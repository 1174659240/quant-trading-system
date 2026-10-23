"""把数据湖打包成可分发的归档 + 清单（发布端）。

设计目标：**同一个地址永远拿到最新数据**，且更新时只重传变化的层。

产物结构（``--out`` 默认 ``dist/data``）::

    dist/data/
      manifest.json                  # 清单：版本、每层文件名、字节数、sha256
      bars-20261022T2130.tar.gz      # 每层一个归档（可再按大小切分）
      ref-20261022T2130.tar.gz
      ...

清单里的 ``base_url`` 是下载前缀（GitHub Release / Hugging Face / 任意静态托管都能用），
``data_pull.py`` 只认清单，不认主机。

命令行::

    python -m scripts.data_pack --plan                 # 只估算体积，不写文件
    python -m scripts.data_pack --layers bars,ref      # 打包指定层
    python -m scripts.data_pack --layers core          # 打包「复现必需」的一组
    python -m scripts.data_pack --layers all --split-mb 900
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

LAKE = Path("D:/quant-data")
OUT = Path("dist/data")
DEFAULT_BASE_URL = "https://github.com/1174659240/quant-trading-system/releases/download/data-latest"

# 层定义：名字 -> (根目录相对路径, 需要排除的子路径)
# 只收「parquet / 小文件」这类不可从公开源低成本重建的东西；raw CSV 与 daily CSV 可重建，默认不收。
LAYERS: dict[str, dict] = {
    "bars": {"roots": ["processed/parquet"], "extra": ["meta/calendar.csv"], "note": "日线列式层（≤2024-12-31）"},
    "holdout": {
        "roots": ["holdout/parquet"],
        "extra": ["holdout/dividends.parquet", "holdout/turn_panel.parquet", "holdout/st.parquet"],
        "note": "样本外列式层（2025-01-01 起，含 .LOCKED 不含）",
    },
    "valuation": {"roots": ["valuation"], "extra": [], "note": "逐日估值/换手/isST（PIT）"},
    "fundamentals": {"roots": ["fundamentals"], "extra": [], "note": "五张财务表，55 个报告期"},
    "ref": {"roots": ["ref"], "extra": [], "note": "universe/行业 PIT/股本 PIT/分红/指数成分/换手宽表"},
    "events": {"roots": ["events"], "extra": [], "note": "龙虎榜/增减持/回购/ST/送转（含 available_date）"},
    "delisted": {"roots": ["delisted"], "extra": [], "note": "退市股日线与退市事件（haircut 用）"},
    "alt": {"roots": ["alt"], "extra": [], "note": "预告/快报/解禁/质押（2026-10 采集）"},
    "history": {"roots": ["growth-history", "early"], "extra": [], "note": "2007–2016 历史日线与历史财务"},
    "meta": {"roots": ["meta"], "extra": [], "note": "日历、universe、manifest、抓取日志"},
    "raw": {"roots": ["raw"], "extra": [], "note": "不复权原始 CSV（体积大，一般用不到）"},
}

#: holdout 层只对所有者开放（一次性验收集，协作者不应持有）
OWNER_ONLY = {"holdout"}

PRESETS = {
    "core": ["meta", "ref", "bars", "fundamentals", "events", "delisted", "alt"],
    "core+valuation": ["meta", "ref", "bars", "fundamentals", "events", "delisted", "alt", "valuation"],
    "all": list(LAYERS),
}
PRESETS["all"] = ["meta", "ref", "bars", "fundamentals", "events", "delisted", "alt", "valuation", "history", "raw"]


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dir_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def layer_files(name: str, lake: Path) -> list[tuple[Path, str]]:
    """返回 [(绝对路径, 归档内相对路径)]。"""
    spec = LAYERS[name]
    out: list[tuple[Path, str]] = []
    for rel in spec["roots"]:
        root = lake / rel
        if not root.exists():
            continue
        if root.is_file():
            out.append((root, rel))
            continue
        for p in dir_files(root):
            out.append((p, p.relative_to(lake).as_posix()))
    for rel in spec.get("extra", []):
        p = lake / rel
        if p.exists():
            out.append((p, rel))
    return out


def estimate(layers: list[str], lake: Path) -> list[dict]:
    rows = []
    for name in layers:
        files = layer_files(name, lake)
        total = sum(p.stat().st_size for p, _ in files)
        rows.append({"layer": name, "files": len(files), "bytes": total, "MB": round(total / 2**20, 1)})
    return rows


@dataclass
class Archive:
    layer: str
    parts: list[str] = field(default_factory=list)
    bytes: int = 0
    sha256: list[str] = field(default_factory=list)
    files: int = 0
    raw_bytes: int = 0


def _writer_for(tmp: Path, split_mb: int):
    """返回 (写句柄, 切换函数, 收尾函数)。split_mb<=0 表示不切分。"""
    limit = split_mb * 2**20 if split_mb > 0 else 0
    state = {"idx": 0, "fh": None, "parts": [], "size": 0}

    def _open():
        name = f"{tmp.name}.part{state['idx']:02d}" if limit else f"{tmp.name}"
        path = tmp.parent / name
        fh = path.open("wb")
        state["fh"] = fh
        state["parts"].append(path.name)
        state["size"] = 0
        return fh

    fh = _open()

    def _rotate_if_needed() -> None:
        nonlocal fh
        if limit and state["size"] >= limit:
            fh.close()
            state["idx"] += 1
            fh = _open()

    def _close() -> None:
        fh.close()

    return fh, _rotate_if_needed, _close, state


def build_layer(name: str, lake: Path, out: Path, version: str, split_mb: int, level: int) -> Archive:
    files = layer_files(name, lake)
    arc = Archive(layer=name, files=len(files), raw_bytes=sum(p.stat().st_size for p, _ in files))
    tar_path = out / f"{name}-{version}.tar.gz"
    fh, rotate, close, state = _writer_for(tar_path, split_mb)

    import io

    class _Sink(io.RawIOBase):
        """把 tarfile 的写入转给当前分片文件。"""

        def writable(self) -> bool:  # noqa: D102
            return True

        def write(self, b) -> int:  # noqa: D102
            fh.write(b)
            state["size"] += len(b)
            rotate()
            return len(b)

    sink = _Sink()
    with tarfile.open(fileobj=sink, mode="w|gz", compresslevel=level) as tar:
        for path, rel in files:
            tar.add(path, arcname=rel, recursive=False)
    close()

    for part in state["parts"]:
        p = out / part
        arc.parts.append(part)
        arc.bytes += p.stat().st_size
        arc.sha256.append(sha256_file(p))
    return arc


def coverage(lake: Path) -> dict:
    """从 meta/manifest.json 与各层目录粗算覆盖区间，写进清单方便下载方核对。"""
    info: dict = {}
    meta = lake / "meta" / "manifest.json"
    if meta.exists():
        try:
            info["lake_manifest"] = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            pass
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="打包数据湖为可分发的归档 + 清单")
    ap.add_argument("--lake", type=Path, default=LAKE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--layers", default="core", help="core / core+valuation / all / 逗号分隔的层名")
    ap.add_argument("--version", default="", help="默认 UTC 时间戳 YYYYMMDDTHHMM")
    ap.add_argument("--split-mb", type=int, default=0, help="按 MB 切分归档（0=不切）")
    ap.add_argument("--level", type=int, default=6, help="gzip 压缩级别 1-9")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--plan", action="store_true", help="只估算体积，不写文件")
    ap.add_argument("--with-holdout", action="store_true",
                    help="所有者：把 holdout 层也打进包（需环境变量 QUANT_OWNER=1）")
    args = ap.parse_args(argv)

    if args.with_holdout:
        if os.environ.get("QUANT_OWNER", "").strip() not in {"1", "true", "yes"}:
            raise SystemExit("[拒绝] --with-holdout 只对所有者开放：请设 QUANT_OWNER=1 再运行。")
        if "holdout" not in PRESETS:
            pass
        extra = ["holdout"]
    else:
        extra = []

    layers = PRESETS.get(args.layers, [s.strip() for s in args.layers.split(",") if s.strip()])
    layers = list(dict.fromkeys([*layers, *extra]))
    unknown = [n for n in layers if n not in LAYERS]
    if unknown:
        raise SystemExit(f"未知的层：{unknown}；可选 {sorted(LAYERS)} 或 {sorted(PRESETS)}")
    blocked = [n for n in layers if n in OWNER_ONLY and not extra]
    if blocked:
        raise SystemExit(f"[拒绝] 这些层只对所有者开放：{blocked}；加 --with-holdout 且设 QUANT_OWNER=1")

    rows = estimate(layers, args.lake)
    total = sum(r["bytes"] for r in rows)
    print(f"{'层':<14}{'文件':>7}{'原始体积':>12}")
    for r in rows:
        print(f"{r['layer']:<14}{r['files']:>7}{r['MB']:>10.1f} MB")
    print(f"{'合计':<14}{sum(r['files'] for r in rows):>7}{total/2**20:>10.1f} MB")
    if args.plan:
        return 0

    version = args.version or time.strftime("%Y%m%dT%H%M", time.gmtime())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url.rstrip("/"),
        "note": "data_pull.py 只读本清单；同一 base_url 下引用不同版本归档，因此地址不变、内容可更新。",
        "layers": {},
    }
    for name in layers:
        print(f"[pack] {name} ...", flush=True)
        arc = build_layer(name, args.lake, args.out, version, args.split_mb, args.level)
        manifest["layers"][name] = {
            "note": LAYERS[name]["note"],
            "files": arc.files,
            "raw_bytes": arc.raw_bytes,
            "archive_bytes": arc.bytes,
            "parts": arc.parts,
            "sha256": arc.sha256,
        }
        print(f"[pack] {name}: {arc.files} 文件 -> {arc.bytes/2**20:.1f} MB，{len(arc.parts)} 片")

    manifest["coverage"] = coverage(args.lake)
    path = args.out / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {path}  ({path.stat().st_size/1024:.1f} KB)")
    print(f"[done] 归档总体积 {sum(a['archive_bytes'] for a in manifest['layers'].values())/2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
