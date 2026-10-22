"""按清单下载并解包数据湖（下载端，给别人用的那个脚本）。

用法::

    # 1) 看清单里有什么、本地缺什么（不下载）
    python -m scripts.data_pull --dry-run

    # 2) 下载全部「复现必需」层到 D:/quant-data
    python -m scripts.data_pull --layers core

    # 3) 只更新某一层（例如日线增量）
    python -m scripts.data_pull --layers bars

    # 4) 换镜像/换主机（GitHub Release、Hugging Face、任意静态托管都行）
    python -m scripts.data_pull --manifest-url https://example.com/data/manifest.json

幂等：本地记录 ``<dest>/meta/data-manifest.json``（已应用的版本 + 每层 sha256）。
清单版本没变就整层跳过；变了才下载，下载后做 sha256 校验再解包。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = "https://github.com/1174659240/quant-trading-system/releases/download/data-latest/manifest.json"
DEFAULT_DEST = Path("D:/quant-data")
STATE_NAME = "data-manifest.json"
UA = "quant-lake-puller/1.0"


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url: str, dest: Path, expect_sha: str | None = None, retries: int = 3) -> Path:
    """下载到 ``dest``（先写 .part，校验后再改名）。已存在且校验通过则直接返回。"""
    if dest.exists() and expect_sha and sha256_file(dest) == expect_sha:
        log(f"  = 已是最新：{dest.name}")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            log(f"  ↓ {url}  ({attempt}/{retries})")
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as resp, part.open("wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                while block := resp.read(1 << 20):
                    fh.write(block)
                    got += len(block)
                    if total:
                        pct = got * 100 // total
                        print(f"\r    {got/2**20:7.1f}/{total/2**20:7.1f} MB  {pct:3d}%", end="", flush=True)
            print()
            if expect_sha and sha256_file(part) != expect_sha:
                raise ValueError("sha256 不匹配")
            part.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            log(f"  ! 失败：{exc}")
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
    raise RuntimeError("unreachable")


def safe_extract(archive: Path, dest: Path) -> int:
    """解包到 ``dest``，拒绝绝对路径与 .. 穿越。"""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts or (len(name) > 1 and name[1] == ":"):
                raise ValueError(f"拒绝可疑路径：{member.name}")
            tar.extract(member, path=dest, filter="data")
            n += 1
    return n


def local_state(dest: Path) -> dict:
    path = dest / "meta" / STATE_NAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def plan(manifest: dict, state: dict, layers: list[str]) -> list[tuple[str, dict, bool]]:
    lstate = (state or {}).get("layers", {})
    out = []
    for name in layers:
        info = manifest["layers"].get(name)
        if info is None:
            log(f"  ! 清单里没有层 {name}，跳过")
            continue
        same = lstate.get(name, {}).get("sha256") == info["sha256"]
        out.append((name, info, same))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按清单下载并解包数据湖")
    ap.add_argument("--manifest-url", default=DEFAULT_MANIFEST, help="清单地址（或本地文件路径）")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--layers", default="core", help="core / all / 逗号分隔的层名")
    ap.add_argument("--base-url", default="", help="覆盖清单里的 base_url（镜像用）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-archives", action="store_true", help="解包后保留下载的归档")
    args = ap.parse_args(argv)

    if args.manifest_url.startswith("http"):
        manifest = fetch_json(args.manifest_url)
    else:
        manifest = json.loads(Path(args.manifest_url).read_text(encoding="utf-8"))

    presets = {"core": ["meta", "ref", "bars", "fundamentals", "events", "delisted", "alt", "holdout"]}
    names = presets.get(args.layers, [s.strip() for s in args.layers.split(",") if s.strip()])
    base = (args.base_url or manifest.get("base_url", "")).rstrip("/")
    state = local_state(args.dest)

    log(f"清单版本：{manifest.get('version')}  本地版本：{state.get('version', '（无）')}")
    log(f"下载前缀：{base}")
    rows = plan(manifest, state, names)
    total = sum(i["archive_bytes"] for _, i, same in rows if not same)
    log(f"需要下载：{sum(0 if same else 1 for _, _, same in rows)} 层 / {total/2**20:.1f} MB")
    for name, info, same in rows:
        flag = "已最新" if same else "需更新"
        log(f"  - {name:<13}{info['files']:>6} 文件  {info['archive_bytes']/2**20:>8.1f} MB  {flag}")

    if args.dry_run:
        return 0

    new_state = {"schema_version": 1, "version": manifest.get("version"), "base_url": base, "layers": dict(state.get("layers", {}))}
    tmpdir = Path(tempfile.mkdtemp(prefix="data-pull-", dir=str(args.dest.parent if args.dest.parent.exists() else Path.cwd())))
    try:
        for name, info, same in rows:
            if same:
                continue
            log(f"[layer] {name}")
            for part, expect in zip(info["parts"], info["sha256"], strict=False):
                archive = tmpdir / part
                download(f"{base}/{part}", archive, expect)
                n = safe_extract(archive, args.dest)
                log(f"  ✓ 解包 {n} 个成员")
                if not args.keep_archives:
                    archive.unlink(missing_ok=True)
            new_state["layers"][name] = {"sha256": info["sha256"], "parts": info["parts"], "files": info["files"]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    path = args.dest / "meta" / STATE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[done] 已更新到版本 {new_state['version']}，状态写入 {path}")
    log("下一步：python -m quant lake status   # 核对行数与覆盖区间")
    return 0


if __name__ == "__main__":
    sys.exit(main())
