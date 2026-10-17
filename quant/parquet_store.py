"""Parquet 存储层：``processed`` / ``holdout`` 的列式副本。

口径（与 ``docs/存储格式核对-Parquet.md`` 一致）：

- ``raw`` 层**保持 CSV 原样不动**（用户要求原始数据原封不动）；
- 精简层 ``processed`` 与最终校核集 ``holdout`` 各写一份 Parquet，按 ``year=YYYY`` 分区，
  每个分区一个文件；分区的划分只用 ``date`` 的年份，语义由 ``quant.lake`` 保证
  （processed ≤ 2024-12-31，holdout ≥ 2025-01-01）；
- 列类型固定：``symbol`` 字符串、``date`` date32、价量 float64、``suspended``/``tradable`` int8，
  不按值推断类型（避免同一列在不同文件里 float/int 混用）；
- 行数、代码数、日期范围、逐列空值数与 CSV 层完全一致才算转换成功。

没有 pyarrow 时本模块整体不可用，CSV 路径不受影响。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "prev_close",
    "adj_factor",
    "suspended",
    "tradable",
)

FLOAT_COLUMNS = ("open", "high", "low", "close", "volume", "amount", "prev_close", "adj_factor")
INT_COLUMNS = ("suspended", "tradable")


def available() -> bool:
    """pyarrow 是否可用（可选依赖）。"""
    try:  # pragma: no cover - 取决于环境
        import pyarrow  # noqa: F401
    except Exception:
        return False
    return True


def require() -> None:
    """没装 pyarrow 时给出可操作的报错。"""
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "Parquet 需要 pyarrow：在当前 Python 里执行 `python -m pip install pyarrow`，"
            "或继续用 CSV 层（lake 默认仍是 CSV）。"
        ) from exc


def _schema():
    import pyarrow as pa

    fields = [pa.field("symbol", pa.string())]
    fields.append(pa.field("date", pa.date32()))
    fields += [pa.field(name, pa.float64()) for name in FLOAT_COLUMNS]
    fields += [pa.field(name, pa.int8()) for name in INT_COLUMNS]
    return pa.schema(fields)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """把一张 CSV 读出来的表整成统一 schema（缺列补 NaN、多余列丢掉）。"""
    import numpy as np

    def col(name: str) -> pd.Series:
        if name in frame.columns:
            return frame[name]
        return pd.Series(np.nan, index=frame.index)

    out = pd.DataFrame(index=frame.index)
    out["symbol"] = col("symbol").astype("string")
    out["date"] = pd.to_datetime(col("date")).dt.normalize()
    for name in FLOAT_COLUMNS:
        out[name] = pd.to_numeric(col(name), errors="coerce").astype("float64")
    for name in INT_COLUMNS:
        series = pd.to_numeric(col(name), errors="coerce").fillna(0)
        out[name] = series.clip(0, 1).round().astype("int8")
    return out[list(COLUMNS)]


def convert(kind_dir: Path, out_dir: Path, *, columns: Iterable[str] = COLUMNS, overwrite: bool = False) -> dict:
    """把一层 CSV 转成按年份分区的 Parquet。

    返回 ``{"files", "rows", "symbols", "first", "last", "mb", "out"}``。
    转换先写临时目录，全部成功后再替换正式目录，失败不会留下「写了一半」的层。
    """
    require()
    import pyarrow as pa
    import pyarrow.parquet as pq

    kind_dir, out_dir = Path(kind_dir), Path(out_dir)
    if not kind_dir.exists():
        raise FileNotFoundError(kind_dir)
    if out_dir.exists() and not overwrite and any(out_dir.glob("year=*")):
        raise FileExistsError(f"{out_dir} 已存在 Parquet 层，需要覆盖请加 overwrite=True")

    schema = _schema()
    buffers: dict[int, list[pd.DataFrame]] = {}
    csv_files = sorted(kind_dir.glob("*.csv"))
    symbols: set[str] = set()
    first, last = None, None
    total = 0
    for path in csv_files:
        frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"symbol": str})
        if frame.empty:
            continue
        frame = _normalize(frame)
        frame["symbol"] = frame["symbol"].fillna(path.stem).astype("string")
        symbols.update(frame["symbol"].astype(str).unique().tolist())
        total += len(frame)
        lo, hi = frame["date"].min(), frame["date"].max()
        first = lo if first is None or lo < first else first
        last = hi if last is None or hi > last else last
        for year, part in frame.groupby(frame["date"].dt.year):
            buffers.setdefault(int(year), []).append(part)

    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    written_files = 0
    for year in sorted(buffers):
        table = pa.Table.from_pandas(pd.concat(buffers.pop(year), ignore_index=True), schema=schema, preserve_index=False)
        part_dir = tmp_dir / f"year={year}"
        part_dir.mkdir(parents=True, exist_ok=True)
        target = part_dir / "data.parquet"
        pq.write_table(table, target, compression="zstd")
        written_files += 1

    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.replace(out_dir)

    mb = sum(p.stat().st_size for p in out_dir.rglob("*.parquet")) / (1024 * 1024)
    summary = {
        "files": written_files,
        "rows": int(total),
        "symbols": len(symbols),
        "first": None if first is None else str(first.date()),
        "last": None if last is None else str(last.date()),
        "mb": round(mb, 1),
        "out": str(out_dir),
    }
    meta = {
        "kind_dir": str(kind_dir),
        "csv_files": len(csv_files),
        "schema": {field.name: str(field.type) for field in schema},
        "columns": list(columns),
        **summary,
    }
    (out_dir.parent / f"{out_dir.name}-meta.json").write_text(
        pd.Series(meta).to_json(force_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def read_symbol(symbol: str, out_dir: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """按代码读 Parquet 层（谓词下推，只碰命中的年份分区）。"""
    require()
    import pyarrow.dataset as ds

    out_dir = Path(out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)
    dataset = ds.dataset(str(out_dir), format="parquet", partitioning="hive")
    table = dataset.to_table(
        filter=ds.field("symbol") == str(symbol),
        columns=list(columns or COLUMNS),
    )
    frame = table.to_pandas()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").set_index("date").drop(columns=["date"], errors="ignore")


def stats(out_dir: Path, *, sample_symbols: int = 0) -> dict:
    """统计 Parquet 层：文件数、行数、代码数、日期范围、空值数。"""
    require()
    import pyarrow.dataset as ds

    out_dir = Path(out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)
    dataset = ds.dataset(str(out_dir), format="parquet", partitioning="hive")
    table = dataset.to_table()
    frame = table.to_pandas()
    info = {
        "files": len(list(out_dir.rglob("*.parquet"))),
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()) if len(frame) else 0,
        "first": None if not len(frame) else str(pd.to_datetime(frame["date"]).min().date()),
        "last": None if not len(frame) else str(pd.to_datetime(frame["date"]).max().date()),
        "nulls": {name: int(frame[name].isna().sum()) for name in frame.columns},
        "mb": round(sum(p.stat().st_size for p in out_dir.rglob("*.parquet")) / (1024 * 1024), 1),
    }
    if sample_symbols and len(frame):
        codes = sorted(frame["symbol"].unique().tolist())
        info["sample_symbols"] = codes[:sample_symbols]
    return info


def compare_csv(csv_dir: Path, out_dir: Path, *, sample: int = 20) -> dict:
    """把 CSV 层与 Parquet 层对拍：统计量 + 抽样逐值比较。"""
    csv_dir, out_dir = Path(csv_dir), Path(out_dir)
    files = sorted(csv_dir.glob("*.csv"))
    rows = 0
    symbols = set()
    first = last = None
    for path in files:
        frame = pd.read_csv(
            path, encoding="utf-8-sig", usecols=["symbol", "date"],
            parse_dates=["date"], dtype={"symbol": str},
        )
        rows += len(frame)
        if len(frame):
            codes_in_file = frame["symbol"].dropna().astype(str)
            symbols.update(codes_in_file.unique().tolist())
            lo, hi = frame["date"].min(), frame["date"].max()
            first = lo if first is None or lo < first else first
            last = hi if last is None or hi > last else last
    parquet = stats(out_dir)
    result = {
        "csv": {"files": len(files), "rows": rows, "symbols": len(symbols),
                "first": None if first is None else str(first.date()),
                "last": None if last is None else str(last.date())},
        "parquet": {k: parquet[k] for k in ("files", "rows", "symbols", "first", "last", "mb")},
        "mismatch": [],
    }
    for key in ("rows", "symbols", "first", "last"):
        if result["csv"][key] != result["parquet"][key]:
            result["mismatch"].append(f"{key}: csv={result['csv'][key]!r} parquet={result['parquet'][key]!r}")
    codes = sorted(symbols)[: max(1, sample)]
    for code in codes:
        path = csv_dir / f"{code}.csv"
        if not path.exists():
            continue
        left = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"symbol": str})
        left = _normalize(left.set_index("date").reset_index()).set_index("date").sort_index()
        right = read_symbol(code, out_dir)
        right = _normalize(right.reset_index()).set_index("date").sort_index()
        if len(left) != len(right):
            result["mismatch"].append(f"{code}: 行数 {len(left)} != {len(right)}")
            continue
        for name in FLOAT_COLUMNS:
            gap = (left[name] - right[name]).abs().max()
            if pd.notna(gap) and gap > 1e-9:
                result["mismatch"].append(f"{code}.{name}: 最大差 {gap:.3e}")
        for name in INT_COLUMNS:
            if not left[name].equals(right[name]):
                result["mismatch"].append(f"{code}.{name}: 取值不一致")
    result["checked_symbols"] = len(codes)
    result["ok"] = not result["mismatch"]
    return result


def remove(out_dir: Path) -> None:
    """删掉某一层的 Parquet 副本（CSV 层不受影响）。"""
    if Path(out_dir).exists():
        shutil.rmtree(out_dir, ignore_errors=True)


def estimate_row_bytes(out_dir: Path) -> float:  # pragma: no cover - 诊断用
    total = sum(p.stat().st_size for p in Path(out_dir).rglob("*.parquet"))
    rows = stats(out_dir)["rows"] or 1
    return total / rows


__all__ = [
    "COLUMNS",
    "available",
    "require",
    "convert",
    "read_symbol",
    "stats",
    "compare_csv",
    "remove",
]
