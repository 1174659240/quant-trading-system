"""Parquet 存储层测试：转换、对拍、读取、锁与覆盖保护。

没有 pyarrow 的环境直接跳过（CSV 层不受影响）。
"""

from __future__ import annotations

import pandas as pd

from quant import parquet_store
from quant.lake import Lake

from _util import setup_console, tmpdir

setup_console()


def _bars(dates, base: float = 10.0, symbol: str = "600000") -> pd.DataFrame:
    rows = []
    for i, day in enumerate(dates):
        close = base + i
        rows.append(
            {
                "date": day,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1000 + i,
                "amount": (1000 + i) * close,
            }
        )
    from quant.data.base import normalize_frame

    return normalize_frame(pd.DataFrame(rows), symbol=symbol)


_DATES = ["2024-12-27", "2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]


def _lake(root, **kwargs):
    return Lake(root=root, min_rows=3, **kwargs)


def _prepare(root):
    lake = _lake(root)
    lake.init()
    for code in ("600011", "600012"):
        lake.persist_raw(code, _bars(_DATES, symbol=code))
    lake.split_holdout()
    lake.build_processed()
    return lake


def test_convert_processed_and_compare():
    if not parquet_store.available():
        print("  . 跳过：未安装 pyarrow")
        return
    with tmpdir("parquet-convert") as root:
        lake = _prepare(root)
        summary = lake.convert_parquet(kinds=("processed",), overwrite=True)
        layer = summary["layers"]["processed"]
        assert summary["ok"], layer["compare"]["mismatch"]
        assert layer["rows"] == 6 and layer["symbols"] == 2
        assert layer["compare"]["ok"]
        # 两个年份 → 两个分区
        assert len(list(lake.processed_parquet.glob("year=*/data.parquet"))) == 1  # 训练区间只有 2024

        csv = lake.read("600011", kind="processed", format="csv")
        pq = lake.read("600011", kind="processed", format="parquet")
        assert (pq.index == csv.index).all()
        for name in ("open", "high", "low", "close", "volume", "amount"):
            gap = (pq[name].astype(float) - csv[name].astype(float)).abs().max()
            assert gap < 1e-9, f"{name} 最大差 {gap}"
        assert (pq["suspended"].astype(int) == csv["suspended"].astype(int)).all()
        assert (pq["tradable"].astype(int) == csv["tradable"].astype(int)).all()
        print("  ✓ processed → Parquet：行数/抽样逐值一致，按年份分区")


def test_convert_requires_overwrite():
    if not parquet_store.available():
        print("  . 跳过：未安装 pyarrow")
        return
    with tmpdir("parquet-overwrite") as root:
        lake = _prepare(root)
        lake.convert_parquet(kinds=("processed",))
        try:
            lake.convert_parquet(kinds=("processed",))
        except FileExistsError as exc:
            assert "overwrite" in str(exc)
        else:
            raise AssertionError("重复转换应要求显式覆盖")
        lake.convert_parquet(kinds=("processed",), overwrite=True)
        print("  ✓ 覆盖保护：默认拒绝重复转换，显式 overwrite 才重建")


def test_parquet_missing_symbol_raises():
    if not parquet_store.available():
        print("  . 跳过：未安装 pyarrow")
        return
    with tmpdir("parquet-missing") as root:
        lake = _prepare(root)
        lake.convert_parquet(kinds=("processed",))
        try:
            lake.read("999999", kind="processed", format="parquet")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Parquet 层里没有的代码应当报 FileNotFoundError")
        print("  ✓ 缺失代码：Parquet 层明确报 FileNotFoundError")


def test_holdout_parquet_needs_unlock():
    if not parquet_store.available():
        print("  . 跳过：未安装 pyarrow")
        return
    with tmpdir("parquet-holdout") as root:
        lake = _prepare(root)
        try:
            lake.convert_parquet(kinds=("holdout",))
        except PermissionError as exc:
            assert "allow_holdout" in str(exc) or "校核集" in str(exc)
        else:
            raise AssertionError("转换 holdout 必须先解锁")
        unlocked = _lake(root, allow_holdout=True)
        summary = unlocked.convert_parquet(kinds=("holdout",))
        assert summary["ok"], summary["layers"]["holdout"]["compare"]["mismatch"]
        frame = unlocked.read("600011", kind="holdout", format="parquet")
        assert len(frame) == 2
        # CSV 层保持原样，没有被改写
        csv = pd.read_csv(unlocked.holdout_path("600011"), encoding="utf-8-sig", dtype={"symbol": str})
        assert len(csv) == 2
        print("  ✓ holdout：转换需解锁，Parquet 副本可读，CSV 原样保留")


def test_parquet_stats_and_verify():
    if not parquet_store.available():
        print("  . 跳过：未安装 pyarrow")
        return
    with tmpdir("parquet-verify") as root:
        lake = _prepare(root)
        lake.convert_parquet(kinds=("processed",))
        info = parquet_store.stats(lake.processed_parquet)
        assert info["rows"] == 6 and info["symbols"] == 2
        assert info["nulls"]["prev_close"] == 2  # 每只首行无前收
        report = lake.verify(limit=10)
        assert report["issue_count"] == 0, report["issues"]
        print("  ✓ 统计与体检：行数/空值/日期范围一致，verify 无问题")
