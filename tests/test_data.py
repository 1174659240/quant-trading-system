"""数据层测试：契约校验、别名映射、CSV/合成源、仓库缓存与对齐。

纯 assert，无 pytest 依赖：``python tests/run_all.py``。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import setup_console, tmpdir

setup_console()

from quant.data import CsvSource, DataError, DataStore, SyntheticSource, make_gbm_frame, normalize_frame
from quant.data.synthetic import trading_calendar


def test_synthetic_contract():
    frame = make_gbm_frame("600519.SH", periods=300, seed=7)
    assert isinstance(frame.index, pd.DatetimeIndex), "索引必须是 DatetimeIndex"
    assert frame.index.name == "date"
    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates
    for col in ("open", "high", "low", "close", "volume", "amount", "prev_close"):
        assert col in frame.columns, f"缺少列 {col}"
    assert (frame["high"] >= frame[["open", "close", "low"]].max(axis=1) - 1e-9).all()
    assert (frame["low"] <= frame[["open", "close", "high"]].min(axis=1) + 1e-9).all()
    assert (frame["volume"] > 0).all()
    print("  ✓ 合成数据满足 K 线契约")


def test_synthetic_deterministic():
    a = make_gbm_frame("X", periods=120, seed=42)
    b = make_gbm_frame("X", periods=120, seed=42)
    pd.testing.assert_frame_equal(a, b)
    c = make_gbm_frame("X", periods=120, seed=43)
    assert not np.allclose(a["close"], c["close"]), "不同 seed 应产生不同行情"
    print("  ✓ 合成数据可复现且随 seed 变化")


def test_price_limit_respected():
    frame = make_gbm_frame("LIMIT", periods=400, seed=11, limit=0.10)
    prev = frame["prev_close"].shift(1).dropna()
    close = frame["close"].loc[prev.index]
    ret = (close / prev - 1.0).abs()
    assert (ret <= 0.1001).all(), f"涨跌停被突破: {ret.max():.4f}"
    print("  ✓ 涨跌停约束生效（最大单日 %.2f%%）" % (ret.max() * 100))


def test_normalize_aliases_and_hand():
    raw = pd.DataFrame(
        {
            "日期": trading_calendar("2024-01-02", 5).strftime("%Y-%m-%d"),
            "开盘": [10.0, 10.1, 10.2, 10.3, 10.4],
            "最高": [10.5, 10.6, 10.7, 10.8, 10.9],
            "最低": [9.9, 10.0, 10.1, 10.2, 10.3],
            "收盘": [10.1, 10.2, 10.3, 10.4, 10.5],
            "成交量(手)": [1000, 1100, 1200, 1300, 1400],
            "成交额": [1.0e6] * 5,
        }
    )
    frame = normalize_frame(raw, symbol="TEST")
    assert frame.index.name == "date"
    assert frame["volume"].iloc[0] == 100000, "手应换算成股（×100）"
    assert "amount" in frame.columns
    assert frame["prev_close"].iloc[0] == frame["open"].iloc[0]
    print("  ✓ 中文列名/手→股/prev_close 回填正确")


def test_normalize_rejects_bad_data():
    bad = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "open": [10.0, 10.0, 10.0],
            "high": [9.0, 11.0, 11.0],  # high < open → 不自洽
            "low": [9.5, 9.5, 9.5],
            "close": [10.0, 10.0, 10.0],
            "volume": [1e5] * 3,
        }
    )
    try:
        normalize_frame(bad, symbol="BAD")
    except DataError as exc:
        assert "不自洽" in str(exc)
        print("  ✓ 不自洽的 high/low 被拒绝")
    else:
        raise AssertionError("应当抛出 DataError")

    missing = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2), "close": [1.0, 2.0]})
    try:
        normalize_frame(missing, symbol="MISS")
    except DataError as exc:
        assert "缺少必需列" in str(exc)
        print("  ✓ 缺列被拒绝并提示列名")
    else:
        raise AssertionError("应当抛出 DataError")


def test_csv_source_roundtrip():
    frame = make_gbm_frame("000001.SZ", periods=250, seed=3)
    with tmpdir("csv") as tmp:
        path = Path(tmp) / "000001.SZ.csv"
        frame.to_csv(path, index=True, index_label="date", encoding="utf-8-sig")
        source = CsvSource(tmp)
        loaded = source.load("000001.SZ", "2020-06-01", "2020-12-31")
        assert len(loaded) > 20
        assert loaded.index.min() >= pd.Timestamp("2020-06-01")
        assert loaded.index.max() <= pd.Timestamp("2020-12-31")
        assert list(source.symbols()) == ["000001.SZ"]
        # 只给裸代码也应能找到
        loaded2 = source.load("000001")
        assert len(loaded2) == len(frame)
    print("  ✓ CSV 读写与区间切片正确")


def test_data_store_cache_and_matrix():
    with tmpdir("store") as tmp:
        store = DataStore(SyntheticSource(["A.SH", "B.SZ"]), cache_dir=tmp)
        bars = store.get_many(["A.SH", "B.SZ"], "2021-01-01", "2021-12-31")
        assert set(bars) == {"A.SH", "B.SZ"}
        mat = store.close_matrix(["A.SH", "B.SZ"], "2021-01-01", "2021-12-31")
        assert mat.shape[1] == 2
        assert mat.index.is_monotonic_increasing
        # 第二次取数应命中磁盘缓存且结果一致
        store2 = DataStore(SyntheticSource(["A.SH"]), cache_dir=tmp)
        again = store2.get("A.SH", "2021-01-01", "2021-12-31")
        pd.testing.assert_frame_equal(mat[["A.SH"]].dropna(), again[["close"]].rename(columns={"close": "A.SH"}).dropna())
        desc = store.describe(["A.SH", "B.SZ"])
        assert len(desc) == 2 and (desc["rows"] > 100).all()
    print("  ✓ DataStore 缓存、矩阵对齐与体检表正常")


def test_calendar_union_intersection():
    store = DataStore(SyntheticSource(["A.SH", "B.SZ"]))
    bars = store.get_many(["A.SH", "B.SZ"])
    u = store.calendar(bars, union=True)
    i = store.calendar(bars, union=False)
    assert len(u) >= len(i)
    assert u.equals(u.sort_values())
    print("  ✓ 日历并集/交集计算正确")


def run() -> None:
    tests = [
        test_synthetic_contract,
        test_synthetic_deterministic,
        test_price_limit_respected,
        test_normalize_aliases_and_hand,
        test_normalize_rejects_bad_data,
        test_csv_source_roundtrip,
        test_data_store_cache_and_matrix,
        test_calendar_union_intersection,
    ]
    for fn in tests:
        fn()


if __name__ == "__main__":
    run()
