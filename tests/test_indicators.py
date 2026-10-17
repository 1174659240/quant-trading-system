"""指标测试：用可手算的序列验证公式，避免“跑通但算错”。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import setup_console

setup_console()

from quant import indicators as ind


def series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values), name="date"), dtype=float)


def test_sma_ema_wma():
    s = series([1, 2, 3, 4, 5])
    ma3 = ind.sma(s, 3)
    assert np.isnan(ma3.iloc[0]) and np.isnan(ma3.iloc[1])
    assert ma3.iloc[2] == 2.0 and ma3.iloc[4] == 4.0
    wma3 = ind.wma(s, 3)
    # (1*1+2*2+3*3)/6 = 14/6
    assert abs(wma3.iloc[2] - 14 / 6) < 1e-9
    ema3 = ind.ema(s, 3)
    assert ema3.iloc[-1] > ema3.iloc[-2]
    print("  ✓ SMA/WMA/EMA 数值正确")


def test_rsi_extremes():
    up = series(np.arange(1, 40, dtype=float))
    r = ind.rsi(up, 14)
    assert r.dropna().iloc[-1] == 100.0, "单边上涨 RSI 应为 100"
    down = series(np.arange(40, 1, -1, dtype=float))
    r2 = ind.rsi(down, 14)
    assert r2.dropna().iloc[-1] == 0.0, "单边下跌 RSI 应为 0"
    zig = series(100 + np.sin(np.arange(60) / 3.0) * 5)
    r3 = ind.rsi(zig, 14).dropna()
    assert ((r3 >= 0) & (r3 <= 100)).all()
    print("  ✓ RSI 边界与取值范围正确")


def test_macd_constant_zero():
    s = series([10.0] * 60)
    m = ind.macd(s)
    tail = m.dropna()
    assert np.allclose(tail.to_numpy(), 0.0, atol=1e-9), "常数序列 MACD 必须为 0"
    print("  ✓ MACD 对常数序列输出 0（无漂移）")


def test_bollinger_and_atr():
    s = series(100 + np.sin(np.arange(80) / 5.0))
    b = ind.bollinger(s, 20, 2.0)
    valid = b.dropna()
    assert (valid["upper"] >= valid["mid"]).all() and (valid["mid"] >= valid["lower"]).all()
    assert (valid["width"] >= 0).all()
    high, low, close = s + 1, s - 1, s
    a = ind.atr(high, low, close, 14).dropna()
    assert (a > 0).all() and (a <= 4).all()
    print("  ✓ 布林带次序与 ATR 正值正确")


def test_crossover():
    fast = series([1, 2, 3, 2, 1, 2, 3])
    slow = series([2, 2, 2, 2, 2, 2, 2])
    up = ind.crossover(fast, slow)
    down = ind.crossunder(fast, slow)
    assert up.iloc[2] and not up.iloc[1]
    assert down.iloc[4]
    assert not (up & down).any()
    print("  ✓ 金叉/死叉识别正确")


def test_kdj_and_zscore():
    s = series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 60)))
    k = ind.kdj(s + 1, s - 1, s).dropna()
    assert ((k["k"] >= 0) & (k["k"] <= 100)).all()
    z = ind.zscore(s, 20).dropna()
    assert abs(z.mean()) < 1.0
    print("  ✓ KDJ 取值范围与 Z-Score 标准化正确")


def test_resample_ohlcv():
    s = series(np.arange(1, 11, dtype=float), start="2024-01-01")
    frame = pd.DataFrame(
        {"open": s, "high": s + 1, "low": s - 1, "close": s + 0.5, "volume": 100.0, "amount": 1000.0},
        index=s.index,
    )
    weekly = ind.resample_ohlcv(frame, "W-FRI")
    assert len(weekly) >= 2
    first_week = frame.loc[: weekly.index[0]]
    assert weekly["open"].iloc[0] == first_week["open"].iloc[0]
    assert weekly["high"].iloc[0] == first_week["high"].max()
    assert weekly["volume"].iloc[0] == first_week["volume"].sum()
    assert weekly.index.name == "date"
    print("  ✓ 日线→周线聚合正确（开高低收量）")


def test_compute_columns():
    from quant.data import make_gbm_frame
    frame = make_gbm_frame("T", periods=200)
    out = ind.compute(frame)
    for col in ("ma20", "macd_dif", "rsi14", "boll_up", "atr14", "kdj_k", "obv", "ret20"):
        assert col in out.columns, f"缺少 {col}"
    assert len(out) == len(frame)
    print("  ✓ compute() 附加大批指标列且不改变行数")


def run() -> None:
    for fn in (
        test_sma_ema_wma,
        test_rsi_extremes,
        test_macd_constant_zero,
        test_bollinger_and_atr,
        test_crossover,
        test_kdj_and_zscore,
        test_resample_ohlcv,
        test_compute_columns,
    ):
        fn()


if __name__ == "__main__":
    run()
