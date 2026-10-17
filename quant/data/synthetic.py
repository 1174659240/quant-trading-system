"""合成行情：离线可复现，用来跑通与压测整条链路。

生成的是「像 A 股」的日线：对数正态随机游走 + 涨跌停截断 + 成交量与波动正相关
+ 可选的低频动量/均值回复结构。所有随机数走独立 ``numpy.random.Generator``，
给定 seed 结果完全一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DataSource, normalize_frame

TRADING_DAYS = 244  # A股一年大约 242~244 个交易日


def trading_calendar(start: str, periods: int, freq: str = "B") -> pd.DatetimeIndex:
    """近似交易日历：去掉周末（工作日频率）；节假日不处理，够回测用。"""
    idx = pd.bdate_range(start=pd.Timestamp(start), periods=periods, name="date")
    return pd.DatetimeIndex(idx.normalize(), name="date")


def make_gbm_frame(
    symbol: str = "SYNTH",
    *,
    start: str = "2020-01-01",
    periods: int = 750,
    seed: int = 20260101,
    start_price: float = 20.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.35,
    limit: float = 0.10,
    momentum: float = 0.0,
    base_volume: float = 2.0e6,
    round_lot: int = 100,
) -> pd.DataFrame:
    """造一只标的的日线。

    参数
    ----
    limit
        涨跌停幅度（主板 0.10，创业板/科创板 0.20 可传 0.20）。0 表示不限制。
    momentum
        动量强度：>0 趋势延续，<0 均值回复。0 为纯随机游走。
    round_lot
        价格最小变动对应股数（A股 0.01 元 → 100 股/手不影响价格）；这里用于价格取整到分。
    """
    rng = np.random.default_rng(seed)
    n = int(periods)
    dt = 1.0 / TRADING_DAYS
    mu = (annual_drift - 0.5 * annual_vol**2) * dt
    sigma = annual_vol * np.sqrt(dt)

    shocks = rng.normal(size=n)
    rets = np.empty(n, dtype=float)
    prev = 0.0
    for i in range(n):
        r = mu + sigma * shocks[i] + momentum * prev
        if limit > 0:
            r = float(np.clip(r, -limit, limit))
        rets[i] = r
        prev = r

    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start_price * float(np.exp(rng.normal(0, sigma * 0.5)))
    if n > 1:
        # 开盘价＝昨日收盘加一个小的跳空
        open_[1:] = close[:-1] * np.exp(rng.normal(0, sigma * 0.4, size=n - 1))
    if limit > 0 and n > 1:
        up = close[:-1] * (1 + limit)
        dn = close[:-1] * (1 - limit)
        open_[1:] = np.clip(open_[1:], dn, up)

    intrabar = np.abs(rng.normal(0, sigma, size=n)) + 0.002
    high = np.maximum(open_, close) * (1 + intrabar)
    low = np.minimum(open_, close) * (1 - intrabar)
    low = np.maximum(low, 0.01)

    if limit > 0 and n > 1:
        pc = np.empty(n)
        pc[0] = open_[0]
        pc[1:] = close[:-1]
        high = np.minimum(high, pc * (1 + limit))
        low = np.maximum(low, pc * (1 - limit))
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

    vol_mult = 1.0 + 3.0 * np.abs(rets) / max(sigma, 1e-9) * 0.25
    volume = np.round(base_volume * vol_mult * rng.lognormal(0, 0.35, size=n))
    volume = np.maximum(volume, 100.0)

    prices = dict(open=open_, high=high, low=low, close=close)
    if round_lot:
        for key in prices:
            prices[key] = np.round(prices[key], 2)
    amount = np.round(volume * (prices["open"] + prices["close"]) / 2.0, 2)

    frame = pd.DataFrame(
        {
            "open": prices["open"],
            "high": prices["high"],
            "low": prices["low"],
            "close": prices["close"],
            "volume": volume,
            "amount": amount,
        },
        index=trading_calendar(start, n),
    )
    return normalize_frame(frame, symbol=symbol)


class SyntheticSource(DataSource):
    """按标的代码派生 seed，保证同一代码每次生成同一条行情。"""

    name = "synthetic"

    def __init__(self, symbols: list[str] | None = None, **kwargs) -> None:
        self._symbols = list(symbols or ["600519.SH", "000001.SZ", "300750.SZ", "601318.SH"])
        self._kwargs = kwargs

    def load(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        seed = (abs(hash(symbol)) % (2**31)) if symbol not in self._symbols else 20260101 + self._symbols.index(symbol)
        kwargs = {k: v for k, v in self._kwargs.items() if k != "seed"}
        frame = make_gbm_frame(symbol, seed=seed, **kwargs)
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index <= pd.Timestamp(end)]
        if len(frame) == 0:
            raise ValueError(f"合成数据在区间 {start}~{end} 内为空")
        return frame

    def symbols(self) -> list[str]:
        return list(self._symbols)
