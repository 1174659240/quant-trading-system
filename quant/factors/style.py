"""红利低波与容量受限因子（新增数据源：PIT 分红、换手、PIT 股本）。

注册进全局因子库后，可直接用 ``python -m quant factor-market --factors div_yield,vol_20 ...``。
所有因子遵守库约定：**值越大越看多**（原始方向见 ``direction``）。

数据依赖与 PIT 口径
- ``div_yield``：``quant.dividends``（东财分红明细，只用 ``ex_date <= t`` 的已实施现金分红，
  过去 365 天每股派息之和 / 收盘价），无未来函数；
- ``turnover_*`` / ``turnover_vol``：baostock 日换手率（``D:/quant-data/valuation/*.parquet`` 的
  ``turn`` 列，裁剪到 2024-12-31，见 ``ref/turn_panel.parquet``）；
- ``mcap`` / ``fmcap``：``收盘价 × 当日可见股本``（``ref/shares.csv`` 的 PIT 变动表），
  自由流通市值用 ``float_share``，若大于总股本按总股本截断（原始数据个别记录不可信）；
- ``listing_days``：``universe.ipo_date`` 到当日的自然日；
- ``rev_smallcap``：短期反转只在小市值（自由流通市值当日前 30%）内取值，其余 NaN。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from quant import dividends, refdata
from quant.factors.exposures import _bare
from quant.factors.library import register_factor
from quant.factors.panel import Panel

__all__ = [
    "HOLDOUT_START",
    "amount_mean",
    "div_yield",
    "float_mcap_panel",
    "listing_days",
    "low_price",
    "mcap_panel",
    "rev_smallcap",
    "spread_proxy",
    "total_mcap",
    "turnover_mean",
    "turnover_vol",
    "turn_panel",
    "vol_20",
    "vol_60",
    "zero_days",
]

HOLDOUT_START = pd.Timestamp("2025-01-01")
TURN_FILE = refdata.REF_DIR / "turn_panel.parquet"
_CACHE: dict[str, object] = {}


def _key(dates: pd.Index, symbols: pd.Index) -> tuple:
    return (
        str(dates[0]),
        str(dates[-1]),
        len(dates),
        len(symbols),
        str(symbols[0]),
        str(symbols[-1]),
    )


def turn_panel(dates: pd.Index, symbols: pd.Index) -> pd.DataFrame:
    """换手率宽表（%），按 ``(日期, 代码)`` 对齐；缺失为 NaN。"""
    key = "turn|" + "|".join(map(str, _key(dates, symbols)))
    hit = _CACHE.get(key)
    if isinstance(hit, pd.DataFrame):
        return hit
    frame = pd.read_parquet(TURN_FILE)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"] <= HOLDOUT_START - pd.Timedelta(days=1)]
    frame = frame.set_index("date")
    frame = frame.reindex(index=dates, columns=[str(column) for column in symbols])
    frame = frame.apply(pd.to_numeric, errors="coerce").astype("float32")
    frame.columns = symbols
    _CACHE[key] = frame
    return frame


def _share_panels(dates: pd.Index, symbols: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PIT 总股本 / 自由流通股本矩阵（股），``searchsorted`` 取当日可见最近一条变动。"""
    key = "shares|" + "|".join(map(str, _key(dates, symbols)))
    hit = _CACHE.get(key)
    if isinstance(hit, tuple):
        return hit
    frame = refdata.load_shares()
    frame = frame[(frame["is_real_code"] == 1) & frame["total_share"].notna()]
    days = pd.DatetimeIndex(dates).to_numpy()
    totals = np.full((len(dates), len(symbols)), np.nan)
    floats = np.full((len(dates), len(symbols)), np.nan)
    for position, symbol in enumerate(symbols):
        part = frame[frame["symbol"] == _bare(symbol)].sort_values("change_date")
        if part.empty:
            continue
        change_dates = pd.to_datetime(part["change_date"]).to_numpy()
        total_values = part["total_share"].to_numpy(dtype=np.float64)
        float_values = pd.to_numeric(part["float_share"], errors="coerce").to_numpy(dtype=np.float64)
        idx = np.searchsorted(change_dates, days, side="right") - 1
        valid = idx >= 0
        totals[valid, position] = total_values[idx[valid]]
        floats[valid, position] = float_values[idx[valid]]
    total_frame = pd.DataFrame(totals, index=dates, columns=symbols)
    float_frame = pd.DataFrame(np.fmin(floats, totals), index=dates, columns=symbols)
    _CACHE[key] = (total_frame, float_frame)
    return total_frame, float_frame


def mcap_panel(panel: Panel) -> pd.DataFrame:
    """总市值（元）＝收盘价 × 当日可见总股本。"""
    total, _ = _share_panels(panel.dates, panel.symbols)
    return total * panel.close


def float_mcap_panel(panel: Panel) -> pd.DataFrame:
    """自由流通市值（元）＝收盘价 × 当日可见自由流通股本（超过总股本时截断）。"""
    _, floats = _share_panels(panel.dates, panel.symbols)
    return floats * panel.close


# --------------------------------------------------------------------------- 因子
def div_yield(panel: Panel, *, window: int = 365) -> pd.DataFrame:
    """股息率（近 365 天已除息每股现金分红 / 收盘价）。"""
    dps = dividends.dps_panel(panel.dates, panel.symbols, window_days=window)
    return dps / panel.close.replace(0, np.nan)


def vol_20(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    return panel.returns.rolling(window, min_periods=max(5, window // 2)).std(ddof=0)


def vol_60(panel: Panel, *, window: int = 60) -> pd.DataFrame:
    return panel.returns.rolling(window, min_periods=max(10, window // 2)).std(ddof=0)


def turnover_mean(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    return turn_panel(panel.dates, panel.symbols).rolling(window, min_periods=max(5, window // 2)).mean()


def turnover_vol(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    return turn_panel(panel.dates, panel.symbols).rolling(window, min_periods=max(5, window // 2)).std(ddof=0)


def zero_days(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """近 N 日零成交（volume<=0 或停牌）天数。"""
    dead = (panel.volume.fillna(0.0) <= 0).astype(float)
    return dead.rolling(window, min_periods=max(5, window // 2)).sum()


def amount_mean(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """近 N 日日均成交额（元）。"""
    return panel.amount.rolling(window, min_periods=max(5, window // 2)).mean()


def spread_proxy(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """买卖价差代理：近 N 日 ``(最高-最低)/收盘`` 均值（越大越缺流动性）。"""
    spread = (panel.high - panel.low) / panel.close.replace(0, np.nan)
    return spread.rolling(window, min_periods=max(5, window // 2)).mean()


def low_price(panel: Panel) -> pd.DataFrame:
    """收盘价（低价股效应：取负后越大越看多）。"""
    return panel.close


def total_mcap(panel: Panel) -> pd.DataFrame:
    """总市值（元）取对数；取负后越大越看多（小市值溢价）。"""
    return np.log(mcap_panel(panel).where(lambda frame: frame > 0))


def free_float_mcap(panel: Panel) -> pd.DataFrame:
    return np.log(float_mcap_panel(panel).where(lambda frame: frame > 0))


def listing_days(panel: Panel) -> pd.DataFrame:
    """上市天数（自然日），缺 ipo_date 为 NaN。"""
    universe = refdata.load_universe()
    ipo = dict(zip(universe["symbol"].astype(str), pd.to_datetime(universe["ipo_date"], errors="coerce")))
    days = pd.DatetimeIndex(panel.dates)
    out = np.full((len(days), len(panel.symbols)), np.nan)
    for position, symbol in enumerate(panel.symbols):
        start = ipo.get(_bare(symbol))
        if start is None or pd.isna(start):
            continue
        out[:, position] = (days - start).days
    return pd.DataFrame(out, index=panel.dates, columns=panel.symbols)


def rev_smallcap(panel: Panel, *, lookback: int = 20, quantile: float = 0.3) -> pd.DataFrame:
    """短期反转 × 小市值：反转分只保留自由流通市值当日前 ``quantile`` 的股票。"""
    mcap = float_mcap_panel(panel)
    small = mcap.rank(axis=1, pct=True) <= quantile
    rev = -(panel.close / panel.close.shift(lookback) - 1.0)
    return rev.where(small)


def size_band(panel: Panel, *, quantile: float = 0.05, source: str = "total") -> pd.DataFrame:
    """规模分档：只在「市值最小的 quantile 比例」内取负对数市值，其余为 NaN。

    与 ``mcap`` / ``fmcap`` 的区别：连续因子在全市场排序，分档因子先把 95%（或 90%、80%）
    的股票剔除，再在最小的一档里排序——用来检验「小市值溢价的收益是不是集中在最尾部」。
    """
    frame = mcap_panel(panel) if source == "total" else float_mcap_panel(panel)
    frame = frame.where(frame > 0)
    inside = frame.rank(axis=1, pct=True) <= quantile
    return (-np.log(frame)).where(inside)


def maxret_smallcap(panel: Panel, *, window: int = 20, quantile: float = 0.3) -> pd.DataFrame:
    """彩票效应 × 小市值：近 N 日单日最大涨幅取负，只保留自由流通市值前 ``quantile`` 的股票。"""
    small = float_mcap_panel(panel).rank(axis=1, pct=True) <= quantile
    best = panel.returns.rolling(window, min_periods=max(5, window // 2)).max()
    return (-best).where(small)


# --------------------------------------------------------------------------- 注册
register_factor(
    "div_yield", div_yield, params={"window": 365}, category="红利",
    description="近 365 天已除息现金分红/收盘价（PIT，已实施口径）",
)
register_factor(
    "vol_20", vol_20, params={"window": 20}, category="波动", sign=-1, direction="negative",
    description="20 日收益波动率取负（低波）",
)
register_factor(
    "vol_60", vol_60, params={"window": 60}, category="波动", sign=-1, direction="negative",
    description="60 日收益波动率取负（低波）",
)
register_factor(
    "turnover_20", turnover_mean, params={"window": 20}, category="流动性", sign=-1, direction="negative",
    description="20 日平均换手率取负（低换手）",
)
register_factor(
    "turnover_vol", turnover_vol, params={"window": 20}, category="流动性", sign=-1, direction="negative",
    description="20 日换手率波动取负",
)
register_factor(
    "zero_days", zero_days, params={"window": 20}, category="流动性", sign=-1, direction="negative",
    description="20 日零成交天数取负",
)
register_factor(
    "amount_20", amount_mean, params={"window": 20}, category="流动性", sign=-1, direction="negative",
    description="20 日日均成交额取负（低成交额）",
)
register_factor(
    "spread_proxy", spread_proxy, params={"window": 20}, category="流动性", sign=-1, direction="negative",
    description="20 日(最高-最低)/收盘均值取负（价差窄）",
)
register_factor(
    "low_price", low_price, category="规模", sign=-1, direction="negative",
    description="收盘价取负（低价股）",
)
register_factor(
    "mcap", total_mcap, category="规模", sign=-1, direction="negative",
    description="ln(总市值) 取负（小市值）",
)
register_factor(
    "fmcap", free_float_mcap, category="规模", sign=-1, direction="negative",
    description="ln(自由流通市值) 取负（小市值）",
)
register_factor(
    "listing_days", listing_days, category="规模", sign=-1, direction="negative",
    description="上市天数取负（越大越看多＝越新越好，检验次新效应）",
)
register_factor(
    "rev_smallcap", rev_smallcap, params={"lookback": 20, "quantile": 0.3}, category="量价",
    description="20 日反转只在小市值（自由流通市值前 30%）内取值",
)
for _tag, _q in (("q05", 0.05), ("q10", 0.10), ("q20", 0.20)):
    register_factor(
        f"mcap_{_tag}", size_band, params={"quantile": _q, "source": "total"}, category="规模",
        description=f"总市值最小 {_q:.0%} 档内的负对数市值（分档小市值）",
    )
    register_factor(
        f"fmcap_{_tag}", size_band, params={"quantile": _q, "source": "float"}, category="规模",
        description=f"自由流通市值最小 {_q:.0%} 档内的负对数市值（分档小市值）",
    )
register_factor(
    "maxret_smallcap", maxret_smallcap, params={"window": 20, "quantile": 0.3}, category="量价",
    description="近 20 日单日最大涨幅取负，只在小市值（自由流通市值前 30%）内取值",
)
