"""因子库（日线低频）：只用量价能算出来的经典因子。

**统一约定：所有因子「值越大越看多」**。需要反向的因子（波动、换手、彩票效应等）
已经在函数内部乘了 -1，并在 ``direction`` 里标注原始方向，避免合成分数时符号搞反
——这是多因子实现里最常见的隐性 bug。

参考的是券商/公募主流的量价因子分类：动量与反转、趋势、波动、流动性/换手、
量价关系、极端收益（彩票）、位置类。基本面因子（估值/质量/成长）需要财报与
股本数据，属于后续接入真实数据后的工作，这里不假装有。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from quant.factors.panel import Panel

__all__ = ["FACTORS", "FactorSpec", "catalog", "compute", "names", "register_factor"]

TRADING_DAYS = 244


@dataclass(frozen=True)
class FactorSpec:
    name: str
    fn: Callable[..., pd.DataFrame]
    params: Mapping[str, object] = field(default_factory=dict)
    direction: str = "positive"
    category: str = "量价"
    description: str = ""
    sign: int = 1
    """原始值 → 因子值 的符号（+1 原样，-1 取负）。"""


FACTORS: dict[str, FactorSpec] = {}


def register_factor(
    name: str,
    fn: Callable[..., pd.DataFrame],
    *,
    params: Mapping[str, object] | None = None,
    direction: str = "positive",
    category: str = "量价",
    description: str = "",
    sign: int = 1,
    replace: bool = False,
) -> FactorSpec:
    if name in FACTORS and not replace:
        raise ValueError(f"因子 {name!r} 已存在（要覆盖请传 replace=True）")
    spec = FactorSpec(name, fn, dict(params or {}), direction, category, description, sign)
    FACTORS[name] = spec
    return spec


def names() -> list[str]:
    return sorted(FACTORS)


def catalog() -> pd.DataFrame:
    rows = [
        {
            "factor": spec.name,
            "category": spec.category,
            "direction": spec.direction,
            "params": ", ".join(f"{k}={v}" for k, v in spec.params.items()),
            "description": spec.description,
        }
        for spec in FACTORS.values()
    ]
    return pd.DataFrame(rows).sort_values(["category", "factor"]).reset_index(drop=True)


def compute(name: str, panel: Panel, **params) -> pd.DataFrame:
    """按名字算因子；返回 ``index=日期, columns=标的`` 的宽表（越大越看多）。"""
    if name not in FACTORS:
        raise KeyError(f"没有因子 {name!r}，可用：{names()}")
    spec = FACTORS[name]
    merged = dict(spec.params)
    merged.update(params)
    frame = spec.fn(panel, **merged)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"因子 {name} 必须返回 DataFrame，实际 {type(frame).__name__}")
    if spec.sign != 1:
        frame = frame * spec.sign
    return frame.reindex(index=panel.dates, columns=panel.symbols)


# --------------------------------------------------------------------------
# 动量与反转
# --------------------------------------------------------------------------


def momentum(panel: Panel, *, lookback: int = 60, skip: int = 5) -> pd.DataFrame:
    """动量：近 ``lookback`` 日收益，跳过最近 ``skip`` 日（避开短期反转）。"""
    close = panel.close
    end = close.shift(skip)
    start = close.shift(skip + lookback)
    return end / start - 1.0


def reversal(panel: Panel, *, lookback: int = 5) -> pd.DataFrame:
    """短期反转（原始＝近 ``lookback`` 日收益，取负：跌得多的后续反弹）。"""
    close = panel.close
    return close / close.shift(lookback) - 1.0


def trend_strength(panel: Panel, *, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """趋势强度：快均线 / 慢均线 − 1（原始为正，多头排列更强）。"""
    close = panel.close
    return close.rolling(fast, min_periods=max(2, fast // 2)).mean() / close.rolling(
        slow, min_periods=max(3, slow // 2)
    ).mean() - 1.0


def high_ratio(panel: Panel, *, window: int = 60) -> pd.DataFrame:
    """位置类：收盘价 / 近 ``window`` 日最高价（接近新高＝强势）。"""
    close = panel.close
    highest = close.rolling(window, min_periods=max(5, window // 4)).max()
    return close / highest


def close_position(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """收盘位置：窗口内 (close−low)/(high−low) 的均值（收在高位＝买盘强）。"""
    close, high, low = panel.close, panel.high, panel.low
    span = (high - low).replace(0, np.nan)
    position = ((close - low) / span).clip(0.0, 1.0)
    return position.rolling(window, min_periods=max(5, window // 4)).mean()


# --------------------------------------------------------------------------
# 波动与极端收益
# --------------------------------------------------------------------------


def volatility(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """年化波动率（原始为正，低波动占优 → ``sign=-1``）。"""
    returns = panel.returns
    return returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(TRADING_DAYS)


def downside_vol(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """下行波动（只用负收益计算；原始为正，偏好下行风险小的 → ``sign=-1``）。"""
    returns = panel.returns
    downside = returns.where(returns < 0, 0.0)
    return downside.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(TRADING_DAYS)


def max_ret(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """彩票效应：窗口内最大单日收益（原始为正，高彩票偏好后续差 → ``sign=-1``）。"""
    returns = panel.returns
    return returns.rolling(window, min_periods=max(5, window // 2)).max()


def skew(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """收益偏度（原始为正，右偏＝彩票特征 → ``sign=-1``）。"""
    returns = panel.returns
    return returns.rolling(window, min_periods=max(8, window // 2)).skew()


# --------------------------------------------------------------------------
# 均线偏离与摆动
# --------------------------------------------------------------------------


def bias(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """乖离率（原始为正，超买回落 → ``sign=-1``）。"""
    close = panel.close
    ma = close.rolling(window, min_periods=max(5, window // 2)).mean()
    return close / ma - 1.0


def rsi(panel: Panel, *, window: int = 14) -> pd.DataFrame:
    """RSI 归一化到 ±1（原始 0~100 越高越超买 → ``sign=-1``：超卖占优）。"""
    returns = panel.returns
    gain = returns.where(returns > 0, 0.0).rolling(window, min_periods=max(5, window // 2)).mean()
    loss = (-returns.where(returns < 0, 0.0)).rolling(window, min_periods=max(5, window // 2)).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - 100 / (1 + rs)
    return (value - 50.0) / 50.0


# --------------------------------------------------------------------------
# 流动性 / 换手 / 量价
# --------------------------------------------------------------------------


def turnover_abn(panel: Panel, *, window: int = 20, base: int = 250) -> pd.DataFrame:
    """异常换手（成交量相对自身长期均值；原始为正，低换手更优 → ``sign=-1``）。"""
    volume = panel.volume
    short = volume.rolling(window, min_periods=max(3, window // 2)).mean()
    long = volume.rolling(base, min_periods=max(20, base // 4)).mean()
    return short / long.replace(0, np.nan)


def amihud(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """Amihud 非流动性：|收益| / 成交额（原始为正，流动性溢价 → 保留正向）。"""
    returns = panel.returns.abs()
    amount = panel.amount.replace(0, np.nan)
    return (returns / amount * 1e8).rolling(window, min_periods=max(5, window // 2)).mean()


def price_volume_corr(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """量价相关性：收益率与成交量变化的滚动相关（量价齐升更优）。"""
    returns = panel.returns
    volume_change = panel.volume.pct_change().replace([np.inf, -np.inf], np.nan)
    min_periods = max(8, window // 2)
    # pandas 3.x：Rolling.cov(other) 只接受 DataFrame/Series，不再接受另一个 Rolling
    cov = returns.rolling(window, min_periods=min_periods).cov(volume_change)
    denom = returns.rolling(window, min_periods=min_periods).std() * volume_change.rolling(
        window, min_periods=min_periods
    ).std()
    return cov / denom.replace(0, np.nan)


def gap(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    """隔夜跳空均值（开盘/昨收−1；正跳空占优说明承接盘强）。"""
    gap_series = panel.open / panel.prev_close - 1.0
    return gap_series.rolling(window, min_periods=max(5, window // 2)).mean()


# --------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------

register_factor(
    "momentum", momentum, params={"lookback": 60, "skip": 5}, category="动量反转",
    description="近 N 日收益，跳过最近 skip 日（经典 12-1 动量）",
)
register_factor(
    "reversal", reversal, params={"lookback": 5}, category="动量反转", sign=-1, direction="negative",
    description="近 N 日收益取负（短期反转）",
)
register_factor(
    "trend_strength", trend_strength, params={"fast": 5, "slow": 20}, category="趋势",
    description="快均线/慢均线−1（多头排列强度）",
)
register_factor(
    "high_ratio", high_ratio, params={"window": 60}, category="位置",
    description="收盘价/近 N 日最高价（接近新高）",
)
register_factor(
    "close_position", close_position, params={"window": 20}, category="位置",
    description="窗口内收盘价在当日振幅中的平均位置",
)
register_factor(
    "low_vol", volatility, params={"window": 20}, category="波动", sign=-1, direction="negative",
    description="年化波动率取负（低波动占优）",
)
register_factor(
    "downside_vol", downside_vol, params={"window": 20}, category="波动", sign=-1, direction="negative",
    description="下行波动率取负",
)
register_factor(
    "max_ret", max_ret, params={"window": 20}, category="极端收益", sign=-1, direction="negative",
    description="窗口内最大单日收益取负（规避彩票股）",
)
register_factor(
    "skew", skew, params={"window": 20}, category="极端收益", sign=-1, direction="negative",
    description="收益偏度取负（规避右偏）",
)
register_factor(
    "bias", bias, params={"window": 20}, category="均线偏离", sign=-1, direction="negative",
    description="乖离率取负（超买回落）",
)
register_factor(
    "rsi", rsi, params={"window": 14}, category="均线偏离", sign=-1, direction="negative",
    description="RSI 反向（超卖占优）",
)
register_factor(
    "turnover_abn", turnover_abn, params={"window": 20, "base": 250}, category="流动性", sign=-1,
    direction="negative", description="异常换手取负（低换手占优）",
)
register_factor(
    "amihud", amihud, params={"window": 20}, category="流动性",
    description="Amihud 非流动性（流动性溢价）",
)
register_factor(
    "price_volume_corr", price_volume_corr, params={"window": 20}, category="量价",
    description="收益率与成交量变化的滚动相关",
)
register_factor(
    "gap", gap, params={"window": 20}, category="量价",
    description="隔夜跳空均值",
)
