"""市场级危机识别与「一键清仓 / 回暖再入场」overlay。

设计依据（机构与学术两条线，详见 ``docs/风险清仓与前沿因子.md``）：

- **趋势**：Faber (2007) 的 10 个月均线择时、Moskowitz-Ooi-Pedersen (2012) 时间序列动量——
  指数跌破 200 日均线是机构最常用的「risk-off」开关。
- **波动**：Moreira & Muir (2017) 波动率管理——已实现波动进入历史高分位时，风险资产预期收益显著恶化。
- **宽度 / 尾部 / 流动性**：Zweig 的宽度推力（breadth thrust）反向使用、A 股「跌停家数」是本土卖方
  报告中最高频的危机确认指标；Kritzman & Li (2010) 的金融动荡指数强调「相关性骤升 + 流动性枯竭」。
- **复合而非单指标**：单一指标在 A 股反复出现假信号（2019-01、2020-03、2022-04 的 V 型反转），
  所以用 6 个分量加权求和，只有多数分量同时报警才清仓；回暖再入场用「分数回落到低位 + 连续确认」。

时点约定：所有分量只用 **t 日收盘及以前** 的数据；引擎把 t 日权重在 t+1 开盘执行，
所以 t 日收盘触发的清仓，实际成交在 t+1 开盘，不存在前视。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 244

#: 六个分量的默认权重（合计 1.0）：趋势与波动权重最高，与机构 risk-off 框架一致。
DEFAULT_WEIGHTS = {
    "trend": 0.25,      # 指数跌破 200 日均线
    "drawdown": 0.15,   # 指数自 250 日高点回撤超过阈值
    "vol": 0.20,        # 20 日已实现波动处于 250 日高分位
    "breadth": 0.15,    # 站上 50 日均线的股票占比过低
    "tail": 0.15,       # 跌停/大跌股票占比过高
    "illiquidity": 0.10,  # 市场 Amihud 非流动性处于高分位
}


def market_index(panel) -> pd.DataFrame:
    """等权市场指数与逐日诊断分量（全部为 PIT 口径）。"""
    rets = panel.returns
    weighted = rets.where(panel.tradable.reindex_like(rets).fillna(False))
    market_ret = weighted.mean(axis=1, skipna=True).fillna(0.0)
    index = (1.0 + market_ret).cumprod()

    close = panel.close
    # 宽度：站上 50 日均线的股票占比（个股需要有 50 日历史）
    ma50 = close.rolling(50, min_periods=40).mean()
    above = (close > ma50) & close.notna() & ma50.notna()
    breadth = above.sum(axis=1) / ma50.notna().sum(axis=1).replace(0, np.nan)

    # 尾部：单日跌幅 <= -9.5%（跌停附近）的股票占比
    tail_share = (rets <= -0.095).sum(axis=1) / rets.notna().sum(axis=1).replace(0, np.nan)

    # 非流动性：个股 |ret| / 成交额的横截面中位数，再取 250 日分位
    amount = panel.amount.replace(0, np.nan)
    illiq = (rets.abs() / amount).replace([np.inf, -np.inf], np.nan).median(axis=1)
    illiq_pct = illiq.rolling(250, min_periods=120).rank(pct=True)

    ma200 = index.rolling(200, min_periods=150).mean()
    dd250 = index / index.rolling(250, min_periods=120).max() - 1.0
    vol20 = market_ret.rolling(20, min_periods=15).std(ddof=1) * np.sqrt(TRADING_DAYS)
    vol_pct = vol20.rolling(250, min_periods=120).rank(pct=True)

    frame = pd.DataFrame(
        {
            "market_ret": market_ret,
            "index": index,
            "ma200": ma200,
            "dd250": dd250,
            "vol20": vol20,
            "vol_pct": vol_pct,
            "breadth": breadth,
            "tail_share": tail_share,
            "illiquidity": illiq,
            "illiq_pct": illiq_pct,
        },
        index=panel.dates,
    )
    return frame


def score_components(
    frame: pd.DataFrame,
    *,
    dd_threshold: float = -0.12,
    vol_quantile: float = 0.90,
    breadth_threshold: float = 0.25,
    tail_threshold: float = 0.01,
    market_crash: float = -0.04,
    illiq_quantile: float = 0.90,
) -> pd.DataFrame:
    """把诊断分量转成 0/1 触发标记（NaN 一律视为不触发）。"""
    f = frame
    flags = pd.DataFrame(index=f.index)
    flags["trend"] = (f["index"] < f["ma200"]).fillna(False)
    flags["drawdown"] = (f["dd250"] <= dd_threshold).fillna(False)
    flags["vol"] = (f["vol_pct"] >= vol_quantile).fillna(False)
    flags["breadth"] = (f["breadth"] <= breadth_threshold).fillna(False)
    flags["tail"] = ((f["tail_share"] >= tail_threshold) | (f["market_ret"] <= market_crash)).fillna(False)
    flags["illiquidity"] = (f["illiq_pct"] >= illiq_quantile).fillna(False)
    return flags.astype(bool)


def crisis_score(flags: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    missing = set(w) - set(flags.columns)
    if missing:
        raise KeyError(f"缺少分量：{sorted(missing)}")
    total = float(sum(w.values()))
    score = sum(flags[name].astype(float) * value for name, value in w.items()) / total
    score.name = "score"
    return score


def crisis_state(
    score: pd.Series,
    *,
    enter: float = 0.60,
    exit: float = 0.30,
    confirm: int = 3,
    gate: pd.Series | None = None,
    recovery: pd.Series | None = None,
) -> pd.Series:
    """滞回状态机：分数 >= enter（且 ``gate`` 为真）进入危机；「回暖」连续 confirm 天解除。

    解除条件 = 分数回落到 ``exit`` 以下 **或** ``recovery``（通常是「指数重新站上 200 日均线」）为真。
    ``gate`` 用来把「慢熊」和「极度危险」区分开：只允许在波动/尾部同时报警的日子触发清仓
    （机构 risk-off 通常要求「价格 + 波动」双确认，单看均线会在阴跌里反复止损）。
    返回 bool Series（True = 危机，应空仓）。只用当日及之前的信息。
    """
    if not 0 < exit < enter <= 1:
        raise ValueError("需要 0 < exit < enter <= 1")
    values = score.to_numpy(dtype=float)
    allow = (
        gate.reindex(score.index).fillna(False).to_numpy(dtype=bool)
        if gate is not None
        else np.ones(len(values), dtype=bool)
    )
    healed = (
        recovery.reindex(score.index).fillna(False).to_numpy(dtype=bool)
        if recovery is not None
        else np.zeros(len(values), dtype=bool)
    )
    state = np.zeros(len(values), dtype=bool)
    current = False
    calm_streak = 0
    for i, value in enumerate(values):
        if current:
            calm = bool(value <= exit or np.isnan(value) or healed[i])
            calm_streak = calm_streak + 1 if calm else 0
            if calm_streak >= confirm:
                current = False
                calm_streak = 0
        else:
            if value >= enter and allow[i]:
                current = True
                calm_streak = 0
        state[i] = current
    return pd.Series(state, index=score.index, name="crisis")


def crisis_overlay(state: pd.Series, *, exposure: float = 0.0, delay: int = 0):
    """把危机状态转成 overlay：危机日把目标权重清零（成交发生在下一交易日开盘）。

    ``delay`` 用于敏感性测试（0 = 收盘触发、次日开盘执行；1 = 再晚一天）。
    """

    def overlay(weights: pd.DataFrame, bars, ctx) -> pd.DataFrame:
        flags = state.reindex(weights.index).ffill().fillna(False).astype(bool)
        if delay:
            flags = flags.shift(delay, fill_value=False)
        out = weights.copy()
        out.loc[flags] = out.loc[flags] * exposure
        return out.fillna(0.0)

    overlay.__name__ = f"crisis_liquidation(enter={state.mean():.2f})"
    return overlay


@dataclass
class CrisisEpisode:
    start: pd.Timestamp
    end: pd.Timestamp | None
    days: int
    max_score: float
    peak_dd: float


def episodes(state: pd.Series, score: pd.Series, frame: pd.DataFrame) -> list[CrisisEpisode]:
    """把连续危机日聚成事件，便于人工核对（例如 2018、2020-03、2022、2024-01）。"""
    out: list[CrisisEpisode] = []
    start = None
    peak = 0.0
    worst = 0.0
    for day, flag in state.items():
        if flag:
            if start is None:
                start = day
                peak = float(score.get(day, np.nan))
                worst = float(frame.loc[day, "dd250"]) if day in frame.index else 0.0
            else:
                peak = max(peak, float(score.get(day, np.nan)))
                worst = min(worst, float(frame.loc[day, "dd250"]) if day in frame.index else 0.0)
        elif start is not None:
            out.append(CrisisEpisode(start, day, int(((state.loc[start:day])).sum()), peak, worst))
            start = None
    if start is not None:
        out.append(CrisisEpisode(start, None, int(state.loc[start:].sum()), peak, worst))
    return out
