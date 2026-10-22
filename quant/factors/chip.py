"""筹码分布因子（自建）：换手率加权的历史成本分布。

口径（用户 2026-10-21 任务书，任务二）：

- 每个交易日按换手率把一部分筹码换成当日价格，旧筹码保留、按 (1 - 换手) 衰减；
  于是 t 日的成本分布权重为 ``w_j = turnover_j · Π_{k=j+1..t}(1 - turnover_k)``（j 在最近 ``window`` 日内）。
- 由此得到：
  - ``profit``    获利盘比例 = 现价高于参考成本的筹码权重占比；
  - ``conc70`` / ``conc90`` 成本集中度 = (q85-q15)/q50、(q95-q05)/q50；
  - ``overhang``  资本利得悬置 = (现价 - 平均成本) / 平均成本。

只依赖日线（收盘价）与换手率，不含任何未来信息：t 日只用 ≤t 的数据。
不注册进 ``library.FACTORS``（保持与共享因子库解耦）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.factors import style
from quant.factors.panel import Panel

WINDOW = 250
MIN_OBS = 60


def _turnover(panel: Panel) -> pd.DataFrame:
    """换手率（小数，0.01 = 1%）；缺失按 0 处理（不衰减）。"""
    turn = style.turn_panel(panel.dates, panel.symbols)
    return (turn / 100.0).clip(lower=0.0, upper=0.95)


def _weighted_quantile(sorted_price: np.ndarray, cum_weight: np.ndarray, q: float) -> np.ndarray:
    """按列取加权分位：``sorted_price``/``cum_weight`` 形状 (T, N)，已按价格升序。"""
    hit = cum_weight >= q
    position = hit.argmax(axis=0)
    return np.take_along_axis(sorted_price, position[None, :], axis=0)[0]


def chip_panels(
    panel: Panel,
    marks: pd.DatetimeIndex,
    *,
    window: int = WINDOW,
    min_obs: int = MIN_OBS,
) -> dict[str, pd.DataFrame]:
    """在给定日期上计算四个筹码因子（返回 ``{name: DataFrame(index=marks, columns=symbols)}``）。"""
    close = panel.close
    turn = _turnover(panel)
    dates = close.index
    symbols = close.columns

    out = {name: pd.DataFrame(np.nan, index=marks, columns=symbols, dtype="float64")
           for name in ("profit", "conc70", "conc90", "overhang")}
    values = close.to_numpy(dtype="float64")
    turns = turn.reindex(index=dates, columns=symbols).to_numpy(dtype="float64")

    for stamp in marks:
        if stamp not in dates:
            continue
        position = dates.get_loc(stamp)
        start = max(0, position - window + 1)
        price = values[start:position + 1]           # (T, N)
        rate = turns[start:position + 1]
        rate = np.nan_to_num(rate, nan=0.0)
        if price.shape[0] < min_obs:
            continue

        keep = price[-1]  # 现价
        # 生存权重：c_j = Π_{k=j..t}(1 - rate_k)，则 w_j = rate_j · c_j / (1 - rate_j)
        decay = 1.0 - rate
        cum = np.cumprod(decay[::-1], axis=0)[::-1]
        weight = np.where(rate > 0, rate * cum / np.maximum(decay, 1e-12), 0.0)
        total = weight.sum(axis=0)
        valid = (total > 0) & np.isfinite(keep) & np.isfinite(price).any(axis=0)
        if not valid.any():
            continue
        weight = weight / np.where(total > 0, total, 1.0)

        # 获利盘：现价 >= 成本价 的权重和
        prof = (weight * (price <= keep[None, :])).sum(axis=0)

        # 加权分位：按价格排序后对权重累加
        order = np.argsort(price, axis=0, kind="stable")
        sorted_price = np.take_along_axis(price, order, axis=0)
        sorted_weight = np.take_along_axis(weight, order, axis=0)
        cum_weight = np.cumsum(sorted_weight, axis=0)
        q05 = _weighted_quantile(sorted_price, cum_weight, 0.05)
        q15 = _weighted_quantile(sorted_price, cum_weight, 0.15)
        q50 = _weighted_quantile(sorted_price, cum_weight, 0.50)
        q85 = _weighted_quantile(sorted_price, cum_weight, 0.85)
        q95 = _weighted_quantile(sorted_price, cum_weight, 0.95)
        avg = (weight * np.nan_to_num(price, nan=0.0)).sum(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            conc70 = (q85 - q15) / q50
            conc90 = (q95 - q05) / q50
            overhang = (keep - avg) / avg

        for name, vector in (("profit", prof), ("conc70", conc70),
                             ("conc90", conc90), ("overhang", overhang)):
            block = out[name].loc[stamp].to_numpy(dtype="float64", copy=True)
            block[:] = np.where(valid, vector, np.nan)
            out[name].loc[stamp] = block

    return out
