"""因子合成：把多个单因子合成一个打分（score）。

成熟的流程里，合成权重不是拍脑袋定的，而是:

* **等权**：最稳的基线，也是"新因子没证明自己之前"的默认选择；
* **IC 加权**：按各因子历史 IC 均值分配权重（IC 为负的因子直接清零，不做空自己）；
* **IC_IR 加权**：按 ``IC均值 / IC标准差`` 分配权重，偏好"稳定有用"而不是"偶尔很准"；
* **贪心筛选**：在 IC_IR 基础上逐个加入候选，只在"合成 IC IR 变好且与已选因子相关性低"时留下
  ——这一步同时解决冗余与过拟合。

所有合成都在**预处理之后**进行（各因子已 z-score），否则不同量纲加权没有意义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant.factors.evaluate import ic_summary, rank_ic
from quant.factors.panel import Panel
from quant.factors.preprocess import standardize, winsorize

__all__ = [
    "CombinedScore",
    "combine",
    "composite_ic",
    "equal_weights",
    "ic_weights",
    "ir_weights",
    "score_panel",
    "select_by_ic_ir",
]


@dataclass
class CombinedScore:
    """合成结果：打分表 + 各因子权重 + 评估指标。"""

    score: pd.DataFrame
    weights: dict[str, float]
    diagnostics: dict[str, object] = field(default_factory=dict)

    def top(self, day, n: int = 10) -> pd.Series:
        row = self.score.loc[day].dropna().sort_values(ascending=False)
        return row.head(n)


def equal_weights(names: Iterable[str]) -> dict[str, float]:
    """等权（自动归一化）。"""
    items = list(names)
    if not items:
        raise ValueError("没有因子可以赋权")
    weight = 1.0 / len(items)
    return {name: weight for name in items}


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(abs(v) for v in weights.values())
    if total <= 0:
        return equal_weights(weights.keys())
    return {k: float(v) / total for k, v in weights.items()}


def ic_weights(
    factors: Mapping[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    *,
    min_abs_ic: float = 0.0,
) -> dict[str, float]:
    """按 IC 均值加权（负 IC 因子清零）。"""
    raw: dict[str, float] = {}
    for name, frame in factors.items():
        stats = ic_summary(rank_ic(frame, forward_returns))
        value = stats["ic_mean"]
        if np.isnan(value) or value <= min_abs_ic:
            value = 0.0
        raw[name] = float(value)
    if sum(raw.values()) <= 0:
        return equal_weights(factors.keys())
    return _normalize(raw)


def ir_weights(
    factors: Mapping[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    *,
    min_ic_ir: float = 0.0,
) -> dict[str, float]:
    """按 IC_IR 加权（负 IR 因子清零）。"""
    raw: dict[str, float] = {}
    for name, frame in factors.items():
        stats = ic_summary(rank_ic(frame, forward_returns))
        value = stats["ic_ir"]
        if np.isnan(value) or value <= min_ic_ir:
            value = 0.0
        raw[name] = float(value)
    if sum(raw.values()) <= 0:
        return equal_weights(factors.keys())
    return _normalize(raw)


def combine(
    factors: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | None = None,
    *,
    standardize_: bool = True,
    winsorize_: bool = True,
) -> pd.DataFrame:
    """按权重把因子相加成打分表（缺失因子按 0 处理，即"中性"）。"""
    if not factors:
        raise ValueError("factors 为空")
    weight_map = dict(weights) if weights else equal_weights(factors.keys())
    unknown = set(weight_map) - set(factors)
    if unknown:
        raise KeyError(f"权重里有未提供的因子：{sorted(unknown)}")
    total: pd.DataFrame | None = None
    for name, frame in factors.items():
        weight = float(weight_map.get(name, 0.0))
        if weight == 0.0:
            continue
        prepared = frame
        if winsorize_:
            prepared = winsorize(prepared)
        if standardize_:
            prepared = standardize(prepared)
        prepared = prepared.fillna(0.0) * weight
        total = prepared if total is None else total.add(prepared, fill_value=0.0)
    if total is None:
        raise ValueError("所有因子权重都是 0")
    total.index.name = "date"
    return total


def composite_ic(
    factors: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """合成打分的 IC 序列（供筛选与评估复用）。"""
    score = combine(factors, weights)
    return rank_ic(score, forward_returns)


def select_by_ic_ir(
    factors: Mapping[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    *,
    max_factors: int = 6,
    corr_threshold: float = 0.7,
    min_ic_ir: float = 0.05,
) -> dict[str, float]:
    """贪心筛选：按 IC_IR 逐个纳入因子，只保留能提升合成 IC_IR 的。

    返回 ``{因子名: 权重}``（IC_IR 加权后的权重）。
    """
    from quant.factors.evaluate import factor_corr

    if not factors:
        raise ValueError("factors 为空")
    stats = {name: ic_summary(rank_ic(frame, forward_returns)) for name, frame in factors.items()}
    ranking = sorted(
        (name for name in factors if not np.isnan(stats[name]["ic_ir"])),
        key=lambda name: stats[name]["ic_ir"],
        reverse=True,
    )
    if not ranking:
        return equal_weights(factors.keys())

    corr = factor_corr(factors) if len(factors) > 1 else None
    selected: list[str] = []
    best_ir = -np.inf
    for name in ranking:
        if stats[name]["ic_ir"] < min_ic_ir:
            break
        if corr is not None and selected:
            too_close = any(
                not np.isnan(corr.loc[name, other]) and abs(corr.loc[name, other]) > corr_threshold
                for other in selected
            )
            if too_close:
                continue
        trial = selected + [name]
        weights = ir_weights({k: factors[k] for k in trial}, forward_returns)
        ic = composite_ic({k: factors[k] for k in trial}, weights, forward_returns)
        combined_ir = ic_summary(ic)["ic_ir"]
        if np.isnan(combined_ir):
            continue
        if combined_ir > best_ir:
            best_ir = combined_ir
            selected = trial
        if len(selected) >= max_factors:
            break
    if not selected:
        return equal_weights(list(factors)[:1])
    weights = ir_weights({k: factors[k] for k in selected}, forward_returns)
    return weights


def score_panel(
    panel: Panel,
    specs: Mapping[str, Mapping[str, object]],
    *,
    weights: Mapping[str, float] | None = None,
    forward_returns: pd.DataFrame | None = None,
    weighting: str = "equal",
    preprocess_kwargs: Mapping[str, object] | None = None,
) -> CombinedScore:
    """完整打分流程：算因子 → 预处理 → 定权 → 合成。

    ``specs`` 形如 ``{"momentum": {"lookback": 60}, "low_vol": {"window": 20}}``。
    ``weighting`` 取 ``equal`` / ``ic`` / ``ir`` / ``auto``（贪心筛选）。
    """
    from quant.factors.library import compute
    from quant.factors.preprocess import preprocess

    raw: dict[str, pd.DataFrame] = {}
    for name, params in specs.items():
        frame = compute(name, panel, **dict(params or {}))
        raw[name] = preprocess(frame, **dict(preprocess_kwargs or {}))

    diagnostics: dict[str, object] = {"weighting": weighting, "factors": list(raw)}
    horizon = forward_returns if forward_returns is not None else panel.forward_returns(1)

    if weights is not None:
        resolved = dict(weights)
    elif weighting == "equal":
        resolved = equal_weights(raw.keys())
    elif weighting == "ic":
        resolved = ic_weights(raw, horizon)
    elif weighting == "ir":
        resolved = ir_weights(raw, horizon)
    elif weighting == "auto":
        resolved = select_by_ic_ir(raw, horizon)
    else:
        raise ValueError("weighting 只能是 equal / ic / ir / auto")

    score = combine(raw, resolved, standardize_=False, winsorize_=False)
    diagnostics["ic"] = ic_summary(rank_ic(score, horizon))
    return CombinedScore(score=score, weights=resolved, diagnostics=diagnostics)
