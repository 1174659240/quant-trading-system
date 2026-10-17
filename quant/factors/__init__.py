"""多因子框架门面：面板、因子库、预处理、评估、合成。"""

from __future__ import annotations

from .composite import (
    CombinedScore,
    combine,
    composite_ic,
    equal_weights,
    ic_weights,
    ir_weights,
    score_panel,
    select_by_ic_ir,
)
from .evaluate import (
    evaluate_factor,
    evaluate_text,
    factor_corr,
    ic_decay,
    ic_summary,
    pearson_ic,
    quantile_returns,
    quantile_summary,
    rank_ic,
)
from .library import FACTORS, FactorSpec, catalog, compute, names, register_factor
from .panel import Panel, build_panel, symbols_with_history
from .preprocess import (
    fill_missing,
    neutralize,
    preprocess,
    rank_normalize,
    standardize,
    winsorize,
)

__all__ = [
    "CombinedScore",
    "FACTORS",
    "FactorSpec",
    "Panel",
    "build_panel",
    "catalog",
    "combine",
    "composite_ic",
    "compute",
    "equal_weights",
    "evaluate_factor",
    "evaluate_text",
    "factor_corr",
    "fill_missing",
    "ic_decay",
    "ic_summary",
    "ic_weights",
    "ir_weights",
    "names",
    "neutralize",
    "pearson_ic",
    "preprocess",
    "quantile_returns",
    "quantile_summary",
    "rank_ic",
    "rank_normalize",
    "register_factor",
    "score_panel",
    "select_by_ic_ir",
    "standardize",
    "symbols_with_history",
    "winsorize",
]
