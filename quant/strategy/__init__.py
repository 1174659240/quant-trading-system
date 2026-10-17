"""策略层。"""

from __future__ import annotations

from .base import CompositeStrategy, Strategy
from .library import (
    BUILTIN,
    BollingerReversion,
    BuyAndHold,
    DonchianBreakout,
    FactorScore,
    MaCross,
    MacdTrend,
    MomentumRank,
    RsiReversion,
    TrendFilteredMomentum,
    build,
)
from .multifactor import MultiFactorStrategy

__all__ = [
    "BUILTIN",
    "BollingerReversion",
    "BuyAndHold",
    "CompositeStrategy",
    "DonchianBreakout",
    "FactorScore",
    "MaCross",
    "MacdTrend",
    "MomentumRank",
    "MultiFactorStrategy",
    "RsiReversion",
    "Strategy",
    "TrendFilteredMomentum",
    "build",
]
