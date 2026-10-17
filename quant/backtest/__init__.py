"""回测层：撮合、账户、引擎、绩效。"""

from __future__ import annotations

from .broker import Broker, Fill, Order, Rejection
from .engine import BacktestResult, Engine, RiskOverlay, run_backtest
from .metrics import (
    PerformanceReport,
    annual_vol,
    beta_alpha,
    build_report,
    cagr,
    calmar,
    conditional_var,
    drawdown_detail,
    drawdown_series,
    information_ratio,
    max_drawdown,
    monthly_returns,
    profit_factor,
    sharpe,
    sortino,
    total_return,
    value_at_risk,
    win_rate,
    yearly_returns,
)
from .portfolio import Portfolio, Position

__all__ = [
    "BacktestResult",
    "Broker",
    "Engine",
    "Fill",
    "Order",
    "PerformanceReport",
    "Portfolio",
    "Position",
    "Rejection",
    "RiskOverlay",
    "annual_vol",
    "beta_alpha",
    "build_report",
    "cagr",
    "calmar",
    "conditional_var",
    "drawdown_detail",
    "drawdown_series",
    "information_ratio",
    "max_drawdown",
    "monthly_returns",
    "profit_factor",
    "run_backtest",
    "sharpe",
    "sortino",
    "total_return",
    "value_at_risk",
    "win_rate",
    "yearly_returns",
]
