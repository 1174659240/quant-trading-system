"""风控：仓位约束、波动率目标、止损，以及风险度量（VaR/CVaR/集中度/风险贡献）。

两类东西：

1. **overlay**（``f(weights, bars, ctx) -> weights``）——插到回测引擎的策略与撮合
   之间，只改目标权重，不碰账户，所以可以任意叠加、单独测试::

       Engine(config, overlays=[max_weight(0.3), vol_target(0.15)])

2. **度量与前置检查**——``RiskLimits`` / ``RiskManager``，供实盘/模拟盘下单前过一遍。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

TRADING_DAYS = 244


# ==================================================================== overlay
def exposure_cap(max_gross: float = 1.0):
    """组合总仓位上限（A股无杠杆时 <=1）。超限按比例缩放。"""

    def overlay(weights: pd.DataFrame, bars: Mapping[str, pd.DataFrame], ctx: Mapping[str, Any]) -> pd.DataFrame:
        gross = weights.abs().sum(axis=1)
        scale = (max_gross / gross).where(gross > max_gross, 1.0)
        return weights.mul(scale, axis=0).fillna(0.0)

    overlay.__name__ = f"exposure_cap({max_gross})"
    return overlay


def max_weight(limit: float = 0.35, renormalize: bool = False):
    """单标的上限。``renormalize=True`` 时把截断掉的部分按比例分给其他标的。"""

    def overlay(weights: pd.DataFrame, bars: Mapping[str, pd.DataFrame], ctx: Mapping[str, Any]) -> pd.DataFrame:
        capped = weights.clip(upper=limit, lower=0.0)
        if not renormalize:
            return capped
        gross = capped.sum(axis=1)
        target = weights.sum(axis=1).clip(upper=1.0)
        scale = (target / gross).replace([np.inf, -np.inf], np.nan)
        return capped.mul(scale.fillna(0.0), axis=0).fillna(0.0)

    overlay.__name__ = f"max_weight({limit})"
    return overlay


def vol_target(target_vol: float = 0.15, window: int = 20, max_scale: float = 1.0):
    """波动率目标：按组合已实现波动缩放总仓位（只减不增）。"""

    def overlay(weights: pd.DataFrame, bars: Mapping[str, pd.DataFrame], ctx: Mapping[str, Any]) -> pd.DataFrame:
        close = pd.DataFrame({s: f["close"] for s, f in bars.items()}).reindex(weights.index).ffill()
        rets = close.pct_change()
        portfolio_ret = (weights.shift(1) * rets).sum(axis=1)
        realized = portfolio_ret.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)
        scale = (target_vol / realized).replace([np.inf, -np.inf], np.nan).clip(upper=max_scale).shift(1)
        return weights.mul(scale.fillna(0.0), axis=0).fillna(0.0)

    overlay.__name__ = f"vol_target({target_vol})"
    return overlay


def trailing_stop(stop_pct: float = 0.12, window: int = 60):
    """个股回撤止损：从近 ``window`` 日最高价回撤超过 ``stop_pct`` 就清掉该标的。"""

    def overlay(weights: pd.DataFrame, bars: Mapping[str, pd.DataFrame], ctx: Mapping[str, Any]) -> pd.DataFrame:
        out = weights.copy()
        for symbol, frame in bars.items():
            if symbol not in out.columns:
                continue
            close = frame["close"].reindex(out.index).ffill()
            rolling_max = close.rolling(window, min_periods=1).max()
            dd = close / rolling_max - 1.0
            out.loc[dd < -abs(stop_pct), symbol] = 0.0
        return out.fillna(0.0)

    overlay.__name__ = f"trailing_stop({stop_pct})"
    return overlay


def turnover_limit(max_turnover: float = 0.3):
    """日换手上限（L1 口径）：超过就把本期权重往上一期拉回来。"""

    def overlay(weights: pd.DataFrame, bars: Mapping[str, pd.DataFrame], ctx: Mapping[str, Any]) -> pd.DataFrame:
        rows: list[pd.Series] = []
        prev = pd.Series(0.0, index=weights.columns)
        for _, row in weights.iterrows():
            diff = (row - prev).abs().sum()
            if diff > max_turnover and diff > 0:
                ratio = max_turnover / diff
                row = prev + (row - prev) * ratio
            rows.append(row)
            prev = row.fillna(0.0).clip(lower=0.0)
        return pd.DataFrame(rows, index=weights.index).fillna(0.0)

    overlay.__name__ = f"turnover_limit({max_turnover})"
    return overlay


def overlays_from_config(raw: Mapping[str, Any] | None) -> list:
    """按配置字典组装 overlay 列表（顺序＝执行顺序）。

    例如 ``{"max_weight": 0.3, "exposure_cap": 0.8, "vol_target": 0.15}``。
    值为 0/None 表示不启用该项。
    """
    raw = dict(raw or {})
    out: list = []

    def _num(key: str, default=None):
        value = raw.get(key, default)
        if value is None:
            return None
        value = float(value)
        return value if value > 0 else None

    limit = _num("max_weight")
    if limit is not None:
        out.append(max_weight(limit, bool(raw.get("max_weight_renormalize", False))))
    gross = _num("exposure_cap", 1.0)
    if gross is not None and gross < 1.0:
        out.append(exposure_cap(gross))
    target_vol = _num("vol_target")
    if target_vol is not None:
        out.append(vol_target(target_vol, int(raw.get("vol_target_window", 20))))
    stop = _num("trailing_stop")
    if stop is not None:
        out.append(trailing_stop(stop, int(raw.get("trailing_stop_window", 60))))
    turnover = _num("turnover_limit")
    if turnover is not None:
        out.append(turnover_limit(turnover))
    return out


# ==================================================================== 风险度量
def historical_var(returns: pd.DataFrame | pd.Series, level: float = 0.05, window: int | None = None):
    """历史法 VaR：返回正数（表示可能损失的幅度）。"""
    if isinstance(returns, pd.Series):
        return float(-np.quantile(returns.dropna().to_numpy(), level))
    if window:
        q = returns.rolling(window, min_periods=max(20, window // 2)).quantile(level)
        return (-q).dropna()
    return (-returns.quantile(level)).rename("var")


def historical_cvar(returns: pd.DataFrame | pd.Series, level: float = 0.05):
    """CVaR / Expected Shortfall（尾部平均损失），返回正数。"""
    if isinstance(returns, pd.Series):
        clean = returns.dropna()
        var = np.quantile(clean.to_numpy(), level)
        tail = clean[clean <= var]
        return float(-tail.mean()) if len(tail) else float(-var)
    out = {}
    for name in returns.columns:
        out[name] = historical_cvar(returns[name].dropna(), level)
    return pd.Series(out, name="cvar")


def correlation_matrix(returns: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    data = returns.tail(window) if window else returns
    return data.corr()


def concentration(weights: pd.Series | pd.DataFrame) -> float | pd.Series:
    """赫芬达尔指数 HHI：1/n 表示完全分散，1 表示单吊一只。"""
    w = weights.abs()
    if isinstance(w, pd.Series):
        total = w.sum()
        return float((w / total).pow(2).sum()) if total > 0 else 0.0
    total = w.sum(axis=1).replace(0.0, np.nan)
    return (w.div(total, axis=0).pow(2).sum(axis=1)).fillna(0.0)


def risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """给定权重与协方差矩阵，各标的对组合波动的贡献占比（边际风险贡献加权后归一）。"""
    aligned = weights.reindex(cov.index).fillna(0.0)
    port_var = float(aligned @ cov @ aligned)
    if port_var <= 0:
        return pd.Series(0.0, index=cov.index)
    mrc = cov @ aligned
    contrib = aligned * mrc / port_var
    return contrib.rename("risk_contribution")


# ==================================================================== 前置检查
@dataclass
class RiskLimits:
    max_weight_per_symbol: float = 0.35
    max_gross_exposure: float = 1.0
    min_cash_ratio: float = 0.0
    max_orders_per_day: int = 200
    max_single_order_value: float = 500_000.0
    banned_symbols: tuple[str, ...] = ()
    min_order_value: float = 0.0
    """低于此金额的订单直接丢弃（省手续费）。"""


@dataclass
class RiskManager:
    limits: RiskLimits = field(default_factory=RiskLimits)
    breaches: list[dict] = field(default_factory=list)

    def trim_targets(self, targets: Mapping[str, float], equity: float) -> dict[str, float]:
        """把目标权重压进限额内，返回调整后的权重（并记录越限事件）。"""
        out = {s: float(w) for s, w in targets.items() if s not in self.limits.banned_symbols}
        for symbol in self.limits.banned_symbols:
            if symbol in targets:
                self.breaches.append({"kind": "banned_symbol", "symbol": symbol, "value": targets[symbol]})
        for symbol, weight in list(out.items()):
            if weight > self.limits.max_weight_per_symbol:
                self.breaches.append({"kind": "max_weight", "symbol": symbol, "value": weight,
                                      "limit": self.limits.max_weight_per_symbol})
                out[symbol] = self.limits.max_weight_per_symbol
        gross = sum(abs(w) for w in out.values())
        cap = min(self.limits.max_gross_exposure, 1.0 - self.limits.min_cash_ratio)
        if gross > cap and gross > 0:
            self.breaches.append({"kind": "max_gross", "value": gross, "limit": cap})
            out = {s: w * cap / gross for s, w in out.items()}
        return out

    def check_order(self, order, equity: float) -> tuple[bool, str]:
        if order.symbol in self.limits.banned_symbols:
            return False, "黑名单标的"
        if order.value > self.limits.max_single_order_value:
            return False, f"单笔金额超限({order.value:,.0f}>{self.limits.max_single_order_value:,.0f})"
        if self.limits.min_order_value and order.value < self.limits.min_order_value:
            return False, f"金额过小({order.value:,.0f})"
        return True, ""

    def max_weight_after(self, portfolio, price: float) -> float:
        return self.limits.max_weight_per_symbol * price

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.breaches) if self.breaches else pd.DataFrame(columns=["kind", "symbol", "value", "limit"])


def risk_summary(returns: pd.DataFrame, weights: pd.Series | None = None, level: float = 0.05) -> dict:
    """一行式风险体检。"""
    out: dict[str, Any] = {
        "var_95": historical_var(returns.mean(axis=1) if isinstance(returns, pd.DataFrame) else returns, level),
        "annual_vol": float(returns.std(ddof=1).mean() * np.sqrt(TRADING_DAYS)) if isinstance(returns, pd.DataFrame) else float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
    }
    if isinstance(returns, pd.DataFrame) and len(returns.columns) > 1:
        corr = correlation_matrix(returns)
        out["avg_correlation"] = float(corr.where(~np.eye(len(corr), dtype=bool)).stack().mean())
        out["cvar_95"] = float(historical_cvar(returns.mean(axis=1), level))
    if weights is not None:
        out["concentration_hhi"] = concentration(weights) if isinstance(weights, pd.DataFrame) else concentration(weights)
    return out
