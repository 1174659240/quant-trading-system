"""绩效与风险指标：只依赖 pandas/numpy，输入净值序列与成交记录。

所有「年化」都按 244 个交易日换算（A股惯例），无风险利率默认 2%。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

TRADING_DAYS = 244


# ------------------------------------------------------------------ 基础
def to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / TRADING_DAYS
    if years <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def annual_vol(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.02) -> float:
    vol = annual_vol(returns)
    if vol == 0:
        return 0.0
    excess = returns.mean() * TRADING_DAYS - rf
    return float(excess / vol)


def sortino(returns: pd.Series, rf: float = 0.02) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    dd = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if dd == 0:
        return 0.0
    return float((returns.mean() * TRADING_DAYS - rf) / dd)


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(drawdown_series(equity).min())


def drawdown_detail(equity: pd.Series) -> dict:
    """最大回撤的完整信息：幅度、开始、谷底、恢复日、持续天数。"""
    if len(equity) < 2:
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None, "recovery_date": None,
                "drawdown_days": 0, "recovery_days": None}
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    after = equity.loc[trough:]
    recovered = after[after >= equity.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    return {
        "max_drawdown": float(dd.min()),
        "peak_date": peak,
        "trough_date": trough,
        "recovery_date": recovery,
        "drawdown_days": int(equity.index.get_loc(trough) - equity.index.get_loc(peak)),
        "recovery_days": None if recovery is None else int(equity.index.get_loc(recovery) - equity.index.get_loc(trough)),
    }


def calmar(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity) / mdd)


def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """历史法单日 VaR（返回负数，表示损失）。"""
    if len(returns) < 20:
        return 0.0
    return float(np.quantile(returns.to_numpy(), level))


def conditional_var(returns: pd.Series, level: float = 0.05) -> float:
    """CVaR / Expected Shortfall：最差 level 分位下的平均损失。"""
    if len(returns) < 20:
        return 0.0
    var = value_at_risk(returns, level)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else var


def win_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def beta_alpha(returns: pd.Series, benchmark: pd.Series, rf: float = 0.02) -> tuple[float, float]:
    joined = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return 0.0, 0.0
    r, b = joined.iloc[:, 0], joined.iloc[:, 1]
    var_b = b.var(ddof=1)
    if var_b == 0:
        return 0.0, 0.0
    beta = float(r.cov(b) / var_b)
    alpha = float((r.mean() - rf / TRADING_DAYS) - beta * (b.mean() - rf / TRADING_DAYS)) * TRADING_DAYS
    return beta, alpha


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    joined = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return 0.0, 0.0
    active = joined.iloc[:, 0] - joined.iloc[:, 1]
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if te == 0:
        return 0.0, 0.0
    return float(active.mean() * TRADING_DAYS / te), te


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    if len(equity) < 2:
        return pd.DataFrame()
    m = equity.resample("ME").last().pct_change().dropna()
    if m.empty:
        return pd.DataFrame()
    frame = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    table = frame.pivot(index="year", columns="month", values="ret")
    table.columns = [f"{c}月" for c in table.columns]
    table["全年"] = (1 + table).prod(axis=1) - 1
    return table


def yearly_returns(equity: pd.Series) -> pd.Series:
    if len(equity) < 2:
        return pd.Series(dtype=float)
    y = equity.resample("YE").last().pct_change()
    first = equity.iloc[0]
    if len(y):
        y.iloc[0] = equity.resample("YE").last().iloc[0] / first - 1.0
    y.index = [d.year for d in y.index]
    return y.dropna()


# ------------------------------------------------------------------ 汇总
@dataclass
class PerformanceReport:
    metrics: dict = field(default_factory=dict)
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    yearly: pd.Series = field(default_factory=pd.Series)

    def to_text(self, title: str = "绩效报告") -> str:
        lines = [f"=== {title} ==="]
        labels = [
            ("start", "起始日"), ("end", "结束日"), ("days", "交易日数"),
            ("total_return", "累计收益"), ("cagr", "年化收益"), ("ann_vol", "年化波动"),
            ("sharpe", "夏普比率"), ("sortino", "索提诺比率"), ("calmar", "卡玛比率"),
            ("max_drawdown", "最大回撤"), ("mdd_peak", "回撤起点"), ("mdd_trough", "回撤谷底"),
            ("mdd_recovery", "回撤恢复"), ("mdd_days", "回撤持续(交易日)"),
            ("win_rate", "日胜率"), ("profit_factor", "盈亏比"), ("var95", "日VaR(95%)"),
            ("cvar95", "日CVaR(95%)"), ("best_day", "最好单日"), ("worst_day", "最差单日"),
            ("avg_turnover", "日均换手"), ("avg_exposure", "日均仓位"), ("total_fees", "累计费用"),
            ("trade_count", "成交笔数"), ("beta", "Beta"), ("alpha", "年化Alpha"),
            ("info_ratio", "信息比率"), ("tracking_error", "跟踪误差"),
        ]
        for key, label in labels:
            if key in self.metrics and self.metrics[key] is not None:
                value = self.metrics[key]
                if isinstance(value, float):
                    if key in {"total_return", "cagr", "ann_vol", "max_drawdown", "win_rate", "best_day",
                               "worst_day", "var95", "cvar95", "avg_turnover", "avg_exposure", "alpha",
                               "tracking_error"}:
                        value = f"{value:.2%}"
                    else:
                        value = f"{value:.3f}"
                lines.append(f"{label:<16}: {value}")
        if not self.yearly.empty:
            lines.append("\n年度收益：")
            for year, ret in self.yearly.items():
                lines.append(f"  {year}: {ret:+.2%}")
        if not self.monthly.empty:
            lines.append("\n月度收益：")
            lines.append(self.monthly.to_string(float_format=lambda x: f"{x:+.2%}"))
        return "\n".join(lines)


def build_report(
    equity: pd.Series,
    *,
    trades: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    rf: float = 0.02,
) -> PerformanceReport:
    if len(equity) < 2:
        return PerformanceReport(metrics={"days": len(equity)})
    returns = to_returns(equity)
    dd = drawdown_detail(equity)
    metrics: dict = {
        "start": equity.index[0].date().isoformat(),
        "end": equity.index[-1].date().isoformat(),
        "days": int(len(equity)),
        "initial_equity": float(equity.iloc[0]),
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "ann_vol": annual_vol(returns),
        "sharpe": sharpe(returns, rf),
        "sortino": sortino(returns, rf),
        "calmar": calmar(equity),
        "max_drawdown": dd["max_drawdown"],
        "mdd_peak": dd["peak_date"].date().isoformat() if dd["peak_date"] is not None else None,
        "mdd_trough": dd["trough_date"].date().isoformat() if dd["trough_date"] is not None else None,
        "mdd_recovery": dd["recovery_date"].date().isoformat() if dd["recovery_date"] is not None else None,
        "mdd_days": dd["drawdown_days"],
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "var95": value_at_risk(returns, 0.05),
        "cvar95": conditional_var(returns, 0.05),
        "best_day": float(returns.max()) if len(returns) else 0.0,
        "worst_day": float(returns.min()) if len(returns) else 0.0,
        "skew": float(returns.skew()) if len(returns) > 3 else 0.0,
        "kurtosis": float(returns.kurtosis()) if len(returns) > 3 else 0.0,
        "positive_days": int((returns > 0).sum()),
        "negative_days": int((returns < 0).sum()),
    }
    if daily is not None and len(daily):
        if "turnover" in daily:
            metrics["avg_turnover"] = float(daily["turnover"].mean())
            metrics["total_turnover"] = float(daily["turnover"].sum())
        if "gross_exposure" in daily:
            metrics["avg_exposure"] = float(daily["gross_exposure"].mean())
        if "fees" in daily:
            metrics["total_fees"] = float(daily["fees"].sum())
    if trades is not None and len(trades):
        metrics["trade_count"] = int(len(trades))
        metrics["buy_count"] = int((trades["side"] == "buy").sum())
        metrics["sell_count"] = int((trades["side"] == "sell").sum())
        metrics["total_fee_paid"] = float(trades["fee"].sum())
        metrics["total_slippage"] = float(trades["slippage_cost"].sum())
        closed = trades[trades["side"] == "sell"]
        if len(closed):
            wins = closed[closed["realized_pnl"] > 0]
            metrics["closed_trades"] = int(len(closed))
            metrics["trade_win_rate"] = float(len(wins) / len(closed))
            gains = float(closed.loc[closed["realized_pnl"] > 0, "realized_pnl"].sum())
            losses = float(-closed.loc[closed["realized_pnl"] < 0, "realized_pnl"].sum())
            metrics["trade_profit_factor"] = float(gains / losses) if losses > 0 else float("inf")
    if benchmark is not None and len(benchmark) > 1:
        bench_ret = to_returns(benchmark)
        beta, alpha = beta_alpha(returns, bench_ret, rf)
        ir, te = information_ratio(returns, bench_ret)
        metrics.update({
            "benchmark_return": total_return(benchmark),
            "benchmark_cagr": cagr(benchmark),
            "benchmark_max_drawdown": max_drawdown(benchmark),
            "beta": beta,
            "alpha": alpha,
            "info_ratio": ir,
            "tracking_error": te,
        })
    return PerformanceReport(metrics=metrics, monthly=monthly_returns(equity), yearly=yearly_returns(equity))
