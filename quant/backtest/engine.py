"""事件驱动回测引擎。

时序约定（避免未来函数）::

    t-1 收盘 → 策略给出目标权重（weights 的第 t-1 行）
    t   开盘 → 按开盘价撮合（受 T+1 / 涨跌停 / 整手 / 现金约束）
    t   收盘 → 按收盘价重估净值，结算 T+1，落一条日度快照

即权重矩阵的第 i 行在日历的第 i+1 个交易日开盘执行，天然不会用到当日信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from .broker import Broker, Fill, Rejection
from .metrics import PerformanceReport, build_report
from .portfolio import Portfolio

RiskOverlay = Callable[[pd.DataFrame, dict[str, pd.DataFrame], dict], pd.DataFrame]


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series
    daily: pd.DataFrame
    trades: pd.DataFrame
    rejections: pd.DataFrame
    targets: pd.DataFrame
    held: pd.DataFrame
    report: PerformanceReport
    benchmark: pd.Series | None = None
    config: BacktestConfig | None = None

    # ---- 便捷 ----
    @property
    def metrics(self) -> dict:
        return self.report.metrics

    def summary(self) -> str:
        return self.report.to_text(title=f"回测结果 · {self.strategy}")

    def to_frame(self) -> pd.DataFrame:
        return self.daily


@dataclass
class Engine:
    config: BacktestConfig
    overlays: list[RiskOverlay] = field(default_factory=list)
    verbose: bool = False
    # 基准收盘价序列（可选）。给了就用它算基准，否则退回「全池等权指数」。
    benchmark_prices: pd.Series | None = None

    # -------------------------------------------------------------- 主流程
    def run(self, bars: dict[str, pd.DataFrame], weights: pd.DataFrame, strategy_name: str = "strategy") -> BacktestResult:
        if not bars:
            raise ValueError("bars 为空，无法回测")
        if weights is None or weights.empty:
            raise ValueError("目标权重矩阵为空")

        bars = {s: f.sort_index() for s, f in bars.items()}
        calendar = self._calendar(bars)
        weights = weights.reindex(index=calendar).reindex(columns=sorted(bars), fill_value=0.0)
        weights = weights.ffill().fillna(0.0)
        for overlay in self.overlays:
            weights = overlay(weights, bars, {"config": self.config})
        weights = weights.reindex(index=calendar).fillna(0.0)
        weights = weights.clip(lower=-1.0, upper=1.0)

        # targets = 过完风控后的目标权重（未滞后），可用来核查 overlay 是否生效
        targets = weights.copy()
        # 权重在 t-1 产生、t 开盘执行
        executable = weights.shift(1).fillna(0.0)

        portfolio = Portfolio(cash=float(self.config.initial_cash))
        broker = Broker(costs=self.config.costs, rules=self.config.rules)
        broker.rejections.clear()

        trades: list[Fill] = []
        held_rows: list[dict] = []
        open_map = self._price_map(bars, "open", calendar)
        close_map = self._price_map(bars, "close", calendar)

        for date in calendar:
            # 1) 以开盘价重估（定单规模与现金判断都用当日已知价格）
            portfolio.mark_all(open_map.get(date, {}), date)
            day_targets = executable.loc[date]
            target_dict = {s: float(v) for s, v in day_targets.items()}

            orders = broker.plan_orders(date, target_dict, portfolio, bars, price_field="open")
            day_fills = broker.execute(date, orders, bars, portfolio)

            # 2) 收盘重估、结算、快照
            portfolio.mark_all(close_map.get(date, {}), date)
            gross = sum(f.gross for f in day_fills)
            fees = sum(f.fee for f in day_fills)
            equity_now = portfolio.equity
            turnover = gross / max(equity_now, 1e-9)
            portfolio.settle()
            portfolio.snapshot(date, turnover=turnover, fees=fees)
            trades.extend(day_fills)
            held_rows.append(portfolio.weights())

        equity = pd.Series(
            {r["date"]: r["equity"] for r in portfolio.records}, name="equity"
        ).sort_index()
        equity = pd.concat([pd.Series({calendar[0]: float(self.config.initial_cash)}), equity])
        equity = equity[~equity.index.duplicated(keep="last")].sort_index()

        daily = portfolio.history()
        trade_df = pd.DataFrame([f.as_dict() for f in trades])
        rejection_df = pd.DataFrame([r.as_dict() for r in broker.rejections])
        held = pd.DataFrame(held_rows, index=calendar).fillna(0.0)
        held.index.name = "date"

        benchmark = self._benchmark_equity(bars, calendar)
        report = build_report(equity, trades=trade_df if len(trade_df) else None, daily=daily, benchmark=benchmark)

        return BacktestResult(
            strategy=strategy_name,
            equity=equity,
            daily=daily,
            trades=trade_df,
            rejections=rejection_df,
            targets=targets,
            held=held,
            report=report,
            benchmark=benchmark,
            config=self.config,
        )

    def run_strategy(self, bars: dict[str, pd.DataFrame], strategy) -> BacktestResult:
        weights = strategy.generate(bars)
        return self.run(bars, weights, strategy_name=strategy.describe())

    # -------------------------------------------------------------- 工具
    @staticmethod
    def _calendar(bars: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
        idx: pd.DatetimeIndex | None = None
        for frame in bars.values():
            idx = frame.index if idx is None else idx.union(frame.index)
        assert idx is not None
        return pd.DatetimeIndex(idx.sort_values().unique(), name="date")

    @staticmethod
    def _price_map(bars: Mapping[str, pd.DataFrame], field: str, calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, dict[str, float]]:
        out: dict[pd.Timestamp, dict[str, float]] = {}
        for symbol, frame in bars.items():
            if field not in frame.columns:
                continue
            series = frame[field].reindex(calendar)
            for date, value in series.items():
                if pd.notna(value):
                    out.setdefault(date, {})[symbol] = float(value)
        return out

    def _benchmark_equity(self, bars: Mapping[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> pd.Series | None:
        symbol = self.config.benchmark
        if self.benchmark_prices is not None:
            close = pd.Series(self.benchmark_prices).sort_index().reindex(calendar).ffill()
        elif not symbol:
            return None
        elif symbol in bars:
            close = bars[symbol]["close"].reindex(calendar).ffill()
        else:
            # 未提供基准数据 → 用全池等权指数代替
            mat = pd.DataFrame({s: f["close"].reindex(calendar).ffill() for s, f in bars.items()})
            close = mat.pct_change().mean(axis=1).add(1.0).cumprod()
        if close.dropna().empty:
            return None
        close = close.dropna()
        return (close / close.iloc[0] * float(self.config.initial_cash)).rename("benchmark")


def run_backtest(
    bars: dict[str, pd.DataFrame],
    strategy,
    config: BacktestConfig | None = None,
    *,
    overlays: Iterable[RiskOverlay] | None = None,
    benchmark_prices: pd.Series | None = None,
) -> BacktestResult:
    """一行跑完：``run_backtest(bars, MaCross(10, 30))``。"""
    config = config or BacktestConfig()
    engine = Engine(
        config=config,
        overlays=list(overlays or []),
        verbose=True,
        benchmark_prices=benchmark_prices,
    )
    return engine.run_strategy(bars, strategy)
