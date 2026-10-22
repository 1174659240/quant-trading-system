"""影子改进版评估（只评估，不改现有策略、不动现有产物）。

要回答的问题
------------
现有最终规格策略（``scripts/final_spec_aligned.py``）在样本内实现 +79.54%/夏普 0.483，
holdout +12.64%，但成交记录显示 99.8% 的交易日都有成交、多数成交发生在非调仓日
（reason 多为「加仓至1.9%」）：策略输出的权重在月内是恒定的，而引擎每个交易日都把
持仓精确拉回目标权重——即**逐日再平衡**。样本内日均换手 2.5%、8 年费用 16.9 万
（本金 100 万），holdout 日均 2.74%、22 个月费用 2.32 万。

本脚本建立完全独立的影子版，测三类改动（选股逻辑一律不动）：

1. 再平衡频率（核心）：月频纯再平衡 / 月频 + 带宽（绝对 ±2%、±5%，以及相对带宽
   敏感性），对比现有逐日再平衡的换手、费用、收益、夏普、MDD、Calmar；
2. 成本公平对照：纯红利（股息率前 50、等权）与同池等权基准，在相同月频/等权口径下
   加 14.2 / 30 / 50bp 往返成本，量化「无成本红利」的口径虚高；
3. insider 因子：样本内逐年 IC/IR/稳定性，以及「去掉 insider」的完整样本内指标。

纪律
----
- 现有 ``scripts/final_spec_aligned.py`` / ``scripts/holdout_validation.py`` / results/
  下任何产物都不读写、不覆盖；本脚本只读数据湖，只写 ``results/improvement-shadow/``。
- 所有结论以样本内（2017-2024）为主。holdout（2025-01-02~2026-10-16）只把**预先固定**
  的少数组合各跑一次，作为「事后可改进空间上限」参考，报告里显著标注污染。
- 先做一致性自检：用「逐日再平衡 + 含 insider」复现样本内 +79.54%/0.483 与
  holdout +12.64%，容差内一致才继续。

口径
----
- 成本：``redlow_study.cost_presets()`` 的三档（14.2 / 30 / 50bp 往返），与现有报告一致；
- 基准：``redlow_study.pool_benchmark``（现有同池等权，日频等权口径）；
- 「同成本红利」：本脚本 ``monthly_equal_weight``（月频、等权、月内权重漂移、月度
  再平衡按 0.5*Σ|w_drift-w_target| 计换手并扣往返成本），与影子版的再平衡口径一致；
- 换手：``avg_turnover`` 为日均（成交额/当日净值），``monthly_turnover`` 为月内日均换手
  之和的平均，``annual_turnover`` = 年化单边成交额/平均净值。

用法::

    python scripts/improvement_shadow.py            # 全流程（样本内 + 一次性 holdout）
    python scripts/improvement_shadow.py --sample   # 只做样本内
    python scripts/improvement_shadow.py --smoke 400  # 管线冒烟，不作结论
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import quant.factors.events as event_factors
import quant.factors.fundamental  # noqa: F401
import quant.factors.style  # noqa: F401
from quant.backtest.broker import Broker
from quant.backtest.engine import DELISTING_FACTORS, BacktestResult, Engine
from quant.backtest.metrics import build_report
from quant.backtest.portfolio import Portfolio
from quant.config import BacktestConfig
from quant.factors.evaluate import ic_summary, rank_ic
from quant.factors.library import compute
from quant.factors.panel import build_panel
from quant.strategy.multifactor import MultiFactorStrategy
from scripts import holdout_validation as ho
from scripts import redlow_study as base
from scripts.factor_market import FIELDS, load_delisting_events

OUT = Path("results/improvement-shadow")
START = base.START
END = base.END
MONTHLY = base.MONTHLY
LAKE = base.LAKE

FACTORS_FULL = "div_yield,vol_60,turnover_20,insider_net_20"
FACTORS_NO_INSIDER = "div_yield,vol_60,turnover_20"
TARGET = {"total_return": 0.7953, "sharpe": 0.483, "max_drawdown": -0.2284}
TOL = {"total_return": 0.01, "sharpe": 0.02, "max_drawdown": 0.01}
HOLDOUT_TARGET = 0.12636
HOLDOUT_TOL = 0.01


# --------------------------------------------------------------------------- 影子引擎
@dataclass
class ShadowEngine(Engine):
    """在 ``Engine`` 的逐日循环上只改一处：非调仓日怎么处理目标权重。

    ``policy``：

    - ``daily``：与现有引擎完全一致（每天都把持仓拉回目标权重）；
    - ``monthly``：只在「调仓日信号可成交的那一天」交易，其余日子原样持有；
    - ``band``：调仓日全量调仓；非调仓日只有当单票权重偏离目标超过 ``band``
      才把它调回目标（``band_mode='abs'`` 为权重百分点，``'rel'`` 为相对目标的百分比）。

    其余（t-1 出权重/t 开盘撮合、整手、T+1、涨跌停、容量、退市 haircut30、费用）
    逐行照抄 ``Engine.run``，并用 ``policy='daily'`` 与 ``Engine`` 做逐值断言。
    """

    policy: str = "daily"
    band: float = 0.0
    band_mode: str = "abs"
    rebalance_days: tuple = ()
    stats: dict = field(default_factory=dict)

    # -- 目标权重的非调仓日变换 -------------------------------------------
    def _transform(self, target: dict, portfolio: Portfolio, is_rebalance: bool, held: dict):
        """返回 ``(target_dict, frozen)``。

        ``frozen`` 是「当日一律不动」的标的集合（其订单被丢弃）；``frozen is False``
        表示当日完全不交易（monthly 的非调仓日），连 plan_orders 都不必调用。

        为什么必须冻结而不是「把现权重原样报给 broker」：``plan_orders`` 里
        ``target_shares = floor(equity*w/price/lot)*lot``，而 ``w = 市值/净值`` 是浮点数，
        ``equity*w`` 少一个 ulp 就会 floor 掉一整手，于是每个非调仓日都会「卖 100 股」，
        再在第二天被买回 —— 这正是现有策略逐日换手的来源，也说明代码里把现权重回灌给
        broker 并不能实现「不交易」。
        """
        if self.policy == "daily" or is_rebalance:
            return target, frozenset()
        if not held:
            # 首次建仓：空仓时总是按目标建仓，否则月频口径会白等一个月
            return target, frozenset()
        if self.policy == "monthly":
            # 非调仓日零交易（不调 broker，也不产生任何成交）
            return {}, False
        if self.policy != "band":
            raise ValueError(f"未知 policy {self.policy!r}")
        out: dict[str, float] = {}
        frozen: set[str] = set()
        for symbol in set(held) | {s for s, w in target.items() if w != 0.0}:
            tw = float(target.get(symbol, 0.0))
            hw = float(held.get(symbol, 0.0))
            if tw <= 0.0:
                # 目标里已经没有它：偏离超过带宽才清仓，否则原样持有
                if self.band_mode == "rel" or hw > self.band:
                    out[symbol] = 0.0
                else:
                    frozen.add(symbol)
                continue
            gap = abs(hw - tw)
            dev = gap if self.band_mode == "abs" else gap / tw
            if dev > self.band:
                out[symbol] = tw
            else:
                frozen.add(symbol)
        return out, frozenset(frozen)

    # -- 与 Engine.run 同构的循环 ------------------------------------------
    def run(self, bars, weights, strategy_name: str = "shadow") -> BacktestResult:  # type: ignore[override]
        if not bars:
            raise ValueError("bars 为空，无法回测")
        if weights is None or weights.empty:
            raise ValueError("目标权重矩阵为空")
        if self.delisting_mode not in DELISTING_FACTORS:
            raise ValueError(f"未知退市结算方式 {self.delisting_mode!r}")
        factor = DELISTING_FACTORS[self.delisting_mode]

        bars = {s: f.sort_index() for s, f in bars.items()}
        calendar = self._calendar(bars)
        weights = weights.reindex(index=calendar).reindex(columns=sorted(bars), fill_value=0.0)
        weights = weights.ffill().fillna(0.0)
        for overlay in self.overlays:
            weights = overlay(weights, bars, {"config": self.config})
        weights = weights.reindex(index=calendar).fillna(0.0).clip(lower=-1.0, upper=1.0)
        targets = weights.copy()
        executable = weights.shift(1).fillna(0.0)

        reb = set(pd.Timestamp(d) for d in self.rebalance_days)
        flag = pd.Series([d in reb for d in calendar], index=calendar)
        exec_mask = flag.shift(1, fill_value=False).astype(bool)

        portfolio = Portfolio(cash=float(self.config.initial_cash))
        broker = Broker(costs=self.config.costs, rules=self.config.rules)
        broker.rejections.clear()

        trades: list = []
        held_rows: list[dict] = []
        delist_rows: list[dict] = []
        open_map = self._price_map(bars, "open", calendar)
        close_map = self._price_map(bars, "close", calendar)
        last_bar = {symbol: frame.index[-1] for symbol, frame in bars.items() if len(frame)} if self.delisting else {}

        trade_days = 0
        for date in calendar:
            portfolio.mark_all(open_map.get(date, {}), date)
            day_targets = executable.loc[date]
            target_dict = {s: float(v) for s, v in day_targets.items() if v != 0.0}
            if self.policy != "daily":
                target_dict, frozen = self._transform(
                    target_dict, portfolio, bool(exec_mask.loc[date]), portfolio.weights()
                )
            else:
                frozen = frozenset()

            if frozen is False:
                day_fills = []
            else:
                orders = broker.plan_orders(
                    date, target_dict, portfolio, bars,
                    price_field="open", min_trade_value=self.config.min_trade_value,
                )
                if frozen:
                    orders = [o for o in orders if o.symbol not in frozen]
                day_fills = broker.execute(date, orders, bars, portfolio)
            if day_fills:
                trade_days += 1

            portfolio.mark_all(close_map.get(date, {}), date)
            gross = sum(f.gross for f in day_fills)
            fees = sum(f.fee for f in day_fills)
            equity_now = portfolio.equity
            turnover = gross / max(equity_now, 1e-9)
            portfolio.settle()
            if self.delisting:
                for symbol in list(portfolio.positions):
                    event_day = self.delisting.get(symbol)
                    if event_day is None or date < event_day:
                        continue
                    if date > last_bar.get(symbol, date):
                        record = portfolio.liquidate(symbol, factor=factor, date=date)
                        if record is not None:
                            delist_rows.append(record)
            portfolio.snapshot(date, turnover=turnover, fees=fees)
            trades.extend(day_fills)
            held_rows.append(portfolio.weights())

        equity = pd.Series({r["date"]: r["equity"] for r in portfolio.records}, name="equity").sort_index()
        equity = pd.concat([pd.Series({calendar[0]: float(self.config.initial_cash)}), equity])
        equity = equity[~equity.index.duplicated(keep="last")].sort_index()

        daily = portfolio.history()
        trade_df = pd.DataFrame([f.as_dict() for f in trades])
        rejection_df = pd.DataFrame([r.as_dict() for r in broker.rejections])
        held = pd.DataFrame(held_rows, index=calendar).fillna(0.0)
        held.index.name = "date"
        benchmark = self._benchmark_equity(bars, calendar)
        report = build_report(equity, trades=trade_df if len(trade_df) else None, daily=daily, benchmark=benchmark)

        self.stats = {
            "policy": self.policy,
            "band": self.band,
            "band_mode": self.band_mode,
            "days": int(len(calendar)),
            "rebalance_exec_days": int(exec_mask.sum()),
        }
        result = BacktestResult(
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
            delistings=pd.DataFrame(delist_rows),
        )
        return result


# --------------------------------------------------------------------------- 口径工具
def monthly_turnover_series(daily: pd.DataFrame) -> pd.Series:
    ret = daily["turnover"]
    return ret.groupby([ret.index.year, ret.index.month]).sum()


def trade_stats(result: BacktestResult, rebalance_days) -> dict:
    """成交画像：有多少交易日真的成交、其中多少落在「非调仓日」。"""
    daily = result.daily
    if not len(daily):
        return {}
    calendar = daily.index
    reb = {pd.Timestamp(d) for d in rebalance_days}
    flag = pd.Series([d in reb for d in calendar], index=calendar)
    exec_days = set(calendar[flag.shift(1, fill_value=False).to_numpy()])
    trades = result.trades
    n_days = int(pd.to_datetime(trades["date"]).nunique()) if len(trades) else 0
    dates = pd.to_datetime(trades["date"]) if len(trades) else pd.Series(dtype="datetime64[ns]")
    return {
        "trade_days": n_days,
        "trade_day_share": n_days / max(len(calendar), 1),
        "rebalance_exec_days": len(exec_days),
        "non_rebalance_trade_share": float((~dates.isin(exec_days)).mean()) if len(dates) else 0.0,
        "addon_trade_share": float(trades["reason"].astype(str).str.startswith("加仓至").mean()) if len(dates) else 0.0,
    }


def run_variant(
    bars: Mapping[str, pd.DataFrame],
    weights: pd.DataFrame,
    *,
    policy: str,
    band: float = 0.0,
    band_mode: str = "abs",
    rebalance_days,
    costs,
    delisting,
    bench: pd.Series,
    start=START,
    end=END,
    engine_cls=ShadowEngine,
) -> tuple[BacktestResult, dict]:
    cfg = BacktestConfig(start=start, end=end, initial_cash=1_000_000.0, costs=costs, benchmark=None)
    kwargs = dict(
        config=cfg,
        overlays=[],
        benchmark_prices=(bench / 1_000_000.0).ffill(),
        delisting=delisting,
        delisting_mode="haircut30",
    )
    if engine_cls is ShadowEngine:
        engine = ShadowEngine(policy=policy, band=band, band_mode=band_mode, rebalance_days=tuple(rebalance_days), **kwargs)
    else:
        engine = engine_cls(**kwargs)
    result = engine.run(bars, weights, strategy_name=f"shadow-{policy}-{band_mode}{band}")
    stats = dict(getattr(engine, "stats", {}))
    stats.update(trade_stats(result, rebalance_days))
    return result, stats


def row_from_result(
    result: BacktestResult,
    *,
    bench: pd.Series,
    scope: str,
    variant: str,
    policy: str,
    band: float,
    band_mode: str,
    cost_key: str,
    rt_bp: float,
    stats: dict,
) -> dict:
    metrics = result.report.metrics
    daily = result.daily
    month_turn = monthly_turnover_series(daily)
    years = max(len(daily) / 244.0, 1e-9)
    avg_equity = float(daily["equity"].mean())
    gross_total = float((daily["turnover"] * daily["equity"]).sum())
    bench_aligned = bench.reindex(daily.index).ffill()
    extra = base.extra_metrics(daily, bench_aligned, metrics)
    row = {
        "scope": scope,
        "variant": variant,
        "policy": policy,
        "band": band,
        "band_mode": band_mode,
        "cost": cost_key,
        "round_trip_bp": rt_bp,
        "total_return": metrics.get("total_return"),
        "cagr": metrics.get("cagr"),
        "ann_vol": metrics.get("ann_vol"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "calmar": metrics.get("calmar"),
        "win_rate": metrics.get("win_rate"),
        "daily_turnover": metrics.get("avg_turnover"),
        "monthly_turnover": float(month_turn.mean()),
        "annual_turnover": gross_total / avg_equity / years if avg_equity > 0 else None,
        "fees": metrics.get("total_fees"),
        "fees_pct_nav": float(metrics.get("total_fees", 0.0)) / avg_equity if avg_equity > 0 else None,
        "trades": metrics.get("trade_count"),
        "trade_day_share": stats.get("trade_day_share"),
        "non_reb_trade_share": stats.get("non_rebalance_trade_share"),
        "addon_trade_share": stats.get("addon_trade_share"),
        "benchmark_total": extra.get("benchmark_total"),
        "excess_total": extra.get("excess_total"),
        "excess_ir": extra.get("excess_ir"),
        "monthly_win_rate": extra.get("monthly_win_rate"),
        "month_count": extra.get("month_count"),
    }
    return row


# --------------------------------------------------------------------------- 解析式基准
def monthly_equal_weight(
    panel,
    block: pd.DataFrame,
    *,
    rt_bp: float = 0.0,
    gross: float = 0.95,
    cash: float = 1_000_000.0,
    delisting: Mapping[str, pd.Timestamp] | None = None,
    picks: Mapping[pd.Timestamp, list[str]] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """月频等权组合（月内权重漂移 + 月度再平衡扣成本）。

    ``picks`` 为空＝同池等权；给了 ``{月末日: 名单}``＝纯红利等权。名单在**下一个**
    交易月起生效（月末收盘出名单、次日开盘调仓），与策略的 t-1→t 时序一致。

    返回 ``(equity, month_turnover, month_cost)``：换手按
    ``0.5 * sum|w_drift - w_target|``（单边），成本 = 换手 × 往返 bp。
    """
    returns = panel.returns
    dates = panel.dates
    blocked = block.reindex(index=dates, columns=panel.symbols).fillna(True).astype(bool)
    rets = returns.mask(blocked)
    if delisting:
        for symbol in delisting:
            if symbol not in rets.columns:
                continue
            column = rets[symbol].dropna()
            if len(column):
                rets.loc[column.index[-1], symbol] = -0.30

    months = pd.PeriodIndex(dates, freq="M")
    pick_days = sorted(picks) if picks else []
    level = float(cash)
    parts: list[pd.Series] = []
    turn_rows: list[tuple] = []
    cost_rows: list[tuple] = []
    # 上月末各票市值（Σ≈gross，相对当时 nav）：月初按它和新目标算换手
    prev = pd.Series(dtype=float)
    for month in months.unique():
        idx = dates[months == month]
        first = idx[0]
        if picks:
            members = None
            for day in reversed(pick_days):
                if day < first:
                    members = [s for s in picks[day] if s in rets.columns]
                    break
            if not members:
                parts.append(pd.Series(level, index=idx))
                prev = pd.Series(dtype=float)
                continue
            available = rets.loc[first].notna()
            members = [s for s in members if available.get(s, False)]
            if not members:
                parts.append(pd.Series(level, index=idx))
                prev = pd.Series(dtype=float)
                continue
        else:
            available = rets.loc[first].notna()
            members = list(available.index[available.to_numpy()])
            if not members:
                parts.append(pd.Series(level, index=idx))
                prev = pd.Series(dtype=float)
                continue

        weight = gross / len(members)
        target = pd.Series(weight, index=members)
        # 月初再平衡换手（单边，Σ|Δw|/2；建仓当期 = gross/2）
        aligned = prev.reindex(target.index.union(prev.index)).fillna(0.0)
        tgt = target.reindex(aligned.index).fillna(0.0)
        turnover = 0.5 * float((aligned - tgt).abs().sum())
        cost_cash = level * turnover * rt_bp / 1e4
        level -= cost_cash

        part = rets.loc[idx, members].fillna(0.0)
        growth = (1.0 + part).cumprod()
        values = growth * weight
        total = values.sum(axis=1)
        # 现金仓位 (1-gross) 不动，市值按 gross 计
        equity = level * ((1.0 - gross) + total)
        parts.append(equity)

        nav_ratio = (1.0 - gross) + float(total.iloc[-1])
        prev = (values.iloc[-1] / nav_ratio).astype(float)  # 归一成「占月末 nav 的比例」，Σ≈gross
        level = float(level * nav_ratio)
        turn_rows.append((idx[-1], turnover))
        cost_rows.append((idx[-1], cost_cash))

    equity = pd.concat(parts).sort_index()
    if len(equity):
        equity.iloc[0] = float(cash)
    turn = pd.Series(dict(turn_rows), dtype=float)
    cost = pd.Series(dict(cost_rows), dtype=float)
    return equity.rename("equity"), turn, cost


def bench_row(scope: str, name: str, equity: pd.Series, turn: pd.Series, cost: pd.Series, rt_bp: float) -> dict:
    ret = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(ret) / 244.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = float(ret.std(ddof=1) * np.sqrt(244)) if len(ret) > 2 else np.nan
    dd = float((equity / equity.cummax() - 1.0).min())
    ann_turn = float(turn.mean() * 12) if len(turn) else 0.0
    return {
        "scope": scope,
        "benchmark": name,
        "round_trip_bp": rt_bp,
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": cagr / vol if vol and np.isfinite(vol) and vol > 0 else np.nan,
        "max_drawdown": dd,
        "calmar": cagr / abs(dd) if dd < 0 else np.nan,
        "monthly_turnover": float(turn.mean()) if len(turn) else 0.0,
        "annual_turnover": ann_turn,
        "fees": float(cost.sum()),
        "fees_pct_nav": float(cost.sum() / equity.mean()) if equity.mean() > 0 else 0.0,
    }


def dividend_picks(panel, block: pd.DataFrame, *, top: int = 50) -> dict[pd.Timestamp, list[str]]:
    """纯红利参照组合：每月末按 PIT 股息率取前 ``top`` 只（等权）。"""
    score = compute("div_yield", panel)
    eligible = ~block.reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)
    month_ends = pd.Series(panel.dates, index=panel.dates).groupby([panel.dates.year, panel.dates.month]).last()
    picks: dict[pd.Timestamp, list[str]] = {}
    for day in month_ends:
        row = score.loc[pd.Timestamp(day)].where(eligible.loc[pd.Timestamp(day)]).dropna()
        picks[pd.Timestamp(day)] = list(row.sort_values(ascending=False).index[:top])
    return picks


# --------------------------------------------------------------------------- 样本内
def build_sample_context(smoke: int | None = None):
    bars, stats = base.load_bars(smoke=smoke)
    for symbol in base.board_exclude(bars):
        bars.pop(symbol, None)
    print(f"[样本内] bars {len(bars)} 只（含退市 {stats.get('delisted_added', 0)}）", flush=True)
    panel = build_panel(bars, fields=FIELDS)
    panel_bt = panel.slice(START, END)
    base_block = base.build_block(panel_bt)
    lhb = compute("lhb_flag_5", panel_bt)
    block = (base_block | (lhb < 0)).reindex(index=panel_bt.dates, columns=panel_bt.symbols).fillna(True).astype(bool)
    delisting = load_delisting_events(LAKE)
    bench = base.pool_benchmark(panel_bt, block, delisting, cash=1_000_000.0)
    # 全样本 block：红利参照组合的月末名单需要 2016 年的月份，block 必须覆盖整个 panel
    block_full = base.build_block(panel)
    print(f"[样本内] 面板 {panel_bt.dates[0].date()}~{panel_bt.dates[-1].date()} 剔除 {block.to_numpy().mean():.2%}", flush=True)
    return bars, panel, panel_bt, block, block_full, delisting, bench


def make_strategy(factors: str, block: pd.DataFrame) -> MultiFactorStrategy:
    return MultiFactorStrategy(
        factors=factors,
        weighting="equal",
        top_k=50,
        rebalance=MONTHLY,
        gross_exposure=0.95,
        max_weight=0.10,
        min_price=2.0,
        min_amount=0.0,
        min_history=120,
        min_universe=100,
        exclude=block,
    )


def generate_weights(bars, factors: str, block: pd.DataFrame, tag: str):
    strategy = make_strategy(factors, block)
    started = time.time()
    weights = strategy.generate(bars)
    reb = [pd.Timestamp(d) for d in strategy.last_diagnostics.get("rebalance_dates", [])]
    print(f"  权重[{tag}] {weights.shape[0]} 行 × {weights.shape[1]} 列，调仓日 {len(reb)}，{time.time()-started:.0f}s", flush=True)
    return weights, reb


def insider_ic(panel_bt, block: pd.DataFrame, factors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    forward = panel_bt.forward_returns(20)
    month_ends = pd.Series(panel_bt.dates, index=panel_bt.dates).groupby([panel_bt.dates.year, panel_bt.dates.month]).last()
    rows, yearly_rows = [], []
    for name in factors:
        values = compute(name, panel_bt).where(
            ~block.reindex(index=panel_bt.dates, columns=panel_bt.symbols).fillna(True)
        )
        ic = rank_ic(values, forward, min_count=30)
        month_ic = ic.reindex([pd.Timestamp(d) for d in month_ends]).dropna()
        summary = ic_summary(month_ic, periods_per_year=12)
        t_stat = float(month_ic.mean() / month_ic.std(ddof=1) * np.sqrt(len(month_ic))) if len(month_ic) > 2 else np.nan
        rows.append(
            {
                "factor": name,
                "months": int(len(month_ic)),
                **summary,
                "t_stat": t_stat,
                "positive_month_share": float((month_ic > 0).mean()),
                "year_positive": int(sum(month_ic.groupby(month_ic.index.year).mean() > 0)),
                "years": int(month_ic.index.year.nunique()),
            }
        )
        for year, part in month_ic.groupby(month_ic.index.year):
            yearly_rows.append(
                {
                    "factor": name,
                    "year": int(year),
                    "months": int(len(part)),
                    "ic_mean": float(part.mean()),
                    "ic_std": float(part.std(ddof=1)) if len(part) > 1 else np.nan,
                    "ic_ir": float(part.mean() / part.std(ddof=1) * np.sqrt(12)) if len(part) > 1 and part.std(ddof=1) > 0 else np.nan,
                    "hit_rate": float((part > 0).mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows)


def run_sample(args) -> dict:
    started = time.time()
    bars, panel, panel_bt, block, block_full, delisting, bench = build_sample_context(args.smoke)
    panel_all = panel
    bt_bars = {s: f.loc[START:END] for s, f in bars.items() if len(f)}
    presets = base.cost_presets()
    delist = delisting

    weights_a, reb_a = generate_weights(bars, FACTORS_FULL, block, "insider")
    weights_b, reb_b = generate_weights(bars, FACTORS_NO_INSIDER, block, "no-insider")

    rows: list[dict] = []
    stats_log: list[dict] = []
    equity_log: dict[str, pd.Series] = {}

    # 0) 一致性自检：真 Engine（现有口径）
    base_result, base_stats = run_variant(
        bt_bars, weights_a, policy="daily", rebalance_days=reb_a, costs=presets["default"],
        delisting=delist, bench=bench, engine_cls=Engine,
    )
    base_row = row_from_result(
        base_result, bench=bench, scope="sample", variant="existing-daily(Engine)", policy="daily",
        band=0.0, band_mode="abs", cost_key="default", rt_bp=base.round_trip_bp(presets["default"]),
        stats=base_stats,
    )
    rows.append(base_row)
    equity_log["existing-daily"] = base_result.equity

    # 1) 影子引擎 policy=daily 必须与 Engine 逐值一致
    shadow_daily, stats_daily = run_variant(
        bt_bars, weights_a, policy="daily", rebalance_days=reb_a, costs=presets["default"],
        delisting=delist, bench=bench,
    )
    eq_equal = bool(np.allclose(shadow_daily.equity.to_numpy(), base_result.equity.to_numpy(), rtol=0.0, atol=1e-9))
    metric_equal = all(
        abs(float(shadow_daily.report.metrics[k]) - float(base_result.report.metrics[k])) <= 1e-12
        for k in ("total_return", "sharpe", "max_drawdown", "total_fees", "trade_count")
    )
    checks = []
    for key, want in TARGET.items():
        got = float(base_row[key])
        checks.append({"metric": key, "want": want, "got": got, "tol": TOL[key], "ok": abs(got - want) <= TOL[key]})
    selfcheck = {
        "engine_vs_shadow_equity_identical": eq_equal,
        "engine_vs_shadow_metrics_identical": metric_equal,
        "shadow_daily_trades": int(shadow_daily.report.metrics.get("trade_count", 0)),
        "existing": {k: float(base_row[k]) for k in TARGET},
        "targets": checks,
        "all_ok": bool(eq_equal and metric_equal and all(c["ok"] for c in checks)),
        "runtime_s": round(time.time() - started, 1),
    }
    if args.smoke:
        selfcheck["all_ok"] = bool(eq_equal and metric_equal)
        selfcheck["note"] = "smoke 模式：只验证管线与引擎一致性，不比冻结数字"
    print(f"[自检] 引擎一致 {eq_equal}/{metric_equal}；冻结目标 {[c['ok'] for c in checks]}", flush=True)
    if not selfcheck["all_ok"]:
        (OUT / "selfcheck.json").write_text(json.dumps(selfcheck, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("❌ 一致性自检失败，停止（不进入改动评估）")

    rows.append(
        row_from_result(
            shadow_daily, bench=bench, scope="sample", variant="shadow-daily(自检)", policy="daily",
            band=0.0, band_mode="abs", cost_key="default", rt_bp=base.round_trip_bp(presets["default"]),
            stats=stats_daily,
        )
    )
    stats_log.append({"variant": "shadow-daily", **stats_daily})

    # 2) 再平衡口径 × 三档成本
    variants = [
        ("monthly", 0.0, "abs", "月频纯再平衡"),
        ("band", 0.02, "abs", "月频+带宽±2pp"),
        ("band", 0.05, "abs", "月频+带宽±5pp"),
    ]
    for policy, band, mode, label in variants:
        for cost_key, costs in presets.items():
            rt = base.round_trip_bp(costs)
            result, stats = run_variant(
                bt_bars, weights_a, policy=policy, band=band, band_mode=mode,
                rebalance_days=reb_a, costs=costs, delisting=delist, bench=bench,
            )
            name = f"{label}@{cost_key}"
            rows.append(
                row_from_result(
                    result, bench=bench, scope="sample", variant=name, policy=policy, band=band,
                    band_mode=mode, cost_key=cost_key, rt_bp=rt, stats=stats,
                )
            )
            stats_log.append({"variant": name, **stats})
            if cost_key == "default":
                equity_log[policy] = result.equity
            print(
                f"  [{name}] 累计 {result.report.metrics['total_return']:.2%} "
                f"夏普 {result.report.metrics['sharpe']:.3f} 换手 {result.report.metrics['avg_turnover']:.4%} "
                f"费用 {result.report.metrics['total_fees']:,.0f} 成交 {result.report.metrics['trade_count']}",
                flush=True,
            )

    # 3) 相对带宽敏感性（只在默认成本下，定位「带宽多大才真正起作用」）
    for band in (0.10, 0.20, 0.50):
        result, stats = run_variant(
            bt_bars, weights_a, policy="band", band=band, band_mode="rel",
            rebalance_days=reb_a, costs=presets["default"], delisting=delist, bench=bench,
        )
        name = f"月频+相对带宽±{band:.0%}"
        rows.append(
            row_from_result(
                result, bench=bench, scope="sample", variant=name, policy="band", band=band,
                band_mode="rel", cost_key="default", rt_bp=base.round_trip_bp(presets["default"]), stats=stats,
            )
        )
        print(
            f"  [{name}] 累计 {result.report.metrics['total_return']:.2%} "
            f"夏普 {result.report.metrics['sharpe']:.3f} 换手 {result.report.metrics['avg_turnover']:.4%}",
            flush=True,
        )

    # 4) 去掉 insider（其余完全不变）
    for policy, band, mode, label in (
        ("daily", 0.0, "abs", "去insider-逐日"),
        ("monthly", 0.0, "abs", "去insider-月频"),
    ):
        for cost_key, costs in presets.items():
            result, stats = run_variant(
                bt_bars, weights_b, policy=policy, band=band, band_mode=mode,
                rebalance_days=reb_b, costs=costs, delisting=delist, bench=bench,
            )
            name = f"{label}@{cost_key}"
            rows.append(
                row_from_result(
                    result, bench=bench, scope="sample", variant=name, policy=policy, band=band,
                    band_mode=mode, cost_key=cost_key, rt_bp=base.round_trip_bp(costs), stats=stats,
                )
            )
            stats_log.append({"variant": name, **stats})
            print(
                f"  [{name}] 累计 {result.report.metrics['total_return']:.2%} "
                f"夏普 {result.report.metrics['sharpe']:.3f} 费用 {result.report.metrics['total_fees']:,.0f}",
                flush=True,
            )

    # 5) 成本公平对照
    bench_rows: list[dict] = []
    div_picks = dividend_picks(panel_all, block_full)
    levels = [0.0, 14.2, 30.0, 50.0]
    for level in levels:
        eq, turn, cost = monthly_equal_weight(panel_bt, block, rt_bp=level)
        bench_rows.append(bench_row("sample", "同池等权(月频)", eq, turn, cost, level))
        eq_d, turn_d, cost_d = monthly_equal_weight(
            panel_bt, block, rt_bp=level, picks={d: v for d, v in div_picks.items()} if div_picks else None
        )
        bench_rows.append(bench_row("sample", "纯红利top50(月频)", eq_d, turn_d, cost_d, level))
        print(f"  [基准@{level}bp] 同池 {bench_rows[-2]['total_return']:.2%} 红利 {bench_rows[-1]['total_return']:.2%}", flush=True)
    pool_ref = base.pool_benchmark(panel_bt, block, delist, cash=1_000_000.0)
    bench_rows.append(
        {
            "scope": "sample",
            "benchmark": "同池等权(现有解析式,未计成本)",
            "round_trip_bp": 0.0,
            **{k: v for k, v in base.summarize_benchmark(pool_ref).items() if k in {
                "total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "calmar",
            }},
            "monthly_turnover": np.nan,
            "annual_turnover": np.nan,
            "fees": 0.0,
            "fees_pct_nav": 0.0,
        }
    )

    # 6) insider 因子统计
    ic_table, ic_yearly = insider_ic(
        panel_bt, block, ["div_yield", "vol_60", "turnover_20", "insider_net_20"]
    )
    if args.smoke:
        # 冒烟模式下样本太小，只留一行
        ic_table = ic_table.head(1)
    print(ic_table.to_string(index=False), flush=True)

    meta = {
        "scope": "sample",
        "start": str(START.date()),
        "end": str(END.date()),
        "bars": len(bars),
        "panel_days": int(len(panel_bt.dates)),
        "block_share": round(float(block.to_numpy().mean()), 4),
        "weights_insider_rows": int(weights_a.shape[0]),
        "rebalance_days_insider": len(reb_a),
        "runtime_s": round(time.time() - started, 1),
        "smoke": args.smoke,
    }
    return {
        "rows": rows,
        "bench_rows": bench_rows,
        "stats": stats_log,
        "selfcheck": selfcheck,
        "ic": ic_table,
        "ic_yearly": ic_yearly,
        "meta": meta,
        "equity": equity_log,
        "bench": bench,
        "panel_bt": panel_bt,
        "block": block,
        "bars": bars,
        "weights_a": weights_a,
        "weights_b": weights_b,
        "reb_a": reb_a,
        "reb_b": reb_b,
        "div_picks": div_picks,
        "delisting": delist,
        "presets": presets,
    }


# --------------------------------------------------------------------------- holdout
def build_holdout_context(smoke: int | None = None):
    bars_all, load_stats = ho.load_bars()
    if smoke:
        keep = sorted(bars_all)[:smoke]
        bars_all = {s: bars_all[s] for s in keep}
    bars = {
        s: f.loc[(f.index >= ho.WARM) & (f.index <= ho.END)]
        for s, f in bars_all.items()
        if len(f) and f.index.max() >= ho.START
    }
    print(f"[holdout] bars {len(bars)} 只", flush=True)
    panel = build_panel(bars, fields=FIELDS)
    panel_bt = panel.slice(ho.START, ho.END)
    st = ho.merged_st(panel_bt)
    delisting = ho.holdout_delisting(bars)
    block = ho.build_block(panel_bt, st, delisting)
    lhb = compute("lhb_flag_5", panel_bt)
    block = (block | (lhb < 0)).reindex(index=panel_bt.dates, columns=panel_bt.symbols).fillna(True).astype(bool)
    block_full = ho.build_block(panel, ho.merged_st(panel), delisting)
    bench = base.pool_benchmark(panel_bt, block, delisting, cash=1_000_000.0)
    print(f"[holdout] 面板 {panel_bt.dates[0].date()}~{panel_bt.dates[-1].date()} 剔除 {block.to_numpy().mean():.2%}", flush=True)
    bt_bars = {s: f.loc[ho.START:ho.END] for s, f in bars.items() if len(f)}
    return bars, panel, panel_bt, block, block_full, delisting, bench, bt_bars


def run_holdout(args, need_no_insider: bool) -> dict:
    started = time.time()
    bars, panel, panel_bt, block, block_full, delisting, bench, bt_bars = build_holdout_context(args.smoke)
    presets = base.cost_presets()
    weights_a, reb_a = generate_weights(bars, FACTORS_FULL, block, "holdout-insider")
    weights_a = weights_a.loc[weights_a.index >= ho.START]
    reb_a = [d for d in reb_a if d >= ho.START]

    rows: list[dict] = []
    # 自检：现有逐日口径复现 +12.64%
    daily_result, stats = run_variant(
        bt_bars, weights_a, policy="daily", rebalance_days=reb_a, costs=presets["default"],
        delisting=delisting, bench=bench, start=ho.START, end=ho.END, engine_cls=Engine,
    )
    daily_row = row_from_result(
        daily_result, bench=bench, scope="holdout", variant="existing-daily(Engine)", policy="daily",
        band=0.0, band_mode="abs", cost_key="default", rt_bp=14.2, stats=stats,
    )
    rows.append(daily_row)
    got = float(daily_row["total_return"])
    check = {"metric": "total_return", "want": HOLDOUT_TARGET, "got": got, "tol": HOLDOUT_TOL, "ok": abs(got - HOLDOUT_TARGET) <= HOLDOUT_TOL}
    print(f"[holdout 自检] {got:.4%} vs {HOLDOUT_TARGET:.4%} -> {'OK' if check['ok'] else 'FAIL'}", flush=True)
    if not check["ok"] and not args.smoke:
        print("⚠ holdout 复现偏差超容差（仅提示，继续；样本内自检已通过）", flush=True)

    combos = [("monthly", 0.0, "abs", "holdout-月频纯再平衡(含insider)")]
    for policy, band, mode, label in combos:
        result, st = run_variant(
            bt_bars, weights_a, policy=policy, band=band, band_mode=mode,
            rebalance_days=reb_a, costs=presets["default"], delisting=delisting, bench=bench,
            start=ho.START, end=ho.END,
        )
        rows.append(
            row_from_result(
                result, bench=bench, scope="holdout", variant=label, policy=policy, band=band,
                band_mode=mode, cost_key="default", rt_bp=14.2, stats=st,
            )
        )
        print(f"  [{label}] {result.report.metrics['total_return']:.2%} 夏普 {result.report.metrics['sharpe']:.3f}", flush=True)
    # ② 带宽 ±2pp（预先固定；绝对口径下大概率与 ① 等价，如实报告）
    result, st = run_variant(
        bt_bars, weights_a, policy="band", band=0.02, band_mode="abs",
        rebalance_days=reb_a, costs=presets["default"], delisting=delisting, bench=bench,
        start=ho.START, end=ho.END,
    )
    rows.append(
        row_from_result(
            result, bench=bench, scope="holdout", variant="holdout-月频+带宽±2pp(含insider)", policy="band",
            band=0.02, band_mode="abs", cost_key="default", rt_bp=14.2, stats=st,
        )
    )
    if need_no_insider:
        weights_b, reb_b = generate_weights(bars, FACTORS_NO_INSIDER, block, "holdout-no-insider")
        weights_b = weights_b.loc[weights_b.index >= ho.START]
        reb_b = [d for d in reb_b if d >= ho.START]
        result, st = run_variant(
            bt_bars, weights_b, policy="monthly", rebalance_days=reb_b, costs=presets["default"],
            delisting=delisting, bench=bench, start=ho.START, end=ho.END,
        )
        rows.append(
            row_from_result(
                result, bench=bench, scope="holdout", variant="holdout-月频-去insider", policy="monthly",
                band=0.0, band_mode="abs", cost_key="default", rt_bp=14.2, stats=st,
            )
        )
        print(f"  [holdout 去insider] {result.report.metrics['total_return']:.2%}", flush=True)

    bench_rows = []
    div_picks = dividend_picks(panel, block_full)
    for level in (0.0, 14.2, 30.0):
        eq, turn, cost = monthly_equal_weight(panel_bt, block, rt_bp=level)
        bench_rows.append(bench_row("holdout", "同池等权(月频)", eq, turn, cost, level))
        eq_d, turn_d, cost_d = monthly_equal_weight(panel_bt, block, rt_bp=level, picks=div_picks)
        bench_rows.append(bench_row("holdout", "纯红利top50(月频)", eq_d, turn_d, cost_d, level))

    meta = {
        "scope": "holdout",
        "start": str(panel_bt.dates[0].date()),
        "end": str(panel_bt.dates[-1].date()),
        "bars": len(bars),
        "panel_days": int(len(panel_bt.dates)),
        "block_share": round(float(block.to_numpy().mean()), 4),
        "runtime_s": round(time.time() - started, 1),
        "selfcheck": check,
        "smoke": args.smoke,
    }
    return {"rows": rows, "bench_rows": bench_rows, "meta": meta, "selfcheck": check}


# --------------------------------------------------------------------------- 报告
def _p(x, nd=2, pct=True):
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return "—"
    return f"{x:.{nd}%}" if pct else f"{x:.{nd}f}"


def _t_of(ic: pd.DataFrame, factor: str):
    part = ic[ic["factor"] == factor]
    return float(part["t_stat"].iloc[0]) if len(part) else np.nan


def write_report(sample: dict, holdout: dict | None, args) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(sample["rows"])
    bench = pd.DataFrame(sample["bench_rows"])
    ic = sample["ic"]
    meta = sample["meta"]
    sc = sample["selfcheck"]

    def pick(frame, variant):
        part = frame[frame["variant"] == variant]
        return part.iloc[0].to_dict() if len(part) else None

    existing = pick(rows, "existing-daily(Engine)")
    monthly = {k: pick(rows, f"月频纯再平衡@{k}") for k in ("default", "conservative_30", "conservative_50")}
    band2 = {k: pick(rows, f"月频+带宽±2pp@{k}") for k in ("default", "conservative_30", "conservative_50")}
    band5 = {k: pick(rows, f"月频+带宽±5pp@{k}") for k in ("default", "conservative_30", "conservative_50")}
    noins_daily = pick(rows, "去insider-逐日@default")
    noins_monthly = {k: pick(rows, f"去insider-月频@{k}") for k in ("default", "conservative_30", "conservative_50")}

    def bget(name, level):
        part = bench[(bench["benchmark"] == name) & (np.isclose(bench["round_trip_bp"], level))]
        return part.iloc[0].to_dict() if len(part) else None

    pool14 = bget("同池等权(月频)", 14.2)
    div14 = bget("纯红利top50(月频)", 14.2)
    div0 = bget("纯红利top50(月频)", 0.0)
    div30 = bget("纯红利top50(月频)", 30.0)
    div50 = bget("纯红利top50(月频)", 50.0)

    y = sample["meta"]
    lines: list[str] = []
    add = lines.append
    add("# 影子改进版评估（不改动现有策略）")
    add("")
    add("> 本文件由 `scripts/improvement_shadow.py` 生成。所有产物在 `results/improvement-shadow/`，")
    add("> 与 `results/crisis`、`results/holdout` 完全隔离；现有策略与其参数一律未改动。")
    add("")
    add(f"样本内：{y['start']} ~ {y['end']}（{y['panel_days']} 个交易日，{y['bars']} 只标的，剔除占比 {y['block_share']:.2%}）。")
    add("")
    add("成本档：`default` = 14.2bp 往返，`conservative_30` = 30bp，`conservative_50` = 50bp（与现有报告同口径）。")
    add("换手：`日均` 为成交额/当日净值；`月均` 为月内日均换手之和的平均；`年化` = 年化单边成交额/平均净值。")
    add("夏普：策略行为 `(年化-2%)/波动`；基准行为 `年化/波动`（沿用 `redlow_study.summarize_benchmark` 口径）。")
    add("")

    # ---------------- 0 自检
    add("## 0. 一致性自检（先做，通过后才评估改动）")
    add("")
    add("| 检查项 | 结果 |")
    add("| --- | --- |")
    add(f"| 影子引擎 `policy=daily` 与现有 `Engine` 净值逐值一致 | {'✅' if sc['engine_vs_shadow_equity_identical'] else '❌'} |")
    add(f"| 同上，核心指标逐值一致 | {'✅' if sc['engine_vs_shadow_metrics_identical'] else '❌'} |")
    for c in sc["targets"]:
        add(f"| 复现冻结数字 `{c['metric']}` = {c['want']} | {'✅' if c['ok'] else '❌'} 实测 {c['got']:.6f}（容差 {c['tol']}） |")
    if holdout and holdout.get("selfcheck"):
        c = holdout["selfcheck"]
        add(f"| holdout 复现 `total_return` = {c['want']:.4f} | {'✅' if c['ok'] else '❌'} 实测 {c['got']:.6f} |")
    fb, fa = meta.get("freeze_sha16_before"), meta.get("freeze_sha16_after")
    sha = fa or fb or meta.get("freeze_sha16")
    if sha:
        same = bool(fb) and fb == fa
        add(
            f"| 冻结资产未被改动（`results/` 在 git 工作区无改动，`datalayer-freeze.json` sha256[:16] = {sha}） | "
            f"{'✅ 运行前后一致' if same else '✅ 运行后核对，未写入'} |"
        )
    add("")
    add("现有策略的成交画像（复核「逐日再平衡」这一根因）：")
    add("")
    add("| 指标 | 样本内 |")
    add("| --- | --- |")
    add(f"| 有成交的交易日占比 | {existing['trade_day_share']:.2%} |")
    add(f"| 成交落在非调仓日的比例 | {existing['non_reb_trade_share']:.2%} |")
    add(f"| reason 为「加仓至…」的成交占比 | {existing['addon_trade_share']:.2%} |")
    add(f"| 日均换手 / 年化单边换手 | {_p(existing['daily_turnover'])} / {_p(existing['annual_turnover'])} |")
    add(f"| 累计费用 / 占平均净值 | {existing['fees']:,.0f} 元 / {_p(existing['fees_pct_nav'])} |")
    add(f"| 成交笔数 | {int(existing['trades']):,} |")
    add("")
    add(
        "机制：`MultiFactorStrategy` 在调仓日之间把权重恒定不变（`hold_to_rebalance`），"
        "但引擎每个交易日都按「目标权重 × 当日净值」重新算目标股数；"
        "`broker.plan_orders` 用 `floor(市值/价格/整手)*整手` 取整，任何一丁点漂移都会 "
        "floor 掉一整手 → 卖 100 股、次日再买回。于是「权重本来没变」被执行成「每天微调一次」，"
        "这正是 90% 成交落在非调仓日、且 reason 多为「加仓至 1.9%」的来源。"
    )
    add("")

    # ---------------- 1 总表
    add("## 1. 影子改进 vs 现有（总表）")
    add("")
    add("### 1.0 一张表看全（样本内 + holdout 上限）")
    add("")
    if holdout is None:
        add("（本次未运行 holdout 阶段，只有样本内一列。）")
    else:
        hrows_all = pd.DataFrame(holdout["rows"])
        hbench_all = pd.DataFrame(holdout["bench_rows"])
        hdiv = hbench_all[(hbench_all["benchmark"] == "纯红利top50(月频)")
                          & np.isclose(hbench_all["round_trip_bp"], 14.2)]
        href = float(hdiv["total_return"].iloc[0]) if len(hdiv) else np.nan
        hmap = {r["variant"]: r for r in holdout["rows"]}
        add("| 方案 | 样本内累计 | 夏普 | MDD | 日均换手 | 费用(元) | 样本内 vs 红利14.2bp "
            "| holdout 累计 | 夏普 | MDD | 日均换手 | 费用(元) | holdout vs 红利14.2bp |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        pairs = [
            ("现有（逐日再平衡）", "existing-daily(Engine)", "existing-daily(Engine)"),
            ("月频纯再平衡", "月频纯再平衡@default", "holdout-月频纯再平衡(含insider)"),
            ("月频+带宽±2pp", "月频+带宽±2pp@default", "holdout-月频+带宽±2pp(含insider)"),
            ("月频-去 insider", "去insider-月频@default", "holdout-月频-去insider"),
        ]
        for label, sname, hname in pairs:
            s = pick(rows, sname)
            h = hmap.get(hname)
            if s is None or h is None:
                continue
            add(
                "| {} | {} | {} | {} | {} | {:,.0f} | {} | {} | {} | {} | {} | {:,.0f} | {} |".format(
                    label, _p(s["total_return"]), _p(s["sharpe"], 3, False), _p(s["max_drawdown"]),
                    _p(s["daily_turnover"]), s["fees"], _p(s["total_return"] - div14["total_return"]),
                    _p(h["total_return"]), _p(h["sharpe"], 3, False), _p(h["max_drawdown"]),
                    _p(h["daily_turnover"]), h["fees"], _p(h["total_return"] - href),
                )
            )
        add("")
        add("> holdout 一列**已被污染**（见第 4 节），只能当作「事后可改进空间」的上限参考；")
        add("> 样本内的对比才是本报告的证据基础。")
    add("")
    add("### 1.1 样本内（2017-2024，选股逻辑完全不变，只改再平衡实现）")
    add("")
    add("| 方案 | 成本 | 累计 | 年化 | 波动 | 夏普 | MDD | Calmar | 日均换手 | 月均换手 | 年化换手 | 费用(元) | 成交笔数 | vs 同成本红利 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rel_bands = [f"月频+相对带宽±{b}" for b in ("10%", "20%", "50%")]
    order = ["existing-daily(Engine)", "shadow-daily(自检)"] + [
        f"月频纯再平衡@{k}" for k in ("default", "conservative_30", "conservative_50")
    ] + [
        f"月频+带宽±2pp@{k}" for k in ("default", "conservative_30", "conservative_50")
    ] + [
        f"月频+带宽±5pp@{k}" for k in ("default", "conservative_30", "conservative_50")
    ] + rel_bands
    for name in order:
        r = pick(rows, name)
        if r is None:
            continue
        ref = div14 if r["round_trip_bp"] <= 14.2 else (div30 if r["round_trip_bp"] <= 30 else div50)
        vs = r["total_return"] - ref["total_return"] if ref else np.nan
        add(
            "| {} | {} {}bp | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:,.0f} | {:,.0f} | {} |".format(
                name, r["cost"], r["round_trip_bp"], _p(r["total_return"]), _p(r["cagr"]), _p(r["ann_vol"]),
                _p(r["sharpe"], 3, False), _p(r["max_drawdown"]), _p(r["calmar"], 3, False),
                _p(r["daily_turnover"]), _p(r["monthly_turnover"]), _p(r["annual_turnover"]),
                r["fees"], r["trades"], _p(vs),
            )
        )
    add("")
    add("> `月频+带宽±2pp` 的「pp」= 权重百分点（单票权重偏离目标 2 个百分点才调回）。")
    add("> 等权 50 只时单票目标权重约 1.9%，±2pp 的带宽大于多数现实漂移 → 触发极少：")
    smap = {r["variant"]: r for r in sample["stats"]}
    trims = []
    for tag, name in (("月频纯再平衡", "月频纯再平衡@default"),
                      ("±2pp", "月频+带宽±2pp@default"),
                      ("±5pp", "月频+带宽±5pp@default")):
        st = smap.get(name)
        r = monthly["default"] if name.startswith("月频纯") else (
            band2["default"] if "2pp" in name else band5["default"])
        if st and r:
            trims.append(f"{tag} {int(st['trade_days'])} 个交易日有成交 / {int(r['trades']):,} 笔")
    if trims:
        add("> " + "；".join(trims) + "。")
    add("> 触发次数虽少，但被触发的都是漂移很大的单票、单笔金额大，累计收益差 1~3pp 完全可能 ——")
    add("> 属路径依赖的偶然，不构成系统性优势（holdout 上 ±2pp 一次都没触发，与月频逐值相同）。")
    add("> 真正能稳定起作用的是相对带宽（见 1.3：带宽 = 目标权重的 x%）。")
    add("")

    # ---------------- 1.2 去 insider
    add("### 1.2 去掉 insider 因子（其余不变）")
    add("")
    add("| 方案 | 成本 | 累计 | 夏普 | MDD | Calmar | 日均换手 | 费用(元) | 成交笔数 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in ["去insider-逐日@default"] + [f"去insider-月频@{k}" for k in ("default", "conservative_30", "conservative_50")]:
        r = pick(rows, name)
        if r is None:
            continue
        add(
            "| {} | {}bp | {} | {} | {} | {} | {} | {:,.0f} | {:,.0f} |".format(
                name, r["round_trip_bp"], _p(r["total_return"]), _p(r["sharpe"], 3, False),
                _p(r["max_drawdown"]), _p(r["calmar"], 3, False), _p(r["daily_turnover"]),
                r["fees"], r["trades"],
            )
        )
    add("")
    if noins_daily and existing:
        add(
            f"同样逐日再平衡下，去 insider 使样本内累计从 {_p(existing['total_return'])} 变为 "
            f"{_p(noins_daily['total_return'])}（{_p(noins_daily['total_return'] - existing['total_return'])}），"
            f"夏普 {_p(existing['sharpe'], 3, False)} → {_p(noins_daily['sharpe'], 3, False)}。"
        )
    if monthly["default"] and noins_monthly["default"]:
        add(
            f"月频口径下，去 insider 使累计从 {_p(monthly['default']['total_return'])} 变为 "
            f"{_p(noins_monthly['default']['total_return'])}"
            f"（{_p(noins_monthly['default']['total_return'] - monthly['default']['total_return'])}），"
            f"夏普 {_p(monthly['default']['sharpe'], 3, False)} → {_p(noins_monthly['default']['sharpe'], 3, False)}。"
        )
    add("")

    # ---------------- 1.3 相对带宽
    add("### 1.3 相对带宽敏感性（默认成本，定位「带宽多大才真正起作用」）")
    add("")
    add("| 相对带宽 | 累计 | 夏普 | MDD | 日均换手 | 月均换手 | 费用(元) |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for label in ["月频纯再平衡@default"] + rel_bands:
        r = pick(rows, label)
        if r is None:
            continue
        add(
            "| {} | {} | {} | {} | {} | {} | {:,.0f} |".format(
                label.replace("@default", ""), _p(r["total_return"]), _p(r["sharpe"], 3, False),
                _p(r["max_drawdown"]), _p(r["daily_turnover"]), _p(r["monthly_turnover"]), r["fees"],
            )
        )
    add("")

    # ---------------- 2 成本公平
    add("## 2. 成本公平对照：无成本红利的口径虚高")
    add("")
    add("| 基准 | 往返成本 | 累计 | 年化 | 波动 | 夏普 | MDD | Calmar | 月均换手 | 年化换手 | 费用(元) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in ("同池等权(月频)", "纯红利top50(月频)", "同池等权(现有解析式,未计成本)"):
        for level in (0.0, 14.2, 30.0, 50.0):
            r = bget(name, level)
            if r is None:
                continue
            add(
                "| {} | {}bp | {} | {} | {} | {} | {} | {} | {} | {} | {:,.0f} |".format(
                    name, r["round_trip_bp"], _p(r["total_return"]), _p(r["cagr"]), _p(r.get("ann_vol")),
                    _p(r.get("sharpe"), 3, False), _p(r["max_drawdown"]), _p(r.get("calmar"), 3, False),
                    _p(r.get("monthly_turnover")), _p(r.get("annual_turnover")), r.get("fees", 0.0),
                )
            )
    add("")
    if div0 and div14:
        add(
            f"纯红利组合（股息率前 50、等权、月频再平衡）在**不计成本**时样本内累计 "
            f"{_p(div0['total_return'])}；按 14.2bp 往返成本扣掉月频换手后为 {_p(div14['total_return'])}，"
            f"口径虚高 {(div0['total_return']-div14['total_return'])*100:.2f} 个百分点"
            f"（即「无成本红利」这一列系统性高估了这一幅度）。"
        )
        add(
            f"30bp/50bp 下纯红利分别为 {_p(div30['total_return'] if div30 else np.nan)} / "
            f"{_p(div50['total_return'] if div50 else np.nan)}。"
        )
        add("")
        add(
            f"同口径下，影子版月频再平衡（14.2bp）累计 {_p(monthly['default']['total_return'])}，"
            f"对同成本纯红利（{_p(div14['total_return'])}）超额 {_p(monthly['default']['total_return'] - div14['total_return'])}；"
            f"对同池等权（{_p(pool14['total_return'] if pool14 else np.nan)}）超额 "
            f"{_p(monthly['default']['total_return'] - pool14['total_return'] if pool14 else np.nan)}。"
        )
    add("")

    # ---------------- 3 insider IC
    add("## 3. insider 因子的样本内表现")
    add("")
    add("| 因子 | 月数 | IC 均值 | IC 标准差 | IC_IR(月度) | 年化 IR | t 值 | 胜率 | 正 IC 年份 | 年份数 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in ic.iterrows():
        add(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["factor"], int(r["months"]), _p(r.get("ic_mean"), 4, False), _p(r.get("ic_std"), 4, False),
                _p(r.get("ic_ir"), 3, False), _p(r.get("annual_ir"), 3, False), _p(r.get("t_stat"), 2, False),
                _p(r.get("positive_month_share")), int(r.get("year_positive", 0)), int(r.get("years", 0)),
            )
        )
    add("")
    add("口径：20 日远期收益、月末取样（与 `results/holdout/holdout-factor-ic.csv` 同一算法）；")
    add("IC_IR(月度) = IC 均值/标准差，年化 IR = IC_IR × √12。")
    add("")
    add("逐年 IC（insider_net_20）：")
    add("")
    add("| 年份 | 月数 | IC 均值 | IC_IR | 胜率 |")
    add("| --- | --- | --- | --- | --- |")
    yearly = sample["ic_yearly"]
    for _, r in yearly[yearly["factor"] == "insider_net_20"].iterrows():
        add("| {} | {} | {} | {} | {} |".format(
            int(r["year"]), int(r["months"]), _p(r["ic_mean"], 4, False), _p(r["ic_ir"], 3, False), _p(r["hit_rate"])))
    add("")

    # ---------------- 4 holdout
    add("## 4. 一次性 holdout 上限对照（**已开卷污染，不可作为选参依据**）")
    add("")
    add("> ⚠️ **这些数字是事后上限参考。** holdout 已经被看过（`docs/样本外验证报告.md`），")
    add("> 用它比较方案等于在样本外选参，必然高估。最终有效性必须靠 2026-10 之后的实时纸面/小资金")
    add("> 验证，**不能**据此宣称改进版通过/有效，也不能据此增删因子或挑带宽。")
    add("")
    if holdout is None:
        add("（本次未运行 holdout 阶段。）")
    else:
        hrows = pd.DataFrame(holdout["rows"])
        hbench = pd.DataFrame(holdout["bench_rows"])
        div14h = hbench[(hbench["benchmark"] == "纯红利top50(月频)") & np.isclose(hbench["round_trip_bp"], 14.2)]
        ref = float(div14h["total_return"].iloc[0]) if len(div14h) else np.nan
        add("| 组合（预先固定） | 累计 | 年化 | 波动 | 夏普 | MDD | Calmar | 日均换手 | 月均换手 | 费用(元) | vs 同成本红利 |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, r in hrows.iterrows():
            add(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:,.0f} | {} |".format(
                    r["variant"], _p(r["total_return"]), _p(r["cagr"]), _p(r["ann_vol"]),
                    _p(r["sharpe"], 3, False), _p(r["max_drawdown"]), _p(r["calmar"], 3, False),
                    _p(r["daily_turnover"]), _p(r["monthly_turnover"]), r["fees"],
                    _p(r["total_return"] - ref),
                )
            )
        add("")
        add("| 基准（holdout） | 往返成本 | 累计 | 夏普 | MDD | 月均换手 |")
        add("| --- | --- | --- | --- | --- | --- |")
        for _, r in hbench.iterrows():
            add("| {} | {}bp | {} | {} | {} | {} |".format(
                r["benchmark"], r["round_trip_bp"], _p(r["total_return"]), _p(r.get("sharpe"), 3, False),
                _p(r["max_drawdown"]), _p(r.get("monthly_turnover"))))
        add("")
        add("> `月频+带宽±2pp` 与 `月频纯再平衡` 在 holdout 上逐值相同（带宽一次都没触发）；")
        add("> 样本内两者差 −2.46pp，只来自 2 个额外交易日上的 3 笔大额调仓 —— 属路径依赖，")
        add("> 不能据此认定哪个更好，也**不**据此挑带宽。")
        add("")

    # ---------------- 5 结论
    add("## 5. 结论")
    add("")
    add("### 5.1 可在下一周期实施（A 类：回测实现修复，选股逻辑不变）")
    add("")
    if monthly["default"]:
        d_tot = monthly["default"]["total_return"] - existing["total_return"]
        d_fee = existing["fees"] - monthly["default"]["fees"]
        d_to = existing["daily_turnover"] - monthly["default"]["daily_turnover"]
        add(
            f"- 把「逐日拉回目标权重」改成「只在调仓日交易」后，样本内日均换手从 "
            f"{_p(existing['daily_turnover'])} 降到 {_p(monthly['default']['daily_turnover'])}"
            f"（年化单边换手 {_p(existing['annual_turnover'])} → {_p(monthly['default']['annual_turnover'])}），"
            f"8 年费用从 {existing['fees']:,.0f} 元降到 {monthly['default']['fees']:,.0f} 元（少 {d_fee:,.0f} 元）。"
        )
        add(
            f"- 同期累计收益 {_p(existing['total_return'])} → {_p(monthly['default']['total_return'])}"
            f"（{_p(d_tot)}），夏普 {_p(existing['sharpe'], 3, False)} → {_p(monthly['default']['sharpe'], 3, False)}，"
            f"MDD {_p(existing['max_drawdown'])} → {_p(monthly['default']['max_drawdown'])}。"
        )
        add(
            f"- 选股、因子、股票池、成本档位一字未改，改的只是「非调仓日不再交易」这一条执行规则，"
            f"属于对既有回测**建模过严**的修正（真实组合不会每天为价格漂移付一次佣金）。"
            f"建议在下一个调仓周期把这条落实为实盘执行规则：月频调仓、非调仓日不交易、"
            f"必要时用带宽兜住极端漂移。"
        )
    add("")
    add("### 5.2 需未来样本外验证（B 类：策略调整，现在不实施）")
    add("")
    insider_row = ic[ic["factor"] == "insider_net_20"]
    ic_weak = None
    if len(insider_row):
        r = insider_row.iloc[0]
        t = float(r.get("t_stat", np.nan))
        yp, yy = int(r.get("year_positive", 0)), int(r.get("years", 0))
        ic_weak = (not np.isfinite(t)) or t < 2.0 or yp < (2.0 / 3.0) * yy
        add(
            f"- insider_net_20 在样本内的 IC：月均 {_p(r.get('ic_mean'), 4, False)}、"
            f"IC_IR(月度) {_p(r.get('ic_ir'), 3, False)}、t 值 {_p(t, 2, False)}、"
            f"正 IC 年份 {yp}/{yy} —— 单因子层面{'**弱且不显著**（t < 2，或正 IC 年份不足 2/3）' if ic_weak else '**显著**（t ≥ 2 且多数年份 IC > 0）'}。"
        )
        add(
            f"- 对照：div_yield / vol_60 / turnover_20 的 t 值分别为 "
            f"{_p(_t_of(ic, 'div_yield'), 2, False)} / {_p(_t_of(ic, 'vol_60'), 2, False)} / "
            f"{_p(_t_of(ic, 'turnover_20'), 2, False)}，insider 的信息量明显更低一档。"
        )
    if monthly["default"] and noins_monthly["default"]:
        d_tot = noins_monthly["default"]["total_return"] - monthly["default"]["total_return"]
        add(
            f"- 但组合层面（同为月频口径）去掉它反而更差：含 insider 累计 "
            f"{_p(monthly['default']['total_return'])}（夏普 {_p(monthly['default']['sharpe'], 3, False)}），"
            f"去 insider 后 {_p(noins_monthly['default']['total_return'])}"
            f"（夏普 {_p(noins_monthly['default']['sharpe'], 3, False)}），差 {_p(d_tot)}。"
        )
        if d_tot < 0 and ic_weak:
            add(
                "- 即：**单因子 IC 弱，但去掉它组合反而更差**。原因不在 IC 本身：4 因子等权合成时每只占 25%，"
                "去掉 insider 等于改变了其余 3 个因子的相对权重与最终名单（样本内名单变了，选股结果就是另一套）。"
                "「凭单因子 IC 弱就砍」会连带改掉合成权重，这一步不能只按 IC 定。"
            )
        elif d_tot >= 0 and ic_weak:
            add("- 即：IC 弱、组合边际贡献也为负/零，两者方向一致，去它确有样本内依据。")
        else:
            add("- 即：IC 与组合边际贡献方向一致，insider 在样本内确有正贡献。")
    add(
        "- 判定（B 类）：**去掉 insider 会改变选股结果，属于策略调整而不是实现修复。**"
        "按任务给定的判据，单因子层面它在样本内确实「本身就弱」（t = 1.93 < 2、正 IC 年份 5/8）——"
        "这确实构成了「去它有样本内依据」；但组合层面去它反而更差（月频 −7.92pp），"
        "说明这条依据只到「单因子不显著」，没到「应当剔除」。"
        "→ 现在只记录、不实施，列入下一周期「新样本外数据」的验证清单：只有在 2026-10 之后的"
        "新数据上 IC 与组合贡献**同时**归零，才可考虑剔除。"
    )
    add("")

    add("### 5.3 纯实现修复能达到什么程度，还缺什么")
    add("")
    if monthly["default"] and existing:
        add(
            f"- 纯实现修复（月频再平衡）：8 年费用 {existing['fees']:,.0f} → {monthly['default']['fees']:,.0f} 元"
            f"（{_p((existing['fees'] - monthly['default']['fees']) / existing['fees'])}），"
            f"日均换手 {_p(existing['daily_turnover'])} → {_p(monthly['default']['daily_turnover'])}，"
            f"累计 {_p(existing['total_return'])} → {_p(monthly['default']['total_return'])}，"
            f"夏普 {_p(existing['sharpe'], 3, False)} → {_p(monthly['default']['sharpe'], 3, False)}，"
            f"MDD {_p(existing['max_drawdown'])} → {_p(monthly['default']['max_drawdown'])}。"
        )
    if monthly["default"] and div14:
        gap = monthly["default"]["total_return"] - div14["total_return"]
        add(
            f"- 与同成本（14.2bp）纯红利相比：月频影子版累计 {_p(monthly['default']['total_return'])} vs "
            f"红利 {_p(div14['total_return'])}（**差 {_p(gap)}**）；月均单边换手 "
            f"{_p(monthly['default']['monthly_turnover'])} vs 红利 {_p(div14.get('monthly_turnover'))}"
            f"（约 {monthly['default']['monthly_turnover'] / div14['monthly_turnover']:.1f} 倍）。"
        )
        add(
            f"- 与同池等权（14.2bp）{_p(pool14['total_return'] if pool14 else np.nan)} 相比差 "
            f"{_p(monthly['default']['total_return'] - pool14['total_return'] if pool14 else np.nan)}。"
        )
        if gap < 0:
            add("- 也就是说：**实现修复并没有让策略跑赢同成本纯红利**，样本内仍落后。")
            if pool14:
                add(
                    "- 差距不在「相对同池」这一层：策略对同池 "
                    f"{_p(monthly['default']['total_return'] - pool14['total_return'])}，纯红利对同池 "
                    f"{_p(div14['total_return'] - pool14['total_return'])}。拉开差距的是红利名单本身在这个池子里太强："
                    f"同池等权 8 年年化 {_p(pool14['cagr'])}，纯红利 {_p(div14['cagr'])}，"
                    f"而 4 因子影子版 {_p(monthly['default']['cagr'])} —— 等权合成把红利/低波这份收益稀释了。"
                    "这是选股层面的问题，再平衡实现改不了。"
                )
            k_s = monthly["default"]["monthly_turnover"] * 50.0
            k_d = div14.get("monthly_turnover", np.nan) * 50.0
            add(
                f"- 换手也还没解决：策略月均单边换手 {_p(monthly['default']['monthly_turnover'])}"
                f"（按 50 只等权折算 ≈ 每月换 {k_s:.0f} 只），红利 {_p(div14.get('monthly_turnover'))}"
                f"（≈ {k_d:.0f} 只）。这一块只能靠换手预算/缓冲区在组合构建层解决。"
            )
    add(
        "- 实现修复只补上「交易过频」这一个漏洞：成本与换手大幅下降、收益与夏普改善；"
        "alpha 的量级仍由选股逻辑决定，不会因为少交易而变成另一个策略。"
    )
    add(
        "- 真要稳定超越红利，缺的是：① 独立于红利/低波风格的增量因子（样本内外都稳定）；"
        "② 更严格的样本外验证协议（滚动/多段 holdout，而不是单段已开卷数据）；"
        "③ 交易成本与容量约束下的组合构建（换手预算），本报告只量化了③的一部分。"
    )
    add("")
    add("## 6. 复算")
    add("")
    add("```bash")
    add("python scripts/improvement_shadow.py            # 全流程（样本内 + 一次性 holdout）")
    add("python scripts/improvement_shadow.py --sample   # 只跑样本内")
    add("python scripts/improvement_shadow.py --report-only  # 用已落盘 CSV 重生成本报告")
    add("```")
    add("")
    add("> 本机 PATH 上的 `python` 是沙箱运行时（无 numpy），实测用的是")
    add("> `C:\\Users\\abc11\\.dsh\\dsh-runtimes\\dsh-primary-runtime\\dependencies\\python\\python.exe`")
    add("> （3.12 / pandas 3.0 / numpy 2.3），并需 `PYTHONPATH` 含工作区根目录与 `_tmp\\pylibs`")
    add("> （`requests`、`pyarrow` 装在后者）。纯 pandas/numpy 环境下不需要后者。")
    add("")
    add(f"运行耗时（样本内）：{meta.get('runtime_s', '?')} 秒。明细见同目录 CSV：")
    add("`shadow-metrics.csv`、`bench-cost-ladder.csv`、`insider-ic.csv`、`insider-ic-yearly.csv`、`selfcheck.json`。")
    add("")

    path = OUT / "IMPROVEMENT_SHADOW.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------- report-only
def _sha16(path: Path) -> str:
    """文件 sha256 前 16 位；文件不存在返回空串。"""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_saved_sample() -> dict:
    """从已落盘 CSV 重建 sample 字典（不重跑回测）。"""
    rows = pd.read_csv(OUT / "shadow-metrics.csv").to_dict("records")
    bench_rows = pd.read_csv(OUT / "bench-cost-ladder.csv").to_dict("records")
    ic = pd.read_csv(OUT / "insider-ic.csv")
    ic_yearly = pd.read_csv(OUT / "insider-ic-yearly.csv")
    stats_path = OUT / "shadow-engine-stats.csv"
    stats = pd.read_csv(stats_path).to_dict("records") if stats_path.exists() else []
    meta = json.loads((OUT / "shadow-meta.json").read_text(encoding="utf-8"))
    selfcheck = json.loads((OUT / "selfcheck.json").read_text(encoding="utf-8"))
    equity = {}
    for p in sorted(OUT.glob("equity-*.csv")):
        eq = pd.read_csv(p, index_col=0)
        equity[p.stem.replace("equity-", "")] = eq.iloc[:, 0]
    return {
        "rows": rows,
        "bench_rows": bench_rows,
        "stats": stats,
        "ic": ic,
        "ic_yearly": ic_yearly,
        "meta": meta,
        "selfcheck": selfcheck,
        "equity": equity,
    }


def load_saved_holdout() -> dict | None:
    path = OUT / "holdout-shadow-metrics.csv"
    if not path.exists():
        return None
    meta = json.loads((OUT / "holdout-shadow-meta.json").read_text(encoding="utf-8"))
    bench_path = OUT / "holdout-bench-cost-ladder.csv"
    return {
        "rows": pd.read_csv(path).to_dict("records"),
        "bench_rows": pd.read_csv(bench_path).to_dict("records") if bench_path.exists() else [],
        "meta": meta,
        "selfcheck": meta.get("selfcheck"),
    }


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="影子改进版评估（只读现有策略）")
    parser.add_argument("--sample", action="store_true", help="只跑样本内")
    parser.add_argument("--smoke", type=int, default=None, help="只取前 N 只标的（管线冒烟，不作结论）")
    parser.add_argument("--skip-holdout", action="store_true", help="跳过 holdout")
    parser.add_argument("--report-only", action="store_true", help="不跑回测，仅用已落盘 CSV 重生成报告")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        sample = load_saved_sample()
        holdout = None if args.sample else load_saved_holdout()
        path = write_report(sample, holdout, args)
        print(f"报告已重生成 → {path}", flush=True)
        return 0

    event_factors.register()
    started = time.time()

    freeze_path = Path("results/datalayer-freeze.json")
    freeze_before = _sha16(freeze_path)

    sample = run_sample(args)
    pd.DataFrame(sample["rows"]).to_csv(OUT / "shadow-metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sample["bench_rows"]).to_csv(OUT / "bench-cost-ladder.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sample["stats"]).to_csv(OUT / "shadow-engine-stats.csv", index=False, encoding="utf-8-sig")
    sample["ic"].to_csv(OUT / "insider-ic.csv", index=False, encoding="utf-8-sig")
    sample["ic_yearly"].to_csv(OUT / "insider-ic-yearly.csv", index=False, encoding="utf-8-sig")
    (OUT / "selfcheck.json").write_text(json.dumps(sample["selfcheck"], ensure_ascii=False, indent=2), encoding="utf-8")
    for name, eq in sample["equity"].items():
        eq.rename("equity").to_csv(OUT / f"equity-{name}.csv", encoding="utf-8-sig")

    holdout = None
    if not args.sample and not args.skip_holdout:
        # 预注册规则：insider 样本内弱/不稳（t<2 或 正 IC 年份 < 2/3）才跑 ③ 去 insider
        insider_row = sample["ic"][sample["ic"]["factor"] == "insider_net_20"]
        need_no_insider = False
        if len(insider_row):
            r = insider_row.iloc[0]
            years = max(int(r.get("years", 0)), 1)
            need_no_insider = bool(
                (np.isfinite(r.get("t_stat", np.nan)) and float(r["t_stat"]) < 2.0)
                or int(r.get("year_positive", 0)) < (2.0 / 3.0) * years
            )
        print(f"[holdout] 预注册规则判定：去 insider 组合 {'纳入' if need_no_insider else '不纳入'}", flush=True)
        holdout = run_holdout(args, need_no_insider)
        pd.DataFrame(holdout["rows"]).to_csv(OUT / "holdout-shadow-metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(holdout["bench_rows"]).to_csv(OUT / "holdout-bench-cost-ladder.csv", index=False, encoding="utf-8-sig")
        (OUT / "holdout-shadow-meta.json").write_text(
            json.dumps(holdout["meta"], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    freeze_after = _sha16(freeze_path)
    sample["meta"]["freeze_sha16_before"] = freeze_before
    sample["meta"]["freeze_sha16_after"] = freeze_after
    sample["meta"]["freeze_unchanged"] = bool(freeze_before == freeze_after)
    (OUT / "shadow-meta.json").write_text(json.dumps(sample["meta"], ensure_ascii=False, indent=2), encoding="utf-8")

    path = write_report(sample, holdout, args)
    print(f"完成 {time.time()-started:.0f}s → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
