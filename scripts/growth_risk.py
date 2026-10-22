"""阶段 3 风控增强：降换手（缓冲带）+ 流动性挂钩滑点 + 波动/回撤控制 + 清仓对照。

设计纪律
--------
- **不修改仓库任何既有文件**：引擎的「非调仓日不交易」与「按流动性分档滑点」都在
  本脚本内用运行时子类/替身实现（``quant.backtest.engine.Broker`` 局部替换），跑完即失效。
- 样本严格 2017-01-01 ~ 2024-12-31，2025+ 一行不读（``growth``/``growth_alt`` 内部硬截断）。
- 所有组合同口径：50 只、等权、gross 95%、单票 4%、行业 30%、月频、次日开盘成交、退市 haircut30。
- 判据事前写死：相对基线 **MDD 改善 ≥3pp 且夏普不降** 才保留，否则去掉。

顺手修的一个语义缺口
--------------------
``Strategy.hold_to_rebalance`` 的注释是「只在调仓日更新权重，其余日子沿用上一期（避免每天换手）」，
但 ``Engine.run`` 只是 ``weights.ffill()``，然后**每个交易日**都用 ``equity × w`` 重算目标股数，
整手向下取整 + 浮点误差 → 每天出现一笔「卖 100 股次日买回」的噪声交易。实测 QN 原始回测：
32,750 笔成交、中位成交额 1,029 元、日均换手 1.49%（中位）、日均费用 103 元。
本脚本用 ``RiskBroker.ALLOW`` 把非调仓日的 ``plan_orders`` 直接返回空，恢复「月内不动」语义；
``legacy`` 规格保留原行为，用来量化这笔噪声的成本。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import quant.backtest.engine as engine_mod
import quant.regime as regime
from quant.backtest.broker import Broker
from quant.backtest.engine import Engine
from quant.config import BacktestConfig, CostModel
from quant.factors import growth, growth_alt
from quant.factors.panel import build_panel

from scripts.factor_market import FIELDS, load_delisting_events
from scripts.growth_backtest import (
    GROSS,
    MAX_WEIGHT,
    MIN_NAMES,
    MONTHLY,
    SECTOR_CAP,
    SPECS,
    TOP_K,
    GrowthTopK,
    composite_score,
    month_ends,
    sector_panel,
    turnover_stats,
)
from scripts.redlow_study import (
    END,
    LAKE,
    START,
    board_exclude,
    build_block,
    load_bars,
    pool_benchmark,
    summarize,
)

# --------------------------------------------------------------------------- #
# 事前参数（写死，不搜索）
# --------------------------------------------------------------------------- #

BUFFER_RANK = 60          # 缓冲带：仍在候选前 60 名就保留（50 的 1.2 倍）
TOP_K = TOP_K
GROSS = GROSS

#: 流动性分档滑点（单边 bp）：档 1 最优 → 档 5 最差。用户给的范围是「最好 2bp、最差 4–6bp」。
SLIP_TIERS: tuple[float, ...] = (2.0, 2.5, 3.0, 4.0, 5.0)

VOL_TARGET = 0.25         # 波动率目标 25%
VOL_WINDOW = 20           # 20 日已实现波动
VOL_MAX_SCALE = 1.0       # 只降不升（天然零杠杆）

DD_PEAK_WINDOW = 250      # 回撤预算：250 日净值高点
DD_FLOOR = 0.30           # 仓位下限 0.3（保留反弹参与）
DD_TRIGGER = 0.75         # NAV/peak − 0.75 → 开始降仓

CRISIS_ENTER = 0.75   # 与既有危机实现（docs/风险清仓与前沿因子.md）一致：0.75 触发、双确认、滞回
CRISIS_EXIT = 0.30
CRISIS_CONFIRM = 3
CRISIS_EXPOSURE = 0.50    # 梯度版危机期的仓位
CRISIS_RAMP = 10          # 危机解除后 10 个交易日线性回补到 1.0
CRISIS_SOFT = 0.45        # 危险分进入 [0.45, 0.60) 的「预备档」仓位
CRISIS_SOFT_EXPOSURE = 0.75

BASE_BPS = 2.0
OUT = Path("results/growth/stage3/risk")


# --------------------------------------------------------------------------- #
# 流动性分档
# --------------------------------------------------------------------------- #


def liquidity_panels(panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ADV20（成交额 20 日均值，越大越流动）与 ILLIQ20（Amihud，越大越不流动）。"""
    amount = panel.amount.replace(0.0, np.nan)
    adv20 = amount.rolling(20, min_periods=10).mean()
    illiq20 = (panel.returns.abs() / amount * 1e8).rolling(20, min_periods=10).mean()
    return adv20, illiq20


def tier_map(
    marks: pd.DatetimeIndex,
    eligible: pd.DataFrame,
    adv20: pd.DataFrame,
    illiq20: pd.DataFrame,
) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    """每个调仓日按「流动性综合排名」切 5 档 → symbol → 单边滑点 bp。

    综合排名 = ADV 分位升序排名 与 ILLIQ 分位降序排名的平均（两者高度负相关，
    平均后对单一指标的极值不敏感）。候选集内切 5 等分，档 1 最优。
    """
    maps: dict[pd.Timestamp, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for day in marks:
        ok = eligible.reindex(index=[day]).iloc[0].reindex(adv20.columns).fillna(False).astype(bool)
        adv = adv20.loc[day].where(ok)
        illiq = illiq20.loc[day].where(ok)
        if adv.notna().sum() < 50:
            maps[day] = {}
            continue
        rank_adv = adv.rank(pct=True)          # 越大越流动
        rank_illiq = 1.0 - illiq.rank(pct=True)  # 越不流动 → 值越小
        combo = 0.5 * rank_adv.fillna(0.0) + 0.5 * rank_illiq.fillna(0.0)
        valid = adv.notna() & illiq.notna()
        combo = combo.where(valid)
        # combo 越大越流动 → 档 1 必须是最流动，所以把升序分位反过来
        bucket = pd.qcut(combo[valid].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
        tier = pd.Series(index=combo.index, dtype=float)
        tier.loc[bucket.index] = bucket.to_numpy(dtype=float)
        bps = tier.map({index + 1: value for index, value in enumerate(SLIP_TIERS)})
        maps[day] = {str(symbol): float(value) for symbol, value in bps.dropna().items()}
        for number, colour in enumerate(SLIP_TIERS, start=1):
            member = tier[tier == number]
            if member.empty:
                continue
            rows.append(
                {
                    "date": day,
                    "tier": number,
                    "bps": colour,
                    "n": int(member.shape[0]),
                    "adv_median_yi": float(adv.reindex(member.index).median() / 1e8),
                    "illiq_median": float(illiq.reindex(member.index).median()),
                }
            )
    frame = pd.DataFrame(rows)
    return maps, frame


def tier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.groupby("tier").agg(
        bps=("bps", "first"),
        n_median=("n", "median"),
        adv_median_yi=("adv_median_yi", "median"),
        illiq_median=("illiq_median", "median"),
    )
    out["share"] = out["n_median"] / out["n_median"].sum()
    return out.reset_index()


# --------------------------------------------------------------------------- #
# 带缓冲带的选股
# --------------------------------------------------------------------------- #


class BufferTopK(GrowthTopK):
    """排名前 ``top_k`` 建仓；已有持仓只要仍在 ``buffer_rank`` 内就保留。"""

    name = "growth-buffer"

    def __init__(self, *args: Any, buffer_rank: int = BUFFER_RANK, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.buffer_rank = int(max(buffer_rank, self.top_k))

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        dates = self.score.index
        symbols = list(self.score.columns)
        mask = self.rebalance_mask(dates, self.rebalance).reindex(dates).fillna(False)
        target = pd.DataFrame(0.0, index=dates, columns=symbols)
        weight_value = min(self.gross / max(self.top_k, 1), self.max_weight)
        picks: dict[str, list[str]] = {}
        held: list[str] = []
        for day in dates[mask.to_numpy()]:
            row = self.score.loc[day].where(self.eligible.reindex(index=[day]).iloc[0]).dropna()
            row = row.sort_values(ascending=False)
            order = list(row.index)
            rank = {symbol: position for position, symbol in enumerate(order)}
            sector_row = self.sector.loc[day] if self.sector is not None and day in self.sector.index else None
            counts: dict[str, int] = {}

            def capacity_left(key: str) -> bool:
                return (counts.get(key, 0) + 1) * weight_value <= self.sector_cap + 1e-12

            chosen: list[str] = []
            for symbol in held:                      # 1) 老仓位：仍在缓冲带内就留
                if rank.get(symbol, 10**9) >= self.buffer_rank:
                    continue
                key = str(sector_row.get(symbol, "UNKNOWN")) if sector_row is not None else "UNKNOWN"
                if not capacity_left(key):
                    continue
                chosen.append(symbol)
                counts[key] = counts.get(key, 0) + 1
                if len(chosen) >= self.top_k:
                    break
            if len(chosen) < self.top_k:             # 2) 新补：按排名顺序补满
                for symbol in order:
                    if len(chosen) >= self.top_k:
                        break
                    if symbol in chosen:
                        continue
                    key = str(sector_row.get(symbol, "UNKNOWN")) if sector_row is not None else "UNKNOWN"
                    if not capacity_left(key):
                        continue
                    chosen.append(symbol)
                    counts[key] = counts.get(key, 0) + 1
            if len(chosen) < self.min_holdings:
                self.last_diagnostics["notes"].append(f"{day.date()} 合格标的仅 {len(chosen)} 只，本期空仓")
                picks[str(day.date())] = []
                held = []
                continue
            held = chosen
            picks[str(day.date())] = chosen
            for symbol in chosen:
                target.at[day, symbol] = weight_value
        target = self.hold_to_rebalance(target, self.rebalance)
        self.last_diagnostics["picks"] = picks
        return self.clip_weights(target)


# --------------------------------------------------------------------------- #
# 运行时替身：非调仓日不交易 + 按标的分档滑点
# --------------------------------------------------------------------------- #


class RiskBroker(Broker):
    """``ALLOW`` 为空 → 原行为；否则只在允许的日子下单。``SLIP`` 为空 → 固定 ``BASE_BPS``。"""

    ALLOW: set[pd.Timestamp] | None = None
    SLIP: dict[str, float] = {}
    BASE_BPS: float = BASE_BPS
    MIN_TRADE: float = 0.0

    def plan_orders(self, date, targets, portfolio, bars, **kwargs):  # type: ignore[override]
        if RiskBroker.ALLOW is not None and date not in RiskBroker.ALLOW:
            return []
        if RiskBroker.MIN_TRADE:
            kwargs["min_trade_value"] = RiskBroker.MIN_TRADE
        return super().plan_orders(date, targets, portfolio, bars, **kwargs)

    def _tradable(self, symbol, bar, side):  # type: ignore[override]
        self._active_symbol = symbol
        return super()._tradable(symbol, bar, side)

    def _slipped(self, price: float, side: str) -> float:  # type: ignore[override]
        bps = RiskBroker.SLIP.get(getattr(self, "_active_symbol", ""), RiskBroker.BASE_BPS)
        adj = bps / 10_000.0
        return price * (1 + adj) if side == "buy" else price * (1 - adj)


def install_broker(min_trade: float = 0.0) -> None:
    RiskBroker.MIN_TRADE = float(min_trade)
    engine_mod.Broker = RiskBroker  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 影子净值与月频缩放
# --------------------------------------------------------------------------- #


def shadow_nav(weights: pd.DataFrame, panel) -> pd.Series:
    """用目标权重 × 面板收益推一条净值（权重已 ffilled，shift(1) 避免当日信息）。"""
    rets = panel.returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    port = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    return (1.0 + port).cumprod()


def vol_scale(nav: pd.Series, *, target: float = VOL_TARGET, window: int = VOL_WINDOW) -> pd.Series:
    rets = nav.pct_change()
    realized = rets.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(244)
    scale = (target / realized).replace([np.inf, -np.inf], np.nan).clip(upper=VOL_MAX_SCALE)
    return scale.fillna(1.0)


def drawdown_scale(nav: pd.Series, *, window: int = DD_PEAK_WINDOW, floor: float = DD_FLOOR) -> pd.Series:
    peak = nav.rolling(window, min_periods=60).max()
    ratio = (nav / peak).fillna(1.0)
    scale = ((ratio - DD_TRIGGER) / (1.0 - DD_TRIGGER)).clip(lower=floor, upper=1.0)
    return scale.fillna(1.0)


def apply_monthly_scale(weights: pd.DataFrame, scale: pd.Series, marks: pd.DatetimeIndex) -> pd.DataFrame:
    """只在调仓日缩放（月内不动），避免把月频策略变成日频。"""
    out = weights.copy()
    for day in marks:
        if day not in out.index:
            continue
        factor = float(scale.reindex([day]).iloc[0]) if day in scale.index else 1.0
        if not np.isfinite(factor):
            factor = 1.0
        out.loc[day] = out.loc[day] * factor
    return out


# --------------------------------------------------------------------------- #
# 危机状态与梯度仓位
# --------------------------------------------------------------------------- #


def crisis_inputs(panel) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    frame = regime.market_index(panel)
    flags = regime.score_components(frame)
    score = regime.crisis_score(flags)
    gate = flags["vol"] | flags["tail"]
    recovery = frame["index"] > frame["ma200"]
    state = regime.crisis_state(score, enter=CRISIS_ENTER, exit=CRISIS_EXIT, confirm=CRISIS_CONFIRM, gate=gate, recovery=recovery)
    return frame, score, state, recovery


def gradient_exposure(score: pd.Series, state: pd.Series) -> pd.Series:
    """危险分 → 仓位：正常 1.0；预备档 [0.45,0.60) 降到 0.75；危机期 0.5；解除后 10 日线性回补。"""
    values = pd.Series(1.0, index=score.index).to_numpy(dtype=float)
    soft = ((score >= CRISIS_SOFT) & (score < CRISIS_ENTER)).fillna(False).to_numpy(dtype=bool)
    flags = state.reindex(score.index).fillna(False).to_numpy(dtype=bool)
    values = np.where(soft, CRISIS_SOFT_EXPOSURE, values)
    values = np.where(flags, CRISIS_EXPOSURE, values)

    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        while index < len(flags) and flags[index]:
            index += 1
        for step in range(CRISIS_RAMP):          # 危机解除后线性回补 0.5 → 1.0
            position = index + step
            if position >= len(flags):
                break
            blend = CRISIS_EXPOSURE + (1.0 - CRISIS_EXPOSURE) * (step + 1) / CRISIS_RAMP
            values[position] = min(values[position], blend)
    return pd.Series(values, index=score.index)


def exposure_to_weights(weights: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    """把仓位系数（逐日）乘到日频权重矩阵上；只在系数变化处产生新行。"""
    frame = exposure.reindex(weights.index).ffill().fillna(1.0)
    return weights.mul(frame, axis=0).fillna(0.0)


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #


def evaluate(
    result,
    bench_equity: pd.Series,
    *,
    cash: float,
    slip_label: str,
) -> dict[str, Any]:
    base = summarize(result, bench_equity, {}, [])
    equity = result.equity
    daily = result.daily
    trades = result.trades
    years = max(len(daily) / 244.0, 1e-9)
    gross = float(trades["gross"].sum()) if len(trades) else 0.0
    slippage = float(trades["slippage_cost"].sum()) if len(trades) else 0.0
    avg_equity = float(daily["equity"].mean())
    out = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": float(base.get("cagr", np.nan)),
        "sharpe": float(base.get("sharpe", np.nan)),
        "max_drawdown": float(base.get("max_drawdown", np.nan)),
        "calmar": float(base.get("calmar", np.nan)),
        "excess_total": float(base.get("excess_total", np.nan)),
        "fees": float(base.get("fees", 0.0)),
        "fees_pct_of_equity": float(base.get("fees", 0.0)) / avg_equity / years,
        "trades": int(len(trades)),
        "ann_turnover_one_way": float(gross / 2.0 / avg_equity / years),
        "weighted_slip_bp": float(slippage / gross * 1e4) if gross > 0 else BASE_BPS,
        "slip_label": slip_label,
        "avg_exposure": float(daily["gross_exposure"].mean()),
        "min_exposure": float(daily["gross_exposure"].min()),
        "cash_days": int((daily["gross_exposure"] < 1e-9).sum()),
        "elapsed": None,
    }
    return out


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def episode_rows(state: pd.Series, score: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
    items = regime.episodes(state, score, frame)
    return pd.DataFrame([item.__dict__ for item in items])


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


@dataclass
class Run:
    name: str
    label: str
    buffer_rank: int
    tiered: bool
    scale: str | None      # None / "vol" / "dd"
    entry: str | None      # None / "crisis" / "gradient"
    min_trade: float = 0.0 # >0 时忽略小于该金额的订单（检验噪声交易是否真能落地）


RUNS: tuple[Run, ...] = (
    Run("legacy", "QN 原样（含逐日噪声交易）", 0, False, None, None),
    Run("legacy5k", "QN 原样 + 忽略 <5000 元订单", 0, False, None, None, 5_000.0),
    Run("legacy5k_slip", "逐日重置 + 忽略<5000元单 + 分档滑点", 0, True, None, None, 5_000.0),
    Run("base", "QN + 非调仓日不交易（无缓冲）", 0, False, None, None),
    Run("buf60", "base + 缓冲带 60", BUFFER_RANK, False, None, None),
    Run("buf60_slip", "buf60 + 流动性分档滑点", BUFFER_RANK, True, None, None),
    Run("vt25", "buf60_slip + 波动率目标 25%/20d", BUFFER_RANK, True, "vol", None),
    Run("ddb", "buf60_slip + 回撤预算（下限0.3）", BUFFER_RANK, True, "dd", None),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--runs", default=",".join(run.name for run in RUNS))
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    bars, load_stats = load_bars()
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    panel = build_panel(bars, fields=FIELDS)
    block = build_block(panel).reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)
    print(f"[panel] {panel.close.shape} {panel.dates[0].date()} ~ {panel.dates[-1].date()}", flush=True)

    facts = growth.GrowthFactors(panel)
    eligible = growth.eligible_mask(facts, block_mask=block)
    marks = month_ends(panel.dates, START, END)
    alt = growth_alt.alt_factors(panel)
    sector = sector_panel(marks, list(panel.symbols))
    score = composite_score(panel, facts, alt, SPECS["QN"]["blocks"], eligible, marks)
    print(f"[score] QN 合成分数就绪，{len(marks)} 个月末", flush=True)

    adv20, illiq20 = liquidity_panels(panel)
    slip_maps, tier_rows = tier_map(marks, eligible, adv20, illiq20)
    summary = tier_summary(tier_rows)
    summary.to_csv(out / "liquidity-tiers.csv", index=False, encoding="utf-8-sig")
    tier_rows.to_csv(out / "liquidity-tiers-by-month.csv", index=False, encoding="utf-8-sig")
    print(f"[tiers]\n{summary.to_string(index=False)}", flush=True)

    frame, crisis_score, crisis_state, _ = crisis_inputs(panel)
    frame.to_csv(out / "market-frame.csv", encoding="utf-8-sig")
    episodes = episode_rows(crisis_state, crisis_score, frame)
    episodes.to_csv(out / "crisis-episodes.csv", index=False, encoding="utf-8-sig")
    print(f"[crisis] 触发 {int(crisis_state.sum())} 天，{len(episodes)} 段", flush=True)

    bench_equity = pool_benchmark(panel, block, load_delisting_events(LAKE), cash=args.cash)
    bench_equity.to_csv(out / "benchmark-equity.csv", encoding="utf-8-sig")
    delisting = load_delisting_events(LAKE)
    bt_bars = {symbol: data.loc[START:END] for symbol, data in bars.items() if len(data)}
    install_broker()

    # 权重只算一次，后续所有规格共用
    strategy = BufferTopK(score, eligible, sector=sector, buffer_rank=BUFFER_RANK)
    base_weights = strategy.generate(bars)
    base_weights.to_csv(out / "weights-buffer.csv.gz", encoding="utf-8-sig", compression="gzip")
    plain = GrowthTopK(score, eligible, sector=sector).generate(bars)
    print(
        f"[weights] 普通 {turnover_stats(plain, marks)['ann_turnover_one_way']:.1%} / "
        f"缓冲 {turnover_stats(base_weights, marks)['ann_turnover_one_way']:.1%}（权重口径，单边/年）",
        flush=True,
    )

    legacy = base_weights.copy()
    nav = shadow_nav(legacy, panel)
    vol_scale_series = vol_scale(nav)
    dd_scale_series = drawdown_scale(nav)
    scaled = {
        "vol": apply_monthly_scale(legacy, vol_scale_series.shift(1), marks),
        "dd": apply_monthly_scale(legacy, dd_scale_series.shift(1), marks),
    }
    print(
        f"[scale] vol 触发降仓 {int((vol_scale_series < 0.999).sum())} 天（均值 {vol_scale_series.mean():.3f}）；"
        f"回撤预算降仓 {int((dd_scale_series < 0.999).sum())} 天（均值 {dd_scale_series.mean():.3f}）",
        flush=True,
    )

    config = dict(
        start=str(START.date()),
        end=str(END.date()),
        initial_cash=args.cash,
        benchmark=None,
        costs=CostModel(slippage_bps=BASE_BPS),
    )

    def run(spec: Run, weights: pd.DataFrame, *, allow: set | None, slip: dict, slip_label: str) -> dict:
        RiskBroker.ALLOW = allow
        RiskBroker.MIN_TRADE = float(spec.min_trade)
        RiskBroker.SLIP = slip
        RiskBroker.BASE_BPS = BASE_BPS
        cfg = BacktestConfig(**config)
        print(f"[{spec.name}] 回测中（allow={len(allow) if allow else '全部'} 天，{slip_label}）…", flush=True)
        t0 = time.time()
        result = Engine(config=cfg, benchmark_prices=None, delisting=delisting, delisting_mode="haircut30").run(
            bt_bars, weights, strategy_name=f"risk-{spec.name}"
        )
        row = {"spec": spec.name, "label": spec.label, **evaluate(result, bench_equity, cash=args.cash, slip_label=slip_label)}
        row["elapsed"] = round(time.time() - t0, 1)
        result.equity.to_csv(out / f"equity-{spec.name}.csv", encoding="utf-8-sig")
        result.daily.to_csv(out / f"daily-{spec.name}.csv", encoding="utf-8-sig")
        if len(result.trades):
            result.trades.to_csv(out / f"trades-{spec.name}.csv", index=False, encoding="utf-8-sig")
        dd = drawdown_series(result.equity)
        row["mdd_date"] = str(dd.idxmin().date())
        print(
            f"  → 累计 {row['total_return']:.2%} 年化 {row['cagr']:.2%} 夏普 {row['sharpe']:.3f} "
            f"MDD {row['max_drawdown']:.2%} 换手 {row['ann_turnover_one_way']:.0%} 滑点 {row['weighted_slip_bp']:.2f}bp "
            f"暴露 {row['avg_exposure']:.3f}（{row['elapsed']}s）",
            flush=True,
        )
        return row

    rebalance_days = set(marks.to_pydatetime())
    tier_slip: dict[str, float] = {}
    for day in marks:
        tier_slip.update(slip_maps.get(day, {}))

    wanted = set(args.runs.split(","))
    rows: list[dict] = []
    for spec in RUNS:
        if spec.name not in wanted:
            continue
        if spec.entry:  # 危机组（3.2）另行处理
            continue
        if spec.name.startswith("legacy"):
            use = tier_slip if spec.tiered else {}
            rows.append(run(spec, legacy, allow=None, slip=use, slip_label="分档" if spec.tiered else "固定 2bp"))
            continue
        if spec.scale:
            weights = scaled[spec.scale]
        else:
            weights = legacy if spec.buffer_rank else plain
        rows.append(
            run(
                spec,
                weights,
                allow=rebalance_days,
                slip=tier_slip if spec.tiered else {},
                slip_label="分档" if spec.tiered else "固定 2bp",
            )
        )

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics-stage31.csv", index=False, encoding="utf-8-sig")
    print(f"[stage31] {len(table)} 组 → {out}（{time.time() - started:.0f}s）", flush=True)

    # ------- 3.2：在 3.1 最优版本上测清仓 -------
    clean = table[table["spec"] == "buf60_slip"]
    if clean.empty:
        print("[stage32] 未跑 buf60_slip，跳过 3.2", flush=True)
    pool = table[~table["spec"].isin(["legacy", "crisis", "gradient", "crisis5k", "gradient5k"])]
    best_name = pool.loc[pool["sharpe"].idxmax(), "spec"] if len(pool) else "buf60_slip"
    print(f"[stage32] 3.1 最优（夏普）= {best_name}", flush=True)

    # 两种基础版：clean=buf60_slip；best=3.1 夏普最优（通常 legacy5k，逐日重置+5000 元门槛）
    BASES = {
        "buf60_slip": dict(weights=legacy, allow=rebalance_days, min_trade=0.0, tag=""),
        "legacy5k": dict(weights=legacy, allow=None, min_trade=5_000.0, tag="5k"),
    }
    exposure_binary = (~crisis_state.fillna(False)).astype(float)
    exposure_grad = gradient_exposure(crisis_score, crisis_state)

    def run_32(base_key: str, mode: str) -> dict:
        cfg = BASES[base_key]
        weights = exposure_to_weights(cfg["weights"], (exposure_binary if mode == "crisis" else exposure_grad))
        changed = set(weights.index[(weights.sum(axis=1).diff().abs() > 1e-9).to_numpy()])
        allow = None if cfg["allow"] is None else (set(cfg["allow"]) | set(pd.DatetimeIndex(changed).to_pydatetime()))
        name = mode + ("5k" if base_key == "legacy5k" else "")
        label = ("逐日重置+5000门槛" if base_key == "legacy5k" else "buf60_slip") + (
            " + 二元清仓" if mode == "crisis" else " + 梯度降仓")
        spec = Run(name, label, BUFFER_RANK, True, None, name, cfg["min_trade"])
        return run(spec, weights, allow=allow, slip=tier_slip, slip_label="分档")

    if {"crisis", "gradient"} & wanted:
        for mode in ("crisis", "gradient"):
            rows.append(run_32("buf60_slip", mode))
        table = pd.DataFrame(rows)
        table.to_csv(out / "metrics-stage32.csv", index=False, encoding="utf-8-sig")
    if {"crisis5k", "gradient5k"} & wanted:
        for mode in ("crisis", "gradient"):
            rows.append(run_32("legacy5k", mode))
        table = pd.DataFrame(rows)
        table.to_csv(out / "metrics-stage32.csv", index=False, encoding="utf-8-sig")


    payload = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "sample": {"start": str(START.date()), "end": str(END.date()), "holdout": "2025-01-01 起未读"},
        "params": {
            "buffer_rank": BUFFER_RANK, "slip_tiers": list(SLIP_TIERS), "vol_target": VOL_TARGET,
            "vol_window": VOL_WINDOW, "dd_floor": DD_FLOOR, "dd_trigger": DD_TRIGGER,
            "crisis": {"enter": CRISIS_ENTER, "exit": CRISIS_EXIT, "confirm": CRISIS_CONFIRM,
                       "exposure": CRISIS_EXPOSURE, "ramp": CRISIS_RAMP, "soft": CRISIS_SOFT},
            "base_version": best_name, "stage32_bases": ["buf60_slip", "legacy5k"],
        },
        "rows": table.to_dict(orient="records"),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out / "risk.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[done] {out}（{payload['elapsed_seconds']}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
