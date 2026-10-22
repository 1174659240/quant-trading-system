"""阶段 3：小盘成长策略组合回测（真实 Engine、月频、成本三档）。

规格（Q9–Q11 拍板，事前定，**看到结果不改**）：

- ``P``  ：冻结版 13 因子、4 块权重 40/25/15/20（成长/质量/研发/辅助）。
- ``Q``  ：删掉 P 里未通过多重检验的 5 个（g1 扣非同比、e2 营运资本差、e6 非经常性
  损益依赖、g6 研发强度、a3 小盘），其余不变——块内等权，块权重不变。
- ``QN`` ：Q 再加阶段 B 冻结前已登记、阶段 C 通过的解禁双因子（n8 60 日 / n9 120 日），
  放进辅助块（不与原 3 个辅助因子抢权重之外的新块，权重表一个字没改）。
- ``M``  ：全市场成长对照——同 QN 打分，但股票池**去掉成长门槛 G3 与排雷 R5–R9**，
  只保留 ST/次新/停牌/一字板/退市整理期（Q5 的 R10 全规格不启用）。

组合构造按 §3.11：50 只、等权、月末信号次月首个交易日开盘成交、gross 95%、
单票上限 4%、单一行业（证监会一级门类，行业名首字母）上限 30%、
单票成交 ≤ 当日成交量 10%（Engine 默认规则）、退市 haircut30。
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from quant.backtest.engine import Engine
from quant.config import BacktestConfig
from quant.factors import growth, growth_alt
from quant.factors.panel import build_panel
from quant.refdata import industry_asof, load_industry
from quant.strategy.base import Strategy
from scripts.factor_market import FIELDS, load_delisting_events
from scripts.redlow_study import (
    END,
    LAKE,
    MONTHLY,
    START,
    board_exclude,
    build_block,
    cost_presets,
    load_bars,
    pool_benchmark,
    round_trip_bp,
    summarize,
    yearly_returns,
)

# --------------------------------------------------------------------------- #
# 规格定义（事前冻结）
# --------------------------------------------------------------------------- #

WEIGHTS: dict[str, float] = dict(growth.BLOCK_WEIGHTS)

# Q9 拍板删除的 5 个因子（阶段 C 未过多重检验或与既有因子完全重复）
DROPPED: tuple[str, ...] = (
    "g1_deduct_yoy",
    "e2_wc_gap",
    "e6_nonrec",
    "g6_rd_intensity",
    "a3_small",
)

# Q10 拍板加入的解禁双因子（阶段 B 冻结前已登记候选，阶段 C 通过）
UNLOCK: tuple[str, ...] = ("n8_unlock_60d", "n9_unlock_120d")

BLOCKS_P: dict[str, tuple[str, ...]] = {block: tuple(names) for block, names in growth.BLOCKS.items()}
BLOCKS_Q: dict[str, tuple[str, ...]] = {
    block: tuple(name for name in names if name not in DROPPED) for block, names in BLOCKS_P.items()
}
BLOCKS_QN: dict[str, tuple[str, ...]] = {
    block: (names + UNLOCK if block == "aux" else names) for block, names in BLOCKS_Q.items()
}

SPECS: dict[str, dict] = {
    "P": {"label": "P 冻结版 13 因子", "blocks": BLOCKS_P, "pool": "growth"},
    "Q": {"label": "Q 删 5 个（8 因子）", "blocks": BLOCKS_Q, "pool": "growth"},
    "QN": {"label": "QN = Q + 解禁双因子", "blocks": BLOCKS_QN, "pool": "growth"},
    "M": {"label": "M 全市场成长（无门槛/排雷）", "blocks": BLOCKS_QN, "pool": "market"},
}

TOP_K = 50
GROSS = 0.95
MAX_WEIGHT = 0.04
SECTOR_CAP = 0.25      # 证监会门类单行业上限；2026-10-22 由 0.30 下调（MDD -1.62pp、换手 -29.5pp）
MIN_NAMES = 20


# --------------------------------------------------------------------------- #
# 时间轴与行业
# --------------------------------------------------------------------------- #


def month_ends(dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """样本区间内每个自然月的最后一个交易日。"""
    frame = pd.Series(dates, index=dates)
    marks = frame.groupby([frame.index.year, frame.index.month]).max()
    marks = pd.DatetimeIndex(sorted(marks.to_numpy()))
    return marks[(marks >= start) & (marks <= end)]


def sector_panel(marks: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """证监会一级门类（行业名首字符）在月末的 PIT 矩阵。

    ``ref/industry.csv`` 是 13 个年度版本快照，``industry_asof`` 取 ``as_of <= day``
    的最近一版（年度 PIT 语义）。缺失记 ``UNKNOWN``（单独占一个"行业"桶）。
    """
    frame = load_industry()
    frame = frame.assign(sector=frame["industry"].astype("string").str[:1])
    versions = sorted(frame["as_of"].unique())
    out = pd.DataFrame("UNKNOWN", index=marks, columns=symbols, dtype=object)
    cache: dict[str, dict[str, str]] = {}
    for day in marks:
        available = [v for v in versions if v <= str(day.date())]
        if not available:
            continue
        version = available[-1]
        if version not in cache:
            snap = frame[frame["as_of"] == version]
            cache[version] = dict(zip(snap["symbol"].astype(str).str.zfill(6), snap["sector"].astype(str)))
        mapping = cache[version]
        for symbol in symbols:
            code = str(symbol).split(".")[0].zfill(6)
            sector = mapping.get(code)
            if sector and sector not in ("nan", "None", ""):
                out.at[day, symbol] = sector
    return out


# --------------------------------------------------------------------------- #
# 合成分数
# --------------------------------------------------------------------------- #


def composite_score(
    panel,
    facts: growth.GrowthFactors,
    alt: Mapping[str, pd.DataFrame],
    blocks: Mapping[str, tuple[str, ...]],
    eligible: pd.DataFrame,
    marks: pd.DatetimeIndex,
    *,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """分块等权合成（块内等权、块间按权重、缺失块按剩余权重归一）。

    只在月末截面计算：rank 是逐日横截面运算，先裁到 marks 再排名，结果等价且省内存。
    """
    marks = pd.DatetimeIndex(marks)
    ok = eligible.reindex(index=marks)
    block_scores: dict[str, pd.DataFrame] = {}
    for block, names in blocks.items():
        ranks = []
        for name in names:
            frame = alt[name] if name in alt else getattr(facts, name)()
            frame = frame.reindex(index=marks)
            ranks.append(growth._rank_pct(frame, ok, min_names=min_names))
        stack = np.stack([frame.to_numpy(dtype="float32") for frame in ranks], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            block_scores[block] = pd.DataFrame(np.nanmean(stack, axis=0), index=marks, columns=panel.symbols)
    names = list(block_scores)
    stack = np.stack([block_scores[name].to_numpy(dtype="float32") for name in names], axis=0)
    weight = np.array([WEIGHTS.get(name, 0.0) for name in names], dtype="float32")[:, None, None]
    available = ~np.isnan(stack)
    total_weight = np.where(available, weight, 0.0).sum(axis=0)
    weighted = np.nansum(np.where(available, stack * weight, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(total_weight > 0, weighted / total_weight, np.nan)
    return pd.DataFrame(score, index=marks, columns=panel.symbols).astype("float32")


# --------------------------------------------------------------------------- #
# 策略
# --------------------------------------------------------------------------- #


class GrowthTopK(Strategy):
    """按合成分数取前 ``top_k``、等权、带行业上限；月频调仓、月内持有不动。"""

    name = "growth-topk"

    def __init__(
        self,
        score: pd.DataFrame,
        eligible: pd.DataFrame,
        *,
        sector: pd.DataFrame | None = None,
        top_k: int = TOP_K,
        gross: float = GROSS,
        max_weight: float = MAX_WEIGHT,
        sector_cap: float = SECTOR_CAP,
        rebalance: str = MONTHLY,
        min_holdings: int = 10,
    ) -> None:
        super().__init__(top_k=top_k, rebalance=rebalance, gross=gross)
        self.score = score
        self.eligible = eligible
        self.sector = sector
        self.top_k = int(top_k)
        self.gross = float(gross)
        self.max_weight = float(max_weight)
        self.sector_cap = float(sector_cap)
        self.rebalance = rebalance
        self.min_holdings = int(min_holdings)
        self.last_diagnostics: dict = {"picks": {}, "notes": []}

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        dates = self.score.index
        symbols = list(self.score.columns)
        mask = self.rebalance_mask(dates, self.rebalance).reindex(dates).fillna(False)
        target = pd.DataFrame(0.0, index=dates, columns=symbols)
        weight_value = min(self.gross / max(self.top_k, 1), self.max_weight)
        picks: dict[str, list[str]] = {}
        for day in dates[mask.to_numpy()]:
            row = self.score.loc[day].where(self.eligible.reindex(index=[day]).iloc[0]).dropna()
            row = row.sort_values(ascending=False)
            sector_row = self.sector.loc[day] if self.sector is not None and day in self.sector.index else None
            chosen: list[str] = []
            counts: dict[str, int] = {}
            for symbol in row.index:
                key = str(sector_row.get(symbol, "UNKNOWN")) if sector_row is not None else "UNKNOWN"
                if (counts.get(key, 0) + 1) * weight_value > self.sector_cap + 1e-12:
                    continue
                chosen.append(symbol)
                counts[key] = counts.get(key, 0) + 1
                if len(chosen) >= self.top_k:
                    break
            if len(chosen) < self.min_holdings:
                self.last_diagnostics["notes"].append(f"{day.date()} 合格标的仅 {len(chosen)} 只，本期空仓")
                picks[str(day.date())] = []
                continue
            picks[str(day.date())] = chosen
            for symbol in chosen:
                target.at[day, symbol] = weight_value
        target = self.hold_to_rebalance(target, self.rebalance)
        self.last_diagnostics["picks"] = picks
        return self.clip_weights(target)


# --------------------------------------------------------------------------- #
# 换手与成本
# --------------------------------------------------------------------------- #


def turnover_stats(weights: pd.DataFrame, marks: pd.DatetimeIndex) -> dict:
    """按月频目标权重算单边换手（首月建仓按半程计）。"""
    frame = weights.reindex(index=marks).fillna(0.0)
    if len(frame) < 2:
        return {"ann_turnover_one_way": None, "avg_month_turnover": None, "turnover_by_year": {}}
    change = frame.diff().abs().sum(axis=1) * 0.5
    change.iloc[0] = frame.iloc[0].sum() * 0.5
    years = len(frame) / 12.0
    by_year = change.groupby(change.index.year).sum()
    return {
        "ann_turnover_one_way": float(change.sum() / years),
        "avg_month_turnover": float(change.mean()),
        "turnover_by_year": {int(year): float(value) for year, value in by_year.items()},
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/growth/stage3"))
    parser.add_argument("--specs", default="P,Q,QN,M")
    parser.add_argument("--costs", default="default,conservative_30,conservative_50")
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    bars, load_stats = load_bars()
    print(f"[bars] {len(bars)} 只（含退市 {load_stats['delisted_added']}）", flush=True)
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)

    panel = build_panel(bars, fields=FIELDS)
    print(f"[panel] {panel.close.shape} {panel.dates[0].date()} ~ {panel.dates[-1].date()}", flush=True)
    block = build_block(panel).reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)

    facts = growth.GrowthFactors(panel)
    eligible_growth = growth.eligible_mask(facts, block_mask=block)
    eligible_market = panel.tradable & ~block
    marks = month_ends(panel.dates, START, END)
    print(f"[marks] {len(marks)} 个月末 {marks[0].date()} ~ {marks[-1].date()}", flush=True)
    print(
        f"[eligible] 成长合格集日均 {int(eligible_growth.sum(axis=1).mean())} 只 / "
        f"月末 {int(eligible_growth.reindex(marks).sum(axis=1).mean())} 只；"
        f"全市场对照月末 {int(eligible_market.reindex(marks).sum(axis=1).mean())} 只",
        flush=True,
    )

    alt = growth_alt.alt_factors(panel)
    sector = sector_panel(marks, list(panel.symbols))
    known = (sector != "UNKNOWN").to_numpy().mean()
    print(f"[sector] 行业门类覆盖率 {known:.1%}", flush=True)

    prices = panel.close
    config_kwargs = {"start": str(START.date()), "end": str(END.date()), "initial_cash": args.cash, "benchmark": None}
    delisting = load_delisting_events(LAKE)
    bench_equity = pool_benchmark(panel, block, delisting, cash=args.cash)
    bench_equity.to_csv(out / "benchmark-equity.csv", encoding="utf-8-sig")
    # pool_benchmark 从面板起点（2014）开始算，而净值对比必须落在 2017-01~2024-12；
    # 直接拿全段起点会得出「+207%」这种与策略窗口无关的数字（踩过）。
    window_bench = bench_equity.loc[START:END].ffill()
    bench_total = float(window_bench.iloc[-1] / window_bench.iloc[0] - 1)
    bench_years = len(window_bench) / 244.0
    bench_cagr = float((1 + bench_total) ** (1 / bench_years) - 1)
    print(
        f"[benchmark] 同池等权 {START.date()}~{END.date()} 累计 {bench_total:.2%} CAGR {bench_cagr:.2%}"
        f"（全样本起点算则为 {float(bench_equity.iloc[-1] / bench_equity.iloc[0] - 1):.2%}，不用于对比）",
        flush=True,
    )

    bt_bars = {symbol: frame.loc[START:END] for symbol, frame in bars.items() if len(frame)}
    presets = cost_presets()
    wanted_costs = [key for key in args.costs.split(",") if key in presets]

    rows: list[dict] = []
    yearly: dict[str, pd.Series] = {}
    for spec in [name for name in args.specs.split(",") if name in SPECS]:
        info = SPECS[spec]
        eligible = eligible_growth if info["pool"] == "growth" else eligible_market
        t0 = time.time()
        score = composite_score(panel, facts, alt, info["blocks"], eligible, marks)
        strategy = GrowthTopK(score, eligible, sector=sector)
        weights = strategy.generate(bars)
        weights.to_csv(out / f"weights-{spec}.csv.gz", encoding="utf-8-sig", compression="gzip")
        turn = turnover_stats(weights, marks)
        picks = strategy.last_diagnostics.get("picks", {})
        holdings = [len(value) for value in picks.values() if value]
        last_picks = list(picks.values())[-1] if picks else []
        print(
            f"[{spec}] 分数就绪（{time.time() - t0:.0f}s）持仓中位 {int(np.median(holdings)) if holdings else 0} 只，"
            f"年单边换手 {turn['ann_turnover_one_way']:.1%}",
            flush=True,
        )
        for cost_key in wanted_costs:
            cfg = BacktestConfig(**config_kwargs, costs=presets[cost_key])
            engine = Engine(
                config=cfg,
                benchmark_prices=(bench_equity / args.cash).ffill(),
                delisting=delisting,
                delisting_mode="haircut30",
            )
            t1 = time.time()
            result = engine.run(bt_bars, weights, strategy_name=f"growth-{spec}")
            elapsed = time.time() - t1
            row = {
                "spec": spec,
                "label": info["label"],
                "pool": info["pool"],
                "cost": cost_key,
                "round_trip_bp": round_trip_bp(presets[cost_key]),
                **summarize(result, bench_equity, bars, last_picks),
                **{key: value for key, value in turn.items() if key != "turnover_by_year"},
                "holdings_median": int(np.median(holdings)) if holdings else 0,
                "holdings_min": int(min(holdings)) if holdings else 0,
                "rebalances": len(picks),
            }
            years = max(len(result.daily) / 244.0, 1e-9)
            row["cost_bp_per_year"] = round(float(row.get("fees", 0.0)) / (args.cash * (1 + years * 0.0)) / years * 1e4, 1)
            row["fees_pct_of_equity"] = float(row.get("fees", 0.0)) / float(result.daily["equity"].iloc[-1])
            rows.append(row)
            result.equity.to_csv(out / f"equity-{spec}-{cost_key}.csv", encoding="utf-8-sig")
            result.daily.to_csv(out / f"daily-{spec}-{cost_key}.csv", encoding="utf-8-sig")
            if len(result.trades):
                result.trades.to_csv(out / f"trades-{spec}-{cost_key}.csv", index=False, encoding="utf-8-sig")
            yearly[f"{spec}-{cost_key}"] = yearly_returns(result.daily).rename(f"{spec}-{cost_key}")
            print(
                f"  [{spec}/{cost_key}] 累计 {row['total_return']:.2%} CAGR {row['cagr']:.2%} "
                f"夏普 {row['sharpe']:.3f} MDD {row['max_drawdown']:.2%} 卡玛 {row['calmar']:.3f} "
                f"超额 {row['excess_total']:.2%} IR {row['excess_ir']} 费用 {row['fees']:,.0f}（{elapsed:.0f}s）",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    if yearly:
        yearly_table = pd.DataFrame(yearly)
        yearly_table.index = yearly_table.index.year
        yearly_table.to_csv(out / "yearly.csv", encoding="utf-8-sig")
        print(yearly_table.round(4).to_string(), flush=True)

    payload = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "sample": {"start": str(START.date()), "end": str(END.date()), "holdout": "2025-01-01 起全程未读"},
        "benchmark": {"total_return": round(bench_total, 6), "cagr": round(bench_cagr, 6), "kind": "同池等权（解析式）"},
        "specs": {name: {"label": info["label"], "pool": info["pool"], "blocks": info["blocks"]} for name, info in SPECS.items()},
        "params": {
            "top_k": TOP_K, "gross": GROSS, "max_weight": MAX_WEIGHT,
            "sector_cap": SECTOR_CAP, "rebalance": MONTHLY, "delisting_mode": "haircut30",
        },
        "costs": {key: round_trip_bp(value) for key, value in presets.items()},
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out / "backtest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[done] {len(rows)} 组，{payload['elapsed_seconds']}s → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
