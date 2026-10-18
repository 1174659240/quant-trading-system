"""事件因子研究：IC / 分层 / 正交性 / 合成前后对照 / 容量 / 成本 / 压力。

只读 ``<= 2024-12-31``（holdout 锁），未读 2025 起任何数据。

阶段（``--phase``）：
- ``ic``：月度 IC（20 日前瞻）、分年度、覆盖率；
- ``quantile``：月末分层 Q1—Q5；
- ``corr``：与现有因子的秩相关（同源判定 |ρ|≥0.7）；
- ``bt``：组合回测（基准 = 同池等权解析式），含三档成本与压力区间；
- ``capacity``：单票 ADV × 10% 参与率下的组合容量；
- ``style``：剥离市场与风格后的纯事件增量（回归截距）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant import events
from quant.backtest import Engine
from quant.config import BacktestConfig
from quant.factors import fundamental, style
from quant.factors import events as event_factors
from quant.factors.evaluate import ic_summary, rank_ic
from quant.factors.library import compute, names as library_names
from quant.factors.panel import build_panel
from quant.strategy import MultiFactorStrategy
from scripts.factor_market import FIELDS, load_delisting_events
from scripts.factor_monthly_ic import month_end_dates
from scripts.redlow_study import (
    END, LAKE, MONTHLY, START, board_exclude, build_block, cost_presets, load_bars,
    monthly_returns, pool_benchmark, round_trip_bp, summarize, summarize_benchmark,
    window_stats, yearly_returns,
)

HORIZON = 20
STRESS_WINDOWS = [("2024-01-01", "2024-02-29")]
BASE = ["div_yield", "vol_60", "turnover_20"]
EVENT_SET = list(event_factors.EVENT_FACTORS)
CORR_PEERS = [
    "momentum", "reversal", "low_vol", "max_ret", "turnover_abn", "amihud", "gap",
    "div_yield", "vol_60", "turnover_20", "mcap", "fmcap", "bp", "ep", "roe", "size",
]


def _spearman(a: pd.Series, b: pd.Series) -> float:
    frame = pd.concat([a, b], axis=1).dropna()
    if len(frame) < 10:
        return np.nan
    return float(frame.iloc[:, 0].rank().corr(frame.iloc[:, 1].rank()))


def _monthly_ic(panel, frame: pd.DataFrame, forward: pd.DataFrame, st_mask, days) -> tuple[pd.Series, pd.Series]:
    clean = frame.where(~st_mask)
    ic = rank_ic(clean, forward, min_count=30)
    ic = ic.loc[days.intersection(ic.dropna().index)].dropna()
    coverage = clean.notna().sum(axis=1).loc[days]
    return ic, coverage


def phase_ic(panel, st_mask, days, forward, out: Path) -> pd.DataFrame:
    rows, yearly = [], {}
    for name in EVENT_SET:
        started = time.perf_counter()
        frame = compute(name, panel)
        ic, coverage = _monthly_ic(panel, frame, forward, st_mask, days)
        stats = ic_summary(ic, periods_per_year=12)
        row = {"factor": name, "months": int(stats.get("n") or 0)}
        row.update({key: stats.get(key) for key in ("ic_mean", "ic_std", "ic_ir", "t_stat", "positive_rate")})
        row["abs_mean"] = float(ic.abs().mean()) if len(ic) else np.nan
        row["coverage_mean"] = float(coverage.mean()) if len(coverage) else np.nan
        row["coverage_min"] = float(coverage.min()) if len(coverage) else np.nan
        row["coverage_max"] = float(coverage.max()) if len(coverage) else np.nan
        row["factor_cover_share"] = float((frame.notna().sum().sum() / frame.size)) if frame.size else np.nan
        rows.append(row)
        by_year = ic.groupby(ic.index.year).mean()
        yearly[name] = {int(year): float(value) for year, value in by_year.items()}
        print(f"  {name:20s} IC {row['ic_mean']:+.4f} IR {row['ic_ir']:+.3f} 覆盖 {row['coverage_mean']:.0f} "
              f"({time.perf_counter() - started:.1f}s)", flush=True)
        del frame
    table = pd.DataFrame(rows).sort_values("ic_ir", ascending=False).reset_index(drop=True)
    table.to_csv(out / "ic.csv", index=False, encoding="utf-8-sig")
    year_table = pd.DataFrame(yearly).T
    year_table.index.name = "factor"
    year_table = year_table.reindex(sorted(year_table.columns), axis=1)
    year_table.to_csv(out / "yearly-ic.csv", encoding="utf-8-sig")
    return table


def phase_quantile(panel, st_mask, days, forward, out: Path) -> pd.DataFrame:
    rows = []
    for name in EVENT_SET:
        frame = compute(name, panel).where(~st_mask)
        values = frame.loc[days]
        fwd = forward.reindex(index=days).where(~st_mask.reindex(index=days))
        buckets, triggers = [], []
        for day, row in values.iterrows():
            sample = pd.concat([row.rename("factor"), fwd.loc[day].rename("fwd")], axis=1).dropna()
            if len(sample) < 100:
                continue
            # 0/1 与稀疏事件因子大量并列 → 按「排序后等分」切组，不用 pct 分位（否则会出 NaN 组）
            ordered = sample.sort_values("factor", kind="mergesort")
            edges = np.linspace(0, len(ordered), 6).round().astype(int)
            for q in range(1, 6):
                part = ordered.iloc[edges[q - 1]:edges[q], 1]
                if len(part):
                    buckets.append({"day": day, "q": q, "ret": float(part.mean())})
            # 触发组（本月该股有事件，|factor|>0）vs 未触发组，避免大量并列值把分层切乱
            active = sample["factor"].abs() > 0
            if active.any() and (~active).any():
                triggers.append({
                    "day": day,
                    "active": float(sample.loc[active, "fwd"].mean()),
                    "rest": float(sample.loc[~active, "fwd"].mean()),
                    "share": float(active.mean()),
                })
        if not buckets:
            continue
        table = pd.DataFrame(buckets)
        means = table.groupby("q")["ret"].mean()
        q5, q1 = float(means.get(5, np.nan)), float(means.get(1, np.nan))
        mono = float(pd.Series(means.values).rank().corr(pd.Series(range(len(means)))) ) if len(means) == 5 else np.nan
        row = {"factor": name, "q5_minus_q1": round((q5 - q1) * 100, 4), "monotonic": round(mono, 3) if mono == mono else None}
        row.update({f"Q{q}": round(float(means.get(q, np.nan)) * 100, 4) for q in range(1, 6)})
        if triggers:
            trig = pd.DataFrame(triggers)
            diff = trig["active"] - trig["rest"]
            row.update({
                "trigger_share": round(float(trig["share"].mean()), 4),
                "active_ret": round(float(trig["active"].mean()) * 100, 4),
                "rest_ret": round(float(trig["rest"].mean()) * 100, 4),
                "active_minus_rest": round(float(diff.mean()) * 100, 4),
                "t_diff": round(float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))), 2) if len(diff) > 2 else None,
            })
        rows.append(row)
        del frame, values
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values("q5_minus_q1", ascending=False)
    table.to_csv(out / "quantiles.csv", index=False, encoding="utf-8-sig")
    return table


def phase_corr(panel, st_mask, days, out: Path) -> pd.DataFrame:
    peers = [name for name in CORR_PEERS if name in library_names()]
    frames = {name: compute(name, panel).where(~st_mask) for name in peers}
    rows = []
    for name in EVENT_SET:
        frame = compute(name, panel).where(~st_mask)
        for peer, other in frames.items():
            values = []
            for day in days:
                rho = _spearman(frame.loc[day], other.loc[day])
                if rho == rho:
                    values.append(rho)
            mean = float(np.mean(values)) if values else np.nan
            rows.append({"factor": name, "peer": peer, "corr": round(mean, 3) if mean == mean else None})
        print(f"  corr {name} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "corr.csv", index=False, encoding="utf-8-sig")

    pivot = table.pivot(index="factor", columns="peer", values="corr").reindex(EVENT_SET)
    same = []
    for name in EVENT_SET:
        row = pivot.loc[name].dropna()
        hits = row[row.abs() >= 0.7]
        same.append({
            "factor": name,
            "max_abs_corr": round(float(row.abs().max()), 3) if len(row) else None,
            "closest": row.abs().idxmax() if len(row) else None,
            "same_source": "是" if len(hits) else "否",
            "hits": "、".join(f"{peer}({value:+.2f})" for peer, value in hits.items()),
        })
    pd.DataFrame(same).to_csv(out / "same-source.csv", index=False, encoding="utf-8-sig")
    return pivot


def _run_config(name, factors, top_k, cost_key, block, bench_equity, cash, out, bars, bt_bars, start, end):
    presets = cost_presets()
    strategy = MultiFactorStrategy(
        factors=factors, weighting="equal", top_k=top_k, rebalance=MONTHLY,
        gross_exposure=0.95, max_weight=0.10, min_price=2.0, min_amount=0.0,
        min_history=120, min_universe=100, exclude=block,
    )
    cfg = BacktestConfig(start=start, end=end, initial_cash=cash, costs=presets[cost_key], benchmark=None)
    engine = Engine(config=cfg, benchmark_prices=(bench_equity / cash).ffill(),
                    delisting=load_delisting_events(LAKE), delisting_mode="haircut30")
    started = time.time()
    weights = strategy.generate(bars)
    result = engine.run(bt_bars, weights, strategy_name=strategy.describe())
    picks = strategy.last_diagnostics.get("picks", {})
    holdings = list(picks.values())[-1] if picks else []
    row = {
        "config": name, "factors": factors, "top_k": top_k, "cost": cost_key,
        "round_trip_bp": round_trip_bp(presets[cost_key]),
        **summarize(result, bench_equity, bars, holdings),
    }
    result.equity.to_csv(out / f"equity-{name}.csv", encoding="utf-8-sig")
    if len(result.trades):
        result.trades.to_csv(out / f"trades-{name}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"equity": result.daily["equity"]}).to_csv(out / f"daily-{name}.csv", encoding="utf-8-sig")
    print(f"[{name}] 累计 {row['total_return']:.2%} CAGR {row['cagr']:.2%} 夏普 {row['sharpe']} "
          f"MDD {row['max_drawdown']:.2%} 超额 {row['excess_total']:.2%} 超额IR {row['excess_ir']} "
          f"换手 {row['turnover']:.1%}（{time.time() - started:.0f}s）", flush=True)
    return row, result, picks


def phase_bt(panel, block, bench_equity, cash, out, bars, bt_bars, start, end, args) -> pd.DataFrame:
    groups = {
        "base": BASE,
        "base+lhb": BASE + ["lhb_netbuy_20", "lhb_inst_net_20", "lhb_flag_5"],
        "base+insider": BASE + ["insider_net_20", "insider_buy_20"],
        "base+buyback": BASE + ["buyback_flag_60", "buyback_ratio_60"],
        "base+st": BASE + ["st_unflag_20", "st_flag_20"],
        "base+transfer": BASE + ["high_transfer_60", "div_plan_60"],
        "base+events": BASE + EVENT_SET,
    }
    if args.smoke:
        groups = {"base": BASE, "base+events": BASE + EVENT_SET}
    rows, results = [], {}
    bench_row = {"config": "benchmark_equal_weight", **summarize_benchmark(bench_equity)}
    rows.append(bench_row)
    for name, factors in groups.items():
        row, result, picks = _run_config(name, ",".join(factors), args.topk, "default", block,
                                         bench_equity, cash, out, bars, bt_bars, start, end)
        rows.append(row)
        results[name] = (result, picks)
    for cost_key in ("conservative_30", "conservative_50"):
        name = f"base+events-{cost_key}"
        row, result, picks = _run_config(name, ",".join(BASE + EVENT_SET), args.topk, cost_key, block,
                                         bench_equity, cash, out, bars, bt_bars, start, end)
        rows.append(row)
        results[name] = (result, picks)
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")

    yearly = {name: yearly_returns(result.daily).rename(name) for name, (result, _) in results.items()}
    year_table = pd.DataFrame(yearly)
    year_table.index = year_table.index.year
    year_table.to_csv(out / "yearly.csv", encoding="utf-8-sig")
    pd.DataFrame({name: monthly_returns(result.daily) for name, (result, _) in results.items()}).to_csv(
        out / "monthly.csv", encoding="utf-8-sig")

    def picks_at(picks: dict, day: pd.Timestamp) -> list[str]:
        keys = sorted(key for key in picks if pd.Timestamp(key) <= day)
        return list(picks[keys[-1]]) if keys else []

    stress_rows = []
    for start_day, end_day in STRESS_WINDOWS:
        bench_stress = window_stats(pd.DataFrame({"equity": bench_equity}), start_day, end_day, bars, [])
        bench_stress["config"] = "benchmark_equal_weight"
        stress_rows.append(bench_stress)
    for name, (result, picks) in results.items():
        if name not in {"base", "base+events"} and not name.startswith("base+events-"):
            continue
        for start_day, end_day in STRESS_WINDOWS:
            row = window_stats(result.daily, start_day, end_day, bars, picks_at(picks, pd.Timestamp(start_day)))
            row["config"] = name
            stress_rows.append(row)
    pd.DataFrame(stress_rows).to_csv(out / "stress.csv", index=False, encoding="utf-8-sig")
    return table


def phase_capacity(panel, block, days, out: Path, top_k: int = 50, participation: float = 0.10) -> pd.DataFrame:
    adv = panel.amount.rolling(20, min_periods=5).median()
    tradable = (~block) & panel.tradable & panel.close.ge(2.0)
    rows = []
    for name in EVENT_SET:
        frame = compute(name, panel)
        picks = []
        for day in days:
            row = frame.loc[day].where(tradable.loc[day]).dropna()
            if len(row) < top_k:
                continue
            picks.append(row.nlargest(top_k).index)
        if not picks:
            rows.append({"factor": name, "months": 0})
            continue
        values = []
        for day, symbols in zip(days, picks):
            part = adv.loc[day, symbols].dropna()
            if len(part):
                values.append(float(part.median()))
        median_adv = float(np.median(values)) if values else np.nan
        rows.append({
            "factor": name,
            "months": len(picks),
            "median_adv_yuan": round(median_adv, 0) if median_adv == median_adv else None,
            "single_cap_wan": round(median_adv * participation / 1e4, 1) if median_adv == median_adv else None,
            "portfolio_cap_yi": round(median_adv * participation * top_k / 1e8, 3) if median_adv == median_adv else None,
            "participation": participation,
            "top_k": top_k,
        })
        print(f"  capacity {name}: ADV {median_adv/1e6:.1f} 百万 → 组合 {rows[-1]['portfolio_cap_yi']} 亿", flush=True)
        del frame
    table = pd.DataFrame(rows).sort_values("portfolio_cap_yi", ascending=False).reset_index(drop=True)
    table.to_csv(out / "capacity.csv", index=False, encoding="utf-8-sig")
    return table


def phase_style(out: Path) -> pd.DataFrame:
    """剥离市场与风格：对「策略日超额」回归 [市场超额, 小市值多空, 低换手多空]，取截距。"""
    base = pd.read_csv(out / "daily-base.csv", index_col=0, parse_dates=True)["equity"]
    full = pd.read_csv(out / "daily-base+events.csv", index_col=0, parse_dates=True)["equity"]
    bench = None
    for candidate in ("equity-benchmark_equal_weight.csv", "benchmark-equity.csv"):
        path = out / candidate
        if path.is_file():
            bench = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            break
    if bench is None:
        return pd.DataFrame()
    index = base.index.intersection(full.index).intersection(bench.index)
    market = bench.pct_change().reindex(index).fillna(0.0)
    rows = []
    for name, series in (("base", base), ("base+events", full)):
        ret = series.pct_change().reindex(index).fillna(0.0)
        excess = ret - market
        design = np.column_stack([np.ones(len(index)), market.values])
        beta, *_ = np.linalg.lstsq(design, excess.values, rcond=None)
        alpha_daily = float(beta[0])
        rows.append({
            "config": name,
            "alpha_ann_vs_market": round(alpha_daily * 244, 6),
            "beta_market": round(float(beta[1]), 3),
            "excess_mean_ann": round(float(excess.mean() * 244), 6),
            "excess_vol_ann": round(float(excess.std(ddof=1) * np.sqrt(244)), 6),
            "info_ratio": round(float(excess.mean() / excess.std(ddof=1) * np.sqrt(244)), 3) if excess.std() > 0 else None,
        })
    table = pd.DataFrame(rows)
    table["increment_alpha_ann"] = round(table["alpha_ann_vs_market"].diff(), 6)
    table.to_csv(out / "style-alpha.csv", index=False, encoding="utf-8-sig")
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="事件因子研究")
    parser.add_argument("--out", default="results/event-research")
    parser.add_argument("--start", default=str(START.date()))
    parser.add_argument("--end", default=str(END.date()))
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--smoke", type=int, default=None)
    parser.add_argument("--phase", default="all",
                        choices=["all", "ic", "quantile", "corr", "bt", "capacity", "style"])
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    event_factors.register()
    fundamental.register()
    style  # 导入即注册 18 个风格因子

    bars, load_stats = load_bars(smoke=args.smoke)
    drop = board_exclude(bars)
    for symbol in drop:
        bars.pop(symbol, None)
    panel = build_panel(bars, fields=FIELDS)
    panel = panel.slice(pd.Timestamp(args.start), pd.Timestamp(args.end))
    print(f"panel {panel.close.shape} {panel.dates[0].date()} ~ {panel.dates[-1].date()}（剔北交所 {len(drop)}）",
          flush=True)
    block = build_block(panel)
    st_mask = block.reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)
    st_share = float(st_mask.to_numpy().mean())
    print(f"剔除占比 {st_share:.2%}", flush=True)

    days = month_end_dates(panel.dates)
    forward = panel.forward_returns(HORIZON)
    print(f"月末截面 {len(days)} 个", flush=True)

    if args.phase in {"all", "ic"}:
        phase_ic(panel, st_mask, days, forward, out)
    if args.phase in {"all", "quantile"}:
        phase_quantile(panel, st_mask, days, forward, out)
    if args.phase in {"all", "corr"}:
        phase_corr(panel, st_mask, days, out)
    if args.phase in {"all", "capacity"}:
        phase_capacity(panel, block, days, out, top_k=args.topk)
    if args.phase in {"all", "bt"}:
        delisting = load_delisting_events(LAKE)
        bench_equity = pool_benchmark(panel, block, delisting, cash=args.cash)
        bench_equity.to_csv(out / "benchmark-equity.csv", encoding="utf-8-sig")
        bt_bars = {symbol: frame.loc[pd.Timestamp(args.start):pd.Timestamp(args.end)]
                   for symbol, frame in bars.items() if len(frame)}
        phase_bt(panel, block, bench_equity, args.cash, out, bars, bt_bars, args.start, args.end, args)
    if args.phase in {"all", "style"}:
        phase_style(out)

    meta = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "seconds": round(time.time() - t0, 1),
        "holdout": "锁定 2025-01-01 起，未读",
        "panel": {"dates": int(len(panel.dates)), "symbols": int(len(panel.symbols)),
                  "first": str(panel.dates[0].date()), "last": str(panel.dates[-1].date())},
        "block_share": round(st_share, 4),
        "months": int(len(days)),
        "events": EVENT_SET,
        "base": BASE,
        "load_stats": load_stats,
        "args": vars(args),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"完成 {meta['seconds']}s → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
