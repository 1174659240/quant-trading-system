"""任务二：容量受限因子研究（第一步，只用现有日线数据）。

对 18 个「小资金容量因子」统一输出六项：

1. 月度 IC（月末截面 × 前瞻 20 日 rank IC）、IR、分年度、正年数；
2. 分层 Q1—Q5 均值与单调性（检验上一轮 D>C>E 非单调是否为分组定义问题）；
3. 与现有全部因子（量价 15 + 基本面 13）的横截面秩相关矩阵，标注同源；
4. 容量测算：单票上限 = 日均成交额 × 10%，组合总容量按等权 k 只的绑定约束；
5. 成本敏感性：默认（往返 14.2bp）/ 双边 0.3% / 双边 0.5%；
6. 尾部压力：样本内 2024-01-01~2024-02-29（holdout 2025 起全程不读）。

用法::

    python -m scripts.capacity_study --out results/capacity
    python -m scripts.capacity_study --smoke 300 --topk 30 --costs default --out results/_smoke-capacity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quant.backtest import Engine  # noqa: E402
from quant.config import BacktestConfig  # noqa: E402
from quant.factors import fundamental, library  # noqa: E402
from quant.factors.evaluate import ic_summary, quantile_returns, rank_ic  # noqa: E402
from quant.factors.panel import build_panel  # noqa: E402
from quant.strategy.multifactor import MultiFactorStrategy  # noqa: E402
from scripts.redlow_study import (  # noqa: E402
    END,
    HOLDOUT,
    LAKE,
    START,
    EqualWeightPool,
    board_exclude,
    build_block,
    cost_presets,
    load_bars,
    load_delisting_events,
    monthly_returns,
    pool_benchmark,
    round_trip_bp,
    summarize,
    summarize_benchmark,
    window_stats,
    yearly_returns,
)

import quant.factors.style  # noqa: E402,F401  （注册红利低波/容量因子）

CAPACITY_FACTORS = [
    # 规模（连续 + 三个分档）
    "mcap",
    "fmcap",
    "mcap_q05",
    "mcap_q10",
    "mcap_q20",
    "fmcap_q05",
    "fmcap_q10",
    "fmcap_q20",
    # 流动性
    "amihud",
    "amount_20",
    "turnover_20",
    "zero_days",
    "turnover_vol",
    "spread_proxy",
    # 低关注
    "low_price",
    "listing_days",
    # 交互
    "rev_smallcap",
    "maxret_smallcap",
]

STRESS = ("2024-01-01", "2024-02-29")
MONTHLY = "ME"


# --------------------------------------------------------------------------- 截面工具
def month_ends(panel) -> pd.DatetimeIndex:
    """样本内（2017-01-01 ~ 2024-12-31）每个月最后一个交易日。"""
    days = panel.dates[(panel.dates >= START) & (panel.dates <= END)]
    frame = pd.DataFrame({"day": days}, index=days)
    grouped = frame.groupby([frame.index.year, frame.index.month])["day"].last()
    return pd.DatetimeIndex(sorted(grouped.tolist()))


def compute_frames(panel, names, *, verbose: bool = True) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name in names:
        started = time.time()
        try:
            frames[name] = library.compute(name, panel)
        except Exception as error:  # noqa: BLE001 —— 单因子失败不应中断整表
            print(f"[skip] {name}: {type(error).__name__}: {error}", flush=True)
            continue
        if verbose:
            print(f"[factor] {name} {time.time() - started:.1f}s", flush=True)
    return frames


def monthly_ic(frame: pd.DataFrame, forward: pd.DataFrame, ends: pd.DatetimeIndex) -> pd.Series:
    ic = rank_ic(frame, forward, min_count=30)
    return ic.reindex(ends).dropna()


def ic_table(frames, forward, ends) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        series = monthly_ic(frame, forward, ends)
        if series.empty:
            rows.append({"factor": name, "months": 0})
            continue
        summary = ic_summary(series, periods_per_year=12)
        by_year = series.groupby(series.index.year).mean()
        rows.append({
            "factor": name,
            "months": int(summary["n"]),
            "ic_mean": summary["ic_mean"],
            "ic_std": summary["ic_std"],
            "ic_ir": summary["ic_ir"],
            "t_stat": summary["t_stat"],
            "positive_rate": summary["positive_rate"],
            "years": int(len(by_year)),
            "positive_years": int((by_year > 0).sum()),
            "worst_year": float(by_year.min()) if len(by_year) else np.nan,
            "best_year": float(by_year.max()) if len(by_year) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("ic_ir", ascending=False, na_position="last")


def yearly_ic_table(frames, forward, ends) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        series = monthly_ic(frame, forward, ends)
        if series.empty:
            continue
        by_year = series.groupby(series.index.year).mean()
        for year, value in by_year.items():
            rows.append({"factor": name, "year": int(year), "ic": float(value)})
    return pd.DataFrame(rows)


def quantile_table(frames, forward, ends) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        quantiles = quantile_returns(frame, forward, quantiles=5, min_count=30)
        if quantiles.empty:
            continue
        at_ends = quantiles.reindex(ends).dropna(how="all")
        means = at_ends.mean()
        if means.isna().all():
            continue
        labels = [f"Q{i}" for i in range(1, len(means) + 1)]
        mean_values = [float(means.get(label, np.nan)) for label in labels]
        # 单调性：Q1..Q5 均值与 1..5 的 Spearman 相关（因子已统一「越大越看多」，故应 >0）
        valid = [(i + 1, value) for i, value in enumerate(mean_values) if np.isfinite(value)]
        if len(valid) >= 3:
            left = pd.Series([item[0] for item in valid], dtype=float)
            right = pd.Series([item[1] for item in valid], dtype=float)
            # 环境里没有 scipy，用「秩 + pearson」等价实现 Spearman
            mono = float(left.rank().corr(right.rank()))
        else:
            mono = np.nan
        top_bottom = mean_values[-1] - mean_values[0] if np.isfinite(mean_values[-1]) and np.isfinite(mean_values[0]) else np.nan
        row = {"factor": name, "months": int(len(at_ends)), "monotonic": mono, "q5_minus_q1": top_bottom}
        row.update({label: value for label, value in zip(labels, mean_values)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("q5_minus_q1", ascending=False, na_position="last")


def correlation_matrix(frames, ends, *, min_count: int = 30) -> pd.DataFrame:
    """各月末横截面秩相关的月均值矩阵。"""
    names = list(frames)
    total = pd.DataFrame(0.0, index=names, columns=names)
    counts = pd.DataFrame(0, index=names, columns=names)
    for day in ends:
        cross = {}
        for name in names:
            frame = frames[name]
            if day not in frame.index:
                continue
            series = frame.loc[day]
            if series.notna().sum() >= min_count:
                cross[name] = series
        if len(cross) < 2:
            continue
        block = pd.DataFrame(cross)
        # 秩相关：环境里没有 scipy，用「先排名再 pearson」等价实现
        corr = block.rank().corr(min_periods=min_count)
        total = total.add(corr.fillna(0.0), fill_value=0.0)
        counts = counts.add(corr.notna().astype(int), fill_value=0)
    return (total / counts.replace(0, np.nan)).round(4)


def capacity_table(frames, panel, block, ends, *, ks=(30, 50, 100), participation: float = 0.10) -> pd.DataFrame:
    """容量：组合 AUM 受「最小持仓股的日均成交额 × 参与率」绑定。

    等权 k 只：``AUM_max = participation * min_i(ADV_i) / (1/k)``；
    单票上限 = ``participation * ADV``（元）。
    """
    adv = panel.amount.rolling(20, min_periods=10).mean()
    rows = []
    for name, frame in frames.items():
        for k in ks:
            totals, singles, names_count = [], [], []
            for day in ends:
                if day not in frame.index or day not in adv.index:
                    continue
                score = frame.loc[day]
                tradable = ~block.loc[day].reindex(score.index).fillna(True).astype(bool)
                score = score.where(tradable)
                top = score.dropna().nlargest(k)
                if top.empty:
                    continue
                liquidity = adv.loc[day].reindex(top.index).dropna()
                if liquidity.empty:
                    continue
                totals.append(participation * float(liquidity.min()) * len(top))
                singles.append(participation * float(liquidity.median()))
                names_count.append(int(len(top)))
            if not totals:
                continue
            rows.append({
                "factor": name,
                "top_k": k,
                "months": len(totals),
                "capacity_median_yi": float(np.median(totals)) / 1e8,
                "capacity_p10_yi": float(np.percentile(totals, 10)) / 1e8,
                "single_cap_median_wan": float(np.median(singles)) / 1e4,
                "avg_names": float(np.mean(names_count)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 回测
def run_cell(name, panel, block, bars, bt_bars, benchmark_equity, top_k, costs, cost_key, *, out, verbose=False):
    strategy = MultiFactorStrategy(
        factors=name,
        weighting="equal",
        top_k=top_k,
        rebalance=MONTHLY,
        gross_exposure=0.95,
        max_weight=0.10,
        min_price=2.0,
        min_history=120,
        # 分档因子（最小 5% 档）天然只有几十只候选，min_universe=100 会让整月空仓；
        # 这里放宽到 10，仍保证横截面不至于只有个位数。
        min_universe=10,
        min_cross_count=5,
        exclude=block,
    )
    engine = Engine(
        config=BacktestConfig(
            start=str(START.date()),
            end=str(END.date()),
            costs=costs,
            params={"top_k": top_k, "cost": cost_key, "factor": name},
        ),
        benchmark_prices=benchmark_equity,
        delisting=None,
        delisting_mode="haircut30",
    )
    result = engine.run(bt_bars, strategy.generate(bars), strategy_name=strategy.describe())
    picks = strategy.last_diagnostics.get("picks", {})
    keys = sorted(key for key in picks if pd.Timestamp(key) <= pd.Timestamp(STRESS[0]))
    holdings = list(picks[keys[-1]]) if keys else []
    metrics = summarize(result, benchmark_equity, bars, holdings)
    metrics.update({"factor": name, "top_k": top_k, "cost": cost_key, "cost_bp": round_trip_bp(costs)})
    if verbose:
        print(
            f"[{name} k={top_k} {cost_key}] 累计 {metrics['total_return']:.2%} "
            f"CAGR {metrics['cagr']:.2%} MDD {metrics['max_drawdown']:.2%} "
            f"超额 {metrics['excess_total']:.2%}",
            flush=True,
        )
    return metrics, result, holdings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="容量受限因子研究（任务二第一步）")
    parser.add_argument("--smoke", type=int, default=None, help="只取前 N 只（冒烟）")
    parser.add_argument("--factors", default="", help="逗号分隔；默认全部 18 个")
    parser.add_argument("--topk", default="50", help="回测持仓数，逗号分隔（容量表固定 30/50/100）")
    parser.add_argument("--costs", default="default,conservative_30,conservative_50")
    parser.add_argument("--corr-limit", type=int, default=0, help="相关性阶段只用前 N 个因子（0=全部）")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-corr", action="store_true")
    parser.add_argument("--out", default="results/capacity")
    args = parser.parse_args(argv)

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)

    names = [item.strip() for item in args.factors.split(",") if item.strip()] or CAPACITY_FACTORS
    topks = [int(item) for item in args.topk.split(",") if item.strip()]
    presets = cost_presets()
    cost_keys = [item.strip() for item in args.costs.split(",") if item.strip()]

    bars, load_stats = load_bars(smoke=args.smoke)
    bars = {symbol: frame for symbol, frame in bars.items() if symbol not in board_exclude(bars)}
    panel = build_panel(bars)
    panel = panel.slice(str(START.date()), str(END.date()))
    block = build_block(panel)
    bt_bars = {symbol: frame.loc[START:END] for symbol, frame in bars.items() if len(frame)}
    ends = month_ends(panel)
    forward = panel.forward_returns(20)
    print(
        f"panel {panel.close.shape} {panel.dates[0].date()} ~ {panel.dates[-1].date()} "
        f"月末 {len(ends)} 个（{ends[0].date()} ~ {ends[-1].date()}）",
        flush=True,
    )

    # 1) 月度 IC
    frames = compute_frames(panel, names)
    ic = ic_table(frames, forward, ends)
    ic.to_csv(out / "monthly-ic.csv", index=False, encoding="utf-8-sig")
    yearly_ic_table(frames, forward, ends).to_csv(out / "yearly-ic.csv", index=False, encoding="utf-8-sig")

    # 2) 分层
    quantiles = quantile_table(frames, forward, ends)
    quantiles.to_csv(out / "quantiles.csv", index=False, encoding="utf-8-sig")

    # 3) 相关性（与全部因子）
    if not args.skip_corr:
        library_names = [name for name in library.names() if name not in frames]
        try:
            fundamental.register()
        except Exception:  # noqa: BLE001
            pass
        extra = [name for name in library.names() if name not in frames and name not in library_names]
        if args.corr_limit:
            library_names = library_names[: args.corr_limit]
            extra = extra[: args.corr_limit]
        other = compute_frames(panel, library_names + extra, verbose=False)
        merged = {**frames, **other}
        corr = correlation_matrix(merged, ends)
        corr.to_csv(out / "corr.csv", encoding="utf-8-sig")
        # 同源判定：与 mcap/fmcap/vol_20/amount_20/turnover_20 的 |corr| >= 0.7
        peers = [name for name in ("mcap", "fmcap", "vol_20", "amount_20", "turnover_20") if name in corr.columns]
        source_rows = []
        for name in frames:
            if name in peers:
                continue
            hits = corr.loc[name, peers].drop(labels=[name], errors="ignore").dropna()
            hits = hits[hits.abs() >= 0.7]
            source_rows.append({
                "factor": name,
                "same_source": "、".join(f"{key}({value:+.2f})" for key, value in hits.items()) or "",
            })
        pd.DataFrame(source_rows).to_csv(out / "same-source.csv", index=False, encoding="utf-8-sig")

    # 4) 容量
    capacity_table(frames, panel, block, ends).to_csv(out / "capacity.csv", index=False, encoding="utf-8-sig")

    # 5/6) 回测 + 成本敏感性 + 压力
    if not args.skip_backtest:
        # 基准用解析式：全池等权在 1e6 本金下每票不足 1 手，走撮合会被全部拒单
        delisting_events = load_delisting_events(LAKE)
        bench_equity = pool_benchmark(panel, block, delisting_events, cash=1_000_000.0)
        bench_metrics = summarize_benchmark(bench_equity)
        (out / "benchmark.json").write_text(
            json.dumps({"metrics": bench_metrics, "cost_bp": round_trip_bp(presets["default"])},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        bench_daily = pd.DataFrame({"equity": bench_equity})

        rows, stress_rows = [], []
        for name in names:
            for top_k in topks:
                for cost_key in cost_keys:
                    slug = f"{name}-k{top_k}-{cost_key}"
                    cell_path = out / "cells" / f"{slug}.json"
                    if cell_path.exists():
                        payload = json.loads(cell_path.read_text(encoding="utf-8"))
                        rows.append(payload["metrics"])
                        continue
                    costs = presets.get(cost_key, presets["default"])
                    metrics, result, holdings = run_cell(
                        name, panel, block, bars, bt_bars, bench_equity, top_k, costs, cost_key,
                        out=out, verbose=True,
                    )
                    result.daily.to_csv(out / "cells" / f"{slug}-daily.csv", encoding="utf-8-sig")
                    cell_path.write_text(
                        json.dumps({"metrics": metrics, "holdings_at_stress": holdings},
                                   ensure_ascii=False, indent=2, default=float),
                        encoding="utf-8",
                    )
                    rows.append(metrics)
                    if cost_key == "default":
                        stress = window_stats(result.daily, STRESS[0], STRESS[1], bars, holdings)
                        stress.update({"factor": name, "top_k": top_k, "cost": cost_key})
                        stress_rows.append(stress)
        pd.DataFrame(rows).to_csv(out / "backtest-metrics.csv", index=False, encoding="utf-8-sig")
        bench_stress = window_stats(bench_daily, STRESS[0], STRESS[1], bars, [])
        bench_stress.update({"factor": "benchmark_equal_weight", "top_k": 0, "cost": "default"})
        stress_rows.insert(0, bench_stress)
        pd.DataFrame(stress_rows).to_csv(out / "stress.csv", index=False, encoding="utf-8-sig")

    meta = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "seconds": round(time.time() - started, 1),
        "load_stats": load_stats,
        "panel": {
            "dates": int(len(panel.dates)),
            "symbols": int(len(panel.symbols)),
            "first": str(panel.dates[0].date()),
            "last": str(panel.dates[-1].date()),
            "month_ends": int(len(ends)),
        },
        "block_share": round(float(block.to_numpy().mean()), 4),
        "factors": names,
        "topk": topks,
        "costs": {key: round(round_trip_bp(presets[key]), 2) for key in cost_keys if key in presets},
        "holdout_locked_from": str(HOLDOUT.date()),
        "stress_window": list(STRESS),
        "participation": 0.10,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"完成 {meta['seconds']}s → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
