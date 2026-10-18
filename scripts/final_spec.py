"""样本内最终规格回测：base 三因子 + 增持加分项 + 龙虎榜排除项。

只读 2017-01-03 ~ 2024-12-31，2025 起锁死。产出 results/event-research/final-spec.{json,csv}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import quant.factors.events as event_factors
import quant.factors.fundamental  # noqa: F401
import quant.factors.style  # noqa: F401
from quant.backtest.engine import Engine
from quant.config import BacktestConfig
from quant.factors.library import compute
from quant.factors.panel import build_panel
from quant.strategy.multifactor import MultiFactorStrategy
from scripts.factor_market import FIELDS, load_delisting_events
from scripts.redlow_study import (
    LAKE,
    START,
    END,
    MONTHLY,
    board_exclude,
    build_block,
    cost_presets,
    load_bars,
    monthly_returns,
    pool_benchmark,
    round_trip_bp,
    summarize,
    summarize_benchmark,
    window_stats,
)

OUT = Path("results/event-research")
FACTORS = "div_yield,vol_60,turnover_20,insider_net_20"


def main() -> None:
    event_factors.register()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    bars, load_stats = load_bars()
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    print(f"bars {len(bars)} 只（含退市 {load_stats['delisted_added']}）", flush=True)

    panel = build_panel(bars, fields=FIELDS)
    panel_bt = panel.slice(pd.Timestamp(START), pd.Timestamp(END))
    block = build_block(panel_bt)
    lhb = compute("lhb_flag_5", panel_bt)
    block = (block | (lhb < 0)).reindex(index=panel_bt.dates, columns=panel_bt.symbols).fillna(True).astype(bool)
    print(f"剔除占比 {block.to_numpy().mean():.2%}（含龙虎榜排除）", flush=True)

    delisting = load_delisting_events(LAKE)
    bench_equity = pool_benchmark(panel_bt, block, delisting, cash=1_000_000.0)
    bench_base = pool_benchmark(panel_bt, build_block(panel_bt).reindex(index=panel_bt.dates, columns=panel_bt.symbols).fillna(True).astype(bool), delisting, cash=1_000_000.0)
    bench_row = {"config": "benchmark_equal_weight", **summarize_benchmark(bench_base)}
    bench_row_same = {"config": "benchmark_same_pool", **summarize_benchmark(bench_equity)}

    bt_bars = {symbol: frame.loc[START:END] for symbol, frame in bars.items() if len(frame)}
    presets = cost_presets()
    rows, stress_rows = [], []
    for cost_key in ("default", "conservative_30", "conservative_50"):
        strategy = MultiFactorStrategy(
            factors=FACTORS,
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
        cfg = BacktestConfig(start=START, end=END, initial_cash=1_000_000.0, costs=presets[cost_key], benchmark=None)
        engine = Engine(
            config=cfg,
            benchmark_prices=(bench_equity / 1_000_000.0).ffill(),
            delisting=delisting,
            delisting_mode="haircut30",
        )
        started = time.time()
        weights = strategy.generate(bars)
        result = engine.run(bt_bars, weights, strategy_name=strategy.describe())
        picks = strategy.last_diagnostics.get("picks", {})
        holdings = list(picks.values())[-1] if picks else []
        row = {
            "config": "final-spec" if cost_key == "default" else f"final-spec-{cost_key}",
            "factors": FACTORS,
            "top_k": 50,
            "cost": cost_key,
            "round_trip_bp": round_trip_bp(presets[cost_key]),
            **summarize(result, bench_equity, bars, holdings),
        }
        rows.append(row)
        if cost_key == "default":
            alt = {
                "config": "final-spec-vs-stdpool",
                "factors": FACTORS,
                "top_k": 50,
                "cost": "default",
                "round_trip_bp": round_trip_bp(presets[cost_key]),
                **summarize(result, bench_base, bars, holdings),
            }
            rows.append(alt)
        print(
            f"[{row['config']}] 累计 {row['total_return']:.2%} CAGR {row['cagr']:.2%} 夏普 {row['sharpe']:.3f} "
            f"MDD {row['max_drawdown']:.2%} 超额 {row['excess_total']:.2%} 换手 {row['turnover']:.2%}（{time.time()-started:.0f}s）",
            flush=True,
        )
        if cost_key == "default":
            daily = result.daily
            stress = window_stats(daily, "2024-01-01", "2024-02-29", bars, holdings)
            stress["config"] = "final-spec"
            stress_rows.append(stress)
            stress_rows.append({**window_stats(daily, "2017-01-01", "2024-12-31", bars, holdings), "config": "full-sample"})
            monthly = monthly_returns(daily)
            result.equity.to_csv(out / "final-spec-equity.csv", encoding="utf-8-sig")
            if len(result.trades):
                result.trades.to_csv(out / "final-spec-trades.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "final-spec.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stress_rows).to_csv(out / "final-spec-stress.csv", index=False, encoding="utf-8-sig")
    meta = {
        "factors": FACTORS,
        "exclusion": "近 5 日上过龙虎榜（lhb_flag_5 < 0）",
        "top_k": 50,
        "rebalance": "ME",
        "delisting_mode": "haircut30",
        "start": str(panel_bt.dates[0].date()),
        "end": str(panel_bt.dates[-1].date()),
        "bt_start": START,
        "bt_end": END,
        "block_share": round(float(block.to_numpy().mean()), 4),
        "holdout": "锁定 2025-01-01 起，未读",
        "benchmark": bench_row,
        "benchmark_same_pool": bench_row_same,
        "metrics": rows,
        "stress": stress_rows,
        "months": int(len(monthly)),
        "monthly_win_rate": round(float((monthly > 0).mean()), 4),
        "worst_month": round(float(monthly.min()), 4),
        "best_month": round(float(monthly.max()), 4),
    }
    (out / "final-spec.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(stress_rows).to_string(index=False))
    print("benchmark:", bench_row["total_return"], bench_row["cagr"])


if __name__ == "__main__":
    main()
