"""阶段 0：把「最终规格」的复现口径对齐 final_spec.py，再在同一 base 上重跑危机 overlay。

与 ``scripts/final_spec.py``（基准，只读）逐项对齐：
- 因子 ``div_yield,vol_60,turnover_20,insider_net_20``、top_k 50、gross 0.95、max_weight 0.10
- ``min_price=2.0``、**``min_amount=0.0``**、``min_history=120``、``min_universe=100``
- 排除项 = ``build_block | (lhb_flag_5 < 0)``
- 基准 = **含龙虎榜排除的同池等权**（``benchmark_same_pool``）
- 退市 haircut30、往返 14.2bp（另跑 30/50bp）

硬自检不通过直接 ``raise SystemExit``，不进入下一阶段。

产物：``results/crisis/aligned-final-spec.csv``（零 overlay 三档成本）、
``results/crisis/aligned-crisis.csv``（危机 overlay 对照）、``results/crisis/aligned-meta.json``。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import quant.factors.events as event_factors
import quant.factors.fundamental  # noqa: F401
import quant.factors.style  # noqa: F401
from quant import regime
from quant.backtest.engine import Engine
from quant.config import BacktestConfig
from quant.factors.library import compute
from quant.factors.panel import build_panel
from quant.strategy.multifactor import MultiFactorStrategy
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
    summarize_benchmark,
)

OUT = Path("results/crisis")
FACTORS = "div_yield,vol_60,turnover_20,insider_net_20"
TARGET = {"total_return": 0.7953, "sharpe": 0.483, "max_drawdown": -0.2284}
TOL = {"total_return": 0.01, "sharpe": 0.02, "max_drawdown": 0.01}


def build_context():
    bars, stats = load_bars()
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    print(f"bars {len(bars)} 只（含退市 {stats['delisted_added']}）", flush=True)
    panel = build_panel(bars, fields=FIELDS)
    panel_bt = panel.slice(pd.Timestamp(START), pd.Timestamp(END))
    base_block = build_block(panel_bt)
    lhb = compute("lhb_flag_5", panel_bt)
    block = (
        (base_block | (lhb < 0))
        .reindex(index=panel_bt.dates, columns=panel_bt.symbols)
        .fillna(True)
        .astype(bool)
    )
    print(f"剔除占比 {block.to_numpy().mean():.2%}（含龙虎榜排除）", flush=True)
    delisting = load_delisting_events(LAKE)
    bench_same_pool = pool_benchmark(panel_bt, block, delisting, cash=1_000_000.0)
    bench_std = pool_benchmark(
        panel_bt,
        base_block.reindex(index=panel_bt.dates, columns=panel_bt.symbols).fillna(True).astype(bool),
        delisting,
        cash=1_000_000.0,
    )
    return bars, panel_bt, block, delisting, bench_same_pool, bench_std


def run(weights, bt_bars, block, delisting, bench, costs, overlays, name):
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
    cfg = BacktestConfig(start=START, end=END, initial_cash=1_000_000.0, costs=costs, benchmark=None)
    engine = Engine(
        config=cfg,
        overlays=overlays,
        benchmark_prices=(bench / 1_000_000.0).ffill(),
        delisting=delisting,
        delisting_mode="haircut30",
    )
    result = engine.run(bt_bars, weights, strategy_name=name)
    return result


def main() -> int:
    event_factors.register()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    bars, panel_bt, block, delisting, bench_same_pool, bench_std = build_context()
    bt_bars = {symbol: frame.loc[START:END] for symbol, frame in bars.items() if len(frame)}
    presets = cost_presets()

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
    weights = strategy.generate(bars)
    picks = strategy.last_diagnostics.get("picks", {})
    holdings = list(picks.values())[-1] if picks else []
    print(f"权重生成完成 {time.time()-started:.0f}s", flush=True)

    zero_rows = []
    zero_results = {}
    for cost_key in ("default", "conservative_30", "conservative_50"):
        result = run(weights, bt_bars, block, delisting, bench_same_pool, presets[cost_key], [], f"aligned-{cost_key}")
        row = {
            "config": f"final-spec-{cost_key}",
            "factors": FACTORS,
            "top_k": 50,
            "cost": cost_key,
            "round_trip_bp": round_trip_bp(presets[cost_key]),
            **summarize(result, bench_same_pool, bars, holdings),
        }
        zero_rows.append(row)
        zero_results[cost_key] = result
        print(
            f"[{row['config']}] 累计 {row['total_return']:.2%} 夏普 {row['sharpe']:.3f} "
            f"MDD {row['max_drawdown']:.2%} 超额 {row['excess_total']:.2%}",
            flush=True,
        )
    pd.DataFrame(zero_rows).to_csv(OUT / "aligned-final-spec.csv", index=False, encoding="utf-8-sig")

    first = zero_rows[0]
    checks = []
    for key, want in TARGET.items():
        got = float(first[key])
        ok = abs(got - want) <= TOL[key]
        checks.append({"metric": key, "want": want, "got": round(got, 4), "tol": TOL[key], "ok": ok})
        print(f"  自检 {key}: {got:.4f} vs {want} (tol {TOL[key]}) -> {'OK' if ok else 'FAIL'}", flush=True)
    if not all(item["ok"] for item in checks):
        print("❌ 阶段 0 自检失败，停止（不进入阶段 1）", flush=True)
        return 1
    print("✅ 阶段 0 自检通过：零 overlay 最终规格与 final_spec.py 一致", flush=True)

    # 在同一 base 上重跑危机 overlay
    frame = regime.market_index(panel_bt)
    flags = regime.score_components(frame)
    score = regime.crisis_score(flags)
    gate = flags["vol"] | flags["tail"]
    recovery = frame["index"] > frame["ma200"]
    variants = {
        "composite": regime.crisis_state(score, enter=0.75, exit=0.30, confirm=3, gate=gate, recovery=recovery),
        "strict": regime.crisis_state(score, enter=0.85, exit=0.30, confirm=3, gate=gate, recovery=recovery),
        "loose": regime.crisis_state(score, enter=0.60, exit=0.30, confirm=3, gate=gate, recovery=recovery),
        "trend-only": regime.crisis_state(flags["trend"].astype(float), enter=0.9, exit=0.4, confirm=3),
        "delay1": regime.crisis_state(score, enter=0.75, exit=0.30, confirm=3, gate=gate, recovery=recovery),
    }
    rows = [{"config": "final-spec (new base)", "crisis_share": 0.0, **{k: first[k] for k in first if k not in ("config", "factors", "top_k", "cost", "round_trip_bp")}}]
    for name, state in variants.items():
        delay = 1 if name == "delay1" else 0
        result = run(weights, bt_bars, block, delisting, bench_same_pool, presets["default"],
                     [regime.crisis_overlay(state, delay=delay)], name)
        row = {"config": name, "crisis_share": round(float(state.mean()), 4), **summarize(result, bench_same_pool, bars, holdings)}
        rows.append(row)
        print(f"[{name}] 累计 {row['total_return']:.2%} 夏普 {row['sharpe']:.3f} MDD {row['max_drawdown']:.2%} 空仓 {row['crisis_share']:.1%}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "aligned-crisis.csv", index=False, encoding="utf-8-sig")

    meta = {
        "purpose": "阶段 0：复现口径对齐 + 在正确 base 上重跑危机 overlay",
        "benchmark_same_pool": summarize_benchmark(bench_same_pool),
        "benchmark_std_pool": summarize_benchmark(bench_std),
        "self_check": checks,
        "block_share": round(float(block.to_numpy().mean()), 4),
        "zero_overlay": zero_rows,
        "crisis_variants": rows,
        "seconds": round(time.time() - started, 1),
    }
    (OUT / "aligned-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"完成 {time.time()-started:.0f}s → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
