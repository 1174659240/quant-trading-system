"""危机清仓 overlay 的回测对比（样本内 2017-2024，holdout 锁死）。

对照组：
- ``base``：红利低波三因子 50 只、月频、等权，不加任何择时；
- ``composite``：六分量危机分数 >= 0.75 且（波动或尾部）报警 → 次日清仓；
  指数重新站上 200 日均线 3 天 → 回场（Faber 式回暖）；
- ``strict`` / ``loose``：入场阈值敏感性（0.85 / 0.60）；
- ``trend-only``：只用「指数跌破 200 日均线」单分量（Faber 原版，验证复合是否有增量）；
- ``delay1``：信号再延后一天执行（执行时点敏感性）。

产物：``results/crisis/{metrics.csv,yearly.csv,stress.csv,episodes.csv,score.csv,meta.json}``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import quant.factors.fundamental  # noqa: F401
import quant.factors.style  # noqa: F401
from quant import regime
from quant.backtest.engine import Engine
from quant.config import BacktestConfig
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
    monthly_returns,
    pool_benchmark,
    round_trip_bp,
    summarize,
    summarize_benchmark,
    window_stats,
    yearly_returns,
)

FACTORS = "div_yield,vol_60,turnover_20"


def build_state(panel, *, enter: float, exit_: float, confirm: int, single: str | None = None, use_gate: bool = True):
    frame = regime.market_index(panel)
    flags = regime.score_components(frame)
    score = flags[single].astype(float) if single else regime.crisis_score(flags)
    gate = None
    recovery = None
    if not single:
        if use_gate:
            gate = flags["vol"] | flags["tail"]
        recovery = frame["index"] > frame["ma200"]
    else:
        recovery = frame["index"] > frame["ma200"]
    state = regime.crisis_state(score, enter=enter, exit=exit_, confirm=confirm, gate=gate, recovery=recovery)
    return frame, flags, score, state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="危机清仓 overlay 对比")
    parser.add_argument("--out", default="results/crisis")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    bars, load_stats = load_bars()
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    panel = build_panel(bars, fields=FIELDS)
    panel_bt = panel.slice(pd.Timestamp(START), pd.Timestamp(END))
    block = build_block(panel_bt).reindex(index=panel_bt.dates, columns=panel_bt.symbols).fillna(True).astype(bool)
    print(f"panel {panel_bt.close.shape} {panel_bt.dates[0].date()} ~ {panel_bt.dates[-1].date()}", flush=True)

    frame, flags, score, state = build_state(panel_bt, enter=0.75, exit_=0.30, confirm=3)
    diag = frame.copy()
    for name in flags.columns:
        diag[f"flag_{name}"] = flags[name]
    diag["score"] = score
    diag["crisis"] = state
    diag.to_csv(out / "score.csv", encoding="utf-8-sig")
    eps = regime.episodes(state, score, frame)
    pd.DataFrame(
        [
            {
                "start": e.start.date(),
                "end": e.end.date() if e.end is not None else None,
                "days": e.days,
                "max_score": round(e.max_score, 3),
                "peak_drawdown": round(e.peak_dd, 4),
            }
            for e in eps
        ]
    ).to_csv(out / "episodes.csv", index=False, encoding="utf-8-sig")
    print(f"危机日 {int(state.sum())} / {len(state)}（{state.mean():.1%}），事件 {len(eps)} 段", flush=True)

    delisting = load_delisting_events(LAKE)
    bench_equity = pool_benchmark(panel_bt, block, delisting, cash=args.cash)
    bench_row = {"config": "benchmark_equal_weight", **summarize_benchmark(bench_equity)}
    presets = cost_presets()
    bt_bars = {symbol: fr.loc[START:END] for symbol, fr in bars.items() if len(fr)}

    variants = [
        ("base", None),
        ("composite", "main"),
        ("strict", (0.85, 0.30, 3, None)),
        ("loose", (0.60, 0.25, 3, None)),
        ("trend-only", (1.0, 0.30, 3, "trend")),
        ("delay1", "delay"),
    ]

    rows, yearly_rows, stress_rows, meta_variants = [bench_row], [], [], {}
    for name, spec in variants:
        overlays = []
        if spec is not None:
            if spec == "main":
                st = state
            elif spec == "delay":
                st = state
            else:
                enter, exit_, confirm, single = spec
                _, _, _, st = build_state(
                    panel_bt, enter=enter, exit_=exit_, confirm=confirm, single=single, use_gate=single is None
                )
            overlays = [regime.crisis_overlay(st, delay=1 if name == "delay1" else 0)]
            meta_variants[name] = {
                "crisis_days": int(st.sum()),
                "crisis_share": round(float(st.mean()), 4),
                "spec": spec if isinstance(spec, str) else {"enter": spec[0], "exit": spec[1], "confirm": spec[2], "single": spec[3]},
            }
        strategy = MultiFactorStrategy(
            factors=FACTORS,
            weighting="equal",
            top_k=args.topk,
            rebalance=MONTHLY,
            gross_exposure=0.95,
            max_weight=0.10,
            min_price=2.0,
            min_amount=2e7,
            min_history=120,
            min_universe=100,
            exclude=block,
        )
        cfg = BacktestConfig(start=START, end=END, initial_cash=args.cash, costs=presets["default"], benchmark=None)
        engine = Engine(
            config=cfg,
            overlays=overlays,
            benchmark_prices=(bench_equity / args.cash).ffill(),
            delisting=delisting,
            delisting_mode="haircut30",
        )
        started = time.time()
        weights = strategy.generate(bars)
        result = engine.run(bt_bars, weights, strategy_name=strategy.describe())
        picks = strategy.last_diagnostics.get("picks", {})
        holdings = list(picks.values())[-1] if picks else []
        row = {
            "config": name,
            "overlay": bool(overlays),
            "round_trip_bp": round_trip_bp(presets["default"]),
            **summarize(result, bench_equity, bars, holdings),
        }
        rows.append(row)
        yearly = yearly_returns(result.daily)
        yearly_rows.append({"config": name, **{str(k): float(v) for k, v in yearly.items()}})
        for start, end, tag in (("2024-01-01", "2024-02-29", "stress"), (START, END, "full")):
            st_row = window_stats(result.daily, start, end, bars, holdings)
            st_row.update({"config": name, "window": tag})
            stress_rows.append(st_row)
        if name == "composite":
            result.equity.to_csv(out / "equity-composite.csv", encoding="utf-8-sig")
            if len(result.trades):
                result.trades.to_csv(out / "trades-composite.csv", index=False, encoding="utf-8-sig")
        print(
            f"[{name}] 累计 {row['total_return']:.2%} CAGR {row['cagr']:.2%} 夏普 {row['sharpe']:.3f} "
            f"MDD {row['max_drawdown']:.2%} Calmar {row['calmar']:.3f} 换手 {row['turnover']:.2%} "
            f"费用 {row['fees']:,.0f}（{time.time()-started:.0f}s）",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly_rows).to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stress_rows).to_csv(out / "stress.csv", index=False, encoding="utf-8-sig")
    eq = pd.read_csv(out / "equity-composite.csv", index_col=0, parse_dates=True).iloc[:, 0]
    monthly_returns(pd.DataFrame({"equity": eq})).rename("monthly_return").to_csv(
        out / "monthly-composite.csv", encoding="utf-8-sig"
    )
    meta = {
        "start": START,
        "end": END,
        "holdout": "锁定 2025-01-01 起，未读",
        "top_k": args.topk,
        "factors": FACTORS,
        "weights": regime.DEFAULT_WEIGHTS,
        "variants": meta_variants,
        "crisis_days": int(state.sum()),
        "episodes": len(eps),
        "seconds": round(time.time() - t0, 1),
        "benchmark": bench_row,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(f"完成 {time.time()-t0:.0f}s → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
