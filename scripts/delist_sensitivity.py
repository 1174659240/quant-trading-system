"""退市结算三档敏感性：同一策略在 last / -30% / -100% 下的净值差多少。

为什么要单独跑：`factor-market --include-delisted` 一次只跑一档，而报告必须给出区间——
「按最后价卖」是乐观假设（老三板还有流动性），「全损」是最保守，两者之间的差距就是
退市处置假设对结论的影响幅度。

    python -m quant delist-sensitivity

只做回测不做 IC 研究，所以比 factor-market 快得多（读取 parquet 层 6 秒，面板 20 秒）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.backtest.engine import Engine, DELISTING_FACTORS  # noqa: E402
from quant.config import BacktestConfig  # noqa: E402
from quant.factors import build_panel  # noqa: E402
from quant.report import to_markdown_table  # noqa: E402
from quant.strategy.multifactor import MultiFactorStrategy  # noqa: E402
from scripts.factor_market import (  # noqa: E402
    FIELDS,
    LAKE,
    load_delisted_bars,
    load_delisting_events,
    load_layer,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m quant delist-sensitivity", description="退市结算三档敏感性")
    parser.add_argument("--lake", default=str(LAKE))
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--factors", default="amihud,bias,close_position,downside_vol,gap,high_ratio,low_vol,max_ret,"
                                             "momentum,price_volume_corr,reversal,rsi,skew,trend_strength,turnover_abn")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--rebalance", default="M")
    parser.add_argument("--cash", type=float, default=10_000_000.0)
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    frames, stats = load_layer(Path(args.lake), "processed", None, format="auto")
    print(f"在市 {len(frames)} 只（{stats['seconds_total']}s）", flush=True)
    dead = load_delisted_bars(Path(args.lake))
    lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end)
    added = 0
    for symbol, frame in dead.items():
        window = frame.loc[(frame.index >= lo) & (frame.index <= hi)]
        if len(window) >= 2 and symbol not in frames:
            frames[symbol] = window
            added += 1
    for symbol in list(frames):
        window = frames[symbol].loc[(frames[symbol].index >= lo) & (frames[symbol].index <= hi)]
        if len(window) >= 2:
            frames[symbol] = window
        else:
            frames.pop(symbol)
    print(f"补进退市股 {added} 只，票池 {len(frames)} 只，窗口 {args.start} ~ {args.end}", flush=True)

    panel = build_panel(frames, fields=FIELDS)
    print(f"面板 {len(panel.dates)} 天 × {len(panel.symbols)} 只", flush=True)
    delisting = load_delisting_events(Path(args.lake))
    close = pd.DataFrame({symbol: frame["close"] for symbol, frame in frames.items()})
    base = close.apply(lambda series: series.dropna().iloc[0] if series.notna().any() else float("nan"))
    benchmark = close.div(base.where(base > 0), axis=1).where(close > 0).mean(axis=1, skipna=True)

    rows = []
    for mode in DELISTING_FACTORS:
        strategy = MultiFactorStrategy(
            factors=args.factors,
            weighting="ic",
            ic_lookback=250,
            ic_min_periods=120,
            weight_refresh=3,
            top_k=args.top_k,
            rebalance=args.rebalance,
            gross_exposure=0.95,
            max_weight=min(0.10, 2.0 / max(args.top_k, 2)),
            winsorize_method="mad",
            winsorize_n=5.0,
            standardize=True,
            min_price=2.0,
            min_amount=2e7,
            min_history=120,
            min_universe=100,
        )
        config = BacktestConfig(
            start=args.start, end=args.end, initial_cash=args.cash, benchmark=None, min_trade_value=5_000.0
        )
        engine = Engine(config=config, benchmark_prices=benchmark, delisting=delisting, delisting_mode=mode)
        result = engine.run_strategy(frames, strategy)
        metrics = result.metrics
        settled = result.delistings if result.delistings is not None else pd.DataFrame()
        rows.append(
            {
                "mode": mode,
                "factor": DELISTING_FACTORS[mode],
                "total_return%": round(float(metrics.get("total_return", float("nan"))) * 100, 2),
                "cagr%": round(float(metrics.get("cagr", float("nan"))) * 100, 2),
                "sharpe": round(float(metrics.get("sharpe", float("nan"))), 3),
                "max_drawdown%": round(float(metrics.get("max_drawdown", float("nan"))) * 100, 2),
                "alpha%": round(float(metrics.get("alpha", float("nan"))) * 100, 2),
                "trades": int(metrics.get("trade_count", 0) or 0),
                "settled": len(settled),
                "settled_pnl": round(float(settled["realized_pnl"].sum()) if len(settled) else 0.0, 0),
            }
        )
        print(f"  {mode}: 累计 {rows[-1]['total_return%']:+.2f}%，结算 {len(settled)} 笔", flush=True)

    table = pd.DataFrame(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "delist-sensitivity.csv", index=False, encoding="utf-8-sig")
    (out_dir / "delist-sensitivity.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 退市结算三档敏感性",
        "",
        f"- 票池：{len(frames)} 只（含 {added} 只退市股），窗口 {args.start} ~ {args.end}，"
        f"{len(panel.dates)} 个交易日",
        f"- 策略：多因子 IC 加权，top_k={args.top_k}，{args.rebalance} 调仓",
        "- 三档假设：`last`=按最后价变现 / `haircut30`=折价 30% / `total_loss`=全损",
        "",
        to_markdown_table(table, floatfmt=".2f", index=False),
    ]
    (out_dir / "delist-sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + table.to_string(index=False))
    print(f"\n报告已写入 {out_dir / 'delist-sensitivity.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
