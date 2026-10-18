"""阶段 2：holdout（2025-01-01 ~ 2026-10-16）一次性样本外验收。

锁定规格（与 ``scripts/final_spec_aligned.py`` 完全一致）：
- 因子 ``div_yield,vol_60,turnover_20,insider_net_20``（等权）、top_k=50、gross 0.95、max_weight 0.10
- ``min_price=2.0`` / ``min_amount=0.0`` / ``min_history=120`` / ``min_universe=100``
- 剔除：ST、次新（<365 天）、停牌、一字板、退市整理期、近 5 日龙虎榜（``lhb_flag_5 < 0``）
- 退市 haircut30；成本 14.2 / 30 / 50 bp

红线：只跑一次、只读；结果只用于判断策略是否有效，不得据此回头调参。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import quant.factors.events as event_factors
import quant.factors.fundamental  # noqa: F401
import quant.factors.style  # noqa: F401
from quant import dividends, events, valuation
from quant.backtest.engine import Engine
from quant.config import BacktestConfig
from quant.factors.evaluate import ic_summary, rank_ic
from quant.factors.library import compute
from quant.factors.panel import build_panel
from quant.refdata import load_universe
from quant.strategy.multifactor import MultiFactorStrategy
from scripts.factor_market import FIELDS, load_delisting_events, load_layer
from scripts.redlow_study import (
    LAKE,
    cost_presets,
    monthly_returns,
    pool_benchmark,
    round_trip_bp,
    summarize,
    summarize_benchmark,
)

OUT = Path("results/holdout")
START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-10-16")
WARM = pd.Timestamp("2024-03-01")
FACTORS = "div_yield,vol_60,turnover_20,insider_net_20"
HOLDOUT_PARQUET = Path("D:/quant-data/holdout/parquet")


def load_bars() -> tuple[dict[str, pd.DataFrame], dict]:
    frames, stats = load_layer(LAKE, "processed")
    hold, hold_stats = load_layer(LAKE, "holdout", format="parquet")
    merged: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        merged[symbol] = frame
    for symbol, frame in hold.items():
        if symbol in merged:
            both = pd.concat([merged[symbol], frame])
            both = both[~both.index.duplicated(keep="last")].sort_index()
            merged[symbol] = both
        else:
            merged[symbol] = frame
    return merged, {"sample": stats, "holdout": hold_stats, "merged": len(merged)}


def merged_st(panel) -> pd.DataFrame:
    base = valuation.st_matrix(list(panel.symbols))
    extra = valuation.st_matrix(list(panel.symbols), root=valuation.HOLDOUT_DIR)
    if base.empty:
        return extra.sort_index()
    if extra.empty:
        return base.sort_index()
    return base.combine_first(extra).sort_index()


def holdout_delisting(bars: dict[str, pd.DataFrame]) -> dict[str, pd.Timestamp]:
    universe = load_universe()
    out: dict[str, pd.Timestamp] = {}
    for symbol, out_date in zip(universe["symbol"].astype(str), pd.to_datetime(universe["out_date"], errors="coerce")):
        if pd.isna(out_date) or not (START <= out_date <= END):
            continue
        code = symbol.zfill(6)
        for key in bars:
            if str(key).split(".")[0] == code:
                out[key] = out_date
        out.setdefault(code, out_date)
    return out


def build_block(panel, st: pd.DataFrame, delisting: dict[str, pd.Timestamp]) -> pd.DataFrame:
    index, columns = panel.dates, panel.symbols
    block = pd.DataFrame(False, index=index, columns=columns)
    bare = [str(c).split(".")[0] for c in columns]
    st_mask = st.reindex(index=index, columns=bare).fillna(False).astype(bool).to_numpy()
    block |= st_mask
    universe = load_universe()
    ipo = dict(zip(universe["symbol"].astype(str), pd.to_datetime(universe["ipo_date"], errors="coerce")))
    for position, symbol in enumerate(columns):
        start = ipo.get(str(symbol).split(".")[0])
        if start is None or pd.isna(start):
            continue
        block.iloc[:, position] |= (index - start).days < 365
    block |= panel.volume.fillna(0.0).le(0)
    one_price = (panel.high - panel.low).abs().le(panel.close * 1e-4)
    limit_move = panel.close.div(panel.prev_close).sub(1.0).abs().ge(0.095)
    block |= one_price & limit_move
    for position, symbol in enumerate(columns):
        last = delisting.get(symbol) or delisting.get(str(symbol).split(".")[0])
        if last is None:
            continue
        block.iloc[:, position] |= (index >= last - pd.Timedelta(days=45)) & (index <= last)
    return block


def portfolio_equity(panel, block: pd.DataFrame, picks: dict[pd.Timestamp, list[str]]) -> pd.Series:
    """按「月末选出的等权名单」解析式合成净值（无交易成本，用于风格参照）。"""
    returns = panel.close.pct_change().fillna(0.0)
    values, equity = [], 1_000_000.0
    month_ends = sorted(picks)
    for i, day in enumerate(panel.dates):
        members = None
        for end in reversed(month_ends):
            if end <= day:
                members = picks[end]
                break
        if members:
            cols = [c for c in members if c in returns.columns]
            if cols:
                equity *= 1.0 + float(returns.loc[day, cols].mean())
        values.append(equity)
    return pd.Series(values, index=panel.dates)


def dividend_proxy(panel, block: pd.DataFrame, top: int = 50) -> dict[pd.Timestamp, list[str]]:
    """自建「红利参照组合」：每月末按 PIT 股息率取前 top 只（等权）。"""
    score = compute("div_yield", panel)
    eligible = ~block.reindex(index=panel.dates, columns=panel.symbols).fillna(True)
    picks: dict[pd.Timestamp, list[str]] = {}
    month_ends = pd.Series(panel.dates, index=panel.dates).groupby([panel.dates.year, panel.dates.month]).last()
    for day in month_ends:
        row = score.loc[day].where(eligible.loc[day]).dropna().sort_values(ascending=False)
        picks[pd.Timestamp(day)] = list(row.index[:top])
    return picks


def try_dividend_index(dates: pd.DatetimeIndex) -> pd.Series | None:
    """尝试取中证红利（000922）日收盘做对照；失败返回 None（不阻断验收）。"""
    try:
        import threading

        import akshare as ak

        box: dict = {}

        def runner():
            try:
                box["df"] = ak.index_zh_a_hist(symbol="000922", period="daily", start_date="20250101", end_date="20261016")
            except BaseException as exc:  # noqa: BLE001
                box["err"] = str(exc)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(40)
        if thread.is_alive() or "df" not in box:
            return None
        frame = box["df"]
        frame["日期"] = pd.to_datetime(frame["日期"])
        series = frame.set_index("日期")["收盘"].astype(float)
        return series.reindex(dates).ffill()
    except BaseException:  # noqa: BLE001
        return None


def factor_ic(panel, block: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    forward = panel.forward_returns(20)
    month_ends = pd.Series(panel.dates, index=panel.dates).groupby([panel.dates.year, panel.dates.month]).last()
    for name in factors:
        values = compute(name, panel)
        values = values.where(~block.reindex(index=panel.dates, columns=panel.symbols).fillna(True))
        ic = rank_ic(values, forward, min_count=30)
        ic_month = ic.reindex([pd.Timestamp(d) for d in month_ends])
        summary = ic_summary(ic_month.dropna(), periods_per_year=12)
        rows.append({"factor": name, "months": int(ic_month.notna().sum()), **summary})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="holdout 一次性验收")
    parser.add_argument("--smoke", type=int, default=None, help="只取前 N 只（管线冒烟，不作结论）")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    guard = Path("results/event-research/final-spec.json")
    guard_before = guard.read_bytes() if guard.is_file() else b""
    started = time.time()
    event_factors.register()

    bars_all, load_stats = load_bars()
    if args.smoke:
        keep = sorted(bars_all)[: args.smoke]
        bars_all = {symbol: bars_all[symbol] for symbol in keep}
    bars = {
        symbol: frame.loc[(frame.index >= WARM) & (frame.index <= END)]
        for symbol, frame in bars_all.items()
        if len(frame) and frame.index.max() >= START
    }
    panels = {s: f for s, f in bars.items()}
    print(f"bars {len(bars)} 只（样本内 {load_stats['sample'].get('symbols', '?')} + holdout {load_stats['holdout'].get('symbols', '?')}）", flush=True)

    panel = build_panel(panels, fields=FIELDS)
    panel_bt = panel.slice(START, END)
    st = merged_st(panel_bt)
    delisting = holdout_delisting(bars)
    block = build_block(panel_bt, st, delisting)
    lhb = compute("lhb_flag_5", panel_bt)
    block = (block | (lhb < 0)).reindex(index=panel_bt.dates, columns=panel_bt.symbols).fillna(True).astype(bool)
    holdout_delisted = len([k for k in delisting if "." not in str(k)])
    cells = int(panel_bt.close.notna().to_numpy().sum())
    print(f"面板 {panel_bt.dates[0].date()}~{panel_bt.dates[-1].date()} {cells:,} 格；剔除 {block.to_numpy().mean():.2%}；holdout 退市 {holdout_delisted} 只", flush=True)

    bench = pool_benchmark(panel_bt, block, delisting, cash=1_000_000.0)
    div_picks = dividend_proxy(panel_bt, block, top=50)
    div_equity = portfolio_equity(panel_bt, block, div_picks)
    index_div = try_dividend_index(panel_bt.dates)

    bt_bars = {s: f.loc[(f.index >= START) & (f.index <= END)] for s, f in bars.items() if len(f)}
    strategy = MultiFactorStrategy(
        factors=FACTORS,
        weighting="equal",
        top_k=50,
        rebalance="ME",
        gross_exposure=0.95,
        max_weight=0.10,
        min_price=2.0,
        min_amount=0.0,
        min_history=120,
        min_universe=100,
        exclude=block,
    )
    weights_all = strategy.generate(panels)
    weights = weights_all.loc[weights_all.index >= START]
    picks = strategy.last_diagnostics.get("picks", {})
    holdings = list(picks.values())[-1] if picks else []
    print(f"权重 {len(weights)} 行；持仓样本 {len(holdings)}", flush=True)

    presets = cost_presets()
    rows, results = [], {}
    for key in ("default", "conservative_30", "conservative_50"):
        cfg = BacktestConfig(start=START, end=END, initial_cash=1_000_000.0, costs=presets[key], benchmark=None)
        engine = Engine(
            config=cfg,
            benchmark_prices=(bench / 1_000_000.0).ffill(),
            delisting=delisting,
            delisting_mode="haircut30",
        )
        result = engine.run(bt_bars, weights, strategy_name=f"final-spec-{key}")
        results[key] = result
        row = {
            "config": key,
            "round_trip_bp": round_trip_bp(presets[key]),
            **summarize(result, bench, bars, holdings),
        }
        bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1.0)
        row["benchmark_total"] = bench_total
        row["excess_total"] = float(row["total_return"] - bench_total)
        rows.append(row)
        print(f"[{key}] 累计 {row['total_return']:.2%} CAGR {row['cagr']:.2%} 夏普 {row['sharpe']:.3f} MDD {row['max_drawdown']:.2%} 基准 {bench_total:.2%}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "holdout-metrics.csv", index=False, encoding="utf-8-sig")
    main_result = results["default"]
    main_result.equity.to_csv(out_dir / "holdout-equity.csv", encoding="utf-8-sig")
    if len(main_result.trades):
        main_result.trades.to_csv(out_dir / "holdout-trades.csv", index=False, encoding="utf-8-sig")
    daily = main_result.daily
    monthly = monthly_returns(daily)
    monthly.to_frame("return").to_csv(out_dir / "holdout-monthly.csv", encoding="utf-8-sig")

    excess = pd.DataFrame(
        {
            "strategy": daily["equity"].pct_change(),
            "pool": bench.pct_change(),
            "dividend": div_equity.pct_change(),
            "dividend_index": (index_div.pct_change() if index_div is not None else np.nan),
        }
    ).dropna(how="all")
    excess["excess_vs_pool"] = excess["strategy"] - excess["pool"]
    excess["excess_vs_dividend"] = excess["strategy"] - excess["dividend"]
    excess.to_csv(out_dir / "holdout-excess.csv", encoding="utf-8-sig")

    ic = factor_ic(panel_bt, block, ["div_yield", "vol_60", "turnover_20", "insider_net_20"])
    ic.to_csv(out_dir / "holdout-factor-ic.csv", index=False, encoding="utf-8-sig")

    # 归因：超额 = α + β_market·市场 + β_style·红利风格
    frame = excess.dropna(subset=["strategy", "pool", "dividend"])
    y = frame["excess_vs_pool"].to_numpy()
    X = np.column_stack([np.ones(len(frame)), frame["pool"].to_numpy(), (frame["dividend"] - frame["pool"]).to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ann = 244
    attribution = pd.DataFrame(
        [
            {"item": "alpha_annual", "value": float(coef[0] * ann)},
            {"item": "beta_market", "value": float(coef[1])},
            {"item": "beta_dividend_style", "value": float(coef[2])},
            {"item": "excess_total", "value": float(metrics.loc[metrics.config == "default", "excess_total"].iloc[0])},
            {"item": "strategy_total", "value": float(metrics.loc[metrics.config == "default", "total_return"].iloc[0])},
            {"item": "benchmark_total", "value": float(metrics.loc[metrics.config == "default", "benchmark_total"].iloc[0])},
            {"item": "dividend_proxy_total", "value": float(div_equity.iloc[-1] / div_equity.iloc[0] - 1.0)},
            {"item": "dividend_index_total", "value": float(index_div.iloc[-1] / index_div.iloc[0] - 1.0) if index_div is not None else float("nan")},
            {"item": "days", "value": len(frame)},
        ]
    )
    attribution.to_csv(out_dir / "holdout-attribution.csv", index=False, encoding="utf-8-sig")

    guard_after = guard.read_bytes() if guard.is_file() else b""
    meta = {
        "start": str(panel_bt.dates[0].date()),
        "end": str(panel_bt.dates[-1].date()),
        "bars": len(bars),
        "cells": cells,
        "block_share": round(float(block.to_numpy().mean()), 4),
        "delisted_in_window": holdout_delisted,
        "factors": FACTORS,
        "top_k": 50,
        "metrics": rows,
        "benchmark_same_pool": summarize_benchmark(bench),
        "dividend_proxy": {"total": float(div_equity.iloc[-1] / div_equity.iloc[0] - 1.0), "months": len(div_picks)},
        "dividend_index": "000922 中证红利（akshare index_zh_a_hist）" if index_div is not None else "不可用（已回退到自建红利参照）",
        "monthly_win_rate": round(float((monthly > 0).mean()), 4),
        "worst_month": round(float(monthly.min()), 4),
        "best_month": round(float(monthly.max()), 4),
        "ic": ic.to_dict("records"),
        "attribution": attribution.to_dict("records"),
        "guard_unchanged": bool(guard_before == guard_after),
        "seconds": round(time.time() - started, 1),
        "note": "holdout 一次性验收；不据此调参",
    }
    (out_dir / "holdout-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    assert panel_bt.dates[0] >= START, "回测窗口早于 holdout 起点"
    assert guard_before == guard_after, "样本内 final-spec.json 被改动"
    print(f"完成 {time.time() - started:.0f}s → {out_dir}", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(attribution.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
