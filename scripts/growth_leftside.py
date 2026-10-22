"""C 规格：梯度减仓 + 左侧分批回场（带趋势闸门）。

与 ``scripts/growth_risk.py`` 的区别只在**入场**：危机期内不等危机分回落到 exit 阈值，
而是在「分数从峰值回落 + 不再创新低 + 波动率退潮 + 已进入下跌下半场」时开始分 3 批买回，
并用趋势闸门（近期新低 / 流动性恶化）暂停加仓。

阈值全部先登记在 ``docs/小盘成长策略-左侧回场预注册.md``，本模块的常量必须与之一致。

用法：
    python -m scripts.growth_leftside --runs A,B,C,C15,C25
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtest.engine import Engine
from quant.config import BacktestConfig, CostModel
from quant.factors.panel import build_panel
from scripts import growth_risk as gr
from quant.factors import growth_alt
from scripts.factor_market import FIELDS, load_delisting_events
from scripts.growth_backtest import SPECS, composite_score, month_ends, sector_panel, turnover_stats
from scripts.redlow_study import END, LAKE, START, board_exclude, build_block, load_bars, pool_benchmark
from quant.factors import growth

OUT = Path("results/growth/leftside")
HIST = Path("D:/quant-data/growth-history")
HIST_START = pd.Timestamp("2008-01-01")
HIST_END = pd.Timestamp("2016-12-30")
HIST_CUTOFF = pd.Timestamp("2016-12-31")
HIST_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "prev_close",
                "adj_factor", "suspended", "tradable"]

# 风控开关：off = 主规格（无风控，A）；leftside = 可选叠加（梯度减仓 + 左侧分批回场，C）
RISK_MODES: dict[str, tuple[str, ...]] = {
    "off": ("A",),
    "leftside": ("A", "B", "C", "C15", "C25"),
}


def load_history_bars() -> dict[str, pd.DataFrame]:
    """读 2007–2016 补采层（``scripts/growth_history_fetch_em.py`` 产物）。"""
    frames: dict[str, list[pd.DataFrame]] = {}
    for path in sorted((HIST / "parquet").glob("year=*/data.parquet")):
        part = pd.read_parquet(path)
        if part.empty:
            continue
        part["date"] = pd.to_datetime(part["date"])
        for symbol, chunk in part.groupby("symbol", sort=False):
            chunk = chunk.sort_values("date").set_index("date")
            chunk = chunk[[c for c in HIST_COLUMNS if c in chunk.columns]]
            frames.setdefault(str(symbol), []).append(chunk)
    out: dict[str, pd.DataFrame] = {}
    for symbol, chunks in frames.items():
        frame = pd.concat(chunks).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume", "amount", "prev_close"])
        if len(frame) >= 60:
            out[symbol] = frame
    return out

# ---- 预注册阈值（docs/小盘成长策略-左侧回场预注册.md） ---------------------- #
SCORE_DROP = 0.20        # C1：危机分从本段滚动峰值回落
QUIET_DAYS = 15          # C2：最近 N 日不再创新低
VOL_CAP = 0.97           # C3：波动率分位已退潮
VOL_DROP = 0.10          # C3：相对本段峰值回落幅度
STEP_DAYS = 20           # 两次加仓的最小间隔（交易日）
TRANCHES = 3             # 最多分 3 批加到满仓
GATE_LOW_DAYS = 5        # G1：最近 N 日出现新低即暂停
GATE_ILLIQ_PCT = 0.95    # G2：非流动性分位过高
GATE_ILLIQ_LOOKBACK = 20  # G2：非流动性仍在恶化（对比 20 日前）
DD_CAP = -0.20           # C4 主规格
DD_SENSITIVITY = (-0.15, -0.25)


def market_frame(panel) -> pd.DataFrame:
    """指数、回撤、波动率分位、非流动性分位 —— 全部因果（只用当日及过去）。"""
    frame = gr.regime.market_index(panel)
    index = frame["index"]
    frame = frame.copy()
    frame["peak"] = index.cummax()
    frame["dd"] = index / frame["peak"] - 1.0
    frame["dd500"] = index / index.rolling(500, min_periods=120).max() - 1.0
    rets = index.pct_change()
    frame["vol20"] = rets.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(244)
    frame["vol_pct"] = frame["vol20"].rolling(500, min_periods=120).rank(pct=True)
    illiq = (panel.close.pct_change().abs() / panel.amount.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    frame["illiq"] = illiq.mean(axis=1)
    frame["illiq_20"] = frame["illiq"].rolling(20, min_periods=20).mean()
    frame["illiq_pct"] = frame["illiq_20"].rolling(250, min_periods=60).rank(pct=True)
    return frame


def leftside_exposure(
    frame: pd.DataFrame,
    score: pd.Series,
    state: pd.Series,
    *,
    dd_cap: float = DD_CAP,
) -> tuple[pd.Series, pd.DataFrame]:
    """逐日的目标仓位路径 + 事件明细（分批、闸门暂停、危机进出）。"""
    index = frame.index
    idx = frame["index"]
    dd = frame["dd500"] if frame["dd500"].notna().any() else frame["dd"]
    vol_pct = frame["vol_pct"]
    illiq_pct = frame["illiq_pct"]
    illiq = frame["illiq_20"]

    new_low = (idx <= idx.rolling(QUIET_DAYS, min_periods=QUIET_DAYS).min()).fillna(False)
    quiet = (new_low.rolling(QUIET_DAYS, min_periods=1).sum().fillna(0) == 0).to_numpy()
    recent_low = (new_low.rolling(GATE_LOW_DAYS, min_periods=1).sum().fillna(0) > 0).to_numpy()
    liq_bad = ((illiq_pct > GATE_ILLIQ_PCT) & (illiq > illiq.shift(GATE_ILLIQ_LOOKBACK))).fillna(False).to_numpy()
    vol_arr = vol_pct.to_numpy(dtype=float)
    dd_arr = dd.to_numpy(dtype=float)
    score_arr = score.reindex(index).fillna(0.0).to_numpy(dtype=float)
    state_arr = state.reindex(index).fillna(False).to_numpy(dtype=bool)
    idx_arr = idx.to_numpy(dtype=float)

    exposure = np.ones(len(index))
    current = 1.0
    in_episode = False
    episode = 0
    peak_score = 0.0
    peak_vol = 0.0
    tranches = 0
    last_step = -10**9
    ramp_start: int | None = None
    ramp_base = 1.0
    pause_logged = False
    events: list[dict] = []

    for position in range(len(index)):
        day = index[position]
        if state_arr[position]:
            if not in_episode:
                in_episode = True
                episode += 1
                current = gr.CRISIS_EXPOSURE
                peak_score = score_arr[position]
                peak_vol = vol_arr[position] if np.isfinite(vol_arr[position]) else 0.0
                tranches = 0
                last_step = -10**9
                ramp_start = None
                pause_logged = False
                events.append({"date": day, "kind": "episode_start", "episode": episode,
                               "exposure_before": 1.0, "exposure_after": current,
                               "score": score_arr[position], "dd": dd_arr[position],
                               "index": idx_arr[position], "note": "进入危机，仓位降至 0.5"})
            else:
                peak_score = max(peak_score, score_arr[position])
                if np.isfinite(vol_arr[position]):
                    peak_vol = max(peak_vol, vol_arr[position])

            vol_now = vol_arr[position]
            c1 = score_arr[position] <= peak_score - SCORE_DROP
            c2 = bool(quiet[position])
            c3 = bool(np.isfinite(vol_now) and vol_now < VOL_CAP
                      and (peak_vol < 0.90 or vol_now <= peak_vol - VOL_DROP))
            c4 = bool(np.isfinite(dd_arr[position]) and dd_arr[position] <= dd_cap)
            gate = bool(recent_low[position] or liq_bad[position])
            spacing = (position - last_step) >= STEP_DAYS
            if c1 and c2 and c3 and c4:
                if gate or not spacing or tranches >= TRANCHES:
                    if not pause_logged:
                        pause_logged = True
                        events.append({"date": day, "kind": "gate_pause", "episode": episode,
                                       "exposure_before": current, "exposure_after": current,
                                       "score": score_arr[position], "dd": dd_arr[position],
                                       "index": idx_arr[position],
                                       "note": ("新低未消" if recent_low[position] else "")
                                               + ("/流动性恶化" if liq_bad[position] else "")
                                               + ("" if (gate or tranches < TRANCHES) else "/批次用尽")})
                else:
                    before = current
                    current = before + (1.0 - before) / TRANCHES
                    tranches += 1
                    last_step = position
                    pause_logged = False
                    events.append({"date": day, "kind": "tranche", "episode": episode,
                                   "exposure_before": round(before, 4), "exposure_after": round(current, 4),
                                   "score": score_arr[position], "dd": dd_arr[position],
                                   "index": idx_arr[position],
                                   "note": f"第 {tranches}/{TRANCHES} 批（分数回落 {peak_score - score_arr[position]:.2f}）"})
            else:
                pause_logged = False
        else:
            if in_episode:
                in_episode = False
                ramp_start = position
                ramp_base = current
                events.append({"date": day, "kind": "episode_end", "episode": episode,
                               "exposure_before": current, "exposure_after": current,
                               "score": score_arr[position], "dd": dd_arr[position],
                               "index": idx_arr[position], "note": "危机解除，开始回补"})
            soft = gr.CRISIS_SOFT <= score_arr[position] < gr.CRISIS_ENTER
            if ramp_start is not None and current < 1.0 - 1e-9:
                step = min(position - ramp_start + 1, gr.CRISIS_RAMP)
                current = ramp_base + (1.0 - ramp_base) * step / gr.CRISIS_RAMP
                if current >= 1.0 - 1e-9:
                    current = 1.0
                    ramp_start = None
            else:
                # 与 B（gradient）口径一致：非危机、非回补期按**当日**分数取固定档位。
                # 旧实现用 min(current, 0.75) 会让 0.75 这个软档"粘住"不再回 1.0
                # （实测 2020-04~2023-12 一直被压在 0.75，等于白扣 25% 仓位）。
                ramp_start = None
                current = gr.CRISIS_SOFT_EXPOSURE if soft else 1.0
        exposure[position] = current

    return pd.Series(exposure, index=index), pd.DataFrame(events)


def leftside_weights(weights: pd.DataFrame, exposure: pd.Series, marks: pd.DatetimeIndex) -> pd.DataFrame:
    """把逐日仓位路径**只在月末调仓日**落到权重矩阵上（月内不动，与主策略同频）。"""
    sampled = exposure.reindex(marks).ffill()
    levels = pd.Series(index=weights.index, dtype=float)
    levels.loc[sampled.index] = sampled.to_numpy()
    levels = levels.ffill().fillna(1.0)
    return weights.mul(levels, axis=0).fillna(0.0)


def history_delistings(panel, end: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """历史层没有退市事件表：以「最后一个有行情的交易日」当退市日（离窗口末日太近的不算）。"""
    events: dict[str, pd.Timestamp] = {}
    horizon = end - pd.Timedelta(days=20)
    for symbol in panel.symbols:
        last = panel.close[symbol].dropna().index
        if len(last) == 0:
            continue
        stamp = pd.Timestamp(last[-1])
        if stamp < horizon:
            events[str(symbol)] = stamp
    return events


def run(args: argparse.Namespace) -> int:
    sample = getattr(args, "sample", "main")
    out = Path(args.out) if args.out is not None else (
        Path("results/growth/leftside-history") if sample == "history" else OUT)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if sample == "history":
        bars = load_history_bars()
        start, end, cutoff = HIST_START, HIST_END, HIST_CUTOFF
        alt_kwargs = {"cutoff": HIST_CUTOFF, "strict": False}
        fund_extra = (HIST / "fundamentals",)
        sector_cap = 1.0
    else:
        bars, _ = load_bars()
        start, end, cutoff = START, END, pd.Timestamp("2024-12-31")
        alt_kwargs = {}
        fund_extra = ()
        sector_cap = gr.SECTOR_CAP
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    panel = build_panel(bars, fields=FIELDS)
    block = build_block(panel).reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)
    print(f"[panel] {sample} {panel.close.shape} {panel.dates[0].date()} ~ {panel.dates[-1].date()}", flush=True)

    facts = growth.GrowthFactors(panel, cutoff=cutoff, extra_dirs=fund_extra)
    eligible = growth.eligible_mask(facts, block_mask=block)
    marks = month_ends(panel.dates, start, end)
    alt = growth_alt.alt_factors(panel, **alt_kwargs)
    sector = None if sample == "history" else sector_panel(marks, list(panel.symbols))
    score = composite_score(panel, facts, alt, SPECS["QN"]["blocks"], eligible, marks)

    adv20, illiq20 = gr.liquidity_panels(panel)
    slip_maps, tier_rows = gr.tier_map(marks, eligible, adv20, illiq20)
    tier_rows.to_csv(out / "liquidity-tiers-by-month.csv", index=False, encoding="utf-8-sig")

    frame, crisis_score, crisis_state, _ = gr.crisis_inputs(panel)
    frame = market_frame(panel)
    frame["score"] = crisis_score.reindex(frame.index)
    frame["state"] = crisis_state.reindex(frame.index).fillna(False)
    frame.to_csv(out / "market-frame.csv", encoding="utf-8-sig")
    episodes = gr.episode_rows(crisis_state, crisis_score, frame)
    episodes.to_csv(out / "crisis-episodes.csv", index=False, encoding="utf-8-sig")
    print(f"[crisis] {int(crisis_state.sum())} 天 / {len(episodes)} 段", flush=True)

    delisting = (history_delistings(panel, end) if sample == "history"
                 else load_delisting_events(LAKE))
    bench_equity = pool_benchmark(panel, block, delisting, cash=args.cash)
    bench_equity.to_csv(out / "benchmark-equity.csv", encoding="utf-8-sig")
    bt_bars = {symbol: data.loc[start:end] for symbol, data in bars.items() if len(data)}
    gr.install_broker()

    base_weights = gr.BufferTopK(score, eligible, sector=sector, sector_cap=sector_cap,
                                 buffer_rank=gr.BUFFER_RANK).generate(bars)
    print(f"[weights] 缓冲换手 {turnover_stats(base_weights, marks)['ann_turnover_one_way']:.1%}", flush=True)

    gradient = gr.gradient_exposure(crisis_score, crisis_state)

    variants = {"C": DD_CAP, "C15": DD_SENSITIVITY[0], "C25": DD_SENSITIVITY[1]}
    exposure_map: dict[str, pd.Series] = {}
    for name, cap in variants.items():
        path, events = leftside_exposure(frame, crisis_score, crisis_state, dd_cap=cap)
        exposure_map[name] = path
        events.to_csv(out / f"events-{name}.csv", index=False, encoding="utf-8-sig")
        print(f"[{name}] 加仓 {int((events['kind'] == 'tranche').sum())} 次，"
              f"闸门暂停 {int((events['kind'] == 'gate_pause').sum())} 次，"
              f"平均仓位 {path.mean():.3f}", flush=True)

    tier_slip: dict[str, float] = {}
    for day in marks:
        tier_slip.update(slip_maps.get(day, {}))

    config = dict(
        start=str(start.date()),
        end=str(end.date()),
        initial_cash=args.cash,
        benchmark=None,
        costs=CostModel(slippage_bps=gr.BASE_BPS),
    )

    def execute(name: str, weights: pd.DataFrame, *, allow: set | None, min_trade: float) -> dict:
        gr.RiskBroker.ALLOW = allow
        gr.RiskBroker.MIN_TRADE = min_trade
        gr.RiskBroker.SLIP = tier_slip
        gr.RiskBroker.BASE_BPS = gr.BASE_BPS
        print(f"[{name}] 回测中…", flush=True)
        t0 = time.time()
        result = Engine(config=BacktestConfig(**config), benchmark_prices=None, delisting=delisting,
                        delisting_mode="haircut30").run(bt_bars, weights, strategy_name=f"left-{name}")
        row = {"spec": name, **gr.evaluate(result, bench_equity, cash=args.cash, slip_label="分档")}
        row["elapsed"] = round(time.time() - t0, 1)
        result.equity.to_csv(out / f"equity-{name}.csv", encoding="utf-8-sig")
        result.daily.to_csv(out / f"daily-{name}.csv", encoding="utf-8-sig")
        if len(result.trades):
            result.trades.to_csv(out / f"trades-{name}.csv", index=False, encoding="utf-8-sig")
        dd = gr.drawdown_series(result.equity)
        row["mdd_date"] = str(dd.idxmin().date())
        print(f"  → 累计 {row['total_return']:.2%} 年化 {row['cagr']:.2%} 夏普 {row['sharpe']:.3f} "
              f"MDD {row['max_drawdown']:.2%} 换手 {row['ann_turnover_one_way']:.0%} "
              f"暴露 {row['avg_exposure']:.3f}（{row['elapsed']}s）", flush=True)
        return row

    weights_map = {
        "A": base_weights,
        "B": gr.exposure_to_weights(base_weights, gradient),
        "C": leftside_weights(base_weights, exposure_map["C"], marks),
        "C15": leftside_weights(base_weights, exposure_map["C15"], marks),
        "C25": leftside_weights(base_weights, exposure_map["C25"], marks),
    }

    if args.risk == "off":
        if args.runs.replace(" ", "").upper() != "A":
            print("[risk=off] 忽略 --runs，仅跑无风控基线 A", flush=True)
        wanted = list(RISK_MODES["off"])
    else:
        wanted = [item.strip() for item in args.runs.split(",") if item.strip()]
    print(f"[risk={args.risk}] 规格 {'/'.join(wanted)}", flush=True)
    rows: list[dict] = []
    for name in wanted:
        rows.append(execute(name, weights_map[name], allow=None, min_trade=5_000.0))

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "risk_mode": args.risk,
        "sample": {"name": sample, "start": str(start.date()), "end": str(end.date()),
                   "holdout": "2025-01-01 起未读"},
        "prereg": {
            "score_drop": SCORE_DROP, "quiet_days": QUIET_DAYS, "vol_cap": VOL_CAP, "vol_drop": VOL_DROP,
            "step_days": STEP_DAYS, "tranches": TRANCHES, "gate_low_days": GATE_LOW_DAYS,
            "gate_illiq_pct": GATE_ILLIQ_PCT, "dd_cap": DD_CAP, "dd_sensitivity": list(DD_SENSITIVITY),
            "crisis": {"enter": gr.CRISIS_ENTER, "exit": gr.CRISIS_EXIT, "confirm": gr.CRISIS_CONFIRM,
                       "exposure": gr.CRISIS_EXPOSURE, "soft": gr.CRISIS_SOFT},
        },
        "rows": table.to_dict(orient="records"),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out / "leftside.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
    print(f"[done] {out}（{payload['elapsed_seconds']}s）", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--runs", default="A,B,C,C15,C25")
    parser.add_argument("--risk", choices=sorted(RISK_MODES), default="off",
                        help="风控开关：off=无风控基线（默认，主规格）；leftside=梯度减仓+左侧分批回场")
    parser.add_argument("--sample", choices=["main", "history"], default="main")
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
