"""稳健改进三项消融：vol target 18% / 流动性尾部剔除 / 删弱因子。

预注册：`docs/稳健改进三项-预注册.md`（运行前已冻结）。
判据：MDD 改善 ≥3.00pp 且夏普不降。
产物：`results/growth/ablation-2026-10-22/`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import growth_risk as gr
from scripts import growth_robust as grb
from scripts.growth_backtest import SPECS, composite_score

OUT = Path("results/growth/ablation-2026-10-22")
SECTOR_CAP = 0.25
BUFFER = 60
MIN_TRADE = 5000.0
TOP_K = 50
GROSS = 0.95
MAX_WEIGHT = 0.04

# 预注册阈值
VOL_TARGET = 0.18
VOL_WINDOW = 20
LIQ_TAIL = 0.05          # ADV20 截面最低 5%
LOCK_LOOKBACK = 250      # 回看 250 个交易日
LOCK_MIN = 2             # ≥2 次一字跌停
DROP_TARGETS = {
    "drop_g5": ["g5_stability"],
    "drop_e3": ["e3_rd_delta"],
    "drop_e2": ["e2_wc_gap"],
    "drop_e6": ["e6_nonrec"],
    "drop_all4": ["g5_stability", "e3_rd_delta", "e2_wc_gap", "e6_nonrec"],
}


# --------------------------------------------------------------------------- #
# 打分配置（删因子 = 从块里移除，块内重新等权）
# --------------------------------------------------------------------------- #
def blocks_without(dropped: list[str]) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for block, names in SPECS["QN"]["blocks"].items():
        kept = tuple(n for n in names if n not in dropped)
        if kept:
            out[block] = kept
    return out


def score_with(prepared: dict, dropped: list[str]) -> pd.DataFrame:
    if not dropped:
        return prepared["score"]
    blocks = blocks_without(dropped)
    return composite_score(prepared["panel"], prepared["facts"], prepared["alt"],
                           blocks, prepared["eligible"], prepared["marks"])


# --------------------------------------------------------------------------- #
# 流动性尾部剔除
# --------------------------------------------------------------------------- #
def _limit_down_threshold(dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """按板块与制度变化给出各日各股票的跌停幅度阈值（负值）。"""
    board_cut = pd.Timestamp("2020-08-24")
    star_cut = pd.Timestamp("2019-07-22")
    thresholds = pd.DataFrame(-0.095, index=dates, columns=symbols, dtype="float64")
    for symbol in symbols:
        code = str(symbol).split(".")[0]
        if code.startswith("688"):
            thresholds.loc[dates >= star_cut, symbol] = -0.19
        elif code.startswith("300") or code.startswith("301"):
            thresholds.loc[dates >= board_cut, symbol] = -0.19
        elif code.startswith(("8", "4")):
            thresholds.loc[:, symbol] = -0.29
    return thresholds


def lock_down_count(panel: Any) -> pd.DataFrame:
    """过去 250 个交易日内「一字跌停」的次数。"""
    high, low, close, prev = panel.high, panel.low, panel.close, panel.prev_close
    flat = (high - low).abs() <= (close.abs() * 1e-4)
    change = close / prev - 1.0
    thresholds = _limit_down_threshold(panel.dates, list(panel.symbols))
    one_word = flat & (change <= thresholds.to_numpy())
    return one_word.rolling(LOCK_LOOKBACK, min_periods=1).sum()


def liquidity_exclusions(prepared: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (剔除掩码, 诊断表)。掩码 True=当日剔除。"""
    panel = prepared["panel"]
    marks = prepared["marks"]
    eligible = prepared["eligible"]
    adv20 = panel.amount.rolling(20, min_periods=10).mean().reindex(marks)
    pct = adv20.where(eligible.reindex(marks)).rank(axis=1, pct=True)
    low_adv = pct <= LIQ_TAIL
    locks = lock_down_count(panel).reindex(marks)
    locked = locks >= LOCK_MIN
    mask = (low_adv | locked).reindex(index=marks, columns=eligible.columns).fillna(False)
    diag = pd.DataFrame({
        "date": marks,
        "eligible": eligible.reindex(marks).sum(axis=1).to_numpy(),
        "excl_adv": low_adv.sum(axis=1).to_numpy(),
        "excl_lock": locked.sum(axis=1).to_numpy(),
        "excl_total": (mask & eligible.reindex(marks)).sum(axis=1).to_numpy(),
    })
    return mask.astype(bool), diag


def counterfactual(prepared: dict, mask: pd.DataFrame) -> dict:
    """被剔除组 vs 保留组的下月等权收益（判断是否误杀 alpha）。"""
    close = prepared["panel"].close
    marks = prepared["marks"]
    rows = []
    for i, day in enumerate(marks[:-1]):
        nxt = marks[i + 1]
        fwd = (close.loc[nxt] / close.loc[day] - 1.0).dropna()
        elig = prepared["eligible"].loc[day]
        pool = [s for s in fwd.index if bool(elig.get(s, False))]
        if len(pool) < 20:
            continue
        excl = [s for s in pool if bool(mask.loc[day].get(s, False))]
        keep = [s for s in pool if not bool(mask.loc[day].get(s, False))]
        rows.append({
            "date": day,
            "excluded": float(fwd.reindex(excl).mean()) if excl else np.nan,
            "kept": float(fwd.reindex(keep).mean()) if keep else np.nan,
            "pool": float(fwd.reindex(pool).mean()),
            "n_excluded": len(excl),
        })
    frame = pd.DataFrame(rows)
    return {
        "months": int(len(frame)),
        "excluded_mean": float(frame["excluded"].mean()),
        "kept_mean": float(frame["kept"].mean()),
        "pool_mean": float(frame["pool"].mean()),
        "excluded_minus_kept_pp": float((frame["excluded"].mean() - frame["kept"].mean()) * 100),
        "excluded_win_rate": float((frame["excluded"] > frame["kept"]).mean()),
        "n_excluded_mean": float(frame["n_excluded"].mean()),
        "detail": frame.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------- #
# vol target
# --------------------------------------------------------------------------- #
def vol_scale(base_equity: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """scale_t = clip(0.18 / σ_{t-1}, 0, 1)，σ 用基线的 20 日实现波动（年化）。"""
    returns = base_equity.pct_change()
    sigma = (returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(244)).shift(1)
    scale = (VOL_TARGET / sigma).replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=1.0)
    return scale.reindex(dates).fillna(1.0)


# --------------------------------------------------------------------------- #
# 运行
# --------------------------------------------------------------------------- #
def build_weights(prepared: dict, score: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    strategy = gr.BufferTopK(score, eligible, sector=prepared["sector"], sector_cap=SECTOR_CAP,
                             buffer_rank=BUFFER, top_k=TOP_K, gross=GROSS, max_weight=MAX_WEIGHT)
    return strategy.generate(prepared["panel"])


def run_one(prepared: dict, name: str, *, weights: pd.DataFrame, slip: dict,
            daily_path: Path) -> dict:
    outcome = grb.run_engine(prepared, weights, name=name, min_trade=MIN_TRADE, slip_map=slip)
    daily = outcome["daily"]
    daily.rename_axis("date").to_csv(daily_path, encoding="utf-8-sig")
    return outcome["row"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="稳健改进三项消融")
    parser.add_argument("--only", default="", help="逗号分隔：baseline,vol18,liqex,drop_g5,...")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    started = time.time()
    prepared = grb.prepare("main")
    panel = prepared["panel"]
    marks = prepared["marks"]
    print(f"[prepare] {time.time() - started:.0f}s", flush=True)

    adv20, illiq20 = gr.liquidity_panels(panel)
    slip_maps, tier_rows = gr.tier_map(marks, prepared["eligible"], adv20, illiq20)
    tier_slip: dict[str, float] = {}
    for day in marks:
        tier_slip.update(slip_maps.get(day, {}))
    tier_rows.to_csv(OUT / "liquidity-tiers.csv", index=False, encoding="utf-8-sig")

    liq_mask, liq_diag = liquidity_exclusions(prepared)
    liq_diag.to_csv(OUT / "liquidity-exclusions.csv", index=False, encoding="utf-8-sig")
    cf = counterfactual(prepared, liq_mask)
    (OUT / "liquidity-counterfactual.json").write_text(
        json.dumps({k: v for k, v in cf.items() if k != "detail"}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    pd.DataFrame(cf["detail"]).to_csv(OUT / "liquidity-counterfactual.csv", index=False, encoding="utf-8-sig")

    eligible_liq = prepared["eligible"].copy()
    for day in marks:
        drop = liq_mask.loc[day]
        drop = drop[drop.astype(bool)].index
        eligible_liq.loc[day, drop] = False
    print(f"[liq] 平均剔除 {liq_diag['excl_total'].mean():.1f} 只/月"
          f"（ADV {liq_diag['excl_adv'].mean():.1f} + 跌停 {liq_diag['excl_lock'].mean():.1f}）", flush=True)

    scores = {"baseline": prepared["score"]}
    for name, dropped in DROP_TARGETS.items():
        scores[name] = score_with(prepared, dropped)

    # 先跑基线，拿到净值用于 vol target
    rows: list[dict] = []
    base_weights = build_weights(prepared, scores["baseline"], prepared["eligible"])
    base_row = run_one(prepared, "baseline", weights=base_weights, slip=tier_slip,
                       daily_path=OUT / "daily-baseline.csv")
    base_row["kind"] = "baseline"
    rows.append(base_row)
    base_daily = pd.read_csv(OUT / "daily-baseline.csv", index_col=0, parse_dates=True)
    base_equity = base_daily["equity"]

    scale = vol_scale(base_equity, panel.dates)
    scale.to_csv(OUT / "vol-scale.csv", encoding="utf-8-sig")
    binding = int(((scale < 0.999) & (scale.index >= pd.Timestamp("2017-01-01"))).sum())
    print(f"[vol] 触发 {binding} 天，均值 {scale.loc['2017':].mean():.3f}", flush=True)

    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    plan = ["vol18", "liqex"] + list(DROP_TARGETS)
    for name in plan:
        if wanted and name not in wanted:
            continue
        step = time.time()
        if name == "vol18":
            weights = base_weights.mul(scale.reindex(base_weights.index).fillna(1.0), axis=0)
        elif name == "liqex":
            weights = build_weights(prepared, scores["baseline"], eligible_liq)
        else:
            weights = build_weights(prepared, scores[name], prepared["eligible"])
        row = run_one(prepared, name, weights=weights, slip=tier_slip,
                      daily_path=OUT / f"daily-{name}.csv")
        row["kind"] = "variant"
        rows.append(row)
        print(f"[{name}] cagr {row['cagr']:.2%} sharpe {row['sharpe']:.3f} "
              f"mdd {row['max_drawdown']:.2%} turnover {row['ann_turnover_one_way']:.1%} "
              f"({time.time() - step:.0f}s)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")

    base = table[table["name"] == "baseline"].iloc[0]
    verdicts = []
    for _, row in table[table["name"] != "baseline"].iterrows():
        mdd_gain = abs(float(base["max_drawdown"])) - abs(float(row["max_drawdown"]))
        verdicts.append({
            "name": row["name"],
            "cagr_delta_pp": float((row["cagr"] - base["cagr"]) * 100),
            "sharpe_delta": float(row["sharpe"] - base["sharpe"]),
            "mdd_gain_pp": float(mdd_gain * 100),
            "turnover_delta_pp": float((row["ann_turnover_one_way"] - base["ann_turnover_one_way"]) * 100),
            "pass": bool(mdd_gain * 100 >= 3.0 and row["sharpe"] >= base["sharpe"] - 1e-9),
        })
    Path(OUT / "verdicts.json").write_text(
        json.dumps({"baseline": {k: float(base[k]) for k in
                                 ("cagr", "sharpe", "max_drawdown", "calmar", "ann_turnover_one_way")},
                    "verdicts": verdicts}, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = ["name", "cagr", "sharpe", "max_drawdown", "calmar", "ann_turnover_one_way",
            "fees_pct_of_equity", "avg_exposure"]
    print(table[cols].to_string(index=False), flush=True)
    print(f"[done] {time.time() - started:.0f}s → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
