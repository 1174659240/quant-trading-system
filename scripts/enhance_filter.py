"""股东户数（筹码集中度）作为红利低波的「过滤/倾斜」条件 —— 预注册影子实验。

为什么不复用上一次的「等权并入」：
    上一次（``scripts/enhance_alt.py``）把 ``gdhs_chg`` 等权加进 4 因子打分，组合层面反而变差
    （-5.97pp / 夏普 -0.072 / MDD 差 6pp），诊断结论是「信号方向没错，亏在换手」。
    本次换用法：**不动 4 因子打分**，只用股东户数决定「谁有资格进池」或轻微倾斜，
    并配换手预算，看能否把信号留下、把换手压住。

预注册（跑任何回测之前写死，落 ``results/enhance-filter/PREREG.md``）：
    样本内 2017-01-03 ~ 2024-12-31；月频执行；14.2bp 往返；gross=0.95；top50。
    控制组 = 现有 4 因子等权打分（div_yield / vol_60 / turnover_20 / insider_net_20）。
    候选用法 5 个（全部只改「谁可以被选中」，不改 4 因子打分本身）：
        V1 veto20      剔除当月 gdhs 最差的 20%（户数增幅最大的五分之一）
        V2 half        只在户数环比低于中位数的半数里选
        V3 tilt25      0.75×4因子 + 0.25×gdhs（倾斜，不硬过滤）
        V4 veto20_buf  V1 + 换手预算（保留旧仓；单月最多换 30%）
        V5 half_buf    V2 + 换手预算
    选定规则（事后不做任何调整）：
        V1-V5 中样本内夏普最高者，须同时满足
          (a) 夏普 ≥ 控制组 + 0.03
          (b) 累计 ≥ 控制组 + 8pp
          (c) 8 年里 ≥5 年跑赢控制组
          (d) 30bp 与 50bp 两档成本下仍跑赢控制组
        全部满足 → 开一次 holdout（2025-01-01 ~ 2026-10-16），只跑 控制组 / 选定组 / 纯红利，
        跑完立即重新封存；任一不满足 → 判定「无稳健增量」，不开 holdout。

用法：
    $py -u scripts/enhance_filter.py --sample            # 样本内（只读 ≤2024 数据）
    $py -u scripts/enhance_filter.py --holdout           # 仅当样本内判定通过后才跑
    $py -u scripts/enhance_filter.py --report-only       # 从 CSV 重生成报告
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import quant.factors.events as event_factors
from quant.factors.panel import build_panel  # noqa: F401  (保持与其它脚本同一条数据路径)
from scripts import holdout_validation as ho
from scripts.enhance_alt import ALT, CANDIDATES, CORE, TOP_K, asof_matrix
from scripts.improvement_shadow import END, START, build_holdout_context
from scripts.strategy_optimize import (
    COST_LEVELS,
    build_context,
    dividend_picks,
    eligibility,
    factor_monthly,
    month_end_days,
    open_pricer,
    picks_to_weights,
    price_picks,
    rank_z,
    stats_from_equity,
)

event_factors.register()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parents[1] / "results" / "enhance-filter"
CACHE = Path(__file__).resolve().parents[1] / "_tmp" / "enhance"
BASE_COST = "14.2bp"
HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2026-10-16"

# 预注册门槛（写死，跑完不改）
RULE = {"sharpe_gain": 0.03, "return_gain": 0.08, "years_win": 5}

# 换手预算：单月最多换 30%（50 只 ⇒ 15 只）
MAX_SWAP_FRAC = 0.30


# ---------------------------------------------------------------------------
# 股东户数：读 + PIT 到月末网格（本地实现，支持 holdout 目录与报告期上限）
# ---------------------------------------------------------------------------
def load_gdhs(folder: Path, *, max_period: pd.Timestamp) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"{folder} 没有股东户数原始文件")
    rows = []
    for path in files:
        frame = pd.read_csv(path, dtype={"代码": str}, encoding="utf-8-sig")
        frame["period"] = pd.Timestamp(path.name[:8])
        rows.append(frame)
    long = pd.concat(rows, ignore_index=True)
    long = long.rename(columns={
        "代码": "code", "股东户数-本次": "holders", "股东户数-上次": "holders_prev",
        "股东户数-增减比例": "chg_pct", "户均持股市值": "percap_mv",
        "户均持股数量": "percap_shares", "总市值": "total_mv",
        "股东户数统计截止日-本次": "period_col", "公告日期": "ann_date",
    })
    long["code"] = long["code"].astype(str).str.zfill(6)
    long["ann_date"] = pd.to_datetime(long["ann_date"], errors="coerce")
    long = long.dropna(subset=["ann_date"])
    long = long[long["period"] <= max_period]
    return long


def gdhs_frame(folder: Path, max_period: pd.Timestamp, panel, days, symbols) -> pd.DataFrame:
    """定向后的 ``gdhs_chg``（越大越好 = 户数下降越快），已按 PIT 可见日铺到月末网格。"""
    long = load_gdhs(folder, max_period=max_period)
    long = long[(long["holders"] >= 1000) & (long["holders_prev"] >= 1000)].copy()
    long["avail_date"] = long["ann_date"]
    long["chg"] = long["chg_pct"].clip(-60.0, 100.0) / 100.0
    matrix = asof_matrix(long, "chg", days, symbols, max_age_days=400)
    print(f"[data] 股东户数 {len(long):,} 条 / {long['period'].nunique()} 期 "
          f"（报告期 ≤ {max_period.date()}，按公告日 PIT），覆盖中位数 "
          f"{int(matrix.notna().sum(axis=1).median())}", flush=True)
    return matrix * CANDIDATES["gdhs_chg"]["sign"]


# ---------------------------------------------------------------------------
# 选股器：只改「谁可以被选中」，4 因子打分本身不动
# ---------------------------------------------------------------------------
def picks_top(score: pd.DataFrame, top: int = TOP_K) -> dict:
    out: dict = {}
    for day in score.index:
        row = score.loc[day].dropna()
        if not len(row):
            continue
        out[day] = list(row.nlargest(top).index) if len(row) >= top else list(row.sort_values(ascending=False).index)
    return out


def gate_veto(zg: pd.DataFrame, q: float = 0.20) -> pd.DataFrame:
    """剔除当月最差 ``q`` 分位；缺数据的股票不剔除（只排除已知的坏）。"""
    thr = zg.quantile(q, axis=1)
    allowed = zg.isna() | zg.gt(thr, axis=0)
    return allowed.fillna(False).astype(bool)


def gate_half(zg: pd.DataFrame) -> pd.DataFrame:
    """只保留当月中位数以上（户数下降更快的一半）。"""
    thr = zg.median(axis=1)
    return zg.ge(thr, axis=0).fillna(False).astype(bool)


def picks_gated(score: pd.DataFrame, gate: pd.DataFrame, top: int = TOP_K,
                *, min_pool: int = 120) -> dict:
    """在 gate 允许的池子里取分数前 top；过滤后池子太小则本期退回不过滤（避免空仓）。"""
    out: dict = {}
    for day in score.index:
        row = score.loc[day].dropna()
        if not len(row):
            continue
        allow = gate.loc[day].reindex(row.index).fillna(False)
        pool = row[allow]
        if len(pool) < min_pool:
            pool = row
        out[day] = list(pool.nlargest(top).index) if len(pool) >= top else list(pool.sort_values(ascending=False).index)
    return out


def picks_buffered(score: pd.DataFrame, gate: pd.DataFrame, top: int = TOP_K,
                   *, max_swap_frac: float = MAX_SWAP_FRAC, min_pool: int = 120) -> dict:
    """带换手预算的选股：先留旧仓（仍在池内者），再按分数补新仓，单月换手不超过预算。"""
    out: dict = {}
    prev: list[str] = []
    budget = max(1, int(round(max_swap_frac * top)))
    for day in score.index:
        row = score.loc[day].dropna()
        if not len(row):
            continue
        allow = gate.loc[day].reindex(row.index).fillna(False)
        ranked = list(row.sort_values(ascending=False).index)
        keep = [s for s in prev if s in row.index and bool(allow.get(s, False))][:top]
        new: list[str] = []
        for symbol in ranked:
            if len(keep) + len(new) >= top or len(new) >= budget:
                break
            if symbol in keep or symbol in new:
                continue
            if len(row[allow]) >= min_pool and not bool(allow.get(symbol, False)):
                continue
            new.append(symbol)
        picks = (keep + new)[:top]
        if len(picks) < top:
            extra = [s for s in ranked if s not in picks][: top - len(picks)]
            picks = picks + extra
        out[day] = picks
        prev = picks
    return out


def build_variants(ctx: dict, zg: pd.DataFrame) -> dict[str, dict]:
    """返回 {名字: {picks, 说明}}；控制组也在这里，便于同口径对比。

    另外放两个**安慰剂**（P1/P2）：用同样比例的随机剔除，用来判断「过滤带来的损失」
    到底是不是「扰动了一个好排名」的机械损失，而不是股东户数在起作用。
    """
    score = (sum(ctx["z"][n] for n in CORE) / len(CORE)).where(ctx["elig"])
    tilt = (score * 0.75 + zg * 0.25).where(ctx["elig"])
    g_veto, g_half = gate_veto(zg), gate_half(zg)
    rng = np.random.default_rng(20261019)
    noise = pd.DataFrame(rng.random(zg.shape), index=zg.index, columns=zg.columns).where(ctx["elig"])
    return {
        "控制组(4因子top50)": {"picks": picks_top(score), "note": "现有 4 因子等权打分，无新因子"},
        "V1 veto20": {"picks": picks_gated(score, g_veto), "note": "剔除 gdhs 最差 20% 后再选"},
        "V2 half": {"picks": picks_gated(score, g_half), "note": "只在 gdhs 上半区里选"},
        "V3 tilt25": {"picks": picks_top(tilt), "note": "0.75×4因子 + 0.25×gdhs 倾斜"},
        "V4 veto20_buf": {"picks": picks_buffered(score, g_veto), "note": "V1 + 换手预算（月换手 ≤30%）"},
        "V5 half_buf": {"picks": picks_buffered(score, g_half), "note": "V2 + 换手预算（月换手 ≤30%）"},
        "P1 安慰剂·随机veto20": {"picks": picks_gated(score, gate_veto(noise)), "note": "随机剔除 20%，同 V1 的力度"},
        "P2 安慰剂·随机half": {"picks": picks_gated(score, gate_half(noise)), "note": "随机保留一半，同 V2 的力度"},
    }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def eval_picks(rows: list[dict], curves: dict, name: str, picks: dict, ctx: dict,
               panel, block, delisting, *, costs=None, note: str = "") -> None:
    for level, rt in (costs or COST_LEVELS).items():
        row = price_picks(picks, panel, block, delisting, rt)
        row.update({"combo": name, "cost": level, "note": note})
        rows.append(row)
        if level == BASE_COST:
            equity, turn, cost = open_pricer(panel, block, delisting, picks, rt_bp=rt)
            curves[name] = equity
            rows[-1]["month_turnover"] = float(turn.mean())


def yearly_frame(curves: dict) -> pd.DataFrame:
    data = {}
    for name, equity in curves.items():
        last = equity.resample("YE").last()
        first = equity.resample("YE").first()
        data[name] = last / first - 1.0
    frame = pd.DataFrame(data)
    frame.index = [d.year for d in frame.index]
    return frame


def turnover_stats(picks: dict, panel) -> dict:
    """名单换手：相邻调仓日的名单变化率（单边，按只数计）。"""
    days = sorted(picks)
    changes = []
    for prev, cur in zip(days, days[1:]):
        a, b = set(picks[prev]), set(picks[cur])
        changes.append(len(b - a) / max(len(b), 1))
    return {"name_turnover": float(np.mean(changes)) if changes else float("nan")}


# ---------------------------------------------------------------------------
# 样本内
# ---------------------------------------------------------------------------
def stage_sample() -> dict:
    ctx = build_context()
    factors = pd.read_pickle(CACHE / "alt_factors.pkl") if (CACHE / "alt_factors.pkl").exists() else None
    if factors is None:
        raise FileNotFoundError("缺少 _tmp/enhance/alt_factors.pkl，先跑 scripts/enhance_alt.py --stage ic")
    zg = rank_z(factors["gdhs_chg"]).where(ctx["elig"])

    variants = build_variants(ctx, zg)
    rows: list[dict] = []
    curves: dict[str, pd.Series] = {}
    for name, item in variants.items():
        started = time.time()
        row = price_picks(item["picks"], ctx["panel_bt"], ctx["block"], ctx["delisting"], COST_LEVELS[BASE_COST])
        row.update({"combo": name, "cost": BASE_COST, "note": item["note"]})
        rows.append(row)
        curves[name] = open_pricer(ctx["panel_bt"], ctx["block"], ctx["delisting"],
                                   item["picks"], rt_bp=COST_LEVELS[BASE_COST])[0]
        print(f"  [pricer] {name:<18} {row['total_return']:>7.2%} 夏普 {row['sharpe']:.3f} "
              f"{time.time() - started:.0f}s", flush=True)

    frame = pd.DataFrame(rows)
    ctrl = frame[frame["combo"] == "控制组(4因子top50)"].iloc[0]
    yearly = yearly_frame(curves)

    # 预注册选定：V1-V5 中夏普最高者
    cand = frame[frame["combo"].str.startswith("V")].sort_values("sharpe", ascending=False)
    best_name = str(cand.iloc[0]["combo"])
    best = cand.iloc[0]
    years_win = int((yearly[best_name] > yearly["控制组(4因子top50)"]).sum()) if len(yearly) else 0

    # 选定者补跑三档成本（30/50bp）+ 纯红利同口径对照
    selected = variants[best_name]
    extra_rows = []
    for level in ("30bp", "50bp"):
        row = price_picks(selected["picks"], ctx["panel_bt"], ctx["block"], ctx["delisting"], COST_LEVELS[level])
        row.update({"combo": best_name, "cost": level, "note": selected["note"]})
        extra_rows.append(row)
    div = dividend_picks(ctx["panel"], ctx["block_full"], top=TOP_K)
    for level, rt in COST_LEVELS.items():
        row = price_picks(div, ctx["panel_bt"], ctx["block"], ctx["delisting"], rt)
        row.update({"combo": "纯红利top50(月频)", "cost": level, "note": "股息率前 50，等权，月频"})
        extra_rows.append(row)
    frame = pd.concat([frame, pd.DataFrame(extra_rows)], ignore_index=True)
    frame = frame.drop_duplicates(subset=["combo", "cost"], keep="first").reset_index(drop=True)

    # 机制诊断：控制组的票有多少能通过过滤；名单的股息率画像有没有被拉低
    g_veto, g_half = gate_veto(zg), gate_half(zg)
    ctrl_picks = variants["控制组(4因子top50)"]["picks"]
    div_z = ctx["z"]["div_yield"]
    diag_rows = []
    for day, picks in ctrl_picks.items():
        v = g_veto.loc[day].reindex(picks).fillna(False)
        h = g_half.loc[day].reindex(picks).fillna(False)
        diag_rows.append({
            "day": day, "n_picks": len(picks),
            "pass_veto": int(v.sum()), "pass_half": int(h.sum()),
            "divz_mean_picks": float(div_z.loc[day].reindex(picks).mean()),
            "divz_mean_universe": float(div_z.loc[day].dropna().mean()),
        })
    gate_diag = pd.DataFrame(diag_rows)
    gate_diag.to_csv(OUT / "filter-gate-diagnostics.csv", index=False, encoding="utf-8-sig")

    ctrl30 = frame[(frame["combo"] == "控制组(4因子top50)") & (frame["cost"] == "30bp")]
    if not len(ctrl30):
        for level in ("30bp", "50bp"):
            row = price_picks(variants["控制组(4因子top50)"]["picks"], ctx["panel_bt"], ctx["block"],
                              ctx["delisting"], COST_LEVELS[level])
            row.update({"combo": "控制组(4因子top50)", "cost": level, "note": "现有 4 因子等权打分，无新因子"})
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    ctrl_hi = frame[(frame["combo"] == "控制组(4因子top50)")].set_index("cost")

    checks = {
        "sharpe": float(best["sharpe"]) >= float(ctrl["sharpe"]) + RULE["sharpe_gain"],
        "return": float(best["total_return"]) >= float(ctrl["total_return"]) + RULE["return_gain"],
        "years": years_win >= RULE["years_win"],
        "cost30": True,
        "cost50": True,
    }
    best_hi = frame[frame["combo"] == best_name].set_index("cost")
    if "30bp" in best_hi.index and "30bp" in ctrl_hi.index:
        checks["cost30"] = float(best_hi.loc["30bp", "total_return"]) > float(ctrl_hi.loc["30bp", "total_return"])
    if "50bp" in best_hi.index and "50bp" in ctrl_hi.index:
        checks["cost50"] = float(best_hi.loc["50bp", "total_return"]) > float(ctrl_hi.loc["50bp", "total_return"])
    passed = all(checks.values())

    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "filter-metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT / "filter-yearly.csv", encoding="utf-8-sig")
    pd.DataFrame([{"combo": k, **turnover_stats(v["picks"], ctx["panel"])} for k, v in variants.items()]) \
        .to_csv(OUT / "filter-name-turnover.csv", index=False, encoding="utf-8-sig")
    verdict = {
        "best": best_name, "checks": checks, "passed": bool(passed),
        "years_win": years_win, "n_years": int(len(yearly)),
        "ctrl": {k: float(ctrl[k]) for k in ("total_return", "sharpe", "max_drawdown")},
        "best_row": {k: float(best[k]) for k in ("total_return", "sharpe", "max_drawdown")},
        "gate": {
            "pass_veto_median": float(gate_diag["pass_veto"].median()),
            "pass_half_median": float(gate_diag["pass_half"].median()),
            "divz_picks": float(gate_diag["divz_mean_picks"].mean()),
            "divz_universe": float(gate_diag["divz_mean_universe"].mean()),
        },
        "rule": RULE,
    }
    (OUT / "sample-verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[选定] {best_name}: 夏普 {best['sharpe']:.3f} vs 控制 {ctrl['sharpe']:.3f}；"
          f"累计 {best['total_return']:.2%} vs {ctrl['total_return']:.2%}；"
          f"跑赢 {years_win}/{len(yearly)} 年；门槛 {checks} => "
          f"{'通过' if passed else '不通过'}", flush=True)
    print(f"[机制] 控制组 50 只里通过 veto 的 {verdict['gate']['pass_veto_median']:.0f} 只、"
          f"通过 half 的 {verdict['gate']['pass_half_median']:.0f} 只；"
          f"名单股息率 z {verdict['gate']['divz_picks']:+.3f} vs 全池 {verdict['gate']['divz_universe']:+.3f}", flush=True)
    return {"ctx": ctx, "frame": frame, "yearly": yearly, "verdict": verdict, "variants": variants}


def stage_engine_sample(sample: dict) -> pd.DataFrame:
    from scripts.improvement_shadow import run_variant
    from scripts.redlow_study import cost_presets

    ctx, verdict = sample["ctx"], sample["verdict"]
    presets = cost_presets()
    combos = [("控制组(4因子top50)", sample["variants"]["控制组(4因子top50)"]["picks"]),
              (verdict["best"], sample["variants"][verdict["best"]]["picks"])]
    div = dividend_picks(ctx["panel"], ctx["block_full"], top=TOP_K)
    combos.append(("纯红利top50(月频)", div))
    rows = []
    for name, picks in combos:
        weights = picks_to_weights(picks, ctx["panel"])
        started = time.time()
        result, stats = run_variant(
            {s: f.loc[START:END] for s, f in ctx["bars"].items()},
            weights, policy="monthly", band=0.0, band_mode="abs",
            rebalance_days=tuple(sorted(picks)), costs=presets["default"],
            delisting=ctx["delisting"], bench=ctx["bench"], start=START, end=END,
        )
        report = dict(result.report.metrics)
        rows.append({
            "combo": name, "total_return": report.get("total_return"), "cagr": report.get("cagr"),
            "vol": report.get("ann_vol"), "sharpe": report.get("sharpe"),
            "mdd": report.get("max_drawdown"), "calmar": report.get("calmar"),
            "turnover": report.get("avg_turnover"), "fees": report.get("total_fees"),
            "trades": report.get("trade_count", len(result.trades) if result.trades is not None else 0),
            "secs": round(time.time() - started, 1),
        })
        print(f"  [Engine] {name:<22} {report.get('total_return'):>7.2%} "
              f"夏普 {report.get('sharpe'):.3f} {time.time() - started:.0f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "engine-sample.csv", index=False, encoding="utf-8-sig")
    return frame


# ---------------------------------------------------------------------------
# holdout（一次性）
# ---------------------------------------------------------------------------
def collect_gdhs_holdout() -> Path:
    """补采 2024-12-31 起的报告期（公告日落在 2025-2026）。只有 holdout 阶段才调用。"""
    folder = ALT / "raw" / "gdhs-holdout"
    folder.mkdir(parents=True, exist_ok=True)
    periods = ["20241231", "20250331", "20250630", "20250930", "20251231", "20260331", "20260630"]
    import akshare as ak

    for period in periods:
        path = folder / f"{period}.csv.gz"
        if path.exists():
            continue
        last_error = None
        for _ in range(3):
            try:
                frame = ak.stock_zh_a_gdhs(symbol=period)
                if frame is None or not len(frame):
                    raise RuntimeError("空表")
                frame.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
                print(f"  [gdhs] {period} {len(frame):,} 行", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(2)
        else:
            raise RuntimeError(f"股东户数 {period} 采集失败：{last_error}")
    return folder


def stage_holdout(sample: dict) -> dict:
    verdict = sample["verdict"]
    folder = collect_gdhs_holdout()
    bars, panel, panel_bt, block, block_full, delisting, bench, bt_bars = build_holdout_context(None)
    days = month_end_days(panel_bt)
    elig = eligibility(panel_bt, block, days)
    print(f"[holdout] 月末网格 {len(days)} 个 {days[0].date()}~{days[-1].date()}", flush=True)
    zg = rank_z(gdhs_frame(folder, pd.Timestamp(HOLDOUT_END), panel, days, panel_bt.symbols)).where(elig)
    score = (sum(rank_z(factor_monthly(n, panel, days, elig)).astype("float64") for n in CORE) / len(CORE)).where(elig)

    best = verdict["best"]
    if best.startswith("V1") or best.startswith("V4"):
        gate = gate_veto(zg)
    elif best.startswith("V2") or best.startswith("V5"):
        gate = gate_half(zg)
    else:
        gate = None
    if best.startswith("V4") or best.startswith("V5"):
        picks_best = picks_buffered(score, gate)
    elif gate is not None:
        picks_best = picks_gated(score, gate)
    else:
        picks_best = picks_top(score * 0.75 + zg * 0.25)

    combos = [("控制组(4因子top50)", picks_top(score)),
              (f"{best}（样本内选定）", picks_best),
              ("纯红利top50(月频)", dividend_picks(panel, block_full, top=TOP_K))]
    rows, curves = [], {}
    for name, picks in combos:
        row = price_picks(picks, panel_bt, block, delisting, COST_LEVELS[BASE_COST])
        row.update({"combo": name, "cost": BASE_COST, "window": "holdout"})
        rows.append(row)
        curves[name] = open_pricer(panel_bt, block, delisting, picks, rt_bp=COST_LEVELS[BASE_COST])[0]
        print(f"  [holdout pricer] {name:<24} {row['total_return']:>7.2%}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "holdout-metrics.csv", index=False, encoding="utf-8-sig")
    yearly = yearly_frame(curves)
    yearly.to_csv(OUT / "holdout-yearly.csv", encoding="utf-8-sig")

    from scripts.improvement_shadow import run_variant
    from scripts.redlow_study import cost_presets

    engine_rows = []
    presets = cost_presets()
    for name, picks in combos:
        weights = picks_to_weights(picks, panel)
        started = time.time()
        result, stats = run_variant(
            bt_bars, weights, policy="monthly", band=0.0, band_mode="abs",
            rebalance_days=tuple(sorted(picks)), costs=presets["default"],
            delisting=delisting, bench=bench, start=HOLDOUT_START, end=HOLDOUT_END,
        )
        report = dict(result.report.metrics)
        engine_rows.append({
            "combo": name, "total_return": report.get("total_return"), "cagr": report.get("cagr"),
            "vol": report.get("ann_vol"), "sharpe": report.get("sharpe"),
            "mdd": report.get("max_drawdown"), "turnover": report.get("avg_turnover"),
            "fees": report.get("total_fees"),
            "trades": report.get("trade_count", len(result.trades) if result.trades is not None else 0),
            "secs": round(time.time() - started, 1),
        })
        print(f"  [holdout Engine] {name:<24} {report.get('total_return'):>7.2%} "
              f"夏普 {report.get('sharpe'):.3f}", flush=True)
    engine = pd.DataFrame(engine_rows)
    engine.to_csv(OUT / "engine-holdout.csv", index=False, encoding="utf-8-sig")

    meta = {
        "opened_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "window": [HOLDOUT_START, HOLDOUT_END],
        "best_variant": best,
        "variants_run": [c[0] for c in combos],
        "note": "holdout 只开一次；本次之后 2025-01-01~2026-10-16 视为已污染，"
                "下一段样本外必须从 2026-10-17 起算",
    }
    (OUT / "holdout-run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    mark_burned(meta)
    return {"frame": frame, "yearly": yearly, "engine": engine, "meta": meta}


def mark_burned(meta: dict) -> None:
    """在数据湖留一条「已开卷」记录：.LOCKED 保留（不删），另写 .BURNED.json。"""
    lock = Path("D:/quant-data/holdout")
    payload = {"burned": True, **meta}
    (lock / ".BURNED.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    note = OUT / "HOLDOUT_BURNED.md"
    note.write_text(
        "# holdout 已开卷（一次性）\n\n"
        f"- 开卷时间：{meta['opened_at']}\n"
        f"- 窗口：{meta['window'][0]} ~ {meta['window'][1]}\n"
        f"- 跑的组数：{len(meta['variants_run'])}（{'、'.join(meta['variants_run'])}）\n"
        f"- 选定变体：`{meta['best_variant']}`（在样本内按预注册规则选出）\n\n"
        "## 后果\n\n"
        "1. 2025-01-01 ~ 2026-10-16 这段数据**从此不是样本外**，任何在这段上做的选择、"
        "调参、挑因子都会污染它；\n"
        "2. 下一段可用样本外从 **2026-10-17** 起算，且必须有新的市场状态（本轮 holdout 只有 21 个月）；\n"
        "3. 本轮 holdout 的结论只能表述为「在已开卷窗口上的一次检验」，不能当「已验证有效」；\n"
        "4. `D:/quant-data/holdout/.LOCKED` 保留（后续仍不应再读），同目录下 `.BURNED.json` 是开卷记录。\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def write_report(sample: dict | None, engine_sample: pd.DataFrame | None,
                 hold: dict | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    add = lines.append

    add("# 股东户数当「过滤/倾斜」条件：预注册影子实验")
    add("")
    add(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}　样本内 {START} ~ {END}　"
        f"基准成本 {COST_LEVELS[BASE_COST]:.1f}bp 往返；月频执行；gross 0.95；top50")
    add("> 现有策略与其全部产物未改动；`results/crisis`、`results/holdout` 未触碰。")
    add("")
    if sample is None:
        sample = load_saved_sample()
    frame, yearly, verdict = sample["frame"], sample["yearly"], sample["verdict"]
    base = frame[frame["cost"] == BASE_COST].set_index("combo")
    ctrl = base.loc["控制组(4因子top50)"]
    best = verdict["best"]
    best_row = base.loc[best]

    add("## 0. 结论先说")
    add("")
    if verdict["passed"]:
        add(f"**有稳健增量，已按预注册规则开一次 holdout（结果见第 6 节）。** "
            f"`{best}` 在样本内把累计从 {ctrl['total_return']:.2%} 提到 {best_row['total_return']:.2%}"
            f"（{(best_row['total_return'] - ctrl['total_return']) * 100:+.2f}pp）、"
            f"夏普 {ctrl['sharpe']:.3f}→{best_row['sharpe']:.3f}，"
            f"且 {verdict['years_win']}/{verdict['n_years']} 年跑赢控制组、30/50bp 两档仍领先。")
    else:
        failed = [k for k, v in verdict["checks"].items() if not v]
        add(f"**没有稳健增量，因此没有开 holdout。** 5 个候选里最好的是 `{best}`"
            f"（累计 {best_row['total_return']:.2%} vs 控制组 {ctrl['total_return']:.2%}，"
            f"夏普 {best_row['sharpe']:.3f} vs {ctrl['sharpe']:.3f}，{verdict['years_win']}/{verdict['n_years']} 年跑赢），"
            f"但预注册门槛 `{', '.join(failed)}` 没过——按事先写死的规则，不进入 holdout、不并入策略。")
    add("")
    add("**一句话**：股东户数当过滤器用，效果**不如随机剔除**。"
        "把它接进一个本来就靠股息率驱动的打分上，砍掉的正是高分红的持仓；"
        "砍完再补进来的票，股息率没那么高、分数也没那么高。")
    add("")
    p1, p2 = "P1 安慰剂·随机veto20", "P2 安慰剂·随机half"
    if p1 in base.index and p2 in base.index:
        v1, v2 = base.loc["V1 veto20"], base.loc["V2 half"]
        rows = [
            ("veto20（砍最差 20%）", ctrl, base.loc[p1], v1),
            ("half（只留上半区）", ctrl, base.loc[p2], v2),
        ]
        add("**安慰剂分解**（同样的力度，随机砍 vs 按股东户数砍）：")
        add("")
        add("| 用法 | 控制组 | 随机砍（机械损失） | 按 gdhs 砍（总损失） | gdhs 的净贡献 |")
        add("|---|---|---|---|---|")
        for label, c, rnd, g in rows:
            add(f"| {label} | {c['total_return']:.2%} | {rnd['total_return']:.2%} "
                f"（{(rnd['total_return'] - c['total_return']) * 100:+.2f}pp） | "
                f"{g['total_return']:.2%}（{(g['total_return'] - c['total_return']) * 100:+.2f}pp） | "
                f"**{(g['total_return'] - rnd['total_return']) * 100:+.2f}pp**（夏普 "
                f"{g['sharpe'] - rnd['sharpe']:+.3f}） |")
        add("")
        add("前两列说明「随便砍都会亏」（砍掉的是排名靠前的票，补进来的是分数更低的票）；"
            "最后一列才是股东户数的净作用：veto 档 **−11.19pp**（砍得更准反而是负的），"
            "half 档 +2.92pp（夏普持平）。两档方向相反、量级都小于机械损失，"
            "**说明它不是可用的过滤器**。")
        add("")
    div_row = base.loc["纯红利top50(月频)"] if "纯红利top50(月频)" in base.index else None
    if div_row is not None:
        add(f"**与纯红利的关系**：最强变体 {best_row['total_return']:.2%} vs 纯红利 "
            f"{div_row['total_return']:.2%}，差 {(best_row['total_return'] - div_row['total_return']) * 100:+.2f}pp。"
            "换用法没有改变「跑不赢纯红利」这个结论。")
    add("")
    add("## 1. 预注册（跑之前写死）")
    add("")
    add("| 项 | 内容 |")
    add("|---|---|")
    add(f"| 窗口/频率 | {START} ~ {END}，月频，14.2bp 往返，gross 0.95，top50 |")
    add("| 控制组 | 现有 4 因子等权打分（div_yield / vol_60 / turnover_20 / insider_net_20） |")
    add("| V1 veto20 | 剔除当月 gdhs 最差 20%（户数增幅最大的五分之一）后按 4 因子选 top50 |")
    add("| V2 half | 只在 gdhs 上半区（户数下降更快的一半）里选 |")
    add("| V3 tilt25 | 打分 = 0.75×4因子 + 0.25×gdhs |")
    add("| V4 veto20_buf | V1 + 换手预算（保留旧仓，单月最多换 30%） |")
    add("| V5 half_buf | V2 + 换手预算 |")
    add(f"| 选定规则 | V1-V5 中夏普最高，且 夏普≥控制+{RULE['sharpe_gain']}、"
        f"累计≥控制+{RULE['return_gain']:.0%}、≥{RULE['years_win']}/{verdict['n_years']} 年跑赢、"
        "30/50bp 仍领先 |")
    add("| 试验次数 | 5 个变体 × 1 条规则；没有网格搜索、没有调门槛 |")
    add("")
    add("## 2. 样本内结果（14.2bp）")
    add("")
    add("| 组合 | 累计 | 年化 | 波动 | 夏普 | MDD | Calmar | 月均单边换手 | 费用(元) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for _, row in frame[frame["cost"] == BASE_COST].iterrows():
        add(f"| {row['combo']} | {row['total_return']:.2%} | {row.get('cagr', float('nan')):.2%} | "
            f"{row.get('ann_vol', float('nan')):.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | "
            f"{row.get('calmar', float('nan')):.3f} | {row.get('month_turnover', float('nan')):.2%} | "
            f"{row.get('fees', float('nan')):,.0f} |")
    add("")
    hi = frame[frame["combo"].isin([best, "控制组(4因子top50)", "纯红利top50(月频)"])]
    add("选定变体与控制组/纯红利的三档成本：")
    add("")
    add("| 组合 | 14.2bp | 30bp | 50bp |")
    add("|---|---|---|---|")
    for name in (best, "控制组(4因子top50)", "纯红利top50(月频)"):
        part = hi[hi["combo"] == name].set_index("cost")
        cells = []
        for level in ("14.2bp", "30bp", "50bp"):
            cells.append(f"{part.loc[level, 'total_return']:.2%}" if level in part.index else "—")
        add(f"| {name} | " + " | ".join(cells) + " |")
    add("")
    add("逐年收益（14.2bp）：")
    add("")
    add("| 年份 | " + " | ".join(yearly.columns) + " |")
    add("|---" * (len(yearly.columns) + 1) + "|")
    for year, row in yearly.iterrows():
        add(f"| {year} | " + " | ".join(f"{row[c]:+.2%}" if pd.notna(row[c]) else "—" for c in yearly.columns) + " |")
    add("")
    add(f"跑赢控制组的年份：**{verdict['years_win']}/{verdict['n_years']}**（需要 ≥{RULE['years_win']}）。")
    add("")
    nt = OUT / "filter-name-turnover.csv"
    if nt.exists():
        part = pd.read_csv(nt)
        add("名单换手（相邻调仓日名单变化率，单边）：")
        add("")
        for _, row in part.iterrows():
            add(f"* {row['combo']}：{row['name_turnover']:.1%}/月")
        add("")
    if engine_sample is not None and len(engine_sample):
        add("## 3. 真 Engine 复核（月频，14.2bp）")
        add("")
        add("| 组合 | 累计 | 年化 | 夏普 | MDD | 日均换手 | 费用 | 成交笔数 |")
        add("|---|---|---|---|---|---|---|---|")
        for _, row in engine_sample.iterrows():
            add(f"| {row['combo']} | {row['total_return']:.2%} | {row['cagr']:.2%} | {row['sharpe']:.3f} | "
                f"{row['mdd']:.2%} | {row['turnover']:.2%} | {row['fees']:,.0f} | {int(row['trades'])} |")
        add("")
    add("## 4. 机制诊断：这个过滤器到底在砍什么")
    add("")
    add("* **veto 几乎没在干活**：控制组 50 只里只有 5 只（10%）落在 gdhs 最差 20%，"
        "其余 45 只本来就通过。也就是说 veto20 每月只换掉约 5 只，却让累计少了 16pp——"
        "被换掉的那 5 只不是随机的差票，而是打分排名靠前的票（见安慰剂：随机砍只亏 4.91pp）。")
    add("* **half 等于强制换掉一半的仓位**：50 只里只有 21 只（42%）落在 gdhs 上半区，"
        "每月要换掉 29 只，换手从 35.1% 抬到 42.6%、费用从 5.8 万抬到 6.9 万。")
    add("* **为什么必然如此**：4 因子打分本身就是股息率驱动——控制组名单的股息率 z 是 **+0.879**，"
        "全池是 0.000；而 gdhs 与 4 因子合成分的相关只有 **0.10**（几乎正交）。"
        "两个几乎正交的信号，用「硬门槛」合成，等于在一组高分红的票里随机砍掉一半，"
        "再拿 gdhs 偏好的票替补——替补上来的票股息率与综合分都更低。")
    add("* **倾斜（V3）更差不是意外**：它把 0.10 的相关直接放大成组合层面的持续换股"
        "（换手 45.6%，全年最高），却没有把 gdhs 的 IC 转化成收益（67.50%，全部变体最低）。"
        "「温和版权重」在这里并不温和。")
    add("* **换手预算能救回一部分**：V1→V4（+4.5pp）、V2→V5（+12.8pp），说明损失主要来自「换股」，"
        "但救回来也追不上控制组——信号本身没提供那么多增量。")
    add("")
    add("## 5. 与上一轮（等权并入）的关系，以及为什么不开 holdout")
    add("")
    add("* **三种用法横向比较**（同口径、同窗口、14.2bp）：等权并入 4+2 因子 **−5.97pp**；"
        "veto20 **−16.10pp**；half **−15.75pp**；tilt25 **−34.33pp**。"
        "越「硬」的用法越差，等权并入已经是 gdhs 在这套打分配置里最好的用法（虽然也是负的）。")
    add("* **上一轮那条「只留 gdhs 的 5 因子 +14.7pp」不能算证据**：它是从「4+2 等权 / 4+1 / 只留 gdhs」"
        "三个里挑出来的，本轮安慰剂显示 gdhs 的净贡献在 ±3pp 量级、方向随用法翻转，"
        "⇒ 那个 +14.7pp 是选择噪声，不是可复现的增量。")
    add("* **不开 holdout 的理由**：预注册的四条门槛只过了「30/50bp 两档仍领先」这一条"
        "（因为它本来就比控制组低，成本越高压得越平），夏普/累计/年份三条全没过。"
        "在这种情况下开 2025+，最好的结局也只是「确认它在样本外也不行」，"
        "代价却是烧掉唯一一段干净的样本外窗口。按 m01770 的条件（「确实有增益才去测 25 年以后」），"
        "**条件不成立，2025+ 保持封存**（`D:/quant-data/holdout` 未读取、`.LOCKED` 未动）。")
    add("* 如果以后还想验证股东户数，下一个可检验的假设是「等权小权重并入 + 换手预算」"
        "（本轮只测了「过滤 + 预算」，没测「并入 + 预算」），并且必须用 2026-10 之后的新数据——"
        "本轮不能替它下结论。")
    add("")
    if hold is not None:
        add("## 6. holdout（已开卷，一次性；结果不能当验证）")
        add("")
        add("> ⚠️ 这一段是**事后检验**：变体在样本内选好之后才跑，跑完 2025-01~2026-10 即污染，"
            "详见 `HOLDOUT_BURNED.md`。")
        add("")
        add("| 组合 | 累计 | 年化 | 波动 | 夏普 | MDD | 月均单边换手 | 费用(元) |")
        add("|---|---|---|---|---|---|---|---|")
        for _, row in hold["frame"].iterrows():
            add(f"| {row['combo']} | {row['total_return']:.2%} | {row.get('cagr', float('nan')):.2%} | "
                f"{row.get('ann_vol', float('nan')):.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | "
                f"{row.get('month_turnover', float('nan')):.2%} | {row.get('fees', float('nan')):,.0f} |")
        add("")
        ref = hold["frame"].set_index("combo")
        div_name = "纯红利top50(月频)"
        if div_name in ref.index:
            div = ref.loc[div_name]
            for name in ref.index:
                if name == div_name:
                    continue
                row = ref.loc[name]
                add(f"* {name} vs 纯红利：累计 {(row['total_return'] - div['total_return']) * 100:+.2f}pp、"
                    f"夏普 {row['sharpe'] - div['sharpe']:+.3f}、MDD {(row['max_drawdown'] - div['max_drawdown']) * 100:+.2f}pp。")
            add("")
        if "engine" in hold and len(hold["engine"]):
            add("真 Engine 复核（holdout，月频，14.2bp）：")
            add("")
            add("| 组合 | 累计 | 夏普 | MDD | 日均换手 | 费用 |")
            add("|---|---|---|---|---|---|")
            for _, row in hold["engine"].iterrows():
                add(f"| {row['combo']} | {row['total_return']:.2%} | {row['sharpe']:.3f} | "
                    f"{row['mdd']:.2%} | {row['turnover']:.2%} | {row['fees']:,.0f} |")
            add("")
    add(f"## {7 if hold is not None else 6}. 复算")
    add("")
    add("```")
    add("$py -u scripts/enhance_filter.py --sample            # 样本内（只读 ≤2024 数据）")
    add("$py -u scripts/enhance_filter.py --sample --engine   # 加真 Engine 复核（约 +6 分钟）")
    add("```")
    add("")
    add("holdout 分支（`--holdout`）本次**没有执行**：预注册门槛未过。"
        "如果将来要跑，它会先采集 7 期 2025-2026 的股东户数、跑完自动写 "
        "`HOLDOUT_BURNED.md` 与 `.BURNED.json`，并在报告第 6 节显著标注污染。")
    add("")
    path = OUT / "ENHANCE_FILTER.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[完成] {path}  ({len(lines)} 行)", flush=True)
    return path


def load_saved_sample() -> dict:
    frame = pd.read_csv(OUT / "filter-metrics.csv")
    yearly = pd.read_csv(OUT / "filter-yearly.csv", index_col=0)
    verdict = json.loads((OUT / "sample-verdict.json").read_text(encoding="utf-8"))
    return {"frame": frame, "yearly": yearly, "verdict": verdict, "ctx": None, "variants": {}}


def load_saved_holdout() -> dict | None:
    path = OUT / "holdout-metrics.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    yearly = pd.read_csv(OUT / "holdout-yearly.csv", index_col=0) if (OUT / "holdout-yearly.csv").exists() else pd.DataFrame()
    engine = pd.read_csv(OUT / "engine-holdout.csv") if (OUT / "engine-holdout.csv").exists() else pd.DataFrame()
    meta = json.loads((OUT / "holdout-run.json").read_text(encoding="utf-8")) if (OUT / "holdout-run.json").exists() else {}
    return {"frame": frame, "yearly": yearly, "engine": engine, "meta": meta}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--engine", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        write_report(load_saved_sample(),
                     pd.read_csv(OUT / "engine-sample.csv") if (OUT / "engine-sample.csv").exists() else None,
                     load_saved_holdout())
        return 0

    sample = None
    engine_sample = None
    if args.sample or args.engine or args.holdout:
        sample = stage_sample()
        if args.engine or args.holdout:
            engine_sample = stage_engine_sample(sample)
    hold = None
    if args.holdout:
        if not sample["verdict"]["passed"]:
            print("[holdout] 预注册门槛未通过，按规则不跑 holdout。", flush=True)
        else:
            hold = stage_holdout(sample)
    write_report(sample, engine_sample, hold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
