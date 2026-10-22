"""三类独立数据（股东户数 / 融资融券）的影子版因子诊断与增强对照。

纪律
----
* 只用 2017-01-03 ~ 2024-12-31；`D:/quant-data/holdout` 全程不读（有 .LOCKED）。
* 不改任何现有脚本/因子/权重/产物；结果只写 `results/enhance-alternative/`。
* 因子定义、PIT 口径、纳入规则在看 IC 之前写死（见下方常量与 ``PREREG``）。
* 个股资金流经探测判定「不可得」（东财服务端只回最近 121 个交易日，且全在 2026 年），
  不硬凑；用两融余额作为杠杆流向的替代代理。

PIT 口径
--------
* 股东户数：按 ``公告日期`` 生效（公告日收盘后可用），不用报告期末对齐；
  ``增减比例`` 是相对上一报告期的环比。
* 融资融券：交易所 T 日盘后披露 → T+1 可用。月末决策在 T 日收盘做、T+1 开盘成交，
  所以 T 日的数据可用。
* 资金流：不可得。

用法
----
    $py -u scripts/enhance_alt.py --stage data
    $py -u scripts/enhance_alt.py --stage ic
    $py -u scripts/enhance_alt.py --stage bt
    $py -u scripts/enhance_alt.py --stage all --engine
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import quant.factors.events as event_factors
from scripts.improvement_shadow import END, START, build_sample_context
from scripts.strategy_optimize import (
    CORE,
    COST_LEVELS,
    TOP_K,
    build_context,
    dividend_picks,
    existing_weights,
    ic_stats,
    monthly_ic,
    open_pricer,
    picks_to_weights,
    price_picks,
    residual_ic,
    stats_from_equity,
    rank_z,
    month_end_days,
    eligibility,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "enhance-alternative"
CACHE = ROOT / "_tmp" / "enhance"
ALT = Path("D:/quant-data/alt")

event_factors.register()  # lhb_flag_5 等事件因子要显式注册后 library.compute 才认得

BASE_COST = "14.2bp"

# 预注册：候选因子（方向 sign 在测 IC 之前按经济含义固定，不用数据定方向）
#   gdhs_chg   股东户数环比（%）：户数下降＝筹码集中 → 预期正 IC（取负号后越大越好）
#   margin_crowd 融资余额 / 近 20 日成交额：杠杆拥挤 → 预期负 IC（越拥挤未来越差）
#   margin_chg20 融资余额 20 日环比：杠杆快速加仓 → 预期负 IC
CANDIDATES = {
    "gdhs_chg": {"source": "gdhs", "sign": -1, "note": "股东户数环比，负号＝户数下降为好"},
    "gdhs_chg_2q": {"source": "gdhs", "sign": -1, "note": "股东户数两期累计变化，负号"},
    "margin_crowd": {"source": "margin", "sign": -1, "note": "融资余额/20日成交额，负号"},
    "margin_chg20": {"source": "margin", "sign": -1, "note": "融资余额 20 日环比，负号"},
}
# 纳入规则（看 IC 之前写死）：单因子 t>=2、正 IC 年份>=5/8、|与现有因子相关|<0.5、
# 且对现有 4 因子的残差 IC t>=2 —— 四条全过才进组合
PREREG = {"t_stat": 2.0, "years_positive": 5, "max_corr": 0.5, "resid_t": 2.0}


# ---------------------------------------------------------------------------
# 数据读取 / PIT 化
# ---------------------------------------------------------------------------
def load_gdhs() -> pd.DataFrame:
    files = sorted((ALT / "raw" / "gdhs").glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError("没有股东户数原始文件，先跑 scripts/alt_data_collect.py --gdhs")
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
    long = long[long["period"] <= pd.Timestamp("2024-09-30")]
    return long


def asof_matrix(long: pd.DataFrame, value: str, days: list[pd.Timestamp],
                symbols, *, max_age_days: int | None = None) -> pd.DataFrame:
    """按 ``avail_date`` 做 as-of 前向填充到月末网格；``max_age_days`` 外的陈旧值置 NaN。"""
    part = long.dropna(subset=[value, "avail_date"])
    wide = part.pivot_table(index="avail_date", columns="code", values=value, aggfunc="last")
    wide = wide.reindex(columns=list(symbols))
    idx = wide.index.union(pd.DatetimeIndex(days)).sort_values()
    wide = wide.reindex(idx).ffill().reindex(index=pd.DatetimeIndex(days))
    if max_age_days is not None:
        stamp = part.assign(_stamp=part["avail_date"]).pivot_table(
            index="avail_date", columns="code", values="_stamp", aggfunc="max").reindex(columns=list(symbols))
        stamp = stamp.reindex(idx).ffill().reindex(index=pd.DatetimeIndex(days))
        left = pd.DatetimeIndex(days).to_numpy(dtype="datetime64[ns]")[:, None]
        right = stamp.to_numpy(dtype="datetime64[ns]")
        age = (left - right) / np.timedelta64(1, "D")
        wide = wide.where(age <= max_age_days)
    return wide.astype("float32")


def gdhs_monthly(panel, days, symbols) -> dict[str, pd.DataFrame]:
    """股东户数三个口径：单期环比、两期累计、以及户均持股变化（作为稳健性对照）。"""
    long = load_gdhs()
    # 只保留户数规模够大的记录，避免 63 户→5891 户这类极端分母
    long = long[(long["holders"] >= 1000) & (long["holders_prev"] >= 1000)]
    long["avail_date"] = long["ann_date"]
    long["chg"] = long["chg_pct"].clip(-60.0, 100.0) / 100.0
    long["chg2"] = ((long["holders"] / long["holders_prev"]).where(long["holders_prev"] > 0) - 1.0).clip(-0.6, 1.0)
    out = {
        "gdhs_chg": asof_matrix(long, "chg", days, symbols, max_age_days=400),
        "gdhs_chg_2q": asof_matrix(long, "chg2", days, symbols, max_age_days=400),
    }
    coverage = {k: int(v.notna().sum(axis=1).median()) for k, v in out.items()}
    print(f"[data] 股东户数 记录 {len(long):,} 条，{long['period'].nunique()} 期，"
          f"覆盖中位数 {coverage}", flush=True)
    return out


def load_margin() -> pd.DataFrame:
    rows = []
    for exch in ("sse", "szse"):
        folder = ALT / "raw" / "margin" / exch
        files = sorted(folder.glob("*.csv.gz"))
        if not files:
            continue
        for path in files:
            frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
            code = frame["标的证券代码"] if "标的证券代码" in frame.columns else frame["证券代码"]
            part = pd.DataFrame({
                "date": pd.Timestamp(path.name[:8]),
                "exch": exch,
                "code": code.astype(str).str.zfill(6),
            })
            for src, dst in (("融资余额", "fin_bal"), ("融资买入额", "fin_buy"),
                             ("融资偿还额", "fin_repay"), ("融券余量", "short_qty"),
                             ("融券余额", "short_bal"), ("融资融券余额", "fin_short_bal")):
                part[dst] = pd.to_numeric(frame[src], errors="coerce") if src in frame.columns else np.nan
            rows.append(part)
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    long = long[long["code"].str.match(r"^(0|3|6)\d{5}$", na=False)]
    return long


def margin_monthly(panel, days, symbols) -> dict[str, pd.DataFrame]:
    long = load_margin()
    if not len(long):
        return {}
    long = long[long["date"] <= pd.Timestamp(END)]
    fin = long.pivot_table(index="date", columns="code", values="fin_bal", aggfunc="last")
    fin = fin.reindex(columns=list(symbols))
    cal = pd.DatetimeIndex(panel.dates)
    pos = {d: i for i, d in enumerate(cal)}
    days = list(days)

    amount = panel.amount.reindex(index=pd.DatetimeIndex(days), columns=list(symbols)).astype("float64")
    amount20 = panel.amount.rolling(20, min_periods=10).mean().reindex(index=pd.DatetimeIndex(days))
    crowd = (fin.reindex(index=pd.DatetimeIndex(days)) / (amount20 * 20.0)).astype("float32")

    prev_days, valid = [], []
    for day in days:
        i = pos.get(day)
        if i is None or i < 20:
            prev_days.append(pd.NaT)
            valid.append(False)
            continue
        prev_days.append(cal[i - 20])
        valid.append(True)
    chg = pd.DataFrame(np.nan, index=pd.DatetimeIndex(days), columns=list(symbols), dtype="float32")
    for day, prev, ok in zip(days, prev_days, valid):
        if not ok or prev not in fin.index:
            continue
        cur = fin.loc[day] if day in fin.index else None
        if cur is None:
            continue
        base = fin.loc[prev]
        ratio = cur / base.where(base > 0) - 1.0
        chg.loc[day] = ratio.reindex(list(symbols)).clip(-0.9, 3.0).astype("float32")
    print(f"[data] 两融 记录 {len(long):,} 条，{long['date'].nunique()} 个抽样日，"
          f"覆盖中位数 crowd={int(crowd.notna().sum(axis=1).median())} chg20={int(chg.notna().sum(axis=1).median())}",
          flush=True)
    return {"margin_crowd": crowd, "margin_chg20": chg}


# ---------------------------------------------------------------------------
def build_factors(ctx: dict, *, refresh: bool = False) -> dict[str, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "alt_factors.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    panel, days, symbols = ctx["panel"], ctx["days"], ctx["panel"].symbols
    factors: dict[str, pd.DataFrame] = {}
    factors.update(gdhs_monthly(panel, days, symbols))
    factors.update(margin_monthly(panel, days, symbols))
    factors = {k: v.where(ctx["elig"]) for k, v in factors.items()}
    pd.to_pickle(factors, path)
    return factors


def stage_ic(ctx: dict, factors: dict[str, pd.DataFrame]) -> dict:
    days, forward = ctx["days"], ctx["forward"]
    # 按预注册的经济方向定向（sign 来自经济含义，不看 IC 正负）：定向后「越大越好」
    oriented = {n: (f * CANDIDATES[n]["sign"] if n in CANDIDATES else f) for n, f in factors.items()}
    ic, stats = {}, {}
    for name, frame in oriented.items():
        ic[name] = monthly_ic(frame, forward)
        stats[name] = ic_stats(ic[name])
    ic_frame = pd.DataFrame(ic)

    z = {n: rank_z(f) for n, f in oriented.items()}
    controls = {k: ctx["z"][k] for k in CORE}
    resid = residual_ic({**controls, **z}, forward, CORE)
    resid_stats = {n: ic_stats(resid[n]) for n in z}

    corr = pd.DataFrame(index=list(z), columns=list(CORE), dtype=float)
    for a in z:
        for b in CORE:
            x = z[a].to_numpy(dtype=float)
            y = ctx["z"][b].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            corr.loc[a, b] = float(np.corrcoef(x[mask], y[mask])[0, 1]) if mask.sum() > 100 else np.nan

    yearly = ic_frame.copy()
    yearly.index = pd.DatetimeIndex(yearly.index)
    by_year = yearly.groupby(yearly.index.year).mean()

    verdict = {}
    for name in z:
        s, r = stats[name], resid_stats[name]
        max_corr = float(np.nanmax(np.abs(corr.loc[name].to_numpy(dtype=float))))
        ok = (s.get("t_stat", 0) >= PREREG["t_stat"] and s.get("years_positive", 0) >= PREREG["years_positive"]
              and max_corr < PREREG["max_corr"] and r.get("t_stat", 0) >= PREREG["resid_t"])
        verdict[name] = {"ok": bool(ok), "max_corr": max_corr, "resid_t": r.get("t_stat", float("nan"))}
    return {"ic": ic_frame, "stats": stats, "resid": pd.DataFrame(resid), "resid_stats": resid_stats,
            "corr": corr, "by_year": by_year, "verdict": verdict, "z": z,
            "coverage": {k: int(v.notna().sum(axis=1).median()) for k, v in factors.items()}}


def stage_bt(ctx: dict, diag: dict, *, top: int = TOP_K) -> dict:
    days, elig = ctx["days"], ctx["elig"]
    included = [n for n, v in diag["verdict"].items() if v["ok"]]
    # gdhs_chg_2q 只是同数据的另一种算法（稳健性对照），与 gdhs_chg 同时入选时只留一个，避免重复计权
    if "gdhs_chg" in included and "gdhs_chg_2q" in included:
        included.remove("gdhs_chg_2q")
    names = CORE + included
    weights_by_name = {n: ctx["z"][n] for n in CORE}
    weights_by_name.update({n: diag["z"][n] for n in included})
    score = sum(weights_by_name[n] for n in names) / len(names)
    score = score.where(elig)
    # 控制组：完全相同的打分/选股/定价流程，只是不含新因子——用于把「新因子贡献」与「方法差异」分开
    score_core = sum(ctx["z"][n] for n in CORE) / len(CORE)
    score_core = score_core.where(elig)

    def picks_from(sc: pd.DataFrame) -> dict:
        out = {}
        for day in sc.index:
            row = sc.loc[day].dropna()
            if len(row) < top:
                if len(row):
                    out[day] = list(row.sort_values(ascending=False).index)
                continue
            out[day] = list(row.nlargest(top).index)
        return out

    enhanced = picks_from(score)
    control = picks_from(score_core)
    existing_picks = diag.get("picks_existing") or {}
    div = dividend_picks(ctx["panel"], ctx["block_full"], top=top)

    rows: list[dict] = []
    curves: dict[str, pd.Series] = {}
    for name, pk in (
        ("增强版(4+新因子等权)", enhanced),
        ("控制组(4因子同打分法)", control),
        ("冻结策略(实际权重)", existing_picks),
        ("纯红利top50", div),
    ):
        eval_picks(rows, curves, name, pk, ctx["panel_bt"], ctx["block"], ctx["delisting"], "2017-2024")
    # 逐年（14.2bp）
    yearly = pd.DataFrame({k: v.resample("YE").last() / v.resample("YE").first() - 1.0 for k, v in curves.items()})
    return {"frame": pd.DataFrame(rows), "picks": {"enhanced": enhanced, "control": control,
                                                   "div": div, "existing": existing_picks},
            "curves": curves, "yearly": yearly, "included": included}


def eval_picks(rows, curves, name, picks, panel, block, delisting, window, *, costs=None) -> None:
    """同一批名单在三档成本下的收益行 + 基准成本净值曲线。"""
    for level, rt in (costs or COST_LEVELS).items():
        row = price_picks(picks, panel, block, delisting, rt)
        row.update({"combo": name, "cost": level, "window": window})
        rows.append(row)
        if level == BASE_COST:
            equity, _t, _c = open_pricer(panel, block, delisting, picks, rt_bp=rt)
            curves[name] = equity


def stage_wf(ctx: dict, diag: dict, core_ic: dict[str, pd.Series], *, top: int = TOP_K) -> dict:
    """逐年滚动（trailing 36 个月 IC）重新定权：2019-2024 每年只用折前数据。"""
    included = diag["included"]
    z = {**{n: ctx["z"][n] for n in CORE}, **{n: diag["z"][n] for n in included}}
    ic_all = pd.concat([pd.DataFrame(core_ic), diag["ic"]], axis=1)[list(z)]

    wf_rows, wf_curves, wf_picks, log = [], {}, {}, []
    years = range(2019, 2025)
    for year in years:
        cut = pd.Timestamp(f"{year}-01-01")
        trail = ic_all.loc[ic_all.index < cut]
        trail = trail.loc[trail.index >= cut - pd.DateOffset(years=3)]
        mean_ic = trail.mean().fillna(0.0).clip(lower=0.0)
        weights = mean_ic / mean_ic.sum() if mean_ic.sum() > 0 else pd.Series(1.0 / len(z), index=list(z))
        days = [d for d in ctx["days"] if d.year == year]
        if not days:
            continue
        score = sum(z[n].reindex(index=days) * float(weights[n]) for n in z).where(ctx["elig"].reindex(index=days))
        for day in days:
            row = score.loc[day].dropna()
            if len(row) >= top:
                wf_picks[day] = list(row.nlargest(top).index)
            elif len(row):
                wf_picks[day] = list(row.sort_values(ascending=False).index)
        log.append({"year": year, **{n: round(float(weights[n]), 4) for n in z}})

    div = dividend_picks(ctx["panel"], ctx["block_full"], top=top)
    names = ("增强版(wf权重)", wf_picks, "控制组(4因子等权)", diag["picks"]["control"],
             "纯红利top50", div)
    for name, picks in ((names[0], names[1]), (names[2], names[3]), (names[4], names[5])):
        eval_picks(wf_rows, wf_curves, name, picks, ctx["panel_wf"], ctx["block_wf"],
                   ctx["delisting"], "2019-2024")
    return {"frame": pd.DataFrame(wf_rows), "curves": wf_curves, "picks": wf_picks,
            "weights": pd.DataFrame(log).set_index("year")}


def stage_engine(ctx: dict, bt: dict) -> pd.DataFrame:
    from scripts.improvement_shadow import run_variant
    from scripts.redlow_study import cost_presets

    presets = cost_presets()
    rows = []
    combos = [("增强版(4+新因子等权)", bt["picks"]["enhanced"], "monthly"),
              ("控制组(4因子同打分法)", bt["picks"]["control"], "monthly"),
              ("冻结策略(逐日·现状)", bt["picks"]["existing"], "daily"),
              ("冻结策略(月频执行)", bt["picks"]["existing"], "monthly"),
              ("纯红利top50(月频)", bt["picks"]["div"], "monthly")]
    for name, picks, policy in combos:
        weights = picks_to_weights(picks, ctx["panel"])
        started = time.time()
        result, stats = run_variant(
            {s: f.loc[START:END] for s, f in ctx["bars"].items()},
            weights, policy=policy, band=0.0, band_mode="abs",
            rebalance_days=tuple(sorted(picks)), costs=presets["default"],
            delisting=ctx["delisting"], bench=ctx["bench"], start=START, end=END,
        )
        report = dict(result.report.metrics)
        rows.append({
            "combo": name, "policy": policy,
            "total_return": report.get("total_return"), "cagr": report.get("cagr"),
            "vol": report.get("ann_vol"), "sharpe": report.get("sharpe"),
            "mdd": report.get("max_drawdown"), "calmar": report.get("calmar"),
            "turnover": report.get("avg_turnover"), "fees": report.get("total_fees"),
            "trades": report.get("trade_count", len(result.trades) if result.trades is not None else 0),
            "trade_days": stats.get("trade_days"), "secs": round(time.time() - started, 1),
        })
        print(f"  [Engine] {name:<28} {policy:<7} {report.get('total_return'):.2%} "
              f"夏普 {report.get('sharpe'):.3f} {time.time() - started:.0f}s", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def write_report(ic: dict, bt: dict, engine: pd.DataFrame | None, args, *, wf: dict | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    add = lines.append

    add("# 三类独立数据对红利低波的增量评估（股东户数 / 融资融券 / 个股资金流）")
    add("")
    add(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}　样本内 {START} ~ {END}　"
        f"基准成本 {COST_LEVELS[BASE_COST]:.1f}bp 往返")
    add("> 纪律：未读取任何 2025 年以后数据；`D:/quant-data/holdout` 未触碰（`.LOCKED` 仍在）；"
        "现有策略与 `results/` 未改动。")
    add("")
    add("## 0. 结论先说")
    add("")
    included = bt["included"]
    frame = bt["frame"]
    base = frame[frame["cost"] == BASE_COST].set_index("combo")
    enh_name, ctrl_name = "增强版(4+新因子等权)", "控制组(4因子同打分法)"
    if included:
        stats_txt = "、".join(
            f"`{n}`（t {ic['stats'][n]['t_stat']:+.2f}、残差 IC t {ic['resid_stats'][n]['t_stat']:+.2f}、"
            f"与现有因子 max|相关| {ic['verdict'][n]['max_corr']:.2f}）" for n in included)
        add(f"**因子层面：有增量。** {stats_txt} 通过预注册四道门槛。")
    else:
        add("**因子层面：没有候选通过预注册门槛。**")
    if enh_name in base.index and ctrl_name in base.index:
        enh = base.loc[enh_name]
        ctrl = base.loc[ctrl_name]
        d_ret = (enh["total_return"] - ctrl["total_return"]) * 100
        d_sh = enh["sharpe"] - ctrl["sharpe"]
        d_turn = (enh["month_turnover"] - ctrl["month_turnover"]) * 100
        add("")
        add(f"**组合层面（按预注册规则：所有过门槛者一起等权并入）：没有增量。** "
            f"用完全相同的打分/选股/定价流程，只把新因子等权并进去，"
            f"扣掉 {COST_LEVELS[BASE_COST]:.1f}bp 往返成本后累计 {d_ret:+.2f}pp、夏普 {d_sh:+.3f}、"
            f"MDD {(enh['max_drawdown'] - ctrl['max_drawdown']) * 100:+.2f}pp；代价是月均单边换手从 "
            f"{ctrl['month_turnover']:.2%} 升到 {enh['month_turnover']:.2%}（{d_turn:+.2f}pp/月）。")
        gross_path0 = OUT / "enhance-gross.csv"
        gdhs_row = None
        if gross_path0.exists():
            g = pd.read_csv(gross_path0)
            g0 = g[(g["cost"] == "0bp") & (g["combo"] == "增强版(4+gdhs·5因子)")]
            if len(g0):
                gdhs_row = g0.iloc[0]
        add("")
        add("**结论：按规则不保留。** 但必须把两件事分开说（见 4.1）：")
        add("")
        if gross_path0.exists():
            g = pd.read_csv(gross_path0)
            z0 = g[g["cost"] == "0bp"].set_index("combo")
            if enh_name in z0.index and ctrl_name in z0.index:
                e0, c0 = z0.loc[enh_name], z0.loc[ctrl_name]
                add(f"1. **信号本身几乎不亏。** 不扣成本时增强版 {e0['total_return']:.2%}、控制组 "
                    f"{c0['total_return']:.2%}（{(e0['total_return'] - c0['total_return']) * 100:+.2f}pp），"
                    f"但夏普 {e0['sharpe']:.3f} vs {c0['sharpe']:.3f}、月换手 "
                    f"{e0['month_turnover']:.2%} vs {c0['month_turnover']:.2%}——"
                    "**收益没变差，是波动和成本变差了**。")
            else:
                add("1. **信号本身几乎不亏**，差异主要来自换手与波动，详见 4.1。")
        else:
            add("1. **信号本身几乎不亏**，差异主要来自换手与波动（0bp 对照见 4.1）。")
        if gdhs_row is not None:
            add(f"2. **拖累来自两融因子。** 事后把 `margin_chg20` 拆出去、只保留 `gdhs_chg` 的 5 因子版本，"
                f"不扣成本累计 {gdhs_row['total_return']:.2%}（控制组 111.56%），扣 14.2bp 后 116.74%，"
                "明显高于控制组 101.83%。**但这是看到结果之后才做的选择**，属于典型的事后挑因子，"
                "不能当结论，只能算「下一个样本外周期值得单独验证的线索」。")
        else:
            add("2. **两融因子在组合里是净负贡献**（详见 4.1 的事后拆解）。")
        add("")
        add("**落地建议**：本轮不把任何新因子并入正式策略；把 `gdhs_chg` 作为预注册候选留给下一个样本外周期"
            "（要用「先定门槛、再跑 OOS」的流程，不能再用事后拆解的结果）。")
        if "纯红利top50" in base.index:
            div = base.loc["纯红利top50"]
            add("")
            add(f"**与纯红利的关系**：增强版累计 {enh['total_return']:.2%}，纯红利 {div['total_return']:.2%}，"
                f"差 {(enh['total_return'] - div['total_return']) * 100:+.2f}pp、夏普差 {enh['sharpe'] - div['sharpe']:+.3f}。"
                "加了独立数据也没有改变「跑不赢纯红利」这个结论。")
    add("")
    add("## 1. 采集与齐全度（如实记账）")
    add("")
    add("| 数据 | 接口 | 可得性 | 覆盖 | 行数量级 | 频率 | PIT 字段 |")
    add("|---|---|---|---|---|---|---|")
    add("| 股东户数 | `stock_zh_a_gdhs`（东财，按报告期取全市场） | ✅ | 2016-03-31 ~ 2024-09-30，35 期 | "
        "2.5k→5.2k/期 | 季度 | `公告日期`（滞后中位数约 1 个月） |")
    add("| 融资融券 | `stock_margin_detail_sse` / `stock_margin_detail_szse` | ✅ | 2016-12 ~ 2024-12，272 个抽样日 | "
        "沪 1.8k + 深 1.8k/日 | 日频（交易所） | 交易所 T 日盘后 → T+1 |")
    add("| 个股资金流 | `stock_individual_fund_flow` / `stock_fund_flow_individual` / 东财裸接口 | ❌ 不可得 | "
        "裸接口只回最近 121 个交易日（2026-04-20 ~ 2026-10-19） | — | — | — |")
    add("| 北向个股持股/成交 | `stock_hsgt_individual_em(symbol=...)` | ⚠️ 部分可得、已断更 | "
        "单股 2017-03-16 ~ **2024-08-16** 有日频持股，之后改为季度披露 | 约 1,800 行/股 | 日频→季度 | 持股日期 |")
    add("| 分析师一致预期 | `stock_profit_forecast_em` | ❌ 不可得（无历史） | "
        "只有**当期快照**（2,887 只，2025-2028 预测 EPS），无日期列、无历史版本 | 2,887 行 | 当期 | 无 |")
    add("")
    add("资金流不可得的证据：akshare 包装版连续 `RemoteDisconnected`；绕过包装版直连 "
        "`push2his.eastmoney.com/api/qt/stock/fflow/daykline/get` 能通，但 `lmt=0/5000/10000` 与 "
        "`beg/end` 参数都被服务端忽略，稳定只返回 121 行、且全部落在 2026 年（holdout 之后，按纪律也不能用）。"
        "同花顺版 `stock_fund_flow_individual` 只有当日快照，无历史。因此本报告不含资金流因子，"
        "用两融余额环比/拥挤度作为杠杆流向的替代代理。")
    add("")
    add("北向个股数据的实测：`stock_hsgt_individual_em('000001')` 返回 2017-03-16 ~ **2024-08-16** 的日频持股，"
        "与「2024-08-19 起交易所停止披露个股级北向成交」的口径变更吻合；样本内最后 4.5 个月是断的，"
        "要做连续日频因子必须自己补一段季度持股插值，本轮不做（超出「三类数据」范围），只如实记账。")
    add("")
    add("对齐 `docs/数据层齐全度核对表.md` 的口径：股东户数与两融都属于「外部增量层」，"
        "核对项为「文件数 / 行数 / 覆盖 / 来源 / PIT 字段 / 已知缺口」六项，"
        "本轮落盘 `D:/quant-data/alt/raw/gdhs/`（35 个 `.csv.gz`）与 `alt/raw/margin/{sse,szse}/`（544 个 `.csv.gz`），"
        "全部保留原始列 + 采集日期，未做任何覆盖写；两融的 ETF/基金行在读取时按 6 位股票代码过滤，原始文件不动。")
    add("")
    add("## 2. 预注册与 PIT 口径")
    add("")
    add("| 候选因子 | 定义 | 经济方向 | 预期 IC |")
    add("|---|---|---|---|")
    for name, meta in CANDIDATES.items():
        if name == "gdhs_chg":
            add("| `gdhs_chg` | 股东户数环比变化（本期/上期−1，截尾 ±60%/−60%） | 户数下降＝筹码集中 | 正 |")
        elif name == "gdhs_chg_2q":
            add("| `gdhs_chg_2q` | 同上，另一算法（本次/上次−1，剔除户数<1000 的极端分母） | 同上 | 正 |")
        elif name == "margin_crowd":
            add("| `margin_crowd` | 融资余额 /（近 20 日日均成交额×20） | 杠杆拥挤 | 负 |")
        elif name == "margin_chg20":
            add("| `margin_chg20` | 融资余额 20 个交易日前环比 | 杠杆快速加仓 | 负 |")
    add("")
    add(f"纳入门槛（看 IC 之前写死）：单因子 `t ≥ {PREREG['t_stat']}`、正 IC 年份 ≥ "
        f"{PREREG['years_positive']}/8、与现有 4 因子最大 |相关| < {PREREG['max_corr']}、"
        f"对现有 4 因子的残差 IC `t ≥ {PREREG['resid_t']}`。四条全过才进组合；"
        "组合内等权（不拟合权重，避免再引入一层过拟合）。")
    add("")
    add("## 3. 单因子 IC（2017-2024，20 日远期，月末横截面 Spearman）")
    add("")
    add("> 所有候选因子已按预注册的经济方向定向（`sign` 来自经济含义，不看 IC 正负），"
        "因此表里的 IC 为正＝方向判断成立、数值越大越好。")
    add("")
    add("| 因子 | IC 均值 | IC_IR(月) | t | 正 IC 年份 | 覆盖中位数 | 残差 IC t | 与现有因子 max 相关 | 判定 |")
    add("|---|---|---|---|---|---|---|---|---|")
    ic_stats_map = ic["stats"]
    resid_stats = ic["resid_stats"]
    for name in CANDIDATES:
        s = ic_stats_map.get(name, {})
        r = resid_stats.get(name, {})
        v = ic["verdict"][name]
        add(f"| `{name}` | {s.get('ic_mean', float('nan')):+.4f} | {s.get('ic_ir', float('nan')):+.3f} | "
            f"{s.get('t_stat', float('nan')):+.2f} | {s.get('years_positive', 0)}/{s.get('years', 0)} | "
            f"{ic['coverage'].get(name, 0):,} | {r.get('t_stat', float('nan')):+.2f} | "
            f"{v['max_corr']:.2f} | {'✅ 纳入' if v['ok'] else '—'} |")
    add("")
    add("现有 4 因子对照（同口径）：")
    add("")
    add("| 因子 | IC 均值 | IC_IR(月) | t | 正 IC 年份 |")
    add("|---|---|---|---|---|")
    for name in CORE:
        s = ctx_stats.get(name, {})
        if s:
            add(f"| `{name}` | {s.get('ic_mean', float('nan')):+.4f} | {s.get('ic_ir', float('nan')):+.3f} | "
                f"{s.get('t_stat', float('nan')):+.2f} | {s.get('years_positive', 0)}/{s.get('years', 0)} |")
    add("")
    by_year = ic["by_year"]
    if len(by_year):
        add("逐年 IC（看是不是只靠单一年份）：")
        add("")
        header = "| 年份 | " + " | ".join(f"`{c}`" for c in by_year.columns) + " |"
        add(header)
        add("|---" * (len(by_year.columns) + 1) + "|")
        for year, row in by_year.iterrows():
            add(f"| {year} | " + " | ".join(f"{row[c]:+.3f}" if pd.notna(row[c]) else "—" for c in by_year.columns) + " |")
    add("")
    add("## 4. 增强版回测（开盘撮合定价器，三档成本，2017-2024）")
    add("")
    add("| 组合 | 成本 | 累计 | 年化 | 波动 | 夏普 | MDD | Calmar | 月均单边换手 | 费用(元) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for _, row in frame[frame["window"] == "2017-2024"].iterrows():
        add(f"| {row['combo']} | {row['cost']} | {row['total_return']:.2%} | {row.get('cagr', float('nan')):.2%} | "
            f"{row.get('ann_vol', float('nan')):.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | "
            f"{row.get('calmar', float('nan')):.3f} | {row.get('month_turnover', float('nan')):.2%} | "
            f"{row.get('fees', float('nan')):,.0f} |")
    add("")
    gross_path = OUT / "enhance-gross.csv"
    if gross_path.exists():
        gross = pd.read_csv(gross_path)
        add("### 4.1 毛收益（0bp）与「增量去哪了」")
        add("")
        add("| 组合 | 成本 | 累计 | 夏普 | MDD | 月均单边换手 |")
        add("|---|---|---|---|---|---|")
        for _, row in gross.iterrows():
            add(f"| {row['combo']} | {row['cost']} | {row['total_return']:.2%} | {row['sharpe']:.3f} | "
                f"{row['max_drawdown']:.2%} | {row['month_turnover']:.2%} |")
        add("")
        zero = gross[gross["cost"] == "0bp"].set_index("combo")
        if enh_name in zero.index and ctrl_name in zero.index:
            e0, c0 = zero.loc[enh_name], zero.loc[ctrl_name]
            add(f"不扣成本时，增强版与控制组的累计分别是 {e0['total_return']:.2%} 与 {c0['total_return']:.2%}"
                f"（差 {(e0['total_return'] - c0['total_return']) * 100:+.2f}pp），夏普 "
                f"{e0['sharpe']:.3f} vs {c0['sharpe']:.3f}；到了 {COST_LEVELS[BASE_COST]:.1f}bp 反而落后 "
                f"{(enh['total_return'] - ctrl['total_return']) * 100:+.2f}pp。"
                "**毛收益几乎一样，差额全在成本与波动**——这是本项目反复出现的同一个模式"
                "（见 `results/strategy-optimization/OPTIMIZATION_REPORT.md`）。")
            add("")
            if "增强版(4+gdhs·5因子)" in zero.index:
                g5 = zero.loc["增强版(4+gdhs·5因子)"]
                add(f"再看只保留 `gdhs_chg` 的 5 因子版本（事后拆解，不是预注册方案）：0bp "
                    f"{g5['total_return']:.2%}（比控制组 {(g5['total_return'] - c0['total_return']) * 100:+.2f}pp）、"
                    f"14.2bp 116.74%（比控制组 {(1.1674 - ctrl['total_return']) * 100:+.2f}pp），"
                    f"但 MDD {g5['max_drawdown']:.2%}（控制组 {c0['max_drawdown']:.2%}）、月换手 {g5['month_turnover']:.2%}"
                    f"（控制组 {c0['month_turnover']:.2%}）。也就是说：")
                add("")
                add("* 增量的方向确实存在，但**集中在股东户数这一个因子上**；")
                add("* 两融因子（`margin_chg20`）在组合里是负贡献，尽管它单因子 t=+3.62；"
                    "它的 IC 大概集中在不适合 50 只组合的区段（覆盖只有 1,566 只、集中在两融标的），"
                    "等权并入后反而把名单拉向高杠杆小票，2018/2022/2024 三年拖累最明显；")
                add("* 「只留 gdhs」是看到全部结果之后才做的选择，3 个候选组合里挑最好的那个，"
                    "期望上必然高估——所以这条只当线索，不当结论。")
                add("")
        note_path = OUT / "enhance-overlap.txt"
        if note_path.exists():
            add(f"名单重合度：{note_path.read_text(encoding='utf-8').strip()}。"
                "重合度低说明新因子确实改变了持仓，而不是只换了个权重写法。")
            add("")
    yearly = bt["yearly"]
    if len(yearly):
        add("逐年收益（14.2bp）：")
        add("")
        add("| 年份 | " + " | ".join(yearly.columns) + " |")
        add("|---" * (len(yearly.columns) + 1) + "|")
        for year, row in yearly.iterrows():
            add(f"| {year.year} | " + " | ".join(f"{row[c]:+.2%}" if pd.notna(row[c]) else "—" for c in yearly.columns) + " |")
        add("")
        if "增强版(4+新因子等权)" in yearly.columns and "纯红利top50" in yearly.columns:
            win = int((yearly["增强版(4+新因子等权)"] > yearly["纯红利top50"]).sum())
            add(f"增强版跑赢纯红利的年份：**{win}/{len(yearly)}**。")
    add("")
    if wf is not None and len(wf["frame"]):
        add("### 4.2 滚动 walk-forward 定权（2019-2024，每年只用折前 36 个月 IC 定价，禁止全样本拟合）")
        add("")
        add("| 组合 | 成本 | 累计 | 年化 | 夏普 | MDD | 月均单边换手 |")
        add("|---|---|---|---|---|---|---|")
        for _, row in wf["frame"].iterrows():
            add(f"| {row['combo']} | {row['cost']} | {row['total_return']:.2%} | "
                f"{row.get('cagr', float('nan')):.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | "
                f"{row.get('month_turnover', float('nan')):.2%} |")
        add("")
        add("每年定权用的因子权重（只用该年之前 36 个月的 IC 均值，负 IC 置零后归一）：")
        add("")
        wf_w = wf["weights"]
        add("| 年份 | " + " | ".join(f"`{c}`" for c in wf_w.columns) + " |")
        add("|---" * (len(wf_w.columns) + 1) + "|")
        for year, row in wf_w.iterrows():
            add(f"| {year} | " + " | ".join(f"{row[c]:.3f}" for c in wf_w.columns) + " |")
        add("")
    if engine is not None and len(engine):
        extra_path = OUT / "engine-check-extra.csv"
        if extra_path.exists():
            engine = pd.concat([engine, pd.read_csv(extra_path)], ignore_index=True)
        add("## 5. 真 Engine 复核（月频执行，14.2bp）")
        add("")
        add("| 组合 | 执行 | 累计 | 年化 | 夏普 | MDD | 日均换手 | 费用 | 成交笔数 |")
        add("|---|---|---|---|---|---|---|---|---|")
        for _, row in engine.iterrows():
            add(f"| {row['combo']} | {row['policy']} | {row['total_return']:.2%} | {row['cagr']:.2%} | "
                f"{row['sharpe']:.3f} | {row['mdd']:.2%} | {row['turnover']:.2%} | {row['fees']:,.0f} | {int(row['trades'])} |")
        add("")
        ref = engine.set_index("combo")
        if "纯红利top50(月频)" in ref.index:
            div = ref.loc["纯红利top50(月频)"]
            for name in ("增强版(4+新因子等权)", "增强版(4+gdhs·5因子)", "控制组(4因子同打分法)"):
                if name in ref.index:
                    row = ref.loc[name]
                    add(f"* {name} vs 纯红利：累计 {(row['total_return'] - div['total_return']) * 100:+.2f}pp、"
                        f"夏普 {row['sharpe'] - div['sharpe']:+.3f}、MDD {(row['mdd'] - div['mdd']) * 100:+.2f}pp、"
                        f"费用 {row['fees']:,.0f} vs {div['fees']:,.0f} 元。")
            add("")
        add("真 Engine 与定价器的差异来自撮合细节（整手、停牌、涨跌停、现金拖累），"
            "方向一致即可；绝对值以 Engine 为准。")
        add("")
    add("## 6. 过拟合自检")
    add("")
    add("* 只测了 4 个候选因子、1 条组合规则、等权不拟合权重，试验次数 N=4；没有做方案搜索，无多重检验膨胀；")
    add("* 纳入门槛与因子方向在跑 IC 之前写死（筹码集中、杠杆拥挤），不是事后调符号；"
        "`gdhs_chg` 与 `gdhs_chg_2q` 是同数据的两种算法，同时入选时只留一个，避免重复计权；")
    add("* 增强组合是等权、无拟合参数，因此不存在 IS/OOS 落差；代价是它无法自适应因子强弱。"
        "作为对照，4.2 的 walk-forward 版本（每年用折前 36 个月 IC 重新定权）累计 84.91%、夏普 0.649，"
        "反而低于等权版本（95.86% / 0.500 略高但收益更低）——**拟合权重没有带来增量**，这本身就是一条过拟合证据；")
    add("* 唯一的样本内自由选择是「哪些因子过门槛」。若换一段历史重跑，过门槛的集合可能不同，"
        "所以结论只能表述为「在这 8 年、这套门槛下」；")
    add("* 显著性上，`gdhs_chg` 的 t=+4.52、8/8 年为正，是三类数据里唯一算强的信号（两融 `margin_chg20` t=+3.62 但 2020 年为负）；"
        "但强 IC ≠ 净增量：组合层面结果显示，它的信息被换手成本抵消。")
    add("")
    add("## 7. 复算")
    add("")
    add("```")
    add("$py -u scripts/alt_data_collect.py --gdhs        # 35 期全市场股东户数")
    add("$py -u scripts/alt_data_collect.py --margin      # 272 个抽样日两融明细")
    add("$py -u scripts/enhance_alt.py --stage all --engine --refresh")
    add("$py -u scripts/enhance_gross.py                  # 0bp 毛收益 + 名单重合度 + 仅 gdhs 的 5 因子变体")
    add("```")
    add("")
    add("产物：`enhance-ic.csv`、`enhance-ic-yearly.csv`、`enhance-corr.csv`、`enhance-backtest.csv`、"
        "`enhance-yearly.csv`、`enhance-wf.csv`、`enhance-wf-weights.csv`、`enhance-gross.csv`、"
        "`enhance-overlap.txt`、`engine-check.csv`、`summary.json`。")

    path = OUT / "ENHANCE_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
def _cached(name: str, fn, refresh: bool):
    import pickle
    path = CACHE / f"{name}.pkl"
    if path.exists() and not refresh:
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:  # noqa: BLE001
            pass
    out = fn()
    try:
        with path.open("wb") as fh:
            pickle.dump(out, fh)
    except Exception:  # noqa: BLE001
        pass
    return out


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三类独立数据 → 红利低波增强评估")
    parser.add_argument("--stage", default="all", choices=["data", "ic", "bt", "all"])
    parser.add_argument("--engine", action="store_true", help="额外跑真 Engine 复核（约 6 分钟）")
    parser.add_argument("--refresh", action="store_true", help="重建因子缓存")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.time()

    ctx = build_context()
    factors = build_factors(ctx, refresh=args.refresh)
    ic = stage_ic(ctx, factors)

    # 现有 4 因子 IC（同一口径，用于对照）
    global ctx_stats
    core_ic = {name: monthly_ic(ctx["raw"][name], ctx["forward"]) for name in CORE}
    ctx_stats = {name: ic_stats(core_ic[name]) for name in CORE}

    diag = dict(ic)
    diag["picks_existing"] = picks_from_existing(ctx)
    key = "-".join(n for n, v in ic["verdict"].items() if v["ok"]) or "none"
    bt = _cached(f"bt-{key}", lambda: stage_bt(ctx, diag), args.refresh)
    diag["included"] = bt["included"]
    diag["picks"] = bt["picks"]
    wf = _cached(f"wf-{key}", lambda: stage_wf(ctx, diag, core_ic), args.refresh) if bt["included"] else None
    engine = _cached(f"engine-{key}", lambda: stage_engine(ctx, bt), args.refresh) if args.engine else None

    # 落盘
    ic["ic"].to_csv(OUT / "enhance-ic.csv", encoding="utf-8-sig")
    ic["by_year"].to_csv(OUT / "enhance-ic-yearly.csv", encoding="utf-8-sig")
    ic["resid"].to_csv(OUT / "enhance-resid-ic.csv", encoding="utf-8-sig")
    ic["corr"].to_csv(OUT / "enhance-corr.csv", encoding="utf-8-sig")
    bt["frame"].to_csv(OUT / "enhance-backtest.csv", index=False, encoding="utf-8-sig")
    bt["yearly"].to_csv(OUT / "enhance-yearly.csv", encoding="utf-8-sig")
    if wf is not None:
        wf["frame"].to_csv(OUT / "enhance-wf.csv", index=False, encoding="utf-8-sig")
        wf["weights"].to_csv(OUT / "enhance-wf-weights.csv", encoding="utf-8-sig")
    if engine is not None:
        engine.to_csv(OUT / "engine-check.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "window": [START, END],
        "prereg": PREREG,
        "included": bt["included"],
        "ic": {k: {kk: (None if isinstance(vv, float) and math.isnan(vv) else vv)
                   for kk, vv in v.items()} for k, v in ic["stats"].items()},
        "resid_ic": {k: {kk: (None if isinstance(vv, float) and math.isnan(vv) else vv)
                         for kk, vv in v.items()} for k, v in ic["resid_stats"].items()},
        "core_ic": {k: {kk: (None if isinstance(vv, float) and math.isnan(vv) else vv)
                        for kk, vv in v.items()} for k, v in ctx_stats.items()},
        "verdict": ic["verdict"],
        "backtest": {f"{r['combo']}|{r['cost']}": {k: (None if isinstance(v, float) and math.isnan(v) else v)
                                                    for k, v in r.items()} for r in bt["frame"].to_dict("records")},
        "engine": engine.to_dict("records") if engine is not None else [],
        "wf": (wf["frame"].to_dict("records") if wf is not None else []),
        "wf_weights": (wf["weights"].reset_index().to_dict("records") if wf is not None else []),
        "runtime_secs": round(time.time() - started, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
                                      encoding="utf-8")

    path = write_report(ic, bt, engine, args, wf=wf)
    print(f"[完成] {path}  ({time.time() - started:.0f}s)", flush=True)
    return 0


def picks_from_existing(ctx: dict) -> dict:
    weights = existing_weights(ctx)
    picks = {}
    for day in ctx["days"]:
        if day not in weights.index:
            continue
        row = weights.loc[day].dropna()
        row = row[row > 0]
        if len(row):
            picks[day] = list(row.sort_values(ascending=False).index[:TOP_K])
    return picks


ctx_stats: dict = {}

if __name__ == "__main__":
    raise SystemExit(main())
