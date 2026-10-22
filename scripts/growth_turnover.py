"""换手 / 稳定性专项：因子稳定性诊断、信号时序平滑、分批建仓、行业口径消融。

设计约束（与 docs 里的预注册一致）：
- 只读 2017-2024 主样本（必要时 2008-2016 历史样本）；2025-01-01 起 holdout 不读。
- 不改 quant/ 核心；复用 scripts.growth_robust / growth_risk / growth_backtest 的现成件。
- 新参数事前定死：平滑半衰期 2 个月、分批 3 期、分批触发门槛单边 20%、细分大类上限 25%。

用法：
    python -X utf8 -m scripts.growth_turnover --diag
    python -X utf8 -m scripts.growth_turnover --runs base,smooth,phased,ind_none,ind_sub25,combo
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant import refdata
from quant.factors import growth

OUT = Path("results/growth/stability")

# --------------------------------------------------------------------------- #
# 事前登记的新参数（不扫、不改）
# --------------------------------------------------------------------------- #
SMOOTH_HALFLIFE_MONTHS = 2.0     # 信号平滑：指数加权，半衰期 2 个月
PHASED_TRANCHES = 3              # 分批建仓：最多 3 期
PHASED_GAP_TRIGGER = 0.20        # 触发门槛：单边（0.5·Σ|Δw|）> 20% 才分批
SUBSECTOR_CAP = 0.25             # 细分大类（证监会行业前 3 位）单个上限 25%
MIN_NAMES = 20


# --------------------------------------------------------------------------- #
# 行业 PIT（门类 / 细分大类）
# --------------------------------------------------------------------------- #
def industry_panels(marks: pd.DatetimeIndex, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (门类, 细分大类) 两张 PIT 矩阵；缺失记 UNKNOWN。

    门类 = 证监会行业名首字符（C/I/F…），细分大类 = 前 3 位（C39/C27…）。
    ``ref/industry.csv`` 是 13 个年度版本快照，按 ``as_of <= day`` 取最近一版。
    """
    frame = refdata.load_industry()
    industry = frame["industry"].astype("string").fillna("UNKNOWN")
    frame = frame.assign(level1=industry.str[:1], level3=industry.str[:3])
    versions = sorted(frame["as_of"].astype(str).unique())

    level1 = pd.DataFrame("UNKNOWN", index=marks, columns=symbols, dtype=object)
    level3 = pd.DataFrame("UNKNOWN", index=marks, columns=symbols, dtype=object)
    stamps = np.array([str(day.date()) for day in marks])
    for idx, version in enumerate(versions):
        start = pd.Timestamp(version)
        end = pd.Timestamp(versions[idx + 1]) if idx + 1 < len(versions) else None
        hit = (marks >= start) & ((marks < end) if end is not None else True)
        if not hit.any():
            continue
        snap = frame[frame["as_of"].astype(str) == version]
        codes = snap["symbol"].astype(str).str.zfill(6)
        one = pd.Series(snap["level1"].to_numpy(), index=codes).groupby(level=0).last()
        three = pd.Series(snap["level3"].to_numpy(), index=codes).groupby(level=0).last()
        wanted = [str(symbol).split(".")[0].zfill(6) for symbol in symbols]
        level1.loc[marks[hit], :] = one.reindex(wanted).fillna("UNKNOWN").to_numpy()
        level3.loc[marks[hit], :] = three.reindex(wanted).fillna("UNKNOWN").to_numpy()
    _ = stamps
    return level1, level3


# --------------------------------------------------------------------------- #
# 诊断：因子 / 块 的月度排名稳定性
# --------------------------------------------------------------------------- #
def monthly_rank(mat: pd.DataFrame, eligible: pd.DataFrame,
                 marks: pd.DatetimeIndex) -> pd.DataFrame:
    """月末截面、在合格集内的排名百分位（0–1，越大越好）。"""
    out = pd.DataFrame(np.nan, index=marks, columns=mat.columns, dtype="float64")
    for day in marks:
        if day not in mat.index:
            continue
        row = mat.loc[day].where(eligible.loc[day].reindex(mat.columns).fillna(False))
        row = row.replace([np.inf, -np.inf], np.nan).dropna()
        if len(row) >= MIN_NAMES:
            out.loc[day] = row.rank(pct=True)
    return out


def _top_set(rank_row: pd.Series, k: int) -> set[str]:
    row = rank_row.dropna()
    if row.empty:
        return set()
    return set(row.nlargest(min(k, len(row))).index)


def stability_stats(rank: pd.DataFrame, *, ks: tuple[int, ...] = (50, 100)) -> dict:
    """相邻两月的截面秩相关 + topN 进出只数。"""
    corrs: list[float] = []
    churn: dict[int, list[float]] = {k: [] for k in ks}
    days = list(rank.index)
    for prev, cur in zip(days[:-1], days[1:]):
        a, b = rank.loc[prev], rank.loc[cur]
        both = a.notna() & b.notna()
        if both.sum() < MIN_NAMES:
            corrs.append(float("nan"))
            for k in ks:
                churn[k].append(float("nan"))
            continue
        corrs.append(float(np.corrcoef(a[both].to_numpy(), b[both].to_numpy())[0, 1]))
        for k in ks:
            prev_top, cur_top = _top_set(a, k), _top_set(b, k)
            churn[k].append(len(cur_top - prev_top) / max(len(prev_top), 1))
    arr = np.array(corrs, dtype="float64")
    out = {
        "months": int(np.isfinite(arr).sum()),
        "rank_corr_mean": float(np.nanmean(arr)),
        "rank_corr_median": float(np.nanmedian(arr)),
        "rank_corr_min": float(np.nanmin(arr)),
        "rank_corr_p10": float(np.nanpercentile(arr, 10)),
    }
    for k in ks:
        out[f"churn_top{k}"] = float(np.nanmean(np.array(churn[k], dtype="float64")))
    out["series"] = corrs
    return out


def diagnosis(prepared: dict) -> dict:
    from scripts import growth_backtest as gb
    from scripts import growth_risk as gr

    facts, alt = prepared["facts"], prepared["alt"]
    eligible, marks = prepared["eligible"], prepared["marks"]

    factor_names: list[str] = []
    for names in gb.BLOCKS_QN.values():
        factor_names.extend(names)

    daily: dict[str, pd.DataFrame] = {}
    for name in factor_names:
        if hasattr(facts, name):
            daily[name] = getattr(facts, name)()
        elif name in alt:
            daily[name] = alt[name]
        else:
            print(f"[diag] 缺因子 {name}", flush=True)
    rank = {name: monthly_rank(mat, eligible, marks) for name, mat in daily.items()}

    rows = []
    for name in factor_names:
        if name not in rank:
            continue
        stats = stability_stats(rank[name])
        block = next(b for b, names in gb.BLOCKS_QN.items() if name in names)
        rows.append({"factor": name, "block": block,
                     **{k: v for k, v in stats.items() if k != "series"}})
    factor_table = pd.DataFrame(rows).sort_values("rank_corr_mean", ascending=False)

    block_rank: dict[str, pd.DataFrame] = {}
    for block in gb.BLOCKS_QN:
        single = grb_composite_single(prepared, block)
        block_rank[block] = monthly_rank(single, eligible, marks)
    block_rows = []
    weights = prepared["blocks"]
    for block, rk in block_rank.items():
        stats = stability_stats(rk)
        block_rows.append({"block": block, "weight": growth.BLOCK_WEIGHTS.get(block),
                           **{k: v for k, v in stats.items() if k != "series"}})
    block_table = pd.DataFrame(block_rows).sort_values("rank_corr_mean", ascending=False)

    total_rank = monthly_rank(prepared["score"], eligible, marks)
    total_stats = stability_stats(total_rank)

    # 基线持仓与剧烈换血月份
    strategy = gr.BufferTopK(prepared["score"], eligible, sector=prepared["sector"],
                             top_k=50, gross=0.95, max_weight=0.04, sector_cap=0.30,
                             buffer_rank=gr.BUFFER_RANK)
    weights = strategy.generate(prepared["bars"])
    holdings = {day: set(weights.loc[day][weights.loc[day] > 0].index) for day in marks}
    overlap_rows = []
    for prev, cur in zip(marks[:-1], marks[1:]):
        a, b = holdings[prev], holdings[cur]
        overlap = len(a & b) / max(len(a), 1)
        overlap_rows.append({"month": str(cur.date()), "overlap": overlap,
                             "total_corr": total_stats["series"][list(marks).index(cur) - 1]})
    overlap_table = pd.DataFrame(overlap_rows)

    # 剧烈月份归因
    violent = overlap_table[overlap_table["overlap"] < 0.30].copy()
    attribution: list[dict] = []
    for _, row in violent.iterrows():
        month = pd.Timestamp(row["month"])
        pos = list(marks).index(month)
        prev = marks[pos - 1]
        blocks_corr = {}
        for block, rk in block_rank.items():
            a, b = rk.loc[prev], rk.loc[month]
            both = a.notna() & b.notna()
            blocks_corr[block] = float(np.corrcoef(a[both], b[both])[0, 1]) if both.sum() >= MIN_NAMES else float("nan")
        factor_corr = {}
        for name, rk in rank.items():
            a, b = rk.loc[prev], rk.loc[month]
            both = a.notna() & b.notna()
            factor_corr[name] = float(np.corrcoef(a[both], b[both])[0, 1]) if both.sum() >= MIN_NAMES else float("nan")
        worst_block = min(blocks_corr, key=lambda k: blocks_corr[k] if np.isfinite(blocks_corr[k]) else 9)
        worst_factors = sorted([(k, v) for k, v in factor_corr.items() if np.isfinite(v)],
                               key=lambda kv: kv[1])[:3]
        attribution.append({"month": str(month.date()), "overlap": float(row["overlap"]),
                            "total_corr": float(row["total_corr"]),
                            "worst_block": worst_block,
                            "worst_block_corr": blocks_corr[worst_block],
                            "block_corr": {k: float(v) for k, v in blocks_corr.items()},
                            "worst_factors": [[k, float(v)] for k, v in worst_factors]})

    payload = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "sample": prepared["sample"],
        "marks": [str(day.date()) for day in marks],
        "factor_table": factor_table.to_dict(orient="records"),
        "block_table": block_table.to_dict(orient="records"),
        "total": {k: v for k, v in total_stats.items() if k != "series"},
        "overlap_mean": float(overlap_table["overlap"].mean()),
        "overlap_median": float(overlap_table["overlap"].median()),
        "violent_months": int(len(violent)),
        "violent_share": float(len(violent) / max(len(overlap_table), 1)),
        "attribution": attribution,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnosis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                                   default=str), encoding="utf-8")
    factor_table.to_csv(OUT / "factor-stability.csv", index=False, encoding="utf-8-sig")
    block_table.to_csv(OUT / "block-stability.csv", index=False, encoding="utf-8-sig")
    overlap_table.to_csv(OUT / "monthly-overlap.csv", index=False, encoding="utf-8-sig")
    _write_diagnosis_md(payload, factor_table, block_table)
    return payload


def grb_composite_single(prepared: dict, block: str) -> pd.DataFrame:
    from scripts import growth_robust as grb_impl
    return grb_impl.composite_score_single(prepared, block)


# --------------------------------------------------------------------------- #
# 变体回测：平滑 / 分批 / 行业口径
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    smooth: bool = False
    sector_level: str | None = "level1"
    sector_cap: float = 0.30
    phased: bool = False
    buffer_rank: int = 60
    growth_g4_weight: float | None = None   # 仅用于「权重结论验证」诊断，不作为新规格
    min_trade: float = 5000.0


VARIANTS: dict[str, Variant] = {
    "base": Variant("base", "基线（v1 legacy5k_slip）"),
    "smooth": Variant("smooth", "信号平滑（半衰期 2 月）", smooth=True),
    "phased": Variant("phased", "分批建仓（3 期·门槛 20%）", phased=True),
    "smooth_phased": Variant("smooth_phased", "平滑 + 分批", smooth=True, phased=True),
    "ind_none": Variant("ind_none", "不设行业上限", sector_level=None, sector_cap=1.0),
    "ind_sub25": Variant("ind_sub25", "细分大类上限 25%", sector_level="level3", sector_cap=SUBSECTOR_CAP),
    "combo": Variant("combo", "平滑 + 分批 + 细分大类 25%",
                     smooth=True, sector_level="level3", sector_cap=SUBSECTOR_CAP, phased=True),
    "buf80": Variant("buf80", "缓冲带 80", buffer_rank=80),
    "buf100": Variant("buf100", "缓冲带 100", buffer_rank=100),
    "buf100_phased": Variant("buf100_phased", "缓冲带 100 + 分批", buffer_rank=100, phased=True),
    "buf100_sub25": Variant("buf100_sub25", "缓冲带 100 + 细分大类 25%",
                            buffer_rank=100, sector_level="level3", sector_cap=SUBSECTOR_CAP),
    "g4_08": Variant("g4_08", "诊断：g4 权重 20%→8%（结论验证，非新规格）",
                     growth_g4_weight=0.20),
    "g4_04": Variant("g4_04", "诊断：g4 权重 20%→4%（结论验证，非新规格）",
                     growth_g4_weight=0.10),
    "combo6": Variant("combo6", "组合（缓冲带100+门类25%+最小单1万）",
                      buffer_rank=100, sector_cap=0.25, min_trade=10000.0),
}


def _reweighted_score(prepared: dict, w_g4: float) -> pd.DataFrame:
    """只改成长块内部权重（g4 占 w_g4，e1 占 1−w_g4），其余块与总权重不变。"""
    from quant.factors import growth as growth_impl
    from scripts import growth_robust as grb_impl

    facts = prepared["facts"]
    eligible = prepared["eligible"]
    r_g4 = growth_impl._rank_pct(facts.g4_eav(), eligible, min_names=MIN_NAMES)
    r_e1 = growth_impl._rank_pct(facts.e1_contract_yoy(), eligible, min_names=MIN_NAMES)
    growth_block = w_g4 * r_g4 + (1.0 - w_g4) * r_e1
    parts = [(0.40, growth_block),
             (0.25, grb_impl.composite_score_single(prepared, "quality")),
             (0.15, grb_impl.composite_score_single(prepared, "rnd")),
             (0.20, grb_impl.composite_score_single(prepared, "aux"))]
    num = sum(weight * part.fillna(0.0) for weight, part in parts)
    den = sum(weight * part.notna().astype("float64") for weight, part in parts)
    return num / den.where(den > 0)


def _smoothed_score(prepared: dict) -> pd.DataFrame:
    """月末采样后做指数加权平滑（半衰期 2 个月），再写回月末。"""
    score = prepared["score"]
    marks = prepared["marks"]
    base = score.loc[marks]
    smoothed = base.ewm(halflife=SMOOTH_HALFLIFE_MONTHS, min_periods=1, adjust=True).mean()
    out = score.astype("float64").copy()
    out.loc[marks, :] = smoothed.to_numpy()
    return out


def _phased_weights(prepared: dict, target: pd.DataFrame) -> pd.DataFrame:
    """把「一步到位」的月末目标改成「不足时分三期补」的月末目标。

    分期规则（事前登记）：单边缺口 0.5·Σ|T−D| > 20% 且已分 <3 期时，本次只补 1/3；
    否则一次到位并重置计数。D 为上一期已部署权重（现金不参与）。
    """
    marks = prepared["marks"]
    columns = target.columns
    deployed = pd.Series(0.0, index=columns)
    rows: dict[pd.Timestamp, pd.Series] = {}
    tranches = 0
    for day in marks:
        goal = target.loc[day].reindex(columns).fillna(0.0)
        gap = goal - deployed
        one_way = float(gap.abs().sum()) / 2.0
        if one_way > PHASED_GAP_TRIGGER and tranches < PHASED_TRANCHES - 1:
            deployed = deployed + gap / (PHASED_TRANCHES - tranches)
            tranches += 1
        else:
            deployed = goal.copy()
            tranches = 0
        rows[day] = deployed.copy()
    out = pd.DataFrame(rows).T.reindex(columns=columns).fillna(0.0)
    return out


def _tier_slip(prepared: dict) -> dict[str, float]:
    from scripts import growth_risk as gr

    adv20, illiq20 = gr.liquidity_panels(prepared["panel"])
    slip_maps, _ = gr.tier_map(prepared["marks"], prepared["eligible"], adv20, illiq20)
    merged: dict[str, float] = {}
    for day in prepared["marks"]:
        merged.update(slip_maps.get(day, {}))
    return merged


def build_weights(prepared: dict, variant: Variant, industries: dict) -> pd.DataFrame:
    from scripts import growth_risk as gr

    score = _smoothed_score(prepared) if variant.smooth else prepared["score"]
    if variant.growth_g4_weight is not None:
        score = _reweighted_score(prepared, variant.growth_g4_weight)
    sector = industries.get(variant.sector_level) if variant.sector_level else None
    strategy = gr.BufferTopK(score, prepared["eligible"], sector=sector, top_k=50,
                             gross=0.95, max_weight=0.04, sector_cap=variant.sector_cap,
                             buffer_rank=variant.buffer_rank)
    weights = strategy.generate(prepared["bars"])
    if variant.phased:
        monthly = weights.loc[prepared["marks"]]
        phased = _phased_weights(prepared, monthly)
        weights = _rebase_monthly(weights, phased, prepared["marks"])
    return weights


def _rebase_monthly(weights: pd.DataFrame, monthly: pd.DataFrame, marks) -> pd.DataFrame:
    """把月末目标替换为 ``monthly``，月内保持（后续 ffill 由引擎做）。"""
    out = pd.DataFrame(np.nan, index=weights.index, columns=weights.columns)
    out.loc[monthly.index] = monthly.reindex(columns=weights.columns).fillna(0.0).to_numpy()
    return out.ffill().fillna(0.0)


def holdings_at(weights: pd.DataFrame, marks) -> dict[pd.Timestamp, set[str]]:
    return {day: set(weights.loc[day][weights.loc[day] > 1e-9].index) for day in marks
            if day in weights.index}


def run_variants(names: list[str]) -> dict:
    from scripts import growth_backtest as gb
    from scripts import growth_robust as grb_impl

    prepared = grb_impl.prepare("main")
    symbols = prepared["panel"].symbols
    industries = {"level1": industry_panels(prepared["marks"], symbols)[0],
                  "level3": industry_panels(prepared["marks"], symbols)[1]}
    tier_slip = _tier_slip(prepared)

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    weights_store: dict[str, pd.DataFrame] = {}
    for name in names:
        variant = VARIANTS[name]
        t0 = time.time()
        weights = build_weights(prepared, variant, industries)
        weights_store[name] = weights
        outcome = grb_impl.run_engine(prepared, weights, name=f"v1-{name}",
                                      min_trade=variant.min_trade, slip_map=tier_slip)
        row = dict(outcome["row"])
        row.update({"variant": name, "label": variant.label,
                    "smooth": variant.smooth, "phased": variant.phased,
                    "sector_level": variant.sector_level, "sector_cap": variant.sector_cap,
                    "elapsed_sec": round(time.time() - t0, 1)})
        rows.append(row)
        outcome["daily"].to_csv(OUT / f"daily-{name}.csv", index=False, encoding="utf-8-sig")
        weights_store[name].to_csv(OUT / f"weights-{name}.csv.gz", compression="gzip")
        print(f"[run] {name} {row['total_return']:.4f} sharpe {row['sharpe']:.3f} "
              f"turnover {row['ann_turnover_one_way']:.3f} ({row['elapsed_sec']}s)", flush=True)

    table = pd.DataFrame(rows)
    marks = prepared["marks"]
    base_hold = holdings_at(weights_store[names[0]], marks) if names else {}
    overlaps = {}
    for name in names[1:]:
        hold = holdings_at(weights_store[name], marks)
        common = [d for d in marks if d in hold and d in base_hold]
        overlaps[name] = float(np.mean([len(hold[d] & base_hold[d]) / max(len(base_hold[d]), 1)
                                        for d in common])) if common else float("nan")
    table["holdings_overlap_vs_base"] = table["variant"].map(lambda v: 1.0 if v == names[0] else overlaps.get(v))
    table.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    payload = {"generated": pd.Timestamp.now("UTC").isoformat(),
               "baseline": names[0],
               "rows": table.to_dict(orient="records"),
               "overlap_vs_base": overlaps}
    (OUT / "variants.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                                  default=str), encoding="utf-8")
    _write_variants_md(payload, table, prepared)
    return payload


def _write_variants_md(payload: dict, table: pd.DataFrame, prepared: dict) -> None:
    base = table.iloc[0]
    lines = ["# 换手 / 稳定性专项：变体回测", "",
             f"生成：{payload['generated']} ｜ 样本：{prepared['sample']} ｜ "
             f"{prepared['start'].date()} ~ {prepared['end'].date()} ｜ 成本：分档滑点 + <5000 元单忽略",
             "", "## 0. 主表", "",
             "| 变体 | 累计 | 年化 | 夏普 | MDD | 卡玛 | 单边年换手 | 费用/净值 | 加权滑点 | 与基线持仓重叠 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in table.to_dict(orient="records"):
        lines.append(
            f"| {row['label']} | {row['total_return']:.2%} | {row['cagr']:.2%} | {row['sharpe']:.3f} | "
            f"{row['max_drawdown']:.2%} | {row['calmar']:.3f} | {row['ann_turnover_one_way']:.1%} | "
            f"{row['fees_pct_of_equity']:.2%} | {row.get('weighted_slip_bp', float('nan')):.2f}bp | "
            f"{(row.get('holdings_overlap_vs_base') or 0):.1%} |")
    lines += ["", "## 1. 相对基线", "",
              f"- 基线单边年换手 **{base['ann_turnover_one_way']:.1%}**、夏普 **{base['sharpe']:.3f}**。",
              "", "| 变体 | Δ换手 | Δ夏普 | Δ年化 | ΔMDD |",
              "|---|---|---|---|---|"]
    for row in table.to_dict(orient="records")[1:]:
        lines.append(f"| {row['label']} | {row['ann_turnover_one_way'] - base['ann_turnover_one_way']:+.1%} | "
                     f"{row['sharpe'] - base['sharpe']:+.3f} | {row['cagr'] - base['cagr']:+.2%} | "
                     f"{row['max_drawdown'] - base['max_drawdown']:+.2%} |")
    (OUT / "stability-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_variant_daily(path: Path, dates: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        frame["date"] = pd.to_datetime(dates.iloc[: len(frame)].to_numpy())
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").sort_index()


def _stats_from_daily(daily: pd.DataFrame, *, rf: float = 0.02) -> dict:
    equity = daily["equity"].astype("float64")
    returns = equity.pct_change().dropna()
    years = len(returns) / 244.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(returns.std(ddof=1) * np.sqrt(244.0))
    excess = float(returns.mean() * 244.0 - rf)
    sharpe = excess / vol if vol > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    mdd = float(drawdown.min())
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")
    turnover = float(daily.get("turnover", pd.Series(dtype=float)).fillna(0.0).sum()) / 2.0 / years
    fees = float(daily.get("fees", pd.Series(dtype=float)).fillna(0.0).sum())
    return {"total_return": total, "cagr": cagr, "ann_vol": vol, "sharpe": sharpe,
            "max_drawdown": mdd, "calmar": calmar, "ann_turnover_one_way": turnover,
            "fees_pct_of_equity": fees / float(equity.mean()) / years,
            "avg_exposure": float(daily.get("gross_exposure", pd.Series(dtype=float)).mean()),
            "mdd_date": str(drawdown.idxmin().date())}


def turnover_decomposition(weights: pd.DataFrame) -> dict:
    """月末权重换手拆分：报告季（4/8/10 月）vs 其他月份。"""
    marks = list(weights.index)
    rows = []
    for prev, cur in zip(marks[:-1], marks[1:]):
        a = weights.loc[prev].fillna(0.0)
        b = weights.loc[cur].fillna(0.0)
        rows.append({"month": cur, "one_way": float((b - a).abs().sum()) / 2.0})
    frame = pd.DataFrame(rows).set_index("month")
    report = frame[frame.index.month.isin((4, 8, 10))]
    other = frame[~frame.index.month.isin((4, 8, 10))]
    total = float(frame["one_way"].sum())
    return {"months": int(len(frame)),
            "total_one_way": total,
            "report_months": int(len(report)),
            "report_share_of_turnover": float(report["one_way"].sum() / total) if total else float("nan"),
            "report_mean": float(report["one_way"].mean()),
            "other_mean": float(other["one_way"].mean()),
            "by_month": {str(k.date()): float(v) for k, v in frame["one_way"].items()}}


def report() -> dict:
    baseline = pd.read_csv("results/growth/stage3/risk/daily-legacy5k_slip.csv")
    dates = baseline["date"]
    rows: list[dict] = []
    for name, variant in VARIANTS.items():
        path = OUT / f"daily-{name}.csv"
        if not path.exists():
            continue
        daily = _load_variant_daily(path, dates)
        stats = _stats_from_daily(daily)
        rows.append({"variant": name, "label": variant.label, **stats})
    table = pd.DataFrame(rows).sort_values("ann_turnover_one_way")
    base_row = table[table["variant"] == "base"].iloc[0]

    weights = pd.read_csv(OUT / "weights-base.csv.gz", index_col=0, parse_dates=True)
    decomposition = turnover_decomposition(weights)

    payload = {"generated": pd.Timestamp.now("UTC").isoformat(),
               "rows": table.to_dict(orient="records"),
               "baseline": base_row.to_dict(),
               "decomposition": decomposition}
    (OUT / "variants-master.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                                         default=str), encoding="utf-8")
    _write_master_md(payload, table, decomposition)
    return payload


def _write_master_md(payload: dict, table: pd.DataFrame, decomposition: dict) -> None:
    base = payload["baseline"]
    lines = ["# 换手 / 稳定性专项：全变体主表", "",
             f"生成：{payload['generated']} ｜ 样本：2017-01-03 ~ 2024-12-31 ｜ "
             "成本：分档滑点 + 忽略 <5000 元单",
             "",
             "| 变体 | 累计 | 年化 | 夏普 | MDD | 卡玛 | 单边年换手 | 费用/净值·年 | 平均暴露 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in table.to_dict(orient="records"):
        lines.append(
            f"| {row['label']} | {row['total_return']:.2%} | {row['cagr']:.2%} | {row['sharpe']:.3f} | "
            f"{row['max_drawdown']:.2%} | {row['calmar']:.3f} | {row['ann_turnover_one_way']:.1%} | "
            f"{row['fees_pct_of_equity']:.2%} | {row['avg_exposure']:.3f} |")
    lines += ["", "## 相对基线", "",
              f"基线：累计 {base['total_return']:.2%}、年化 {base['cagr']:.2%}、夏普 {base['sharpe']:.3f}、"
              f"MDD {base['max_drawdown']:.2%}、换手 {base['ann_turnover_one_way']:.1%}。", "",
              "| 变体 | Δ换手 | Δ年化 | Δ夏普 | ΔMDD |", "|---|---|---|---|---|"]
    for row in table.to_dict(orient="records"):
        if row["variant"] == "base":
            continue
        lines.append(f"| {row['label']} | {row['ann_turnover_one_way'] - base['ann_turnover_one_way']:+.1%} | "
                     f"{row['cagr'] - base['cagr']:+.2%} | {row['sharpe'] - base['sharpe']:+.3f} | "
                     f"{row['max_drawdown'] - base['max_drawdown']:+.2%} |")
    lines += ["", "## 换手来自哪里", "",
              f"- 月末权重换手：报告季（4/8/10 月，{decomposition['report_months']} 个月）"
              f"平均 **{decomposition['report_mean']:.1%}**，其他月份平均 **{decomposition['other_mean']:.1%}**；"
              f"报告季占全部换手的 **{decomposition['report_share_of_turnover']:.0%}**。"]
    (OUT / "stability-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def backfill_dates() -> int:
    """早期版本把 daily 存成无日期列；用基线日期回填，保持文件可独立复算。"""
    dates = pd.read_csv("results/growth/stage3/risk/daily-legacy5k_slip.csv")["date"]
    fixed = 0
    for path in sorted(OUT.glob("daily-*.csv")):
        frame = pd.read_csv(path)
        if "date" in frame.columns:
            continue
        if len(frame) != len(dates):
            print(f"[skip] {path.name} 长度 {len(frame)} ≠ 基线 {len(dates)}", flush=True)
            continue
        frame.insert(0, "date", dates.to_numpy())
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        fixed += 1
    return fixed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="换手 / 稳定性专项")
    parser.add_argument("--diag", action="store_true", help="只跑诊断")
    parser.add_argument("--runs", default="", help="逗号分隔的变体名")
    parser.add_argument("--report", action="store_true", help="只汇总已跑变体")
    parser.add_argument("--backfill", action="store_true", help="给无日期列的 daily 文件回填日期")
    args = parser.parse_args(argv)

    if args.backfill:
        print(f"[backfill] 回填 {backfill_dates()} 个文件", flush=True)
        return 0

    if args.report:
        payload = report()
        print(f"[report] {len(payload['rows'])} 个变体；报告季占比 "
              f"{payload['decomposition']['report_share_of_turnover']:.0%}", flush=True)
        return 0
    if args.diag or not args.runs:
        from scripts import growth_robust as grb_impl
        payload = diagnosis(grb_impl.prepare("main"))
        print(f"[diag] 总分秩相关 {payload['total']['rank_corr_mean']:.3f} "
              f"重叠 {payload['overlap_mean']:.3f} 剧烈月 {payload['violent_months']}", flush=True)
    if args.runs:
        names = [name.strip() for name in args.runs.split(",") if name.strip()]
        unknown = [name for name in names if name not in VARIANTS]
        if unknown:
            raise SystemExit(f"未知变体：{unknown}；可选 {list(VARIANTS)}")
        run_variants(names)
    return 0


def _write_diagnosis_md(payload: dict, factor_table: pd.DataFrame, block_table: pd.DataFrame) -> None:
    lines = ["# 换手根因诊断：因子 / 块的月度稳定性", "",
             f"生成：{payload['generated']} ｜ 样本：{payload['sample']} ｜ 月末数：{len(payload['marks'])}",
             "", "## 0. 一句话", ""]
    lines.append(f"总分相邻月秩相关均值 **{payload['total']['rank_corr_mean']:.3f}**，"
                 f"基线持仓相邻月重叠率均值 **{payload['overlap_mean']:.3f}**（中位 {payload['overlap_median']:.3f}），"
                 f"**{payload['violent_months']} 个月（{payload['violent_share']:.0%}）重叠率 <0.30**。")
    lines += ["", "## 1. 因子稳定性（越靠上越稳定）", "",
              "| 因子 | 块 | 秩相关均值 | 中位 | 最差 | P10 | top50 进出 | top100 进出 | 月数 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in factor_table.to_dict(orient="records"):
        lines.append(f"| `{row['factor']}` | {row['block']} | {row['rank_corr_mean']:.3f} | "
                     f"{row['rank_corr_median']:.3f} | {row['rank_corr_min']:.3f} | {row['rank_corr_p10']:.3f} | "
                     f"{row['churn_top50']:.1%} | {row['churn_top100']:.1%} | {row['months']} |")
    lines += ["", "## 2. 块稳定性", "",
              "| 块 | 权重 | 秩相关均值 | 中位 | 最差 | top50 进出 |",
              "|---|---|---|---|---|---|"]
    for row in block_table.to_dict(orient="records"):
        lines.append(f"| {row['block']} | {row['weight']} | {row['rank_corr_mean']:.3f} | "
                     f"{row['rank_corr_median']:.3f} | {row['rank_corr_min']:.3f} | {row['churn_top50']:.1%} |")
    lines += ["", "## 3. 剧烈换血月份归因（重叠率 <0.30）", "",
              "| 月份 | 重叠率 | 总分秩相关 | 最差块（相关） | 三个最翻转因子 |",
              "|---|---|---|---|---|"]
    for row in payload["attribution"]:
        worst = "、".join(f"`{k}` {v:.2f}" for k, v in row["worst_factors"])
        lines.append(f"| {row['month']} | {row['overlap']:.2f} | {row['total_corr']:.3f} | "
                     f"{row['worst_block']}（{row['worst_block_corr']:.3f}） | {worst} |")
    (OUT / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
