"""阶段 C：统一单因子诊断（新核心 13 + 补充 13 + 旧因子全量，同池同口径）。

纪律（见 ``docs/小盘成长策略设计与数据核对.md`` §4.2）：
- 只在「成长合格集」内排秩、算 IC、算分层；
- 月末截面 vs 次月收益，2017-01 ~ 2024-12，**2025+ 一行不读**；
- 旧因子全量跑一遍（§4.2.1 事前登记），N0 一次性打印并落盘锁定。

用法::

    python -m scripts.growth_factor_ic --registry          # 只登记 N0
    python -m scripts.growth_factor_ic --only new --limit 3
    python -m scripts.growth_factor_ic                    # 全量诊断
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from quant.factors import events, frontier, fundamental, growth, growth_alt, library, style
from quant.factors.evaluate import rank_ic
from scripts.redlow_study import END, START, build_block, load_bars

OUT_DIR = Path("results/growth")
HOLDOUT = pd.Timestamp("2025-01-01")
MIN_NAMES = 50          # 截面最少 50 只才出 IC（合格集 1,200–1,800，月频约 1,000+）
QUANTILES = 5
EVENT_FACTORS = (
    "buyback_flag",
    "buyback_ratio",
    "div_plan",
    "high_transfer",
    "insider_buy",
    "insider_net",
    "lhb_flag",
    "lhb_inst_net",
    "lhb_netbuy",
    "st_flag",
    "st_unflag",
)
FRONTIER_FACTORS = (
    "downside_beta",
    "fip_momentum",
    "high_52w",
    "intraday_ret_20",
    "ivol_250",
    "overnight_ret_20",
    "residual_momentum",
)
NEW_CORE = tuple(growth.BLOCKS["growth"] + growth.BLOCKS["quality"] + growth.BLOCKS["rnd"] + growth.BLOCKS["aux"])
# style.py 注册进 library.FACTORS 的 20 个（模块里没有汇总常量，按名字硬编码以免误判）
STYLE_NAMES = frozenset(
    {
        "div_yield", "vol_20", "vol_60", "turnover_20", "turnover_vol", "zero_days", "amount_20",
        "spread_proxy", "low_price", "mcap", "fmcap", "listing_days", "rev_smallcap",
        "mcap_q05", "mcap_q10", "mcap_q20", "fmcap_q05", "fmcap_q10", "fmcap_q20", "maxret_smallcap",
    }
)


# 旧因子方向对齐：``events.register()`` / ``frontier.register()`` 在注册时带 ``sign``，
# 直接把原始函数返回的矩阵当因子用，方向会和 ``library.FACTORS`` 里「值越大越看多」的约定相反
# （实测 ``library.compute("overnight_ret_20")`` 与 ``frontier.overnight_ret_20()`` 相关系数 −1.0）。
# 方向一律从注册表取，不在这里硬编码 —— 硬编码第一版漏了 ``ivol_250``，被自检抓了出来。
_REGISTERED: set[str] = set()


def _ensure_registered(module_name: str) -> None:
    """``frontier.register()`` / ``events.register()`` 只跑一次。

    ``frontier.register()`` 内部没有 ``replace=True``，重复调用会对已注册的名字抛 ``ValueError``。
    """
    if module_name in _REGISTERED:
        return
    if module_name == "frontier":
        frontier.register()
    else:
        events.register()
    _REGISTERED.add(module_name)


def _orientation(module_name: str, name: str) -> int:
    """返回该旧因子在因子库约定下的方向（+1 / −1）；查不到注册项直接报错，不猜。"""
    _ensure_registered(module_name)
    entry = library.FACTORS.get(name)
    if entry is None and module_name == "events":
        # 事件因子注册名带窗口后缀（如 lhb_flag_5）
        for suffix in ("_5", "_20", "_60"):
            if name + suffix in library.FACTORS:
                entry = library.FACTORS[name + suffix]
                break
    if entry is None:
        raise AssertionError(f"{module_name}.{name} 不在 library.FACTORS 里，方向未知")
    return -1 if int(getattr(entry, "sign", 1)) < 0 else 1


# --------------------------------------------------------------------------- #
# 截面工具
# --------------------------------------------------------------------------- #


def month_ends(dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """样本区间的月末交易日（最后一个有行情的交易日）。"""
    inside = dates[(dates >= start) & (dates <= end)]
    if inside.empty:
        return pd.DatetimeIndex([])
    marks = inside.to_series().groupby([inside.year, inside.month]).max()
    return pd.DatetimeIndex(sorted(pd.DatetimeIndex(marks.to_numpy())))


def forward_month_returns(
    close: pd.DataFrame,
    marks: pd.DatetimeIndex,
    *,
    entry: str = "close",
    open_px: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """月末 → 下一个月末的收益（最后一个月为 NaN）。

    ``entry="close"``：月末收盘进场（IC 诊断传统口径，含次日跳空，偏乐观）。
    ``entry="open"``：月末之后第一个交易日开盘进场，与回测的 t+1 开盘成交一致。
    """
    at_marks = close.loc[marks]
    if entry == "close":
        return at_marks.pct_change().shift(-1)
    if open_px is None:
        raise ValueError("entry=open 需要传入开盘价面板")
    entry_px = open_px.shift(-1).reindex(index=marks).reindex(columns=close.columns)
    return at_marks.shift(-1) / entry_px - 1.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 5:
        return float("nan")
    ra = np.array(pd.Series(a).rank(), dtype="float64", copy=True)
    rb = np.array(pd.Series(b).rank(), dtype="float64", copy=True)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _rank_ic_series(factor: pd.DataFrame, forward: pd.DataFrame, marks: pd.DatetimeIndex) -> pd.Series:
    frame = factor.reindex(index=marks)
    fwd = forward.reindex(index=marks, columns=frame.columns)
    ic = rank_ic(frame, fwd, min_count=MIN_NAMES)
    return ic.replace([np.inf, -np.inf], np.nan).dropna()


def _quantile_table(factor: pd.DataFrame, forward: pd.DataFrame, marks: pd.DatetimeIndex) -> pd.DataFrame:
    frame = factor.reindex(index=marks)
    fwd = forward.reindex(index=marks, columns=frame.columns)
    rows = []
    for stamp in frame.index:
        values = frame.loc[stamp]
        rets = fwd.loc[stamp]
        keep = values.notna() & rets.notna()
        if keep.sum() < MIN_NAMES:
            continue
        ranked = values[keep].rank(pct=True)
        bucket = np.ceil(ranked * QUANTILES).clip(1, QUANTILES).astype(int)
        rows.append(rets[keep].groupby(bucket).mean())
    if not rows:
        return pd.DataFrame(columns=[f"Q{i}" for i in range(1, QUANTILES + 1)])
    return pd.DataFrame(rows).reindex(columns=range(1, QUANTILES + 1))


def _residual_returns(forward: pd.DataFrame, controls: Mapping[str, pd.DataFrame], marks: pd.DatetimeIndex) -> pd.DataFrame:
    """每月对 controls 做截面 OLS，返回残差收益矩阵（索引=月末，列=股票）。"""
    out = pd.DataFrame(np.nan, index=marks, columns=forward.columns, dtype="float64")
    for stamp in marks:
        y = forward.loc[stamp]
        cols = []
        for frame in controls.values():
            if stamp in frame.index:
                cols.append(frame.loc[stamp].reindex(y.index).to_numpy(dtype="float64"))
            else:
                cols.append(np.full(len(y), np.nan))
        if not cols:
            continue
        design = np.column_stack(cols)
        valid = np.isfinite(y.to_numpy(dtype="float64")) & np.isfinite(design).all(axis=1)
        if valid.sum() < MIN_NAMES:
            continue
        x = np.column_stack([np.ones(valid.sum()), design[valid]])
        yv = y.to_numpy(dtype="float64")[valid]
        try:
            beta, *_ = np.linalg.lstsq(x, yv, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - 极端数值问题
            continue
        resid = yv - x @ beta
        block = out.loc[stamp].to_numpy(dtype="float64", copy=True)
        block[valid] = resid
        out.loc[stamp] = block
    return out


def _max_abs_corr(factor: pd.DataFrame, others: Mapping[str, pd.DataFrame], marks: pd.DatetimeIndex) -> tuple[float, str]:
    """与该因子月均秩相关最高的对照因子（绝对值）。"""
    best, best_name = 0.0, ""
    frame = factor.reindex(index=marks)
    for name, other in others.items():
        if name == "self":
            continue
        other_at = other.reindex(index=marks, columns=frame.columns)
        corrs = []
        for stamp in marks:
            a = frame.loc[stamp].to_numpy(dtype="float64")
            b = other_at.loc[stamp].to_numpy(dtype="float64")
            keep = np.isfinite(a) & np.isfinite(b)
            if keep.sum() < MIN_NAMES:
                continue
            corrs.append(_spearman(a[keep], b[keep]))
        if not corrs:
            continue
        mean = float(np.nanmean(corrs))
        if np.isfinite(mean) and abs(mean) > best:
            best, best_name = abs(mean), name
    return best, best_name


# --------------------------------------------------------------------------- #
# 因子来源
# --------------------------------------------------------------------------- #


def old_factor_registry() -> dict[str, list[str]]:
    """锁定 N0：旧因子清单（一个都不少，也不许事后挑）。"""
    fundamental.register()
    registered = sorted(library.FACTORS)
    old_price = [n for n in registered if n not in STYLE_NAMES]
    old_style = [n for n in registered if n in STYLE_NAMES]
    old_fund = sorted(set(registered) - set(old_price) - set(old_style))
    return {
        "price": old_price,
        "style": old_style,
        "fundamental": old_fund,
        "events": list(EVENT_FACTORS),
        "frontier": list(FRONTIER_FACTORS),
    }


def iter_factors(panel, facts: growth.GrowthFactors, core_frames: Mapping[str, pd.DataFrame], family_filter: set[str] | None) -> Iterable[tuple[str, str, pd.DataFrame]]:
    """(family, name, frame) 生成器；一次只持有一个因子矩阵，避免内存爆掉。"""
    if family_filter is None or "new" in family_filter:
        for name in NEW_CORE:
            yield "新(核心)", name, core_frames[name]
        for name in ("g2_parent_yoy", "g10_revenue_yoy"):
            yield "新(诊断)", "diag_" + name, getattr(facts, name)()
    if family_filter is None or "alt" in family_filter:
        alt = growth_alt.alt_factors(panel)
        for name, frame in alt.items():
            yield "新(补充)", name, frame
        del alt
    registry = old_factor_registry()
    if family_filter is None or "old" in family_filter:
        fundamental.register()
        for name in registry["price"] + registry["style"] + registry["fundamental"]:
            try:
                frame = library.compute(name, panel)
            except Exception as exc:  # noqa: BLE001 - 单个旧因子失败不该中断全量诊断
                print(f"[warn] old factor {name} 计算失败: {exc}")
                continue
            yield "旧(量价/风格/基本面)", name, frame
            del frame
        for name in registry["events"]:
            try:
                frame = getattr(events, name)(panel)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] event factor {name} 计算失败: {exc}")
                continue
            sign = _orientation("events", name)
            yield "旧(事件)", name, frame if sign > 0 else -frame
            del frame
        for name in registry["frontier"]:
            try:
                frame = getattr(frontier, name)(panel)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] frontier factor {name} 计算失败: {exc}")
                continue
            sign = _orientation("frontier", name)
            yield "旧(前沿)", name, frame if sign > 0 else -frame
            del frame


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", action="store_true", help="只打印/落盘旧因子登记表 N0")
    parser.add_argument("--only", default=None, help="new / alt / old，逗号分隔")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个因子（冒烟用）")
    parser.add_argument("--coverage", action="store_true", help="只出补充因子的逐年覆盖率（阶段 B）")
    parser.add_argument("--entry", choices=["close", "open"], default="close",
                        help="IC 进场口径：close=月末收盘（默认）；open=次日开盘（与回测一致）")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    started = time.time()
    registry = old_factor_registry()
    n_old = sum(len(v) for v in registry.values())
    n_new = len(NEW_CORE) + 2
    n_alt = len(growth_alt.FACTOR_META)
    print(f"[registry] 新核心 {len(NEW_CORE)} + 新诊断 2 + 新补充 {n_alt} + 旧 {n_old} = {len(NEW_CORE) + 2 + n_alt + n_old}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    payload = {
        "generated": stamp,
        "sample": {"start": str(START), "end": str(END), "universe": "成长合格集（§3.7 R1–R9 + G3）"},
        "min_names": MIN_NAMES,
        "registry": {**registry, "counts": {k: len(v) for k, v in registry.items()}, "N_old": n_old, "N_new": n_new, "N_alt": n_alt},
        "factors": {},
    }
    (args.out / "old-factors-registry.json").write_text(
        json.dumps(payload["registry"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.registry:
        for key, names in registry.items():
            print(f"  {key}: {len(names)} -> {names}")
        return 0

    bars, stats = load_bars()
    panel = __import__("quant.factors.panel", fromlist=["build_panel"]).build_panel(bars)
    del bars
    print(f"[panel] {len(panel.dates)} 天 × {len(panel.symbols)} 只 ({stats.get('rows', '?')} 行)")

    block = build_block(panel)
    facts = growth.GrowthFactors(panel)
    eligible = growth.eligible_mask(facts, block_mask=block)
    print(f"[eligible] 日均 {int(eligible.sum(axis=1).mean())} 只，月均 {int(eligible.reindex(month_ends(panel.dates, START, END)).sum(axis=1).mean())} 只")

    marks = month_ends(panel.dates, START, END)
    forward = forward_month_returns(panel.close, marks, entry=args.entry, open_px=panel.open)
    print(f"[marks] {len(marks)} 个月末：{marks[0].date()} → {marks[-1].date()}（进场={args.entry}）")

    if args.coverage:
        write_coverage(panel, facts, eligible, args.out)
        return 0

    # 残差回归的控制变量（§4.2：ln 市值、amihud、A1、bp、ep）
    base_controls = {
        "ln_mcap": np.log(style.mcap_panel(panel).where(lambda f: f > 0)),
        "amihud": library.compute("amihud", panel),
        "a1_reversal": facts.a1_reversal(),
        "bp": fundamental.fundamental_wide("bp", panel),
        "ep": fundamental.fundamental_wide("ep", panel),
    }
    base_controls = {k: v.reindex(index=marks) for k, v in base_controls.items()}
    base_resid = _residual_returns(forward, base_controls, marks)

    # 新 13 个因子（旧因子的残差回归要再控这 13 个）
    core_frames = {name: getattr(facts, name)().reindex(index=marks) for name in NEW_CORE}
    core_resid = _residual_returns(forward, {**base_controls, **core_frames}, marks)

    family_filter = set(args.only.split(",")) if args.only else None
    results: list[dict] = []
    for family, name, frame in iter_factors(panel, facts, core_frames, family_filter):
        frame = core_frames.get(name, frame)
        frame = frame.reindex(index=marks, columns=forward.columns)
        frame = frame.where(eligible.reindex(index=marks, columns=forward.columns))
        ic = _rank_ic_series(frame, forward, marks)
        if ic.empty:
            print(f"[skip] {family}/{name}: 无有效 IC")
            continue
        qret = _quantile_table(frame, forward, marks)
        q_ann = ((1.0 + qret.mean()).pow(12) - 1.0) if not qret.empty else pd.Series(dtype="float")
        spread = float(q_ann.iloc[-1] - q_ann.iloc[0]) if len(q_ann) >= 2 else float("nan")
        mono = _spearman(np.arange(len(q_ann)), q_ann.to_numpy(dtype="float64")) if len(q_ann) >= 3 else float("nan")
        years = ic.groupby(ic.index.year).mean()
        is_core = name in core_frames
        resid_source = base_resid if is_core else core_resid
        resid_ic = _rank_ic_series(frame, resid_source, marks)
        entry = {
            "family": family,
            "name": name,
            "months": int(ic.size),
            "coverage": float(frame.notna().sum(axis=1).mean()),
            "ic_mean": float(ic.mean()),
            "ic_std": float(ic.std()),
            "ic_ir": float(ic.mean() / ic.std()) if ic.std() else float("nan"),
            "t": float(ic.mean() / ic.std() * np.sqrt(ic.size)) if ic.std() else float("nan"),
            "pos_years": int((years > 0).sum()),
            "n_years": int(years.size),
            "q_ann": [float(x) for x in q_ann.to_numpy()] if not q_ann.empty else [],
            "spread": spread,
            "monotonic": float(mono) if np.isfinite(mono) else None,
            "resid_ic": float(resid_ic.mean()) if not resid_ic.empty else float("nan"),
            "resid_t": float(resid_ic.mean() / resid_ic.std() * np.sqrt(resid_ic.size)) if not resid_ic.empty and resid_ic.std() else float("nan"),
            "max_corr": None,
            "max_corr_with": None,
        }
        if not is_core:
            corr, who = _max_abs_corr(frame, core_frames, marks)
            entry["max_corr"] = float(corr)
            entry["max_corr_with"] = who
        payload["factors"][f"{family}|{name}"] = entry
        results.append(entry)
        print(
            f"[{len(results):3d}] {family:<10} {name:<22} IC {entry['ic_mean']:+.4f} t {entry['t']:+.2f} "
            f"resid t {entry['resid_t']:+.2f} 正年 {entry['pos_years']}/{entry['n_years']}"
        )
        if args.limit and len(results) >= args.limit:
            break

    payload["elapsed_sec"] = round(time.time() - started, 1)
    (args.out / "factor-ic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload, args.out / "factor-ic.md")
    print(f"[done] {len(results)} 个因子，{payload['elapsed_sec']}s → {args.out/'factor-ic.json'}")
    return 0


def write_coverage(panel, facts: growth.GrowthFactors, eligible: pd.DataFrame, out: Path) -> None:
    """阶段 B：补充因子（13 个）在成长合格集内的逐年覆盖率，不看 IC。"""
    alt = growth_alt.alt_factors(panel)
    years = [y for y in sorted(set(panel.dates.year)) if y >= 2016]
    lines = [
        "# 阶段 B：补充因子覆盖率（成长合格集内，逐年）",
        "",
        f"- 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}；分母 = 当年该日合格集股票数（R1–R9 + G3）",
        "",
        "| 因子 | 定义 | " + " | ".join(str(y) for y in years) + " |",
        "|---|---|" + "---|" * len(years),
    ]
    detail: dict[str, dict[str, float]] = {}
    for name, frame in alt.items():
        masked = frame.where(eligible.reindex_like(frame))
        cells = []
        detail[name] = {}
        for year in years:
            block = masked[masked.index.year == year]
            if block.empty:
                cells.append("—")
                detail[name][str(year)] = float("nan")
                continue
            ratio = float(block.notna().sum(axis=1).mean() / max(eligible[eligible.index.year == year].sum(axis=1).mean(), 1))
            detail[name][str(year)] = round(ratio, 4)
            cells.append(f"{ratio:.0%}")
        meta = growth_alt.FACTOR_META.get(name, {})
        lines.append(f"| {name} | {meta.get('desc', '')} | " + " | ".join(cells) + " |")
    lines += ["", "## 首批可用日", ""]
    for name, frame in alt.items():
        first = frame.where(eligible.reindex_like(frame)).dropna(how="all").index.min()
        lines.append(f"- {name}: {first.date() if pd.notna(first) else '—'}")
    path = out / "alt-factor-coverage.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (out / "alt-factor-coverage.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[coverage] {len(alt)} 个补充因子 → {path}")


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{digits}f}"


def _negated_names() -> list[str]:
    """库内注册方向为 −1 的旧因子名（用于报告里的方向说明）。"""
    names = []
    for group, pool in (("events", EVENT_FACTORS), ("frontier", FRONTIER_FACTORS)):
        for name in pool:
            if _orientation(group, name) < 0:
                names.append(f"`{name}`")
    return sorted(names)


def write_report(payload: Mapping, path: Path) -> None:
    rows = list(payload["factors"].values())
    lines = [
        "# 阶段 C：单因子诊断（成长合格集内，同池同口径）",
        "",
        f"- 生成：{payload['generated']}；样本 {payload['sample']['start']} ~ {payload['sample']['end']}；"
        f"截面最少 {payload['min_names']} 只",
        f"- 旧因子登记 N₀ = {payload['registry']['N_old']}（量价 {payload['registry']['counts']['price']} / "
        f"风格 {payload['registry']['counts']['style']} / 基本面 {payload['registry']['counts']['fundamental']} / "
        f"事件 {payload['registry']['counts']['events']} / 前沿 {payload['registry']['counts']['frontier']}）",
        f"- 新因子：核心 {payload['registry']['N_new']} + 补充 {payload['registry']['N_alt']}",
        "",
        "判定线（§4.2 事前）：单因子 `|t|≥2 且 正IC年≥5/8` = 可用；`残差|t|≥2` = 有增量。",
        "",
        f"- 方向约定：全部因子按 `quant/factors` 注册方向（值越大越看多）。"
        f"{'、'.join(_negated_names())} 在库里注册为 `sign=−1`，本表已按注册方向取值；"
        "若取原始函数方向，这几个的 IC / t / 正年数整体取反 —— 它们与全市场研究的期望方向相反，"
        "属「方向不稳定」，判断时单列说明、不因取反后过线就采用。",
        "",
    ]
    for family in ("新(核心)", "新(诊断)", "新(补充)", "旧(量价/风格/基本面)", "旧(事件)", "旧(前沿)"):
        subset = [r for r in rows if r["family"] == family]
        if not subset:
            continue
        lines += [f"## {family}（{len(subset)}）", "",
                  "| 因子 | 月数 | 覆盖 | rank IC | IC_IR | t | 正 IC 年 | Q1→Q5 年化 | 多空 | 单调 | 残差 IC | 残差 t | 对照最大ρ |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        subset.sort(key=lambda r: -abs(r["t"]) if np.isfinite(r["t"]) else 0)
        for r in subset:
            q = r["q_ann"]
            q_txt = f"{q[0]:+.1%}→{q[-1]:+.1%}" if len(q) >= 2 else "—"
            lines.append(
                f"| {r['name']} | {r['months']} | {r['coverage']:,.0f} | {_fmt(r['ic_mean'])} | {_fmt(r['ic_ir'], 3)} | "
                f"{_fmt(r['t'], 2)} | {r['pos_years']}/{r['n_years']} | {q_txt} | {_fmt(r['spread'], 3)} | "
                f"{_fmt(r['monotonic'], 2)} | {_fmt(r['resid_ic'])} | {_fmt(r['resid_t'], 2)} | "
                f"{_fmt(r['max_corr'], 2)}{('·' + str(r['max_corr_with'])) if r['max_corr_with'] else ''} |"
            )
        lines.append("")

    passed = [r for r in rows if np.isfinite(r["t"]) and abs(r["t"]) >= 2 and r["pos_years"] >= 5]
    incr = [r for r in rows if np.isfinite(r["resid_t"]) and abs(r["resid_t"]) >= 2]
    lines += ["## 判定汇总", "", f"- 单因子可用（|t|≥2 且正年≥5/8）：{len(passed)} 个", ""]
    for r in passed:
        lines.append(f"  - {r['family']}/{r['name']}：t {_fmt(r['t'], 2)}，正年 {r['pos_years']}/{r['n_years']}")
    lines += ["", f"- 残差有增量（|t|≥2）：{len(incr)} 个", ""]
    for r in incr:
        lines.append(f"  - {r['family']}/{r['name']}：残差 t {_fmt(r['resid_t'], 2)}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
