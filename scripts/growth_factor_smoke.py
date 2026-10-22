"""阶段 1 冒烟：在真实数据湖上检查 ``quant.factors.growth`` 的覆盖率、PIT 与符号。

这不是正式评估（正式协议见 ``docs/小盘成长策略设计与数据核对.md`` §4，阶段 2 才跑），
只回答四个问题：

1. 13 个因子每年有多少截面覆盖（是不是像设计那样"研发 2018 起、毛利率 2021 起"）；
2. 合格集（R1–R9 + G3）每年剩多少只（应落在 §1.10 的 1,200–1,760 量级）；
3. 月频 rank IC 的符号与量级（**只看方向对不对**，不当结论用）；
4. 有没有前视：把公告日之后才该出现的值提前拿到（抽样核对）。

用法::

    python -m scripts.growth_factor_smoke --out results/growth

产物：``results/growth/stage1-smoke.json``（可复算，数字都能在上面三个口径下重跑）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.factors import growth as G
from quant.factors.panel import build_panel
from scripts.redlow_study import END, START, build_block, load_bars


def _month_ends(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每个自然月的最后一个交易日。"""
    series = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(series.groupby([series.index.year, series.index.month]).max().to_numpy())


def _forward_month_return(close: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """月末 → 次月末的收益（含前视，只用于 IC 诊断）。"""
    at_ends = close.reindex(month_ends)
    return at_ends.shift(-1) / at_ends - 1.0


def _rank_ic(factor: pd.DataFrame, forward: pd.DataFrame) -> pd.Series:
    common = factor.index.intersection(forward.index)
    left = factor.reindex(common).rank(axis=1)
    right = forward.reindex(common).rank(axis=1)
    out = {}
    for day in common:
        a = left.loc[day]
        b = right.loc[day]
        ok = a.notna() & b.notna()
        if ok.sum() < 30:
            out[day] = np.nan
            continue
        out[day] = a[ok].corr(b[ok])
    return pd.Series(out).sort_index()


def _ic_stats(ic: pd.Series) -> dict:
    clean = ic.dropna()
    if clean.empty:
        return {"months": 0}
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    n = int(len(clean))
    annual = clean.groupby(clean.index.year).mean()
    return {
        "months": n,
        "ic_mean": round(mean, 4),
        "ic_std": round(std, 4),
        "ic_ir": round(float(mean / std) if std else float("nan"), 4),
        "t": round(float(mean / std * np.sqrt(n)) if std else float("nan"), 3),
        "positive_years": int((annual > 0).sum()),
        "years": int(len(annual)),
        "annual_ic": {int(year): round(float(value), 4) for year, value in annual.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="成长因子冒烟检查")
    parser.add_argument("--out", default="results/growth")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("读取数据湖 processed 层 …")
    bars, _stats = load_bars()
    panel = build_panel(bars)
    panel = panel.slice("2016-01-01", "2024-12-31")
    print(f"面板：{len(panel.dates)} 个交易日 × {len(panel.symbols)} 只标的")

    print("读取财报并按公告日散列 …")
    factors = G.GrowthFactors(panel)
    frames = factors.all_factors()

    # 1) 覆盖率：按年，分母 = 该年最后一个交易日的上市满 365 天标的数
    listing = factors.listing_days()
    day = panel.dates[-1]
    year_ends = pd.DatetimeIndex([panel.dates[panel.dates.year == y][-1] for y in range(2017, 2025)])
    coverage = {}
    first_valid = {}
    for name, frame in frames.items():
        row = {}
        for stamp in year_ends:
            mask = (listing.loc[stamp].reindex(frame.columns) >= 365).fillna(False).to_numpy(dtype=bool)
            universe = int(mask.sum())
            valid = frame.loc[stamp].to_numpy(dtype="float64")[mask]
            row[int(stamp.year)] = round(float(np.count_nonzero(~np.isnan(valid)) / universe) if universe else 0.0, 3)
        coverage[name] = row
        present = frame.notna().any(axis=1)
        first_valid[name] = str(present[present].index[0].date()) if present.any() else None

    # 2) 合格集规模
    print("构建合格集（R1–R9 + G3）…")
    block = build_block(panel)
    eligible = G.eligible_mask(factors, block_mask=block)
    pool = {int(stamp.year): int(eligible.loc[stamp].sum()) for stamp in year_ends}

    # 3) 月频 rank IC（符号检查）
    print("月频 rank IC（仅看方向）…")
    close = panel.close
    month_ends = _month_ends(panel.dates)
    month_ends = month_ends[(month_ends >= pd.Timestamp(START)) & (month_ends <= pd.Timestamp(END))]
    forward = _forward_month_return(close, month_ends)
    in_pool = eligible.reindex(month_ends)
    ic_report = {}
    for name, frame in frames.items():
        masked = frame.reindex(month_ends).where(in_pool)
        ic_report[name] = _ic_stats(_rank_ic(masked, forward))

    # 3b) 对照组：同样剔 ST/次新/停牌/一字板，但**不做成长筛选**（看"池内 IC 衰减"）
    print("全市场口径 rank IC（对照组）…")
    base_mask = panel.tradable & (listing >= 365) & ~block.reindex_like(panel.tradable).fillna(False)
    in_base = base_mask.reindex(month_ends)
    ic_base = {}
    for name, frame in frames.items():
        masked = frame.reindex(month_ends).where(in_base)
        ic_base[name] = _ic_stats(_rank_ic(masked, forward))

    # 4) PIT 实测（不依赖计算路径）：因子只允许在"公告日"或"公告日后的第一个交易日"跳变。
    #    若有人误用 report_date、或把未披露的报表提前散列，跳变日就会落到这两类日子之外。
    g1 = frames["g1_deduct_yoy"]
    long = factors.long("profit")[["code", "report_date", "notice_date"]]
    picks = [symbol for symbol in ("000001", "600519", "300750", "002415") if symbol in panel.symbols]
    allowed_after = {}
    for symbol in picks:
        code = symbol.split(".")[0]
        notices = long.loc[long["code"] == code, "notice_date"].dropna().sort_values()
        allowed = set()
        for notice in notices:
            later = panel.dates[panel.dates >= notice]
            allowed.add(notice)
            if len(later):
                allowed.add(later[0])
        allowed_after[symbol] = allowed
    pit_check = {}
    for symbol in picks:
        series = g1[symbol].dropna()
        moves = series[series.diff().fillna(1.0) != 0.0].index
        unexpected = [str(stamp.date()) for stamp in moves if stamp not in allowed_after[symbol]]
        pit_check[symbol] = {
            "moves": int(len(moves)),
            "unexpected": unexpected[:5],
            "unexpected_count": len(unexpected),
        }

    payload = {
        "panel": {"days": int(len(panel.dates)), "symbols": int(len(panel.symbols)), "start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date())},
        "coverage_by_year": coverage,
        "first_valid": first_valid,
        "eligible_by_year": pool,
        "ic": ic_report,
        "ic_control": ic_base,
        "pit_check": pit_check,
        "note": "阶段 1 冒烟：IC 只用于检查符号，不构成结论；正式评估见 §4。",
    }
    path = out_dir / "stage1-smoke.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n覆盖率（按年，分母=上市满 365 天）")
    header = "factor".ljust(20) + "".join(str(y).rjust(7) for y in range(2017, 2025))
    print(header)
    for name, row in coverage.items():
        print(name.ljust(20) + "".join(f"{row[y]:7.2f}" for y in range(2017, 2025)))
    print("\n合格集：" + "  ".join(f"{y}:{pool[y]}" for y in range(2017, 2025)))
    print("\n月频 IC（月份数 / 均值 / IR / t / 正 IC 年）——「成长合格集内」与「对照口径」")
    print("factor".ljust(20) + "pool: n / IC / t".ljust(26) + "control: IC / t")
    for name in ic_report:
        a = ic_report[name]
        b = ic_base.get(name, {})
        left = f"{a['months']:4d} {a['ic_mean']:+.4f} {a['t']:+.2f}" if a.get("months") else "      无"
        right = f"{b['ic_mean']:+.4f} {b['t']:+.2f}" if b.get("months") else "无"
        print(name.ljust(20) + left.ljust(26) + right)
    print("\nPIT 跳变日核对（只允许落在公告日/公告日后第一个交易日）")
    for symbol, info in pit_check.items():
        flag = "OK" if info["unexpected_count"] == 0 else f"异常 {info['unexpected_count']} 次: {info['unexpected']}"
        print(f"  {symbol}  跳变 {info['moves']:3d} 次  {flag}")
    print(f"\n已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
