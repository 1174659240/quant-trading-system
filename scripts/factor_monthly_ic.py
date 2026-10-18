"""基本面/量价因子的**月度** IC 与分年度稳定性。

为什么单独做一份：日频 IC 会被「噪声 + 交易成本」稀释，基本面因子在日频上几乎为零
（合并报告的 roe 1 日 IC 只有 0.0055），但月度调仓的持仓周期是 20 个交易日，
真正该看的是「月末截面排序 → 未来 20 日收益」的秩相关。这份脚本只做三件事：

1. **月末 20 日 IC**：每个因子在每个月末截面上的 rank IC 序列 → 均值/IR/t 值/胜率；
2. **分年度稳定性**：把 IC 按年份拆开，看「是不是只有某一年在赚钱」；
3. **覆盖度**：每个因子每个截面有多少只可用，避免拿 2,000 只的因子和 300 只的因子比 IC。

口径与 `--factor-market` 一致：在市值 + 退市股（recovery 用 haircut30 只影响回测，
IC 计算只用真实收益）、逐日剔 ST、区间 2016-01-04 ~ 2024-12-31。

用法::

    python -m scripts.factor_monthly_ic
    python -m scripts.factor_monthly_ic --factors ep,bp,roe,size --start 2016-01-04

产出 ``results/factor-monthly-ic/{monthly-ic.csv,yearly-ic.csv,factor-monthly-ic.md}``。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.factors import build_panel, compute, names as factor_names, rank_ic  # noqa: E402
from quant.factors.evaluate import ic_summary  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factor_market import FIELDS, LAKE, load_delisted_bars, load_layer  # noqa: E402

HORIZON = 20


def month_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每个自然月的最后一个交易日。"""
    index = pd.DatetimeIndex(dates)
    frame = pd.DataFrame({"day": index}, index=index)
    frame["year"] = index.year
    frame["month"] = index.month
    return pd.DatetimeIndex(frame.groupby(["year", "month"])["day"].last())


def _summary(factor: str, ic: pd.Series, coverage: pd.Series) -> dict:
    stats = ic_summary(ic, periods_per_year=12)
    row = {"factor": factor, "months": int(stats.get("n") or 0)}
    row.update({key: stats.get(key) for key in ("ic_mean", "ic_std", "ic_ir", "t_stat", "positive_rate")})
    row["abs_mean"] = float(ic.abs().mean()) if len(ic) else np.nan
    row["coverage_mean"] = float(coverage.mean()) if len(coverage) else np.nan
    row["coverage_min"] = float(coverage.min()) if len(coverage) else np.nan
    row["coverage_max"] = float(coverage.max()) if len(coverage) else np.nan
    return row


def render(monthly: pd.DataFrame, yearly: pd.DataFrame, args, meta: dict) -> str:
    lines = [
        "# 因子月度 IC 与分年度稳定性",
        "",
        f"- 区间：{args.start} ~ {args.end}；口径：月末截面 → 未来 {HORIZON} 个交易日收益（rank IC）",
        f"- 票池：在市值 + 退市股（{meta['symbols']} 只 / {meta['days']} 个交易日），逐日剔 ST（ST 占比 {meta['st_share'] * 100:.2f}%）",
        f"- 因子：{len(monthly)} 个；月末截面 {meta['months']} 个",
        "",
        "## 一、月度 IC 汇总（按 IC IR 排序）",
        "",
        "| 因子 | 月份数 | IC 均值 | IC IR | t 值 | 胜率 | 覆盖（均值/最小） |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    ordered = monthly.sort_values("ic_ir", ascending=False)
    for row in ordered.itertuples():
        lines.append(
            f"| {row.factor} | {int(row.months)} | {row.ic_mean:+.4f} | {row.ic_ir:+.3f} | "
            f"{row.t_stat:+.2f} | {row.positive_rate * 100:.1f}% | {row.coverage_mean:.0f} / {row.coverage_min:.0f} |"
        )
    lines += [
        "",
        "## 二、分年度 IC 均值（稳定性）",
        "",
        "| 因子 | " + " | ".join(str(year) for year in yearly.columns) + " | 正年数 |",
        "| --- | " + " | ".join("---" for _ in yearly.columns) + " | --- |",
    ]
    for factor, row in yearly.iterrows():
        positive = int((row > 0).sum())
        cells = " | ".join(
            "—" if pd.isna(value) else f"{value:+.4f}" for value in row.to_numpy(dtype=float)
        )
        lines.append(f"| {factor} | {cells} | {positive}/{int(row.notna().sum())} |")
    lines += [
        "",
        "## 三、读法",
        "",
        "- **IC IR > 0.3 且正年数过半**才值得进合成；单项 IC 均值 0.02 看起来小，但月度截面上 0.02 已相当于不错的选股信号。",
        "- 覆盖度差距大（价值/质量因子 2,000–3,000 只 vs 量价因子 5,000 只）时不要直接比 IC：覆盖面小的因子",
        "  只在「有报表的那部分票」里排序，等于自带一个流动性/规模筛选。",
        "- 分年度表要看「有没有单年独大」：某因子只有 2020 或 2015 一年为正，其余全负，基本是风格押注而非选股能力。",
        "- 财报因子有公告滞后（Q1 约 28 天、年报约 115 天），这里按 NOTICE_DATE 已对齐，但仍是修订后值（轻微前视）。",
        "",
        "## 已知偏差",
        "",
        "1. 指数成分只有当前快照（不做历史成分过滤）；",
        "2. 财务报表为修订后值；",
        "3. 行业分类年度 PIT；",
        "4. 退市股 2014 年前退市的 81 只无日线（本区间起点 2016，影响仅限极少数）。",
        "",
        f"复现：`python -m scripts.factor_monthly_ic --start {args.start} --end {args.end}`",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.factor_monthly_ic", description=__doc__.splitlines()[0])
    parser.add_argument("--lake", default=str(LAKE))
    parser.add_argument("--kind", default="processed", choices=["processed", "raw"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--factors", default="all", help="all（含基本面）或逗号分隔")
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--out", default=str(ROOT / "results" / "factor-monthly-ic"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== 读取 {args.kind} 层 ==", flush=True)
    frames, stats = load_layer(Path(args.lake), args.kind, args.limit or None)
    from quant.factors import fundamental
    from quant.factors import style  # noqa: F401  注册 18 个风格/容量因子
    from quant.factors import events as event_factors  # noqa: F401

    fundamental.register()
    event_factors.register()
    dead = load_delisted_bars(Path(args.lake))
    added = [symbol for symbol in dead if symbol not in frames]
    frames.update({symbol: dead[symbol] for symbol in added})
    print(f"== 退市股补进 {len(added)} 只（票池 {len(frames)} 只） ==", flush=True)

    lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end)
    frames = {
        symbol: window
        for symbol, frame in frames.items()
        if len(window := frame.loc[(frame.index >= lo) & (frame.index <= hi)]) >= 2
    }
    panel = build_panel(frames, fields=FIELDS)
    window = (panel.dates >= lo) & (panel.dates <= hi)
    if not window.all():
        panel = type(panel)({key: value.loc[panel.dates[window]] for key, value in panel.fields.items()})

    from quant.valuation import st_matrix

    st_mask = st_matrix()
    st_share = 0.0
    if not st_mask.empty:
        st_mask = st_mask.reindex(index=panel.dates, columns=panel.symbols).fillna(False).astype(bool)
        st_share = float(st_mask.to_numpy().mean())
    print(f"== 面板 {len(panel.dates)} 天 × {len(panel.symbols)} 只；ST 占比 {st_share * 100:.2f}% ==", flush=True)

    monthly_days = month_end_dates(panel.dates)
    monthly_mask = pd.Series(panel.dates.isin(monthly_days), index=panel.dates)
    forward = panel.forward_returns(HORIZON)
    print(f"== 月末截面 {len(monthly_days)} 个；前瞻 {HORIZON} 日收益已就绪 ==", flush=True)

    wanted = factor_names() if args.factors == "all" else [s.strip() for s in args.factors.split(",") if s.strip()]
    unknown = [name for name in wanted if name not in factor_names()]
    if unknown:
        raise SystemExit(f"未知因子：{unknown}")

    monthly_rows: list[dict] = []
    yearly_rows: dict[str, dict[int, float]] = {}
    for order, name in enumerate(wanted, 1):
        started = time.perf_counter()
        frame = compute(name, panel)
        ic_all = rank_ic(frame, forward, min_count=args.min_count)
        if st_mask is not None and not st_mask.empty:
            # 剔除 ST 后的 IC：把 ST 格的因子置 NaN 最省事（rank_ic 会自动跳过）
            frame = frame.where(~st_mask)
            ic_all = rank_ic(frame, forward, min_count=args.min_count)
        ic_monthly = ic_all.loc[monthly_mask.values & ic_all.notna().values].dropna()
        coverage = frame.notna().sum(axis=1).loc[monthly_days]
        row = _summary(name, ic_monthly, coverage)
        monthly_rows.append(row)
        by_year = ic_monthly.groupby(ic_monthly.index.year).mean()
        yearly_rows[name] = {int(year): float(value) for year, value in by_year.items()}
        print(
            f"  [{order}/{len(wanted)}] {name:22s} 月份 {row['months']:3d}  IC {row['ic_mean']:+.4f}  "
            f"IR {row['ic_ir']:+.3f}  覆盖 {row['coverage_mean']:.0f}  {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        del frame, ic_all
        gc.collect()

    monthly = pd.DataFrame(monthly_rows).sort_values("ic_ir", ascending=False).reset_index(drop=True)
    yearly = pd.DataFrame(yearly_rows).T
    yearly.index.name = "factor"
    yearly = yearly.reindex(sorted(yearly.columns), axis=1).sort_index()

    monthly.to_csv(out_dir / "monthly-ic.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "yearly-ic.csv", encoding="utf-8-sig")
    meta = {
        "symbols": len(panel.symbols),
        "days": len(panel.dates),
        "months": len(monthly_days),
        "st_share": st_share,
        "args": vars(args),
        "stats": stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "factor-monthly-ic.md").write_text(render(monthly, yearly, args, meta), encoding="utf-8")
    print(f"\n== 完成：{out_dir / 'factor-monthly-ic.md'} ==", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
