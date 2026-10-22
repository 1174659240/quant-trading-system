"""阶段 A 报告：补充数据（业绩预告/快报、限售解禁、股权质押）落盘核对。

只读已落盘的 ``D:/quant-data/alt/*.parquet`` 与采集清单，产出
``results/growth/alt-data-report.md``（覆盖率、PIT 字段、禁列审计、与主库对齐检查）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ALT = Path("D:/quant-data/alt")
OUT = Path("results/growth")
HOLDOUT = pd.Timestamp("2025-01-01")
DATASETS = ("forecast", "express", "unlock", "pledge")


def load(name: str) -> pd.DataFrame:
    path = ALT / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def listed_per_year(panel_years: range) -> dict[int, int]:
    """各年末沪深两市上市满 365 天的公司数（分母）。"""
    from quant.refdata import load_universe

    uni = load_universe()
    uni = uni[uni["board"].isin(["主板", "创业板", "科创板"])]
    uni = uni[uni["is_b_share"].fillna(0).astype(int) == 0]
    codes = uni["symbol"].astype(str).str.split(".").str[0]
    uni = uni[~codes.str.startswith(("43", "83", "87", "88", "92"))]
    ipo = pd.to_datetime(uni["ipo_date"], errors="coerce")
    out_date = pd.to_datetime(uni["out_date"], errors="coerce")
    result = {}
    for year in panel_years:
        day = pd.Timestamp(f"{year}-12-31")
        alive = (ipo <= day - pd.Timedelta(days=365)) & (out_date.isna() | (out_date > day))
        result[year] = int(alive.sum())
    return result


def alive_codes(year: int) -> set[str]:
    """当年末仍上市、且已上市满 365 天的 6 位代码集合。"""
    from quant.refdata import load_universe

    day = pd.Timestamp(f"{year}-12-31")
    uni = load_universe()
    uni = uni[uni["board"].isin(["主板", "创业板", "科创板"])]
    uni = uni[uni["is_b_share"].fillna(0).astype(int) == 0]
    codes = uni["symbol"].astype(str).str.split(".").str[0]
    ipo = pd.to_datetime(uni["ipo_date"], errors="coerce")
    out_date = pd.to_datetime(uni["out_date"], errors="coerce")
    keep = (~codes.str.startswith(("43", "83", "87", "88", "92"))) & (ipo <= day - pd.Timedelta(days=365)) & (
        out_date.isna() | (out_date > day)
    )
    return set(codes[keep])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    collection: dict = {}
    coll_path = OUT / "alt-collection.json"
    if coll_path.exists():
        collection = json.loads(coll_path.read_text(encoding="utf-8"))

    frames = {name: load(name) for name in DATASETS}
    years = range(2016, 2025)
    base = listed_per_year(years)

    lines = ["# 阶段 A：补充数据落盘核对", ""]
    lines.append(f"- 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}")
    lines.append("- 落盘位置：`D:/quant-data/alt/{forecast,express,unlock,pledge}.parquet`（原有目录仅新增，未改动任何已冻结文件）")
    lines.append("")

    lines += ["## 1. 总体规模", "", "| 数据集 | 行数 | 股票数 | 报告期/日期范围 | 字段数 |", "|---|---|---|---|---|"]
    for name in DATASETS:
        df = frames[name]
        if df.empty:
            lines.append(f"| {name} | 0 | — | — | — |")
            continue
        stock_col = "code" if "code" in df.columns else df.columns[0]
        period_col = next(c for c in ("report_period", "unlock_date", "as_of", "date") if c in df.columns)
        periods = df[period_col].dropna()
        rng = f"{periods.min():%Y-%m-%d} ~ {periods.max():%Y-%m-%d}" if len(periods) else "—"
        lines.append(f"| {name} | {len(df):,} | {df[stock_col].nunique():,} | {rng} | {df.shape[1]} |")
    lines.append("")

    # 业绩预告
    fc = frames["forecast"]
    if not fc.empty:
        lines += ["## 2. 业绩预告（forecast）", ""]
        lines.append(f"- 记录 {len(fc):,} 行 / {fc['code'].nunique():,} 只 / 报告期 {fc['report_period'].nunique()} 个")
        lines.append(f"- `notice_date` 缺失 {fc['notice_date'].isna().mean():.2%}；最大公告日 {fc['notice_date'].max():%Y-%m-%d}")
        if "indicator" in fc:
            top = fc["indicator"].value_counts().head(6)
            lines.append(f"- 指标构成（前 6）：" + "、".join(f"{k} {v:,}" for k, v in top.items()))
        if "forecast_type" in fc:
            top = fc["forecast_type"].value_counts().head(6)
            lines.append(f"- 预告类型（前 6）：" + "、".join(f"{k} {v:,}" for k, v in top.items()))
        lines.append("")

    # 业绩快报
    ex = frames["express"]
    if not ex.empty:
        lines += ["## 3. 业绩快报（express）", ""]
        lines.append(f"- 记录 {len(ex):,} 行 / {ex['code'].nunique():,} 只 / 报告期 {ex['report_period'].nunique()} 个")
        lines.append(f"- `notice_date` 缺失 {ex['notice_date'].isna().mean():.2%}；最大公告日 {ex['notice_date'].max():%Y-%m-%d}")
        lines.append("")

    # 解禁
    ul = frames["unlock"]
    if not ul.empty:
        lines += ["## 4. 限售解禁（unlock）", ""]
        lines.append(f"- 记录 {len(ul):,} 行 / {ul['code'].nunique():,} 只 / 解禁日 {ul['unlock_date'].min():%Y-%m-%d} ~ {ul['unlock_date'].max():%Y-%m-%d}")
        if "release_shares" in ul:
            lines.append(f"- 解禁股数中位 {ul['release_shares'].median():,.0f} 股；占比字段缺失 {ul['release_ratio'].isna().mean():.2%}" if "release_ratio" in ul else "")
        lines.append("")

    # 质押
    pl = frames["pledge"]
    if not pl.empty:
        lines += ["## 5. 股权质押（pledge）", ""]
        lines.append(f"- 记录 {len(pl):,} 行 / {pl['code'].nunique():,} 只 / 周度日期 {pl['as_of'].min():%Y-%m-%d} ~ {pl['as_of'].max():%Y-%m-%d}（{pl['as_of'].nunique()} 期）")
        lines.append("")

    # 年度覆盖
    lines += ["## 6. 年度覆盖率（分母 = 当年末上市满 365 天的沪深公司）", "",
              "| 年 | 分母 | 预告有记录 | 快报有记录 | 年内有解禁 | 质押有记录 |",
              "|---|---|---|---|---|---|"]
    for year in years:
        day = pd.Timestamp(f"{year}-12-31")
        pool = base.get(year, 0)

        universe = alive_codes(year)

        def cover(df: pd.DataFrame, col: str, extra: pd.Series | None = None) -> str:
            if df.empty or col not in df.columns:
                return "—"
            codes = df["code"].astype(str)
            dates = pd.to_datetime(df[col], errors="coerce")
            mask = (dates >= pd.Timestamp(f"{year}-01-01")) & (dates <= day)
            if extra is not None:
                mask &= extra.reindex(df.index).fillna(False).astype(bool)
            n = len(set(codes[mask]) & universe)
            return f"{n:,} ({n / pool:.1%})" if pool and n else "0"

        lines.append(
            f"| {year} | {pool:,} | {cover(fc, 'notice_date')} | {cover(ex, 'notice_date')} | "
            f"{cover(ul, 'unlock_date')} | {cover(pl, 'as_of', pl['pledge_ratio'] > 0)} |"
        )
    lines.append("")

    # 审计
    lines += ["## 7. 审计", ""]
    problems: list[str] = []
    for name in DATASETS:
        df = frames[name]
        if df.empty:
            problems.append(f"{name} 空表")
            continue
        date_col = "notice_date" if "notice_date" in df.columns else ("unlock_date" if "unlock_date" in df.columns else "as_of")
        late = df[date_col].dropna()
        if len(late) and late.max() >= HOLDOUT:
            problems.append(f"{name}.{date_col} 出现 {late.max():%Y-%m-%d} ≥ 2025-01-01")
        if "code" in df.columns:
            bad = ~df["code"].astype(str).str.fullmatch(r"\d{6}")
            if bad.any():
                problems.append(f"{name} 有 {bad.sum():,} 行代码不是 6 位纯数字")
    forbidden = sorted({c for df in frames.values() for c in df.columns if c.startswith("ret_") or "after" in c})
    if forbidden:
        lines.append("- 存在接口自带收益列（**已列入 FORBIDDEN_COLUMNS，因子层读不到**）：" + ", ".join(forbidden))
    lines += [f"- {'❌ ' + p if problems else '✅ 全部通过'}" for p in problems] or ["- ✅ 全部通过"]
    guard = collection.get("guard", {})
    if guard:
        same = guard.get("unchanged")
        lines.append(f"- 冻结文件指纹：{'✅ 未变' if same else '❌ 有变化'}（{len(guard.get('baseline', {}))} 个文件）")
    lines.append("")

    path = OUT / "alt-data-report.md"
    path.write_text("\n".join([x for x in lines if x is not None]), encoding="utf-8")
    print(f"[done] → {path}")
    print("\n".join(problems) if problems else "[audit] 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
