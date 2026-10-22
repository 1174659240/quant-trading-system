"""小盘成长策略 · 数据可得性核对（只读；不改数据湖、不动现有策略）。

回答五个问题：

1. **代码口径**：财务表里有多少行不是 6 位 A 股代码（东财「A 字头」非上市主体）？
2. **PIT 可信度**：``NOTICE_DATE`` 的公告滞后分布、是否有明显后置/修订痕迹、
   三表（profit/balance/cashflow）与 indicator 的公告日是否一致。
3. **字段覆盖**：研发费用、商誉、各增速字段、ROIC、经营现金流等，逐字段在
   「沪深两市当期已上市未退市」基数下的覆盖率（按年报期给出，另给全期分位）。
4. **可算性**：TTM（连续 4 季）、TTM 同比（连续 8 季）、连续多季正增长、
   研发强度、商誉/净资产、经营现金流 vs 扣非净利润 —— 每个派生指标在当期
   有多少只股票真的算得出来。
5. **流动性/容量**：日线成交额分布、一字板频率、单票 10% 参与率对应的容量。

口径（写进报告时必须原样保留）
- 只读 ``D:/quant-data/fundamentals`` 中**报告期 <= 2024-12-31** 的文件：2025 年起的
  报告期文件一个不打开（holdout 纪律）。
- 股票池分母 = ``D:/quant-data/ref/universe.csv`` 中「沪深两市（无北交所/B 股）、
  ipo_date <= 报告期、out_date 为空或 > 报告期」的股票。
- ``NOTICE_DATE`` 缺失时的口径不在本脚本内兜底（只统计缺失率），兜底规则在策略模块定。

用法::

    python -m scripts.growth_data_probe --out results/growth
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LAKE = Path("D:/quant-data")
FUND = LAKE / "fundamentals"
CUTOFF = pd.Timestamp("2024-12-31")
REAL_CODE = re.compile(r"^\d{6}$")
NORTH = ("43", "83", "87", "88", "92")

FIELDS = {
    "profit": [
        "TOTAL_OPERATE_INCOME",
        "OPERATE_INCOME",
        "OPERATE_COST",
        "PARENT_NETPROFIT",
        "DEDUCT_PARENT_NETPROFIT",
        "NETPROFIT",
        "RESEARCH_EXPENSE",
        "BASIC_EPS",
    ],
    "balance": [
        "GOODWILL",
        "TOTAL_ASSETS",
        "TOTAL_LIABILITIES",
        "TOTAL_PARENT_EQUITY",
        "ACCOUNTS_RECE",
        "INVENTORY",
    ],
    "cashflow": ["NETCASH_OPERATE", "CONSTRUCT_LONG_ASSET"],
    "indicator": [
        "KCFJCXSYJLR",
        "KCFJCXSYJLRTZ",
        "PARENTNETPROFIT",
        "PARENTNETPROFITTZ",
        "TOTALOPERATEREVE",
        "TOTALOPERATEREVETZ",
        "ROIC",
        "ROEJQ",
        "ROEKCJQ",
        "XSMLL",
        "XSJLL",
        "ZCFZL",
        "MGJYXJJE",
        "BPS",
        "TOTAL_SHARE",
        "A_FREE_SHARE",
    ],
}
# 报告里要单独点名的「成长核心字段」
GROWTH_FIELDS = [
    ("indicator", "KCFJCXSYJLRTZ"),
    ("indicator", "PARENTNETPROFITTZ"),
    ("indicator", "TOTALOPERATEREVETZ"),
    ("indicator", "ROIC"),
    ("indicator", "XSMLL"),
    ("indicator", "MGJYXJJE"),
    ("profit", "DEDUCT_PARENT_NETPROFIT"),
    ("profit", "RESEARCH_EXPENSE"),
    ("balance", "GOODWILL"),
    ("cashflow", "NETCASH_OPERATE"),
]


# --------------------------------------------------------------------------- 读取
def periods_available(table: str) -> list[pd.Timestamp]:
    """只列报告期 <= CUTOFF 的文件（2025+ 的文件不打开）。"""
    out = []
    for path in sorted((FUND / table).glob("????-??-??.parquet")):
        stamp = pd.Timestamp(path.stem)
        if stamp <= CUTOFF:
            out.append(stamp)
    return out


def load_table(table: str) -> tuple[pd.DataFrame, dict]:
    """读一张财务表（报告期 <= 2024-12-31），返回长表与读取统计。"""
    frames = []
    raw_rows = 0
    for stamp in periods_available(table):
        path = FUND / table / f"{stamp.date()}.parquet"
        frame = pd.read_parquet(path)
        raw_rows += len(frame)
        keep = [
            c
            for c in ["SECURITY_CODE", "SECURITY_NAME_ABBR", "REPORT_DATE", "NOTICE_DATE", *FIELDS[table]]
            if c in frame.columns
        ]
        part = frame[keep].copy()
        part["REPORT_DATE"] = pd.to_datetime(part["REPORT_DATE"], errors="coerce")
        part["NOTICE_DATE"] = pd.to_datetime(part["NOTICE_DATE"], errors="coerce")
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)
    long["SECURITY_CODE"] = long["SECURITY_CODE"].astype(str).str.zfill(6)
    long["is_real_code"] = long["SECURITY_CODE"].str.match(REAL_CODE)
    for column in FIELDS[table]:
        if column in long.columns:
            long[column] = pd.to_numeric(long[column], errors="coerce")
    stats = {
        "periods": len(frames),
        "first_period": str(periods_available(table)[0].date()),
        "last_period": str(periods_available(table)[-1].date()),
        "raw_rows": raw_rows,
        "pseudo_rows": int((~long["is_real_code"]).sum()),
        "pseudo_share": round(float((~long["is_real_code"]).mean()), 4),
        "real_rows": int(long["is_real_code"].sum()),
        "real_codes": int(long.loc[long["is_real_code"], "SECURITY_CODE"].nunique()),
        "dup_real_pairs": int(
            long.loc[long["is_real_code"]].duplicated(subset=["SECURITY_CODE", "REPORT_DATE"]).sum()
        ),
        "notice_missing_share": round(float(long.loc[long["is_real_code"], "NOTICE_DATE"].isna().mean()), 4),
    }
    return long, stats


# --------------------------------------------------------------------------- 池
def universe() -> pd.DataFrame:
    frame = pd.read_csv(LAKE / "ref" / "universe.csv", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame = frame[frame["board"].isin(["主板", "创业板", "科创板"])]
    frame = frame[frame["is_b_share"].fillna(0).astype(int) == 0]
    frame = frame[~frame["symbol"].str[:2].isin(NORTH)]
    frame["ipo_date"] = pd.to_datetime(frame["ipo_date"], errors="coerce")
    frame["out_date"] = pd.to_datetime(frame["out_date"], errors="coerce")
    return frame


def listed_at(uni: pd.DataFrame, day: pd.Timestamp) -> set[str]:
    alive = (uni["ipo_date"] <= day) & (uni["out_date"].isna() | (uni["out_date"] > day))
    return set(uni.loc[alive, "symbol"])


# --------------------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="小盘成长数据可得性核对")
    parser.add_argument("--out", default="results/growth")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tables: dict[str, pd.DataFrame] = {}
    inventory: dict[str, dict] = {}
    for table in FIELDS:
        tables[table], inventory[table] = load_table(table)
        print(f"[load] {table}: {inventory[table]}", flush=True)

    uni = universe()
    report: dict = {"generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                    "cutoff": str(CUTOFF.date()), "inventory": inventory,
                    "universe_rows": int(len(uni))}

    # ---------------------------------------------------------------- 1. 代码口径
    detail = {}
    for table, frame in tables.items():
        bad = frame.loc[~frame["is_real_code"]]
        detail[table] = {
            "pseudo_rows": int(len(bad)),
            "pseudo_codes": int(bad["SECURITY_CODE"].nunique()),
            "samples": bad["SECURITY_CODE"].drop_duplicates().head(6).tolist(),
        }
        if table == "profit":
            real = frame.loc[frame["is_real_code"], "SECURITY_CODE"].astype(str)
            detail[table]["real_codes_by_prefix"] = {
                "60x/68x(沪)": int(real[real.str[:2].isin(["60", "68"])].nunique()),
                "00x/30x(深)": int(real[real.str[:2].isin(["00", "30"])].nunique()),
                "其他6位": int(real[~real.str[:2].isin(["60", "68", "00", "30"])].nunique()),
            }
            in_uni = real[real.isin(set(uni["symbol"]))].nunique()
            detail[table]["real_codes_in_universe"] = int(in_uni)
            detail[table]["real_codes_not_in_universe"] = int(real.nunique() - in_uni)
            extra = sorted(set(real.unique()) - set(uni["symbol"]))[:8]
            detail[table]["not_in_universe_samples"] = extra
    report["code_scope"] = detail

    # ---------------------------------------------------------------- 2. PIT 可信度
    pit: dict = {}
    for table, frame in tables.items():
        real = frame.loc[frame["is_real_code"] & frame["NOTICE_DATE"].notna()].copy()
        lag = (real["NOTICE_DATE"] - real["REPORT_DATE"]).dt.days
        pit[table] = {
            "lag_days_p05_p25_p50_p75_p95": [float(np.nanpercentile(lag, q)) for q in (5, 25, 50, 75, 95)],
            "lag_negative_share": round(float((lag < 0).mean()), 5),
            "lag_gt_180_share": round(float((lag > 180).mean()), 5),
            "lag_gt_365_share": round(float((lag > 365).mean()), 5),
        }
    # 单期细看：年报/半年报/三季报的公告日分散度
    profit = tables["profit"]
    per_period = {}
    for stamp in (pd.Timestamp("2019-12-31"), pd.Timestamp("2022-12-31"), pd.Timestamp("2024-12-31"),
                  pd.Timestamp("2024-06-30"), pd.Timestamp("2024-09-30")):
        part = profit[(profit["REPORT_DATE"] == stamp) & profit["is_real_code"] & profit["NOTICE_DATE"].notna()]
        if not len(part):
            continue
        lag = (part["NOTICE_DATE"] - part["REPORT_DATE"]).dt.days
        per_period[str(stamp.date())] = {
            "n": int(len(part)),
            "notice_min": str(part["NOTICE_DATE"].min().date()),
            "notice_median": str(part["NOTICE_DATE"].median().date()),
            "notice_max": str(part["NOTICE_DATE"].max().date()),
            "lag_p50": round(float(lag.median()), 1),
            "lag_p95": round(float(lag.quantile(0.95)), 1),
            "late_gt_180": int((lag > 180).sum()),
        }
    pit["by_period_profit"] = per_period
    # 三表 vs indicator 公告日一致性
    key = ["SECURITY_CODE", "REPORT_DATE"]
    left = profit.loc[profit["is_real_code"], key + ["NOTICE_DATE"]].rename(columns={"NOTICE_DATE": "notice_profit"})
    ind = tables["indicator"]
    right = ind.loc[ind["is_real_code"], key + ["NOTICE_DATE"]].rename(columns={"NOTICE_DATE": "notice_indicator"})
    merged = left.merge(right, on=key, how="inner")
    diff = (merged["notice_indicator"] - merged["notice_profit"]).dt.days
    pit["indicator_vs_profit_notice"] = {
        "matched": int(len(merged)),
        "same_day_share": round(float((diff == 0).mean()), 4),
        "within_3d_share": round(float((diff.abs() <= 3).mean()), 4),
        "abs_median": float(diff.abs().median()),
        "indicator_later_share": round(float((diff > 0).mean()), 4),
    }
    bal = tables["balance"]
    lb = bal.loc[bal["is_real_code"], key + ["NOTICE_DATE"]].rename(columns={"NOTICE_DATE": "notice_balance"})
    merged2 = merged.merge(lb, on=key, how="inner")
    d2 = (merged2["notice_balance"] - merged2[["notice_profit", "notice_indicator"]].max(axis=1)).dt.days
    pit["balance_vs_max_notice"] = {
        "matched": int(len(merged2)),
        "balance_latest_share": round(float((d2 >= 0).mean()), 4),
        "median_days_later": float(d2.median()),
    }
    report["pit"] = pit

    # ---------------------------------------------------------------- 3. 字段覆盖
    coverage: dict = {}
    for stamp in sorted(profit["REPORT_DATE"].dropna().unique()):
        stamp = pd.Timestamp(stamp)
        alive = listed_at(uni, stamp)
        if len(alive) < 100:
            continue
        row = {"listed": len(alive)}
        for table, frame in tables.items():
            part = frame[(frame["REPORT_DATE"] == stamp) & frame["is_real_code"]]
            inpool = part[part["SECURITY_CODE"].isin(alive)]
            row[f"{table}_rows"] = int(len(inpool))
            row[f"{table}_codes"] = int(inpool["SECURITY_CODE"].nunique())
            for column in FIELDS[table]:
                if column in inpool.columns:
                    row[f"{table}.{column}"] = round(float(inpool[column].notna().mean()), 4) if len(inpool) else None
        coverage[str(stamp.date())] = row
    report["coverage_by_period"] = coverage

    # 年报期（12-31）汇总 + 全期分位
    annual = {k: v for k, v in coverage.items() if k.endswith("12-31")}
    report["coverage_annual"] = annual
    field_stats = {}
    for table in FIELDS:
        for column in FIELDS[table]:
            key_name = f"{table}.{column}"
            values = [v[key_name] for v in coverage.values() if v.get(key_name) is not None]
            if values:
                field_stats[key_name] = {
                    "min": round(float(np.min(values)), 4),
                    "median": round(float(np.median(values)), 4),
                    "max": round(float(np.max(values)), 4),
                }
    report["coverage_field_stats"] = field_stats

    # ---------------------------------------------------------------- 4. 可算性
    def pivot(table: str, column: str) -> pd.DataFrame:
        part = tables[table]
        part = part[part["is_real_code"] & part[column].notna()]
        return part.pivot_table(index="REPORT_DATE", columns="SECURITY_CODE", values=column, aggfunc="last").sort_index()

    deduct = pivot("profit", "DEDUCT_PARENT_NETPROFIT")
    rnd = pivot("profit", "RESEARCH_EXPENSE")
    revenue = pivot("profit", "TOTAL_OPERATE_INCOME")
    goodwill = pivot("balance", "GOODWILL")
    equity = pivot("balance", "TOTAL_PARENT_EQUITY")
    ocf = pivot("cashflow", "NETCASH_OPERATE")
    roic = pivot("indicator", "ROIC")
    deduct_tz = pivot("indicator", "KCFJCXSYJLRTZ")

    feasibility: dict = {}
    for stamp in sorted(deduct.index):
        stamp = pd.Timestamp(stamp)
        if stamp.month != 12 or stamp.year < 2015:
            continue
        alive = listed_at(uni, stamp)
        position = list(deduct.index).index(stamp)
        inpool = lambda frame: frame.columns.isin(alive)  # noqa: E731
        window4 = deduct.iloc[max(0, position - 3): position + 1][list(alive & set(deduct.columns))]
        window8 = deduct.iloc[max(0, position - 7): position + 1][list(alive & set(deduct.columns))]
        have4 = int(window4.notna().all().sum()) if len(window4) == 4 else 0
        have8 = int(window8.notna().all().sum()) if len(window8) == 8 else 0
        row = {
            "listed": len(alive),
            "profit_rows": int(deduct.iloc[position][inpool(deduct)].notna().sum()),
            "ttm4_quarters": have4,
            "ttm8_quarters": have8,
            "rd_present": int((rnd.iloc[position][inpool(rnd)].notna() & rnd.iloc[position][inpool(rnd)].ne(0)).sum()),
            "rd_positive": int(rnd.iloc[position][inpool(rnd)].gt(0).sum()),
            "rd_intensity_median": round(float(
                (rnd.iloc[position][inpool(rnd)] / revenue.iloc[position].reindex(rnd.columns)[inpool(rnd)].replace(0, np.nan)
                 ).median()), 4),
            "goodwill_present": int(goodwill.iloc[position][inpool(goodwill)].notna().sum()),
            "goodwill_positive": int(goodwill.iloc[position][inpool(goodwill)].gt(0).sum()),
            "goodwill_over_equity_p50": round(float(
                (goodwill.iloc[position][inpool(goodwill)] / equity.iloc[position].reindex(goodwill.columns)[inpool(goodwill)].replace(0, np.nan)
                 ).median()), 4),
            "goodwill_over_equity_p90": round(float(
                (goodwill.iloc[position][inpool(goodwill)] / equity.iloc[position].reindex(goodwill.columns)[inpool(goodwill)].replace(0, np.nan)
                 ).quantile(0.90)), 4),
            "ocf_present": int(ocf.iloc[position][inpool(ocf)].notna().sum()),
            "roic_present": int(roic.iloc[position][inpool(roic)].notna().sum()),
            "deduct_growth_field_present": int(deduct_tz.iloc[position][inpool(deduct_tz)].notna().sum()),
        }
        feasibility[str(stamp.date())] = row
    report["feasibility"] = feasibility

    # ---------------------------------------------------------------- 5. 增速字段语义
    semantic: dict = {}
    for value_col, growth_col in (
        ("KCFJCXSYJLR", "KCFJCXSYJLRTZ"),
        ("PARENTNETPROFIT", "PARENTNETPROFITTZ"),
        ("TOTALOPERATEREVE", "TOTALOPERATEREVETZ"),
    ):
        value = pivot("indicator", value_col)
        growth = pivot("indicator", growth_col)
        base = value.shift(4)
        yoy = value / base.replace(0, np.nan) - 1.0
        yoy = yoy.where((value > 0) & (base > 0))
        mine = yoy.stack(future_stack=True) if hasattr(yoy, "stack") else yoy.stack()
        theirs = (growth / 100.0).stack()
        both = pd.concat({"mine": mine, "theirs": theirs}, axis=1).dropna()
        if not len(both):
            continue
        delta = (both["theirs"] - both["mine"]).abs()
        semantic[f"{growth_col}_vs_yoy_{value_col}"] = {
            "n": int(len(both)),
            "within_1pp": round(float((delta <= 0.01).mean()), 4),
            "within_5pp": round(float((delta <= 0.05).mean()), 4),
            "median_abs_diff": round(float(delta.median()), 4),
            "corr": round(float(both["mine"].corr(both["theirs"])), 4),
        }
        # 换一种口径：单季同比（先差分再同比）
        single = value.diff()
        single.loc[single.index.month == 3] = value.loc[single.index.month == 3]
        single_base = single.shift(4)
        single_yoy = single / single_base.replace(0, np.nan) - 1.0
        single_yoy = single_yoy.where((single > 0) & (single_base > 0))
        both_q = pd.concat({"mine": single_yoy.stack(), "theirs": theirs}, axis=1).dropna()
        if len(both_q):
            dq = (both_q["theirs"] - both_q["mine"]).abs()
            semantic[f"{growth_col}_single_quarter_vs_yoy"] = {
                "n": int(len(both_q)),
                "within_5pp": round(float((dq <= 0.05).mean()), 4),
                "median_abs_diff": round(float(dq.median()), 4),
            }
    report["growth_field_semantics"] = semantic

    # ------------------------------------------------ 5b. 现成增速字段的陷阱与缺失语义
    traps: dict = {}
    value = pivot("indicator", "KCFJCXSYJLR")
    growth = pivot("indicator", "KCFJCXSYJLRTZ")
    base = value.shift(4)
    stacked = pd.concat(
        {"value": value.stack(), "base": base.stack(), "growth": (growth / 100.0).stack()}, axis=1
    ).dropna(subset=["growth"])
    neg = stacked[stacked["base"] <= 0]
    traps["rows_with_growth_field"] = int(len(stacked))
    traps["share_base_le_0"] = round(float((stacked["base"] <= 0).mean()), 4)
    traps["share_abs_growth_gt_10x"] = round(float((stacked["growth"].abs() > 10).mean()), 4)
    traps["share_abs_growth_gt_100x"] = round(float((stacked["growth"].abs() > 100).mean()), 4)
    if len(neg):
        traps["negbase_rows"] = int(len(neg))
        traps["negbase_growth_median"] = round(float(neg["growth"].median()), 4)
        traps["negbase_abs_gt_5x_share"] = round(float((neg["growth"].abs() > 5).mean()), 4)
    # 商誉：缺失 vs 显式 0
    balance = tables["balance"]
    last = balance[(balance["REPORT_DATE"] == CUTOFF) & balance["is_real_code"]]
    alive_end = listed_at(uni, CUTOFF)
    last = last[last["SECURITY_CODE"].isin(alive_end)]
    has_assets = last["TOTAL_ASSETS"].notna()
    traps["goodwill_at_2024"] = {
        "rows": int(len(last)),
        "null_share": round(float(last["GOODWILL"].isna().mean()), 4),
        "zero_rows": int(last["GOODWILL"].eq(0).sum()),
        "null_but_has_assets": int((last["GOODWILL"].isna() & has_assets).sum()),
        "has_assets_rows": int(has_assets.sum()),
    }
    # 三表是否真的排除金融业：2024-12-31 缺 OPERATE_COST 的公司是谁
    pf_last = tables["profit"][(tables["profit"]["REPORT_DATE"] == CUTOFF) & tables["profit"]["is_real_code"]]
    pf_last = pf_last[pf_last["SECURITY_CODE"].isin(alive_end)]
    missing_cost = pf_last[pf_last["OPERATE_COST"].isna()]
    traps["financial_check_2024"] = {
        "profit_rows": int(len(pf_last)),
        "missing_operate_cost": int(len(missing_cost)),
        "missing_operate_cost_names": missing_cost["SECURITY_NAME_ABBR"].head(12).tolist(),
        "missing_operate_income": int(pf_last["OPERATE_INCOME"].isna().sum()),
    }
    report["field_traps"] = traps

    # ------------------------------------------------ 5c. 候选集漏斗（成长门槛逐层收缩）
    single = deduct.diff()
    single.loc[single.index.month == 3] = deduct.loc[single.index.month == 3]
    ttm = single.rolling(4, min_periods=4).sum()
    ttm_base = ttm.shift(4)
    ttm_growth = (ttm / ttm_base.replace(0, np.nan) - 1.0).where((ttm > 0) & (ttm_base > 0))
    sq_growth = (single / single.shift(4).replace(0, np.nan) - 1.0).where((single > 0) & (single.shift(4) > 0))
    positive_quarters = (sq_growth > 0).rolling(4, min_periods=4).sum()
    zcfzl = pivot("indicator", "ZCFZL")
    funnel: dict = {}
    for stamp in sorted(ttm_growth.index):
        stamp = pd.Timestamp(stamp)
        if stamp.month != 12 or stamp.year < 2016:
            continue
        alive = listed_at(uni, stamp)
        cols = [c for c in ttm_growth.columns if c in alive]
        if not cols:
            continue

        def take(frame):
            return frame.reindex(columns=cols).loc[stamp] if stamp in frame.index else pd.Series(np.nan, index=cols)

        step1 = take(ttm).gt(0)                                  # 真赚钱
        step2 = take(ttm_growth).gt(0)                           # 且同比正增长
        step3 = take(positive_quarters).ge(2)                    # 且最近 4 单季至少 2 季正增长
        step4 = take(goodwill).fillna(0.0) / take(equity).replace(0, np.nan) <= 0.30
        step5 = take(zcfzl).le(70.0)
        step6 = take(ocf) / take(ttm).replace(0, np.nan) >= 0.0
        gate = step1 & step2 & step3
        full = gate & step4 & step5 & step6
        funnel[str(stamp.date())] = {
            "listed": len(alive),
            "ttm_profit_gt0": int(step1.sum()),
            "ttm_growth_gt0": int(step2.sum()),
            "positive_quarters_ge2": int(step3.sum()),
            "growth_gate": int(gate.sum()),
            "after_risk_filters": int(full.sum()),
        }
    report["candidate_funnel"] = funnel

    # ---------------------------------------------------------------- 6. 流动性
    bars_files = sorted((LAKE / "processed" / "parquet").glob("year=*/data.parquet"))
    liquidity: dict = {}
    for path in bars_files:
        frame = pd.read_parquet(path, columns=["symbol", "date", "close", "high", "low", "volume", "amount", "prev_close", "tradable"])
        frame["code"] = frame["symbol"].astype(str).str.split(".").str[0]
        frame = frame[~frame["code"].str[:2].isin(NORTH)]
        frame = frame[frame["volume"].gt(0) & frame["amount"].gt(0) & frame["tradable"].fillna(1).astype(int).eq(1)]
        if not len(frame):
            continue
        frame["ret"] = frame["close"] / frame["prev_close"] - 1.0
        one_price = (frame["high"] - frame["low"]).abs() <= frame["close"] * 1e-4
        locked_up = one_price & frame["ret"].ge(0.095)
        locked_down = one_price & frame["ret"].le(-0.095)
        per_stock = frame.groupby("code")["amount"].median()
        rank = per_stock.rank(pct=True)
        frame["grp"] = np.where(frame["code"].map(rank) <= 1 / 3, "low", np.where(frame["code"].map(rank) >= 2 / 3, "high", "mid"))
        small = frame[frame["grp"] == "low"]
        year = path.parent.name.split("=")[-1]
        liquidity[year] = {
            "stock_days": int(len(frame)),
            "stocks": int(frame["code"].nunique()),
            "median_adv_p10": round(float(per_stock.quantile(0.10)) / 1e4, 1),   # 万元
            "median_adv_p25": round(float(per_stock.quantile(0.25)) / 1e4, 1),
            "median_adv_p50": round(float(per_stock.quantile(0.50)) / 1e4, 1),
            "median_adv_p75": round(float(per_stock.quantile(0.75)) / 1e4, 1),
            "median_adv_p90": round(float(per_stock.quantile(0.90)) / 1e4, 1),
            "locked_up_share": round(float(locked_up.mean()), 5),
            "locked_down_share": round(float(locked_down.mean()), 5),
            "low_adv_tercile": {
                "stocks": int(small["code"].nunique()),
                "median_adv_wan": round(float(small.groupby("code")["amount"].median().median()) / 1e4, 1),
                "locked_up_share": round(float((small["high"] - small["low"]).abs().le(small["close"] * 1e-4).mul(small["ret"].ge(0.095)).mean()), 5),
                "locked_down_share": round(float((small["high"] - small["low"]).abs().le(small["close"] * 1e-4).mul(small["ret"].le(-0.095)).mean()), 5),
            },
        }
    report["liquidity"] = liquidity

    # 容量：1000 万本金 + 50 只等权 + 单票 10% 参与率
    capital = 1_000_000.0
    notice = 10
    per_stock = capital * 0.95 / 50
    required_adv = per_stock / (notice / 100.0)
    report["capacity"] = {
        "capital": capital,
        "holdings": 50,
        "gross": 0.95,
        "per_stock_notional": round(per_stock, 0),
        "participation_pct": notice,
        "required_adv_yuan": round(required_adv, 0),
        "required_adv_wan": round(required_adv / 1e4, 1),
    }

    report["seconds"] = round(time.time() - t0, 1)
    (out / "data-availability.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"完成 {report['seconds']}s → {out / 'data-availability.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
