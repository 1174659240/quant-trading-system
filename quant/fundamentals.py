"""财务三表（东财数据中心 RPT_F10_FINANCE_*）：按报告期全市场拉取，落 parquet。

为什么不用 akshare 的 ``stock_lrb_em`` 等封装：它们只给 15/15/12 列摘要，
没有短期借款、商誉、资本开支、折旧摊销、应计项，做不了质量/现金流因子。

口径与偏差（任何用到它的回测报告都要照抄）：
1. **每报告期只保留修订后的一行**（东财同 baostock，无版本号），因此存在轻微后视偏差；
2. ``NOTICE_DATE`` 是公告日，因子可用性上界用它而不是 ``REPORT_DATE``；
3. 季报/中报是**年初至今累计值**，单季值需同一年内差分；TTM = 最近四个单季之和；
4. 只保留 ``SECURITY_TYPE_CODE in ("058001001","058001002")``（A 股与科创板），
   北交所不在 baostock universe 内，本表也不覆盖；
5. 报告期早于上市日或晚于退市日的行，由下游按 universe 区间裁剪。
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests

FUND_DIR = Path("D:/quant-data/fundamentals")
EM_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
PAGE_SIZE = 500

REPORTS = {
    "profit": "RPT_F10_FINANCE_GINCOME",
    "balance": "RPT_F10_FINANCE_GBALANCE",
    "cashflow": "RPT_F10_FINANCE_GCASHFLOW",
    "indicator": "RPT_F10_FINANCE_MAINFINADATA",
}

KEYS = ["SECUCODE", "SECURITY_CODE", "SECURITY_NAME_ABBR", "ORG_CODE", "REPORT_DATE", "REPORT_TYPE", "NOTICE_DATE", "CURRENCY"]

COLUMNS = {
    "profit": KEYS + [
        "TOTAL_OPERATE_INCOME", "OPERATE_INCOME", "OPERATE_COST", "TOTAL_OPERATE_COST",
        "OPERATE_PROFIT", "TOTAL_PROFIT", "NETPROFIT", "PARENT_NETPROFIT", "DEDUCT_PARENT_NETPROFIT",
        "BASIC_EPS", "DILUTED_EPS", "RESEARCH_EXPENSE", "SALE_EXPENSE", "MANAGE_EXPENSE",
        "FINANCE_EXPENSE", "OPERATE_TAX_ADD", "ASSET_IMPAIRMENT_INCOME", "CREDIT_IMPAIRMENT_INCOME",
        "OTHER_INCOME", "INVEST_INCOME", "FAIRVALUE_CHANGE_INCOME", "ASSET_DISPOSAL_INCOME",
        "NONBUSINESS_INCOME", "NONBUSINESS_EXPENSE", "INCOME_TAX", "MINORITY_INTEREST",
        "INTEREST_INCOME", "FEE_COMMISSION_INCOME",
    ],
    "balance": KEYS + [
        "MONETARYFUNDS", "ACCOUNTS_RECE", "NOTE_RECE", "INVENTORY", "TOTAL_CURRENT_ASSETS",
        "FIXED_ASSET", "CIP", "INTANGIBLE_ASSET", "GOODWILL", "LONG_PREPAID_EXPENSE",
        "TOTAL_ASSETS", "SHORT_LOAN", "ACCOUNTS_PAYABLE", "NOTE_PAYABLE", "ADVANCE_RECEIVABLES",
        "CONTRACT_LIAB", "TOTAL_CURRENT_LIAB", "LONG_LOAN", "BOND_PAYABLE", "TOTAL_LIABILITIES",
        "TOTAL_PARENT_EQUITY", "MINORITY_EQUITY", "TOTAL_EQUITY", "SHARE_CAPITAL",
        "CAPITAL_RESERVE", "UNASSIGN_RPOFIT", "SURPLUS_RESERVE", "NONCURRENT_LIAB_1YEAR",
        "LEASE_LIAB", "DEVELOP_EXPENSE", "TRADE_FINASSET", "TRADE_FINLIAB",
        "DEFER_TAX_ASSET", "DEFER_TAX_LIAB",
    ],
    "cashflow": KEYS + [
        "NETCASH_OPERATE", "NETCASH_INVEST", "NETCASH_FINANCE", "CONSTRUCT_LONG_ASSET",
        "SALES_SERVICES", "BUY_SERVICES", "PAY_STAFF_CASH", "PAY_ALL_TAX",
        "RECEIVE_INVEST_INCOME", "DISPOSAL_LONG_ASSET", "ACCEPT_INVEST_CASH",
        "RECEIVE_LOAN_CASH", "PAY_DEBT_CASH", "ASSIGN_DIVIDEND_PORFIT",
        "CCE_ADD", "END_CCE", "BEGIN_CCE", "TOTAL_OPERATE_INFLOW", "TOTAL_OPERATE_OUTFLOW",
        "TOTAL_INVEST_INFLOW", "TOTAL_INVEST_OUTFLOW", "TOTAL_FINANCE_INFLOW", "TOTAL_FINANCE_OUTFLOW",
        # 现金流量表补充资料（间接法）：折旧摊销/减值/递延税，用来算应计与真实现金流
        "FA_IR_DEPR", "IA_AMORTIZE", "LPE_AMORTIZE", "DEFER_INCOME_AMORTIZE",
        "ASSET_IMPAIRMENT", "FAIRVALUE_CHANGE_LOSS", "DEFER_TAX",
        "INVENTORY_REDUCE", "OPERATE_RECE_REDUCE", "OPERATE_PAYABLE_ADD", "NETPROFIT",
    ],
    # 主要财务指标：三表覆盖不到金融业（银行/保险/证券没有通用资产负债表的行），
    # 但这张表对全部公司都齐全，且自带 ROE/毛利率/同比与股本（BPS×股本＝归母净资产）。
    "indicator": KEYS + [
        "ORG_TYPE", "EPSJB", "EPSKCJB", "BPS", "MGJYXJJE", "MGWFPLR", "MGZBGJ",
        "TOTALOPERATEREVE", "MLR", "PARENTNETPROFIT", "KCFJCXSYJLR",
        "TOTALOPERATEREVETZ", "PARENTNETPROFITTZ", "KCFJCXSYJLRTZ",
        "ROEJQ", "ROEKCJQ", "ROIC", "XSJLL", "XSMLL", "ZCFZL",
        "TOTAL_SHARE", "A_FREE_SHARE", "B_FREE_SHARE",
    ],
}

TYPE_FILTER = '(SECURITY_TYPE_CODE in ("058001001","058001002"))'

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})


def _log(record: dict) -> None:
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    with (FUND_DIR / "fetch_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def page(report: str, flt: str, columns: str, number: int, *, size: int = PAGE_SIZE, retries: int = 4):
    """取一页；返回 (DataFrame, 总页数)。"""
    params = {
        "reportName": report,
        "columns": columns,
        "pageSize": str(size),
        "pageNumber": str(number),
        "source": "HSF10",
        "client": "PC",
        "filter": flt,
    }
    last = None
    for attempt in range(1, retries + 1):
        try:
            resp = _SESSION.get(EM_URL, params=params, timeout=45)
            payload = resp.json()
            result = payload.get("result") or {}
            return pd.DataFrame(result.get("data") or []), int(result.get("pages") or 0)
        except Exception as exc:  # noqa: BLE001 - 网络抖动统一重试
            last = exc
            time.sleep(1.0 * attempt)
    raise RuntimeError(f"东财请求失败 {report} p{number}: {last}")


def period_frame(table: str, report_date: str, *, pause: float = 0.05, verbose: bool = False) -> pd.DataFrame:
    """拉一个报告期的全表（自动翻页），只保留选中的列。"""
    report = REPORTS[table]
    wanted = COLUMNS[table]
    flt = f"(REPORT_DATE='{report_date}'){TYPE_FILTER}"
    first, pages = page(report, flt, ",".join(wanted), 1)
    frames = [first]
    pages = max(pages, 1)
    for number in range(2, pages + 1):
        if pause:
            time.sleep(pause)
        chunk, _ = page(report, flt, ",".join(wanted), number)
        frames.append(chunk)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=wanted)
    for column in wanted:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[wanted]
    if verbose:
        print(f"    {table} {report_date}: {len(frame)} 行 / {pages} 页", flush=True)
    return frame


def path_for(table: str, report_date: str) -> Path:
    return FUND_DIR / table / f"{report_date}.parquet"


def periods(start: str = "2013-03-31", end: str | None = None) -> list[str]:
    """季度报告期序列（含最新已结束季度，不含未披露完的当期）。

    默认从 2013Q1 起：2014Q1 起的 TTM 需要往前数四个单季。
    """
    end = end or date.today().isoformat()
    out = []
    year = int(start[:4])
    while True:
        for month_day in ("03-31", "06-30", "09-30", "12-31"):
            day = f"{year}-{month_day}"
            if day < start:
                continue
            if day > end:
                return out
            out.append(day)
        year += 1


def build(
    *,
    tables: Sequence[str] = ("profit", "balance", "cashflow", "indicator"),
    start: str = "2013-03-31",
    end: str | None = None,
    pause: float = 0.05,
    limit: int | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> dict:
    """按报告期×表批量落盘；已存在的文件跳过（resume）。"""
    days = periods(start, end)
    jobs = [(t, d) for t in tables for d in days]
    if resume:
        jobs = [(t, d) for t, d in jobs if not path_for(t, d).exists()]
    if limit:
        jobs = jobs[:limit]
    FUND_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    rows = 0
    failures: dict[str, str] = {}
    t_all = time.perf_counter()
    for idx, (table, day) in enumerate(jobs, 1):
        t0 = time.perf_counter()
        try:
            frame = period_frame(table, day, pause=pause)
        except RuntimeError as exc:
            failures[f"{table}:{day}"] = str(exc)
            continue
        if len(frame) == 0:
            failures[f"{table}:{day}"] = "空表"
            continue
        path = path_for(table, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
        done += 1
        rows += len(frame)
        _log(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "table": table,
                "report_date": day,
                "rows": int(len(frame)),
                "cols": len(frame.columns),
                "seconds": round(time.perf_counter() - t0, 2),
                "bytes": path.stat().st_size,
            }
        )
        if verbose:
            elapsed = time.perf_counter() - t_all
            rate = idx / elapsed if elapsed else 0
            print(
                f"  财务 {idx}/{len(jobs)} {table} {day} {len(frame)} 行 "
                f"{time.perf_counter() - t0:.1f}s / 累计 {rate:.3f} 期每秒 / 剩余 {((len(jobs) - idx) / rate / 60 if rate else 0):.0f} 分钟",
                flush=True,
            )

    summary = {
        "tables": list(tables),
        "periods": len(days),
        "jobs": len(jobs),
        "written": done,
        "rows": rows,
        "failed": len(failures),
        "failures": failures,
        "seconds": round(time.perf_counter() - t_all, 1),
        "dir": str(FUND_DIR),
    }
    (FUND_DIR / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def load(table: str, report_date: str) -> pd.DataFrame:
    return pd.read_parquet(path_for(table, report_date))


def load_all(table: str, *, start: str | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted((FUND_DIR / table).glob("????-??-??.parquet")):
        if start and path.stem < start:
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def coverage() -> dict:
    out = {}
    for table in REPORTS:
        files = sorted((FUND_DIR / table).glob("????-??-??.parquet"))
        out[table] = {
            "periods": len(files),
            "first": files[0].stem if files else "",
            "last": files[-1].stem if files else "",
            "rows": int(sum(len(pd.read_parquet(p, columns=["SECURITY_CODE"])) for p in files)),
            "mb": round(sum(p.stat().st_size for p in files) / 1024 / 1024, 1),
        }
    return out
