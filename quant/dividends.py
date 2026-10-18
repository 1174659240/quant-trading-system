"""PIT 分红数据：东财 RPT_SHAREBONUS_DET（分红送配明细）。

设计要点
- 只有**已实施**（有除权除息日）的现金分红才进入 TTM 口径：``ex_date <= t`` 是公开事实，
  因此 ``div_yield(t) = 过去 365 天每股现金分红之和 / 收盘价`` 不含未来函数；
- 每 10 股派息税前 ``PRETAX_BONUS_RMB`` 是「元/10 股」，每股 = /10；
- 报告期与预案公告日一并保留，便于换用「预案口径」做敏感性；
- 落盘 ``D:/quant-data/ref/dividends.parquet``，断点续跑按报告期合并写。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

REF_DIR = Path("D:/quant-data/ref")
DIVIDENDS = REF_DIR / "dividends.parquet"
HOLDOUT_DIVIDENDS = Path("D:/quant-data/holdout/dividends.parquet")
META = REF_DIR / "dividends_meta.json"
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/yjfp/"}
COLUMNS = (
    "SECURITY_CODE,SECURITY_NAME_ABBR,REPORT_DATE,PLAN_NOTICE_DATE,NOTICE_DATE,"
    "EQUITY_RECORD_DATE,EX_DIVIDEND_DATE,PRETAX_BONUS_RMB,BONUS_IT_RATIO,"
    "DIVIDENT_RATIO,ASSIGN_PROGRESS,TOTAL_SHARES,SECUCODE"
)
PAGE_SIZE = 500
HOLDOUT_START = pd.Timestamp("2025-01-01")


def periods(start: str = "2013-12-31", end: str = "2024-12-31") -> list[str]:
    """按季度末报告期生成区间（含端点）。"""
    days = pd.date_range(start=start, end=end, freq="QE")
    return [day.strftime("%Y-%m-%d") for day in days]


def _page(report_date: str, page: int, *, retries: int = 4, timeout: float = 30.0) -> dict:
    params = {
        "sortColumns": "PLAN_NOTICE_DATE",
        "sortTypes": "-1",
        "pageSize": str(PAGE_SIZE),
        "pageNumber": str(page),
        "reportName": "RPT_SHAREBONUS_DET",
        "columns": COLUMNS,
        "source": "WEB",
        "client": "WEB",
        "filter": f"(REPORT_DATE='{report_date}')",
    }
    last = None
    for attempt in range(retries):
        try:
            response = requests.get(URL, params=params, headers=HEADERS, timeout=timeout)
            payload = response.json()
            result = payload.get("result") or {}
            return {
                "rows": result.get("data") or [],
                "pages": int(result.get("pages") or 0),
                "count": int(result.get("count") or 0),
            }
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 + attempt)
    return {"rows": [], "pages": 0, "count": 0, "error": last}


def clean(rows: list[dict], *, allow_holdout: bool = False) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.rename(
        columns={
            "SECURITY_CODE": "symbol",
            "SECURITY_NAME_ABBR": "code_name",
            "REPORT_DATE": "report_date",
            "PLAN_NOTICE_DATE": "plan_notice",
            "NOTICE_DATE": "notice_date",
            "EQUITY_RECORD_DATE": "record_date",
            "EX_DIVIDEND_DATE": "ex_date",
            "PRETAX_BONUS_RMB": "pretax_bonus_10",
            "BONUS_IT_RATIO": "bonus_ratio",
            "DIVIDENT_RATIO": "div_ratio",
            "ASSIGN_PROGRESS": "progress",
            "TOTAL_SHARES": "total_shares",
            "SECUCODE": "secucode",
        }
    )
    for column in ("report_date", "plan_notice", "notice_date", "record_date", "ex_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["cash_per_share"] = pd.to_numeric(frame["pretax_bonus_10"], errors="coerce").fillna(0.0) / 10.0
    frame["bonus_ratio"] = pd.to_numeric(frame["bonus_ratio"], errors="coerce")
    frame["total_shares"] = pd.to_numeric(frame["total_shares"], errors="coerce")
    frame["div_ratio"] = pd.to_numeric(frame["div_ratio"], errors="coerce")
    if not allow_holdout:
        frame = frame[frame["report_date"] < HOLDOUT_START]  # holdout 锁：默认不读 2025 起的报告期
    return frame.drop_duplicates(["symbol", "report_date", "ex_date", "cash_per_share"], keep="last")


def fetch_period(report_date: str, *, allow_holdout: bool = True, pause: float = 0.05) -> pd.DataFrame:
    """只抓单个报告期（holdout 用），返回清洗后的明细，不落盘。"""
    first = _page(report_date, 1)
    rows = first["rows"]
    for page in range(2, first["pages"] + 1):
        rows += _page(report_date, page)["rows"]
        time.sleep(pause)
    return clean(rows, allow_holdout=allow_holdout)


def fetch(start: str = "2013-12-31", end: str = "2024-12-31", *, pause: float = 0.05, resume: bool = True) -> dict:
    """抓取区间内所有季度报告期的分红明细，按报告期增量合并落盘。"""
    wanted = periods(start, end)
    existing = pd.read_parquet(DIVIDENDS) if resume and DIVIDENDS.is_file() else pd.DataFrame()
    have = set(existing["report_date"].dt.strftime("%Y-%m-%d")) if len(existing) else set()
    frames = [existing] if len(existing) else []
    log: list[dict] = []
    t_all = time.time()
    for report_date in wanted:
        if report_date in have:
            continue
        t0 = time.time()
        first = _page(report_date, 1)
        rows = first["rows"]
        for page in range(2, first["pages"] + 1):
            rows += _page(report_date, page)["rows"]
            time.sleep(pause)
        frame = clean(rows)
        frames.append(frame)
        item = {
            "report_date": report_date,
            "rows": int(len(frame)),
            "count": first["count"],
            "pages": first["pages"],
            "error": first.get("error"),
            "seconds": round(time.time() - t0, 1),
        }
        log.append(item)
        print(f"  {report_date}: {len(frame)} 行 / {first['pages']} 页 / {item['seconds']}s", flush=True)
        time.sleep(pause)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out = out.sort_values(["symbol", "ex_date", "report_date"]).reset_index(drop=True)
        REF_DIR.mkdir(parents=True, exist_ok=True)
        tmp = DIVIDENDS.with_suffix(".tmp.parquet")
        out.to_parquet(tmp, index=False)
        tmp.replace(DIVIDENDS)
    meta = {
        "rows": int(len(out)),
        "symbols": int(out["symbol"].nunique()) if len(out) else 0,
        "periods": sorted(out["report_date"].dt.strftime("%Y-%m-%d").unique().tolist()) if len(out) else [],
        "with_ex_date": int(out["ex_date"].notna().sum()) if len(out) else 0,
        "seconds": round(time.time() - t_all, 1),
        "log": log,
        "source": "eastmoney RPT_SHAREBONUS_DET",
        "fetched_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def fetch_holdout(start: str = "2025-03-31", end: str = "2026-09-30", *, pause: float = 0.05) -> dict:
    """抓 holdout 报告期分红，落 ``D:/quant-data/holdout/dividends.parquet``（不改样本内文件）。"""
    wanted = periods(start, end)
    existing = pd.read_parquet(HOLDOUT_DIVIDENDS) if HOLDOUT_DIVIDENDS.is_file() else pd.DataFrame()
    have = set(existing["report_date"].dt.strftime("%Y-%m-%d")) if len(existing) else set()
    frames = [existing] if len(existing) else []
    log: list[dict] = []
    for report_date in wanted:
        if report_date in have:
            continue
        frame = fetch_period(report_date, allow_holdout=True, pause=pause)
        frames.append(frame)
        log.append({"report_date": report_date, "rows": int(len(frame))})
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out = out.sort_values(["symbol", "ex_date", "report_date"]).reset_index(drop=True)
        HOLDOUT_DIVIDENDS.parent.mkdir(parents=True, exist_ok=True)
        tmp = HOLDOUT_DIVIDENDS.with_suffix(".tmp.parquet")
        out.to_parquet(tmp, index=False)
        tmp.replace(HOLDOUT_DIVIDENDS)
    meta = {
        "rows": int(len(out)),
        "symbols": int(out["symbol"].nunique()) if len(out) else 0,
        "periods": sorted(out["report_date"].dt.strftime("%Y-%m-%d").unique().tolist()) if len(out) else [],
        "log": log,
        "source": "eastmoney RPT_SHAREBONUS_DET",
        "note": "holdout 层：2025-01-01 起，只用于最终一次性验收",
    }
    (HOLDOUT_DIVIDENDS.parent / "dividends_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load(include_holdout: bool = False) -> pd.DataFrame:
    """读分红明细；默认把 ``ex_date`` 晚于 2024-12-31 的记录一并裁掉（holdout 锁）。

    ``include_holdout=True`` 时并上 ``holdout/dividends.parquet``（仅最终验收用）。
    """
    frames = []
    if DIVIDENDS.is_file():
        frames.append(pd.read_parquet(DIVIDENDS))
    if include_holdout and HOLDOUT_DIVIDENDS.is_file():
        frames.append(pd.read_parquet(HOLDOUT_DIVIDENDS))
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["symbol", "report_date", "ex_date", "cash_per_share"], keep="last"
    )
    for column in ("report_date", "plan_notice", "notice_date", "record_date", "ex_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    if not include_holdout:
        cutoff = HOLDOUT_START - pd.Timedelta(days=1)
        frame = frame[frame["ex_date"].isna() | (frame["ex_date"] <= cutoff)]
    return frame.reset_index(drop=True)


def _load_legacy() -> pd.DataFrame:
    """旧实现（保留以兼容调用方）：读分红明细并按 holdout 截断。"""
    frame = pd.read_parquet(DIVIDENDS)
    for column in ("report_date", "plan_notice", "notice_date", "record_date", "ex_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    cutoff = HOLDOUT_START - pd.Timedelta(days=1)
    keep = frame["ex_date"].isna() | (frame["ex_date"] <= cutoff)
    return frame[(frame["report_date"] <= cutoff) & keep]


def dps_panel(dates: pd.DatetimeIndex, symbols, *, window_days: int = 365) -> pd.DataFrame:
    """每个交易日「过去 window_days 内已除息」的每股现金分红之和（元/股）。

    按**自然日**铺开再滚动：除息日未必是面板里的交易日（如临时停牌），
    直接对面板日期 reindex 会把它丢掉。
    """
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates)))
    symbols = [str(symbol).zfill(6) for symbol in symbols]
    empty = pd.DataFrame(0.0, index=dates, columns=symbols)
    holdout = len(dates) > 0 and pd.Timestamp(dates[-1]) > HOLDOUT_START
    frame = load(include_holdout=holdout)
    frame = frame[frame["ex_date"].notna() & (frame["cash_per_share"] > 0)]
    frame = frame[(frame["ex_date"] >= dates.min() - pd.Timedelta(days=window_days)) & (frame["ex_date"] <= dates.max())]
    frame = frame[frame["symbol"].isin(symbols)]
    if frame.empty:
        return empty
    calendar = pd.date_range(dates.min() - pd.Timedelta(days=window_days), dates.max(), freq="D")
    daily = frame.groupby(["ex_date", "symbol"])["cash_per_share"].sum().unstack(fill_value=0.0)
    daily = daily.reindex(index=calendar, columns=symbols, fill_value=0.0).astype("float32")
    rolled = daily.rolling(window_days, min_periods=1).sum().astype("float32")
    return rolled.reindex(index=dates)


def coverage(dates: pd.DatetimeIndex, symbols) -> dict:
    """分红事件覆盖率：池内多少标的在窗口内至少有 1 次已实施现金分红。"""
    wanted = {str(symbol).zfill(6) for symbol in symbols}
    frame = load(include_holdout=pd.Timestamp(dates[-1]) > HOLDOUT_START)
    frame = frame[frame["ex_date"].notna() & (frame["ex_date"] >= pd.Timestamp(dates.min())) & (frame["ex_date"] <= pd.Timestamp(dates.max()))]
    frame = frame[frame["symbol"].isin(wanted)]
    pairs = frame.drop_duplicates("symbol")
    return {
        "events": int(len(frame)),
        "symbols": int(pairs["symbol"].nunique()),
        "universe": int(len(wanted)),
        "share": round(pairs["symbol"].nunique() / max(len(wanted), 1), 4),
    }
