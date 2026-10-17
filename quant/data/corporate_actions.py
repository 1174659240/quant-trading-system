"""除权除息事件与复权因子。

行情接口给的是**不复权**价格（东财 push2his 与腾讯 fqkline 两个「带复权的」接口在本机
长期不可用/被限流），所以复权必须自己算，而自己算的前提是拿到除权除息事件：

* 事件来源：东方财富数据中心 ``RPT_SHAREBONUS_DET``（按除权除息日全市场批量拉，一次
  约 4 万条，覆盖 2015 年至今），字段语义都是「每 10 股」；
* 前复权因子：对每个除权日，``k = (P0 - 每股派息) / (1 + 每股送转) / P0``，
  其中 ``P0`` 是除权日前一交易日的原始收盘价；除权日之前的每个交易日都乘上 k，
  多个事件连乘。最新交易日的因子恒为 1，历史价格向下缩放。

这个口径与券商/东财的前复权一致：分红与送转都会让历史价格变低，除权当天不产生假跌。
"""

from __future__ import annotations

import numpy as np

import json
import time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
_BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_COLUMNS = "SECURITY_CODE,EX_DIVIDEND_DATE,BONUS_IT_RATIO,BONUS_RATIO,PRETAX_BONUS_RMB"


class ActionError(RuntimeError):
    """事件拉取/解析失败。"""


def _get_json(params: dict, *, timeout: float) -> dict:
    url = _BASE + "?" + urlencode(params, quote_via=quote, safe="(),")
    request = Request(url, headers={"User-Agent": _UA, "Referer": "https://data.eastmoney.com/"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def fetch_events(
    start: str = "2015-01-01",
    *,
    page_size: int = 500,
    timeout: float = 20.0,
    pause: float = 0.15,
    max_pages: int = 300,
    verbose: bool = False,
) -> pd.DataFrame:
    """全市场除权除息事件。

    返回列：``symbol, ex_date, bonus_shares, cash_per_share``（后两个已是「每股」口径）。
    """
    rows: list[dict] = []
    page = 1
    total_pages = 1
    while page <= total_pages and page <= max_pages:
        params = {
            "reportName": "RPT_SHAREBONUS_DET",
            "columns": _COLUMNS,
            "pageSize": str(page_size),
            "pageNumber": str(page),
            "sortColumns": "EX_DIVIDEND_DATE",
            "sortTypes": "-1",
            "filter": f"(EX_DIVIDEND_DATE>='{start}')",
        }
        data = _get_json(params, timeout=timeout)
        result = data.get("result") or {}
        if not result:
            raise ActionError(f"第 {page} 页没有结果：{str(data)[:200]}")
        total_pages = int(result.get("pages") or 1)
        rows.extend(result.get("data") or [])
        if verbose:
            print(f"  事件 {page}/{total_pages} 页，累计 {len(rows)} 条")
        page += 1
        time.sleep(max(0.0, pause))
    if not rows:
        raise ActionError("没有拿到任何除权除息事件")

    frame = pd.DataFrame(rows)
    frame["symbol"] = frame["SECURITY_CODE"].astype(str).str.zfill(6)
    frame["ex_date"] = pd.to_datetime(frame["EX_DIVIDEND_DATE"]).dt.normalize()
    bonus = pd.to_numeric(frame["BONUS_IT_RATIO"], errors="coerce").fillna(0.0)
    transfer = pd.to_numeric(frame["BONUS_RATIO"], errors="coerce").fillna(0.0)
    frame["bonus_shares"] = (bonus + transfer) / 10.0  # 每 10 股送转 → 每股
    frame["cash_per_share"] = pd.to_numeric(frame["PRETAX_BONUS_RMB"], errors="coerce").fillna(0.0) / 10.0
    out = frame[["symbol", "ex_date", "bonus_shares", "cash_per_share"]].copy()
    out = out[(out["bonus_shares"] > 0) | (out["cash_per_share"] > 0)]
    return out.sort_values(["symbol", "ex_date"]).reset_index(drop=True)


def save_events(events: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = events.copy()
    out["ex_date"] = pd.to_datetime(out["ex_date"]).dt.strftime("%Y-%m-%d")
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)
    return path


def load_events(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "ex_date", "bonus_shares", "cash_per_share"])
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str}, parse_dates=["ex_date"])
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    return frame


def build_factor_index(close: pd.Series | pd.DataFrame, events: pd.DataFrame | None) -> pd.Series:
    """单只标的的前复权因子序列（index 与传入价格一致）。

    ``close`` 可以是 Series 或列名含 ``close`` 的 DataFrame。

    口径：因子(d) = 所有 ex_date > d 的事件 (p0 - cash) / (1 + bonus) / p0 的乘积，
    p0 取事件前最近一根收盘价（不复权）。事件早于数据起点、或 k 越界（<=0 或 >2）、
    或分红送股都为 0 的，跳过。

    实现用 numpy 前缀积 + searchsorted（O(事件数 log 事件数)），不再逐事件扫全表。
    """
    if isinstance(close, pd.DataFrame):
        close = close["close"]
    close = close.astype(float)
    index = pd.DatetimeIndex(close.index)
    factor = pd.Series(1.0, index=close.index, dtype=float)
    if events is None or len(events) == 0 or len(index) == 0:
        return factor

    events = events.sort_values("ex_date")
    index_arr = index.values.astype("datetime64[ns]")
    close_arr = close.to_numpy(dtype=float)

    dates, factors = [], []
    unique_dates = pd.DatetimeIndex(pd.to_datetime(events["ex_date"]).dt.normalize())
    bonus_arr = pd.to_numeric(events.get("bonus_shares"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cash_arr = pd.to_numeric(events.get("cash_per_share"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for ex_date, bonus, cash in zip(unique_dates.values.astype("datetime64[ns]"), bonus_arr, cash_arr):
        if bonus == 0.0 and cash == 0.0:
            continue
        pos = int(index_arr.searchsorted(ex_date, side="left")) - 1
        if pos < 0:
            continue  # 事件早于上市/数据起点，没有前收盘价可参照
        p0 = close_arr[pos]
        if not p0 > 0:
            continue
        k = (p0 - cash) / (1.0 + bonus) / p0
        if not (0.0 < k <= 2.0):  # 异常事件（如数据错位）跳过，避免污染整段历史
            continue
        dates.append(ex_date)
        factors.append(k)
    if not dates:
        return factor

    order = np.argsort(np.array(dates))
    sorted_dates = np.array(dates)[order]
    sorted_k = np.array(factors, dtype=float)[order]
    # 后缀积：suffix[i] = k(i..n-1) 的乘积；suffix[n] = 1
    suffix = np.ones(len(sorted_k) + 1, dtype=float)
    suffix[:-1] = np.cumprod(sorted_k[::-1])[::-1]
    count = np.searchsorted(sorted_dates, index_arr, side="right")  # ex_date <= d 的事件数
    factor.iloc[:] = suffix[count]
    return factor


def apply_qfq(frame: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    """给不复权日线加上 ``adj_factor`` 与前复权价（列名不变，价格被就地缩放）。"""
    out = frame.copy()
    factor = build_factor_index(out["close"].astype(float), events)
    out["adj_factor"] = factor.round(10)
    for column in ("open", "high", "low", "close"):
        out[column] = (out[column].astype(float) * factor).round(3)
    return out


def sanity_report(frame: pd.DataFrame) -> dict:
    """复权后体检：单日涨跌幅异常（>±21%）的条数。"""
    if len(frame) < 2:
        return {"rows": len(frame), "max_abs_pct": 0.0, "outliers": 0}
    pct = frame["close"].astype(float).pct_change().dropna()
    return {
        "rows": int(len(frame)),
        "max_abs_pct": round(float(pct.abs().max()), 4),
        "outliers": int((pct.abs() > 0.21).sum()),
    }
