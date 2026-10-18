"""中性化用的暴露矩阵：历史行业标签（PIT）与对数市值（PIT 股本）。

两个暴露都必须**当日可见**：

* 行业：用 baostock 的年度 PIT（``classify_date <= 当日``），退市股在 2025 年后的版本里
  被清空，但回测窗口截止 2024-12-31，所以退市股仍有归属；
* 市值：``收盘价 × shares_at(当日) 的总股本``，**不能用** ``MAINFINADATA.TOTAL_SHARE``
  （那是最新股本，2013 年的横截面里会提前知道 2026 年的股本）。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from quant import refdata

__all__ = ["industry_labels", "labels_from_frame", "log_size", "size_from_shares"]

_SUFFIX = re.compile(r"\.(SH|SZ|BJ)$", re.IGNORECASE)


def _bare(symbol: str) -> str:
    """``600519.SH`` → ``600519``（参考数据的代码不带交易所后缀）。"""
    return _SUFFIX.sub("", str(symbol))


def labels_from_frame(frame: pd.DataFrame, dates: pd.Index, symbols: pd.Index) -> pd.DataFrame:
    """把行业长表（``symbol / as_of / industry``）铺成「日期 × 代码」，按版本向前沿用。

    参考数据的代码是 6 位裸码，回测里用的是 ``600519.SH``，两边都要能对上；
    早于第一版分类日的日期用第一版（历史上第一版就是当时的分类）。
    """
    if len(frame) == 0:
        return pd.DataFrame(np.nan, index=dates, columns=symbols)
    frame = frame.copy()
    frame["as_of"] = frame["as_of"].astype(str)
    versions = sorted(frame["as_of"].unique())
    bare = pd.Index([_bare(symbol) for symbol in symbols])
    pivots = {}
    for version in versions:
        pivot = frame[frame["as_of"] == version].drop_duplicates("symbol").set_index("symbol")["industry"]
        pivot = pivot.reindex(bare)
        pivot.index = symbols
        pivots[version] = pivot
    version_dates = pd.to_datetime(pd.Index(versions)).sort_values()
    ordered = [pivots[str(stamp.date())] for stamp in version_dates]
    rows = {}
    for day in dates:
        pos = int(np.searchsorted(version_dates.to_numpy(), np.datetime64(day), side="right")) - 1
        rows[day] = ordered[max(pos, 0)]
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.reindex(index=dates, columns=symbols)


def industry_labels(dates: pd.Index, symbols: pd.Index) -> pd.DataFrame:
    """构造「日期 × 代码」的行业标签矩阵，按最近的年度版本向前沿用。"""
    return labels_from_frame(refdata.load_industry(), dates, symbols)


def _share_series(frame: pd.DataFrame | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """每只股票的「变动日 → 总股本」排序数组（只保留真实 A 股代码）。"""
    if frame is None:
        frame = refdata.load_shares()
    if len(frame) == 0:
        return {}
    frame = frame[(frame["is_real_code"] == 1) & frame["total_share"].notna()]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        group = group.sort_values("change_date")
        out[str(symbol)] = (
            pd.to_datetime(group["change_date"]).to_numpy(),
            group["total_share"].to_numpy(dtype=np.float64),
        )
    return out


def size_from_shares(close: pd.DataFrame, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """总市值矩阵（元）：``收盘价 × 当日可见总股本``，缺失为 NaN。"""
    series = _share_series(frame)
    days = close.index.to_numpy()
    out = np.full(close.shape, np.nan, dtype=np.float64)
    columns = list(close.columns)
    for position, symbol in enumerate(columns):
        entry = series.get(_bare(symbol))
        if entry is None:
            continue
        change_dates, totals = entry
        idx = np.searchsorted(change_dates, days, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        prices = close.iloc[:, position].to_numpy(dtype=np.float64, na_value=np.nan)
        shares = np.full(len(days), np.nan)
        shares[valid] = totals[idx[valid]]
        out[:, position] = prices * shares
    return pd.DataFrame(out, index=close.index, columns=columns)


def log_size(close: pd.DataFrame, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """``ln(总市值)``；市值为 0/负/缺失的格子留 NaN（中性化时按缺失处理）。"""
    mcap = size_from_shares(close, frame)
    out = np.log(mcap.where(mcap > 0))
    return out.astype("float32")
