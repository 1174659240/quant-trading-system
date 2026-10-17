"""数据源抽象与 K 线表规范。

**K 线表契约**（全系统唯一约定，跨模块一律照此处理）::

    索引：DatetimeIndex，名字 ``date``，升序、唯一、无时区
    列  ：open high low close volume（必需，float64；volume 单位＝股）
          amount（可选，成交额，元）
          prev_close（可选，前收盘价；缺失由 close.shift(1) 补，首行用 open）

所有外部数据（CSV / 爬虫 / 导出文件）进入系统的第一站都是
``normalize_frame``，列名别名与排序问题都在这里一次性解决。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS: tuple[str, ...] = ("amount", "prev_close")

#: 常见外部列名 → 规范列名。覆盖 akshare / tushare / 东方财富 / 同花顺导出。
COLUMN_ALIASES: dict[str, str] = {
    "date": "date", "datetime": "date", "trade_date": "date", "日期": "date", "时间": "date",
    "代码": "symbol", "symbol": "symbol", "ts_code": "symbol", "股票代码": "symbol",
    "open": "open", "开盘": "open", "开盘价": "open", "今开": "open",
    "high": "high", "最高": "high", "最高价": "high",
    "low": "low", "最低": "low", "最低价": "low",
    "close": "close", "收盘": "close", "收盘价": "close", "最新价": "close", "今收": "close",
    "volume": "volume", "vol": "volume", "成交量": "volume", "成交量(股)": "volume", "成交量(手)": "volume_hand",
    "amount": "amount", "成交额": "amount", "成交额(元)": "amount", "turnover": "amount",
    "pre_close": "prev_close", "prev_close": "prev_close", "昨收": "prev_close", "前收盘": "prev_close",
    "pct_chg": "pct_chg", "涨跌幅": "pct_chg", "change": "change", "涨跌额": "change",
}


class DataError(RuntimeError):
    """数据不满足 K 线表契约时抛出，消息里带具体是哪一列/哪一行的问题。"""


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for col in frame.columns:
        key = str(col).strip().lower()
        target = COLUMN_ALIASES.get(key) or COLUMN_ALIASES.get(str(col).strip())
        if target:
            renamed[col] = target
    return frame.rename(columns=renamed)


def _parse_index(frame: pd.DataFrame) -> pd.DataFrame:
    """把日期放进 DatetimeIndex；支持日期是列、也是索引两种输入。"""
    if "date" in frame.columns:
        series = frame.pop("date")
    elif isinstance(frame.index, pd.DatetimeIndex):
        series = pd.Series(frame.index, index=frame.index)
    else:
        series = pd.Series(frame.index, index=frame.index)
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        bad = series[parsed.isna()].head(3).tolist()
        raise DataError(f"无法解析的日期值（前 3 个）: {bad}")
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(parsed.values, name="date")
    frame.index = frame.index.normalize()
    return frame


def normalize_frame(
    frame: pd.DataFrame,
    *,
    symbol: str = "",
    fill_prev_close: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """把任意来源的 K 线表整成契约格式。

    参数
    ----
    symbol
        仅用于报错信息，不影响结果。
    fill_prev_close
        是否需要 ``prev_close`` 列（涨跌停判定要用）。缺失时用
        ``close.shift(1)`` 补，首行退回 ``open``。
    drop_duplicates
        同一交易日重复行保留最后一条（外部导出常见重复）。
    """
    if frame is None or len(frame) == 0:
        raise DataError(f"{symbol or '<unknown>'}: 行情数据为空")

    frame = _rename_columns(pd.DataFrame(frame).copy())
    frame = _parse_index(frame)

    # 手数 → 股（A股外部数据常以“手”计）
    if "volume_hand" in frame.columns:
        if "volume" not in frame.columns:
            frame["volume"] = pd.to_numeric(frame["volume_hand"], errors="coerce") * 100
        frame = frame.drop(columns=["volume_hand"])

    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise DataError(
            f"{symbol or '<unknown>'}: 缺少必需列 {missing}；"
            f"现有列 {list(frame.columns)}。请参考 COLUMN_ALIASES 增加别名映射。"
        )

    for col in OHLCV_COLUMNS + tuple(c for c in OPTIONAL_COLUMNS if c in frame.columns):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.sort_index()
    if drop_duplicates:
        frame = frame[~frame.index.duplicated(keep="last")]

    if frame[list(OHLCV_COLUMNS)].isna().any().any():
        bad_rows = frame.index[frame[list(OHLCV_COLUMNS)].isna().any(axis=1)]
        raise DataError(
            f"{symbol or '<unknown>'}: OHLCV 存在缺失值，共 {len(bad_rows)} 行，"
            f"首个为 {bad_rows[0].date() if len(bad_rows) else 'n/a'}"
        )

    if (frame["volume"] < 0).any():
        raise DataError(f"{symbol or '<unknown>'}: 存在负成交量")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataError(f"{symbol or '<unknown>'}: 存在非正价格")

    bad_range = (frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (
        frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    )
    if bad_range.any():
        first = frame.index[bad_range][0]
        raise DataError(f"{symbol or '<unknown>'}: {first.date()} 的 high/low 与其 OHLC 不自洽")

    if fill_prev_close and "prev_close" not in frame.columns:
        frame["prev_close"] = frame["close"].shift(1)
        if len(frame):
            frame.iloc[0, frame.columns.get_loc("prev_close")] = frame.iloc[0]["open"]

    ordered = list(OHLCV_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in frame.columns]
    extra = [c for c in frame.columns if c not in ordered]
    return frame[ordered + extra]


class DataSource(ABC):
    """数据源接口：给标的与区间，返回契约格式的 K 线表。"""

    name = "base"

    @abstractmethod
    def load(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """返回单个标的的日线 K 线表。"""

    def symbols(self) -> Iterable[str]:
        """数据源里可用的标的列表；不支持的实现返回空。"""
        return ()
