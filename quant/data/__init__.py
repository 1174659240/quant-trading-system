"""行情数据层。"""

from __future__ import annotations

from .akshare_source import AkshareSource
from .base import (
    COLUMN_ALIASES,
    OHLCV_COLUMNS,
    DataError,
    DataSource,
    normalize_frame,
)
from .csv_source import CsvSource
from .store import DataStore
from .synthetic import SyntheticSource, make_gbm_frame
from .web_source import WebKlineSource, fetch_many, list_symbols

__all__ = [
    "COLUMN_ALIASES",
    "OHLCV_COLUMNS",
    "AkshareSource",
    "CsvSource",
    "DataError",
    "DataSource",
    "DataStore",
    "SyntheticSource",
    "WebKlineSource",
    "fetch_many",
    "list_symbols",
    "make_gbm_frame",
    "normalize_frame",
]
