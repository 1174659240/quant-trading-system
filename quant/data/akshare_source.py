"""AKShare 数据源（可选）。

akshare **不是**本系统的依赖，只有联网取真实 A 股行情时才需要::

    pip install akshare
    python -m scripts.fetch_data --symbols 600519.SH,000001.SZ --source akshare

取到后建议立刻落盘成 CSV，回测阶段一律走 CsvSource，避免联网带来的不可复现。
"""

from __future__ import annotations

import pandas as pd

from .base import DataSource, DataError, normalize_frame


def _require_akshare():
    try:
        import akshare as ak  # type: ignore
    except Exception as exc:  # pragma: no cover - 取决于本机是否装了 akshare
        raise DataError(
            "未安装 akshare，无法联网取数。安装：pip install akshare；"
            "或改用 CSV / 合成数据源（--source csv|synthetic）。"
        ) from exc
    return ak


def to_ak_symbol(symbol: str) -> str:
    """``600519.SH`` → ``600519``；``sh600519`` → ``600519``。"""
    s = symbol.strip()
    if "." in s:
        s = s.split(".")[0]
    for prefix in ("sh", "sz", "bj", "SH", "SZ", "BJ"):
        if s.startswith(prefix) and s[len(prefix):].isdigit():
            return s[len(prefix):]
    return s


def exchange_of(symbol: str) -> str:
    s = symbol.upper()
    code = to_ak_symbol(s)
    if s.endswith(".SH") or s.startswith("SH") or code.startswith(("6", "9", "5")):
        return "SH"
    if s.endswith(".BJ") or code.startswith(("4", "8")):
        return "BJ"
    return "SZ"


class AkshareSource(DataSource):
    """A 股日线（前复权）。字段：日期/开盘/最高/最低/收盘/成交量/成交额。"""

    name = "akshare"

    def __init__(self, *, adjust: str = "qfq", symbols: list[str] | None = None) -> None:
        self.adjust = adjust
        self._symbols = list(symbols or [])

    def load(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        ak = _require_akshare()
        code = to_ak_symbol(symbol)
        start_date = pd.Timestamp(start or "2015-01-01").strftime("%Y%m%d")
        end_date = pd.Timestamp(end or pd.Timestamp.today()).strftime("%Y%m%d")
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=self.adjust,
            )
        except Exception as exc:
            raise DataError(f"akshare 取 {symbol} 失败: {exc}") from exc
        if raw is None or len(raw) == 0:
            raise DataError(f"akshare 返回空数据: {symbol} ({start_date}~{end_date})")
        frame = normalize_frame(raw, symbol=symbol)
        frame["symbol"] = symbol
        return frame

    def load_index(self, symbol: str = "000300.SH", start: str | None = None, end: str | None = None) -> pd.DataFrame:
        ak = _require_akshare()
        code = to_ak_symbol(symbol)
        try:
            raw = ak.stock_zh_index_daily(symbol=("sh" if exchange_of(symbol) == "SH" else "sz") + code)
        except Exception as exc:
            raise DataError(f"akshare 取指数 {symbol} 失败: {exc}") from exc
        frame = normalize_frame(raw, symbol=symbol)
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index <= pd.Timestamp(end)]
        return frame

    def symbols(self) -> list[str]:
        return list(self._symbols)
