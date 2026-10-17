"""交易日历与停牌补齐。

上游日线接口有个共同的「聪明」行为：**停牌日不返回行**。如果直接把它当作序列，
停牌就变成了「价格连续」和「波动率为零」的假象，会系统性低估风险：

* 停牌期间无法交易，却可能在下一次复牌时一次性跳空（几十个百分点的缺口）；
* 停牌期间成交量为 0，任何按成交额/换手率算的因子都会失真；
* 回测若在停牌日成交，等于用不存在的流动性。

所以这里把交易日历作为**第一等公民**：日历来自指数日线（每个交易日必有），
个股在上市区间内按日历重排，缺的行补成「停牌行」——价格取前收、量为 0、打上
``suspended=1`` 标记。原始层保持上游原样（只增不改），停牌补齐只发生在派生层。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: 用作日历锚的指数：上证指数，沪深两市每个交易日都有该指数的一根日线。
CALENDAR_SYMBOL = "sh000001"


def date_range_from_bars(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """从任一 ``date`` 列的行情表里取出交易日序列。"""
    if frame is None or len(frame) == 0:
        return pd.DatetimeIndex([])
    column = "date" if "date" in frame.columns else frame.columns[0]
    return pd.DatetimeIndex(pd.to_datetime(frame[column]).dt.normalize().unique()).sort_values()


def fetch_calendar(start: str = "2015-01-01", *, kind: str = "tencent") -> pd.DatetimeIndex:
    """联网取交易日历（指数日线的日期序列）。

    只取日期：指数日线的 OHLC 常有小数位不自洽（上游四舍五入），走
    ``normalize_frame`` 的个股校验会被拒；日历本来也不需要价格。
    """
    from .web_source import SOURCES

    fetcher = SOURCES[kind]
    frame = fetcher(CALENDAR_SYMBOL, start=start, adjust="none")
    dates = date_range_from_bars(frame)
    if len(dates) == 0:
        raise RuntimeError("没能取到指数日线，交易日历为空")
    return dates


def save_calendar(dates: pd.DatetimeIndex, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"date": pd.to_datetime(dates).strftime("%Y-%m-%d")})
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)
    return path


def load_calendar(path: str | Path) -> pd.DatetimeIndex:
    path = Path(path)
    if not path.exists():
        return pd.DatetimeIndex([])
    frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
    return pd.DatetimeIndex(pd.to_datetime(frame["date"]).dt.normalize()).sort_values()


def materialize_suspensions(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """把个股序列按交易日历重排，并把缺失的交易日补成停牌行。

    只在该标的「首根日线 → 末根日线」之间补，上市前/退市后不凭空造行。
    返回值保证：索引严格递增、无重复、停牌行 ``volume == 0``、``suspended == 1``。
    """
    if len(frame) == 0:
        return frame
    out = frame.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index).normalize())
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if len(calendar) == 0:
        out["suspended"] = 0
        return out

    first, last = out.index[0], out.index[-1]
    window = calendar[(calendar >= first) & (calendar <= last)]
    if len(window) == 0:
        window = out.index
    missing = window.difference(out.index)
    if len(missing):
        filler = pd.DataFrame(index=missing, columns=out.columns, dtype=float)
        out = pd.concat([out, filler]).sort_index()
    out["suspended"] = out.index.isin(missing).astype(int) if len(missing) else 0
    for column in ("close", "open", "high", "low"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").ffill()
    for column in ("volume", "amount"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    # 停牌行价差为 0（开高低收都等于前收），避免凭空产生振幅
    suspended = out["suspended"] == 1
    if suspended.any():
        for column in ("open", "high", "low"):
            if column in out.columns:
                out.loc[suspended, column] = out.loc[suspended, "close"]
    return out
