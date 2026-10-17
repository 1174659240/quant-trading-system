"""面板数据（panel）：把「按标的存的多只 K 线」拼成「按字段存的宽表」。

多因子的第一课是**对齐**：所有因子必须在同一张 `index=交易日, columns=标的` 的表上
计算，否则错一位就是未来函数。本模块只做这一件事，且明确禁止前视：

* ``Panel.returns`` 是**当日相对昨收**的收益（用 ``prev_close``，没有前视）；
* ``Panel.forward_returns(n)`` 是**未来 n 日**收益，只用于**因子评估**，
  绝不能进策略的信号链路（名字里带 forward 就是提醒）；
* ``Panel.tradable`` 是「当天能否成交」的掩码（停牌/无量/价格异常），
  选股与因子计算都应优先用它过滤。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["Panel", "build_panel"]

_FIELDS = ("open", "high", "low", "close", "volume", "amount", "prev_close")


@dataclass
class Panel:
    """对齐后的多标的面板。``fields[field]`` 都是 ``index=日期, columns=标的``。"""

    fields: dict[str, pd.DataFrame]
    _cache: dict[str, object] = field(default_factory=dict, repr=False)

    # -- 基本属性 -----------------------------------------------------------

    @property
    def dates(self) -> pd.DatetimeIndex:
        for frame in self.fields.values():
            return frame.index
        return pd.DatetimeIndex([])

    @property
    def symbols(self) -> list[str]:
        for frame in self.fields.values():
            return list(frame.columns)
        return []

    def field(self, name: str) -> pd.DataFrame:
        if name not in self.fields:
            raise KeyError(f"面板里没有字段 {name!r}，现有：{sorted(self.fields)}")
        return self.fields[name]

    def has(self, name: str) -> bool:
        return name in self.fields

    def __len__(self) -> int:
        return len(self.dates)

    def __repr__(self) -> str:
        return f"<Panel {len(self.dates)} 个交易日 × {len(self.symbols)} 只标的，字段 {sorted(self.fields)}>"

    # -- 派生字段（都向后看，不含前视） -------------------------------------

    @property
    def close(self) -> pd.DataFrame:
        return self.field("close")

    @property
    def open(self) -> pd.DataFrame:
        return self.field("open") if self.has("open") else self.close

    @property
    def high(self) -> pd.DataFrame:
        return self.field("high") if self.has("high") else self.close

    @property
    def low(self) -> pd.DataFrame:
        return self.field("low") if self.has("low") else self.close

    @property
    def volume(self) -> pd.DataFrame:
        return self.field("volume") if self.has("volume") else self.close * np.nan

    @property
    def amount(self) -> pd.DataFrame:
        if self.has("amount"):
            return self.field("amount")
        return self.volume * self.close

    @property
    def prev_close(self) -> pd.DataFrame:
        if self.has("prev_close"):
            return self.field("prev_close")
        return self.close.shift(1)

    @property
    def returns(self) -> pd.DataFrame:
        """当日收益（相对昨收），无前视。"""
        if "returns" not in self._cache:
            base = self.prev_close.replace(0, np.nan)
            self._cache["returns"] = self.close / base - 1.0
        return self._cache["returns"]  # type: ignore[return-value]

    @property
    def log_amount(self) -> pd.DataFrame:
        if "log_amount" not in self._cache:
            self._cache["log_amount"] = np.log(self.amount.replace(0, np.nan))
        return self._cache["log_amount"]  # type: ignore[return-value]

    @property
    def tradable(self) -> pd.DataFrame:
        """可交易掩码：有量、有价、有昨收（停牌当天为 False）。"""
        if "tradable" not in self._cache:
            mask = (
                self.close.notna()
                & (self.close > 0)
                & self.volume.notna()
                & (self.volume > 0)
                & self.prev_close.notna()
                & (self.prev_close > 0)
            )
            self._cache["tradable"] = mask
        return self._cache["tradable"]  # type: ignore[return-value]

    # -- 评估用（含前视，禁止进信号链路） -----------------------------------

    def forward_returns(self, periods: int = 1) -> pd.DataFrame:
        """未来 ``periods`` 日收益（用于 IC / 分层评估，**含前视**）。"""
        if periods <= 0:
            raise ValueError("periods 必须为正整数")
        future = self.close.shift(-periods)
        return future / self.close - 1.0

    # -- 工具 ---------------------------------------------------------------

    def mask(self, frame: pd.DataFrame, *, fill: float = np.nan) -> pd.DataFrame:
        """把不可交易的格点抹成 NaN（因子/收益都应先过这一层）。"""
        return frame.where(self.tradable, fill)

    def count(self) -> pd.Series:
        """每个交易日有效标的数。"""
        return self.tradable.sum(axis=1)

    def to_long(self) -> pd.DataFrame:
        """转成长表（date, symbol, open…），落盘与人工核对方便。"""
        frames = []
        for name, frame in self.fields.items():
            # pandas 3.x 起 stack() 不再接受 dropna；新实现本来就不补 NA 行
            stacked = frame.stack()
            stacked.name = name
            frames.append(stacked)
        long = pd.concat(frames, axis=1)
        long.index.names = ["date", "symbol"]
        return long.reset_index()

    def slice(self, start=None, end=None) -> "Panel":
        return Panel({k: v.loc[start:end] for k, v in self.fields.items()})


def build_panel(
    bars: Mapping[str, pd.DataFrame],
    *,
    fields: Sequence[str] = _FIELDS,
) -> Panel:
    """把 ``{symbol: K线DataFrame}`` 拼成 ``Panel``（各字段一张宽表，日期取并集）。"""
    if not bars:
        raise ValueError("bars 为空，无法构建面板")
    out: dict[str, pd.DataFrame] = {}
    for field_name in fields:
        series_map: dict[str, pd.Series] = {}
        for symbol, frame in bars.items():
            if field_name not in frame.columns:
                if field_name in ("open", "high", "low") and "close" in frame.columns:
                    series_map[symbol] = frame["close"].astype(float)
                continue
            series = frame[field_name].astype(float)
            series = series[~series.index.duplicated(keep="last")]
            series_map[symbol] = series
        if not series_map:
            continue
        wide = pd.DataFrame(series_map)
        wide.index = pd.to_datetime(wide.index)
        out[field_name] = wide.sort_index()
    if "close" not in out:
        raise ValueError("bars 里没有 close 字段，无法构建面板")
    return Panel(out)


def symbols_with_history(panel: Panel, *, min_days: int = 60) -> list[str]:
    """过滤掉历史太短的标的（次新股在因子上不稳定）。"""
    counts = panel.close.notna().sum(axis=0)
    return [s for s in panel.symbols if int(counts.get(s, 0)) >= min_days]


def top_by(series_map: Mapping[str, pd.Series], names: Iterable[str]) -> dict[str, pd.Series]:
    return {name: series_map[name] for name in names if name in series_map}
