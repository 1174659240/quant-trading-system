"""事件因子：把事件层（``quant.events``）转成可回测的横截面信号。

口径（全部只用 ``available_date <= t`` 的记录，事件因子生命线是 PIT）
- ``lhb_netbuy_20``：过去 20 个交易日龙虎榜净买额之和 / 当日总市值。
- ``lhb_inst_net_20``：同上，但只算机构席位净买额。
- ``lhb_flag_5``：过去 5 个交易日是否上过榜（0/1，关注度/过热代理，方向取负）。
- ``buyback_flag_60``：过去 60 个交易日是否有回购公告（0/1）。
- ``buyback_ratio_60``：过去 60 个交易日回购计划金额上限之和 / 总市值。
- ``insider_net_20``：过去 20 个交易日大股东/董监高净增持金额 / 总市值。
- ``insider_buy_20``：过去 20 个交易日增持金额 / 总市值（只看买方向）。
- ``st_unflag_20``：过去 20 个交易日是否摘帽（0/1）。
- ``st_flag_20``：过去 20 个交易日是否戴帽（0/1，方向取负）。
- ``high_transfer_60``：过去 60 个交易日是否有高送转预案（送转 ≥ 每 10 股 5 股）。
- ``div_plan_60``：过去 60 个交易日是否有现金分红预案。

窗口口径天然可交易：t 日收盘可见 → 因子在 t 日成立 → 组合 t+1 开盘成交。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import events
from quant.factors.exposures import size_from_shares
from quant.factors.library import register_factor
from quant.factors.panel import Panel

_MATRIX_CACHE: dict[tuple, pd.DataFrame] = {}
_MCAP_CACHE: dict[tuple, pd.DataFrame] = {}


def _panel_key(panel: Panel, tag: str) -> tuple:
    return (tag, str(panel.dates[0]), str(panel.dates[-1]), len(panel.dates),
            str(panel.symbols[0]), str(panel.symbols[-1]), len(panel.symbols))


def _holdout(panel: Panel) -> bool:
    """面板末日超过 2024-12-31 → 读 holdout 事件（只在最终验收里发生）。"""
    return len(panel.dates) > 0 and pd.Timestamp(panel.dates[-1]) > events.CUTOFF


def _event_matrix(panel: Panel, name: str, value: str, *, prepare=None, tag: str = "") -> pd.DataFrame:
    """事件 → 日频矩阵：``available_date`` 对齐到第一个 >= 该日的面板交易日。"""
    key = _panel_key(panel, f"{name}:{value}:{tag}:{_holdout(panel)}")
    if key in _MATRIX_CACHE:
        return _MATRIX_CACHE[key]
    frame = events.load(name, allow_holdout=_holdout(panel))
    if prepare is not None:
        frame = prepare(frame)
    arr = np.zeros((len(panel.dates), len(panel.symbols)), dtype="float64")
    if len(frame):
        frame = frame.dropna(subset=["available_date"])
        index = panel.dates.searchsorted(pd.DatetimeIndex(frame["available_date"]).values, side="left")
        cols = pd.Index(panel.symbols).get_indexer(frame["symbol"].astype(str).str.zfill(6))
        valid = (index >= 0) & (index < len(panel.dates)) & (cols >= 0)
        if valid.any():
            np.add.at(arr, (np.asarray(index)[valid], np.asarray(cols)[valid]),
                      pd.to_numeric(frame[value], errors="coerce").fillna(0.0).values[valid])
    matrix = pd.DataFrame(arr, index=panel.dates, columns=panel.symbols)
    _MATRIX_CACHE[key] = matrix
    return matrix


def _ones_matrix(panel: Panel, frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    key = _panel_key(panel, f"ones:{tag}")
    if key in _MATRIX_CACHE:
        return _MATRIX_CACHE[key]
    arr = np.zeros((len(panel.dates), len(panel.symbols)), dtype="float64")
    if len(frame):
        frame = frame.dropna(subset=["available_date"])
        index = panel.dates.searchsorted(pd.DatetimeIndex(frame["available_date"]).values, side="left")
        cols = pd.Index(panel.symbols).get_indexer(frame["symbol"].astype(str).str.zfill(6))
        valid = (index >= 0) & (index < len(panel.dates)) & (cols >= 0)
        if valid.any():
            np.add.at(arr, (np.asarray(index)[valid], np.asarray(cols)[valid]), 1.0)
    matrix = pd.DataFrame(arr, index=panel.dates, columns=panel.symbols)
    matrix = (matrix > 0).astype("float64")
    _MATRIX_CACHE[key] = matrix
    return matrix


def _rolling_sum(matrix: pd.DataFrame, window: int) -> pd.DataFrame:
    return matrix.rolling(window, min_periods=1).sum()


def _rolling_any(matrix: pd.DataFrame, window: int) -> pd.DataFrame:
    return (matrix > 0).rolling(window, min_periods=1).sum().gt(0).astype("float64")


def _mcap(panel: Panel) -> pd.DataFrame:
    key = _panel_key(panel, "mcap")
    if key not in _MCAP_CACHE:
        _MCAP_CACHE[key] = size_from_shares(panel.close)
    return _MCAP_CACHE[key]


def _scaled(panel: Panel, name: str, value: str, window: int, *, prepare=None, tag: str = "") -> pd.DataFrame:
    total = _rolling_sum(_event_matrix(panel, name, value, prepare=prepare, tag=tag), window)
    mcap = _mcap(panel)
    out = total.divide(mcap.where(mcap > 0))
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# 因子实现
# ---------------------------------------------------------------------------


def lhb_netbuy(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    return _scaled(panel, "lhb", "net_buy", window)


def lhb_inst_net(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    return _scaled(panel, "lhb_inst", "inst_net", window)


def lhb_flag(panel: Panel, *, window: int = 5) -> pd.DataFrame:
    return _rolling_any(_ones_matrix(panel, events.load("lhb", allow_holdout=_holdout(panel)), "lhb"), window)


def buyback_flag(panel: Panel, *, window: int = 60) -> pd.DataFrame:
    return _rolling_any(_ones_matrix(panel, events.load("repurchase", allow_holdout=_holdout(panel)), "repurchase"), window)


def buyback_ratio(panel: Panel, *, window: int = 60) -> pd.DataFrame:
    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["plan_amount"] = out["plan_amount_high"].fillna(out["plan_amount_low"])
        return out

    return _scaled(panel, "repurchase", "plan_amount", window, prepare=prepare, tag="ratio")


def _signed_amount(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    amount = out["amount"].abs()
    out["amount"] = amount * np.sign(out["shares"].fillna(0.0))
    return out


def insider_net(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    return _scaled(panel, "hold", "amount", window, prepare=_signed_amount, tag="net")


def insider_buy(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        return _signed_amount(frame[frame["shares"] > 0])

    return _scaled(panel, "hold", "amount", window, prepare=prepare, tag="buy")


def st_unflag(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    left = events.load("st", allow_holdout=_holdout(panel))
    left = left[left["direction"] == "摘帽"]
    return _rolling_any(_ones_matrix(panel, left, "st_unflag"), window)


def st_flag(panel: Panel, *, window: int = 20) -> pd.DataFrame:
    became = events.load("st", allow_holdout=_holdout(panel))
    became = became[became["direction"] == "戴帽"]
    return _rolling_any(_ones_matrix(panel, became, "st_flag"), window)


def high_transfer(panel: Panel, *, window: int = 60) -> pd.DataFrame:
    frame = events.load("fhps", allow_holdout=_holdout(panel))
    picked = frame[frame["is_high_transfer"] > 0]
    return _rolling_any(_ones_matrix(panel, picked, "high_transfer"), window)


def div_plan(panel: Panel, *, window: int = 60) -> pd.DataFrame:
    frame = events.load("fhps")
    picked = frame[frame["div_ratio"].fillna(0) > 0]
    return _rolling_any(_ones_matrix(panel, picked, "div_plan"), window)


EVENT_FACTORS = {
    "lhb_netbuy_20": (lhb_netbuy, {"window": 20}, "龙虎榜净买额/总市值（20 日累计）"),
    "lhb_inst_net_20": (lhb_inst_net, {"window": 20}, "龙虎榜机构净买额/总市值（20 日累计）"),
    "lhb_flag_5": (lhb_flag, {"window": 5}, "近 5 日是否上榜（过热代理）"),
    "buyback_flag_60": (buyback_flag, {"window": 60}, "近 60 日是否有回购公告"),
    "buyback_ratio_60": (buyback_ratio, {"window": 60}, "近 60 日回购计划金额上限/总市值"),
    "insider_net_20": (insider_net, {"window": 20}, "近 20 日大股东/董监高净增持金额/总市值"),
    "insider_buy_20": (insider_buy, {"window": 20}, "近 20 日增持金额/总市值"),
    "st_unflag_20": (st_unflag, {"window": 20}, "近 20 日是否摘帽"),
    "st_flag_20": (st_flag, {"window": 20}, "近 20 日是否戴帽（方向取负）"),
    "high_transfer_60": (high_transfer, {"window": 60}, "近 60 日是否有高送转预案"),
    "div_plan_60": (div_plan, {"window": 60}, "近 60 日是否有现金分红预案"),
}

NEGATIVE = {"lhb_flag_5", "st_flag_20"}


def register() -> list[str]:
    """注册全部事件因子（``replace=True``，幂等）。"""
    names = []
    for name, (fn, params, description) in EVENT_FACTORS.items():
        sign = -1 if name in NEGATIVE else 1
        register_factor(
            name, fn, params=params,
            direction="negative" if sign < 0 else "positive",
            category="事件", description=description, sign=sign, replace=True,
        )
        names.append(name)
    return names
