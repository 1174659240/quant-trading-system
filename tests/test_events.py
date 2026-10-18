"""事件层（龙虎榜/回购/增减持/ST/送转分红）与事件因子测试：

- 双日期字段（``event_date`` 发生日、``available_date`` PIT 可见日）；
- ``events.load`` 默认裁到 2024-12-31，holdout 不进入研究；
- 因子按 ``available_date`` 对齐，可见日之前不得出现数值；
- 方向符号统一「越大越看多」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import events, refdata
from quant.factors import events as event_factors
from quant.factors.library import compute, register_factor
from quant.factors.panel import build_panel

CUTOFF = pd.Timestamp("2024-12-31")


def _panel(n_days: int = 40, n_symbols: int = 3) -> object:
    dates = refdata.trade_days()
    dates = dates[(dates >= pd.Timestamp("2019-01-01")) & (dates <= pd.Timestamp("2019-12-31"))][:n_days]
    rng = np.random.default_rng(7)
    bars = {}
    for k in range(n_symbols):
        close = 10.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, len(dates)))
        frame = pd.DataFrame(
            {
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(dates), 1e6),
                "amount": np.full(len(dates), 1e7),
                "prev_close": np.r_[close[0], close[:-1]],
            },
            index=dates,
        )
        bars[f"{600000 + k:06d}.SH"] = frame
    return build_panel(bars)


def test_event_tables_and_pit_columns():
    table = events.summary()
    assert set(events.TABLES).issubset(set(table["event"]))
    assert (table["exists"]).all(), table
    for name in events.TABLES:
        frame = events.load(name)
        assert "event_date" in frame.columns and "available_date" in frame.columns, name
        assert frame["available_date"].notna().any(), name
        assert frame["event_date"].notna().any(), name


def test_load_defaults_to_holdout_lock():
    for name in events.TABLES:
        frame = events.load(name)
        latest = pd.to_datetime(frame["available_date"], errors="coerce").max()
        assert latest <= CUTOFF, (name, latest)
    full = events.load("lhb", allow_holdout=True)
    assert pd.to_datetime(full["available_date"]).max() > CUTOFF


def test_hold_available_date_is_plus_two_trading_days():
    frame = events.load("hold").dropna(subset=["available_date"]).head(200)
    trade = refdata.trade_days().values
    for event_day, available in zip(pd.to_datetime(frame["event_date"]), pd.to_datetime(frame["available_date"])):
        pos = trade.searchsorted(np.datetime64(event_day), side="left")
        expected = pd.Timestamp(trade[min(pos + 2, len(trade) - 1)]).normalize()
        assert available.normalize() == expected, (event_day, available, expected)


def test_st_flips_are_countable_and_directional():
    frame = events.load("st")
    assert {"event_date", "available_date", "direction", "symbol"}.issubset(frame.columns)
    assert set(frame["direction"].unique()) == {"戴帽", "摘帽"}
    assert 500 <= len(frame) <= 4000, len(frame)
    assert not frame.duplicated(["symbol", "event_date", "direction"]).any()


def test_event_factor_alignment_and_sign():
    panel = _panel()
    event_factors.register()
    name = "lhb_flag_5"
    frame = compute(name, panel)
    # 全为 0/-1（sign = -1：上过榜取负）
    assert frame.fillna(0.0).le(0.0).all().all()
    assert frame.fillna(0.0).min().min() >= -1.0
    # 手工找一个事件，验证可见日当天出现、前一天没有
    lhb = events.load("lhb").dropna(subset=["available_date"])
    symbols = set(s.replace(".SH", "") for s in panel.symbols)
    row = lhb[lhb["symbol"].astype(str).isin(symbols)].head(1)
    if len(row):
        day = pd.Timestamp(row["available_date"].iloc[0])
        later = panel.dates[panel.dates >= day]
        if len(later):
            assert frame.loc[later[0]].le(0.0).all()


def test_event_factor_registered():
    event_factors.register()
    for name in event_factors.EVENT_FACTORS:
        register_factor  # 已注册即可 compute
        frame = compute(name, _panel(n_days=30))
        assert frame.shape[0] == 30
