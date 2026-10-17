"""复权与停牌补齐测试：纯离线，只用构造数据。"""

from __future__ import annotations

import pandas as pd

from quant.data.calendar import materialize_suspensions, save_calendar, load_calendar
from quant.data.corporate_actions import (
    apply_qfq,
    build_factor_index,
    load_events,
    save_events,
    sanity_report,
)
from quant.data.web_source import parse_symbol, tx_symbol

from _util import setup_console, tmpdir

setup_console()


def _frame(dates, closes):
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
            "amount": [1000 * c for c in closes],
        },
        index=pd.DatetimeIndex(dates),
    )


def test_parse_symbol_beijing_codes():
    assert parse_symbol("920000") == ("920000", "BJ")
    assert parse_symbol("920000.BJ") == ("920000", "BJ")
    assert parse_symbol("bj920000") == ("920000", "BJ")
    assert tx_symbol("920000") == "bj920000"
    assert parse_symbol("600519") == ("600519", "SH")
    assert parse_symbol("000001") == ("000001", "SZ")


def test_factor_index_cash_dividend():
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = pd.Series([10.0, 10.2, 10.4, 9.95, 10.1, 10.3], index=dates)
    events = pd.DataFrame(
        [{"symbol": "600000", "ex_date": dates[3], "bonus_shares": 0.0, "cash_per_share": 0.5}]
    )
    factor = build_factor_index(close, events)
    k = (10.4 - 0.5) / 10.4
    assert abs(factor.iloc[0] - k) < 1e-12
    assert abs(factor.iloc[2] - k) < 1e-12
    assert factor.iloc[3] == 1.0  # 除权日及之后不再缩放
    # 复权后除权日不出现假跌
    qfq = close * factor
    assert abs(float(qfq.iloc[3] / qfq.iloc[2] - 1) - (9.95 / (10.4 * k) - 1)) < 1e-12
    assert abs(float(qfq.iloc[3] / qfq.iloc[2] - 1)) < 0.01


def test_factor_index_bonus_shares():
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series([12.0, 12.0, 8.0, 8.2], index=dates)  # 10 送 5：12 → 8
    events = pd.DataFrame(
        [{"symbol": "600000", "ex_date": dates[2], "bonus_shares": 0.5, "cash_per_share": 0.0}]
    )
    factor = build_factor_index(close, events)
    assert abs(factor.iloc[0] - 12.0 / 1.5 / 12.0) < 1e-12
    assert abs(factor.iloc[2] - 1.0) < 1e-12
    qfq = close * factor
    assert abs(float(qfq.iloc[2] / qfq.iloc[1] - 1)) < 0.01  # 除权日不跳空


def test_apply_qfq_keeps_columns_and_scales_prices():
    dates = pd.bdate_range("2024-01-01", periods=5)
    frame = _frame(dates, [10.0, 10.2, 10.4, 9.95, 10.1])
    events = pd.DataFrame(
        [{"symbol": "600000", "ex_date": dates[3], "bonus_shares": 0.0, "cash_per_share": 0.5}]
    )
    out = apply_qfq(frame, events)
    assert "adj_factor" in out.columns
    assert set(["open", "high", "low", "close", "volume", "amount"]).issubset(out.columns)
    assert out["close"].iloc[0] < frame["close"].iloc[0]
    assert out["close"].iloc[-1] == frame["close"].iloc[-1]  # 最新价不动
    assert (out["volume"] == frame["volume"]).all()  # 成交量不下调
    report = sanity_report(out)
    assert report["rows"] == 5 and report["outliers"] == 0


def test_factor_index_skips_events_outside_range():
    dates = pd.bdate_range("2024-06-01", periods=3)
    close = pd.Series([10.0, 10.1, 10.2], index=dates)
    events = pd.DataFrame(
        [{"symbol": "600000", "ex_date": pd.Timestamp("2020-01-01"), "bonus_shares": 0.0, "cash_per_share": 1.0}]
    )
    factor = build_factor_index(close, events)
    assert (factor == 1.0).all()  # 早于数据起点的事件没有前收可参照，跳过


def test_events_roundtrip():
    with tmpdir("actions") as tmp:
        events = pd.DataFrame(
            [
                {"symbol": "600519", "ex_date": pd.Timestamp("2024-06-19"), "bonus_shares": 0.0, "cash_per_share": 30.876},
                {"symbol": "000001", "ex_date": pd.Timestamp("2024-06-14"), "bonus_shares": 0.0, "cash_per_share": 0.719},
            ]
        )
        path = save_events(events, tmp / "actions.csv")
        back = load_events(path)
        assert len(back) == 2
        assert set(back["symbol"]) == {"600519", "000001"}
        assert back["ex_date"].dtype.kind == "M"


def test_materialize_fills_suspended_days():
    dates = pd.bdate_range("2024-01-01", periods=10)
    calendar = pd.DatetimeIndex(dates)
    # 第 4、5 天停牌（原始数据里缺行）
    kept = [d for i, d in enumerate(dates) if i not in (3, 4)]
    frame = _frame(kept, [10.0 + i * 0.1 for i in range(len(kept))])
    out = materialize_suspensions(frame, calendar)
    assert len(out) == 10
    assert out.index.is_monotonic_increasing
    gap = out.loc[dates[3] : dates[4]]
    assert (gap["suspended"] == 1).all()
    assert (gap["volume"] == 0).all()
    assert (gap["open"] == gap["close"]).all() and (gap["high"] == gap["low"]).all()
    # 停牌日价格等于前一日收盘（不产生假收益）
    assert abs(float(out["close"].loc[dates[3]] - float(out["close"].loc[dates[2]]))) < 1e-12
    assert (out["suspended"] == 0).sum() == 8


def test_materialize_respects_listing_window():
    dates = pd.bdate_range("2024-01-01", periods=10)
    calendar = pd.DatetimeIndex(dates)
    frame = _frame(dates[3:6], [5.0, 5.1, 5.2])  # 第 4 天上市
    out = materialize_suspensions(frame, calendar)
    assert len(out) == 3  # 上市前不凭空造行
    assert out.index[0] == dates[3]
    assert (out["suspended"] == 0).all()


def test_calendar_roundtrip():
    with tmpdir("calendar") as tmp:
        dates = pd.bdate_range("2024-01-01", periods=20)
        path = save_calendar(dates, tmp / "calendar.csv")
        back = load_calendar(path)
        assert len(back) == 20
        assert back.equals(pd.DatetimeIndex(dates))
        assert load_calendar(tmp / "missing.csv").empty
