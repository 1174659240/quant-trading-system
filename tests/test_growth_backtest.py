"""阶段 3 组合构造的离线单测（合成数据，不碰数据湖）。

只测「选股/赋权/行业上限/现金缓冲/换手口径」这些纯逻辑——真实回测走
``scripts/growth_backtest.py``，需要数据湖，不进离线测试。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.growth_backtest import GrowthTopK, month_ends, turnover_stats

MONTHS = pd.DatetimeIndex(
    [
        "2020-01-31",
        "2020-02-28",
        "2020-03-31",
        "2020-04-30",
        "2020-05-29",
        "2020-06-30",
    ]
)
SYMBOLS = ["000001", "000002", "000003", "000004", "000005", "000006"]
SECTOR = {"000001": "A", "000002": "A", "000003": "A", "000004": "B", "000005": "B", "000006": "C"}


def _frame(values: dict[str, float], dates=MONTHS) -> pd.DataFrame:
    return pd.DataFrame({symbol: [values[symbol]] * len(dates) for symbol in SYMBOLS}, index=dates, dtype="float64")


def _sector_frame(dates=MONTHS) -> pd.DataFrame:
    return pd.DataFrame({symbol: [SECTOR[symbol]] * len(dates) for symbol in SYMBOLS}, index=dates, dtype=object)


def _bars(dates=MONTHS) -> dict[str, pd.DataFrame]:
    index = pd.DatetimeIndex([day + pd.Timedelta(days=1) for day in dates])
    return {
        symbol: pd.DataFrame(
            {
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 1_000_000.0,
                "amount": 10_000_000.0,
                "prev_close": 10.0,
            },
            index=index,
        )
        for symbol in SYMBOLS
    }


def test_sector_cap_limits_names_per_sector() -> None:
    score = _frame({symbol: 1.0 - index * 0.1 for index, symbol in enumerate(SYMBOLS)})
    eligible = pd.DataFrame(True, index=MONTHS, columns=SYMBOLS)
    strategy = GrowthTopK(
        score, eligible, sector=_sector_frame(), top_k=6, gross=0.95, max_weight=1.0,
        sector_cap=0.30, min_holdings=1,
    )
    weights = strategy.generate(_bars())
    per_name = 0.95 / 6
    allowed = int(0.30 / per_name + 1e-9)  # 每个行业最多几只
    assert allowed == 1, f"本测试假设单票 {per_name:.4f} 时每个行业只能进 1 只，实际 {allowed}"
    picks = strategy.last_diagnostics["picks"]["2020-01-31"]
    assert len(picks) == 3, f"3 个行业各 1 只，应该选 3 只，实际 {picks}"
    for sector in ("A", "B", "C"):
        assert sum(1 for symbol in picks if SECTOR[symbol] == sector) == 1
    row = weights.loc["2020-01-31"]
    assert abs(row.sum() - 3 * per_name) < 1e-9, "选不满 top_k 时应留现金，而不是把权重放大"
    assert row.max() <= 0.04 + 1e-9 or row.max() <= per_name + 1e-9


def test_ineligible_never_picked() -> None:
    score = _frame({"000001": 9.0, "000002": 8.0, "000003": 7.0, "000004": 6.0, "000005": 5.0, "000006": 4.0})
    eligible = pd.DataFrame(True, index=MONTHS, columns=SYMBOLS)
    eligible[["000001", "000004"]] = False
    strategy = GrowthTopK(
        score, eligible, sector=_sector_frame(), top_k=3, gross=0.95, max_weight=1.0,
        sector_cap=1.0, min_holdings=1,
    )
    weights = strategy.generate(_bars())
    picks = strategy.last_diagnostics["picks"]["2020-01-31"]
    assert "000001" not in picks and "000004" not in picks
    assert picks == ["000002", "000003", "000005"], picks
    assert weights.loc["2020-01-31", "000001"] == 0.0


def test_monthly_rebalance_holds_within_month() -> None:
    score = _frame({symbol: 1.0 - index * 0.1 for index, symbol in enumerate(SYMBOLS)})
    eligible = pd.DataFrame(True, index=MONTHS, columns=SYMBOLS)
    strategy = GrowthTopK(
        score, eligible, sector=_sector_frame(), top_k=2, gross=0.95, max_weight=1.0,
        sector_cap=1.0, min_holdings=1,
    )
    weights = strategy.generate(_bars())
    # 只有月末行有权重，且两个月末之间权重不变（月内不交易）
    assert weights.loc["2020-01-31"].sum() > 0
    assert weights.loc["2020-02-28"].sum() > 0
    held = weights.loc["2020-01-31"]
    assert held.max() <= 0.95 + 1e-9


def test_turnover_stats_matches_hand_calculation() -> None:
    weights = pd.DataFrame(0.0, index=MONTHS, columns=["a", "b"])
    weights.loc["2020-01-31", ["a", "b"]] = [0.5, 0.5]
    weights.loc["2020-02-28", ["a", "b"]] = [0.5, 0.5]
    weights.loc["2020-03-31", ["a", "b"]] = [0.0, 1.0]
    for rest in MONTHS[3:]:
        weights.loc[rest, ["a", "b"]] = [0.0, 1.0]
    stats = turnover_stats(weights, MONTHS)
    # 建仓 1.0 单边 0.5；第 2 月 0；第 3 月 |Δ|/2 = (0.5+0.5)/2 = 0.5 → 合计 1.0，6 个月 = 0.5 年
    assert abs(stats["ann_turnover_one_way"] - 2.0) < 1e-9, stats
    assert abs(stats["avg_month_turnover"] - 1.0 / 6) < 1e-9, stats
    assert stats["turnover_by_year"] == {2020: 1.0}, stats


def test_month_ends_picks_last_trading_day() -> None:
    dates = pd.DatetimeIndex(
        ["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02", "2020-03-31"]
    )
    marks = month_ends(dates, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-31"))
    assert list(marks) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-31")]
    trimmed = month_ends(dates, pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-31"))
    assert list(trimmed) == [pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-31")]


def test_empty_when_too_few_names() -> None:
    score = _frame({symbol: float(index) for index, symbol in enumerate(SYMBOLS)})
    eligible = pd.DataFrame(False, index=MONTHS, columns=SYMBOLS)
    strategy = GrowthTopK(score, eligible, top_k=5, min_holdings=3)
    weights = strategy.generate(_bars())
    assert float(np.nanmax(weights.to_numpy())) == 0.0
    assert strategy.last_diagnostics["notes"], "空仓要留下说明"
