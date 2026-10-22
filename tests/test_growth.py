"""``quant.factors.growth`` 的离线单测：全部用 ``_tmp/`` 下的合成数据湖，不读真实数据。

覆盖的是最容易出前视/口径错的几处：

1. 累计值 → 单季 → TTM 的换算（含缺季、跨年）；
2. 增长率两端为正的硬约束（Q2：负基期必须记 NaN）；
3. 按 ``NOTICE_DATE`` 散列（公告日之前取不到值）、上市日之前取不到财报；
4. 合同负债 + 预收款必须相加（CAS14 两字段覆盖期互补）；
5. 营运资本扩张差、非经常性依赖、盈利稳定性的符号；
6. 分块合成在某块空时按剩余块归一（Q1）。

真实数据湖的冒烟见 ``scripts/growth_factor_smoke.py``（不进单测，避免依赖 D 盘）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _util import tmpdir  # noqa: E402

from quant.factors import growth as G  # noqa: E402
from quant.factors import growth_alt as AL  # noqa: E402
from quant.factors.panel import build_panel  # noqa: E402

PERIODS = pd.period_range("2017Q1", "2019Q4", freq="Q").to_timestamp(how="end").normalize()
NOTICE_LAG = pd.Timedelta(days=30)

# 单季原值（12 个季度，2017Q1 起）：累计值由 _cumulative 在年内累加
SINGLES = {
    "000001": dict(
        deduct=[10, 10, 10, 10, 12, 13, 14, 15, 20, 20, 20, 20],
        parent=[10, 10, 10, 10, 12, 13, 14, 15, 20, 20, 20, 20],
        revenue=[100] * 12,
        cost=[80] * 4 + [75] * 8,
        rd=[5] * 4 + [6] * 4 + [7] * 4,
        ocf=[12] * 12,
        rece=[20] * 4 + [40] * 4 + [60] * 4,
        inventory=[10] * 4 + [20] * 4 + [30] * 4,
        contract=[50] * 4 + [75] * 4 + [100] * 4,
    ),
    "000002": dict(  # 基期为负：2017 年扣非单季 -5
        deduct=[-5, -5, -5, -5, 5, 5, 5, 5, 5, 5, 5, 5],
        parent=[-5, -5, -5, -5, 5, 5, 5, 5, 5, 5, 5, 5],
        revenue=[100] * 12,
        cost=[90] * 12,
        rd=[0] * 12,
        ocf=[5] * 12,
        rece=[20] * 12,
        inventory=[10] * 12,
        contract=[40] * 12,
    ),
    "000003": dict(  # 2019-01-01 才上市，但财报里有上市前的季度
        deduct=[10, 10, 10, 10, 12, 13, 14, 15, 20, 20, 20, 20],
        parent=[10, 10, 10, 10, 12, 13, 14, 15, 20, 20, 20, 20],
        revenue=[100] * 12,
        cost=[80] * 12,
        rd=[5] * 12,
        ocf=[12] * 12,
        rece=[20] * 12,
        inventory=[10] * 12,
        contract=[50] * 12,
    ),
}

IPO = {"000001": "2010-01-01", "000002": "2010-01-01", "000003": "2019-01-01"}


def _cumulative(values: list[float], start: str = "2017Q1") -> dict[pd.Timestamp, float]:
    """年内累加（A 股财报口径），返回 ``{报告期: 累计值}``。"""
    out: dict[pd.Timestamp, float] = {}
    running: dict[int, float] = {}
    for period, value in zip(pd.period_range(start, periods=len(values), freq="Q"), values):
        year = period.year
        running[year] = running.get(year, 0.0) + float(value)
        out[period.to_timestamp(how="end").normalize()] = running[year]
    return out


def _levels(values: list[float], start: str = "2017Q1") -> dict[pd.Timestamp, float]:
    """时点值（资产负债表），不做累加。"""
    return {
        period.to_timestamp(how="end").normalize(): float(value)
        for period, value in zip(pd.period_range(start, periods=len(values), freq="Q"), values)
    }


def _write_lake(root: Path, *, blank_rd: bool = False, drop: tuple[str, str] | None = None, extra_2025: bool = False) -> Path:
    lake = root / "fundamentals"
    rows = {"profit": [], "balance": [], "cashflow": [], "indicator": []}
    for code, profile in SINGLES.items():
        for period in PERIODS:
            if drop and drop == (code, str(period.date())):
                continue
            deduct = _cumulative(profile["deduct"])[period]
            revenue = _cumulative(profile["revenue"])[period]
            cost = _cumulative(profile["cost"])[period]
            rd = _cumulative(profile["rd"])[period]
            ocf = _cumulative(profile["ocf"])[period]
            base = {"SECURITY_CODE": code, "REPORT_DATE": str(period.date()), "NOTICE_DATE": str((period + NOTICE_LAG).date())}
            rows["profit"].append(
                {
                    **base,
                    "DEDUCT_PARENT_NETPROFIT": deduct,
                    "PARENT_NETPROFIT": _cumulative(profile["parent"])[period],
                    "RESEARCH_EXPENSE": np.nan if blank_rd else rd,
                    "OPERATE_INCOME": revenue,
                    "OPERATE_COST": cost,
                }
            )
            rows["balance"].append(
                {
                    **base,
                    "GOODWILL": 0.0,
                    "TOTAL_ASSETS": 100.0,
                    "TOTAL_LIABILITIES": 30.0,
                    "TOTAL_PARENT_EQUITY": 70.0,
                    "ACCOUNTS_RECE": _levels(profile["rece"])[period],
                    "INVENTORY": _levels(profile["inventory"])[period],
                    "CONTRACT_LIAB": _levels(profile["contract"])[period],
                    "ADVANCE_RECEIVABLES": np.nan,
                }
            )
            rows["cashflow"].append({**base, "NETCASH_OPERATE": ocf})
            rows["indicator"].append({**base, "ROIC": 8.5})
    for table, table_rows in rows.items():
        frame = pd.DataFrame(table_rows)
        folder = lake / table
        folder.mkdir(parents=True, exist_ok=True)
        for period, part in frame.groupby("REPORT_DATE"):
            part.to_parquet(folder / f"{pd.Timestamp(period).date()}.parquet", index=False)
    if extra_2025:  # 不该被读到的 holdout 期文件
        profit = lake / "profit"
        fake = pd.DataFrame(
            [
                {
                    "SECURITY_CODE": "000001",
                    "REPORT_DATE": "2025-03-31",
                    "NOTICE_DATE": "2025-04-30",
                    "DEDUCT_PARENT_NETPROFIT": 999.0,
                    "PARENT_NETPROFIT": 999.0,
                    "RESEARCH_EXPENSE": 999.0,
                    "OPERATE_INCOME": 999.0,
                    "OPERATE_COST": 999.0,
                }
            ]
        )
        fake.to_parquet(profit / "2025-03-31.parquet", index=False)
    return lake


def _panel(symbols: tuple[str, ...] = ("000001.SZ", "000002.SZ", "000003.SZ")) -> object:
    dates = pd.date_range("2018-01-02", "2019-12-31", freq="B")
    bars = {}
    for symbol in symbols:
        frame = pd.DataFrame(
            {
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 1e6,
                "amount": 1e7,
                "prev_close": 10.0,
            },
            index=dates,
        )
        bars[symbol] = frame
    return build_panel(bars)


def _factors(root: Path, **kwargs) -> G.GrowthFactors:
    lake = _write_lake(root, **kwargs)
    return G.GrowthFactors(_panel(), fund_dir=lake, ipo_dates=IPO)


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #


def test_single_quarter_keeps_q1_and_diffs_rest() -> None:
    index = PERIODS[:8]
    values = [10, 21, 33, 46, 12, 25, 39, 54]
    wide = pd.DataFrame({"A": values}, index=index)
    single = G.single_quarter(wide)["A"]
    assert abs(single.iloc[0] - 10) < 1e-9
    assert abs(single.iloc[1] - 11) < 1e-9
    assert abs(single.iloc[2] - 12) < 1e-9
    assert abs(single.iloc[3] - 13) < 1e-9
    assert abs(single.iloc[4] - 12) < 1e-9, "跨年第一季应原样保留，不能减上一年 Q4"


def test_single_quarter_missing_period_turns_nan() -> None:
    index = PERIODS[:4]
    wide = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0]}, index=index)
    wide = wide.drop(index[2])
    single = G.single_quarter(wide)["A"]
    assert np.isnan(single.loc[index[2]]), "缺了 Q3，Q3 应为空"
    assert np.isnan(single.loc[index[3]]), "缺了 Q3，Q4 的差分也应为空（不能拿 Q2 硬减）"


def test_ttm_needs_four_quarters() -> None:
    single = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=PERIODS[:5])
    values = G.ttm(single)["A"]
    assert values.iloc[:3].isna().all()
    assert abs(values.iloc[3] - 10) < 1e-9
    assert abs(values.iloc[4] - 14) < 1e-9


def test_level_yoy_rejects_negative_base() -> None:
    wide = pd.DataFrame({"A": [-5.0, np.nan, np.nan, np.nan, 20.0]}, index=PERIODS[:5])
    assert G.level_yoy(wide)["A"].isna().all(), "基期为负时增速必须记 NaN（Q2）"


def test_level_yoy_formula() -> None:
    wide = pd.DataFrame({"A": [40.0, np.nan, np.nan, np.nan, 54.0]}, index=PERIODS[:5])
    value = G.level_yoy(wide)["A"].iloc[4]
    assert abs(value - 0.35) < 1e-9


# --------------------------------------------------------------------------- #
# 读盘与 PIT
# --------------------------------------------------------------------------- #


def test_load_table_filters_pseudo_codes_and_cutoff() -> None:
    with tmpdir("growth-load") as root:
        lake = _write_lake(root, extra_2025=True)
        long = G.load_table("profit", fund_dir=lake)
        assert long["code"].str.fullmatch(r"\d{6}").all()
        assert long["report_date"].max() <= G.REPORT_CUTOFF, "2025 报告期文件不得被读入"
        assert (long["notice_date"] - long["report_date"]).dt.days.max() == 30


def test_notice_date_fallback_adds_110_days() -> None:
    with tmpdir("growth-notice") as root:
        lake = _write_lake(root)
        path = lake / "profit" / f"{PERIODS[0].date()}.parquet"
        frame = pd.read_parquet(path)
        frame["NOTICE_DATE"] = None
        frame.to_parquet(path, index=False)
        long = G.load_table("profit", fund_dir=lake)
        row = long[long["report_date"] == PERIODS[0]].iloc[0]
        assert (row["notice_date"] - row["report_date"]).days == 110


def test_scatter_waits_for_notice_date() -> None:
    with tmpdir("growth-pit") as root:
        factors = _factors(root)
        series = factors.g1_deduct_yoy()["000001.SZ"]
        report_day = pd.Timestamp("2018-12-31")
        notice_day = report_day + NOTICE_LAG
        assert np.isnan(series.loc[notice_day - pd.Timedelta(days=1)])
        assert abs(series.loc[notice_day] - 0.35) < 1e-6


def test_pre_ipo_financials_are_masked() -> None:
    with tmpdir("growth-ipo") as root:
        factors = _factors(root)
        series = factors.g1_deduct_yoy()["000003.SZ"]
        assert series.loc[:"2018-12-31"].isna().all(), "上市前不得有财报值（招股书历史的坑）"
        assert series.loc["2019-03-01":].notna().any()


# --------------------------------------------------------------------------- #
# 因子口径
# --------------------------------------------------------------------------- #


def test_g1_uses_ttm_and_rejects_negative_base() -> None:
    with tmpdir("growth-g1") as root:
        factors = _factors(root)
        good = factors.g1_deduct_yoy()
        assert abs(good["000001.SZ"].loc["2019-03-01"] - 0.35) < 1e-6
        assert np.isnan(good["000002.SZ"].loc["2019-03-01"]), "2018 年报的正 TTM 对应 2017 的负 TTM，必须记 NaN"


def test_contract_advance_sums_two_fields() -> None:
    with tmpdir("growth-e1") as root:
        factors = _factors(root)
        advance = factors.contract_advance()
        assert advance.loc["2018-12-31", "000001"] == 75.0, "ADVANCE_RECEIVABLES 为空时按 0 相加"
        value = factors.e1_contract_yoy()["000001.SZ"].loc["2019-03-01"]
        assert abs(value - 0.5) < 1e-6


def test_e2_working_capital_gap_sign() -> None:
    with tmpdir("growth-e2") as root:
        factors = _factors(root)
        value = factors.e2_wc_gap()["000001.SZ"].loc["2019-03-01"]
        assert value < 0, "应收+存货 翻倍而营收零增长 → 扩张差为正 → 因子取负"


def test_e6_nonrec_and_g5_stability_sign() -> None:
    with tmpdir("growth-e6") as root:
        factors = _factors(root)
        assert abs(factors.e6_nonrec()["000001.SZ"].loc["2019-03-01"]) < 1e-9, "归母=扣非时依赖度为 0"
        assert factors.g5_stability()["000001.SZ"].loc["2019-03-01"] < 0, "波动越大分数越负"


def test_rd_factors_start_in_2018_and_g8_in_2021() -> None:
    with tmpdir("growth-rd") as root:
        factors = _factors(root)
        assert factors.g6_rd_intensity().loc[:"2018-04-29"].isna().all().all(), "2018-03 报告期公告前没有研发块"
        assert factors.g6_rd_intensity()["000001.SZ"].loc["2019-03-01"] > 0
        assert factors.g8_gm_yoy().isna().all().all(), "2021-03 之前毛利率同比一律为空"


# --------------------------------------------------------------------------- #
# 合格集与合成
# --------------------------------------------------------------------------- #


def test_eligible_mask_excludes_negative_and_pre_ipo() -> None:
    with tmpdir("growth-mask") as root:
        factors = _factors(root)
        mask = G.eligible_mask(factors)
        day = pd.Timestamp("2019-03-01")
        assert bool(mask.loc[day, "000001.SZ"]), "正常成长股应在合格集内"
        assert not bool(mask.loc[day, "000002.SZ"]), "基期负增长 → G3 门槛不过"
        assert not bool(mask.loc[day, "000003.SZ"]), "上市不满 365 天"


def test_growth_score_renormalizes_missing_block() -> None:
    with tmpdir("growth-score") as root:
        factors = _factors(root, blank_rd=True)
        score = G.growth_score(factors.panel, factors=factors, min_names=2)
        day = pd.Timestamp("2019-03-01")
        row = score.loc[day]
        assert row.notna().any(), "研发块全空时应按剩余块归一，而不是整天为空"
        assert row.dropna().between(0, 1).all()


def test_rank_only_inside_eligible_set() -> None:
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [np.nan, np.nan, np.nan], "C": [3.0, 2.0, 1.0]}, index=pd.date_range("2020-01-01", periods=3))
    eligible = pd.DataFrame(False, index=values.index, columns=values.columns)
    eligible["A"] = True
    ranked = G._rank_pct(values, eligible, min_names=1)
    assert ranked["A"].notna().all()
    assert ranked["C"].isna().all(), "不合格的标的不能参与排名"


def test_growth_factors_returns_thirteen_plus_diagnostics() -> None:
    with tmpdir("growth-all") as root:
        factors = _factors(root)
        frames = factors.all_factors()
        core = [name for name in frames if not name.startswith("diag_")]
        assert len(core) == 13
        assert set(core) == set(sum(G.BLOCKS.values(), ()))
        for name, frame in frames.items():
            assert list(frame.columns) == list(factors.panel.symbols), name
            assert len(frame) == len(factors.panel.dates), name


# --------------------------------------------------------------------------- #
# 补充因子（growth_alt）：PIT、窗口与禁用列
# --------------------------------------------------------------------------- #

ALT_IPO = {"000001": "2010-01-01", "000002": "2010-01-01", "000003": "2010-01-01"}


def _alt_panel(symbols: tuple[str, ...] = ("000001.SZ",)) -> object:
    dates = pd.date_range("2020-01-02", "2020-12-31", freq="B")
    bars = {}
    for symbol in symbols:
        bars[symbol] = pd.DataFrame(
            {
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 1e6,
                "amount": 1e7,
                "prev_close": 10.0,
            },
            index=dates,
        )
    return build_panel(bars)


def _write_alt(root: Path, *, with_2025: bool = False) -> Path:
    alt = root / "alt"
    alt.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(code="000001", report_period="2020-03-31", notice_date="2020-04-15",
             indicator=AL.PARENT_INDICATOR, value=30.0, last_year=20.0,
             forecast_type="预增", ret_20d_after=0.5),
        dict(code="000001", report_period="2020-06-30", notice_date="2020-07-15",
             indicator=AL.PARENT_INDICATOR, value=10.0, last_year=40.0,
             forecast_type="预减", ret_20d_after=-0.2),
    ]
    if with_2025:
        rows.append(
            dict(code="000001", report_period="2025-03-31", notice_date="2025-04-15",
                 indicator=AL.PARENT_INDICATOR, value=5.0, last_year=4.0,
                 forecast_type="预增", ret_20d_after=0.0)
        )
    pd.DataFrame(rows).to_parquet(alt / "forecast.parquet", index=False)
    pd.DataFrame(
        [dict(code="000001", unlock_date="2020-06-30", unlock_shares=1e6, ret_20d_after=0.1)]
    ).to_parquet(alt / "unlock.parquet", index=False)
    pd.DataFrame(
        [
            dict(code="000001", as_of="2020-03-06", pledge_ratio=10.0, pledge_count=5),
            dict(code="000001", as_of="2020-03-13", pledge_ratio=12.0, pledge_count=6),
        ]
    ).to_parquet(alt / "pledge.parquet", index=False)
    pd.DataFrame(
        [dict(code="000001", report_period="2020-03-31", notice_date="2020-04-25",
              net_profit=33.0, net_profit_last=20.0, ret_20d_after=0.0)]
    ).to_parquet(alt / "express.parquet", index=False)
    return alt


def _alt(root: Path, **kwargs) -> AL.AltGrowthFactors:
    alt = _write_alt(root, **kwargs)
    return AL.AltGrowthFactors(_alt_panel(), alt_dir=alt, ipo_dates=ALT_IPO)


def test_alt_fc_yoy_waits_for_notice_date() -> None:
    with tmpdir("alt") as root:
        factors = _alt(root)
        yoy = factors.fc_yoy()
        assert np.isnan(yoy.loc["2020-04-14", "000001.SZ"]), "公告日前一天不能有预告值"
        assert abs(yoy.loc["2020-04-15", "000001.SZ"] - 0.5) < 1e-6, "30/20-1=0.5"
        assert abs(yoy.loc["2020-07-15", "000001.SZ"] - (-0.75)) < 1e-6, "10/40-1=-0.75"


def test_alt_fc_type_score_and_pit() -> None:
    with tmpdir("alt") as root:
        factors = _alt(root)
        score = factors.fc_type()
        assert np.isnan(score.loc["2020-04-14", "000001.SZ"])
        assert abs(score.loc["2020-04-15", "000001.SZ"] - 1.0) < 1e-6, "预增=+1"
        assert abs(score.loc["2020-07-15", "000001.SZ"] - (-1.0)) < 1e-6, "预减=-1"


def test_alt_window_sum_forward_and_backward() -> None:
    dates = pd.date_range("2020-01-02", "2020-12-31", freq="B")
    ctx = AL._Context(dates, ("000001.SZ",), (pd.Timestamp("2010-01-01"),))
    events = pd.DataFrame(
        [dict(code="000001", unlock_date="2020-06-30", unlock_shares=1e6)]
    )
    forward = AL._window_sum(events, ctx, days=60, forward=True)["000001.SZ"]
    assert abs(forward.loc["2020-05-15"] - 1e6) < 1e-3, "60 日窗口内应计入 6/30 的解禁"
    assert abs(forward.loc["2020-04-01"]) < 1e-6, "60 日窗口够不到的日期记 0"
    backward = AL._window_sum(events, ctx, days=20, forward=False)["000001.SZ"]
    assert abs(backward.loc["2020-07-10"] - 1e6) < 1e-3, "过去 20 日窗口应包含 6/30"
    assert abs(backward.loc["2020-08-14"]) < 1e-6


def test_alt_pledge_carries_forward_between_snapshots() -> None:
    with tmpdir("alt") as root:
        factors = _alt(root)
        ratio = factors.pledge_ratio()
        assert np.isnan(ratio.loc["2020-03-05", "000001.SZ"]), "首个快照之前不能有值"
        assert abs(ratio.loc["2020-03-09", "000001.SZ"] - (-10.0)) < 1e-6, "负向因子：10% -> -10"
        assert abs(ratio.loc["2020-03-16", "000001.SZ"] - (-12.0)) < 1e-6, "新快照覆盖旧值"


def test_alt_rejects_forbidden_columns_and_holdout_rows() -> None:
    with tmpdir("alt") as root:
        alt = _write_alt(root, with_2025=True)
        frame = AL.load_alt("forecast", alt_dir=alt, columns=["value", "ret_20d_after"])
        assert "ret_20d_after" not in frame.columns, "未来收益列必须读不到"
        assert "value" in frame.columns
        try:
            AL.load_alt("forecast", alt_dir=alt)
        except ValueError as exc:
            assert "holdout" in str(exc)
        else:
            raise AssertionError("含 2025+ 记录的表必须拒绝读取")


def test_alt_forecast_period_cap_excludes_annual_report() -> None:
    with tmpdir("alt") as root:
        alt = _write_alt(root)
        capped = AL.AltGrowthFactors(
            _alt_panel(), alt_dir=alt, ipo_dates=ALT_IPO, forecast_cap=pd.Timestamp("2020-03-31")
        )
        long = capped.forecast_long()
        assert len(long) == 1, "报告期 2020-06-30 的预告必须被 cap 掉"
        assert pd.to_datetime(long["report_period"]).max() <= pd.Timestamp("2020-03-31")
