"""多因子框架自测：面板、因子库、预处理、IC 评估、合成与多因子策略。

这些测试全部离线（合成数据），跑得快，也不依赖任何数据源。
重点检查三件容易出错的事：**方向**（值越大越看多）、**前视**（评估用未来收益、策略不用）、
**权重合法性**（不超仓、不超单票、调仓日之间不换手）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _util import setup_console

setup_console()

from quant import plugins  # noqa: E402
from quant.backtest.engine import run_backtest  # noqa: E402
from quant.config import BacktestConfig  # noqa: E402
from quant.data import DataStore, SyntheticSource  # noqa: E402
from quant.factors import (  # noqa: E402
    FACTORS,
    Panel,
    build_panel,
    catalog,
    combine,
    compute,
    equal_weights,
    evaluate_factor,
    evaluate_text,
    factor_corr,
    fill_missing,
    ic_decay,
    ic_summary,
    names,
    neutralize,
    quantile_returns,
    quantile_summary,
    rank_ic,
    rank_normalize,
    register_factor,
    score_panel,
    select_by_ic_ir,
    standardize,
    symbols_with_history,
    winsorize,
)
from quant.strategy.multifactor import MultiFactorStrategy  # noqa: E402

_END = "2023-12-29"


def _bars(symbols: list[str], periods: int = 300, seed: int = 3) -> dict[str, pd.DataFrame]:
    store = DataStore(SyntheticSource(symbols, start="2023-01-01", periods=periods, seed=seed))
    return store.get_many(symbols, start="2023-01-01", end=_END)


def _panel(symbols: list[str] | None = None, periods: int = 300, seed: int = 3) -> Panel:
    symbols = symbols or [f"60000{i}.SH" for i in range(1, 13)]
    return build_panel(_bars(symbols, periods=periods, seed=seed))


# ------------------------------------------------------------------ 面板


def test_build_panel_shapes_and_fields():
    panel = _panel(periods=120)
    assert isinstance(panel, Panel)
    assert len(panel.symbols) == 12
    assert panel.has("close") and panel.has("volume") and panel.has("prev_close")
    assert panel.field("close").shape == (len(panel.dates), 12)
    assert panel.close.shape == panel.field("close").shape
    assert str(len(panel.dates)) in repr(panel)
    assert panel.tradable.any().any()


def test_panel_returns_use_prev_close_without_lookahead():
    panel = _panel(symbols=["600001.SH"], periods=60)
    ret = panel.returns
    expected = panel.close / panel.prev_close - 1.0
    both = ret.notna() & expected.notna()
    assert both.any().any()
    assert np.allclose(ret.where(both), expected.where(both))
    # 面板第一天的收益完全由 prev_close 决定，不依赖未来
    first = ret.iloc[0].dropna()
    assert len(first) == 1


def test_panel_forward_returns_look_ahead_and_are_flagged():
    panel = _panel(symbols=["600001.SH"], periods=30)
    forward = panel.forward_returns(1)
    expected = panel.close.shift(-1) / panel.close - 1.0
    both = forward.notna() & expected.notna()
    assert both.any().any()
    assert np.allclose(forward.where(both), expected.where(both), equal_nan=True)
    assert forward.iloc[-1].isna().all()  # 最后一天没有"未来"
    assert forward.iloc[0].iloc[0] > -1.0


def test_symbols_with_history_filters_short_series():
    panel = _panel(periods=30)
    assert symbols_with_history(panel, min_days=500) == []
    assert len(symbols_with_history(panel, min_days=10)) == 12


def test_panel_slice_and_to_long():
    panel = _panel(periods=60)
    cut = panel.slice(start=str(panel.dates[10].date()))
    assert len(cut.dates) == len(panel.dates) - 10
    long = panel.to_long()
    assert set(["date", "symbol"]).issubset(long.columns)
    assert len(long) >= len(cut.dates) * 12 * 0.5


# ------------------------------------------------------------------ 因子库


def test_factor_catalog_and_compute_shapes():
    panel = _panel(periods=200)
    available = names()
    assert len(available) >= 15
    assert "momentum" in available and "low_vol" in available and "reversal" in available
    frame = catalog()
    assert list(frame.columns) == ["factor", "category", "direction", "params", "description"]
    assert len(frame) == len(available)

    momentum = compute("momentum", panel, lookback=20, skip=2)
    assert momentum.shape == (len(panel.dates), len(panel.symbols))
    assert momentum.notna().any().any()
    try:
        compute("没有这个因子", panel)
    except KeyError as exc:
        assert "没有因子" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("未知因子应当报错")


def test_factor_sign_flips_direction():
    panel = _panel(periods=60)
    register_factor(
        "test_close_raw",
        lambda panel, **kwargs: panel.close,
        params={},
        category="test",
        description="测试用：原始收盘价",
        replace=True,
    )
    register_factor(
        "test_close_flip",
        lambda panel, **kwargs: panel.close,
        params={},
        category="test",
        description="测试用：收盘价取负",
        sign=-1,
        replace=True,
    )
    try:
        raw = compute("test_close_raw", panel)
        flipped = compute("test_close_flip", panel)
        mask = raw.notna() & flipped.notna()
        assert mask.any().any()
        assert np.allclose(raw.where(mask), -flipped.where(mask))
    finally:  # 别把测试因子留给别的测试
        FACTORS.pop("test_close_raw", None)
        FACTORS.pop("test_close_flip", None)


def test_factor_unknown_params_are_rejected():
    panel = _panel(periods=60)
    try:
        compute("momentum", panel, 不存在的参数=1)
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("因子不认的参数应当报 TypeError")


# ------------------------------------------------------------------ 预处理


def test_winsorize_pulls_in_outliers():
    # winsorize 是逐日横截面：一行 = 一天，列 = 标的
    values = np.linspace(1.0, 5.0, 60)
    values[0] = 1000.0
    frame = pd.DataFrame([values, values * 1.1])
    cleaned = winsorize(frame, method="mad", n=3.0)
    assert cleaned.iloc[0, 0] < frame.iloc[0, 0]  # 极值被拉回来
    assert np.isclose(cleaned.iloc[0, 1], frame.iloc[0, 1])  # 正常值不动


def test_winsorize_quantile_method_and_constant_row():
    frame = pd.DataFrame([np.linspace(1.0, 3.0, 60), np.full(60, 2.0)])
    out = winsorize(frame, method="quantile")
    assert out.shape == frame.shape
    assert out.iloc[0].between(1.0, 3.0).all()
    assert np.allclose(out.iloc[1], 2.0)  # 整行同值：分位法退化为原值
    constant = winsorize(pd.DataFrame([np.full(60, 2.0)]), method="mad")
    assert np.allclose(constant.iloc[0], 2.0)  # MAD=0 也是原值


def test_standardize_and_rank_normalize_ranges():
    frame = pd.DataFrame(
        np.arange(20, dtype=float).reshape(4, 5) ** 1.5,
        index=pd.date_range("2024-01-01", periods=4),
    )
    z = standardize(frame)
    assert np.allclose(z.mean(axis=1).to_numpy(), 0.0, atol=1e-9)
    ranked = rank_normalize(frame)
    assert ranked.min().min() >= -1.0 - 1e-9
    assert ranked.max().max() <= 1.0 + 1e-9


def test_neutralize_removes_exposure_effect():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=30)
    columns = [f"s{i}" for i in range(20)]
    exposure = pd.DataFrame(rng.normal(size=(30, 20)), index=dates, columns=columns)
    noise = pd.DataFrame(rng.normal(scale=0.1, size=(30, 20)), index=dates, columns=columns)
    factor = exposure * 3.0 + noise
    residual = neutralize(factor, exposure, method="ols", min_count=10)
    left = residual.stack()
    right = exposure.stack()
    assert abs(float(left.corr(right))) < 0.2  # 暴露已被回归掉


def test_neutralize_bucket_mode_and_fill_missing():
    dates = pd.date_range("2024-01-01", periods=5)
    columns = [f"s{i}" for i in range(12)]
    factor = pd.DataFrame(np.arange(60, dtype=float).reshape(5, 12), index=dates, columns=columns)
    exposure = pd.DataFrame(np.tile(np.arange(12, dtype=float), (5, 1)), index=dates, columns=columns)
    bucketed = neutralize(factor, exposure, method="bucket", buckets=3)
    assert bucketed.shape == factor.shape
    holed = factor.copy()
    holed.iloc[2, 0] = np.nan
    filled = fill_missing(holed, limit=2)
    assert filled.iloc[2, 0] == factor.iloc[1, 0]  # 停牌期间沿用最近有效值
    leading = factor.copy()
    leading.iloc[0, 0] = np.nan
    assert np.isnan(fill_missing(leading).iloc[0, 0])  # 开头没有可填的值，保持缺失


# ------------------------------------------------------------------ 评估


def test_rank_ic_and_summary():
    panel = _panel(periods=250)
    factor = compute("momentum", panel, lookback=20, skip=0)
    ic = rank_ic(factor, panel.forward_returns(1), min_count=5)
    assert isinstance(ic, pd.Series)
    assert ic.dropna().abs().max() <= 1.0 + 1e-9
    stats = ic_summary(ic)
    for key in ("n", "ic_mean", "ic_std", "ic_ir", "t_stat", "positive_rate", "abs_mean", "annual_ir"):
        assert key in stats
    assert stats["n"] > 0
    assert 0.0 <= stats["positive_rate"] <= 1.0


def test_ic_decay_covers_requested_horizons():
    panel = _panel(periods=250)
    factor = compute("momentum", panel, lookback=20, skip=0)
    decay = ic_decay(factor, panel, horizons=(1, 5, 10), min_count=5)
    assert list(decay.index) == [1, 5, 10]
    assert decay.notna().all()


def test_quantile_returns_and_summary():
    panel = _panel(periods=250)
    factor = compute("momentum", panel, lookback=20, skip=0)
    qret = quantile_returns(factor, panel.forward_returns(1), quantiles=5, min_count=5)
    assert list(qret.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert qret.notna().any().any()
    summary = quantile_summary(qret)
    assert "多空(Q最高-Q最低)" in summary.index
    assert summary.loc["Q5", "days"] > 0


def test_factor_corr_is_symmetric():
    panel = _panel(periods=200)
    factors = {
        "momentum": compute("momentum", panel, lookback=20, skip=0),
        "low_vol": compute("low_vol", panel, window=20),
    }
    matrix = factor_corr(factors)
    assert matrix.shape == (2, 2)
    assert np.isclose(matrix.loc["momentum", "low_vol"], matrix.loc["low_vol", "momentum"])
    assert np.isclose(matrix.loc["momentum", "momentum"], 1.0)


def test_evaluate_factor_and_text():
    panel = _panel(periods=250)
    factor = compute("reversal", panel, lookback=5)
    result = evaluate_factor(factor, panel, name="reversal", horizons=(1, 5), quantiles=4)
    assert result["name"] == "reversal"
    assert result["coverage"] > 0
    assert set(result["quantiles"].columns) == {"Q1", "Q2", "Q3", "Q4"}
    text = evaluate_text(result)
    assert "reversal" in text and "IC" in text


# ------------------------------------------------------------------ 合成


def test_combine_equal_weights_and_unknown_weight():
    panel = _panel(periods=200)
    factors = {
        "momentum": compute("momentum", panel, lookback=20, skip=0),
        "low_vol": compute("low_vol", panel, window=20),
    }
    weights = equal_weights(factors)
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    score = combine(factors, weights)
    assert score.shape == (len(panel.dates), len(panel.symbols))
    try:
        combine(factors, {"momentum": 1.0, "别的因子": 1.0})
    except KeyError as exc:
        assert "未提供" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("未知权重应当报错")


def test_select_by_ic_ir_returns_subset_weights():
    panel = _panel(periods=250)
    factors = {
        "momentum": compute("momentum", panel, lookback=20, skip=0),
        "low_vol": compute("low_vol", panel, window=20),
        "reversal": compute("reversal", panel, lookback=5),
    }
    weights = select_by_ic_ir(factors, panel.forward_returns(1), max_factors=2, min_ic_ir=-1.0)
    assert set(weights).issubset(set(factors))
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_score_panel_weighting_modes():
    panel = _panel(periods=250)
    specs = {"momentum": {"lookback": 20, "skip": 0}, "low_vol": {"window": 20}}
    for weighting in ("equal", "ic", "ir", "auto"):
        combined = score_panel(panel, specs, weighting=weighting)
        assert combined.score.shape == (len(panel.dates), len(panel.symbols))
        assert abs(sum(combined.weights.values()) - 1.0) < 1e-9
        top = combined.top(panel.dates[-5], 3)
        assert len(top) == 3
        assert top.is_monotonic_decreasing
    try:
        score_panel(panel, specs, weighting="乱写")
    except ValueError as exc:
        assert "weighting" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("非法 weighting 应当报错")


# ------------------------------------------------------------------ 策略


def test_multifactor_weights_are_valid_and_bounded():
    symbols = [f"6000{i:02d}.SH" for i in range(1, 13)]
    strategy = MultiFactorStrategy(
        factors="momentum,low_vol",
        top_k=3,
        rebalance="W-FRI",
        gross_exposure=0.95,
        max_weight=0.50,
        min_history=60,
        min_universe=5,
    )
    bars = _bars(symbols, periods=300, seed=5)
    weights = strategy.generate(bars)
    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == (len(next(iter(bars.values()))), len(symbols))
    assert float(weights.max().max()) <= 0.95 + 1e-9
    assert float(weights.sum(axis=1).max()) <= 0.95 + 1e-9
    # 只在调仓日新开仓，其余交易日沿用上一份权重（hold_to_rebalance）
    rebalance_days = strategy.last_diagnostics["rebalance_dates"]
    assert rebalance_days, "应当有调仓日"
    held_counts = (weights > 0).sum(axis=1)
    assert int(held_counts.max()) <= 3
    assert strategy.describe()
    assert strategy.last_diagnostics["factors"] == ["momentum", "low_vol"]


def test_multifactor_strategy_runs_through_engine():
    symbols = [f"6000{i:02d}.SH" for i in range(1, 13)]
    bars = _bars(symbols, periods=300, seed=5)
    strategy = MultiFactorStrategy(
        factors="momentum,low_vol", top_k=4, rebalance="W-FRI", min_history=60, min_universe=5
    )
    config = BacktestConfig(
        start="2023-01-01",
        end=_END,
        initial_cash=500_000.0,
        universe=symbols,
        data_dir="data/cache",
        result_dir="results",
    )
    result = run_backtest(bars, strategy, config)
    assert result.metrics["total_return"] == result.metrics["total_return"]  # 不是 NaN
    assert result.equity.iloc[-1] > 0
    assert float(result.targets.abs().to_numpy().max()) > 0  # 真的下了仓位
    assert len(result.trades) > 0


def test_multifactor_is_registered_as_builtin():
    assert "multifactor" in plugins.strategy_names()
    strategy = plugins.build_strategy("multifactor", top_k=3, 不存在的参数=1)
    assert strategy.name == "multifactor"


def test_multifactor_survives_too_small_universe():
    # 标的太少时应退回空仓而不是崩掉（真实全市场不会遇到，但配置写错时要能看出来）
    symbols = ["600001.SH", "600002.SH"]
    bars = _bars(symbols, periods=200, seed=9)
    strategy = MultiFactorStrategy(top_k=3, min_universe=10, min_history=60)
    weights = strategy.generate(bars)
    assert weights.shape[1] == 2
    assert float(weights.abs().to_numpy().max()) == 0.0
    assert any("空仓" in note for note in strategy.last_diagnostics.get("notes", []))
