"""方法论自测：行业+市值中性化、滚动 IC 无前视、分组选股（select_bucket）。

这三个点都是「跑得通但可能悄悄错了」的类型，所以测试只盯可证伪的性质：

* 中性化后的残差与行业均值、ln(市值) 应当**数值上正交**（不是"看起来像"）；
* 调仓日的因子权重只能用到当日之前的信息——把未来 K 线截掉，权重必须一模一样；
* ``select_bucket`` 取的分组必须与 ``evaluate.quantile_returns`` 的编号一致（1=最低分）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _util import setup_console

setup_console()

from quant.data import DataStore, SyntheticSource  # noqa: E402
from quant.factors.exposures import labels_from_frame  # noqa: E402
from quant.factors.neutral import codes_from_labels, residualize, residualize_many  # noqa: E402
from quant.strategy.multifactor import MultiFactorStrategy  # noqa: E402

_END = "2023-12-29"
_DAY = pd.Timestamp("2023-06-30")


def _bars(symbols: list[str], periods: int = 300, seed: int = 3) -> dict[str, pd.DataFrame]:
    store = DataStore(SyntheticSource(symbols, start="2023-01-01", periods=periods, seed=seed))
    return store.get_many(symbols, start="2023-01-01", end=_END)


# ------------------------------------------------------------- 暴露


def test_industry_labels_are_point_in_time_and_map_suffix():
    frame = pd.DataFrame(
        {
            "symbol": ["600001", "600001", "600002", "600002"],
            "as_of": ["2014-12-31", "2015-12-31", "2014-12-31", "2015-12-31"],
            "industry": ["K70房地产业", "C36汽车制造业", "J66货币金融服务", "J66货币金融服务"],
        }
    )
    dates = pd.DatetimeIndex(["2014-06-30", "2015-06-30", "2016-06-30"])
    symbols = pd.Index(["600001.SH", "600002.SH"])
    labels = labels_from_frame(frame, dates, symbols)
    # 早于第一版时用第一版；跨年后用新一版（不复用未来的分类）
    assert labels.loc[dates[0], "600001.SH"] == "K70房地产业"
    assert labels.loc[dates[1], "600001.SH"] == "K70房地产业"
    assert labels.loc[dates[2], "600001.SH"] == "C36汽车制造业"
    assert labels.loc[dates[2], "600002.SH"] == "J66货币金融服务"
    assert list(labels.columns) == list(symbols)


# ------------------------------------------------------------- 中性化


def test_residualize_removes_industry_and_size():
    rng = np.random.default_rng(0)
    n, groups = 600, 6
    labels = [f"ind{index % groups}" for index in range(n)]
    ln_size = rng.normal(10.0, 1.0, n)
    # 因子 = 2×ln市值 + 行业偏移 + 噪声：不中性化时横截面几乎就是市值的复制品
    industry_shift = np.array([float(labels.index(label) % groups) for label in labels]) * 0.5
    values = 2.0 * ln_size + industry_shift + rng.normal(0.0, 0.3, n)
    symbols = [f"S{index:04d}" for index in range(n)]

    frame = pd.DataFrame([values], index=[_DAY], columns=symbols)
    label_frame = pd.DataFrame([labels], index=[_DAY], columns=symbols)
    size_frame = pd.DataFrame([ln_size], index=[_DAY], columns=symbols)
    codes = codes_from_labels(label_frame)
    residual = residualize(frame, codes=codes, size=size_frame, min_count=20)

    row = residual.loc[_DAY].to_numpy(dtype=float)
    assert np.isfinite(row).all()
    # 与 ln(市值) 的相关系数应当近似为 0（FWL 的性质，不是调参调出来的）
    assert abs(float(np.corrcoef(row, ln_size)[0, 1])) < 1e-6
    # 每个行业的残差均值应当为 0
    labels_array = np.array(labels)
    for label in set(labels):
        mask = labels_array == label
        assert abs(float(row[mask].mean())) < 1e-9


def test_residualize_many_matches_single_and_keeps_nan():
    rng = np.random.default_rng(7)
    n = 240
    labels = [f"ind{index % 4}" for index in range(n)]
    ln_size = rng.normal(9.0, 0.8, n)
    symbols = [f"S{index:04d}" for index in range(n)]
    base = rng.normal(0.0, 1.0, n) + ln_size
    broken = base.copy()
    broken[:30] = np.nan  # 缺数据的格子不能被填成 0（那等于给最低分）
    frames = {
        "a": pd.DataFrame([base], index=[_DAY], columns=symbols),
        "b": pd.DataFrame([broken], index=[_DAY], columns=symbols),
    }
    codes = codes_from_labels(pd.DataFrame([labels], index=[_DAY], columns=symbols))
    size = pd.DataFrame([ln_size], index=[_DAY], columns=symbols)

    together = residualize_many(frames, codes=codes, size=size, min_count=20)
    alone = residualize(frames["a"], codes=codes, size=size, min_count=20)
    assert np.allclose(together["a"].to_numpy(), alone.to_numpy(), equal_nan=True)
    # 30 个缺失格必须仍是 NaN，其余格子有值
    assert together["b"].iloc[0, :30].isna().all()
    assert together["b"].iloc[0, 30:].notna().all()


def test_neutralize_requires_exposures():
    try:
        MultiFactorStrategy(neutralize="industry")
    except ValueError as error:
        assert "暴露" in str(error)
    else:  # pragma: no cover
        raise AssertionError("neuralize 没给暴露矩阵应当报错")


def test_strategy_neutralize_end_to_end():
    symbols = [f"6000{index:02d}.SH" for index in range(1, 13)]
    bars = _bars(symbols, periods=300, seed=11)
    close = pd.DataFrame({symbol: frame["close"] for symbol, frame in bars.items()})
    labels = pd.DataFrame(
        [["I1" if index % 2 else "I2" for index in range(len(symbols))]],
        index=close.index,
        columns=close.columns,
    )
    size = np.log(close * 3.0e9)
    strategy = MultiFactorStrategy(
        factors="momentum,low_vol",
        top_k=3,
        rebalance="W-FRI",
        min_history=60,
        min_universe=5,
        neutralize="industry_size",
        industry=labels,
        size=size,
    )
    weights = strategy.generate(bars)
    assert float(weights.abs().to_numpy().max()) > 0
    assert strategy.neutralize == "industry_size"
    assert int((weights > 0).sum(axis=1).max()) <= 3


# ------------------------------------------------------------- 前视


def test_ic_weights_ignore_future_bars():
    symbols = [f"6000{index:02d}.SH" for index in range(1, 13)]
    bars = _bars(symbols, periods=300, seed=5)
    kwargs = dict(
        factors="momentum,reversal,low_vol",
        weighting="ic",
        ic_lookback=40,
        ic_min_periods=8,
        weight_refresh=1,
        top_k=3,
        rebalance="W-FRI",
        min_history=60,
        min_universe=5,
    )
    full = MultiFactorStrategy(**kwargs)
    full.generate(bars)
    full_log = full.last_diagnostics["factor_weights_by_date"]
    assert full_log

    cutoff = sorted(full_log)[len(full_log) // 2]
    truncated = {
        symbol: frame.loc[frame.index <= pd.Timestamp(cutoff)] for symbol, frame in bars.items()
    }
    partial = MultiFactorStrategy(**kwargs)
    partial.generate(truncated)
    partial_log = partial.last_diagnostics["factor_weights_by_date"]
    assert cutoff in partial_log, "截断窗口里也应当算出同一个调仓日的权重"
    for name, weight in full_log[cutoff].items():
        assert abs(weight - partial_log[cutoff][name]) < 1e-12


# ------------------------------------------------------------- 分组


def test_bucket_members_follow_quantile_numbering():
    ranked = [f"S{index:03d}" for index in range(100)]
    top = MultiFactorStrategy(select_bucket=5, select_quantiles=5, top_k=5, min_universe=5)
    bottom = MultiFactorStrategy(select_bucket=1, select_quantiles=5, top_k=5, min_universe=5)
    middle = MultiFactorStrategy(select_bucket=3, select_quantiles=5, top_k=5, min_universe=5)
    assert top._bucket_members(ranked) == ranked[0:20]
    # 最低分组里「越极端越靠前」= 分数最低的先选
    assert bottom._bucket_members(ranked) == list(reversed(ranked[80:100]))
    assert middle._bucket_members(ranked) == ranked[40:60]
    # 样本不够切组时不硬切
    assert top._bucket_members(ranked[:9]) == []
    assert top._bucket_members(ranked[:10]) == ranked[0:2]


def test_bucket_validation_and_backtest_picks():
    try:
        MultiFactorStrategy(select_bucket=6, select_quantiles=5)
    except ValueError as error:
        assert "select_bucket" in str(error)
    else:  # pragma: no cover
        raise AssertionError("越界的分组应当报错")

    symbols = [f"6000{index:02d}.SH" for index in range(1, 31)]
    bars = _bars(symbols, periods=300, seed=13)
    strategy = MultiFactorStrategy(
        factors="momentum,low_vol",
        top_k=4,
        rebalance="W-FRI",
        min_history=60,
        min_universe=10,
        select_bucket=5,
        select_quantiles=5,
    )
    weights = strategy.generate(bars)
    picks = strategy.last_diagnostics["picks"]
    assert picks, "应当有建仓记录"
    sizes = [len(value) for value in picks.values()]
    assert max(sizes) <= 4
