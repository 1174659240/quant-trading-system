"""危机识别（quant.regime）与前沿因子（quant.factors.frontier）的离线测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import regime
from quant.data import make_gbm_frame
from quant.factors import frontier
from quant.factors.library import compute
from quant.factors.panel import build_panel


def _panel(periods: int = 400, symbols=("600519.SH", "000001.SZ", "300750.SZ", "601318.SH", "000002.SZ")):
    bars = {s: make_gbm_frame(s, periods=periods, seed=200 + i) for i, s in enumerate(symbols)}
    return build_panel(bars)


def test_market_index_and_score_are_bounded():
    panel = _panel()
    frame = regime.market_index(panel)
    flags = regime.score_components(frame)
    score = regime.crisis_score(flags)
    assert list(frame.columns)[:3] == ["market_ret", "index", "ma200"]
    assert flags.dtypes.eq(bool).all()
    clean = score.dropna()
    assert len(clean) > 0 and clean.between(0.0, 1.0).all()
    assert abs(sum(regime.DEFAULT_WEIGHTS.values()) - 1.0) < 1e-12, "分量权重必须归一"
    print("  ✓ 危机分数落在 [0,1]，六分量权重合计 1")


def test_state_has_no_lookahead():
    panel = _panel()
    score = regime.crisis_score(regime.score_components(regime.market_index(panel)))
    state = regime.crisis_state(score, enter=0.6, exit=0.3, confirm=3)
    cut = 250
    truncated = regime.crisis_state(score.iloc[:cut], enter=0.6, exit=0.3, confirm=3)
    assert state.iloc[:cut].equals(truncated), "截断之后的数据不能影响截断之前的状态"
    print("  ✓ 状态机无前视：截断未来数据不改变历史状态")


def test_gate_blocks_and_recovery_exits():
    idx = pd.bdate_range("2020-01-01", periods=12, name="date")
    score = pd.Series([0.9] * 12, index=idx)
    gate = pd.Series([False] * 12, index=idx)
    assert not regime.crisis_state(score, enter=0.6, exit=0.3, confirm=2, gate=gate).any()

    gate = pd.Series([False] * 3 + [True] * 9, index=idx)
    state = regime.crisis_state(score, enter=0.6, exit=0.3, confirm=2, gate=gate)
    assert state.iloc[:3].sum() == 0 and state.iloc[3:].all(), "gate 打开后才允许清仓"

    score_slow = pd.Series([0.9] * 8 + [0.1] * 4, index=idx)
    recovery = pd.Series([False] * 8 + [True] * 4, index=idx)
    state = regime.crisis_state(score_slow, enter=0.6, exit=0.3, confirm=2, gate=gate, recovery=recovery)
    assert state.iloc[8] and not state.iloc[9], "回暖需连续 confirm 天才解除"
    print("  ✓ gate 阻断入场；回暖连续确认后解除")


def test_overlay_zeroes_crisis_rows_only():
    idx = pd.bdate_range("2020-01-01", periods=8, name="date")
    weights = pd.DataFrame(0.2, index=idx, columns=["A", "B", "C"])
    crisis = pd.Series([False, False, True, True, False, False, False, False], index=idx)
    overlay = regime.crisis_overlay(crisis)
    out = overlay(weights, {}, {"config": None})
    assert out.loc[idx[2]].abs().sum() == 0.0 and out.loc[idx[3]].abs().sum() == 0.0
    assert np.allclose(out.loc[idx[4]], 0.2), "解除后恢复原目标权重"

    delayed = regime.crisis_overlay(crisis, delay=1)(weights, {}, {"config": None})
    assert delayed.loc[idx[2]].abs().sum() > 0 and delayed.loc[idx[3]].abs().sum() == 0.0
    print("  ✓ overlay 只在危机行清零；delay=1 时推迟一天")


def test_episodes_merge_consecutive_days():
    idx = pd.bdate_range("2020-01-01", periods=10, name="date")
    state = pd.Series([False, True, True, False, True, False, False, False, False, False], index=idx)
    score = pd.Series(0.9, index=idx)
    frame = pd.DataFrame({"dd250": -0.2}, index=idx)
    eps = regime.episodes(state, score, frame)
    assert len(eps) == 2 and eps[0].days == 2 and eps[0].end == idx[3]
    print("  ✓ 连续危机日合并为事件，间隔日切开")


def test_frontier_factors_shape_and_sign():
    panel = _panel()
    frontier.register()
    for name in ("overnight_ret_20", "intraday_ret_20", "high_52w", "residual_momentum", "downside_beta", "ivol_250", "fip_momentum"):
        frame = compute(name, panel)
        assert frame.shape == (len(panel.dates), len(panel.symbols)), name
        assert frame.notna().any().any(), f"{name} 全为空"
    peak = compute("high_52w", panel)
    clean = peak.stack().dropna()
    assert clean.gt(0).all() and clean.max() < 1.5, "距 52 周高点比例应在 (0, 1.5)"
    print("  ✓ 7 个前沿因子可计算、形状正确、high_52w 值域合理")
