"""关卡 2/3、holdout 隔离的离线测试（不碰数据湖、不联网、不消费 holdout）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import data_pack, gate_holdout as gh, gate_robustness as gr, reproduce_lake as rl  # noqa: E402
from tests._util import tmpdir  # noqa: E402


def _fake_daily(n: int = 1000, *, strat_mu: float = 0.0012, bench_mu: float = 0.0002,
                seed: int = 11) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    strat = pd.Series(rng.normal(strat_mu, 0.008, n), index=idx)
    bench = pd.Series(rng.normal(bench_mu, 0.008, n), index=idx)
    daily = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"),
                          "equity": 1e6 * (1 + strat).cumprod()})
    return daily, 1e6 * (1 + bench).cumprod()


# ---------------------------------------------------------------- 关卡 2
def test_gate2_criteria_sha_stable():
    first = gr.script_sha()
    assert len(first) == 16
    assert first == gr.script_sha(), "脚本 sha 必须可复现（同一文件同一结果）"
    assert set(gr.CRITERIA) >= {"dsr", "pbo", "bootstrap_lower", "is_oos_ratio", "perm_p"}


def test_gate2_markdown_renders():
    with tmpdir("gate2-md") as tmp:
        path = Path(tmp) / "gate2.md"
        result = {"strategy": "growth_cap25", "generated": "2026-10-23T00:00:00Z",
                  "git_commit": "abc1234", "script_sha": "0123456789abcdef", "elapsed_s": 1.0,
                  "passed": False, "criteria": {"dsr": 0.95},
                  "checks": [{"name": "dsr", "value": 0.59, "pass": False, "note": "N=4"}]}
        gr.write_markdown(result, path)
        text = path.read_text(encoding="utf-8")
        assert "| dsr | 0.59 |" in text
        assert "未通过" in text


# ---------------------------------------------------------------- 关卡 3
def _floor(cagr: float, sharpe: float, *, regime: str = "上涨·低波动",
           basis: str = "same_regime", fallback: bool = False) -> dict:
    """手搓一个底线包，用来测 evaluate 的判决逻辑（不读数据湖）。"""
    return {"regime": regime, "windows_total": 40, "windows_same_regime": 12, "n_used": 12,
            "basis": basis, "fallback": fallback, "window_days": 500,
            "cagr_floor": cagr, "sharpe_floor": sharpe,
            "cagr_median": cagr + 0.05, "sharpe_median": sharpe + 0.1,
            "cagr_best": cagr + 0.30, "cagr_worst": cagr - 0.20}


def _cagr_of(daily: pd.DataFrame) -> float:
    return float((daily["equity"].iloc[-1] / daily["equity"].iloc[0]) ** (244 / len(daily)) - 1)


def test_gate3_criteria_are_v3_regime_floor():
    """v3：底线来自样本内同类行情的分位，不再是全样本保留率。"""
    assert gh.CRITERIA_VERSION == "v3"
    assert set(gh.CRITERIA) == {"cagr_floor_quantile", "sharpe_floor_quantile",
                               "excess_cagr_min", "min_regime_windows", "window_step"}
    assert len(gh.criteria_sha()) == 16
    assert "v3" in gh.CRITERIA_NOTE


def test_gate3_evaluate_pass_above_floor():
    daily, bench = _fake_daily()
    outcome = gh.evaluate(daily.set_index(pd.to_datetime(daily["date"])), bench,
                          cagr=_cagr_of(daily), sharpe=1.2, floor=_floor(0.05, 0.20))
    checks = {c["name"]: c["pass"] for c in outcome["checks"]}
    assert all(checks.values()), outcome["checks"]
    ref = outcome["reference"]
    assert ref["regime"] == "上涨·低波动"
    assert ref["floor_basis"] == "same_regime"
    assert ref["cagr_floor"] == 0.05 and ref["sharpe_floor"] == 0.2


def test_gate3_evaluate_fails_below_cagr_floor():
    daily, bench = _fake_daily(strat_mu=0.0001)
    outcome = gh.evaluate(daily.set_index(pd.to_datetime(daily["date"])), bench,
                          cagr=_cagr_of(daily), sharpe=1.2, floor=_floor(0.60, 0.20))
    checks = {c["name"]: c["pass"] for c in outcome["checks"]}
    assert checks["cagr_floor"] is False
    assert checks["sharpe_floor"] is True


def test_gate3_evaluate_fails_below_sharpe_floor():
    daily, bench = _fake_daily()
    outcome = gh.evaluate(daily.set_index(pd.to_datetime(daily["date"])), bench,
                          cagr=_cagr_of(daily), sharpe=0.05, floor=_floor(0.05, 0.50))
    checks = {c["name"]: c["pass"] for c in outcome["checks"]}
    assert checks["sharpe_floor"] is False
    assert checks["cagr_floor"] is True


def test_gate3_evaluate_fails_when_far_behind_benchmark():
    daily, bench = _fake_daily(strat_mu=0.0002, bench_mu=0.0030)
    outcome = gh.evaluate(daily.set_index(pd.to_datetime(daily["date"])), bench,
                          cagr=_cagr_of(daily), sharpe=1.2, floor=_floor(-0.50, -1.0))
    checks = {c["name"]: c["pass"] for c in outcome["checks"]}
    assert checks["excess_cagr_min"] is False, "兜底：相对同池等权落后超过 5pp 必须否决"
    assert outcome["excess_cagr"] < -0.05


def test_gate3_reference_block_has_market_context():
    daily, bench = _fake_daily()
    ref = gh.evaluate(daily.set_index(pd.to_datetime(daily["date"])), bench,
                      cagr=_cagr_of(daily), sharpe=1.2, floor=_floor(0.05, 0.20))["reference"]
    for key in ("bench_cagr", "bench_vol", "bench_max_drawdown", "beta", "corr",
                "monthly_win_rate", "regime", "excess_t", "cagr_floor", "sharpe_floor",
                "floor_basis", "windows_same_regime"):
        assert key in ref, key


def test_gate3_regime_reference_builds_floor_from_same_regime():
    """样本内滚动窗口要能按行情分组，且底线落在分布之内（不高于最好、不低于最差）。"""
    daily, bench = _fake_daily(n=1200, strat_mu=0.0010, bench_mu=0.0008)
    daily = daily.set_index(pd.to_datetime(daily["date"]))
    ref = gh.regime_reference(daily, bench, window_days=250)
    assert ref["windows_total"] > 10, ref["windows_total"]
    labels = {row["label"] for row in ref["rows"]}
    assert all(label.count("·") == 1 for label in labels), labels
    label = ref["rows"][0]["label"]
    floor = gh.regime_floor(ref, label)
    assert floor["basis"] in {"same_regime", "all_windows"}
    assert floor["cagr_worst"] <= floor["cagr_floor"] <= floor["cagr_best"]
    assert floor["sharpe_floor"] <= floor["sharpe_median"]
    assert floor["n_used"] >= 1
    assert set(floor) >= {"regime", "cagr_floor", "sharpe_floor", "windows_same_regime"}


def test_gate3_marker_blocks_second_run():
    with tmpdir("gate3-marker") as tmp:
        tmp = Path(tmp)
        old_marker, old_gate1 = gh.MARKER, gh.gate1_offline
        try:
            gh.MARKER = tmp / "holdout-consumed.json"
            gh.MARKER.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
            gh.gate1_offline = lambda *a, **k: {"pass": True, "returncode": 0, "seconds": 0.0, "tail": []}
            rc = gh.main(["--out", str(tmp)])
            assert rc == 2, "已消费过的 holdout 必须拒绝重复验收"
            assert gh.MARKER.exists()
        finally:
            gh.MARKER, gh.gate1_offline = old_marker, old_gate1


def test_gate3_dry_run_does_not_consume():
    with tmpdir("gate3-dry") as tmp:
        tmp = Path(tmp)
        old_marker, old_gate1 = gh.MARKER, gh.gate1_offline
        try:
            gh.MARKER = tmp / "holdout-consumed.json"
            gh.gate1_offline = lambda *a, **k: {"pass": True, "returncode": 0, "seconds": 0.0, "tail": []}
            rc = gh.main(["--dry-run", "--out", str(tmp)])
            assert rc == 0
            assert not gh.MARKER.exists(), "dry-run 不得消费 holdout"
            verdict = json.loads((tmp / "holdout-verdict.json").read_text(encoding="utf-8"))
            assert verdict["dry_run"] is True
        finally:
            gh.MARKER, gh.gate1_offline = old_marker, old_gate1


# ---------------------------------------------------------------- holdout 隔离
def test_owner_mode_is_off_by_default():
    import os
    old = os.environ.pop("QUANT_OWNER", None)
    try:
        assert rl.is_owner() is False
        os.environ["QUANT_OWNER"] = "1"
        assert rl.is_owner() is True
    finally:
        os.environ.pop("QUANT_OWNER", None)
        if old is not None:
            os.environ["QUANT_OWNER"] = old


def test_reproduce_stages_mark_holdout():
    assert rl.STAGE_MAP["split"].holdout is True
    assert rl.STAGE_MAP["parquet"].owner_cmds, "parquet 阶段的 holdout 命令只能在所有者模式追加"
    for name in ("process", "fund", "events", "alt"):
        assert rl.STAGE_MAP[name].holdout is False


def test_data_pack_keeps_holdout_owner_only():
    assert "holdout" in data_pack.OWNER_ONLY
    for preset in ("core", "all"):
        assert "holdout" not in data_pack.PRESETS[preset], f"{preset} 预设不得默认带 holdout"


def test_data_pack_rejects_holdout_without_owner():
    import os
    old = os.environ.pop("QUANT_OWNER", None)
    try:
        try:
            data_pack.main(["--layers", "holdout", "--plan"])
        except SystemExit as exc:  # 预期：拒绝并退出
            assert "所有者" in str(exc.code) or "holdout" in str(exc.code)
        else:  # pragma: no cover - 只有出 bug 才会走到
            raise AssertionError("非所有者打包 holdout 必须被拒绝")
    finally:
        if old is not None:
            os.environ["QUANT_OWNER"] = old
