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
    assert set(gr.HARD) == {"perm_p_max", "cost_cagr_min", "capacity_participation_max"}
    assert set(gr.SOFT) >= {"dsr_min", "pbo_max", "bootstrap_lower_min", "is_oos_ratio_min",
                            "sensitivity_sharpe_range_max", "sensitivity_cagr_range_max",
                            "wfa_positive_share_min", "state_share_max"}
    assert len(gr.criteria_sha()) == 16


def test_gate2_archetypes_cost_tier():
    """成本档按策略类型分，不一刀切：小盘高换手 30bp、大盘低换手 50bp。"""
    assert gr.ARCHETYPES["small_cap_fast"]["cost_bp"] == 30.0
    assert gr.ARCHETYPES["large_cap_slow"]["cost_bp"] == 50.0
    assert gr.criteria_payload()["archetypes"]["small_cap_fast"]["cost_bp"] == 30.0


def test_gate2_markdown_renders():
    with tmpdir("gate2-md") as tmp:
        path = Path(tmp) / "gate2.md"
        result = _gate2_payload(hard_ok=False, multi_alarm=True)
        result.update({"strategy": "growth_cap25", "generated": "2026-10-23T00:00:00Z",
                       "git_commit": "abc1234", "script_sha": "0123456789abcdef", "elapsed_s": 1.0,
                       "archetype": "small_cap_fast", "archetype_label": "小盘高换手型",
                       "cost_bp": 30.0, "target_aum": 5e7,
                       "prereg": {"file": "x", "sha": "y", "declared_at": "z"},
                       "criteria": gr.criteria_payload(),
                       "hard": {"passed": False, "failed": ["perm_p"],
                                "checks": [{"name": "perm_p", "value": 0.12, "threshold": 0.05,
                                            "pass": False, "note": "真实 IC 0.01"}]},
                       "soft": {"multi_alarm": True, "failed": ["pbo"], "failed_count": 1,
                                "severe": [], "verdict": "多重稳健性告警",
                                "checks": [{"name": "pbo", "value": 0.65, "threshold": 0.5,
                                            "pass": False, "severe": False, "note": ""}]}})
        gr.write_markdown(result, path)
        text = path.read_text(encoding="utf-8")
        assert "硬否决层" in text and "强制参考层" not in text
        assert "强参考层" in text
        assert "| perm_p | 0.12 | 0.05 |" in text
        assert "硬否决未通过" in text


def _gate2_payload(*, hard_ok: bool = True, multi_alarm: bool = False,
                   version: str = gr.CRITERIA_VERSION, light: bool = False) -> dict:
    return {"version": version, "criteria_sha": gr.criteria_sha(), "light": light,
            "hard": {"passed": hard_ok, "failed": [] if hard_ok else ["perm_p"]},
            "soft": {"multi_alarm": multi_alarm,
                     "failed": ["pbo"] if multi_alarm else []},
            "verdict": ("CLEAR" if hard_ok and not multi_alarm
                        else "MULTI-ALARM" if hard_ok else "HARD-FAIL")}


def _write_gate2(tmp: Path, payload: dict) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "gate2-growth_cap25-20260101-0000.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_gate2_precheck_rejects_missing_report():
    with tmpdir("gate2-missing") as tmp:
        out = gh.gate2_precheck("growth_cap25", gates=Path(tmp))
        assert out["ok"] is False and "未找到关卡 2" in out["reason"]


def test_gate2_precheck_rejects_old_advisory_format():
    with tmpdir("gate2-old") as tmp:
        _write_gate2(Path(tmp), _gate2_payload(version="gate2-v1"))
        out = gh.gate2_precheck("growth_cap25", gates=Path(tmp))
        assert out["ok"] is False and "旧格式" in out["reason"]


def test_gate2_precheck_rejects_stale_criteria():
    with tmpdir("gate2-stale") as tmp:
        payload = _gate2_payload()
        payload["criteria_sha"] = "deadbeefdeadbeef"
        _write_gate2(Path(tmp), payload)
        out = gh.gate2_precheck("growth_cap25", gates=Path(tmp))
        assert out["ok"] is False and "判据已变" in out["reason"]


def test_gate2_precheck_rejects_hard_fail():
    with tmpdir("gate2-hard") as tmp:
        _write_gate2(Path(tmp), _gate2_payload(hard_ok=False))
        old = gr.verify_prereg
        gr.verify_prereg = lambda *a, **k: {"ok": True, "reason": "", "archetype": "small_cap_fast",
                                            "cost_bp": 30.0, "target_aum": 5e7}
        try:
            out = gh.gate2_precheck("growth_cap25", gates=Path(tmp))
            assert out["ok"] is False and "硬否决未通过" in out["reason"]
        finally:
            gr.verify_prereg = old


def test_gate2_precheck_accepts_clear():
    with tmpdir("gate2-clear") as tmp:
        _write_gate2(Path(tmp), _gate2_payload())
        old = gr.verify_prereg
        gr.verify_prereg = lambda *a, **k: {"ok": True, "reason": "", "archetype": "small_cap_fast",
                                            "cost_bp": 30.0, "target_aum": 5e7}
        try:
            out = gh.gate2_precheck("growth_cap25", gates=Path(tmp))
            assert out["ok"] is True
            assert out["multi_alarm"] is False
            assert out["verdict"] == "CLEAR"
        finally:
            gr.verify_prereg = old


def test_gate2_precheck_reports_multi_alarm():
    with tmpdir("gate2-alarm") as tmp:
        _write_gate2(Path(tmp), _gate2_payload(multi_alarm=True))
        old = gr.verify_prereg
        gr.verify_prereg = lambda *a, **k: {"ok": True, "reason": "", "archetype": "small_cap_fast",
                                            "cost_bp": 30.0, "target_aum": 5e7}
        try:
            out = gh.gate2_precheck("growth_cap25", gates=Path(tmp))
            assert out["ok"] is True and out["multi_alarm"] is True
        finally:
            gr.verify_prereg = old


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


def _stub_gate2(gh_module):
    """把关卡 2 前置校验打桩成「通过」，避免测试依赖真实报告。"""
    return lambda *a, **k: {"ok": True, "file": "stub", "verdict": "CLEAR",
                            "multi_alarm": False, "soft_failed": []}


def test_gate3_verdict_requires_gate2_and_gate1():
    assert gh.gate3_verdict(True, True)["verdict"] == "PASS"
    assert gh.gate3_verdict(True, False)["verdict"] == "FAIL"
    assert gh.gate3_verdict(False, True)["verdict"] == "FAIL"


def test_gate3_verdict_holds_on_multi_alarm():
    out = gh.gate3_verdict(True, True, multi_alarm=True)
    assert out["verdict"] == "HOLD" and out["passed"] is False
    assert out["blocked_by"] == "gate2-multi-alarm"


def test_gate3_verdict_owner_confirm_releases_hold():
    out = gh.gate3_verdict(True, True, multi_alarm=True, owner_confirm="已看过强参考层")
    assert out["verdict"] == "PASS" and out["passed"] is True
    assert out["owner_confirmation"] == "已看过强参考层"


def test_gate3_refuses_when_gate2_hard_veto():
    """关卡 2 硬否决不过，关卡 3 必须直接拒绝，连关卡 1 都不跑。"""
    with tmpdir("gate3-veto") as tmp:
        tmp = Path(tmp)
        old_pre, old_gate1 = gh.gate2_precheck, gh.gate1_offline
        calls = {"gate1": 0}

        def _gate1(*a, **k):
            calls["gate1"] += 1
            return {"pass": True, "returncode": 0, "seconds": 0.0, "tail": []}

        try:
            gh.gate2_precheck = lambda *a, **k: {"ok": False, "reason": "关卡 2 硬否决未通过：perm_p"}
            gh.gate1_offline = _gate1
            rc = gh.main(["--out", str(tmp)])
            assert rc == 3, "硬否决未过必须非 0 退出且不进关卡 3"
            assert calls["gate1"] == 0, "拒绝时不该白跑离线测试"
            verdict = json.loads((tmp / "holdout-verdict.json").read_text(encoding="utf-8"))
            assert verdict["verdict"] == "REJECTED"
        finally:
            gh.gate2_precheck, gh.gate1_offline = old_pre, old_gate1


def test_gate3_marker_blocks_second_run():
    with tmpdir("gate3-marker") as tmp:
        tmp = Path(tmp)
        old_marker, old_gate1, old_pre = gh.MARKER, gh.gate1_offline, gh.gate2_precheck
        try:
            gh.MARKER = tmp / "holdout-consumed.json"
            gh.MARKER.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
            gh.gate1_offline = lambda *a, **k: {"pass": True, "returncode": 0, "seconds": 0.0, "tail": []}
            gh.gate2_precheck = _stub_gate2(gh)
            rc = gh.main(["--out", str(tmp)])
            assert rc == 2, "已消费过的 holdout 必须拒绝重复验收"
            assert gh.MARKER.exists()
        finally:
            gh.MARKER, gh.gate1_offline, gh.gate2_precheck = old_marker, old_gate1, old_pre


def test_gate3_dry_run_does_not_consume():
    with tmpdir("gate3-dry") as tmp:
        tmp = Path(tmp)
        old_marker, old_gate1, old_pre = gh.MARKER, gh.gate1_offline, gh.gate2_precheck
        try:
            gh.MARKER = tmp / "holdout-consumed.json"
            gh.gate1_offline = lambda *a, **k: {"pass": True, "returncode": 0, "seconds": 0.0, "tail": []}
            gh.gate2_precheck = _stub_gate2(gh)
            rc = gh.main(["--dry-run", "--out", str(tmp)])
            assert rc == 0
            assert not gh.MARKER.exists(), "dry-run 不得消费 holdout"
            verdict = json.loads((tmp / "holdout-verdict.json").read_text(encoding="utf-8"))
            assert verdict["dry_run"] is True
        finally:
            gh.MARKER, gh.gate1_offline, gh.gate2_precheck = old_marker, old_gate1, old_pre


def test_gate2_state_shares_are_contributions_not_compounded():
    """行情状态贡献占比：互斥分桶之和为 1；集中度必须落在 [0,1]。"""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2022-01-03", periods=300)
    daily = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
    buckets = {"mom_up": daily[daily > 0], "mom_down": daily[daily <= 0]}
    shares = gr.state_shares(buckets, daily)
    assert abs(shares["mom_up"] + shares["mom_down"] - 1.0) < 1e-9
    conc = gr.state_concentration(buckets, daily)
    assert 0.0 <= conc <= 1.0, conc
    assert gr.state_shares({"empty": pd.Series(dtype=float)}, daily) == {}
    # 单边行情：正贡献全在上涨日 → 集中度 1
    only_up = {"mom_up": daily[daily > 0], "mom_down": daily[daily > 0] * 0}
    assert gr.state_concentration(only_up, daily) == 1.0


def test_gate2_state_concentration_uses_excess_and_skips_empty_pairs():
    """E 层整段没触发（一侧 0 天）时该维度不参与计算，否则会凭空算出 1.0。"""
    up = pd.Series([0.1] * 100)
    down = pd.Series([0.3] * 100)
    buckets = {"mom_up": up, "mom_down": down,
               "e_trigger": pd.Series(dtype=float), "e_calm": pd.Series([0.2] * 1000)}
    assert abs(gr.state_concentration(buckets, pd.Series([0.0] * 100)) - 0.75) < 1e-12


def test_gate2_capacity_ignores_zero_weights():
    """权重文件是稠密矩阵（未持仓写 0）；不剔零会把 p95 参与率错算成 0。"""
    import types

    with tmpdir("gate2-cap") as tmp:
        tmp = Path(tmp)
        days = pd.bdate_range("2022-01-03", periods=25)
        codes = [f"{i:06d}" for i in range(1, 201)]
        panel = types.SimpleNamespace(amount=pd.DataFrame(1e8, index=days, columns=codes))
        weights = pd.DataFrame(0.0, index=days[::5], columns=codes)
        for day in weights.index:
            weights.loc[day, codes[:10]] = 0.02
        path = tmp / "weights.csv"
        weights.to_csv(path)
        out = gr.capacity({"panel": panel}, path, target_aum=5e7)
        p95 = out["target"]["p95"]
        assert p95 is not None and 0.0 < p95 < 0.05, p95


def test_prereg_locks_archetype_and_blocks_silent_change():
    """档位在预注册时锁定；想改必须 --relock 且留理由，旧档进 history。"""
    with tmpdir("gate2-prereg") as tmp:
        tmp = Path(tmp)
        info = gr.declare("growth_cap25", "small_cap_fast", 5e7,
                          prereg_dir=tmp / "prereg", lock_path=tmp / "lock.json")
        assert info["ok"] and info["sha"]
        ok = gr.verify_prereg("growth_cap25", prereg_dir=tmp / "prereg", lock_path=tmp / "lock.json")
        assert ok["ok"] and ok["cost_bp"] == 30.0 and ok["target_aum"] == 5e7
        try:
            gr.declare("growth_cap25", "large_cap_slow", 5e7,
                       prereg_dir=tmp / "prereg", lock_path=tmp / "lock.json")
        except SystemExit as exc:
            assert "relock" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("已锁定的档位不得静默改档")
        relocked = gr.declare("growth_cap25", "large_cap_slow", 5e7, relock=True,
                              reason="规模变大", prereg_dir=tmp / "prereg",
                              lock_path=tmp / "lock.json")
        assert relocked["relocked"] is True
        lock = json.loads((tmp / "lock.json").read_text(encoding="utf-8"))
        assert lock["strategies"]["growth_cap25"]["history"][0]["reason"] == "规模变大"


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
