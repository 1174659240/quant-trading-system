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
def test_gate3_evaluate_pass_when_excess_and_decay_ok():
    daily, bench = _fake_daily()
    cagr = float((daily["equity"].iloc[-1] / daily["equity"].iloc[0]) ** (244 / len(daily)) - 1)
    outcome = gh.evaluate(daily.set_index(pd.to_datetime(daily["date"])), bench, cagr, is_cagr_value=0.05)
    assert outcome["excess_cagr"] > 0
    assert outcome["decay_ratio"] > 1.5
    assert all(c["pass"] for c in outcome["checks"]), outcome


def test_gate3_evaluate_fails_decay():
    daily, bench = _fake_daily(strat_mu=0.0001)
    cagr = float((daily["equity"].iloc[-1] / daily["equity"].iloc[0]) ** (244 / len(daily)) - 1)
    outcome = gh.evaluate(daily.set_index(pd.to_datetime(daily["date"])), bench, cagr, is_cagr_value=0.60)
    checks = {c["name"]: c["pass"] for c in outcome["checks"]}
    assert checks["decay_ratio_min"] is False


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
