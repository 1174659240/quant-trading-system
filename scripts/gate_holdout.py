"""关卡 3：一次性样本外（holdout）验收——机器执行，不看人脸色。

设计意图：holdout 只能被这个脚本消费一次，判据在 ``CRITERIA`` 里写死并带 sha256。
脚本会：
1. 跑关卡 1（离线测试，``--skip-lake``），记录退出码；
2. 读最近的关卡 2 报告（**仅作参考**，不阻断）；
3. 用冻结姿态跑一次 holdout（``scripts/growth_holdout``），写进 ``results/gates/holdout-<stamp>/``；
4. 按预注册判据算三项：年化超额 > 0、样本外/样本内 ≥ 0.5、日频超额 t ≥ 2；
5. 三项全过才 ``PASS``，否则 ``FAIL``，并把结论写进 ``results/gates/holdout-verdict.json``。

**没有 --tune、没有 --relax**：想改判据只能改这份文件（sha256 会变，报告里会显示），
且必须在 ``docs/放行流程.md`` 里先写预注册。

用法::

    python -X utf8 -m scripts.gate_holdout            # 正式验收（会消费 holdout）
    python -X utf8 -m scripts.gate_holdout --dry-run  # 只做关卡 1 + 读现状，不动 holdout
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

GATES = Path("results/gates")
MARKER = GATES / "holdout-consumed.json"
IS_METRICS = Path("results/risk-system/cap25/metrics-main.csv")

#: 预注册判据（改这里 = 改协议，必须同步 docs/放行流程.md）
CRITERIA = {
    "excess_cagr_gt": 0.0,
    "decay_ratio_min": 0.50,
    "excess_t_min": 2.0,
}

TRADING_DAYS = 244


def criteria_sha() -> str:
    payload = json.dumps(CRITERIA, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def gate1_offline(timeout: int = 900) -> dict:
    """关卡 1：跑离线测试（不碰数据湖）。"""
    started = time.time()
    proc = subprocess.run([sys.executable, "-X", "utf8", "tests/run_all.py", "--skip-lake"],
                          capture_output=True, text=True, timeout=timeout)
    tail = (proc.stdout or "").strip().splitlines()[-3:]
    return {"pass": proc.returncode == 0, "returncode": proc.returncode,
            "seconds": round(time.time() - started, 1), "tail": tail}


def gate2_record(strategy: str = "growth_cap25") -> dict:
    """读最近的关卡 2 报告（只作参考）。"""
    files = sorted(GATES.glob(f"gate2-{strategy}-*.json"))
    if not files:
        return {"found": False}
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    return {"found": True, "file": str(files[-1]), "passed": data.get("passed"),
            "elapsed_s": data.get("elapsed_s")}


def is_cagr() -> float:
    frame = pd.read_csv(IS_METRICS)
    return float(frame["cagr"].iloc[0])


def run_holdout(out_dir: Path, *, force: bool) -> dict:
    from scripts import growth_holdout as gh

    gh.OUT = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = gh.run(force=force)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    return {"returncode": rc, "summary": summary}


def evaluate(daily: pd.DataFrame, bench_series: pd.Series, cagr: float, is_cagr_value: float) -> dict:
    """纯函数：给定 holdout 的日频净值与基准，按预注册判据出结论（可单测）。"""
    marks = pd.to_datetime(daily.index)
    bench_win = bench_series.reindex(marks).ffill().dropna()
    years = len(daily) / TRADING_DAYS
    bench_total = float(bench_win.iloc[-1] / bench_win.iloc[0] - 1) if len(bench_win) > 1 else 0.0
    bench_cagr = (1 + bench_total) ** (1 / years) - 1 if years > 0 else float("nan")
    excess = cagr - bench_cagr
    decay = cagr / is_cagr_value if is_cagr_value else float("nan")

    strat = daily["equity"].astype(float).pct_change().fillna(0.0)
    bench_daily = bench_win.pct_change().reindex(strat.index).fillna(0.0)
    excess_daily = strat - bench_daily
    std = excess_daily.std(ddof=1)
    t_stat = float(excess_daily.mean() / std * np.sqrt(len(excess_daily))) if std > 0 else float("nan")

    checks = [
        {"name": "excess_cagr_gt", "value": round(float(excess), 6), "threshold": CRITERIA["excess_cagr_gt"],
         "pass": bool(excess > CRITERIA["excess_cagr_gt"])},
        {"name": "decay_ratio_min", "value": round(float(decay), 4), "threshold": CRITERIA["decay_ratio_min"],
         "pass": bool(np.isfinite(decay) and decay >= CRITERIA["decay_ratio_min"])},
        {"name": "excess_t_min", "value": round(t_stat, 4), "threshold": CRITERIA["excess_t_min"],
         "pass": bool(np.isfinite(t_stat) and t_stat >= CRITERIA["excess_t_min"])},
    ]
    return {"cagr": float(cagr), "bench_cagr": bench_cagr, "excess_cagr": excess,
            "decay_ratio": decay, "excess_t": t_stat, "checks": checks,
            "years": years, "bench_total": bench_total}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="关卡 3：一次性 holdout 验收（机器执行）")
    ap.add_argument("--strategy", default="growth_cap25")
    ap.add_argument("--dry-run", action="store_true", help="只跑关卡 1 与读现状，不消费 holdout")
    ap.add_argument("--force", action="store_true", help="已经消费过时强制再跑（会覆盖旧结论，谨慎）")
    ap.add_argument("--out", type=Path, default=GATES)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record: dict = {"gate": 3, "strategy": args.strategy, "generated": pd.Timestamp.now("UTC").isoformat(),
                    "git_commit": git_commit(), "criteria": CRITERIA, "criteria_sha": criteria_sha()}

    print("[gate3] 关卡 1：离线测试…", flush=True)
    record["gate1"] = gate1_offline()
    print(f"[gate3]   关卡1 {'通过' if record['gate1']['pass'] else '未通过'} "
          f"({record['gate1']['seconds']}s)")

    record["gate2"] = gate2_record(args.strategy)
    if not record["gate2"].get("found"):
        print("[gate3]   未找到关卡 2 报告——按约定只作参考，继续")
    else:
        print(f"[gate3]   关卡2 参考值：passed={record['gate2']['passed']}")

    if MARKER.exists() and not args.force:
        print(f"[gate3] holdout 已被消费过：{MARKER}，拒绝重复验收（要重跑加 --force）")
        record["blocked"] = "holdout-already-consumed"
        (args.out / "holdout-verdict.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return 2

    if args.dry_run:
        print("[gate3] dry-run：不消费 holdout，到此为止")
        record["dry_run"] = True
        (args.out / "holdout-verdict.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M")
    out_dir = args.out / f"holdout-{stamp}"
    print(f"[gate3] 关卡 3：跑 holdout（{out_dir}）…", flush=True)
    run = run_holdout(out_dir, force=True)
    summary = run["summary"]

    row = summary["metrics"]
    daily = pd.read_csv(out_dir / "daily.csv")
    daily.index = pd.to_datetime(daily["date"])
    bench_series = pd.read_csv(out_dir / "benchmark-pool.csv", index_col=0, parse_dates=True).iloc[:, 0]
    outcome = evaluate(daily, bench_series, float(row["cagr"]), is_cagr())
    cagr = outcome["cagr"]
    bench_cagr = outcome["bench_cagr"]
    excess = outcome["excess_cagr"]
    decay = outcome["decay_ratio"]
    t_stat = outcome["excess_t"]
    checks = outcome["checks"]
    passed = all(c["pass"] for c in checks) and record["gate1"]["pass"]

    record.update({
        "holdout_dir": str(out_dir),
        "gate3": {"cagr": cagr, "bench_cagr": bench_cagr, "excess_cagr": excess,
                  "is_cagr": is_cagr_value, "decay_ratio": decay, "excess_t": t_stat,
                  "sharpe": float(row["sharpe"]), "max_drawdown": float(row["max_drawdown"]),
                  "turnover": float(row.get("ann_turnover_one_way", float("nan")))},
        "checks": checks,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "elapsed_s": round(time.time() - started, 1),
    })
    (args.out / "holdout-verdict.json").write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    marker = {"consumed_at": record["generated"], "commit": record["git_commit"],
              "verdict": record["verdict"], "holdout_dir": str(out_dir),
              "criteria_sha": record["criteria_sha"]}
    MARKER.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[gate3] 结论：{record['verdict']}（{record['elapsed_s']}s）")
    for c in checks:
        print(f"  {'✓' if c['pass'] else '✗'} {c['name']:<20}{c['value']} (需 {c['threshold']})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
