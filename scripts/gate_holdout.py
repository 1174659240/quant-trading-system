"""关卡 3：一次性样本外（holdout）验收——机器执行，不看人脸色。

设计意图：holdout 只能被这个脚本消费一次，判据在 ``CRITERIA`` 里写死并带 sha256。
脚本会：
0. **先校验关卡 2 的硬否决层**（新版 gate2-v2）：关卡 1 通过、置换 p 达标、成本档下年化为正、
   目标规模参与率达标，缺一项就直接拒绝（非 0 退出），不碰 holdout；
1. 跑关卡 1（离线测试，``--skip-lake``），记录退出码；
2. 读最近的关卡 2 报告，取它的强参考层结论（是否「多重稳健性告警」）；
3. 用冻结姿态跑一次 holdout（``scripts/growth_holdout``），写进 ``results/gates/holdout-<stamp>/``；
4. 按预注册判据判断「换一个新样本后是否掉太多」，分四步：
   a) 先认行情：用同池等权基准算 holdout 窗口的年化/波动/最大回撤，打一个行情标签；
   b) 回样本内找同类行情：把样本内切成与 holdout 等长的滚动窗口，逐窗打同样的标签；
   c) 定底线：取同类窗口里策略年化、夏普的 25% 分位作为底线（同类窗口太少时退化为全样本分位）；
   d) 判决：holdout 的年化、夏普都不低于底线，且相对同池等权不落后超过 5pp/年，才算 PASS。
   新样本的行情与策略在其中的相对位置（beta、相关、月度胜率、日频超额 t）如实记录，供人看；
5. 三项全过才 ``PASS``，否则 ``FAIL``；若关卡 2 有「多重稳健性告警」，即使行情达标也判 ``HOLD``，
   要所有者用 ``--owner-confirm "理由"`` 才能放行；
6. 结论写进 ``results/gates/holdout-verdict.json``。

**没有 --tune、没有 --relax**：想改判据只能改这份文件（sha256 会变，报告里会显示），
且必须在 ``docs/放行流程.md`` 里先写预注册。

用法::

    python -X utf8 -m scripts.gate_holdout            # 正式验收（会消费 holdout）
    python -X utf8 -m scripts.gate_holdout --dry-run  # 只做关卡 2 校验 + 关卡 1，不动 holdout
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
IS_DAILY = Path("results/risk-system/cap25/daily-A-main.csv")
IS_BENCH = Path("results/risk-system/cap25/benchmark-main.csv")

#: 预注册判据 v3（2026-10-23）：先认行情、再找样本内同类行情、用同类行情的下分位做底线。
#: 改这里 = 改协议，必须同步 docs/放行流程.md；``criteria_sha`` 会变，记录里留痕。
CRITERIA_VERSION = "v3"
CRITERIA = {
    "cagr_floor_quantile": 0.25,    # 底线 = 同类行情窗口年化的 25% 分位
    "sharpe_floor_quantile": 0.25,  # 底线 = 同类行情窗口夏普的 25% 分位
    "excess_cagr_min": -0.05,       # 相对同池等权不得落后超过 5pp/年
    "min_regime_windows": 4,        # 同类行情窗口少于 4 个则退化为全样本分位
    "window_step": 5,               # 样本内滚动窗口的步长（交易日）
}

#: 判据变更记录（照抄进每次的 verdict 文件，方便日后追溯）
CRITERIA_NOTE = (
    "v3 于 2026-10-23 由所有者确定：先看 holdout 是什么行情，再回到样本内找同类行情，"
    "用策略在同类行情里的 25% 分位当底线，达不到就判不通过；相对基准只作兜底与参考。"
    "v1（日频超额 t≥2）在 1.8 年样本上不可达，v2（全样本保留率）不区分行情，均被取代。"
    "注意：这份 holdout 已被消费过，本次判决不具备盲测性质，只能作为流程演练与参考记录。"
)

TRADING_DAYS = 244
RISK_FREE = 0.02


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


def gate2_precheck(strategy: str = "growth_cap25", *, gates: Path | None = None) -> dict:
    """关卡 3 的前置闸门：关卡 2 的硬否决层必须过，否则不许碰 holdout。

    校验四件事：报告存在且是新版（gate2-v2）、判据 sha 与当前一致（判据变过要重跑）、
    预注册仍与锁文件一致、硬否决层全过。强参考层的「多重告警」不在这里拦，
    它会在判决阶段把 PASS 降级为 HOLD。
    """
    from scripts import gate_robustness as g2

    gates = gates or GATES
    files = sorted(gates.glob(f"gate2-{strategy}-*.json"))
    if not files:
        return {"ok": False, "reason": f"未找到关卡 2 报告（{gates}/gate2-{strategy}-*.json）"}
    path = files[-1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "reason": f"关卡 2 报告读不出来：{exc}", "file": str(path)}
    if data.get("version") != g2.CRITERIA_VERSION:
        return {"ok": False, "file": str(path),
                "reason": f"关卡 2 报告是旧格式（{data.get('version')}，无硬否决层），需重跑"}
    if data.get("criteria_sha") != g2.criteria_sha():
        return {"ok": False, "file": str(path),
                "reason": "关卡 2 判据已变（criteria_sha 不一致），需重跑关卡 2"}
    if data.get("light"):
        return {"ok": False, "file": str(path), "reason": "关卡 2 是轻量模式，硬否决未计算"}
    prereg = g2.verify_prereg(strategy)
    if not prereg["ok"]:
        return {"ok": False, "file": str(path), "reason": f"预注册失效：{prereg['reason']}"}
    hard = data.get("hard") or {}
    if not hard.get("passed"):
        failed = ", ".join(hard.get("failed") or []) or "未知"
        return {"ok": False, "file": str(path), "reason": f"关卡 2 硬否决未通过：{failed}"}
    soft = data.get("soft") or {}
    return {"ok": True, "file": str(path), "verdict": data.get("verdict"),
            "archetype": data.get("archetype"), "target_aum": data.get("target_aum"),
            "multi_alarm": bool(soft.get("multi_alarm")),
            "soft_failed": soft.get("failed") or [], "generated": data.get("generated")}


def gate3_verdict(checks_passed: bool, gate1_passed: bool, *, multi_alarm: bool = False,
                  owner_confirm: str | None = None) -> dict:
    """把三层结果合成最终判决（纯函数，便于单测）。

    行情达标 + 关卡 1 达标 = 通过；若关卡 2 报「多重稳健性告警」，默认降级为 HOLD，
    只有所有者显式 ``--owner-confirm`` 才放行。
    """
    passed = bool(checks_passed and gate1_passed)
    if passed and multi_alarm and not owner_confirm:
        return {"passed": False, "verdict": "HOLD", "blocked_by": "gate2-multi-alarm",
                "owner_confirmation": None}
    return {"passed": passed, "verdict": "PASS" if passed else "FAIL", "blocked_by": None,
            "owner_confirmation": (owner_confirm if (passed and multi_alarm) else None)}


def is_frames() -> tuple[pd.DataFrame, pd.Series]:
    """样本内（冻结口径）的日频净值与同池等权基准。"""
    daily = pd.read_csv(IS_DAILY, parse_dates=["date"]).set_index("date")
    bench = pd.read_csv(IS_BENCH, parse_dates=["date"]).set_index("date")
    return daily, bench.iloc[:, 0].astype(float)


def _sharpe(returns: pd.Series) -> float:
    vol = float(returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    if not np.isfinite(vol) or vol <= 0:
        return float("nan")
    return (float(returns.mean()) * TRADING_DAYS - RISK_FREE) / vol


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def _cagr(equity: pd.Series) -> float:
    years = len(equity) / TRADING_DAYS
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def _regime_label(bench_cagr: float, bench_vol: float, bench_max_drawdown: float) -> str:
    """给一段行情起个粗标签（趋势 × 波动形态），用来在样本内找同类窗口。"""
    if not np.isfinite(bench_cagr):
        return "未知"
    trend = "上涨" if bench_cagr >= 0.10 else ("下跌" if bench_cagr <= -0.10 else "震荡")
    if bench_vol >= 0.30 or bench_max_drawdown <= -0.30:
        shape = "高波动/深回撤"
    elif bench_vol <= 0.18:
        shape = "低波动"
    else:
        shape = "中等波动"
    return f"{trend}·{shape}"


def _window_stats(equity: pd.Series, bench: pd.Series) -> dict:
    """一段窗口里：策略自己的表现 + 同池等权基准的行情标签。"""
    bench = bench.reindex(equity.index).ffill().dropna()
    equity = equity.loc[bench.index]
    returns = equity.pct_change().dropna()
    b_returns = bench.pct_change().dropna()
    b_vol = float(b_returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    b_mdd = _max_drawdown(bench / bench.iloc[0])
    b_cagr = _cagr(bench)
    return {"cagr": _cagr(equity), "sharpe": _sharpe(returns),
            "bench_cagr": b_cagr, "bench_vol": b_vol, "bench_max_drawdown": b_mdd,
            "label": _regime_label(b_cagr, b_vol, b_mdd)}


def regime_reference(daily: pd.DataFrame, bench: pd.Series, *, window_days: int) -> dict:
    """把样本内切成与 holdout 等长的滚动窗口，逐窗打行情标签。"""
    equity = daily["equity"].astype(float)
    bench = bench.reindex(equity.index).ffill()
    keep = bench.notna()
    equity, bench = equity[keep], bench[keep]
    step = int(CRITERIA["window_step"])
    rows = []
    for start in range(0, len(equity) - window_days + 1, step):
        rows.append(_window_stats(equity.iloc[start:start + window_days],
                                  bench.iloc[start:start + window_days]))
    return {"window_days": int(window_days), "windows_total": len(rows), "rows": rows}


def _quantile(values, q: float) -> float:
    arr = np.asarray([v for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else float("nan")


def regime_floor(reference: dict, label: str) -> dict:
    """同类行情窗口的表现下分位就是底线；同类窗口太少时退化为全样本分位。"""
    same = [r for r in reference["rows"] if r["label"] == label]
    fallback = len(same) < int(CRITERIA["min_regime_windows"])
    used = reference["rows"] if fallback else same
    cagrs = [r["cagr"] for r in used]
    sharpes = [r["sharpe"] for r in used]
    return {
        "regime": label,
        "windows_total": reference["windows_total"],
        "windows_same_regime": len(same),
        "n_used": len(used),
        "basis": "all_windows" if fallback else "same_regime",
        "fallback": bool(fallback),
        "window_days": reference["window_days"],
        "cagr_floor": _quantile(cagrs, CRITERIA["cagr_floor_quantile"]),
        "sharpe_floor": _quantile(sharpes, CRITERIA["sharpe_floor_quantile"]),
        "cagr_median": _quantile(cagrs, 0.50),
        "sharpe_median": _quantile(sharpes, 0.50),
        "cagr_best": _quantile(cagrs, 1.0),
        "cagr_worst": _quantile(cagrs, 0.0),
    }


def evaluate(daily: pd.DataFrame, bench_series: pd.Series, *, cagr: float, sharpe: float,
             floor: dict) -> dict:
    """纯函数：按 v3 判据出结论（可单测）。

    判的是「换一个新样本后，是否跌破了同类行情下的底线」，看两件事：
    年化是否低于同类行情窗口的 25% 分位、夏普是否低于同类行情窗口的 25% 分位；
    再加一条兜底：相对同池等权年化落后不得超过 5pp，防「底线很低但仍被市场甩开」。
    新样本的行情与策略相对位置（beta、相关、月度胜率、日频超额 t）如实记录，只作参考。
    """
    marks = pd.to_datetime(daily.index)
    bench_win = bench_series.reindex(marks).ffill().dropna()
    years = len(daily) / TRADING_DAYS
    bench_total = float(bench_win.iloc[-1] / bench_win.iloc[0] - 1) if len(bench_win) > 1 else 0.0
    bench_cagr = (1 + bench_total) ** (1 / years) - 1 if years > 0 else float("nan")
    bench_returns = bench_win.pct_change().dropna()
    bench_vol = float(bench_returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    bench_mdd = _max_drawdown(bench_win / bench_win.iloc[0])

    equity = daily["equity"].astype(float)
    strat_returns = equity.pct_change().fillna(0.0)
    strat_mdd = _max_drawdown(equity)
    excess_cagr = cagr - bench_cagr

    bench_daily = bench_returns.reindex(strat_returns.index).fillna(0.0)
    excess_daily = strat_returns - bench_daily
    std = excess_daily.std(ddof=1)
    excess_t = float(excess_daily.mean() / std * np.sqrt(len(excess_daily))) if std > 0 else float("nan")

    both = pd.concat([strat_returns, bench_daily], axis=1).dropna()
    beta = corr = float("nan")
    if len(both) > 2 and both.iloc[:, 1].var() > 0:
        beta = float(both.cov().iloc[0, 1] / both.iloc[:, 1].var())
        corr = float(both.corr().iloc[0, 1])

    monthly = equity.resample("ME").last().pct_change().dropna()
    bench_monthly = (bench_win / bench_win.iloc[0]).resample("ME").last().pct_change().dropna()
    monthly_excess = (monthly - bench_monthly.reindex(monthly.index)).dropna()
    monthly_win_rate = float((monthly_excess > 0).mean()) if len(monthly_excess) else float("nan")

    checks = [
        {"name": "cagr_floor", "value": round(float(cagr), 4),
         "threshold": round(float(floor["cagr_floor"]), 4),
         "pass": bool(np.isfinite(cagr) and np.isfinite(floor["cagr_floor"])
                      and cagr >= floor["cagr_floor"])},
        {"name": "sharpe_floor", "value": round(float(sharpe), 4),
         "threshold": round(float(floor["sharpe_floor"]), 4),
         "pass": bool(np.isfinite(sharpe) and np.isfinite(floor["sharpe_floor"])
                      and sharpe >= floor["sharpe_floor"])},
        {"name": "excess_cagr_min", "value": round(float(excess_cagr), 6),
         "threshold": CRITERIA["excess_cagr_min"],
         "pass": bool(np.isfinite(excess_cagr) and excess_cagr > CRITERIA["excess_cagr_min"])},
    ]
    reference = {
        "regime": floor["regime"], "window_days": floor["window_days"],
        "windows_total": floor["windows_total"], "windows_same_regime": floor["windows_same_regime"],
        "floor_basis": floor["basis"], "floor_fallback": floor["fallback"],
        "cagr_floor": round(floor["cagr_floor"], 4), "sharpe_floor": round(floor["sharpe_floor"], 4),
        "cagr_median_same_regime": round(floor["cagr_median"], 4),
        "sharpe_median_same_regime": round(floor["sharpe_median"], 4),
        "cagr_best_same_regime": round(floor["cagr_best"], 4),
        "cagr_worst_same_regime": round(floor["cagr_worst"], 4),
        "bench_cagr": round(float(bench_cagr), 4), "bench_vol": round(bench_vol, 4),
        "bench_max_drawdown": round(bench_mdd, 4), "bench_total": round(bench_total, 4),
        "excess_cagr": round(float(excess_cagr), 4), "excess_t": round(float(excess_t), 3),
        "beta": round(beta, 3) if np.isfinite(beta) else None,
        "corr": round(corr, 3) if np.isfinite(corr) else None,
        "monthly_win_rate": round(monthly_win_rate, 3) if np.isfinite(monthly_win_rate) else None,
        "years": round(years, 2),
    }
    return {"cagr": float(cagr), "bench_cagr": bench_cagr, "excess_cagr": excess_cagr,
            "is_cagr": float("nan"), "sharpe": float(sharpe),
            "strat_max_drawdown": strat_mdd, "checks": checks, "reference": reference,
            "years": years, "bench_total": bench_total, "floor": floor}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="关卡 3：一次性 holdout 验收（机器执行）")
    ap.add_argument("--strategy", default="growth_cap25")
    ap.add_argument("--dry-run", action="store_true", help="只跑关卡 2 校验 + 关卡 1，不消费 holdout")
    ap.add_argument("--force", action="store_true", help="已经消费过时强制再跑（会覆盖旧结论，谨慎）")
    ap.add_argument("--owner-confirm", default=None, metavar="REASON",
                    help="关卡 2 报「多重稳健性告警」时，所有者写理由强行放行（会写进结论文件）")
    ap.add_argument("--out", type=Path, default=GATES)
    ap.add_argument("--evaluate-existing", type=Path, default=None,
                    help="用当前判据重算一份已消费的 holdout 产物（只读，不覆盖一次性标记）")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record: dict = {"gate": 3, "strategy": args.strategy, "generated": pd.Timestamp.now("UTC").isoformat(),
                    "git_commit": git_commit(), "criteria": CRITERIA, "criteria_sha": criteria_sha(),
                    "criteria_version": CRITERIA_VERSION, "criteria_note": CRITERIA_NOTE}

    print("[gate3] 关卡 2 前置校验：硬否决层…", flush=True)
    pre = gate2_precheck(args.strategy, gates=args.out)
    record["gate2"] = pre
    if not pre["ok"]:
        print(f"[gate3] 拒绝进入关卡 3：{pre['reason']}")
        record["blocked"] = "gate2-hard-veto"
        record["verdict"] = "REJECTED"
        record["elapsed_s"] = round(time.time() - started, 1)
        (args.out / "holdout-verdict.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return 3
    print(f"[gate3]   通过（{Path(pre['file']).name}，"
          f"多重告警={'是' if pre['multi_alarm'] else '否'}）")

    print("[gate3] 关卡 1：离线测试…", flush=True)
    record["gate1"] = gate1_offline()
    print(f"[gate3]   关卡1 {'通过' if record['gate1']['pass'] else '未通过'} "
          f"({record['gate1']['seconds']}s)")

    multi_alarm = bool(pre.get("multi_alarm"))
    if multi_alarm:
        print("[gate3]   注意：关卡 2 报「多重稳健性告警」，行情达标也只会判 HOLD，"
              "需要 --owner-confirm 才放行")

    if args.evaluate_existing is not None:
        hold_dir = args.evaluate_existing
        summary = json.loads((hold_dir / "summary.json").read_text(encoding="utf-8"))
        record["mode"] = "recheck-existing"
        record["holdout_dir"] = str(hold_dir)
        record["note"] = "重算：这份 holdout 已被消费，结论不是盲测结果，只用于核对判据与留痕。"
        assess(hold_dir, record, summary, multi_alarm=multi_alarm,
               owner_confirm=args.owner_confirm)
        return finish(record, args.out, started, consume=False,
                      filename="holdout-verdict-recheck.json")

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
    record["holdout_dir"] = str(out_dir)
    assess(out_dir, record, run["summary"], multi_alarm=multi_alarm,
           owner_confirm=args.owner_confirm)
    return finish(record, args.out, started, consume=True)


def assess(hold_dir: Path, record: dict, summary: dict, *, multi_alarm: bool = False,
           owner_confirm: str | None = None) -> dict:
    """读一份 holdout 产物，按当前判据算结论并写进 record（不碰标记、不写文件）。"""
    row = summary["metrics"]
    daily = pd.read_csv(hold_dir / "daily.csv")
    daily.index = pd.to_datetime(daily["date"])
    bench_series = pd.read_csv(hold_dir / "benchmark-pool.csv", index_col=0, parse_dates=True).iloc[:, 0]
    is_daily, is_bench = is_frames()
    is_ref = regime_reference(is_daily, is_bench, window_days=len(daily))
    hold_stats = _window_stats(daily["equity"].astype(float), bench_series.reindex(daily.index).ffill())
    floor = regime_floor(is_ref, hold_stats["label"])
    outcome = evaluate(daily, bench_series, cagr=float(row["cagr"]),
                       sharpe=float(row["sharpe"]), floor=floor)
    checks = outcome["checks"]
    verdict = gate3_verdict(all(c["pass"] for c in checks), record["gate1"]["pass"],
                            multi_alarm=multi_alarm, owner_confirm=owner_confirm)

    record.update({
        "gate3": {"cagr": outcome["cagr"], "sharpe": outcome["sharpe"],
                  "cagr_floor": outcome["floor"]["cagr_floor"],
                  "sharpe_floor": outcome["floor"]["sharpe_floor"],
                  "floor_basis": outcome["floor"]["basis"],
                  "bench_cagr": outcome["bench_cagr"], "excess_cagr": outcome["excess_cagr"],
                  "max_drawdown": float(row["max_drawdown"]),
                  "turnover": float(row.get("ann_turnover_one_way", float("nan")))},
        "reference": outcome["reference"],
        "checks": checks,
        "passed": verdict["passed"],
        "verdict": verdict["verdict"],
        "blocked_by": verdict["blocked_by"],
        "owner_confirmation": verdict["owner_confirmation"],
    })
    return record


def finish(record: dict, out_dir: Path, started: float, *, consume: bool,
           filename: str = "holdout-verdict.json") -> int:
    """落盘结论、打印、必要时写一次性标记。"""
    record["elapsed_s"] = round(time.time() - started, 1)
    (out_dir / filename).write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    if consume:
        marker = {"consumed_at": record["generated"], "commit": record["git_commit"],
                  "verdict": record["verdict"], "holdout_dir": record.get("holdout_dir"),
                  "criteria_sha": record["criteria_sha"]}
        MARKER.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[gate3] 结论：{record['verdict']}（{record['elapsed_s']}s）")
    for c in record["checks"]:
        print(f"  {'✓' if c['pass'] else '✗'} {c['name']:<20}{c['value']} (需 {c['threshold']})")
    ref = record["reference"]
    basis = "同类行情" if not ref["floor_fallback"] else "同类窗口不足，退化为全样本"
    print(f"  行情：新样本 {ref['regime']}，同池等权年化 {ref['bench_cagr']:.2%}、"
          f"波动 {ref['bench_vol']:.1%}、最大回撤 {ref['bench_max_drawdown']:.1%}")
    print(f"  底线：{basis}（{ref['windows_same_regime']}/{ref['windows_total']} 个等长窗口）"
          f"年化下分位 {ref['cagr_floor']:.2%}、夏普下分位 {ref['sharpe_floor']:.3f}；"
          f"同类中位年化 {ref['cagr_median_same_regime']:.2%}")
    print(f"  参考：beta {ref['beta']}、相关 {ref['corr']}、月度胜率 {ref['monthly_win_rate']}、"
          f"日超额 t {ref['excess_t']}、相对基准超额 {ref['excess_cagr']:.2%}/年")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
