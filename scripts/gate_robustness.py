"""关卡 2：策略「毕业」前的稳健性门禁（手动触发，不进 CI）。

分两层，语义完全不同：

**硬否决层**（任何一项不过 → 禁止进入关卡 3、禁止消费 holdout）
1. `gate1_offline`：关卡 1 的离线测试全过（无未来函数 / 数据泄露的回归检查都在里面）；
2. `perm_p`：置换检验 p ≤ 0.05（信号不是随机选股选出来的）；
3. `cost_tier_cagr`：在本策略类型对应的成本档下，年化 > 0；
4. `capacity_p95`：在本策略预注册的目标规模下，单票日参与率 p95 ≤ 10%。

**强参考层**（不自动否决，但必须完整写进报告）
DSR、PBO、bootstrap 下限、IS/OOS 比、参数扰动夏普极差、参数扰动年化极差、
WFA 正年占比、单一状态贡献。其中「≥2 项不达标」或「任一项严重不足」→ 结论标
「多重稳健性告警」；此时即使关卡 3 行情达标，也默认不放行，必须所有者人工确认。

**策略类型分档**：成本档不按一刀切。档位在预注册时锁定（``--declare``），
看到结果后再改档会被拒绝（``--relock --reason`` 才能改，且旧档进 history）。

判据写死在 ``HARD`` / ``SOFT`` / ``SEVERE`` 里，``criteria_sha`` 留痕。
改判据的顺序见 ``docs/放行流程.md`` 第七节。

用法::

    python -X utf8 -m scripts.gate_robustness --declare --strategy growth_cap25 \\
        --archetype small_cap_fast --target-aum 50000000     # 首次预注册（锁定档位）
    python -X utf8 -m scripts.gate_robustness                  # 完整两层检查
    python -X utf8 -m scripts.gate_robustness --light          # 只跑轻量项（开发用，不产判决）
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import growth_robust as R  # noqa: E402

ROOT = Path("results/gates")
PY = sys.executable
CRITERIA_VERSION = "gate2-v2"

#: 策略类型档位：成本检验按档取，不搞一刀切。档位在预注册时锁定。
ARCHETYPES = {
    "small_cap_fast": {
        "label": "小盘高换手型",
        "cost_bp": 30.0,
        "note": "持仓偏小盘、年换手 3~5 倍、容量小；成本检验放宽到 30bp，不要求 50bp",
    },
    "large_cap_slow": {
        "label": "大盘低换手型",
        "cost_bp": 50.0,
        "note": "大市值、低换手、容量大；成本检验按 50bp",
    },
}

#: 硬否决层判定线
HARD = {
    "perm_p_max": 0.05,
    "cost_cagr_min": 0.0,
    "capacity_participation_max": 0.10,
}

#: 强参考层判定线
SOFT = {
    "dsr_min": 0.95,
    "pbo_max": 0.50,
    "bootstrap_lower_min": 0.0,
    "is_oos_ratio_min": 0.50,
    "sensitivity_sharpe_range_max": 0.15,
    "sensitivity_cagr_range_max": 0.50,
    "wfa_positive_share_min": 0.50,
    "state_share_max": 0.70,
}

#: 「严重不足」的界线（比判定线更宽），用于多重告警里的「某项特别差」
SEVERE = {
    "dsr_min": 0.50,
    "pbo_max": 0.70,
    "bootstrap_lower_min": -0.10,
    "is_oos_ratio_min": 0.25,
    "sensitivity_sharpe_range_max": 0.30,
    "sensitivity_cagr_range_max": 1.00,
    "wfa_positive_share_min": 0.30,
    "state_share_max": 0.85,
}

PREREG_DIR = Path("results/gates/prereg")
PREREG_LOCK = Path("results/gates/prereg-lock.json")

STRATEGIES = {
    "growth_cap25": {
        "daily": Path("results/risk-system/cap25/daily-A-main.csv"),
        "weights": Path("results/risk-system/cap25/weights-cap25.csv.gz"),
        "bench": Path("results/risk-system/cap25/benchmark-main.csv"),
        "sector_cap": 0.25,
    },
}


# ---------------------------------------------------------------- 判据指纹 / 预注册
def criteria_payload() -> dict:
    return {"version": CRITERIA_VERSION, "hard": HARD, "soft": SOFT,
            "severe": SEVERE,
            "archetypes": {k: {"label": v["label"], "cost_bp": v["cost_bp"]}
                           for k, v in ARCHETYPES.items()}}


def criteria_sha() -> str:
    return hashlib.sha256(json.dumps(criteria_payload(), sort_keys=True).encode()).hexdigest()[:16]


def script_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _payload_sha(payload: dict) -> str:
    keep = {k: payload.get(k) for k in ("strategy", "archetype", "target_aum")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


def prereg_path(strategy: str, prereg_dir: Path | None = None) -> Path:
    return (prereg_dir or PREREG_DIR) / f"{strategy}.json"


def load_prereg(strategy: str, prereg_dir: Path | None = None) -> dict | None:
    path = prereg_path(strategy, prereg_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_prereg(strategy: str, *, prereg_dir: Path | None = None,
                  lock_path: Path | None = None) -> dict:
    """校验预注册存在、档位合法、且与锁文件一致（防「看完结果再改档」）。"""
    path = prereg_path(strategy, prereg_dir)
    prereg = load_prereg(strategy, prereg_dir)
    if prereg is None:
        return {"ok": False, "reason": f"缺少预注册 {path}（先跑 --declare 锁定档位）",
                "file": str(path)}
    archetype = prereg.get("archetype")
    if archetype not in ARCHETYPES:
        return {"ok": False, "reason": f"档位非法：{archetype}", "file": str(path)}
    if float(prereg.get("target_aum") or 0) <= 0:
        return {"ok": False, "reason": "target_aum 必须为正数", "file": str(path)}
    lock_path = lock_path or PREREG_LOCK
    if not Path(lock_path).is_file():
        return {"ok": False, "reason": f"缺少锁文件 {lock_path}", "file": str(path)}
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    entry = (lock.get("strategies") or {}).get(strategy)
    if not entry:
        return {"ok": False, "reason": "锁文件里没有这条策略", "file": str(path)}
    if entry.get("sha") != _payload_sha(prereg):
        return {"ok": False, "reason": "预注册内容与锁文件不一致（档位或目标规模被改过）",
                "file": str(path)}
    return {"ok": True, "file": str(path), "archetype": archetype,
            "archetype_label": ARCHETYPES[archetype]["label"],
            "cost_bp": ARCHETYPES[archetype]["cost_bp"],
            "target_aum": float(prereg["target_aum"]), "sha": entry.get("sha"),
            "declared_at": prereg.get("declared_at"), "note": prereg.get("note", "")}


def declare(strategy: str, archetype: str, target_aum: float, *, note: str = "",
            relock: bool = False, reason: str = "", prereg_dir: Path | None = None,
            lock_path: Path | None = None) -> dict:
    """写预注册并上锁。已锁定时必须 --relock 且给理由，旧档进 history。"""
    if strategy not in STRATEGIES:
        raise SystemExit(f"未知策略：{strategy}（可选 {sorted(STRATEGIES)}）")
    if archetype not in ARCHETYPES:
        raise SystemExit(f"未知档位：{archetype}（可选 {sorted(ARCHETYPES)}）")
    if float(target_aum) <= 0:
        raise SystemExit("--target-aum 必须为正数")
    prereg_dir = prereg_dir or PREREG_DIR
    lock_path = Path(lock_path or PREREG_LOCK)
    lock = (json.loads(lock_path.read_text(encoding="utf-8"))
            if lock_path.is_file() else {"strategies": {}})
    prev = (lock.get("strategies") or {}).get(strategy)
    if prev and not relock:
        raise SystemExit(f"{strategy} 的档位已锁定（{prev.get('archetype')}，"
                         f"{prev.get('target_aum')}）。改档必须 --relock --reason '...'")

    prereg_dir.mkdir(parents=True, exist_ok=True)
    payload = {"strategy": strategy, "archetype": archetype, "target_aum": float(target_aum),
               "declared_at": pd.Timestamp.now("UTC").isoformat(),
               "criteria_version": CRITERIA_VERSION, "note": note}
    path = prereg_path(strategy, prereg_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    history = list((prev or {}).get("history") or [])
    if prev:
        history.append({"sha": prev.get("sha"), "archetype": prev.get("archetype"),
                        "target_aum": prev.get("target_aum"),
                        "declared_at": prev.get("declared_at"),
                        "replaced_at": payload["declared_at"], "reason": reason})
    lock.setdefault("strategies", {})[strategy] = {
        "sha": _payload_sha(payload), "archetype": archetype, "target_aum": float(target_aum),
        "declared_at": payload["declared_at"], "history": history}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "file": str(path), "sha": _payload_sha(payload), "archetype": archetype,
            "target_aum": float(target_aum), "relocked": bool(prev)}


# ---------------------------------------------------------------- 判定
def _check(name: str, *, layer: str, value, threshold, passed: bool, note: str = "",
           severe: bool = False) -> dict:
    return {"name": name, "layer": layer, "value": value, "threshold": threshold,
            "pass": bool(passed), "severe": bool(severe), "note": note}


def _is_severe(key: str, value) -> bool:
    """比判定线更宽的那条线：过了它就只是「不达标」，过不了就是「严重不足」。"""
    if value is None or not np.isfinite(value):
        return False
    if key.endswith("_min"):
        return float(value) < float(SEVERE[key])
    return float(value) > float(SEVERE[key])


def hard_verdict(checks: list[dict]) -> dict:
    failed = [c["name"] for c in checks if not c["pass"]]
    return {"passed": not failed, "failed": failed, "checks": checks}


def soft_summary(checks: list[dict]) -> dict:
    failed = [c["name"] for c in checks if not c["pass"]]
    severe = [c["name"] for c in checks if c.get("severe")]
    multi_alarm = len(failed) >= 2 or bool(severe)
    if multi_alarm:
        verdict = "多重稳健性告警"
    elif failed:
        verdict = "单项不达标（不触发多重告警）"
    else:
        verdict = "全部达标"
    return {"checks": checks, "failed": failed, "failed_count": len(failed),
            "severe": severe, "multi_alarm": bool(multi_alarm), "verdict": verdict}


def overall_verdict(hard: dict, soft: dict, *, light: bool = False) -> str:
    if light:
        return "LIGHT"
    if not hard["passed"]:
        return "HARD-FAIL"
    if soft["multi_alarm"]:
        return "MULTI-ALARM"
    return "CLEAR"


# ---------------------------------------------------------------- 硬否决：三项实算
def gate1_offline(timeout: int = 900) -> dict:
    started = time.time()
    proc = subprocess.run([PY, "-X", "utf8", "tests/run_all.py", "--skip-lake"],
                          capture_output=True, text=True, timeout=timeout)
    tail = (proc.stdout or "").strip().splitlines()[-3:]
    return {"pass": proc.returncode == 0, "returncode": proc.returncode,
            "seconds": round(time.time() - started, 1), "tail": tail}


def baseline_slippage_for(cost_bp: float) -> dict:
    """把「往返成本档」映射成引擎的单边滑点：用既有的三档成本模型反解。"""
    from scripts import redlow_study as rs
    presets = rs.cost_presets()
    round_trip = {name: rs.round_trip_bp(cm) for name, cm in presets.items()}
    name = min(round_trip, key=lambda k: abs(round_trip[k] - cost_bp))
    return {"preset": name, "round_trip_bp": round(float(round_trip[name]), 2),
            "slippage_bps": float(presets[name].slippage_bps),
            "presets": {k: round(float(v), 2) for k, v in round_trip.items()}}


def cost_at_tier(prepared: dict, spec: dict, cost_bp: float) -> dict:
    """按成本档真跑一遍引擎（权重、成交假设与冻结口径一致，只换滑点）。"""
    info = baseline_slippage_for(cost_bp)
    weights = R._build_weights(prepared, {"sector_cap": spec["sector_cap"]})
    outcome = R.run_engine(prepared, weights, name=f"gate2-cost{info['round_trip_bp']:g}",
                           min_trade=5000.0, slip_map=None,
                           baseline_bps=info["slippage_bps"])
    row = outcome["row"]
    return {"cost_bp": cost_bp, "preset": info["preset"],
            "round_trip_bp": info["round_trip_bp"], "slippage_bps": info["slippage_bps"],
            "cagr": float(row.get("cagr", float("nan"))),
            "sharpe": float(row.get("sharpe", float("nan"))),
            "max_drawdown": float(row.get("max_drawdown", float("nan")))}


def cost_ladder_analytic(prepared: dict) -> dict:
    """参考用：解析式月频在 14.2 / 30 / 50bp 下的年化（快，不跑引擎）。"""
    out = {}
    for bp in (14.2, 30.0, 50.0):
        series = R.analytic_monthly(prepared["score"], prepared["panel"].close, prepared["marks"],
                                    top_k=50, gross=0.95, cost_bp=bp)
        stats = R.stats_block(series, periods_per_year=12)
        out[f"{bp:g}bp"] = {"cagr": float(stats["cagr"]), "sharpe": float(stats["sharpe"])}
    return out


def capacity(prepared: dict, weights_path: Path, *, target_aum: float,
             aums: tuple[float, ...] = (1e7, 5e7, 1e8)) -> dict:
    """按冻结权重估「单票日参与率」：持仓市值 / ADV20，取全部 (日, 票) 的 p95。"""
    if not Path(weights_path).is_file():
        return {"note": f"缺权重文件 {weights_path}", "target_aum": target_aum}
    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    weights.index = pd.to_datetime(weights.index)
    amount = prepared["panel"].amount
    adv = amount.rolling(20, min_periods=10).mean()
    per_day: list[tuple[pd.DatetimeIndex, pd.DataFrame]] = []
    for day in weights.index:
        if day not in adv.index:
            continue
        held = weights.loc[day].dropna()
        held = held[held > 0]  # 权重文件是稠密矩阵，未持仓写 0，必须去掉
        if held.empty:
            continue
        per_day.append((day, held))

    def _p95(aum: float) -> dict:
        ratios = []
        for day, held in per_day:
            adv_row = adv.loc[day].reindex(held.index)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (held * aum / adv_row).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio):
                ratios.append(ratio.to_numpy(dtype=float))
        if not ratios:
            return {"p95": None, "max": None, "n": 0}
        arr = np.concatenate(ratios)
        return {"p95": float(np.nanpercentile(arr, 95)), "max": float(np.nanmax(arr)),
                "n": int(arr.size), "days": len(per_day)}

    ladder = {f"{aum:g}": _p95(aum) for aum in aums}
    return {"target_aum": target_aum, "target": _p95(target_aum), "ladder": ladder,
            "days": len(per_day)}


def state_shares(buckets: dict[str, pd.Series], series: pd.Series) -> dict:
    """各行情状态对总收益的贡献占比（日收益之和口径）。

    注意：分桶互相重叠时占比之和不一定等于 1；而且当另一侧是亏的时候，
    单个桶的占比可以 >1（例如全样本 +10%，其中上涨日 +13%、下跌日 −3%）。
    所以「单状态贡献」检查用 :func:`state_concentration`，不用这个数字。
    """
    total_sum = float(series.sum()) if len(series) else 0.0
    if abs(total_sum) < 1e-12:
        return {name: 0.0 for name in buckets}
    return {name: float(bucket.sum() / total_sum)
            for name, bucket in buckets.items() if len(bucket)}


def state_concentration(buckets: dict[str, pd.Series], series: pd.Series,
                        *, min_days: int = 20) -> float:
    """单一行情状态对**超额收益**的贡献集中度，落在 [0, 1]。

    只在互斥分桶对里算（上涨/下跌、E 触发/未触发）：把负贡献截成 0 之后，
    最大桶占该维度全部正贡献的比例。数值越大 = 超额越只靠单一状态。
    某一侧样本不足 ``min_days`` 天（例如 E 层整段没触发）时该维度不计，
    否则会把它算成 1.0，等于凭空多一个「严重告警」。

    口径说明：这里用「策略 − 同池等权基准」的日超额，不用裸收益。
    纯多头策略在上涨日和下跌日的裸收益天然不对称（上涨日几乎必然贡献大头），
    用裸收益算会把所有多头策略都判成 1.0，指标失去分辨力。
    """
    worst = 0.0
    for pair in (("mom_up", "mom_down"), ("e_trigger", "e_calm")):
        sides = [buckets[name] for name in pair if name in buckets]
        if len(sides) < 2 or min(len(side) for side in sides) < min_days:
            continue
        parts = {name: float(buckets[name].sum()) for name in pair}
        positive = {name: max(value, 0.0) for name, value in parts.items()}
        total = sum(positive.values())
        if total > 0:
            worst = max(worst, max(positive.values()) / total)
    return float(min(max(worst, 0.0), 1.0))


def _bench_returns(path: Path | None, index: pd.DatetimeIndex) -> pd.Series | None:
    """基准日收益（同池等权口径），对齐到策略日频索引。"""
    if path is None or not Path(path).is_file():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    equity = frame.iloc[:, 0].astype(float)
    returns = R.to_returns(equity)
    return returns.reindex(index).fillna(0.0)


def multi_state(prepared: dict, daily: pd.Series,
                bench_daily: pd.Series | None = None) -> dict:
    """策略日收益按市场状态切分：E 层触发/未触发、指数 60 日动量的正/负。

    同时给出超额口径的分桶，用于 :func:`state_concentration`。
    """
    from scripts import growth_leftside as gl
    from scripts import risk_system as rs

    frame = gl.market_frame(prepared["panel"])
    exposure = rs.default_layer(prepared["panel"])[0].reindex(frame.index).fillna(1.0)
    close = prepared["panel"].close
    idx = frame["index"] if "index" in frame else close.mean(axis=1)
    mom60 = idx / idx.shift(60) - 1

    excess = daily - (bench_daily.reindex(daily.index).fillna(0.0)
                      if bench_daily is not None else 0.0)

    buckets: dict[str, pd.Series] = {
        "e_trigger": daily[exposure.reindex(daily.index).fillna(1.0) < 0.999],
        "e_calm": daily[exposure.reindex(daily.index).fillna(1.0) >= 0.999],
        "mom_up": daily[(mom60.reindex(daily.index) > 0).fillna(False)],
        "mom_down": daily[(mom60.reindex(daily.index) <= 0).fillna(False)],
    }
    excess_buckets: dict[str, pd.Series] = {
        "e_trigger": excess[exposure.reindex(daily.index).fillna(1.0) < 0.999],
        "e_calm": excess[exposure.reindex(daily.index).fillna(1.0) >= 0.999],
        "mom_up": excess[(mom60.reindex(daily.index) > 0).fillna(False)],
        "mom_down": excess[(mom60.reindex(daily.index) <= 0).fillna(False)],
    }
    total = float((1 + daily).prod() - 1) if len(daily) else 0.0
    out = {}
    for name, series in buckets.items():
        if len(series) < 20:
            out[name] = {"days": int(len(series)), "note": "样本太短"}
            continue
        stats = R.stats_block(series)
        out[name] = {"days": int(len(series)), "cagr": float(stats["cagr"]),
                     "sharpe": float(stats["sharpe"]), "total": float(stats["total"]),
                     "sum_ret": float(series.sum())}
    if bench_daily is not None:
        out["_excess_share"] = state_shares(excess_buckets, excess)
        out["_excess_sum"] = {name: float(bucket.sum())
                              for name, bucket in excess_buckets.items() if len(bucket)}
    out["_share"] = state_shares(buckets, daily)
    out["_concentration"] = state_concentration(
        excess_buckets if bench_daily is not None else buckets,
        excess if bench_daily is not None else daily)
    out["_benchmark"] = "same_pool_equal_weight" if bench_daily is not None else "none"
    out["_buckets"] = buckets
    out["_excess_buckets"] = excess_buckets
    out["_excess"] = excess
    out["_total_return"] = total
    return out


# ---------------------------------------------------------------- 主流程
def run(strategy: str = "growth_cap25", *, light: bool = False, skip_gate1: bool = False,
        perm_reps: int = 1000, prereg_dir: Path | None = None,
        lock_path: Path | None = None) -> dict:
    spec = STRATEGIES[strategy]
    started = time.time()
    prereg = verify_prereg(strategy, prereg_dir=prereg_dir, lock_path=lock_path)
    if not prereg["ok"]:
        raise SystemExit(f"[gate2] 预注册校验失败：{prereg['reason']}")

    archetype = prereg["archetype"]
    cost_bp = float(prereg["cost_bp"])
    target_aum = float(prereg["target_aum"])

    checks_hard: list[dict] = []
    checks_soft: list[dict] = []

    # 硬否决 1：关卡 1（无未来函数 / 数据泄露的回归检查在离线测试里）
    if skip_gate1:
        g1 = {"pass": False, "returncode": None, "seconds": 0.0,
              "tail": ["--skip-gate1：本次未跑关卡 1"]}
    else:
        print("[gate2] 关卡 1：离线测试…", flush=True)
        g1 = gate1_offline()
    checks_hard.append(_check("gate1_offline", layer="hard", value=g1["returncode"],
                              threshold=0, passed=g1["pass"],
                              note="离线测试全过 = 无已知前视/泄露回归"
                              if not skip_gate1 else "本次跳过，硬否决层不完整"))

    # 强参考四项（轻量）
    fast = R.run_fast(daily_path=spec["daily"], tag=f"-gate2-{strategy}")
    dsr = fast["dsr"]
    dsr_best = max((v["dsr"] for v in dsr.values()), default=0.0)
    checks_soft.append(_check("dsr", layer="soft", value=round(dsr_best, 4),
                              threshold=SOFT["dsr_min"], passed=dsr_best >= SOFT["dsr_min"],
                              note=f"取最宽松的 N={min(dsr, key=lambda k: dsr[k]['n_trials'])}",
                              severe=_is_severe("dsr_min", dsr_best)))
    pbo = fast["pbo"]["pbo"]
    checks_soft.append(_check("pbo", layer="soft", value=round(pbo, 4),
                              threshold=SOFT["pbo_max"], passed=pbo <= SOFT["pbo_max"],
                              severe=_is_severe("pbo_max", pbo)))
    lower = fast["bootstrap"]["main_monthly"]["sharpe_ci"][0]
    checks_soft.append(_check("bootstrap_lower", layer="soft", value=round(lower, 4),
                              threshold=SOFT["bootstrap_lower_min"],
                              passed=lower > SOFT["bootstrap_lower_min"],
                              severe=_is_severe("bootstrap_lower_min", lower)))
    ratio = fast["is_oos"]["cagr_ratio"]
    checks_soft.append(_check("is_oos_ratio", layer="soft",
                              value=None if ratio is None else round(ratio, 4),
                              threshold=SOFT["is_oos_ratio_min"],
                              passed=(ratio is not None) and ratio >= SOFT["is_oos_ratio_min"],
                              severe=_is_severe("is_oos_ratio_min", ratio)))

    result = {
        "gate": 2, "version": CRITERIA_VERSION, "strategy": strategy,
        "archetype": archetype, "archetype_label": prereg["archetype_label"],
        "cost_bp": cost_bp, "target_aum": target_aum,
        "prereg": {"file": prereg["file"], "sha": prereg["sha"],
                   "declared_at": prereg["declared_at"], "note": prereg.get("note", "")},
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "git_commit": git_commit(), "script_sha": script_sha(),
        "criteria": criteria_payload(), "criteria_sha": criteria_sha(),
        "light": bool(light), "elapsed_s": None, "fast": fast,
    }
    if light:
        result["hard"] = hard_verdict(checks_hard)
        result["soft"] = soft_summary(checks_soft)
        result["verdict"] = overall_verdict(result["hard"], result["soft"], light=True)
        result["note"] = ("轻量模式只算预注册与四项快指标；硬否决的置换/成本/容量未计算，"
                          "因此不能作为放行依据。")
        result["elapsed_s"] = round(time.time() - started, 1)
        return result

    prepared = R.prepare("main", sector_cap=spec["sector_cap"])

    # 硬否决 2：置换检验
    print(f"[gate2] 置换检验（{perm_reps} 次）…", flush=True)
    perm = R.permutation_test(prepared["score"], prepared["panel"].close, prepared["marks"],
                              prepared["eligible"], reps=perm_reps)
    checks_hard.append(_check("perm_p", layer="hard", value=round(float(perm["p_value"]), 4),
                              threshold=HARD["perm_p_max"],
                              passed=float(perm["p_value"]) <= HARD["perm_p_max"],
                              note=f"真实 IC {perm['actual_mean_ic']:.4f}"))

    # 硬否决 3：策略类型对应成本档下的年化
    print(f"[gate2] 成本档 {cost_bp:g}bp（{prereg['archetype_label']}）…", flush=True)
    cost = cost_at_tier(prepared, spec, cost_bp)
    checks_hard.append(_check("cost_tier_cagr", layer="hard", value=round(cost["cagr"], 4),
                              threshold=HARD["cost_cagr_min"],
                              passed=np.isfinite(cost["cagr"]) and cost["cagr"] > HARD["cost_cagr_min"],
                              note=f"实际往返 {cost['round_trip_bp']:g}bp（预设 {cost['preset']}，"
                                   f"滑点 {cost['slippage_bps']:g}bp）"))

    # 硬否决 4：目标规模下参与率
    print(f"[gate2] 容量（目标规模 {target_aum:,.0f}）…", flush=True)
    cap = capacity(prepared, spec["weights"], target_aum=target_aum)
    p95 = (cap.get("target") or {}).get("p95")
    checks_hard.append(_check("capacity_p95", layer="hard",
                              value=None if p95 is None else round(p95, 4),
                              threshold=HARD["capacity_participation_max"],
                              passed=(p95 is not None and p95 <= HARD["capacity_participation_max"]),
                              note=f"目标规模 {target_aum:,.0f} 元下单票日参与率 p95"))

    # 强参考四项（重）
    print("[gate2] 参数扰动…", flush=True)
    sens = R.sensitivity_scan(prepared, tag=f"gate2-{strategy}")
    sharpe_range = float(sens["sharpe"].max() - sens["sharpe"].min())
    cagr_range = float(sens["cagr"].max() - sens["cagr"].min())
    checks_soft.append(_check("sensitivity_sharpe_range", layer="soft", value=round(sharpe_range, 4),
                              threshold=SOFT["sensitivity_sharpe_range_max"],
                              passed=sharpe_range <= SOFT["sensitivity_sharpe_range_max"],
                              severe=_is_severe("sensitivity_sharpe_range_max", sharpe_range)))
    checks_soft.append(_check("sensitivity_cagr_range", layer="soft", value=round(cagr_range, 4),
                              threshold=SOFT["sensitivity_cagr_range_max"],
                              passed=cagr_range <= SOFT["sensitivity_cagr_range_max"],
                              severe=_is_severe("sensitivity_cagr_range_max", cagr_range)))

    print("[gate2] walk-forward…", flush=True)
    wf = R.wfa(prepared, first_year=2020, tag=f"gate2-{strategy}")
    wfa_years = int(wf.get("years", 0))
    positive_share = (int(wf.get("positive_years", 0)) / wfa_years) if wfa_years else 0.0
    checks_soft.append(_check("wfa_positive_share", layer="soft", value=round(positive_share, 4),
                              threshold=SOFT["wfa_positive_share_min"],
                              passed=positive_share >= SOFT["wfa_positive_share_min"],
                              note=f"{wf.get('positive_years')}/{wfa_years} 年为正",
                              severe=_is_severe("wfa_positive_share_min", positive_share)))

    print("[gate2] 行情状态切分…", flush=True)
    daily = R.to_returns(R.load_daily(spec["daily"]))
    bench_daily = _bench_returns(spec.get("bench"), daily.index)
    states = multi_state(prepared, daily, bench_daily)
    concentration = state_concentration(states["_excess_buckets"], states["_excess"])
    states.pop("_excess_buckets", None)
    states.pop("_excess", None)
    states.pop("_buckets", None)  # 分桶序列不进 JSON（体积大且无法序列化）
    checks_soft.append(_check("state_share", layer="soft", value=round(concentration, 4),
                              threshold=SOFT["state_share_max"],
                              passed=concentration <= SOFT["state_share_max"],
                              note="超额收益（策略−同池等权基准）在单一行情状态上的集中度",
                              severe=_is_severe("state_share_max", concentration)))

    hard = hard_verdict(checks_hard)
    soft = soft_summary(checks_soft)
    result.update({
        "hard": hard, "soft": soft,
        "verdict": overall_verdict(hard, soft),
        "blocking": not hard["passed"],
        "requires_owner_confirmation": bool(soft["multi_alarm"]),
        "permutation": perm, "cost": cost, "cost_ladder_analytic": cost_ladder_analytic(prepared),
        "capacity": cap, "sensitivity": sens.to_dict(orient="records"), "wfa": wf,
        "states": states,
        "elapsed_s": round(time.time() - started, 1),
    })
    return result


def write_markdown(result: dict, path: Path) -> None:
    hard = result.get("hard", {"checks": [], "passed": False, "failed": []})
    soft = result.get("soft", {"checks": [], "failed": [], "multi_alarm": False, "verdict": ""})
    verdict = {"CLEAR": "可进入关卡 3", "MULTI-ALARM": "多重稳健性告警（需所有者确认）",
               "HARD-FAIL": "硬否决未通过（禁止进入关卡 3）",
               "LIGHT": "轻量模式（不产判决）"}.get(result.get("verdict"), result.get("verdict"))
    lines = [
        f"# 关卡 2 稳健性报告：{result['strategy']}",
        "",
        f"- 生成：{result['generated']}",
        f"- 提交：`{result['git_commit']}`，脚本 sha `{result['script_sha']}`，"
        f"判据 sha `{result['criteria_sha']}`（{result['version']}）",
        f"- 策略类型：**{result['archetype_label']}**（`{result['archetype']}`，"
        f"成本档 {result['cost_bp']:g}bp，目标规模 {result['target_aum']:,.0f} 元）",
        f"- 预注册：`{result['prereg']['file']}`，sha `{result['prereg']['sha']}`，"
        f"锁定于 {result['prereg']['declared_at']}",
        f"- 耗时：{result['elapsed_s']} 秒",
        f"- 结论：**{verdict}**",
        "",
        "## 硬否决层（任何一项不过即禁止进入关卡 3）",
        "",
        "| 检查 | 数值 | 判定线 | 结果 | 备注 |",
        "|---|---|---|---|---|",
    ]
    for c in hard["checks"]:
        lines.append(f"| {c['name']} | {c['value']} | {c['threshold']} | "
                     f"{'✓' if c['pass'] else '✗'} | {c.get('note', '')} |")
    lines += ["", "## 强参考层（必须完整记录）", "",
              "| 检查 | 数值 | 判定线 | 结果 | 严重不足 | 备注 |",
              "|---|---|---|---|---|---|"]
    for c in soft["checks"]:
        lines.append(f"| {c['name']} | {c['value']} | {c['threshold']} | "
                     f"{'✓' if c['pass'] else '✗'} | {'是' if c.get('severe') else ''} | "
                     f"{c.get('note', '')} |")
    lines += ["", f"强参考小结：{soft.get('verdict', '')}；不达标 {soft.get('failed_count', 0)} 项"
                  f"（{', '.join(soft.get('failed', [])) or '无'}）；"
                  f"严重不足 {', '.join(soft.get('severe', [])) or '无'}。"]
    if soft.get("multi_alarm"):
        lines += ["", "> 多重稳健性告警：即使关卡 3 行情达标，也默认不放行，必须所有者人工确认。"]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="关卡 2：稳健性门禁（硬否决 + 强参考）")
    ap.add_argument("--strategy", default="growth_cap25", choices=sorted(STRATEGIES))
    ap.add_argument("--declare", action="store_true", help="写预注册并锁定档位")
    ap.add_argument("--archetype", choices=sorted(ARCHETYPES), default=None)
    ap.add_argument("--target-aum", type=float, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--relock", action="store_true", help="改已锁定的档位（需 --reason）")
    ap.add_argument("--reason", default="")
    ap.add_argument("--light", action="store_true", help="只跑轻量项（开发用，不产判决）")
    ap.add_argument("--skip-gate1", action="store_true", help="不跑离线测试（硬否决层会不完整）")
    ap.add_argument("--perm-reps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=ROOT)
    args = ap.parse_args(argv)

    if args.declare:
        if not args.archetype or not args.target_aum:
            raise SystemExit("--declare 需要 --archetype 与 --target-aum")
        info = declare(args.strategy, args.archetype, args.target_aum, note=args.note,
                       relock=args.relock, reason=args.reason)
        print(f"[gate2] 预注册已写入 {info['file']}，锁 sha {info['sha']}，"
              f"档位 {info['archetype']}（{ARCHETYPES[info['archetype']]['label']}）"
              f"{'，已重锁' if info['relocked'] else ''}")
        return 0

    result = run(args.strategy, light=args.light, skip_gate1=args.skip_gate1,
                 perm_reps=args.perm_reps)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    suffix = "-light" if args.light else ""
    json_path = args.out / f"gate2-{args.strategy}-{stamp}{suffix}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    write_markdown(result, args.out / f"gate2-{args.strategy}-{stamp}{suffix}.md")
    print(f"[gate2] {json_path}")
    print(f"[gate2] verdict={result['verdict']} elapsed={result['elapsed_s']}s "
          f"criteria_sha={result['criteria_sha']}")
    for c in result["hard"]["checks"]:
        print(f"  [硬] {'✓' if c['pass'] else '✗'} {c['name']:<24}{c['value']}")
    for c in result["soft"]["checks"]:
        print(f"  [参考] {'✓' if c['pass'] else '✗'} {c['name']:<24}{c['value']}"
              f"{'  严重' if c.get('severe') else ''}")
    print(f"  强参考小结：{result['soft']['verdict']}")
    if result["verdict"] == "CLEAR":
        return 0
    if result["verdict"] == "LIGHT":
        return 0
    if result["verdict"] == "MULTI-ALARM":
        return 3
    return 4


if __name__ == "__main__":
    sys.exit(main())
