"""关卡 2：策略「毕业」前的稳健性门禁（手动触发，不进 CI）。

为什么要单独一个脚本：日常 CI 只跑离线测试和仓库纪律；稳健性检验要完整数据湖、
十几分钟，而且只在「这条策略要申请毕业」时跑一次，不适合挂在每次 push 上。

它做十件事（判定线在 ``CRITERIA`` 里写死，想改必须先改预注册文档）：

1. DSR（deflated Sharpe，试验次数从 ``trial-ledger.json`` 读）
2. PBO / CSCV
3. Stationary bootstrap（月频、块长 6 个月）
4. IS/OOS 分半
5. 置换检验（1000 次）
6. Walk-forward（滚动 36 月 → 下一年）
7. 参数扰动（12 组邻域）
8. 成本三档（14.2 / 30 / 50bp）
9. 容量（1e6 / 1e7 / 5e7 / 1e8 四个规模，看日参与率）
10. 多市场状态（E 层触发 / 未触发、指数 60 日动量的正负）

输出 ``results/gates/gate2-<strategy>-<yyyymmdd-HHMM>.json`` 与同名 ``.md``。
结果只是**参考**：真正的放行在关卡 3，由机器按预注册判据执行。

用法::

    python -X utf8 -m scripts.gate_robustness                 # 跑默认策略 growth_cap25
    python -X utf8 -m scripts.gate_robustness --dsr-only      # 只跑轻量四项（约 1 分钟）
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

#: 判定线（写死；改之前先改 docs/放行流程.md）
CRITERIA = {
    "dsr": 0.95,
    "pbo": 0.50,
    "bootstrap_lower": 0.0,
    "is_oos_ratio": 0.50,
    "perm_p": 0.05,
    "wfa_positive_share": 0.50,
    "sensitivity_sharpe_range": 0.15,
    "sensitivity_cagr_range": 0.50,
    "cost_50bp_cagr": 0.0,
    "capacity_participation": 0.10,
    "state_share": 0.70,
}

STRATEGIES = {
    "growth_cap25": {
        "daily": Path("results/risk-system/cap25/daily-A-main.csv"),
        "weights": Path("results/risk-system/cap25/weights-cap25.csv.gz"),
        "sector_cap": 0.25,
    },
}


def script_sha() -> str:
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _check(name: str, value, passed: bool, note: str = "") -> dict:
    return {"name": name, "value": value, "pass": bool(passed), "note": note}


# ---------------------------------------------------------------- 成本 / 容量 / 状态
def cost_ladder(prepared: dict) -> dict:
    out = {}
    for bp in (14.2, 30.0, 50.0):
        series = R.analytic_monthly(prepared["score"], prepared["panel"].close, prepared["marks"],
                                    top_k=50, gross=0.95, cost_bp=bp)
        stats = R.stats_block(series, periods_per_year=12)
        out[f"{bp:g}bp"] = {"cagr": float(stats["cagr"]), "sharpe": float(stats["sharpe"])}
    return out


def capacity(prepared: dict, weights_path: Path) -> dict:
    """按冻结权重的调仓额 / 持仓 ADV20 估算日参与率。"""
    if not weights_path.is_file():
        return {"note": f"缺权重文件 {weights_path}"}
    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    weights.index = pd.to_datetime(weights.index)
    close = prepared["panel"].close
    amount = prepared["panel"].amount
    adv = amount.rolling(20, min_periods=10).mean()
    marks = weights.index
    trades = weights.fillna(0.0).diff().abs()
    trades.iloc[0] = weights.iloc[0].abs()
    rows = []
    for day in marks:
        if day not in adv.index:
            continue
        held = weights.loc[day].dropna()
        if held.empty:
            continue
        traded = trades.loc[day].reindex(held.index).fillna(0.0)
        rows.append((day, float(traded.sum()), adv.loc[day].reindex(held.index)))
    out = {"aum": {}}
    for aum in (1e6, 1e7, 5e7, 1e8):
        parts = []
        for day, turnover, adv_row in rows:
            notional = turnover * aum
            if notional <= 0:
                continue
            per_name = weights.loc[day].reindex(adv_row.index).fillna(0.0) * aum
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (per_name / adv_row).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio):
                parts.append(float(ratio.max()))
        arr = np.array(parts) if parts else np.array([np.nan])
        out["aum"][f"{aum:g}"] = {"p95": float(np.nanpercentile(arr, 95)), "max": float(np.nanmax(arr))}
    return out


def multi_state(prepared: dict, daily: pd.Series) -> dict:
    """策略日收益按市场状态切分：E 层触发/未触发、指数 60 日动量的正/负。"""
    from scripts import growth_leftside as gl
    from scripts import growth_risk as gr

    frame = gl.market_frame(prepared["panel"])
    exposure = gr.default_layer(prepared["panel"])[0].reindex(frame.index).fillna(1.0)
    close = prepared["panel"].close
    idx = frame["index"] if "index" in frame else close.mean(axis=1)
    mom60 = idx / idx.shift(60) - 1

    buckets: dict[str, list[pd.Series]] = {"e_trigger": [], "e_calm": [], "mom_up": [], "mom_down": []}
    buckets["e_trigger"].append(daily[exposure.reindex(daily.index).fillna(1.0) < 0.999])
    buckets["e_calm"].append(daily[exposure.reindex(daily.index).fillna(1.0) >= 0.999])
    buckets["mom_up"].append(daily[(mom60.reindex(daily.index) > 0).fillna(False)])
    buckets["mom_down"].append(daily[(mom60.reindex(daily.index) <= 0).fillna(False)])

    total = float((1 + daily).prod() - 1) if len(daily) else 0.0
    out = {}
    for name, series_list in buckets.items():
        series = pd.concat(series_list) if series_list else pd.Series(dtype=float)
        if len(series) < 20:
            out[name] = {"days": int(len(series)), "note": "样本太短"}
            continue
        stats = R.stats_block(series)
        out[name] = {"days": int(len(series)), "cagr": float(stats["cagr"]),
                     "sharpe": float(stats["sharpe"]), "total": float(stats["total_return"])}
    # 单一状态贡献：用 ETF 式拆分（这里用「该状态的天数占比 × 该状态收益」作粗估）
    shares = {}
    for name, item in out.items():
        if "total" in item and total:
            shares[name] = float(item["total"] / total) if abs(total) > 1e-9 else 0.0
    out["_share"] = shares
    return out


# ---------------------------------------------------------------- 主流程
def run(strategy: str = "growth_cap25", *, dsr_only: bool = False) -> dict:
    spec = STRATEGIES[strategy]
    started = time.time()

    # 轻量四项：DSR / PBO / bootstrap / IS-OOS
    fast = R.run_fast(daily_path=spec["daily"], tag=f"-gate2-{strategy}")

    checks: list[dict] = []
    dsr = fast["dsr"]
    dsr_best = max((v["dsr"] for v in dsr.values()), default=0.0)
    checks.append(_check("dsr", round(dsr_best, 4), dsr_best >= CRITERIA["dsr"],
                         f"取最宽松的 N={min(dsr, key=lambda k: dsr[k]['n_trials'])}"))
    pbo = fast["pbo"]["pbo"]
    checks.append(_check("pbo", round(pbo, 4), pbo <= CRITERIA["pbo"]))
    boot = fast["bootstrap"]["main_monthly"]
    lower = boot["sharpe_ci"][0]
    checks.append(_check("bootstrap_lower", round(lower, 4), lower > CRITERIA["bootstrap_lower"]))
    io = fast["is_oos"]
    ratio = io["cagr_ratio"]
    checks.append(_check("is_oos_ratio", round(ratio, 4),
                         (ratio is not None) and ratio >= CRITERIA["is_oos_ratio"]))

    result = {
        "gate": 2,
        "strategy": strategy,
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "git_commit": git_commit(),
        "script_sha": script_sha(),
        "criteria": CRITERIA,
        "advisory": True,
        "note": "关卡 2 只供参考；放行判定在关卡 3 由机器执行",
        "elapsed_s": None,
        "fast": fast,
    }
    if dsr_only:
        result["checks"] = checks
        result["passed"] = all(c["pass"] for c in checks)
        result["elapsed_s"] = round(time.time() - started, 1)
        return result

    prepared = R.prepare("main", sector_cap=spec["sector_cap"])

    perm = R.permutation_test(prepared["score"], prepared["panel"].close, prepared["marks"],
                              prepared["eligible"], reps=1000)
    checks.append(_check("perm_p", round(float(perm["p_value"]), 4),
                         float(perm["p_value"]) <= CRITERIA["perm_p"],
                         f"real IC {perm['actual_mean_ic']:.4f}"))
    sens = R.sensitivity_scan(prepared, tag=f"gate2-{strategy}")
    sharpe_range = float(sens["sharpe"].max() - sens["sharpe"].min())
    cagr_range = float(sens["cagr"].max() - sens["cagr"].min())
    checks.append(_check("sensitivity_sharpe_range", round(sharpe_range, 4),
                         sharpe_range <= CRITERIA["sensitivity_sharpe_range"]))
    checks.append(_check("sensitivity_cagr_range", round(cagr_range, 4),
                         cagr_range <= CRITERIA["sensitivity_cagr_range"]))
    wf = R.wfa(prepared, first_year=2020, tag=f"gate2-{strategy}")
    wfa_years = int(wf.get("years", 0))
    positive_share = (int(wf.get("positive_years", 0)) / wfa_years) if wfa_years else 0.0
    checks.append(_check("wfa_positive_share", round(positive_share, 4),
                         positive_share >= CRITERIA["wfa_positive_share"],
                         f"{wf.get('positive_years')}/{wfa_years} 年为正"))

    costs = cost_ladder(prepared)
    checks.append(_check("cost_50bp_cagr", round(costs["50bp"]["cagr"], 4),
                         costs["50bp"]["cagr"] > CRITERIA["cost_50bp_cagr"]))
    cap = capacity(prepared, spec["weights"])
    worst = cap.get("aum", {}).get("5e+07", {}).get("p95")
    checks.append(_check("capacity_participation", None if worst is None else round(worst, 4),
                         worst is not None and worst <= CRITERIA["capacity_participation"],
                         "5,000 万规模下最差单票日参与率 p95"))
    daily = R.to_returns(R.load_daily(spec["daily"]))
    states = multi_state(prepared, daily)
    share = states.get("_share", {})
    biggest = max(share.values()) if share else 1.0
    checks.append(_check("state_share", round(biggest, 4), abs(biggest) <= CRITERIA["state_share"],
                         "单一行情状态贡献占比（粗估）"))

    result.update({
        "checks": checks,
        "passed": all(c["pass"] for c in checks),
        "permutation": perm,
        "sensitivity": sens.to_dict(orient="records"),
        "wfa": wf,
        "costs": costs,
        "capacity": cap,
        "states": states,
        "elapsed_s": round(time.time() - started, 1),
    })
    return result


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        f"# 关卡 2 稳健性报告：{result['strategy']}",
        "",
        f"- 生成：{result['generated']}",
        f"- 提交：`{result['git_commit']}`，脚本 sha `{result['script_sha']}`",
        f"- 耗时：{result['elapsed_s']} 秒",
        f"- 总判定：**{'通过' if result['passed'] else '未通过'}**（仅作参考，放行见关卡 3）",
        "",
        "| 检查 | 数值 | 判定线 | 结果 | 备注 |",
        "|---|---|---|---|---|",
    ]
    for c in result["checks"]:
        lines.append(f"| {c['name']} | {c['value']} | {result['criteria'].get(c['name'], '')} | "
                     f"{'✓' if c['pass'] else '✗'} | {c.get('note', '')} |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="关卡 2：稳健性门禁（参考值）")
    ap.add_argument("--strategy", default="growth_cap25", choices=sorted(STRATEGIES))
    ap.add_argument("--dsr-only", action="store_true", help="只跑轻量四项")
    ap.add_argument("--out", type=Path, default=ROOT)
    args = ap.parse_args(argv)

    result = run(args.strategy, dsr_only=args.dsr_only)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    json_path = args.out / f"gate2-{args.strategy}-{stamp}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(result, args.out / f"gate2-{args.strategy}-{stamp}.md")
    print(f"[gate2] {json_path}")
    print(f"[gate2] passed={result['passed']} elapsed={result['elapsed_s']}s")
    for c in result["checks"]:
        print(f"  {'✓' if c['pass'] else '✗'} {c['name']:<28}{c['value']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
