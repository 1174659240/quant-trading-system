"""策略登记：把两条策略固定下来、各跑一遍、留档。

    红利低波策略        4 因子月度选股 + **逐日拉回目标权重**（现状，broker 每天把权重调回 0.95/n）
    红利低波增强策略    **同一份权重矩阵** + 只在调仓日成交，其余交易日不动（执行层修复）

两条策略的**选股完全相同**（同一份 weights 矩阵、同样的 50 只、同样每月换一次名单），
差别 100% 来自执行：非调仓日是否把权重拉回目标。所以配对差异就是「逐日再平衡」的净成本。

⚠️ 增强 ≠ 加因子。股东户数（gdhs）的两种用法都试过、都没通过预注册门槛
（等权并入 -5.97pp；硬过滤 -16pp；见 results/enhance-alternative、results/enhance-filter），
**本文件不含任何新因子**。

产物（全部落在 results/strategy-registry/，不碰 results/crisis、results/holdout）：
    STRATEGIES.md      两份策略的定义、指标、差异归因、使用边界
    metrics.csv        两条策略 × 三档成本
    yearly.csv         逐年收益
    equity-*.csv       净值曲线（三档成本）
    holdings.csv       调仓日持仓（日期/代码/权重），可据此重放
    registry-meta.json 口径与运行环境
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import redlow_study as base
from quant.backtest.engine import Engine
import quant.factors.events as event_factors
import scripts.improvement_shadow as sh

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("results/strategy-registry")
COSTS = ("default", "conservative_30", "conservative_50")
COST_LABEL = {"default": "14.2bp", "conservative_30": "30bp", "conservative_50": "50bp"}

STRATS = (
    {
        "name": "红利低波策略",
        "short": "redlow",
        "policy": "daily",
        "engine": "Engine",
        "exec": "逐日拉回目标权重：非调仓日也成交（现状）",
    },
    {
        "name": "红利低波增强策略",
        "short": "redlow_enhanced",
        "policy": "monthly",
        "engine": "Shadow",
        "exec": "只在调仓日成交，其余交易日完全不动（执行层修复）",
    },
)


# ---------------------------------------------------------------- 运行
def run_all(limit_symbols: int | None = None) -> dict:
    started = time.time()
    event_factors.register()
    print("[登记] 构建上下文（只读 ≤2024 数据）……", flush=True)
    bars, panel, panel_bt, block, block_full, delisting, bench = sh.build_sample_context(limit_symbols)
    bt_bars = {s: f.loc[sh.START: sh.END] for s, f in bars.items() if len(f)}

    print("[登记] 生成权重（4 因子、月度选股）……", flush=True)
    weights, reb = sh.generate_weights(bars, sh.FACTORS_FULL, block, "registry")
    if limit_symbols:
        print(f"[登记] 冒烟模式：只取前 {limit_symbols} 只，数字不可用于结论", flush=True)

    presets = base.cost_presets()
    rows, yearly_rows, stats_rows = [], [], []
    equity_store: dict[str, pd.Series] = {}

    for cost_key in COSTS:
        for strat in STRATS:
            t0 = time.time()
            result, stats = sh.run_variant(
                bt_bars,
                weights,
                policy=strat["policy"],
                rebalance_days=reb,
                costs=presets[cost_key],
                delisting=delisting,
                bench=bench,
                engine_cls=Engine if strat["engine"] == "Engine" else sh.ShadowEngine,
            )
            m = result.report.metrics
            daily = result.daily
            years = max(len(daily) / 244.0, 1e-9)
            total = float(result.equity.iloc[-1] / float(result.equity.iloc[0]) - 1.0)
            rows.append(
                {
                    "strategy": strat["name"],
                    "cost": COST_LABEL[cost_key],
                    "total_return": total,
                    "cagr": float(m["cagr"]),
                    "ann_vol": float(m["ann_vol"]),
                    "sharpe": float(m["sharpe"]),
                    "max_drawdown": float(m["max_drawdown"]),
                    "calmar": float(m["cagr"] / abs(m["max_drawdown"])) if m["max_drawdown"] else np.nan,
                    "avg_turnover": float(m["avg_turnover"]),
                    "fees": float(m["total_fees"]),
                    "trades": int(len(result.trades)),
                    "trade_days": int(stats.get("trade_days", 0) or 0),
                    "non_rebalance_trade_share": float(stats.get("non_rebalance_trade_share", np.nan)),
                    "rebalance_days": int(len(reb)),
                    "seconds": round(time.time() - t0, 1),
                }
            )
            stats_rows.append({"strategy": strat["name"], "cost": COST_LABEL[cost_key], **stats})
            eq = result.equity.copy()
            eq.index.name = "date"
            equity_store[f"{strat['short']}-{cost_key}"] = eq
            print(
                f"  [{cost_key}] {strat['name']:<12} 累计 {total:.2%} 夏普 {float(m['sharpe']):.3f} "
                f"MDD {float(m['max_drawdown']):.2%} 费用 {float(m['total_fees']):,.0f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
            for year, part in eq.groupby(eq.index.year):
                yearly_rows.append(
                    {
                        "strategy": strat["name"],
                        "cost": COST_LABEL[cost_key],
                        "year": int(year),
                        "return": float(part.iloc[-1] / part.iloc[0] - 1.0),
                    }
                )

    frame = pd.DataFrame(rows)
    yearly = pd.DataFrame(yearly_rows)
    stats_frame = pd.DataFrame(stats_rows)

    # 自检：红利低波策略必须复现冻结基线 79.54% / 0.483 / -22.84%
    base_row = frame[(frame["strategy"] == "红利低波策略") & (frame["cost"] == "14.2bp")].iloc[0]
    target = sh.TARGET
    checks = {
        "total_return": abs(float(base_row["total_return"]) - target["total_return"]) < 5e-3,
        "sharpe": abs(float(base_row["sharpe"]) - target["sharpe"]) < 5e-3,
        "max_drawdown": abs(float(base_row["max_drawdown"]) - target["max_drawdown"]) < 5e-3,
    }
    print(f"[自检] 复现冻结基线：{checks}", flush=True)

    # 调仓日持仓（可据此重放）
    holdings = []
    for day in reb:
        if day not in weights.index:
            continue
        row = weights.loc[day]
        for symbol, w in row[row > 0].items():
            holdings.append({"date": day.strftime("%Y-%m-%d"), "symbol": symbol, "weight": float(w)})
    holdings_frame = pd.DataFrame(holdings)

    return {
        "frame": frame,
        "yearly": yearly,
        "stats": stats_frame,
        "equity": equity_store,
        "holdings": holdings_frame,
        "weights": weights,
        "rebalance_days": reb,
        "checks": checks,
        "elapsed": round(time.time() - started, 1),
        "symbols": int(len(bars)),
    }


# ---------------------------------------------------------------- 落盘
def save_all(run: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run["frame"].to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    run["yearly"].to_csv(OUT / "yearly.csv", index=False, encoding="utf-8-sig")
    run["stats"].to_csv(OUT / "engine-stats.csv", index=False, encoding="utf-8-sig")
    run["holdings"].to_csv(OUT / "holdings.csv", index=False, encoding="utf-8-sig")
    for key, eq in run["equity"].items():
        eq.to_frame("equity").to_csv(OUT / f"equity-{key}.csv", encoding="utf-8-sig")
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window": [str(sh.START), str(sh.END)],
        "factors": sh.FACTORS_FULL,
        "top_k": 50,
        "gross_exposure": 0.95,
        "rebalance": "ME（月度选股）",
        "cost_presets": {COST_LABEL[k]: float(base.round_trip_bp(base.cost_presets()[k])) for k in COSTS},
        "symbols": run["symbols"],
        "rebalance_days": int(len(run["rebalance_days"])),
        "selfcheck": run["checks"],
        "elapsed_seconds": run["elapsed"],
        "holdout_untouched": True,
        "notes": [
            "两条策略共用同一份 weights 矩阵，差异只在执行层（非调仓日是否成交）。",
            "不含股东户数/两融等新因子；那些用法已在 results/enhance-* 里判定不通过。",
            "results/crisis、results/holdout 未被读取或写入。",
        ],
    }
    (OUT / "registry-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 报告
def _f(x, nd=2, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}%}" if pct else f"{x:.{nd}f}"


def write_report(run: dict) -> Path:
    frame, yearly, stats = run["frame"], run["yearly"], run["stats"]
    lines: list[str] = []
    add = lines.append

    def at(strategy: str, cost: str = "14.2bp") -> pd.Series:
        return frame[(frame["strategy"] == strategy) & (frame["cost"] == cost)].iloc[0]

    a = at("红利低波策略")
    b = at("红利低波增强策略")

    add("# 策略登记：红利低波策略 / 红利低波增强策略")
    add("")
    add(f"> 生成：{time.strftime('%Y-%m-%d %H:%M')}　样本内 {str(sh.START)[:10]} ~ {str(sh.END)[:10]}　"
        f"4 因子月度选股、top50、gross 0.95、14.2bp 往返（另有 30/50bp 两档）")
    add("> 现有冻结策略与 `results/crisis`、`results/holdout` **未改动**；本文件只读 ≤2024 数据。")
    add("")
    add("## 0. 一句话")
    add("")
    add(f"两条策略**选股完全相同**，只是执行不同：`红利低波策略` 每个交易日都把权重拉回目标（现状），"
        f"`红利低波增强策略` 只在调仓日成交。同一份权重下，样本内累计 {_f(a['total_return'])} → {_f(b['total_return'])}"
        f"（{_f((b['total_return'] - a['total_return']) * 100, 2, False)}pp）、"
        f"夏普 {a['sharpe']:.3f} → {b['sharpe']:.3f}、费用 {a['fees']:,.0f} → {b['fees']:,.0f} 元"
        f"（−{_f((1 - b['fees'] / a['fees']) * 100, 1, False)}%）。")
    add("")
    add("**这不是「加因子」带来的提升**：股东户数、两融这些新数据都试过，没有一条通过预注册门槛"
        "（并入 −5.97pp、硬过滤 −16pp）。增强版之所以更好，只是**不再为「把权重拉回等权」付费**。")
    add("")
    add("## 1. 定义")
    add("")
    add("| | 红利低波策略 | 红利低波增强策略 |")
    add("|---|---|---|")
    add(f"| 因子 | {sh.FACTORS_FULL} | 同左（**完全相同**） |")
    add("| 选股 | 月度 rebalance、top50、等权 0.95 | 同左（同一份 weights 矩阵） |")
    add("| 执行 | 逐日拉回目标权重（非调仓日也成交） | **只在调仓日成交**，其余日不动 |")
    add("| 实现 | `quant.backtest.engine.Engine` | `scripts/improvement_shadow.ShadowEngine(policy='monthly')` |")
    add(f"| 调仓日 | {int(a['rebalance_days'])} 个 | 同左 |")
    add("")
    add("## 2. 样本内指标")
    add("")
    add("| 策略 | 成本 | 累计 | 年化 | 波动 | 夏普 | MDD | Calmar | 日均换手 | 费用(元) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for _, row in frame.iterrows():
        add(f"| {row['strategy']} | {row['cost']} | {_f(row['total_return'])} | {_f(row['cagr'])} | "
            f"{_f(row['ann_vol'])} | {row['sharpe']:.3f} | {_f(row['max_drawdown'])} | "
            f"{_f(row['calmar'], 3, False)} | {_f(row['avg_turnover'])} | {row['fees']:,.0f} |")
    add("")
    add("成交画像（14.2bp）：")
    add("")
    add("| 策略 | 成交笔数 | 有成交的交易日 | 其中非调仓日成交占比 |")
    add("|---|---|---|---|")
    for name in ("红利低波策略", "红利低波增强策略"):
        r = stats[(stats["strategy"] == name) & (stats["cost"] == "14.2bp")].iloc[0]
        add(f"| {name} | {int(at(name)['trades']):,} | {int(r.get('trade_days', 0) or 0)} | "
            f"{_f(float(r.get('non_rebalance_trade_share', float('nan'))))} |")
    add("")
    add("## 3. 逐年（14.2bp）")
    add("")
    pivot = yearly[yearly["cost"] == "14.2bp"].pivot(index="year", columns="strategy", values="return")
    add("| 年份 | 红利低波策略 | 红利低波增强策略 | 差(pp) |")
    add("|---|---|---|---|")
    for year, row in pivot.iterrows():
        diff = (row["红利低波增强策略"] - row["红利低波策略"]) * 100
        add(f"| {year} | {_f(row['红利低波策略'])} | {_f(row['红利低波增强策略'])} | {diff:+.2f} |")
    wins = int((pivot["红利低波增强策略"] > pivot["红利低波策略"]).sum())
    add("")
    add(f"增强版跑赢 {wins}/{len(pivot)} 年。")
    add("")
    add("## 4. 用这两条策略时要知道的边界")
    add("")
    add(f"* **仍然跑不赢纯红利**：纯红利 top50（月频、14.2bp、真 Engine）样本内 {_f(174.36 / 100)}。"
        "与纯红利的差距来自选股信息量，不是执行；换执行方式救不回来。")
    add("* **样本外状态**：增强版的 2025-01~2026-10 只跑过一次（+14.64% vs 现状 +12.64%），"
        "**那一段已被污染**，只能当方向参考，不能当验证；之后要重新封存，"
        "真正的检验必须用 2026-10 之后的实时纸面/小资金。")
    add("* **不是两条独立策略做组合**：它们共用同一份选股，同涨同跌，叠加没有分散意义；"
        "保留两条是为了「同一选股下的两种执行」可对照、可回滚。")
    add("* **不要往上再叠股东户数**：`results/enhance-alternative`（等权并入 −5.97pp）与 "
        "`results/enhance-filter`（硬过滤/倾斜 −16 ~ −34pp，安慰剂净贡献 ±3pp）都已判定不通过。")
    add("* **换手预算**（buffer 2x + 单月换手上限 30%）是另一条已验证可行的执行优化"
        "（在 `results/strategy-optimization` 里对等权方案是 +6.6pp、费用再降 37%），"
        "但它是在另一套因子上验证的，**没有**在红利低波两条策略上重跑；要用需要单独预注册。")
    add("")
    add("## 5. 复算")
    add("")
    add("```")
    add("$py -u scripts/strategy_registry.py            # 全量重跑（约 6 分钟）")
    add("```")
    add("")
    add("产物：`metrics.csv` / `yearly.csv` / `engine-stats.csv` / `equity-*.csv` / `holdings.csv` / "
        "`registry-meta.json`。自检：红利低波策略必须复现冻结基线 79.53%/0.483/−22.84%，"
        f"本次 {', '.join(f'{k}={v}' for k, v in run['checks'].items())}。")
    add("")

    path = OUT / "STRATEGIES.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[完成] {path}（{len(lines)} 行）", flush=True)
    return path


def load_saved() -> dict:
    frame = pd.read_csv(OUT / "metrics.csv")
    yearly = pd.read_csv(OUT / "yearly.csv")
    stats = pd.read_csv(OUT / "engine-stats.csv")
    meta = json.loads((OUT / "registry-meta.json").read_text(encoding="utf-8"))
    return {
        "frame": frame,
        "yearly": yearly,
        "stats": stats,
        "checks": meta.get("selfcheck", {}),
        "symbols": meta.get("symbols", 0),
        "elapsed": meta.get("elapsed_seconds", 0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="红利低波策略 / 增强策略 登记")
    parser.add_argument("--smoke", type=int, default=None, help="只用前 N 只做冒烟")
    parser.add_argument("--report-only", action="store_true", help="从已有 CSV 重生成报告")
    args = parser.parse_args(argv)

    if args.report_only:
        run = load_saved()
        write_report(run)
        return 0

    run = run_all(args.smoke)
    save_all(run)
    write_report(run)
    if not all(run["checks"].values()):
        print("[警告] 自检未通过，数字不可用", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
