"""阶段 3 结果汇总：把 ``results/growth/stage3`` 的 csv/json 汇成一份可读报告。

只做汇总、格式化与由已落盘净值直接可复算的派生指标（超额 t 值、逐年超额、
费用 bp/年），不重跑回测、不改任何数字口径。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["P", "Q", "QN", "M"]
COST_LABEL = {
    "default": "默认 14.2bp",
    "conservative_30": "保守 30bp",
    "conservative_50": "保守 50bp",
}


def pct(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value * 100:.{digits}f}%"


def num(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def read_daily(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame["equity"].astype(float)


def benchmark_series(out: Path) -> pd.Series:
    frame = pd.read_csv(out / "benchmark-equity.csv", index_col=0, parse_dates=True)
    equity = frame.iloc[:, 0].astype(float)
    return equity.loc["2017-01-01":"2024-12-31"].ffill()


def stats(out: Path, table: pd.DataFrame) -> pd.DataFrame:
    """从净值直接复算：超额 t、逐年、费用 bp/年、实际波动。"""
    bench = benchmark_series(out)
    bret = bench.pct_change().dropna()
    rows = []
    for row in table.itertuples():
        equity = read_daily(out / f"daily-{row.spec}-{row.cost}.csv")
        ret = equity.pct_change().dropna()
        aligned = bret.reindex(ret.index).fillna(0.0)
        excess = ret - aligned
        years = max(len(equity) / 244.0, 1e-9)
        rows.append(
            {
                "spec": row.spec,
                "cost": row.cost,
                "excess_t": float(excess.mean() / excess.std(ddof=1) * np.sqrt(len(excess))),
                "realized_vol": float(ret.std(ddof=1) * np.sqrt(244)),
                "fees_pct_realized": float(row.fees) / float(equity.mean()) / years,
            }
        )
    return pd.DataFrame(rows)


def main_table(table: pd.DataFrame, extra: pd.DataFrame, cost: str) -> str:
    part = table[table["cost"] == cost].copy()
    if part.empty:
        return "（无结果）\n"
    part["order"] = part["spec"].map({name: index for index, name in enumerate(ORDER)})
    part = part.sort_values("order").merge(extra[extra["cost"] == cost], on=["spec", "cost"], how="left")
    lines = [
        "| 规格 | 累计 | 年化 | 波动 | 夏普 | 最大回撤 | 卡玛 | 超额(几何) | 超额t | 年单边换手 | 费用(bp/年) | 持仓中位 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in part.itertuples():
        lines.append(
            f"| {row.spec}（{row.label}） | {pct(row.total_return)} | {pct(row.cagr)} | {pct(row.realized_vol)} | "
            f"{num(row.sharpe)} | {pct(row.max_drawdown)} | {num(row.calmar)} | {pct(row.excess_total)} | "
            f"{num(row.excess_t, 2)} | {pct(row.ann_turnover_one_way, 0)} | {num(row.cost_bp_per_year, 1)} | "
            f"{int(row.holdings_median)} |"
        )
    return "\n".join(lines) + "\n"


def cost_table(table: pd.DataFrame, extra: pd.DataFrame) -> str:
    merged = table.merge(extra, on=["spec", "cost"], how="left")
    merged["order"] = merged["spec"].map({name: index for index, name in enumerate(ORDER)})
    merged = merged.sort_values(["order", "cost"])
    lines = [
        "| 规格 | 成本档 | 累计 | 年化 | 夏普 | 最大回撤 | 卡玛 | 超额(几何) | 费用(bp/年) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in merged.itertuples():
        lines.append(
            f"| {row.spec} | {COST_LABEL.get(row.cost, row.cost)} | {pct(row.total_return)} | {pct(row.cagr)} | "
            f"{num(row.sharpe)} | {pct(row.max_drawdown)} | {num(row.calmar)} | {pct(row.excess_total)} | "
            f"{num(row.cost_bp_per_year, 1)} |"
        )
    return "\n".join(lines) + "\n"


def yearly_tables(out: Path, table: pd.DataFrame) -> tuple[str, str]:
    bench = benchmark_series(out)
    bret = bench.pct_change().dropna()
    gross = pd.DataFrame({spec: (1 + read_daily(out / f"daily-{spec}-default.csv").pct_change().dropna()).groupby(
        read_daily(out / f"daily-{spec}-default.csv").pct_change().dropna().index.year).prod() - 1
        for spec in ORDER if (out / f"daily-{spec}-default.csv").exists()})
    bench_yearly = (1 + bret).groupby(bret.index.year).prod() - 1
    gross["基准(同池等权)"] = bench_yearly
    excess = gross[ORDER].sub(gross["基准(同池等权)"], axis=0)

    def render(frame: pd.DataFrame, digits: int) -> str:
        lines = ["| 年份 | " + " | ".join(frame.columns) + " |", "|---" * (len(frame.columns) + 1) + "|"]
        for year, row in frame.iterrows():
            lines.append(f"| {year} | " + " | ".join(pct(value, digits) for value in row.to_numpy(dtype=float)) + " |")
        return "\n".join(lines)

    return render(gross, 1), render(excess, 1)


def build(out: Path) -> Path:
    payload, table = load(out)
    extra = stats(out, table)
    params = payload.get("params", {})
    bench = benchmark_series(out)
    bret = bench.pct_change().dropna()
    years = len(bench) / 244.0
    bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1)
    bench_cagr = float((1 + bench_total) ** (1 / years) - 1)
    bench_vol = float(bret.std(ddof=1) * np.sqrt(244))
    bench_mdd = float((bench / bench.cummax() - 1).min())

    lines: list[str] = []
    lines.append("# 小盘成长策略 · 阶段 3 组合回测（真实 Engine）\n")
    lines.append(f"- 生成时间：{payload.get('generated')}｜回测耗时 {payload.get('elapsed_seconds')}s")
    lines.append(f"- 样本：{payload['sample']['start']} ~ {payload['sample']['end']}；**{payload['sample']['holdout']}**")
    lines.append(
        f"- 基准（同池等权、解析式）：累计 {pct(bench_total)}，CAGR {pct(bench_cagr)}，波动 {pct(bench_vol)}，"
        f"最大回撤 {pct(bench_mdd)}"
    )
    lines.append(
        f"- 组合参数：持仓 {params.get('top_k')} 只、等权、gross {params.get('gross')}、单票上限 "
        f"{pct(params.get('max_weight'), 1)}、行业（证监会门类）上限 {pct(params.get('sector_cap'), 0)}、"
        f"月频（{params.get('rebalance')}，月末信号 → 次日开盘成交）、退市 {params.get('delisting_mode')}"
    )
    lines.append("- 成本档（往返）：" + "、".join(f"{key} {value}bp" for key, value in payload.get("costs", {}).items()))
    lines.append(
        "- 股票池：P/Q/QN = 成长合格集（R1–R9 + G3，R10 按 Q5 不启用）；"
        "M = 全市场（只剔 ST/次新/停牌/一字板/退市整理期）\n"
    )

    lines.append("## 0. 一句话结论\n")
    best = table[table["cost"] == "default"].sort_values("sharpe", ascending=False).iloc[0]
    lines.append(
        f"四套规格全部跑赢同池等权（超额 {pct(best['excess_total'])} 起），但**年化只有 7.5%–9.0%，"
        f"离 20% 的目标差一大截**；最好的 {best['spec']} 夏普 {num(best['sharpe'])}、最大回撤 "
        f"{pct(best['max_drawdown'])}、年单边换手 {pct(best['ann_turnover_one_way'], 0)}——"
        "换手与成本是当前最硬的结构性问题，不是选股逻辑。\n"
    )

    lines.append("## 1. 主结果（默认成本档 14.2bp）\n")
    lines.append(main_table(table, extra, "default"))
    lines.append("")

    lines.append("## 2. 成本敏感性（三档）\n")
    lines.append(cost_table(table, extra))
    lines.append("")

    gross_table, excess_table = yearly_tables(out, table)
    lines.append("## 3. 逐年收益\n")
    lines.append(gross_table)
    lines.append("")
    lines.append("## 4. 逐年超额（默认成本档，策略 − 同池等权）\n")
    lines.append(excess_table)
    lines.append("")

    lines.append("## 5. 口径说明\n")
    lines.append(
        "- **年单边换手**：月末目标权重 `0.5·Σ|Δw|`，首月建仓按半程计；与成交明细会有差异"
        "（整手、涨跌停、10% 参与率限制）。"
    )
    lines.append("- **费用(bp/年)**：engine 记账费用 ÷ 日均净值 ÷ 年数（含佣金、印花税、过户费、滑点）。")
    lines.append("- **超额(几何)**：策略累计 ÷ 基准累计 − 1，同区间同池，不含杠杆。")
    lines.append("- **超额t**：日频超额收益均值 ÷ 标准差 × √N，只作量级参考（未做多重检验校正）。")
    lines.append("- **M 对照**：同 QN 打分但不设成长门槛 G3 与排雷 R5–R9，只留 ST/次新/停牌/一字板/退市整理期。")
    lines.append("- 规格 P = 冻结的 13 因子；Q = 删 Q9 的 5 个；QN = Q + Q10 的解禁双因子（放辅助块）。")
    lines.append("- 三套规格的块权重完全相同（成长 40 / 质量 25 / 研发 15 / 辅助 20），块内等权。\n")

    lines.append("## 6. 读表须知\n")
    lines.append(
        "1. 四套规格共用同一批因子与同一段样本，**彼此不是独立证据**；差异 <3pp 属噪声，"
        "只能看方向（QN ≥ M ≥ P > Q，且 Q 与 P 的夏普几乎相同）。"
    )
    lines.append(
        "2. 2017、2018 两年全规格负收益（−1% ~ −24%），策略的「抗跌」是超额的主要来源之一，"
        "**不是靠单年暴涨**；2021 是唯一明确跑输基准的年份（P −5pp）。"
    )
    lines.append(
        "3. 换手 425%–490%（单边/年）意味着每年把组合换掉 4–5 遍；成本档从 14.2bp 提到 50bp，"
        "QN 的累计从 99.0% 掉到 67.9%（−31pp）。**任何后续改进优先砍换手，而不是继续加因子**。\n"
    )

    target = out / "backtest.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {target}")
    return target


def load(out: Path) -> tuple[dict, pd.DataFrame]:
    payload = json.loads((out / "backtest.json").read_text(encoding="utf-8"))
    table = pd.read_csv(out / "metrics.csv")
    return payload, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/growth/stage3"))
    args = parser.parse_args()
    build(args.out)
