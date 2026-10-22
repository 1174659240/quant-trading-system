"""阶段 3.1/3.2 风险控制对照报告：读 ``results/growth/stage3/risk/`` → 写 ``risk-report.md``。

只读已落盘的 CSV/JSON，不重跑回测；所有数字与 ``risk.json`` 一一对应，便于复算。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("results/growth/stage3/risk")
REPORT = "risk-report.md"

GROUP_31 = ["legacy", "legacy5k", "legacy5k_slip", "base", "buf60", "buf60_slip", "vt25", "ddb"]
GROUP_32 = ["buf60_slip", "crisis", "gradient", "legacy5k_slip", "crisis5k", "gradient5k"]
JUDGE = {"vt25": "buf60_slip", "ddb": "buf60_slip", "crisis": "buf60_slip",
         "gradient": "buf60_slip", "crisis5k": "legacy5k_slip", "gradient5k": "legacy5k_slip"}


def pct(value: float) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.2%}"


def num(value: float, digits: int = 3) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def table_rows(rows: list[dict], names: list[str]) -> list[dict]:
    by_name = {row["spec"]: row for row in rows}
    out = []
    for name in names:
        row = by_name.get(name)
        if row is None:
            continue
        out.append(
            {
                "规格": name,
                "说明": row["label"],
                "累计": pct(row["total_return"]),
                "年化": pct(row["cagr"]),
                "夏普": num(row["sharpe"]),
                "MDD": pct(row["max_drawdown"]),
                "卡玛": num(row["calmar"]),
                "单边换手": pct(row["ann_turnover_one_way"]),
                "加权滑点": f"{row['weighted_slip_bp']:.2f}bp",
                "费用/净值·年": pct(row["fees_pct_of_equity"]),
                "平均暴露": num(row["avg_exposure"]),
            }
        )
    return out


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_（无数据）_"
    head = "| " + " | ".join(rows[0].keys()) + " |"
    sep = "|" + "|".join(["---"] * len(rows[0])) + "|"
    body = ["| " + " | ".join(str(value) for value in row.values()) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def yearly(daily: pd.DataFrame) -> pd.Series:
    equity = daily.set_index(pd.to_datetime(daily["date"]))["equity"]
    year_end = equity.groupby(equity.index.year).last()
    start = pd.concat([pd.Series([equity.iloc[0]], index=[equity.index[0].year - 1]), year_end])
    return start.pct_change().dropna()


def judge_rows(rows: list[dict]) -> list[dict]:
    by_name = {row["spec"]: row for row in rows}
    out = []
    for name, control in JUDGE.items():
        row, base = by_name.get(name), by_name.get(control)
        if row is None or base is None:
            continue
        # MDD 都是负数，越接近 0 越好 → 改善幅度 = |对照| − |方案|
        mdd_gain = abs(base["max_drawdown"]) - abs(row["max_drawdown"])
        sharpe_ok = row["sharpe"] >= base["sharpe"]
        mdd_ok = mdd_gain >= 0.03
        out.append(
            {
                "方案": name,
                "对照": control,
                "MDD 改善": f"{mdd_gain:+.2%}",
                "≥3pp": "✅" if mdd_ok else "❌",
                "夏普变化": f"{base['sharpe']:.3f} → {row['sharpe']:.3f}",
                "夏普不降": "✅" if sharpe_ok else "❌",
                "累计变化": f"{base['total_return']:.2%} → {row['total_return']:.2%}",
                "结论": "保留" if (mdd_ok and sharpe_ok) else "剔除",
            }
        )
    return out


def main() -> int:
    payload = json.loads((OUT / "risk.json").read_text(encoding="utf-8"))
    rows = payload["rows"]
    tiers = pd.read_csv(OUT / "liquidity-tiers.csv") if (OUT / "liquidity-tiers.csv").exists() else pd.DataFrame()
    episodes = pd.read_csv(OUT / "crisis-episodes.csv") if (OUT / "crisis-episodes.csv").exists() else pd.DataFrame()
    if len(episodes) and "start" in episodes:
        episodes = episodes[pd.to_datetime(episodes["start"]) >= pd.Timestamp("2017-01-01")].reset_index(drop=True)

    lines = ["# 阶段 3.1 / 3.2 风险控制对照", ""]
    lines += [
        f"样本 {payload['sample']['start']} ~ {payload['sample']['end']}，{payload['sample']['holdout']}；"
        "持仓 50 只、等权、月末信号、次日开盘成交、gross 95%、单票 4%、行业 30%。",
        "",
        "## 1. 阶段 3.1：降换手 / 流动性滑点 / 波动控制",
        "",
        md_table(table_rows(rows, GROUP_31)),
        "",
        "### 1.1 流动性分档（按月、合格集内 5 等分，事前写死）",
        "",
        md_table(tiers.round(4).to_dict(orient="records")) if len(tiers) else "_（无分档数据）_",
        "",
        "## 2. 阶段 3.2：叠加清仓 / 梯度降仓",
        "",
        md_table(table_rows(rows, GROUP_32)),
        "",
        "## 3. 事前判据检查（MDD 改善 ≥3pp 且夏普不降）",
        "",
        md_table(judge_rows(rows)),
        "",
        "## 4. 危机触发明细（六分量 + 双确认 + 滞回，enter 0.75 / exit 0.30 / 确认 3 日）",
        "",
    ]
    if len(episodes):
        shown = episodes.copy()
        for column in ("start", "end"):
            if column in shown:
                shown[column] = pd.to_datetime(shown[column]).dt.strftime("%Y-%m-%d")
        lines += [md_table(shown.round(4).to_dict(orient="records")), ""]
    else:
        lines += ["_（无触发）_", ""]

    lines += ["## 5. 逐年收益", ""]
    yearly_rows: dict[str, dict] = {}
    for name in GROUP_31 + [item for item in GROUP_32 if item not in GROUP_31]:
        path = OUT / f"daily-{name}.csv"
        if not path.exists():
            continue
        series = yearly(pd.read_csv(path))
        for year, value in series.items():
            yearly_rows.setdefault(str(int(year)), {})[name] = pct(value)
    if yearly_rows:
        header = ["年份"] + [name for name in GROUP_31 + GROUP_32 if any(name in row for row in yearly_rows.values())]
        header = list(dict.fromkeys(header))
        lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for year in sorted(yearly_rows):
            lines.append("| " + " | ".join([year] + [yearly_rows[year].get(name, "—") for name in header[1:]]) + " |")
        lines.append("")

    lines += [
        "## 6. 口径",
        "",
        "- 单边换手 = 成交额 / 2 / 平均净值 / 年数（成交口径，含被最小成交额过滤掉的尝试之外的全部成交）。",
        "- 加权滑点 = 滑点成本合计 / 成交额合计。分档滑点按「ADV20 分位 + Amihud 分位」合成流动性分数，"
        "最好档 2bp、最差档 5bp，档位在每月末事先确定。",
        "- 费用/净值·年 = 累计费用 / 平均净值 / 年数。",
        "- 平均暴露 = 逐日 gross_exposure 均值；清仓方案的空仓天数见 risk.json 的 cash_days。",
        "- 真实 Engine（含佣金最低 5 元、印花税、过户费、整手、涨跌停、10% 参与率、退市 haircut30）；"
        "2025-01-01 之后的数据未读。",
    ]
    (OUT / REPORT).write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUT / REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
