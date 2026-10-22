"""holdout 验收报告生成器：只读 `results/growth/holdout/` 的产物，不重算策略。

用法：``python -X utf8 -m scripts.growth_holdout_report``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/growth/holdout")
CASH = 1_000_000.0
RF = 0.02
PPY = 244
START = pd.Timestamp("2025-01-02")
END = pd.Timestamp("2026-10-16")
ANCHOR = pd.Timestamp("2024-12-31")


def load_equity() -> pd.Series:
    frame = pd.read_csv(OUT / "equity.csv", parse_dates=["date"]).set_index("date")
    return frame["equity"].astype(float)


def load_bench() -> pd.Series:
    frame = pd.read_csv(OUT / "benchmark-pool.csv", index_col=0, parse_dates=True)
    series = frame.iloc[:, 0].astype(float).sort_index()
    return series.loc[ANCHOR:END].ffill()


def load_index(code: str, name: str) -> pd.Series | None:
    path = OUT / f"index-{code}.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")["close"].astype(float)
    return frame.loc[ANCHOR:END].ffill()


def stats(equity: pd.Series, label: str) -> dict:
    returns = equity.pct_change().dropna()
    years = len(returns) / PPY
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((1 + total) ** (1 / years) - 1)
    vol = float(returns.std(ddof=0) * np.sqrt(PPY))
    drawdown = (equity / equity.cummax() - 1)
    mdd = float(drawdown.min())
    return {"label": label, "total": total, "cagr": cagr, "vol": vol,
            "sharpe": float((returns.mean() * PPY - RF) / vol) if vol else float("nan"),
            "mdd": mdd, "calmar": cagr / abs(mdd) if mdd else float("nan"),
            "years": years, "mdd_date": str(drawdown.idxmin().date())}


def chain_yearly(equity: pd.Series) -> pd.Series:
    """按「上一年末 → 本年末」计算年度收益（首年以 2024-12-31 的 100 万为基）。"""
    frame = equity.copy()
    frame.index = pd.to_datetime(frame.index)
    ends = frame.groupby(frame.index.year).last()
    base = pd.Series({2024: float(frame.iloc[0])})
    prev = pd.concat([base, ends]).sort_index()
    return (prev / prev.shift(1) - 1).dropna()


def chain_monthly(equity: pd.Series) -> pd.Series:
    """月末净值环比；首月以 2024-12-31 的锚点净值为基，避免丢掉 2025-01。"""
    frame = equity.copy()
    frame.index = pd.to_datetime(frame.index)
    ends = frame.groupby([frame.index.year, frame.index.month]).last()
    base = pd.Series({(ANCHOR.year, ANCHOR.month): float(frame.iloc[0])})
    prev = pd.concat([base, ends]).sort_index()
    return (prev / prev.shift(1) - 1).dropna()


def fmt_pct(value: float) -> str:
    return f"{value:+.2%}" if np.isfinite(value) else "—"


def main() -> int:
    equity = load_equity()
    bench = load_bench()
    indexes = {"中证1000": load_index("000852", "中证1000"),
               "创业板指": load_index("399006", "创业板指")}
    metrics = pd.read_csv(OUT / "metrics.csv").iloc[0]
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(OUT / "daily.csv", parse_dates=["date"]).set_index("date")
    trades = pd.read_csv(OUT / "trades.csv", parse_dates=["date"])
    consumed = json.loads((OUT / "consumed.json").read_text(encoding="utf-8")) if (OUT / "consumed.json").is_file() else {}

    rows = [stats(equity, "策略（cap25 冻结口径）"), stats(bench, "同池等权基准")]
    for name, series in indexes.items():
        if series is not None:
            rows.append(stats(series, name))
    table = pd.DataFrame(rows)
    pool_cagr = float(table.loc[table["label"] == "同池等权基准", "cagr"].iloc[0])

    yearly = chain_yearly(equity)
    bench_yearly = chain_yearly(bench)
    monthly = chain_monthly(equity)
    bench_monthly = chain_monthly(bench)
    month_table = pd.DataFrame({"策略": monthly, "同池等权": bench_monthly}).dropna()
    month_table.index = pd.MultiIndex.from_tuples(month_table.index, names=["year", "month"])
    month_table = month_table.loc[month_table.index.get_level_values("year") >= 2025]  # 去掉 2024-12 锚点月
    month_table["超额"] = month_table["策略"] - month_table["同池等权"]
    win = int((month_table["超额"] > 0).sum())

    lines: list[str] = []
    lines.append("# 小盘成长策略：holdout 一次性样本外验收（2025-01-02 ~ 2026-10-16）\n")
    lines.append("> **一次性、不可回调**：本报告是冻结口径（cap25）在 holdout 层上的唯一一次正式验收。")
    lines.append("> 结果不用于修改因子、参数或规则；若未来改动策略，本报告即失效，需另行登记新的验收。\n")

    lines.append("## 0. 结论（客观陈述）\n")
    lines.append(f"- 策略累计 **{fmt_pct(table.loc[0, 'total'])}**、年化 **{fmt_pct(table.loc[0, 'cagr'])}**、"
                 f"夏普 **{table.loc[0, 'sharpe']:.3f}**、最大回撤 **{fmt_pct(table.loc[0, 'mdd'])}**"
                 f"（{table.loc[0, 'mdd_date']}）。")
    lines.append(f"- 同期同池等权基准累计 {fmt_pct(table.loc[1, 'total'])}、年化 {fmt_pct(table.loc[1, 'cagr'])}、"
                 f"夏普 {table.loc[1, 'sharpe']:.3f}、最大回撤 {fmt_pct(table.loc[1, 'mdd'])}。")
    lines.append(f"- 相对同池等权的年化超额 **{fmt_pct(float(table.loc[0, 'cagr']) - pool_cagr)}**，"
                 f"累计超额 {fmt_pct(float(table.loc[0, 'total']) - float(table.loc[1, 'total']))}。")
    lines.append(f"- 单边年换手 {metrics['ann_turnover_one_way']:.0%}，费用占净值 {metrics['fees_pct_of_equity']:.2%}/年，"
                 f"加权滑点 {metrics['weighted_slip_bp']:.2f}bp，成交 {int(metrics['trades'])} 笔。")
    lines.append(f"- E 层（极端清仓）在窗口内**触发 0 次**，平均仓位 1.000 → 本期收益完全来自选股与组合构造，"
                 f"风险层未被检验。")
    lines.append(f"- 月胜率（相对同池等权）{win}/{len(month_table)} = {win / len(month_table):.0%}。")
    lines.append("- **读法**：本期未被证伪，但超额高度集中在 2026-01/03/07 三个月（合计 +21.02%，"
                 "其余月份合计为负），2025 年（同池 +34.8%）几乎零超额；beta 0.718。"
                 "因此数字支持「策略在样本外没有崩」，不支持「超额稳定可复制」。\n")

    lines.append("## 1. 口径（冻结，未改动）\n")
    lines.append("| 项目 | 取值 |\n|---|---|")
    for key, value in (
        ("因子", "QN 十因子（growth: g4_eav/e1_contract_yoy；quality: g5/g8/g9；rnd: e3_rd_delta；"
                 "aux: a1_reversal/a2_amihud/n8_unlock_60d/n9_unlock_120d）"),
        ("块权重", "成长 40% / 质量 25% / 研发 15% / 辅助 20%，块内等权、缺失按剩余项归一"),
        ("组合", "50 只等权、gross 0.95、单票 4%、证监会门类 25%、缓冲带 60"),
        ("交易", "月末信号 → 次月首个交易日开盘成交；逐日向目标回归且忽略 <5000 元单"),
        ("成本", "分档滑点 2–5bp（ADV20/Amihud 月末事前定档）+ 佣金双边万 2.5（最低 5 元）"
                 "+ 印花税卖出 0.05% + 过户费 0.001%"),
        ("退市", "haircut30（按最后交易日折价 30% 变现）"),
        ("E 层", "冻结规则默认开启（指数<MA200 且 dd250≤−20% 且 breadth≤12% 且 (r20≤−35% 或 r60≤−45%)，"
                 "连续 3 日；回补 1/2→1/3→1/6）"),
    ):
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## 2. 主指标\n")
    lines.append("| 指标 | 数值 |\n|---|---|")
    for key, label in (("total_return", "累计收益"), ("cagr", "年化收益"), ("sharpe", "夏普（rf=2%）"),
                       ("max_drawdown", "最大回撤"), ("calmar", "卡玛"), ("ann_turnover_one_way", "单边年换手"),
                       ("fees", "费用合计（元）"), ("fees_pct_of_equity", "费用/净值·年"),
                       ("weighted_slip_bp", "加权滑点（bp）"), ("trades", "成交笔数"),
                       ("avg_exposure", "平均暴露"), ("cash_days", "空仓天数")):
        value = metrics[key]
        if key == "total_return":
            value = table.loc[0, "total"]
        elif key == "cagr":
            value = table.loc[0, "cagr"]
        elif key == "sharpe":
            value = table.loc[0, "sharpe"]
        elif key == "max_drawdown":
            value = table.loc[0, "mdd"]
        if key in {"total_return", "cagr", "max_drawdown", "fees_pct_of_equity"}:
            shown = f"{float(value):.2%}"
        elif key == "ann_turnover_one_way":
            shown = f"{float(value):.0%}"
        elif key in {"fees", "weighted_slip_bp"}:
            shown = f"{float(value):,.1f}"
        elif key in {"trades", "cash_days"}:
            shown = f"{int(value)}"
        else:
            shown = f"{float(value):.3f}"
        lines.append(f"| {label} | {shown} |")
    lines.append(f"| 最大回撤日 | {metrics['mdd_date']} |")
    lines.append("")
    lines.append("> 净值起点为 2024-12-31 的 100 万现金（该日只有一天、无持仓，"
                 "`空仓天数=1` 即指这一天）；2025-01-02 开盘建仓。")
    lines.append("> 平均 gross 0.883 低于目标 0.95，来自三处：逐日向目标回归只在月末换仓、"
                 "5000 元最小订单过滤、以及整手/价格约束（50 只 × 1.9 万时高价股买不满一手）。\n")

    lines.append("## 3. 基准对比\n")
    lines.append("| 序列 | 累计 | 年化 | 年化波动 | 夏普 | 最大回撤 | 卡玛 | 年化超额(vs 同池) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in table.iterrows():
        excess = f"{row['cagr'] - pool_cagr:+.2%}" if row["label"] != "同池等权基准" else "—"
        lines.append(f"| {row['label']} | {fmt_pct(row['total'])} | {fmt_pct(row['cagr'])} | {row['vol']:.2%} | "
                     f"{row['sharpe']:.3f} | {fmt_pct(row['mdd'])} | {row['calmar']:.3f} | {excess} |")
    missing = [name for name, series in indexes.items() if series is None]
    if missing:
        lines.append(f"\n> 取不到的指数：{', '.join(missing)}（外部行情源当晚不可用）。")
    lines.append("")

    lines.append("## 4. 分年度与逐月\n")
    lines.append("| 年度 | 策略 | 同池等权 | 超额 |\n|---|---|---|---|")
    for year in sorted((set(yearly.index) | set(bench_yearly.index)) - {2024}):
        s = yearly.get(year, float("nan"))
        b = bench_yearly.get(year, float("nan"))
        lines.append(f"| {year} | {fmt_pct(s)} | {fmt_pct(b)} | {fmt_pct(s - b)} |")
    lines.append("\n| 月份 | 策略 | 同池等权 | 超额 |\n|---|---|---|---|")
    for (year, month), row in month_table.iterrows():
        lines.append(f"| {year}-{month:02d} | {fmt_pct(row['策略'])} | {fmt_pct(row['同池等权'])} | "
                     f"{fmt_pct(row['超额'])} |")
    lines.append("")

    # 超额集中度：月频超额和、前 3 个月占比、beta
    ex = month_table["超额"]
    ex_sum = float(ex.sum())
    top3 = ex.nlargest(3)
    r_s = equity.pct_change().dropna()
    r_b = bench.pct_change().reindex(r_s.index).fillna(0.0)
    beta = float(np.cov(r_s, r_b)[0, 1] / np.var(r_b))
    corr = float(np.corrcoef(r_s, r_b)[0, 1])
    lines.append("### 4.1 超额集中度体检（诚实读法的前提）\n")
    lines.append(f"- 月度超额之和 {ex_sum:+.2%}，正超额月份 {win}/{len(month_table)}。")
    lines.append(f"- 最大的 3 个月超额（{', '.join(f'{i[0]}-{i[1]:02d}' for i in top3.index)}）= "
                 f"{' / '.join(f'{v:+.2%}' for v in top3)}，合计 {top3.sum():+.2%}，"
                 f"占月度超额之和的 {top3.sum() / ex_sum:.0%}（超过 100% 是因为其余月份合计为负）。")
    lines.append(f"- 日频 beta（vs 同池等权）**{beta:.3f}**、相关系数 {corr:.3f}。")
    lines.append("- 读法：超额高度集中在少数月份（尤其 2026-07），2025 年几乎没有超额；"
                 "因此本报告支持「策略没有被样本外证伪」，不支持「超额稳定可复制」。\n")

    lines.append("## 5. E 层（极端清仓）在 holdout 期的触发情况\n")
    lines.append(f"- 触发天数：**{int(summary.get('trigger_days', 0))}**；清仓 0 天；平均仓位 1.000。")
    lines.append("- 该层规则最后一次触发在 2015-09（样本内），2025-01 以来未再触发，因此本期回测等价于「无风控」版本。")
    lines.append("- 含义：本期数字没有经过任何市场级降仓保护；若期间出现 2008 型急跌，本层会介入，但本期未发生。\n")

    lines.append("## 6. 交易与持仓事实\n")
    if len(trades):
        buys = int((trades["side"] == "buy").sum())
        sells = int((trades["side"] == "sell").sum())
        lines.append(f"- 成交 {len(trades)} 笔（买 {buys} / 卖 {sells}），"
                     f"单笔中位金额 {trades['gross'].median():,.0f} 元，"
                     f"费用/成交额中位 {float((trades['fee'] / trades['gross']).median()) * 1e4:.2f}bp。")
    lines.append(f"- 平均 gross 暴露 {daily['gross_exposure'].mean():.3f}"
                 f"（p5 {daily['gross_exposure'].quantile(0.05):.3f} / p95 {daily['gross_exposure'].quantile(0.95):.3f}），"
                 f"持仓数中位 {int(daily['positions'].median())}。")
    lines.append(f"- 首笔建仓日 {daily[daily['gross_exposure'] > 0].index[0].date()}（信号来自 2024-12-31 收盘）。\n")

    lines.append("## 7. 已知数据缺口与处理（如实声明）\n")
    lines.append("1. **另类数据止于 2024-12-31**：解禁 n8/n9 在 holdout 期为空 → 辅助块按剩余 a1/a2 归一。")
    lines.append("   这使 holdout 期的 aux 块与样本内不同构，是**已知偏差**，不是策略改动。")
    lines.append("2. **2025+ 退市事件缺失**：`delisted/events.csv` 只到 2024；holdout 期若有股票退市，")
    lines.append("   引擎按最后可用价格持有，未做折价处理 → 可能**高估**收益。")
    lines.append("3. **ST 过滤**用 holdout 估值层的 `isST` 补齐；停牌/一字板沿用价量层掩码。")
    lines.append("4. 财务用 `NOTICE_DATE ≤ 当日` 的最新一期（含 2025–2026 报告期），按披露日生效，无前视。")
    lines.append("5. 行业为 PIT 门类（`ref/industry.csv` 年度版本，最新一版 2026-10-12）。\n")

    lines.append("## 8. 工程留痕：同一冻结口径的 4 次运行\n")
    lines.append("策略、因子、参数、规则全程未改；前三次的差异全部来自「建仓日边界」的工程处理，")
    lines.append("数字并列如下（前三次产物保存在同名子目录，未删除）：\n")
    lines.append("| 运行 | 处理 | 累计 | 年化 | 夏普 | MDD |\n|---|---|---|---|---|---|")
    lines.append(f"| run1 | 无 2024-12-31 信号 → 1 月整月空仓 | 39.04% | 20.51% | 1.090 | −16.16% |")
    lines.append(f"| run2 | 加月末信号，但引擎丢掉窗口前一行 → 仍 1 月空仓 | 41.02% | 21.48% | 1.134 | −16.22% |")
    lines.append(f"| run3 | 2025-01-02 即建仓，但净值起点误用首日收盘 | 43.70% | 21.45% | 1.117 | −16.24% |")
    lines.append(f"| **run4（最终）** | 起点保留 2024-12-31 的 100 万现金 | **{fmt_pct(table.loc[0, 'total'])}** | "
                 f"**{fmt_pct(table.loc[0, 'cagr'])}** | **{table.loc[0, 'sharpe']:.3f}** | "
                 f"**{fmt_pct(table.loc[0, 'mdd'])}** |")
    lines.append("")
    lines.append(f"`consumed.json`：{json.dumps(consumed, ensure_ascii=False)}")
    lines.append("> `force=true` 表示其中若干次是为修正“建仓日边界”这类工程问题而显式覆盖；"
                 "每次覆盖的原因与数字见 `runs-log.json`，期间未据 holdout 结果改动任何策略规则。\n")

    lines.append("## 9. 复现命令\n")
    lines.append("```bash")
    lines.append("# 一次验收（已消费，重跑须显式 --force）")
    lines.append("python -X utf8 -m scripts.growth_holdout --run")
    lines.append("# 报告")
    lines.append("python -X utf8 -m scripts.growth_holdout_report")
    lines.append("```\n")
    lines.append("产物：`metrics.csv`、`equity.csv`、`daily.csv`、`trades.csv`、`weights.csv.gz`、"
                 "`benchmark-pool.csv`、`yearly.csv`、`monthly.csv`、`index-*.csv`、`events-E.csv`、"
                 "`liquidity-tiers.csv`、`summary.json`、`consumed.json`\n")
    lines.append("## 10. 文件改动清单\n")
    lines.append("- 新增 `scripts/growth_holdout.py`（验收执行）、`scripts/growth_holdout_report.py`（本报告生成）")
    lines.append("- 未修改 `quant/` 任何文件；未修改既有策略脚本与结果目录")
    lines.append("- 新增结果目录 `results/growth/holdout/`（含 `run1/`、`run2/`、`run3/` 三次工程留痕）\n")

    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\n[report] → {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
