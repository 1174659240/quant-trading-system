"""回测报表：Markdown + CSV + 自绘 SVG 图（不依赖 matplotlib）。

产物目录::

    results/<name>/
        equity.csv    净值曲线（含基准）
        daily.csv     日度账户快照（现金/市值/仓位/换手/费用）
        trades.csv    成交流水
        rejections.csv 未成交原因（T+1/涨跌停/现金不足…）
        metrics.json  全部绩效指标
        report.md     人读版报告
        equity.svg    净值 + 回撤双面板图
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ SVG
def _points(values: np.ndarray, width: float, height: float, vmin: float, vmax: float) -> str:
    n = len(values)
    if n == 0:
        return ""
    span = max(vmax - vmin, 1e-9)
    step = width / max(n - 1, 1)
    coords = []
    for i, value in enumerate(values):
        x = i * step
        y = height - (value - vmin) / span * height
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def render_svg(equity: pd.Series, benchmark: pd.Series | None = None, *, width: int = 900, height: int = 480) -> str:
    pad_l, pad_r, pad_t, pad_b = 64, 20, 28, 34
    panel_h = (height - pad_t - pad_b - 60) / 2
    plot_w = width - pad_l - pad_r

    eq = equity.astype(float)
    if len(eq) and float(eq.iloc[0]):
        eq = eq / float(eq.iloc[0])  # 归一化到 1.0，多策略/多区间才好比
    dd = (eq / eq.cummax() - 1.0) * 100.0
    bench = None
    if benchmark is not None and len(benchmark):
        bench = benchmark.reindex(eq.index).ffill().astype(float)
        if len(bench) and float(bench.iloc[0]):
            bench = bench / float(bench.iloc[0])

    vmin = float(min(eq.min(), bench.min() if bench is not None else eq.min()))
    vmax = float(max(eq.max(), bench.max() if bench is not None else eq.max()))
    eq_pts = _points(eq.to_numpy(), plot_w, panel_h, vmin, vmax)
    bench_pts = _points(bench.to_numpy(), plot_w, panel_h, vmin, vmax) if bench is not None else ""
    dd_min = float(dd.min()) if len(dd) else 0.0
    dd_pts = _points(dd.to_numpy(), plot_w, panel_h, dd_min, 0.0)

    def grid(y0: float, h: float, vmin_: float, vmax_: float, fmt: str) -> str:
        out = []
        for i in range(5):
            y = y0 + h * i / 4
            value = vmax_ - (vmax_ - vmin_) * i / 4
            out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="#e6e8eb" stroke-width="1"/>')
            out.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" fill="#6b7280" text-anchor="end">{fmt.format(value)}</text>')
        return "".join(out)

    def xlabels(y: float) -> str:
        out = []
        if len(eq) == 0:
            return ""
        n = len(eq)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            i = min(int(frac * (n - 1)), n - 1)
            x = pad_l + plot_w * frac
            out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="11" fill="#6b7280" text-anchor="middle">{eq.index[i].date()}</text>')
        return "".join(out)

    y_eq = pad_t
    y_dd = pad_t + panel_h + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Segoe UI, Microsoft YaHei, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad_l}" y="18" font-size="14" fill="#111827">净值曲线（初始 = 1.0）</text>',
        f'<g transform="translate(0,{y_eq})">',
        grid(0, panel_h, vmin, vmax, "{:.2f}"),
        "</g>",
        f'<g transform="translate({pad_l},{y_eq})">',
    ]
    parts.append(f'<polyline points="{eq_pts}" fill="none" stroke="#2563eb" stroke-width="2"/>')
    if bench_pts:
        parts.append(f'<polyline points="{bench_pts}" fill="none" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="5,4"/>')
    parts.append("</g>")

    parts.append(f'<g transform="translate(0,{y_dd})">')
    parts.append(f'<text x="{pad_l}" y="-8" font-size="13" fill="#111827">回撤（%）</text>')
    parts.append(grid(0, panel_h, dd_min, 0.0, "{:.1f}%"))
    if dd_pts:
        area = f"{pad_l},0 {pad_l + plot_w},0 " + " ".join(
            f"{pad_l + float(p.split(',')[0]):.2f},{float(p.split(',')[1]):.2f}" for p in dd_pts.split()
        )
        parts.append(f'<polygon points="{area}" fill="#fca5a5" fill-opacity="0.45" stroke="none"/>')
        shifted = " ".join(f"{pad_l + float(p.split(',')[0]):.2f},{float(p.split(',')[1]):.2f}" for p in dd_pts.split())
        parts.append(f'<polyline points="{shifted}" fill="none" stroke="#dc2626" stroke-width="1.5"/>')
    parts.append("</g>")
    parts.append(xlabels(height - 12))
    parts.append("</svg>")
    return "\n".join(parts)


PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]


def render_multi_svg(curves: pd.DataFrame, *, title: str = "多策略净值对比", width: int = 900, height: int = 420) -> str:
    """把多条净值曲线画进一张图（归一化到 1.0），带图例。"""
    pad_l, pad_r, pad_t, pad_b = 64, 20, 46, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    data = curves.astype(float).ffill().dropna(how="all")
    if data.empty:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'
    norm = data.div(data.iloc[0].replace(0.0, np.nan)).dropna(axis=1, how="all")
    vmin, vmax = float(norm.min().min()), float(norm.max().max())

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Segoe UI, Microsoft YaHei, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad_l}" y="20" font-size="14" fill="#111827">{title}（初始 = 1.0）</text>',
    ]
    for i in range(5):
        y = pad_t + plot_h * i / 4
        value = vmax - (vmax - vmin) * i / 4
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="#e6e8eb" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" fill="#6b7280" text-anchor="end">{value:.2f}</text>')
    parts.append(f'<g transform="translate({pad_l},{pad_t})">')
    for i, column in enumerate(norm.columns):
        color = PALETTE[i % len(PALETTE)]
        parts.append(f'<polyline points="{_points(norm[column].to_numpy(), plot_w, plot_h, vmin, vmax)}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append("</g>")
    for i, column in enumerate(norm.columns):
        color = PALETTE[i % len(PALETTE)]
        lx = pad_l + i * 150
        ly = pad_t - 12
        parts.append(f'<rect x="{lx}" y="{ly - 8}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx + 15}" y="{ly}" font-size="11" fill="#374151">{column}</text>')
    n = len(norm)
    if n:
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            i = min(int(frac * (n - 1)), n - 1)
            parts.append(f'<text x="{pad_l + plot_w * frac:.1f}" y="{height - 12}" font-size="11" fill="#6b7280" text-anchor="middle">{norm.index[i].date()}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------------------------ Markdown 表格（不依赖 tabulate）
def _fmt_cell(value, floatfmt: str) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return ""
        return format(float(value), floatfmt)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, np.bool_):
        return "是" if bool(value) else "否"
    return str(value)


def to_markdown_table(df: pd.DataFrame, *, floatfmt: str = ".4f", index: bool = True) -> str:
    """DataFrame → Markdown 表格。

    不用 ``DataFrame.to_markdown``，因为它依赖 pandas 的可选包 ``tabulate``；
    本项目只依赖 pandas/numpy。
    """
    header = ([str(df.index.name) if df.index.name else ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for idx, row in df.iterrows():
        cells = ([_fmt_cell(idx, floatfmt)] if index else []) + [_fmt_cell(v, floatfmt) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ 报告
def write_report(result, out_dir: str | Path, name: str = "backtest") -> Path:
    """把回测结果落盘，返回报告目录。``result`` 是 ``BacktestResult``。"""
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    equity = result.equity.rename("equity")
    frame = equity.to_frame()
    if result.benchmark is not None:
        frame["benchmark"] = result.benchmark.reindex(equity.index).ffill()
    frame["drawdown"] = (equity / equity.cummax() - 1.0)
    frame.to_csv(out / "equity.csv", encoding="utf-8-sig")

    if len(result.daily):
        result.daily.to_csv(out / "daily.csv", encoding="utf-8-sig")
    if len(result.trades):
        result.trades.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    if len(result.rejections):
        result.rejections.to_csv(out / "rejections.csv", index=False, encoding="utf-8-sig")
    if len(result.targets):
        result.targets.to_csv(out / "targets.csv", encoding="utf-8-sig")

    metrics = {k: (None if v is None else (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v))
               for k, v in result.metrics.items()}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "equity.svg").write_text(render_svg(equity, result.benchmark), encoding="utf-8")

    md = [
        f"# 回测报告 · {result.strategy}",
        "",
        f"- 区间：{result.metrics.get('start')} ~ {result.metrics.get('end')}（{result.metrics.get('days')} 个交易日）",
        f"- 初始资金：{result.metrics.get('initial_equity', 0):,.0f} → 期末：{result.metrics.get('final_equity', 0):,.0f}",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
    ]
    rows = [
        ("累计收益", result.metrics.get("total_return"), "{:.2%}"),
        ("年化收益", result.metrics.get("cagr"), "{:.2%}"),
        ("年化波动", result.metrics.get("ann_vol"), "{:.2%}"),
        ("夏普比率", result.metrics.get("sharpe"), "{:.3f}"),
        ("索提诺比率", result.metrics.get("sortino"), "{:.3f}"),
        ("卡玛比率", result.metrics.get("calmar"), "{:.3f}"),
        ("最大回撤", result.metrics.get("max_drawdown"), "{:.2%}"),
        ("最长回撤(交易日)", result.metrics.get("mdd_days"), "{:.0f}"),
        ("日胜率", result.metrics.get("win_rate"), "{:.2%}"),
        ("盈亏比", result.metrics.get("profit_factor"), "{:.3f}"),
        ("日VaR(95%)", result.metrics.get("var95"), "{:.2%}"),
        ("日CVaR(95%)", result.metrics.get("cvar95"), "{:.2%}"),
        ("日均换手", result.metrics.get("avg_turnover"), "{:.2%}"),
        ("日均仓位", result.metrics.get("avg_exposure"), "{:.2%}"),
        ("成交笔数", result.metrics.get("trade_count"), "{:.0f}"),
        ("累计费用", result.metrics.get("total_fees"), "{:,.0f}"),
        ("Beta", result.metrics.get("beta"), "{:.3f}"),
        ("年化Alpha", result.metrics.get("alpha"), "{:.2%}"),
        ("信息比率", result.metrics.get("info_ratio"), "{:.3f}"),
    ]
    for label, value, fmt in rows:
        if value is None:
            continue
        md.append(f"| {label} | {fmt.format(value)} |")

    if not result.report.yearly.empty:
        md += ["", "## 年度收益", "", "| 年份 | 收益 |", "| --- | --- |"]
        for year, ret in result.report.yearly.items():
            md.append(f"| {year} | {ret:+.2%} |")
    if not result.report.monthly.empty:
        md += ["", "## 月度收益", "", to_markdown_table(result.report.monthly, floatfmt="+.2%")]
    if len(result.trades):
        md += ["", "## 最近成交（末 20 笔）", "", to_markdown_table(result.trades.tail(20), index=False, floatfmt=".3f")]
    if len(result.rejections):
        counts = result.rejections["reason"].value_counts().head(10)
        md += ["", "## 未成交原因 Top 10", "", "| 原因 | 次数 |", "| --- | --- |"]
        for reason, count in counts.items():
            md.append(f"| {reason} | {count} |")

    md += ["", "![净值曲线](equity.svg)", ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    return out


def compare_report(results: dict[str, object], out_dir: str | Path, name: str = "compare") -> Path:
    """多策略对比：一张表 + 一张叠加图。"""
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = {}
    for label, result in results.items():
        m = result.metrics
        curves[label] = result.equity / result.equity.iloc[0]
        rows.append({
            "策略": label,
            "累计收益": m.get("total_return", 0.0),
            "年化收益": m.get("cagr", 0.0),
            "年化波动": m.get("ann_vol", 0.0),
            "夏普": m.get("sharpe", 0.0),
            "最大回撤": m.get("max_drawdown", 0.0),
            "卡玛": m.get("calmar", 0.0),
            "日胜率": m.get("win_rate", 0.0),
            "日均换手": m.get("avg_turnover", 0.0),
            "成交笔数": m.get("trade_count", 0),
        })
    table = pd.DataFrame(rows).set_index("策略")
    table.to_csv(out / "compare.csv", encoding="utf-8-sig")

    combined = pd.DataFrame(curves)
    reference = next(iter(results.values()))
    if getattr(reference, "benchmark", None) is not None:
        combined["基准"] = reference.benchmark.reindex(combined.index).ffill()
    (out / "compare.svg").write_text(render_multi_svg(combined), encoding="utf-8")
    (out / "compare.md").write_text(
        "# 策略对比\n\n" + to_markdown_table(table, floatfmt=".4f") + "\n\n![对比曲线](compare.svg)\n",
        encoding="utf-8",
    )
    combined.to_csv(out / "curves.csv", encoding="utf-8-sig")
    return out
