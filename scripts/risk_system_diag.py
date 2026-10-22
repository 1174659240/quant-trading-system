"""极端风险清仓系统的**标定诊断**（只读、不交易）。

任务一第一步：报告 2008 / 2015 / 2018 / 2022 / 2024 五个事件段里 regime 六分量
（趋势、回撤、波动、广度、尾部、非流动性）的峰值与**因果滚动分位**。
本脚本不改任何规则，只产出数字，供随后的事前登记（``docs/极端风险清仓预注册.md``）引用。

用法：
    python -m scripts.risk_system_diag --sample main
    python -m scripts.risk_system_diag --sample history
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant import regime
from quant.factors.panel import build_panel
from scripts import growth_leftside as gl
from scripts.factor_market import FIELDS
from scripts.redlow_study import board_exclude, build_block, load_bars

OUT = Path("results/risk-system")

#: 事件段（标签, 起, 止）。两个样本各取自己覆盖的段。
WINDOWS = {
    "main": [
        ("2018-bear", "2018-01-01", "2018-12-31"),
        ("2022-bear", "2022-01-01", "2022-12-31"),
        ("2024-microcap", "2024-01-01", "2024-02-29"),
    ],
    "history": [
        ("2008-crash", "2008-01-01", "2008-12-31"),
        ("2015-crash", "2015-06-01", "2015-09-30"),
    ],
}


def load_panel(sample: str):
    if sample == "history":
        bars = gl.load_history_bars()
        start, end = gl.HIST_START, gl.HIST_END
    else:
        bars, _ = load_bars()
        start, end = pd.Timestamp("2016-01-01"), pd.Timestamp("2024-12-31")
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    panel = build_panel(bars, fields=FIELDS)
    print(f"[panel] {sample} {panel.close.shape} {panel.dates[0].date()} ~ {panel.dates[-1].date()}",
          flush=True)
    return panel, start, end


def component_frame(panel) -> pd.DataFrame:
    """六分量 + 因果滚动分位（全部只用当日及过去）。"""
    frame = regime.market_index(panel).copy()
    tail = frame["tail_share"]
    breadth = frame["breadth"]
    frame["tail_pct"] = tail.rolling(250, min_periods=120).rank(pct=True)
    frame["breadth_pct"] = breadth.rolling(250, min_periods=120).rank(pct=True)
    flags = regime.score_components(frame)
    score = regime.crisis_score(flags)          # 先用原始列名算分
    frame = frame.join(flags.add_prefix("flag_"))
    frame["score"] = score
    return frame


def window_stats(frame: pd.DataFrame, start: str, end: str) -> dict:
    part = frame.loc[start:end].dropna(how="all")
    if part.empty:
        return {"days": 0}

    def mx(col: str) -> float:
        return float(part[col].max()) if col in part else float("nan")

    def mn(col: str) -> float:
        return float(part[col].min()) if col in part else float("nan")

    row = {
        "days": int(len(part)),
        "dd_min": mn("dd250"),
        "vol_pct_max": mx("vol_pct"),
        "breadth_min": mn("breadth"),
        "breadth_pct_min": mn("breadth_pct"),
        "tail_max": mx("tail_share"),
        "tail_pct_max": mx("tail_pct"),
        "illiq_pct_max": mx("illiq_pct"),
        "score_max": mx("score"),
        "trend_share": float(part["flag_trend"].mean()) if "flag_trend" in part else float("nan"),
    }
    for name in regime.DEFAULT_WEIGHTS:
        row[f"flag_{name}"] = float(part[f"flag_{name}"].mean())
    return row


def render(sample: str, rows: list[dict]) -> str:
    head = ["段", "交易日", "回撤min", "vol分位max", "广度min", "广度分位min",
            "尾部max", "尾部分位max", "非流动分位max", "危机分max", "趋势触发占比"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        def f(value, pct: bool = False, nd: int = 3) -> str:
            if value is None or (isinstance(value, (int, float)) and not np.isfinite(value)):
                return "—"
            return f"{value:.1%}" if pct else f"{value:.{nd}f}"
        lines.append("| " + " | ".join([
            row["label"], str(row["days"]), f(row["dd_min"], pct=True),
            f(row["vol_pct_max"], nd=3), f(row["breadth_min"], pct=True),
            f(row["breadth_pct_min"], pct=True), f(row["tail_max"], pct=True),
            f(row["tail_pct_max"], pct=True), f(row["illiq_pct_max"], nd=3),
            f(row["score_max"], nd=3), f(row["trend_share"], pct=True),
        ]) + " |")
    flags = ["trend", "drawdown", "vol", "breadth", "tail", "illiquidity"]
    lines.append("")
    lines.append("各分量触发天数占比：" + "、".join(
        f"{name} {rows[0].get(f'flag_{name}', float('nan')):.0%}" for name in flags) if len(rows) == 1 else "")
    if len(rows) > 1:
        lines.append("")
        lines.append("| 段 | " + " | ".join(flags) + " |")
        lines.append("|" + "---|" * (len(flags) + 1))
        for row in rows:
            lines.append("| " + row["label"] + " | " + " | ".join(
                f"{row.get(f'flag_{name}', float('nan')):.0%}" for name in flags) + " |")
    lines.append("")
    lines.append(f"样本段：{sample}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="极端风险清仓系统标定诊断")
    parser.add_argument("--sample", choices=["main", "history"], default="main")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    panel, start, end = load_panel(args.sample)
    frame = component_frame(panel)
    frame.to_csv(args.out / f"components-{args.sample}.csv", encoding="utf-8-sig")

    rows = []
    for label, w_start, w_end in WINDOWS[args.sample]:
        part = frame.loc[w_start:w_end]
        if part.empty:
            continue
        row = {"label": label, "start": w_start, "end": w_end,
               **window_stats(frame, w_start, w_end)}
        rows.append(row)

    table = render(args.sample, rows)
    (args.out / f"diagnostic-{args.sample}.md").write_text(table, encoding="utf-8")
    (args.out / f"diagnostic-{args.sample}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(table)
    print(f"[done] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
