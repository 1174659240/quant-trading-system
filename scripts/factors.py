"""因子研究报告：算 IC、分层收益、因子相关性，落 CSV + Markdown。

    python -m quant factor --source csv --csv-dir data/csv
    python -m quant factor --factors momentum,low_vol --horizon 5 --quantiles 5
    python -m quant factor --source synthetic --periods 600 --out results/factor-synth

这一步回答的是「这个因子在我的票池里到底有没有信息」，而不是「它能赚多少钱」。
IC 均值、IC IR、IC>0 占比、分层单调性，缺一样都别急着上策略。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.data import CsvSource, DataStore, SyntheticSource  # noqa: E402
from quant.factors import (  # noqa: E402
    build_panel,
    compute,
    evaluate_factor,
    evaluate_text,
    factor_corr,
    names as factor_names,
)
from quant.report import to_markdown_table  # noqa: E402

SPREAD_ROW = "多空(Q最高-Q最低)"


def build_source(args):
    if args.source == "synthetic":
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
        return SyntheticSource(symbols, start=args.start or "2020-01-01", periods=args.periods)
    return CsvSource(args.csv_dir)


def _fmt_bucket_table(table: pd.DataFrame) -> str:
    """分层汇总表：收益/胜率乘 100 显示百分比，`days` 保持整数。"""
    display = table.copy()
    for column in display.columns:
        if column == "days":
            continue
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value) * 100:+.3f}%"
        )
    if "days" in display.columns:
        display["days"] = display["days"].astype("Int64")
    return display.to_string()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m quant factor",
        description="因子研究报告：IC / 分层 / 相关性",
    )
    parser.add_argument("--source", default="csv", choices=["csv", "synthetic", "web"])
    parser.add_argument("--csv-dir", default="data/csv")
    parser.add_argument("--symbols", default="", help="逗号分隔；空＝数据源里的全部标的")
    parser.add_argument("--start", default=None)
    parser.add_argument("--periods", type=int, default=600, help="synthetic 模式的行数")
    parser.add_argument("--factors", default="momentum,low_vol,reversal")
    parser.add_argument("--horizon", type=int, default=5, help="IC 衰减报告的观察期")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--out", default=None, help="报告目录，默认 results/factor-report")
    args = parser.parse_args(argv)

    wanted = [item.strip() for item in args.factors.split(",") if item.strip()]
    unknown = [name for name in wanted if name not in factor_names()]
    if unknown:
        print(f"没有这些因子：{unknown}\n可选：{factor_names()}")
        return 1

    store = DataStore(build_source(args))
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
    bars = store.get_many(symbols)
    if not bars:
        print("没有取到任何行情数据")
        return 1
    panel = build_panel(bars)
    print(
        f"标的 {len(panel.symbols)} 只 · 交易日 {len(panel.dates)} 天 · "
        f"{panel.dates[0].date()} ~ {panel.dates[-1].date()}"
    )

    rows: list[dict] = []
    frames: dict[str, pd.DataFrame] = {}
    for name in wanted:
        frame = compute(name, panel)
        result = evaluate_factor(
            frame,
            panel,
            name=name,
            horizons=(1, args.horizon),
            quantiles=args.quantiles,
        )
        frames[name] = frame
        summary = result["ic"]
        decay = result["decay"]
        qsum = result["quantile_summary"]
        spread = float(qsum.loc[SPREAD_ROW, "mean"]) if SPREAD_ROW in qsum.index else float("nan")
        rows.append(
            {
                "factor": name,
                "coverage": round(float(result["coverage"]), 1),
                "ic_mean": round(float(summary["ic_mean"]), 4),
                "ic_std": round(float(summary["ic_std"]), 4),
                "ic_ir": round(float(summary["ic_ir"]), 3),
                "t_stat": round(float(summary["t_stat"]), 2),
                "positive_rate": round(float(summary["positive_rate"]), 3),
                f"ic_{args.horizon}d": round(float(decay.get(args.horizon, float("nan"))), 4),
                f"q{args.quantiles}-q1": spread,
            }
        )
        print("\n" + "-" * 64)
        print(evaluate_text(result))
        print("分层（每期收益，按因子值从低到高）：")
        print(_fmt_bucket_table(qsum))

    table = pd.DataFrame(rows)
    if table.empty:
        print("没有可报告的因子")
        return 1

    corr = factor_corr(frames)
    print("\n" + "=" * 64)
    print("因子摘要：")
    print(table.to_string(index=False))
    print("\n因子秩相关：")
    print(corr.to_string(float_format=lambda v: f"{v:+.3f}"))

    out_dir = Path(args.out or (ROOT / "results" / "factor-report"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "factor-summary.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(out_dir / "factor-corr.csv", encoding="utf-8-sig")

    lines = ["# 因子研究报告", ""]
    lines.append(
        f"- 标的：{len(panel.symbols)} 只；交易日：{len(panel.dates)} 天"
        f"（{panel.dates[0].date()} ~ {panel.dates[-1].date()}）"
    )
    lines.append(f"- IC 观察期：{args.horizon} 个交易日；分层：{args.quantiles} 组")
    lines.append("- 分层收益是「每期按因子值分组的当期收益」，多空行＝最高组减最低组")
    lines.append("")
    lines.append("## 因子摘要")
    lines.append("")
    lines.append(to_markdown_table(table, floatfmt=".4f", index=False))
    lines.append("")
    lines.append("## 因子相关性（秩相关）")
    lines.append("")
    lines.append(to_markdown_table(corr, floatfmt="+.3f"))
    lines.append("")
    for name in wanted:
        if name not in frames:
            continue
        result = evaluate_factor(
            frames[name],
            panel,
            name=name,
            horizons=(1, args.horizon),
            quantiles=args.quantiles,
        )
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```")
        lines.append(evaluate_text(result))
        lines.append("```")
        lines.append("")
    lines.append("> 经验门槛：IC 均值 > 0.02 且 IC IR > 0.3 才算有点信息；分层收益要单调。")
    lines.append("> 票池只有几十只时 IC 噪声很大，一次结果不能当结论。")
    (out_dir / "factor-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
