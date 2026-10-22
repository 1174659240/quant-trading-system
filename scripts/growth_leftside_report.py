"""把两个样本（主样本 2017–2024 / 历史样本 2008–2016）的 A/B/C 对照写成一张表。

只读 ``results/growth/leftside*`` 下已经跑完的产物，不重跑回测、不改任何数字。

用法：
    python -m scripts.growth_leftside_report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MAIN = Path("results/growth/leftside")
HIST = Path("results/growth/leftside-history")
OUT = Path("results/growth/leftside-comparison.md")

LABELS = {
    "A": "A 无风控基线（legacy5k_slip）",
    "B": "B 梯度降仓（gradient5k，危机期降到 0.5、解除后线性回补）",
    "C": "C 梯度 + 左侧分批回场（DD 上限 −20%，主规格）",
    "C15": "C15 同上、DD 上限 −15%",
    "C25": "C25 同上、DD 上限 −25%",
}
ORDER = ["A", "B", "C", "C15", "C25"]
COLS = [
    ("total_return", "累计", "pct"),
    ("cagr", "年化", "pct"),
    ("sharpe", "夏普", "num"),
    ("max_drawdown", "最大回撤", "pct"),
    ("calmar", "卡玛", "num"),
    ("ann_turnover_one_way", "单边年换手", "pct"),
    ("avg_exposure", "平均暴露", "num"),
    ("min_exposure", "最低暴露", "num"),
    ("cash_days", "空仓天数", "int"),
    ("fees_pct_of_equity", "费用/净值·年", "pct"),
    ("excess_total", "超额（几何）", "pct"),
    ("weighted_slip_bp", "加权滑点 bp", "num"),
]


def fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if kind == "pct":
        return f"{float(value) * 100:.2f}%"
    if kind == "int":
        return f"{int(value)}"
    return f"{float(value):.3f}"


def load(folder: Path) -> tuple[pd.DataFrame, dict]:
    path = folder / "metrics.csv"
    if not path.exists():
        return pd.DataFrame(), {}
    table = pd.read_csv(path, dtype={"spec": str})
    meta: dict = {}
    payload = folder / "leftside.json"
    if payload.exists():
        meta = json.loads(payload.read_text(encoding="utf-8"))
    return table, meta


def table_block(table: pd.DataFrame, title: str) -> list[str]:
    if table.empty:
        return [f"### {title}", "", "_该样本尚未跑完，无数据。_", ""]
    rows = ["| 规格 | " + " | ".join(label for _, label, _ in COLS) + " |"]
    rows.append("|---" * (len(COLS) + 1) + "|")
    index = {spec: row for spec, row in table.set_index("spec").iterrows()}
    for spec in ORDER:
        if spec not in index:
            continue
        row = index[spec]
        cells = [fmt(row.get(key), kind) for key, _, kind in COLS]
        rows.append(f"| {LABELS.get(spec, spec)} | " + " | ".join(cells) + " |")
    return [f"### {title}", "", *rows, ""]


def yearly_block(folder: Path, title: str) -> list[str]:
    if not folder.exists():
        return []
    frames = {}
    for spec in ORDER:
        path = folder / f"equity-{spec}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        column = "equity" if "equity" in frame.columns else frame.columns[0]
        equity = frame[column].astype(float)
        daily = equity.pct_change().fillna(0.0)
        frames[spec] = (1.0 + daily).groupby(daily.index.year).prod() - 1.0
    if not frames:
        return []
    bench_path = folder / "benchmark-equity.csv"
    series = pd.DataFrame(frames).sort_index()
    if bench_path.exists():
        bench = pd.read_csv(bench_path, index_col=0, parse_dates=True)
        column = "equity" if "equity" in bench.columns else bench.columns[0]
        bench_daily = bench[column].astype(float).pct_change().fillna(0.0)
        series["基准"] = (1.0 + bench_daily).groupby(bench_daily.index.year).prod() - 1.0
    series.index = [str(index) for index in series.index]
    series = series.dropna(how="all")
    if series.empty:
        return []
    header = "| 年份 | " + " | ".join(ORDER) + " | 基准 |"
    rows = ["### " + title, "", header, "|---" * (len(ORDER) + 2) + "|"]
    for year, row in series.iterrows():
        cells = [fmt(row.get(spec), "pct") for spec in ORDER]
        rows.append(f"| {year} | " + " | ".join(cells) + f" | {fmt(row.get('基准'), 'pct')} |")
    return [*rows, ""]


def episode_block(folder: Path, title: str) -> list[str]:
    path = folder / "crisis-episodes.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path)
    if table.empty:
        return []
    payload = folder / "leftside.json"
    if payload.exists():
        window = json.loads(payload.read_text(encoding="utf-8")).get("sample", {})
        if window.get("start") and window.get("end"):
            start = pd.Timestamp(window["start"])
            end = pd.Timestamp(window["end"])
            stamp = pd.to_datetime(table["start"], errors="coerce")
            table = table[(stamp >= start) & (stamp <= end)]
            if table.empty:
                return [f"### {title}", "", "_样本区间内没有危机段。_", ""]
    rows = ["### " + title, "",
            "| # | 开始 | 结束 | 天数 | 最高危机分 | 峰谷回撤 |", "|---" * 6 + "|"]
    for position, row in table.reset_index(drop=True).iterrows():
        rows.append(f"| {position + 1} | {str(row.get('start'))[:10]} | {str(row.get('end'))[:10]} | "
                    f"{int(row.get('days', 0))} | {float(row.get('max_score', 0)):.3f} | "
                    f"{fmt(row.get('peak_dd'), 'pct')} |")
    return [*rows, ""]


def build(main_dir: Path | None = None, hist_dir: Path | None = None,
          out_path: Path | None = None) -> Path:
    main_dir = Path(main_dir) if main_dir is not None else MAIN
    hist_dir = Path(hist_dir) if hist_dir is not None else HIST
    out_path = Path(out_path) if out_path is not None else OUT
    main_table, main_meta = load(main_dir)
    hist_table, hist_meta = load(hist_dir)

    lines = ["# 左侧分批回场：主样本 vs 历史样本对照", ""]
    lines += [
        "> 由 `scripts/growth_leftside_report.py` 生成，只读已跑完的产物。",
        "> 主样本 2017-01~2024-12（成长因子可用、行业上限 30%）；",
        "> 历史样本 2008-01~2016-12（财务补采到 2006 年，**无 PIT 行业分类 → 行业上限关闭**、"
        "ST 标记缺失 → 排雷缺少 ST 维度）。",
        "> 2025-01-01 起的 holdout 全程未读。",
        "",
        "## 一、主样本（2017–2024）",
        "",
    ]
    lines += table_block(main_table, "A / B / C 主表")
    lines += yearly_block(main_dir, "逐年收益（主样本）")
    lines += episode_block(main_dir, "危机段（主样本）")

    lines += ["## 二、历史样本（2008–2016）", ""]
    if hist_table.empty:
        lines += ["_历史样本尚未跑完。_", ""]
    else:
        lines += table_block(hist_table, "A / B / C 主表")
        lines += yearly_block(hist_dir, "逐年收益（历史样本）")
        lines += episode_block(hist_dir, "危机段（历史样本）")

    lines += [
        "## 三、口径差异（必须一起读）",
        "",
        "| 项目 | 主样本 | 历史样本 |",
        "|---|---|---|",
        "| 区间 | 2017-01~2024-12（96 个月末） | 2008-01~2016-12 |",
        "| 因子 | QN 合成（成长+质量+研发+辅助，含解禁 n8/n9） | 同公式；2006–2012 财务来自东财补采层 |",
        "| 行业上限 | 30%（证监会门类，年度 PIT） | **关闭**（2008–2013 无 PIT 行业分类） |",
        "| ST 排雷 | 有（valuation.isST） | **无**（2014 前无 ST 标记） |",
        "| 退市处置 | 事件表 haircut30 | 以最后一个交易日当退市日 + haircut30 |",
        "| 基准 | 同池等权（解析式） | 同池等权（解析式） |",
        "",
        "**判决口径（预注册）**：C 相对 A —— 历史样本 MDD 改善 ≥3pp 且夏普不降；"
        "主样本结论已在 `results/growth/leftside/leftside.json` 冻结。",
        "",
    ]

    if main_meta or hist_meta:
        lines += ["## 四、运行元数据", "", "```json",
                  json.dumps({"main": main_meta.get("sample", {}), "history": hist_meta.get("sample", {})},
                             ensure_ascii=False, indent=2), "```", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=MAIN)
    parser.add_argument("--hist", type=Path, default=HIST)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    path = build(args.main, args.hist, args.out)
    print(f"[report] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
