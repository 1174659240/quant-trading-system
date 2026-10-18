"""方法论对照组：把「换了什么」和「差多少」放在同一张表里。

五组配置共用同一批 bars（含退市股、haircut30、剔 ST），只改一个变量：

    A 等权合成 + 不中性化 + 全池取头部     ← 起点
    B 滚动 IC 加权 + 不中性化 + 全池取头部
    C 滚动 IC 加权 + 行业/市值中性化 + 全池取头部
    D 同 C，但只买「分数最低的 20%」       ← 反向对照：若信号为真，D 应显著更差
    E 同 C，但只买「中间的 20%」           ← 分组有效性对照：中间组不该有超额

用法::

    python -m scripts.methodology_ablation
    python -m scripts.methodology_ablation --start 2016-01-04 --end 2024-12-31 --top-k 30

产出 ``results/methodology-ablation/{ablation.csv,ablation.md,configs.json}``。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.backtest.engine import Engine  # noqa: E402
from quant.config import BacktestConfig  # noqa: E402
from quant.factors import build_panel, factor_corr, names as factor_names  # noqa: E402
from quant.factors.exposures import industry_labels, log_size  # noqa: E402
from quant.strategy.multifactor import MultiFactorStrategy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factor_market import (  # noqa: E402
    FIELDS,
    LAKE,
    dedup_factors,
    load_delisted_bars,
    load_delisting_events,
    load_layer,
    run_factor_study,
)

CONFIGS = [
    {"code": "A", "label": "等权 / 不中性化 / 头部", "weighting": "equal", "neutralize": None, "bucket": None},
    {"code": "B", "label": "滚动 IC / 不中性化 / 头部", "weighting": "ic", "neutralize": None, "bucket": None},
    {"code": "C", "label": "滚动 IC / 行业+市值中性化 / 头部", "weighting": "ic", "neutralize": "industry_size", "bucket": None},
    {"code": "D", "label": "滚动 IC / 中性化 / 最低 20%", "weighting": "ic", "neutralize": "industry_size", "bucket": 1},
    {"code": "E", "label": "滚动 IC / 中性化 / 中间 20%", "weighting": "ic", "neutralize": "industry_size", "bucket": 3},
]

METRIC_KEYS = [
    ("total_return", "累计收益", "pct"),
    ("cagr", "年化（CAGR）", "pct"),
    ("ann_vol", "年化波动", "pct"),
    ("sharpe", "夏普", "num"),
    ("max_drawdown", "最大回撤", "pct"),
    ("calmar", "卡玛", "num"),
    ("alpha", "Alpha（年化）", "pct"),
    ("beta", "Beta", "num"),
    ("info_ratio", "信息比率", "num"),
    ("trade_count", "成交笔数", "int"),
]


def _benchmark(bars: dict, start: str, end: str) -> pd.Series:
    close = pd.DataFrame({symbol: frame["close"] for symbol, frame in bars.items()})
    close = close.loc[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]
    base = close.apply(lambda series: series.dropna().iloc[0] if series.notna().any() else np.nan)
    return close.div(base.where(base > 0), axis=1).where(close > 0).mean(axis=1, skipna=True)


def _fmt(metrics: dict, key: str, kind: str) -> str:
    value = metrics.get(key)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if kind == "pct":
        return f"{float(value) * 100:.2f}%"
    if kind == "int":
        return f"{int(value):,}"
    return f"{float(value):.3f}"


def run_config(
    config: dict,
    bars: dict,
    wanted: list[str],
    *,
    args,
    industry: pd.DataFrame | None,
    size: pd.DataFrame | None,
    benchmark: pd.Series,
    st_mask: pd.DataFrame | None,
    out_dir: Path,
) -> dict:
    neutral = config["neutralize"]
    strategy = MultiFactorStrategy(
        factors=",".join(wanted),
        weighting=config["weighting"],
        ic_lookback=250,
        ic_min_periods=120,
        weight_refresh=3,
        top_k=args.top_k,
        rebalance=args.rebalance,
        gross_exposure=0.95,
        max_weight=min(0.10, 2.0 / max(args.top_k, 2)),
        winsorize_method="mad",
        winsorize_n=5.0,
        standardize=True,
        min_price=2.0,
        min_amount=args.min_amount,
        min_history=120,
        min_universe=100,
        exclude=st_mask,
        neutralize=neutral,
        industry=industry if neutral else None,
        size=size if neutral else None,
        select_bucket=config["bucket"],
    )
    backtest_config = BacktestConfig(
        start=args.start,
        end=args.end,
        initial_cash=args.cash,
        benchmark=None,
        min_trade_value=args.min_trade_value,
    )
    engine = Engine(
        config=backtest_config,
        benchmark_prices=benchmark,
        delisting=load_delisting_events(),
        delisting_mode="haircut30",
    )
    started = time.perf_counter()
    result = engine.run_strategy(bars, strategy)
    elapsed = time.perf_counter() - started
    picks = strategy.last_diagnostics.get("picks", {})
    counts = [len(names) for names in picks.values()]
    row = {
        "code": config["code"],
        "label": config["label"],
        "seconds": round(elapsed, 1),
        "rebalances": len(picks),
        "avg_holdings": round(float(np.mean(counts)), 1) if counts else 0.0,
    }
    for key, _, _ in METRIC_KEYS:
        row[key] = result.metrics.get(key)
    result.equity.to_csv(out_dir / f"equity-{config['code']}.csv", encoding="utf-8-sig")
    if len(result.trades):
        result.trades.to_csv(out_dir / f"trades-{config['code']}.csv", index=False, encoding="utf-8-sig")
    del result, strategy
    gc.collect()
    return row


def _verdict(rows: list[dict]) -> list[str]:
    """把「C>D>E」这类方向写死成判决，避免事后看着数字自我说服。"""
    by_code = {row["code"]: row for row in rows}

    def ret(code: str) -> float | None:
        row = by_code.get(code)
        if row is None:
            return None
        value = row.get("total_return")
        return float(value) if value is not None and np.isfinite(value) else None

    lines: list[str] = []
    a, b, c, d, e = (ret(code) for code in "ABCDE")
    if a is not None and b is not None:
        delta = (b - a) * 100
        verb = "改善" if delta > 0 else "拖累"
        lines.append(f"- **A → B（滚动 IC 相对等权）**：合计收益 {verb} {abs(delta):.2f}pp。")
    if b is not None and c is not None:
        delta = (c - b) * 100
        verb = "改善" if delta > 0 else "拖累"
        lines.append(f"- **B → C（加行业+市值中性化）**：合计收益 {verb} {abs(delta):.2f}pp。")
    if c is not None and d is not None:
        if d >= c:
            lines.append(
                f"- **分组方向不成立**：最低分组 D（{d * 100:.2f}%）≥ 头部 C（{c * 100:.2f}%），"
                "说明中性化后的打分在横截面上没有区分度，C 的超额不能当作用选股能力。"
            )
        else:
            lines.append(
                f"- **分组方向成立**：头部 C（{c * 100:.2f}%）> 最低分组 D（{d * 100:.2f}%），"
                f"相差 {(c - d) * 100:.2f}pp。"
            )
    if c is not None and e is not None:
        if e >= c:
            lines.append(f"- **中间分组 E（{e * 100:.2f}%）≥ 头部 C，分组不单调**，打分更像风格暴露而非排序信号。")
    return lines


def render(rows: list[dict], kept: list[str], args) -> str:
    lines = [
        "# 方法论对照组（行业+市值中性化 / 滚动 IC / 分组）",
        "",
        f"- 区间：{args.start} ~ {args.end}；调仓 `{args.rebalance}`；持仓 {args.top_k} 只；单笔最小 {args.min_trade_value:,.0f} 元",
        f"- 票池：在市值 + 退市股（haircut30 结算），按 isST 逐日剔除；因子 {len(kept)} 个（全池去冗余后）",
        f"- 因子：{', '.join(kept)}",
        "- 除 A 组以外全部用滚动 IC 加权（250 日窗口、120 日最少样本、每 3 次调仓刷新一次权重）",
        "",
        "| 组 | 配置 | " + " | ".join(name for _, name, _ in METRIC_KEYS) + " | 平均持仓 |",
        "| --- | --- | " + " | ".join("---" for _ in METRIC_KEYS) + " | --- |",
    ]
    for row in rows:
        cells = [_fmt(row, key, kind) for key, _, kind in METRIC_KEYS]
        lines.append(f"| {row['code']} | {row['label']} | " + " | ".join(cells) + f" | {row['avg_holdings']} |")
    lines += [
        "",
        "## 判决（按上表机械得出，不做事后挑选）",
        "",
    ]
    verdict = _verdict(rows)
    lines += verdict if verdict else ["- 组未跑全，无法判决。"]
    lines += [
        "",
        "## 读法",
        "",
        "- **A → B** 只改合成方式：滚动 IC 相对等权的差异，就是「让历史有效的因子多说话」的边际收益。",
        "- **B → C** 只加行业+市值中性化：差异体现「行业/市值暴露」被剥掉后还剩多少选股能力。",
        "- **C → D/E** 是分组反向与中性对照：若信号为真，D（最低分）应明显劣于 C，E（中间组）不该有超额；",
        "  如果 D 与 C 收益接近，说明打分在横截面上没有区分度，C 的超额更可能来自 beta/行业暴露而非选股。",
        "- 所有收益都含 haircut30 的退市结算、ST 剔除与双边费用；**不含**涨跌停/停牌无法成交的额外约束。",
        "",
        "## 已知偏差",
        "",
        "1. 指数成分只有当前快照，回测中不按历史成分过滤（幸存者偏差的一份残留）；",
        "2. 财务指标是东财修订后值，存在轻微前视（公告日已按 NOTICE_DATE 对齐）；",
        "3. 行业分类为年度 PIT，重组后可能滞后一年；",
        "4. 退市结算只有 3 笔真正持有到退市日，三档假设对结论影响约 7pp（见 results/delist-sensitivity.md）。",
        "",
        f"复现：`python -m scripts.methodology_ablation --start {args.start} --end {args.end} --top-k {args.top_k}`",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.methodology_ablation", description=__doc__.splitlines()[0])
    parser.add_argument("--lake", default=str(LAKE))
    parser.add_argument("--kind", default="processed", choices=["processed", "raw"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--factors", default="all")
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--rebalance", default="M")
    parser.add_argument("--min-amount", type=float, default=2e7)
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--dedup-threshold", type=float, default=0.7)
    parser.add_argument("--cash", type=float, default=10_000_000.0)
    parser.add_argument("--min-trade-value", type=float, default=5_000.0)
    parser.add_argument("--codes", default="ABCDE", help="只跑部分组（冒烟用），如 AC")
    parser.add_argument("--out", default=str(ROOT / "results" / "methodology-ablation"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== 读取 {args.kind} 层 ==", flush=True)
    frames, stats = load_layer(Path(args.lake), args.kind, args.limit or None)
    from quant.factors import fundamental

    fundamental.register()
    dead = load_delisted_bars(Path(args.lake))
    added = [symbol for symbol in dead if symbol not in frames]
    frames.update({symbol: dead[symbol] for symbol in added})
    print(f"== 退市股补进 {len(added)} 只（票池 {len(frames)} 只） ==", flush=True)

    lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end)
    frames = {
        symbol: window
        for symbol, frame in frames.items()
        if len(window := frame.loc[(frame.index >= lo) & (frame.index <= hi)]) >= 2
    }

    print("== 构建面板 ==", flush=True)
    panel = build_panel(frames, fields=FIELDS)
    window = (panel.dates >= lo) & (panel.dates <= hi)
    if not window.all():
        panel = type(panel)({key: value.loc[panel.dates[window]] for key, value in panel.fields.items()})
    print(f"  {len(panel.dates)} 天 × {len(panel.symbols)} 只", flush=True)

    from quant.valuation import st_matrix

    st_mask = st_matrix()
    if not st_mask.empty:
        st_mask = st_mask.reindex(index=panel.dates, columns=panel.symbols).fillna(False).astype(bool)
    print(f"== ST 矩阵 {st_mask.shape}，占比 {float(st_mask.to_numpy().mean()) * 100:.2f}% ==", flush=True)

    wanted = factor_names() if args.factors == "all" else [s.strip() for s in args.factors.split(",") if s.strip()]
    print(f"== 因子研究（{len(wanted)} 个） ==", flush=True)
    rows, factor_frames, _timings = run_factor_study(
        panel, wanted, horizon=1, quantiles=5, min_count=args.min_count, block=st_mask
    )
    kept, dedup = dedup_factors(factor_frames, rows, threshold=args.dedup_threshold)
    dedup.to_csv(out_dir / "factor-dedup.csv", index=False, encoding="utf-8-sig")
    print(f"== 去冗余保留 {len(kept)}/{len(rows)}：{', '.join(kept)} ==", flush=True)
    if 1 < len(factor_frames) <= 10:
        factor_corr(factor_frames).to_csv(out_dir / "factor-corr.csv", encoding="utf-8-sig")
    del factor_frames, rows
    gc.collect()

    close = pd.DataFrame({symbol: frame["close"] for symbol, frame in frames.items()})
    industry = industry_labels(close.index, close.columns)
    size = log_size(close)
    print(f"== 暴露矩阵：行业 {industry.shape}（非空 {float(industry.notna().to_numpy().mean()) * 100:.1f}%）、"
          f"市值 {size.shape}（非空 {float(size.notna().to_numpy().mean()) * 100:.1f}%） ==", flush=True)
    benchmark = _benchmark(frames, args.start, args.end)

    selected: set[str] = set()
    for token in args.codes.replace(",", " ").split():
        selected.update(letter for letter in token.upper() if letter.isalpha())
    results: list[dict] = []
    for config in CONFIGS:
        if config["code"] not in selected:
            continue
        print(f"== 组 {config['code']}：{config['label']} ==", flush=True)
        row = run_config(
            config, frames, kept, args=args, industry=industry, size=size,
            benchmark=benchmark, st_mask=st_mask, out_dir=out_dir,
        )
        print("   " + "  ".join(f"{_fmt(row, key, kind)}" for key, _, kind in METRIC_KEYS), flush=True)
        results.append(row)
        pd.DataFrame(results).to_csv(out_dir / "ablation.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(results).to_csv(out_dir / "ablation.csv", index=False, encoding="utf-8-sig")
    (out_dir / "configs.json").write_text(
        json.dumps({"args": vars(args), "configs": CONFIGS, "factors": kept, "stats": stats},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "ablation.md").write_text(render(results, kept, args), encoding="utf-8")
    print(f"\n== 完成：{out_dir / 'ablation.md'} ==", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
