"""诊断 D>C>E：分组定义是否正确、差异来自 IC 加权还是中性化。

四组对照（同一面板、同一剔除、同一成本、k=30）：
  eq-q5 / eq-q1     等权、不中性化、最高/最低分组
  ic-q5 / ic-q1     滚动 IC 加权、不中性化、最高/最低分组
另外补 icn-q5 / icn-q1（IC 加权 + 行业市值中性化）用来复核消融结果。

输出：results/bucket-diag/{table.csv,report.md,meta.json}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.backtest import Engine
from quant.config import BacktestConfig, CostModel
from quant.factors.exposures import industry_labels, log_size
from quant.factors.panel import build_panel
from quant.factors import fundamental
from quant.strategy.multifactor import MultiFactorStrategy

import importlib.util
import sys

_spec = importlib.util.spec_from_file_location("redlow", Path(__file__).with_name("redlow_study.py"))
redlow = importlib.util.module_from_spec(_spec)
sys.modules["redlow"] = redlow
_spec.loader.exec_module(redlow)

FACTORS = ",".join([
    # 量价（12）
    "momentum", "reversal", "high_ratio", "close_position", "downside_vol", "max_ret",
    "skew", "bias", "turnover_abn", "amihud", "price_volume_corr", "gap",
    # 基本面（12）
    "ep", "bp", "sp", "cfp", "gross_margin", "net_margin", "accruals", "asset_growth",
    "revenue_growth", "debt_ratio", "goodwill_ratio", "size",
])
START = "2017-01-01"
END = "2024-12-31"
TopK = 30


def summarize(result, bench: pd.Series) -> dict[str, Any]:
    metrics = dict(result.report.metrics)
    equity = result.equity
    ret = equity.pct_change().dropna()
    month = (1.0 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1.0
    return {
        "total_return": metrics.get("total_return"),
        "cagr": metrics.get("cagr"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "calmar": metrics.get("calmar"),
        "monthly_win_rate": float((month > 0).mean()),
        "beta": metrics.get("beta"),
        "alpha": metrics.get("alpha"),
        "excess_total": float(equity.iloc[-1] / equity.iloc[0] - 1 - (bench.iloc[-1] / bench.iloc[0] - 1)),
        "picks_median": float(np.median((result.targets > 0).sum(axis=1))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/bucket-diag")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    fundamental.register()
    bars = redlow.load_bars()
    bars = bars[0] if isinstance(bars, tuple) else bars
    bars = {symbol: frame.loc["2014-01-01":END] for symbol, frame in bars.items()}
    panel = build_panel(bars)
    panel = panel.slice(pd.Timestamp(START), pd.Timestamp(END))
    block = redlow.build_block(panel)
    bench_equity = redlow.pool_benchmark(panel, block, redlow.load_delisting_events(redlow.LAKE))
    industry = industry_labels(panel.close.index, panel.close.columns)
    size = log_size(panel.close)
    bt_bars = {symbol: frame.loc[START:END] for symbol, frame in bars.items() if symbol in panel.symbols}

    rows = []
    for weighting in ("equal", "ic"):
        for neutralize, tag in ((None, ""),):
            for bucket in (5, 1):
                name = f"{weighting}{tag}-q{bucket}"
                strategy = MultiFactorStrategy(
                    factors=FACTORS,
                    weighting=weighting,
                    ic_lookback=250,
                    ic_min_periods=120,
                    weight_refresh=3,
                    top_k=TopK,
                    rebalance="ME",
                    gross_exposure=0.95,
                    max_weight=min(0.10, 2.0 / TopK),
                    winsorize_method="mad",
                    winsorize_n=5.0,
                    standardize=True,
                    min_price=2.0,
                    min_amount=2e7,
                    min_history=120,
                    min_universe=100,
                    neutralize=neutralize,
                    industry=industry if neutralize else None,
                    size=size if neutralize else None,
                    select_bucket=bucket,
                    select_quantiles=5,
                    exclude=block,
                )
                config = BacktestConfig(
                    start=START, end=END, initial_cash=1_000_000.0,
                    costs=CostModel(), params={"name": name},
                )
                engine = Engine(
                    config=config,
                    benchmark_prices=(bench_equity / 1_000_000.0).ffill(),
                    delisting=redlow.load_delisting_events(redlow.LAKE),
                    delisting_mode="haircut30",
                )
                weights = strategy.generate(bars)
                result = engine.run(bt_bars, weights, strategy_name=name)
                row = {"config": name, **summarize(result, bench_equity)}
                rows.append(row)
                print(f"[{name}] {row['total_return']:.2%} CAGR {row['cagr']:.2%} "
                      f"夏普 {row['sharpe']} MDD {row['max_drawdown']:.2%} β {row['beta']}", flush=True)
                pd.DataFrame({"equity": result.equity}).to_csv(out / f"equity-{name}.csv", encoding="utf-8-sig")

    table = pd.DataFrame(rows)
    table.to_csv(out / "table.csv", index=False, encoding="utf-8-sig")
    (out / "meta.json").write_text(json.dumps({
        "factors": FACTORS.split(","),
        "start": START, "end": END, "top_k": TopK,
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 {time.time() - t0:.0f}s → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
