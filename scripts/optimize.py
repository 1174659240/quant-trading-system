"""策略参数网格搜索（内置 walk-forward 样本外检验）。

示例::

    python -m scripts.optimize --strategy ma_cross --grid fast=5,10,20 slow=30,60 --objective sharpe
    python -m scripts.optimize --strategy momentum_rank --grid "lookback=20,60,120" --objective calmar

为什么要样本外：网格搜出来的最优参数在样本内几乎必然好看，
所以这里默认把区间切成「前 70% 训练 / 后 30% 检验」，两个都报出来。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant import plugins
from quant import strategy as strategy_lib
from quant.backtest import run_backtest
from quant.config import load_config
from quant.data import CsvSource, DataStore, SyntheticSource
from quant.risk import overlays_from_config

DEFAULT_SYMBOLS = "600519.SH,000001.SZ,300750.SZ,601318.SH"
OBJECTIVES = ("sharpe", "calmar", "total_return", "cagr", "sortino")


def parse_grid(text: str) -> dict[str, list]:
    """``fast=5,10 slow=30,60`` → ``{"fast": [5.0, 10.0], "slow": [30.0, 60.0]}``"""
    grid: dict[str, list] = {}
    for chunk in text.replace(";", " ").split():
        if "=" not in chunk:
            raise SystemExit(f"网格项格式应为 name=v1,v2：{chunk}")
        key, values = chunk.split("=", 1)
        parsed = []
        for raw in values.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed.append(int(raw))
            except ValueError:
                try:
                    parsed.append(float(raw))
                except ValueError:
                    parsed.append(raw)
        grid[key.strip()] = parsed
    if not grid:
        raise SystemExit("网格为空")
    return grid


def slice_bars(bars: dict[str, pd.DataFrame], start=None, end=None) -> dict[str, pd.DataFrame]:
    out = {}
    for symbol, frame in bars.items():
        sub = frame
        if start is not None:
            sub = sub[sub.index >= pd.Timestamp(start)]
        if end is not None:
            sub = sub[sub.index <= pd.Timestamp(end)]
        if len(sub) > 30:
            out[symbol] = sub
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="参数网格搜索")
    p.add_argument("--config", default="config.toml")
    p.add_argument("--strategy", default="ma_cross")
    p.add_argument("--grid", required=True, help='例如 "fast=5,10,20 slow=30,60"')
    p.add_argument("--objective", default="sharpe", choices=OBJECTIVES)
    p.add_argument("--source", default="csv", choices=["csv", "synthetic"])
    p.add_argument("--csv-dir", default="data/csv")
    p.add_argument("--symbols", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--oos-split", type=float, default=0.7, help="训练集占比（0 表示不做样本外）")
    p.add_argument("--top", type=int, default=15, help="打印前 N 名")
    p.add_argument("--out", default=None, help="结果 CSV 路径（默认 results/optimize-<strategy>.csv）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plugins.load_plugins_from_config_file(args.config if args.config and Path(args.config).is_file() else None, root=ROOT)
    cfg = load_config(args.config) if Path(args.config).is_file() else None
    if cfg is None:
        from quant.config import BacktestConfig

        cfg = BacktestConfig()
    symbols = [s.strip() for s in (args.symbols or ",".join(cfg.universe) or DEFAULT_SYMBOLS).split(",") if s.strip()]
    source = SyntheticSource(symbols) if args.source == "synthetic" else CsvSource(args.csv_dir)
    store = DataStore(source)
    bars = store.get_many(symbols, args.start or cfg.start, args.end or cfg.end)
    if not bars:
        raise SystemExit("没有取到行情")
    overlays = overlays_from_config(getattr(cfg, "_risk", None))

    index = sorted(set().union(*[set(f.index) for f in bars.values()]))
    split_at = index[int(len(index) * args.oos_split)] if 0 < args.oos_split < 1 else None
    train = slice_bars(bars, None, split_at)
    test = slice_bars(bars, split_at, None) if split_at is not None else {}

    grid = parse_grid(args.grid)
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"策略 {args.strategy}：{len(combos)} 组参数，样本内 {len(train.get(symbols[0], []))} 根 / "
          f"样本外 {len(test.get(symbols[0], [])) if test else 0} 根 K 线")

    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        row = dict(params)
        strategy = plugins.build_strategy(args.strategy, **params)
        for label, data in (("train", train), ("test", test)):
            if not data:
                continue
            result = run_backtest(data, plugins.build_strategy(args.strategy, **params), cfg)
            m = result.metrics
            row[f"{label}_return"] = m.get("total_return")
            row[f"{label}_sharpe"] = m.get("sharpe")
            row[f"{label}_mdd"] = m.get("max_drawdown")
            row[f"{label}_calmar"] = m.get("calmar")
            row[f"{label}_trades"] = m.get("trade_count")
        rows.append(row)

    table = pd.DataFrame(rows)
    sort_col = f"train_{args.objective}" if f"train_{args.objective}" in table else args.objective
    table = table.sort_values(sort_col, ascending=False).reset_index(drop=True)
    out = Path(args.out or f"results/optimize-{args.strategy}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")

    show_cols = list(keys) + [c for c in table.columns if c not in keys]
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(table[show_cols].head(args.top).to_string(index=False))
    if test:
        print("\n提示：看 train_* 与 test_* 的落差。落差大＝过拟合，优先选两边都稳的那组，而不是样本内冠军。")
    print(f"\n完整结果：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
