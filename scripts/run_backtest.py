"""跑一次回测并出报表（命令行入口）。

示例::

    # 用合成数据跑均线交叉（离线、可复现）
    python -m scripts.run_backtest --source synthetic --strategy ma_cross --symbols 600519.SH,000001.SZ

    # 用本地 CSV + 配置文件里的参数，并叠加风控
    python -m scripts.run_backtest --config config.toml --source csv --strategy momentum_rank

    # 多策略对比
    python -m scripts.run_backtest --strategies buy_and_hold,ma_cross,momentum_rank --compare

参数优先顺序：命令行 > config.toml > 代码默认值。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant import plugins
from quant import strategy as strategy_lib
from quant.backtest import run_backtest
from quant.config import BacktestConfig, load_config
from quant.data import CsvSource, DataStore, SyntheticSource
from quant.report import compare_report, write_report
from quant.risk import overlays_from_config

DEFAULT_SYMBOLS = "600519.SH,000001.SZ,300750.SZ,601318.SH"


def build_source(name: str, symbols: list[str], csv_dir: str, cache_dir: str):
    if name == "synthetic":
        return SyntheticSource(symbols)
    if name == "csv":
        return CsvSource(csv_dir)
    if name == "akshare":
        from quant.data.akshare_source import AkshareSource

        return AkshareSource()
    if name in plugins.source_names():
        # 插件注册的自定义数据源
        return plugins.open_source(name)
    raise SystemExit(f"不认识的数据源: {name}（可选：{', '.join(plugins.source_names())}）")


def load_config_file(path: str | None) -> tuple[BacktestConfig, dict, str | None]:
    """读 config.toml；同时把 ``[risk]`` 段与 ``[strategy].name`` 单独返回。"""
    if not path:
        return BacktestConfig(), {}, None
    import tomllib

    cfg = load_config(path)
    with Path(path).open("rb") as fh:
        doc = tomllib.load(fh)
    return cfg, dict(doc.get("risk", {})), (doc.get("strategy", {}) or {}).get("name")


def load_benchmark(store: DataStore, cfg: BacktestConfig):
    """单独取基准的收盘价序列；取不到就返回 None（引擎退回全池等权指数）。

    注意：基准**不能**塞进 ``bars``，否则它会被当成一只可交易标的参与选股。
    """
    symbol = cfg.benchmark
    if not symbol:
        return None
    try:
        frame = store.get(symbol, cfg.start, cfg.end)
    except Exception as exc:  # noqa: BLE001
        print(f"基准 {symbol} 取不到（{str(exc)[:80]}）→ 基准改用全池等权指数")
        return None
    series = frame["close"].rename(symbol)
    print(f"基准 {symbol}：{len(series)} 天，{series.index[0].date()} ~ {series.index[-1].date()}，"
          f"区间涨幅 {(series.iloc[-1] / series.iloc[0] - 1) * 100:+.2f}%")
    return series


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="回测入口")
    p.add_argument("--config", default=None, help="TOML 配置文件（默认项目根 config.toml 如存在）")
    p.add_argument("--source", default=None, help=f"数据源: {', '.join(plugins.source_names())}（含插件注册的）")
    p.add_argument(
        "--strategy",
        default=None,
        help=f"策略名之一: {', '.join(plugins.strategy_names())}（含插件注册的）",
    )
    p.add_argument("--strategies", default=None, help="逗号分隔的多个策略（配合 --compare）")
    p.add_argument("--compare", action="store_true", help="多策略对比")
    p.add_argument("--symbols", default=None, help="逗号分隔标的，覆盖配置")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--cash", type=float, default=None, help="初始资金")
    p.add_argument("--name", default=None, help="报表目录名（results/<name>/）")
    p.add_argument("--csv-dir", default="data/csv", help="CSV 数据源目录")
    p.add_argument("--cache-dir", default=None, help="行情缓存目录")
    p.add_argument("--max-weight", type=float, default=None, help="单标的上限，例如 0.3")
    p.add_argument("--exposure-cap", type=float, default=None, help="总仓位上限，例如 0.8")
    p.add_argument("--vol-target", type=float, default=None, help="目标年化波动，例如 0.15")
    p.add_argument("--no-overlay", action="store_true", help="忽略配置里的风控 overlay")
    p.add_argument("--slippage-bps", type=float, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = args.config
    if config_path is None and (ROOT / "config.toml").is_file():
        config_path = str(ROOT / "config.toml")
    cfg, risk_cfg, default_strategy = load_config_file(config_path)
    # ---- 自定义插件（strategies/ 数据源/ 风控）----
    plugin_info = plugins.load_plugins_from_config_file(config_path, root=ROOT)
    if plugin_info["loaded"] and not args.quiet:
        print("已加载插件：" + ", ".join(Path(p).name for p in plugin_info["loaded"]))
    if plugin_info["errors"]:
        for err in plugin_info["errors"]:
            print(f"[警告] 插件加载失败 {err['path']}：{err['error'].strip().splitlines()[-1]}")

    # ---- 命令行覆盖（frozen dataclass → dataclasses.replace）----
    overrides: dict = {}
    if args.start:
        overrides["start"] = args.start
    if args.end:
        overrides["end"] = args.end
    if args.cash:
        overrides["initial_cash"] = args.cash
    if args.symbols:
        overrides["universe"] = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    if args.cache_dir:
        overrides["data_dir"] = args.cache_dir
    costs = cfg.costs
    if args.slippage_bps is not None:
        costs = replace(costs, slippage_bps=args.slippage_bps)
    cfg = replace(cfg, costs=costs, **overrides)

    source_name = args.source or "synthetic"
    symbols = list(cfg.universe) or [s for s in DEFAULT_SYMBOLS.split(",")]
    if not cfg.universe:
        cfg = replace(cfg, universe=tuple(symbols))

    source = build_source(source_name, symbols, args.csv_dir, cfg.data_dir)
    store = DataStore(source, cache_dir=None if source_name == "synthetic" else cfg.data_dir, data_dir=cfg.data_dir)
    bars = store.get_many(symbols, cfg.start, cfg.end)
    if not bars:
        raise SystemExit(f"没有取到任何行情：source={source_name} symbols={symbols} 区间={cfg.start}~{cfg.end}")
    print(f"取到 {len(bars)} 只标的，共 {sum(len(f) for f in bars.values())} 根 K 线（source={source_name}）")

    # ---- 基准：单独取一次，绝不混进 bars（否则会被当成可交易标的）----
    benchmark_prices = load_benchmark(store, cfg)

    # ---- 风控 overlay ----
    overlays = []
    if not args.no_overlay:
        overlays = overlays_from_config(risk_cfg)
    if args.max_weight is not None:
        from quant.risk import max_weight

        overlays.append(max_weight(args.max_weight))
    if args.exposure_cap is not None:
        from quant.risk import exposure_cap

        overlays.append(exposure_cap(args.exposure_cap))
    if args.vol_target is not None:
        from quant.risk import vol_target

        overlays.append(vol_target(args.vol_target))
    if overlays:
        print("风控 overlay：" + " → ".join(getattr(o, "__name__", str(o)) for o in overlays))

    names = [s.strip() for s in (args.strategies.split(",") if args.strategies else []) if s.strip()]
    if not names:
        names = [args.strategy or default_strategy or "buy_and_hold"]
    if args.compare and len(names) < 2:
        names = ["buy_and_hold", "ma_cross", "momentum_rank"]
        print("--compare 至少需要两个策略，自动改用：" + ", ".join(names))

    results: dict[str, object] = {}
    for name in names:
        strategy = plugins.build_strategy(name, **dict(cfg.params))
        result = run_backtest(bars, strategy, cfg, overlays=overlays, benchmark_prices=benchmark_prices)
        results[strategy.describe()] = result
        if not args.quiet:
            print("\n" + result.summary())
        out_name = args.name or name
        if len(names) > 1:
            out_name = f"{args.name or 'batch'}-{name}"
        out = write_report(result, cfg.result_dir, out_name)
        print(f"报表已写入 {out}")

    if len(results) > 1:
        out = compare_report(results, cfg.result_dir, args.name or "compare")
        print(f"对比报表已写入 {out}")
        table = pd.DataFrame({k: v.metrics for k, v in results.items()}).T
        cols = [c for c in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "win_rate", "trade_count") if c in table.columns]
        print(table[cols].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
