"""插件注册表（策略 / 数据源 / 风控）与统一 CLI 的测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import setup_console, tmpdir  # noqa: E402
from quant import plugins  # noqa: E402
from quant.data import SyntheticSource, make_gbm_frame  # noqa: E402
from quant.strategy.base import Strategy  # noqa: E402

setup_console()

SYMBOLS = ["600519.SH", "000001.SZ", "300750.SZ"]


def _bars(periods: int = 160, seed: int = 7) -> dict[str, pd.DataFrame]:
    return {
        symbol: make_gbm_frame(symbol, start="2023-01-02", periods=periods, seed=seed + i, momentum=0.15 * (i + 1))
        for i, symbol in enumerate(SYMBOLS)
    }


def test_builtin_registry_visible():
    names = plugins.strategy_names()
    assert "momentum_rank" in names and "ma_cross" in names, names
    assert {"synthetic", "csv", "akshare"}.issubset(set(plugins.source_names()))
    assert {"max_weight", "exposure_cap", "vol_target"}.issubset(set(plugins.overlay_names()))


def test_register_custom_strategy_and_build():
    @plugins.strategy("always_full_test", replace=True)
    class AlwaysFull(Strategy):
        name = "always_full_test"

        def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
            index = sorted(set().union(*[set(frame.index) for frame in bars.values()]))
            return pd.DataFrame(1.0 / max(1, len(bars)), index=index, columns=list(bars))

    strategy = plugins.build_strategy("always_full_test")
    assert isinstance(strategy, AlwaysFull)
    weights = strategy.generate(_bars(60))
    assert weights.shape == (60, len(SYMBOLS)), weights.shape
    assert abs(float(weights.iloc[-1].sum()) - 1.0) < 1e-9
    assert "always_full_test" in plugins.describe()["strategies"]["custom"]
    # 未知策略仍走内置库的报错路径
    try:
        plugins.build_strategy("没有这个策略")
    except KeyError:
        pass
    else:
        raise AssertionError("未知策略应报 KeyError")


def test_duplicate_registration_requires_replace():
    plugins.register_strategy("dup_test", Strategy, replace=True)
    try:
        plugins.register_strategy("dup_test", Strategy)
    except ValueError:
        pass
    else:
        raise AssertionError("重复注册同名策略应报 ValueError")
    plugins.register_strategy("dup_test", Strategy, replace=True)
    assert "dup_test" in plugins.strategy_names()


def test_custom_source_factory_and_open_source():
    @plugins.source("synth_alias_test", replace=True)
    def open_alias(symbols=None, **kwargs):
        return SyntheticSource(symbols, periods=80, seed=3, **kwargs)

    source = plugins.open_source("synth_alias_test", symbols=SYMBOLS[:2])
    frame = source.load(SYMBOLS[0])
    assert len(frame) == 80 and "close" in frame.columns

    builtin = plugins.open_source("synthetic", symbols=SYMBOLS[:1], periods=60)
    assert len(builtin.load(SYMBOLS[0])) == 60

    try:
        plugins.open_source("不存在的数据源")
    except KeyError:
        pass
    else:
        raise AssertionError("未知数据源应报 KeyError")
    assert "synth_alias_test" in plugins.describe()["sources"]["custom"]


def test_overlay_factory():
    @plugins.overlay("half_test", replace=True)
    def half():
        return lambda weights, bars, ctx: weights * 0.5

    fn = plugins.build_overlay("half_test")
    frame = pd.DataFrame({"a": [1.0, 0.6]})
    out = fn(frame, {}, {})
    assert abs(float(out.iloc[0, 0]) - 0.5) < 1e-12
    assert callable(plugins.build_overlay("max_weight", limit=0.2))
    try:
        plugins.build_overlay("没有这个 overlay")
    except KeyError:
        pass
    else:
        raise AssertionError("未知 overlay 应报 KeyError")


def test_plugins_dir_is_loaded_from_config():
    info = plugins.load_plugins_from_config({}, root=ROOT)
    assert any(path.endswith("example_dual_momentum.py") for path in info["loaded"]), info
    assert not info["errors"], info["errors"]
    assert {"dual_momentum", "cash_floor", "csv_alias"}.issubset(
        set(plugins.strategy_names()) | set(plugins.overlay_names()) | set(plugins.source_names())
    )
    # 重复加载必须幂等（不能因为二次注册而报错）
    again = plugins.load_plugins_from_config({}, root=ROOT)
    assert not again["errors"], again["errors"]


def test_example_plugin_strategy_weights():
    plugins.load_plugins_from_config({}, root=ROOT)
    strategy = plugins.build_strategy("dual_momentum", lookback=20, top_k=2, abs_threshold=-1.0)
    weights = strategy.generate(_bars(120))
    assert weights.shape == (120, len(SYMBOLS)), weights.shape
    last = weights.iloc[-1]
    assert float(last.max()) <= 0.5 + 1e-9, last.to_dict()
    assert float(last.sum()) <= 1.0 + 1e-9
    assert strategy.describe().startswith("dual_momentum(")
    assert float(weights.min().min()) >= 0.0


def test_plugin_overlay_flows_through_engine():
    from quant.backtest import run_backtest
    from quant.config import BacktestConfig

    plugins.load_plugins_from_config({}, root=ROOT)
    strategy = plugins.build_strategy("dual_momentum", lookback=20, top_k=1, abs_threshold=-1.0)
    cfg = BacktestConfig(start="2023-01-02", end="2024-12-31", initial_cash=1_000_000, universe=tuple(SYMBOLS))
    result = run_backtest(bars := _bars(120), strategy, cfg, overlays=[plugins.build_overlay("cash_floor", min_cash=0.5)])
    assert len(result.equity) == len(next(iter(bars.values())))
    # targets 是过完风控的目标权重：cash_floor(0.5) 之后任何一天都不该超过 0.5
    assert float(result.targets.max().max()) <= 0.5 + 1e-9
    assert result.metrics["total_return"] is not None


def test_unknown_params_are_filtered_out():
    """config.toml 的 [params] 是全局段，插件策略不认的键必须被丢掉而不是报错。"""

    @plugins.strategy("only_lookback_test", replace=True)
    class OnlyLookback(Strategy):
        name = "only_lookback_test"

        def __init__(self, lookback: int = 20):
            self.lookback = lookback

        def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
            index = sorted(set().union(*[set(frame.index) for frame in bars.values()]))
            return pd.DataFrame(0.0, index=index, columns=list(bars))

    strategy = plugins.build_strategy("only_lookback_test", lookback=30, top_k=2, skip=1, rebalance="W-FRI")
    assert strategy.lookback == 30
    assert not hasattr(strategy, "top_k")


def test_broken_plugin_file_is_collected():
    with tmpdir("plugins-broken") as directory:
        (directory / "broken_plugin.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        (directory / "_skipped.py").write_text("raise RuntimeError('should be skipped')\n", encoding="utf-8")
        loaded = plugins.load_plugin_dir(directory)
        assert loaded == [], loaded
        assert any("broken_plugin.py" in path for path, _ in plugins.LOAD_ERRORS)
        assert not any("_skipped.py" in path for path, _ in plugins.LOAD_ERRORS)
    assert plugins.describe()["errors"]


def test_cli_entrypoints():
    from quant.cli import main

    assert main(["plugins"]) == 0
    assert main(["plugins", "--json"]) == 0
    assert main(["--version"]) == 0
    assert main(["--help"]) == 0
    assert main(["没有这个子命令"]) == 1
    assert main([]) == 1
