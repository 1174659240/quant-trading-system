"""示例插件：演示三种扩展怎么注册（放进 ``plugins/`` 就会被自动加载）。

注册后即可被命令行直接引用::

    python -m quant plugins                                  # 确认 dual_momentum / cash_floor / csv_alias 在列
    python -m quant backtest --strategy dual_momentum --source csv --csv-dir data/csv

三种声明风格（本文件用装饰器，另外两种见 ``quant/plugins.py`` 文档）：

1. 装饰器 ``@strategy`` / ``@source`` / ``@overlay``（推荐，注册与定义在一起）；
2. 模块级字典 ``STRATEGIES = {...}`` / ``SOURCES = {...}`` / ``OVERLAYS = {...}``；
3. 函数钩子 ``def register(api): api.strategy(...)``（适合动态生成一批名字）。

**契约不变**：策略输出目标权重矩阵（行=交易日、列=标的、值=收盘后想持有的仓位比例），
数据源实现 ``load(symbol, start, end)``，风控 overlay 是 ``(weights, bars, ctx) -> weights``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.metrics import TRADING_DAYS
from quant.plugins import overlay, source, strategy
from quant.strategy.base import Strategy


@strategy("dual_momentum")
class DualMomentum(Strategy):
    """双重动量：先做绝对动量过滤（不看跌），再在合格的标的里取相对动量前 top_k 等权。

    与内置 ``momentum_rank`` 的差别：多了 ``abs_threshold`` 这道「市场不对就空仓」的闸门，
    若用 ``vol_adjust`` 还会按 20 日波动把仓位缩放到目标波动附近。
    """

    name = "dual_momentum"

    def __init__(
        self,
        lookback: int = 60,
        top_k: int = 2,
        abs_threshold: float = 0.0,
        rebalance: str | None = "W-FRI",
        vol_adjust: bool = False,
        target_vol: float = 0.15,
        vol_window: int = 20,
    ) -> None:
        super().__init__(
            lookback=lookback,
            top_k=top_k,
            abs_threshold=abs_threshold,
            rebalance=rebalance,
            vol_adjust=vol_adjust,
        )
        self.lookback = int(lookback)
        self.top_k = max(1, int(top_k))
        self.abs_threshold = float(abs_threshold)
        self.rebalance = rebalance
        self.vol_adjust = bool(vol_adjust)
        self.target_vol = float(target_vol)
        self.vol_window = int(vol_window)

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame()
        close = pd.DataFrame({symbol: frame["close"] for symbol, frame in bars.items()}).sort_index()
        momentum = close / close.shift(self.lookback) - 1.0
        eligible = momentum.where(momentum > self.abs_threshold)
        ranks = eligible.rank(axis=1, ascending=False, method="first")
        picked = (ranks <= self.top_k) & eligible.notna()
        counts = picked.sum(axis=1)
        weights = picked.div(counts.replace(0, np.nan), axis=0).fillna(0.0)

        if self.vol_adjust:
            vol = close.pct_change().rolling(self.vol_window).std() * np.sqrt(TRADING_DAYS)
            scale = (self.target_vol / vol.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
            weights = weights * scale

        return self.clip_weights(self.hold_to_rebalance(weights, self.rebalance))


@overlay("cash_floor")
def cash_floor(min_cash: float = 0.1):
    """至少留 ``min_cash`` 比例的现金：把目标权重整体缩放到 ``1 - min_cash``。

    比 ``exposure_cap`` 更弱——它只在满仓时起作用，不改变标的之间的相对比例。
    """
    keep = max(0.0, min(1.0, 1.0 - float(min_cash)))

    def apply(weights: pd.DataFrame, bars: dict[str, pd.DataFrame], ctx: dict) -> pd.DataFrame:
        return weights * keep

    apply.__name__ = f"cash_floor(min_cash={min_cash})"
    return apply


@source("csv_alias")
def open_csv_alias(root: str = "data/csv", symbol_column: str | None = None):
    """数据源示例：把 CSV 源包装成带默认目录的别名，``--source csv_alias`` 即可用。"""
    from quant.data import CsvSource

    return CsvSource(root, symbol_column=symbol_column)
