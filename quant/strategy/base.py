"""策略基类。

**唯一约定**：策略输出「目标权重矩阵」而不是买卖指令::

               600519.SH  000001.SZ
    2023-01-03      0.50       0.00
    2023-01-04      0.50       0.30
    ...

- 行＝交易日（与输入 bars 的并集日历一致，升序）
- 列＝标的代码
- 值＝该标的当日**收盘后**想要持有的仓位比例（1.0 表示满仓这一只）
- 引擎负责把它翻译成订单、并按 A 股规则撮合（T+1、涨跌停、整手、费用）

这样策略只管「想拿什么仓」，成交约束、成本、风控全在引擎侧，策略代码干净且
可单独测试；也天然支持多标的轮动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class Strategy(ABC):
    name = "strategy"
    #: True 表示允许做空（A股默认 False，权重会被截断到 >=0）
    allow_short = False

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    # ---- 子类实现 ----
    @abstractmethod
    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """返回目标权重矩阵。"""

    # ---- 公共工具 ----
    def rebalance_mask(self, index: pd.DatetimeIndex, freq: str | None) -> pd.Series:
        """哪些日子允许调仓；``freq`` 走 pandas 频率别名（'W-FRI' / 'ME' / 'QE' 等）。"""
        mask = pd.Series(True, index=index)
        if not freq:
            return mask
        try:
            marks = pd.Series(index, index=index).resample(freq).last().dropna()
        except Exception:
            return mask
        allowed = set(marks.values)
        return pd.Series([d in allowed for d in index], index=index)

    def hold_to_rebalance(self, weights: pd.DataFrame, freq: str | None) -> pd.DataFrame:
        """只在调仓日更新权重，其余日子沿用上一期（避免每天换手）。"""
        if not freq:
            return weights
        mask = self.rebalance_mask(weights.index, freq)
        out = weights.where(mask, other=np.nan)
        return out.ffill().fillna(0.0)

    def clip_weights(self, weights: pd.DataFrame) -> pd.DataFrame:
        out = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = out.clip(lower=-1.0 if self.allow_short else 0.0, upper=1.0)
        return out

    def describe(self) -> str:
        if not self.params:
            return self.name
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({inner})"


class CompositeStrategy(Strategy):
    """把多个策略的权重按权重加起来（等权或指定比例）。"""

    def __init__(self, strategies: list[Strategy], weights: list[float] | None = None, **params: Any) -> None:
        super().__init__(**params)
        if not strategies:
            raise ValueError("CompositeStrategy 至少需要一个子策略")
        self.strategies = strategies
        w = np.array(weights if weights is not None else [1.0] * len(strategies), dtype=float)
        if len(w) != len(strategies):
            raise ValueError("weights 长度必须与 strategies 一致")
        self.weights = w / w.sum() if w.sum() != 0 else w

    @property
    def name(self) -> str:  # type: ignore[override]
        return "composite(" + "+".join(s.name for s in self.strategies) + ")"

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        total: pd.DataFrame | None = None
        for w, strat in zip(self.weights, self.strategies):
            part = strat.generate(bars) * w
            total = part if total is None else total.add(part, fill_value=0.0)
        assert total is not None
        return self.clip_weights(total.fillna(0.0))
