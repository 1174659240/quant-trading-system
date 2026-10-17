"""配置对象：交易成本、市场规则、回测参数。

全部是 ``dataclass``，可从 TOML 读入，也可直接构造。TOML 用标准库
``tomllib`` 解析，不引入额外依赖。
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CostModel:
    """交易成本模型（A股默认值，2023 年后费率）。

    ``slippage_bps`` 是**单边**冲击成本，按基点计：买入成交价上浮、
    卖出成交价下压，模拟排队与冲击。
    """

    commission_rate: float = 0.00025
    """佣金费率（双边），万 2.5。"""

    commission_min: float = 5.0
    """单笔最低佣金（元）。"""

    stamp_duty_rate: float = 0.0005
    """印花税，仅卖出收取，0.05%。"""

    transfer_fee_rate: float = 0.00001
    """过户费，双边 0.001%。"""

    slippage_bps: float = 2.0
    """单边滑点（基点），2bp = 万分之二。"""

    def buy_cost(self, price: float, shares: int) -> float:
        """买入一笔的费用（不含成交金额本身）。"""
        amount = price * shares
        commission = max(amount * self.commission_rate, self.commission_min)
        return commission + amount * self.transfer_fee_rate

    def sell_cost(self, price: float, shares: int) -> float:
        """卖出一笔的费用。"""
        amount = price * shares
        commission = max(amount * self.commission_rate, self.commission_min)
        return commission + amount * self.transfer_fee_rate + amount * self.stamp_duty_rate


@dataclass(frozen=True)
class MarketRules:
    """市场微观规则。板块差异通过 ``price_limit_overrides`` 逐标的覆盖。"""

    lot_size: int = 100
    """一手股数，买入必须是整数倍。"""

    t_plus_one: bool = True
    """当日买入的股票当日不可卖出。"""

    price_limit_pct: float = 0.10
    """默认涨跌停幅度（主板 ±10%）。"""

    price_limit_overrides: Mapping[str, float] = field(default_factory=dict)
    """逐标的覆盖，例如 ``{"300750": 0.20, "688981": 0.20, "ST000001": 0.05}``。"""

    limit_epsilon: float = 0.001
    """判定触板的容差（相对涨跌停价），避免浮点误差误判。"""

    block_buy_at_limit_up: bool = True
    """涨停不可买入（一字板买不到）。"""

    block_sell_at_limit_down: bool = True
    """跌停不可卖出。"""

    max_volume_participation: float = 0.10
    """单笔委托最多吃掉当日成交量的比例，超出部分按比例截断。"""

    def limit_pct(self, symbol: str) -> float:
        return float(self.price_limit_overrides.get(symbol, self.price_limit_pct))

    def limit_prices(self, symbol: str, prev_close: float) -> tuple[float, float]:
        """返回 ``(跌停价, 涨停价)``，按 A股四舍五入到分。"""
        pct = self.limit_pct(symbol)
        down = round(prev_close * (1 - pct) + 1e-9, 2)
        up = round(prev_close * (1 + pct) + 1e-9, 2)
        return down, up


@dataclass(frozen=True)
class BacktestConfig:
    """一次回测的完整参数。"""

    start: str = "2022-01-01"
    end: str = "2024-12-31"
    initial_cash: float = 1_000_000.0
    universe: tuple[str, ...] = ()
    benchmark: str | None = None
    """基准指数代码，仅用于报表对比（需数据里存在该标的）。"""

    data_dir: str = "data/cache"
    result_dir: str = "results"
    costs: CostModel = field(default_factory=CostModel)
    rules: MarketRules = field(default_factory=MarketRules)
    params: Mapping[str, Any] = field(default_factory=dict)
    """策略超参，原样透传给 ``Strategy(**params)``。"""

    min_trade_value: float = 0.0
    """单笔调仓的最小金额（元）；低于它的零碎调整会被跳过，避免每天为价格漂移付佣金。"""

    def with_params(self, **overrides: Any) -> "BacktestConfig":
        merged = dict(self.params)
        merged.update(overrides)
        return replace(self, params=merged)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, raw: Mapping[str, Any] | None):
    """按 dataclass 字段过滤未知键，便于配置向前兼容。"""
    raw = dict(raw or {})
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{cls.__name__} 不认识的配置项: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> BacktestConfig:
    """从 TOML 文件读取回测配置。

    文件形如::

        [backtest]
        start = "2022-01-01"
        end   = "2024-12-31"
        initial_cash = 1000000
        universe = ["600519", "000001"]

        [costs]
        slippage_bps = 3

        [rules]
        price_limit_pct = 0.10

        [params]
        fast = 10
        slow = 30
    """
    path = Path(path)
    with path.open("rb") as fh:
        doc = tomllib.load(fh)
    backtest = dict(doc.get("backtest", {}))
    costs = _build(CostModel, doc.get("costs"))
    rules = _build(MarketRules, doc.get("rules"))
    if "universe" in backtest:
        backtest["universe"] = tuple(backtest["universe"])
    backtest.setdefault("params", doc.get("params", {}))
    return _build(BacktestConfig, {**backtest, "costs": costs, "rules": rules})


def load_params(path: str | Path) -> dict[str, Any]:
    """只读 ``[params]`` 段，方便策略单独使用。"""
    with Path(path).open("rb") as fh:
        return dict(tomllib.load(fh).get("params", {}))
