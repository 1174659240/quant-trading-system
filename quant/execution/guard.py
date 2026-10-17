"""盘前风控护栏（pre-trade guard）：委托出门前的最后一道闸。

成熟的券商/资管流程里，策略信号和券商柜台之间至少还有两层：**组合级限额**与
**订单级风控**。本模块做的就是订单级那层，且刻意做成"能拦就拦、拦不住就缩量、
缩不了就拒单"的保守逻辑：

* **硬拒绝**：急停开关、非交易时段（可配）、标的白名单/黑名单、缺限价、
  价格偏离过大、涨跌停方向错误、重复委托、当日委托数超限；
* **缩量**：整手取整、单笔金额上限、成交量参与度、现金不足、可卖不足、
  单票权重上限、总仓位上限、单日换手上限；
* **拒单**：缩量后低于最小委托金额。

每条结论都带**中文原因与具体数字**，方便事后复盘"为什么这笔没下出去"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from quant.config import CostModel, MarketRules

if TYPE_CHECKING:  # 只在类型检查时导入，避免与 quant.gateway 包形成导入环
    from quant.gateway.base import GatewayOrder

__all__ = ["GuardBatch", "GuardContext", "GuardDecision", "PreTradeGuard", "TradeLimits"]


@dataclass(frozen=True)
class TradeLimits:
    """订单级限额（默认值即"稳健档"，可在 config.toml 的 ``[guard]`` 段覆盖）。"""

    max_weight_per_symbol: float = 0.35
    """单票市值占权益上限（买入时不得突破）。"""

    max_gross_exposure: float = 1.0
    """总仓位上限（含买入后的总市值/权益）。"""

    min_cash_ratio: float = 0.0
    """下单后必须留下的现金比例（留作缓冲）。"""

    max_orders_per_day: int = 200
    """当日最多委托笔数（防程序失控刷单）。"""

    max_single_order_value: float = 500_000.0
    """单笔委托金额上限（元）。"""

    min_order_value: float = 5_000.0
    """缩量后仍低于此金额就拒单（省手续费与废单）。"""

    max_price_deviation_pct: float = 0.10
    """限价相对最新价的允许偏离；超过直接拒单（防手误/脏数据）。"""

    max_volume_participation: float = 0.10
    """单笔不超过上一日成交额的比例（冲击控制）。"""

    max_turnover_of_equity: float = 0.50
    """当日累计成交额 / 总权益 上限（防过度交易）。"""

    allowed_symbols: tuple[str, ...] = ()
    """白名单：非空时只允许列表内标的。"""

    banned_symbols: tuple[str, ...] = ()
    """黑名单：ST、问题股、已退市等。"""

    require_limit_price: bool = True
    """是否强制限价委托（半自动推荐限价）。"""

    session_only: bool = False
    """是否只在连续竞价时段允许提交（半自动盘前生成订单文件时可关）。"""


@dataclass
class GuardContext:
    """一次批量校验所需的市场与账户状态。"""

    cash: float = 0.0
    equity: float = 0.0
    position_value: Mapping[str, float] = field(default_factory=dict)
    available: Mapping[str, int] = field(default_factory=dict)
    latest: Mapping[str, float] = field(default_factory=dict)
    """最新价（估值与偏离检查用）。"""

    prev_close: Mapping[str, float] = field(default_factory=dict)
    prev_amount: Mapping[str, float] = field(default_factory=dict)
    """上一日成交额（参与度检查用）。"""

    submitted_keys: Iterable[str] = ()
    orders_today: int = 0
    daily_turnover: float = 0.0
    kill_switch: bool = False
    session_ok: bool = True

    def copy(self) -> "GuardContext":
        return GuardContext(
            cash=self.cash,
            equity=self.equity,
            position_value=dict(self.position_value),
            available=dict(self.available),
            latest=dict(self.latest),
            prev_close=dict(self.prev_close),
            prev_amount=dict(self.prev_amount),
            submitted_keys=set(self.submitted_keys),
            orders_today=self.orders_today,
            daily_turnover=self.daily_turnover,
            kill_switch=self.kill_switch,
            session_ok=self.session_ok,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["submitted_keys"] = sorted(self.submitted_keys)
        return data


@dataclass
class GuardDecision:
    """单笔委托的判定结果。"""

    order: GatewayOrder
    ok: bool
    reasons: list[str] = field(default_factory=list)
    adjusted_shares: int = 0
    shrunk: bool = False

    @property
    def final_order(self) -> GatewayOrder:
        if self.ok and self.shrunk and self.adjusted_shares > 0:
            from quant.gateway.base import GatewayOrder

            return GatewayOrder(
                symbol=self.order.symbol,
                side=self.order.side,
                shares=int(self.adjusted_shares),
                price=self.order.price,
                order_type=self.order.order_type,
                client_id=self.order.client_id,
                note=self.order.note,
            )
        return self.order

    def to_dict(self) -> dict:
        return {
            "symbol": self.order.symbol,
            "side": self.order.side,
            "shares": self.order.shares,
            "price": self.order.price,
            "ok": self.ok,
            "adjustedShares": self.adjusted_shares,
            "shrunk": self.shrunk,
            "reasons": self.reasons,
        }


@dataclass
class GuardBatch:
    """一批委托的判定结果。"""

    accepted: list[GatewayOrder] = field(default_factory=list)
    rejected: list[tuple[GatewayOrder, str]] = field(default_factory=list)
    decisions: list[GuardDecision] = field(default_factory=list)

    @property
    def adjusted(self) -> list[GuardDecision]:
        return [d for d in self.decisions if d.ok and d.shrunk]

    def summary(self) -> str:
        return (
            f"通过 {len(self.accepted)} 笔（其中缩量 {len(self.adjusted)} 笔），"
            f"拦截 {len(self.rejected)} 笔"
        )


class PreTradeGuard:
    """订单级风控。用法：``guard.check_batch(orders, context)``。"""

    def __init__(
        self,
        limits: TradeLimits | None = None,
        *,
        rules: MarketRules | None = None,
        costs: CostModel | None = None,
    ) -> None:
        self.limits = limits or TradeLimits()
        self.rules = rules or MarketRules()
        self.costs = costs or CostModel()

    # -- 对外入口 -----------------------------------------------------------

    def check_batch(self, orders: Sequence[GatewayOrder], context: GuardContext) -> GuardBatch:
        """逐笔判定，并在过程中**累计**账户状态（后一笔看得到前一笔的影响）。"""
        state = context.copy()
        state.submitted_keys = set(state.submitted_keys)
        batch = GuardBatch()
        for order in orders:
            decision = self._check_one(order, state)
            if not decision.ok:
                batch.decisions.append(decision)
                batch.rejected.append((order, "；".join(decision.reasons)))
                continue
            final = decision.final_order
            decision.adjusted_shares = int(final.shares)
            decision.shrunk = final.shares != order.shares
            batch.decisions.append(decision)
            batch.accepted.append(final)
            self._apply(final, state)
        return batch

    # -- 单笔判定 -----------------------------------------------------------

    def _check_one(self, order: GatewayOrder, state: GuardContext) -> GuardDecision:
        limits = self.limits
        hard: list[str] = []
        notes: list[str] = []

        if state.kill_switch:
            return GuardDecision(order, False, ["急停开关已触发（存在 data/KILL_SWITCH），全部委托被拒"])
        if limits.session_only and not state.session_ok:
            return GuardDecision(order, False, ["当前不在连续竞价时段，按配置拒绝提交"])
        if limits.allowed_symbols and order.symbol not in limits.allowed_symbols:
            return GuardDecision(order, False, [f"{order.symbol} 不在白名单内"])
        if order.symbol in limits.banned_symbols:
            return GuardDecision(order, False, [f"{order.symbol} 在黑名单内（ST/问题股等）"])
        if limits.require_limit_price and not order.price:
            return GuardDecision(order, False, ["按配置要求限价委托，但这笔没有价格"])

        price = float(order.price or state.latest.get(order.symbol, 0.0) or 0.0)
        if price <= 0:
            return GuardDecision(order, False, [f"{order.symbol} 既无限价也没有最新价，无法校验"])

        latest = float(state.latest.get(order.symbol, price))
        if latest > 0:
            deviation = abs(price / latest - 1.0)
            if deviation > limits.max_price_deviation_pct + 1e-9:
                hard.append(
                    f"限价 {price:.2f} 偏离最新价 {latest:.2f} 达 {deviation:.1%}，"
                    f"超过上限 {limits.max_price_deviation_pct:.0%}"
                )

        key = f"{date.today().isoformat()}|{order.key}"
        if key in state.submitted_keys:
            hard.append("同一交易日已提交过完全相同的委托（幂等保护，防重复下单）")
        if state.orders_today >= limits.max_orders_per_day:
            hard.append(f"当日委托数已达上限 {limits.max_orders_per_day} 笔")

        # 涨跌停方向检查
        prev_close = float(state.prev_close.get(order.symbol, 0.0) or 0.0)
        if prev_close > 0:
            down, up = self.rules.limit_prices(order.symbol, prev_close)
            eps = self.rules.limit_epsilon
            if order.side == "buy" and self.rules.block_buy_at_limit_up and price >= up * (1 - eps):
                hard.append(f"涨停价 {up:.2f} 不追（开盘一字板/触板买单成交概率极低）")
            if order.side == "sell" and self.rules.block_sell_at_limit_down and price <= down * (1 + eps):
                hard.append(f"跌停价 {down:.2f} 不砸（触板卖单成交概率极低）")

        if hard:
            return GuardDecision(order, False, hard)

        shares = int(order.shares)
        shares -= shares % self.rules.lot_size
        if shares != int(order.shares):
            notes.append(f"整手取整：{order.shares} → {shares} 股（一手 {self.rules.lot_size} 股）")

        equity = float(state.equity or (state.cash + sum(state.position_value.values())))
        latest_price = latest if latest > 0 else price

        # 单笔金额上限
        cap_shares = int(limits.max_single_order_value // latest_price)
        if cap_shares < shares:
            shares = cap_shares
            notes.append(f"单笔金额上限 {limits.max_single_order_value:,.0f} 元 → 缩到 {shares} 股")

        # 成交量参与度
        prev_amount = float(state.prev_amount.get(order.symbol, 0.0) or 0.0)
        if prev_amount > 0:
            participation_cap = int(limits.max_volume_participation * prev_amount // latest_price)
            if participation_cap < shares:
                shares = participation_cap
                notes.append(
                    f"上一日成交额 {prev_amount:,.0f} 元的 {limits.max_volume_participation:.0%} "
                    f"→ 缩到 {shares} 股（冲击控制）"
                )

        # 单日换手
        if equity > 0:
            turnover_left = limits.max_turnover_of_equity * equity - state.daily_turnover
            turnover_cap = int(max(turnover_left, 0.0) // latest_price)
            if turnover_cap < shares:
                shares = turnover_cap
                notes.append(f"单日换手上限 {limits.max_turnover_of_equity:.0%} 权益 → 缩到 {shares} 股")

        if order.side == "buy":
            # 单票权重上限
            if equity > 0:
                cap_value = limits.max_weight_per_symbol * equity - float(state.position_value.get(order.symbol, 0.0))
                weight_cap = int(max(cap_value, 0.0) // latest_price)
                if weight_cap < shares:
                    shares = weight_cap
                    notes.append(
                        f"单票权重上限 {limits.max_weight_per_symbol:.0%} → 缩到 {shares} 股"
                    )
                gross = sum(state.position_value.values())
                gross_cap_value = limits.max_gross_exposure * equity - gross
                gross_cap = int(max(gross_cap_value, 0.0) // latest_price)
                if gross_cap < shares:
                    shares = gross_cap
                    notes.append(f"总仓上限 {limits.max_gross_exposure:.0%} → 缩到 {shares} 股")
                reserve = limits.min_cash_ratio * equity
                cash_cap = self._max_affordable(latest_price, state.cash - reserve)
                if cash_cap < shares:
                    shares = cash_cap
                    notes.append(f"现金约束（保留 {limits.min_cash_ratio:.0%} 现金）→ 缩到 {shares} 股")
        else:
            available = int(state.available.get(order.symbol, 0) or 0)
            if available < shares:
                shares = available
                notes.append(f"可卖只有 {available} 股（T+1/未成交部分不可卖）→ 缩到 {shares} 股")

        shares -= shares % self.rules.lot_size
        value = shares * latest_price
        if shares <= 0:
            return GuardDecision(order, False, notes + ["缩量后为 0 股，拒单"])
        if value < limits.min_order_value:
            return GuardDecision(
                order, False, notes + [f"缩量后金额 {value:,.0f} 元低于最小委托额 {limits.min_order_value:,.0f} 元，拒单"]
            )
        return GuardDecision(order, True, notes, adjusted_shares=shares, shrunk=shares != int(order.shares))

    # -- 内部工具 -----------------------------------------------------------

    def _max_affordable(self, price: float, cash_available: float) -> int:
        """在含费的前提下算最多能买多少股（整手）。"""
        lot = self.rules.lot_size
        shares = int(max(cash_available, 0.0) // price)
        shares -= shares % lot
        while shares > 0 and price * shares + self.costs.buy_cost(price, shares) > cash_available:
            shares -= lot
        return max(shares, 0)

    def _apply(self, order: GatewayOrder, state: GuardContext) -> None:
        """把通过的委托计入状态：现金、持仓市值、可卖、幂等键、当日笔数与换手。"""
        price = float(order.price or state.latest.get(order.symbol, 0.0) or 0.0)
        value = price * int(order.shares)
        if order.side == "buy":
            fee = self.costs.buy_cost(price, int(order.shares))
            state.cash -= value + fee
            state.position_value[order.symbol] = float(state.position_value.get(order.symbol, 0.0)) + value
        else:
            fee = self.costs.sell_cost(price, int(order.shares))
            state.cash += value - fee
            state.position_value[order.symbol] = max(
                0.0, float(state.position_value.get(order.symbol, 0.0)) - value
            )
            state.available[order.symbol] = max(0, int(state.available.get(order.symbol, 0)) - int(order.shares))
        state.submitted_keys.add(f"{date.today().isoformat()}|{order.key}")
        state.orders_today += 1
        state.daily_turnover += value

    # -- 描述 ---------------------------------------------------------------

    def describe(self) -> dict:
        return {"limits": asdict(self.limits), "rules": asdict(self.rules), "costs": asdict(self.costs)}


def limits_from_mapping(raw: Mapping | None) -> TradeLimits:
    """从 TOML 的 ``[guard]`` 段构造 ``TradeLimits``（列表型字段自动转 tuple）。"""
    raw = dict(raw or {})
    known = set(TradeLimits.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"[guard] 不认识的配置项：{sorted(unknown)}")
    for key in ("allowed_symbols", "banned_symbols"):
        if key in raw and raw[key] is not None:
            raw[key] = tuple(str(x) for x in raw[key])
    return TradeLimits(**raw)
