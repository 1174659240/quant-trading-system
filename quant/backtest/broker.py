"""撮合与执行：把「目标权重」翻译成 A 股可成交的订单并撮合。

真实约束在这里，而不是在策略里：

* **T+1**：当日买入的股票当日不可卖（卖出量受 ``Position.available`` 限制）
* **涨跌停**：涨停买不到、跌停卖不出（一字板同样如此）
* **整手**：买入必须是 100 股整数倍（卖出允许零股，因为可能有配股碎股）
* **资金**：现金不足时按比例缩量，不足一手的买单直接放弃
* **容量**：单笔不超过当日成交量的一定比例（默认 10%）
* **成本**：佣金（含最低 5 元）、印花税（卖出）、过户费、滑点
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CostModel, MarketRules


@dataclass
class Order:
    date: pd.Timestamp
    symbol: str
    side: str  # 'buy' | 'sell'
    shares: int
    reason: str = ""
    target_weight: float = 0.0
    ref_price: float = 0.0
    """下单价参考（通常是当日开盘价）。"""

    @property
    def value(self) -> float:
        return self.shares * self.ref_price


@dataclass
class Fill:
    date: pd.Timestamp
    symbol: str
    side: str
    shares: int
    price: float
    gross: float
    fee: float
    slippage_cost: float
    realized_pnl: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "date": self.date,
            "symbol": self.symbol,
            "side": self.side,
            "shares": self.shares,
            "price": round(self.price, 4),
            "gross": round(self.gross, 2),
            "fee": round(self.fee, 2),
            "slippage_cost": round(self.slippage_cost, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "reason": self.reason,
        }


@dataclass
class Rejection:
    date: pd.Timestamp
    symbol: str
    side: str
    shares: int
    reason: str

    def as_dict(self) -> dict:
        return {"date": self.date, "symbol": self.symbol, "side": self.side, "shares": self.shares, "reason": self.reason}


@dataclass
class Broker:
    costs: CostModel = field(default_factory=CostModel)
    rules: MarketRules = field(default_factory=MarketRules)
    rejections: list[Rejection] = field(default_factory=list)

    # ---- 工具 ----
    def _tradable(self, symbol: str, bar: pd.Series | None, side: str) -> tuple[bool, str]:
        if bar is None:
            return False, "停牌/无行情"
        open_px = float(bar["open"])
        prev_close = float(bar["prev_close"]) if "prev_close" in bar and not pd.isna(bar["prev_close"]) else float(bar["close"])
        down, up = self.rules.limit_prices(symbol, prev_close)
        eps = self.rules.limit_epsilon
        if side == "buy" and self.rules.block_buy_at_limit_up:
            if open_px >= up * (1 - eps):
                return False, f"开盘涨停({open_px:.2f}≥{up:.2f})"
        if side == "sell" and self.rules.block_sell_at_limit_down:
            if open_px <= down * (1 + eps):
                return False, f"开盘跌停({open_px:.2f}≤{down:.2f})"
        return True, ""

    def _capacity(self, bar: pd.Series | None) -> float:
        if bar is None or self.rules.max_volume_participation <= 0:
            return float("inf")
        return float(bar["volume"]) * self.rules.max_volume_participation

    def _slipped(self, price: float, side: str) -> float:
        adj = self.costs.slippage_bps / 10_000.0
        return price * (1 + adj) if side == "buy" else price * (1 - adj)

    # ---- 计划 ----
    def plan_orders(
        self,
        date: pd.Timestamp,
        targets: dict[str, float],
        portfolio,
        bars: dict[str, pd.DataFrame],
        *,
        price_field: str = "open",
        min_trade_value: float = 0.0,
    ) -> list[Order]:
        """目标权重 → 订单列表（不改变账户状态，可直接给模拟盘用）。"""
        equity = portfolio.equity
        orders: list[Order] = []
        # 只处理有目标权重或已持仓的标的：目标矩阵通常是全市场宽表，
        # 逐日遍历全部 5000+ 只会让「查行情」从几万次放大到上千万次（回测慢一个量级）。
        symbols = {s for s, w in targets.items() if float(w) != 0.0} | {
            s for s, p in portfolio.positions.items() if p.shares > 0
        }

        for symbol in sorted(symbols):
            weight = float(targets.get(symbol, 0.0))
            if not np.isfinite(weight):
                weight = 0.0
            frame = bars.get(symbol)
            bar = None
            if frame is not None and date in frame.index:
                bar = frame.loc[date]
            pos = portfolio.positions.get(symbol)
            current = pos.shares if pos else 0
            if bar is None:
                if abs(weight) < 1e-12 and current == 0:
                    continue
                price = float(pos.last_price) if pos and pos.last_price else 0.0
                if abs(weight) < 1e-12 and current > 0:
                    orders.append(Order(date, symbol, "sell", current, "清仓(无行情待卖)", 0.0, price))
                continue

            price = float(bar[price_field])
            if price <= 0:
                continue
            target_value = equity * weight
            lot = max(int(self.rules.lot_size), 1)
            target_shares = int(np.floor(target_value / price / lot) * lot)
            if target_shares < 0:
                target_shares = 0 if not self.rules.t_plus_one else target_shares
            delta = target_shares - current
            if delta == 0:
                continue
            if min_trade_value > 0 and abs(delta) * price < min_trade_value:
                continue
            if delta > 0:
                orders.append(Order(date, symbol, "buy", delta, f"加仓至{weight:.1%}", weight, price))
            else:
                qty = -delta
                if self.rules.t_plus_one and pos is not None:
                    qty = min(qty, max(pos.available, 0))
                if qty > 0:
                    orders.append(Order(date, symbol, "sell", qty, f"减仓至{weight:.1%}", weight, price))
        return orders

    # ---- 撮合 ----
    def execute(
        self,
        date: pd.Timestamp,
        orders: list[Order],
        bars: dict[str, pd.DataFrame],
        portfolio,
    ) -> list[Fill]:
        fills: list[Fill] = []
        def bar_of(symbol: str):
            frame = bars.get(symbol)
            return None if frame is None or date not in frame.index else frame.loc[date]

        # 1) 先卖：释放现金并受 T+1 与跌停限制
        for order in [o for o in orders if o.side == "sell"]:
            bar = bar_of(order.symbol)
            ok, why = self._tradable(order.symbol, bar, "sell")
            qty = order.shares
            cap = self._capacity(bar)
            if qty > cap:
                qty = int(qty // max(self.rules.lot_size, 1) * max(self.rules.lot_size, 1)) if cap > 1 else 0
                why = why or f"超出成交量容量({cap:.0f}股)"
            if not ok or qty <= 0:
                if qty <= 0 and why:
                    self.rejections.append(Rejection(date, order.symbol, "sell", order.shares, why))
                elif not ok:
                    self.rejections.append(Rejection(date, order.symbol, "sell", order.shares, why))
                continue
            pos = portfolio.positions.get(order.symbol)
            if pos is None or pos.shares <= 0:
                continue
            qty = min(qty, pos.shares, max(pos.available, 0))
            if qty <= 0:
                self.rejections.append(Rejection(date, order.symbol, "sell", order.shares, "T+1 当日买入不可卖"))
                continue
            raw = float(bar["open"])
            price = self._slipped(raw, "sell")
            fee = self.costs.sell_cost(price, qty)
            realized = portfolio.apply_fill(order.symbol, -qty, price, fee, date)
            fills.append(
                Fill(date, order.symbol, "sell", qty, price, price * qty, fee,
                     (raw - price) * qty, realized, order.reason)
            )

        # 2) 再买：现金约束，按缺口金额从大到小依次成交
        buys = [o for o in orders if o.side == "buy"]
        buys.sort(key=lambda o: o.value, reverse=True)
        for order in buys:
            bar = bar_of(order.symbol)
            ok, why = self._tradable(order.symbol, bar, "buy")
            if not ok:
                self.rejections.append(Rejection(date, order.symbol, "buy", order.shares, why))
                continue
            raw = float(bar["open"])
            price = self._slipped(raw, "buy")
            lot = max(int(self.rules.lot_size), 1)
            qty = int(order.shares // lot * lot)
            cap = self._capacity(bar)
            if qty > cap:
                qty = int(cap // lot * lot)
                if qty <= 0:
                    self.rejections.append(Rejection(date, order.symbol, "buy", order.shares, f"超出成交量容量({cap:.0f}股)"))
                    continue
            # 现金约束：留出费用余量
            usable = portfolio.cash * 0.9999 if portfolio.cash > 0 else 0.0
            affordable = int(usable / price // lot * lot)
            if affordable < qty:
                qty = affordable
                if qty < lot:
                    self.rejections.append(Rejection(date, order.symbol, "buy", order.shares, "现金不足"))
                    continue
            if qty <= 0:
                continue
            fee = self.costs.buy_cost(price, qty)
            if price * qty + fee > portfolio.cash:
                qty -= lot
                if qty <= 0:
                    self.rejections.append(Rejection(date, order.symbol, "buy", order.shares, "现金不足(含费用)"))
                    continue
                fee = self.costs.buy_cost(price, qty)
            portfolio.apply_fill(order.symbol, qty, price, fee, date)
            fills.append(
                Fill(date, order.symbol, "buy", qty, price, price * qty, fee,
                     (price - raw) * qty, 0.0, order.reason)
            )
        return fills
