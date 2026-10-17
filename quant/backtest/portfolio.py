"""账户与持仓：现金、持仓、T+1 可卖数量、逐日净值快照。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    symbol: str
    shares: int = 0
    available: int = 0
    """T+1 规则下当日可卖出的股数。"""
    cost_basis: float = 0.0
    """摊薄成本价（含费用），用于计算已实现盈亏。"""
    last_price: float = 0.0
    last_date: pd.Timestamp | None = None

    @property
    def market_value(self) -> float:
        return self.shares * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.cost_basis) * self.shares if self.shares else 0.0

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"Position({self.symbol}, {self.shares}股, 成本{self.cost_basis:.3f}, 现价{self.last_price:.3f})"


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    records: list[dict] = field(default_factory=list)

    # ---- 查询 ----
    def position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def shares(self, symbol: str) -> int:
        pos = self.positions.get(symbol)
        return pos.shares if pos else 0

    @property
    def market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    def weights(self) -> dict[str, float]:
        eq = self.equity
        if eq <= 0:
            return {}
        return {s: p.market_value / eq for s, p in self.positions.items() if p.shares}

    # ---- 行情推进 ----
    def mark(self, symbol: str, price: float, date: pd.Timestamp | None = None) -> None:
        pos = self.position(symbol)
        pos.last_price = float(price)
        if date is not None:
            pos.last_date = date

    def mark_all(self, prices: dict[str, float], date: pd.Timestamp | None = None) -> None:
        """按当日行情标记；停牌（无行情）的标的沿用上一价格。"""
        for symbol, pos in self.positions.items():
            price = prices.get(symbol)
            if price is not None and price > 0:
                self.mark(symbol, price, date)

    # ---- 成交 ----
    def apply_fill(self, symbol: str, shares_delta: int, price: float, fee: float, date: pd.Timestamp) -> float:
        """执行一笔成交（买入 shares_delta>0，卖出 <0），返回该笔已实现盈亏。"""
        pos = self.position(symbol)
        realized = 0.0
        if shares_delta > 0:
            cost = price * shares_delta
            total_cost = cost + fee
            new_shares = pos.shares + shares_delta
            pos.cost_basis = (pos.cost_basis * pos.shares + total_cost) / new_shares if new_shares else 0.0
            pos.shares = new_shares
            self.cash -= total_cost
            # 当日买入不计入可卖（T+1）
        elif shares_delta < 0:
            qty = -shares_delta
            proceeds = price * qty - fee
            realized = (price - pos.cost_basis) * qty - fee
            pos.shares -= qty
            pos.available -= qty
            self.cash += proceeds
            self.realized_pnl += realized
            if pos.shares <= 0:
                pos.shares = 0
                pos.available = 0
                pos.cost_basis = 0.0
        pos.last_price = float(price)
        pos.last_date = date
        self.total_fees += fee
        return realized

    def settle(self) -> None:
        """收盘结算：把当日买入的股票变为次日可卖。"""
        for pos in self.positions.values():
            pos.available = pos.shares

    # ---- 快照 ----
    def snapshot(self, date: pd.Timestamp, *, turnover: float = 0.0, fees: float = 0.0) -> dict:
        row = {
            "date": date,
            "equity": self.equity,
            "cash": self.cash,
            "market_value": self.market_value,
            "positions": sum(1 for p in self.positions.values() if p.shares > 0),
            "gross_exposure": self.market_value / self.equity if self.equity > 0 else 0.0,
            "turnover": turnover,
            "fees": fees,
            "realized_pnl": self.realized_pnl,
        }
        self.records.append(row)
        return row

    def history(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["equity", "cash", "market_value", "positions", "gross_exposure", "turnover", "fees"])
        frame = pd.DataFrame(self.records).set_index("date")
        frame.index.name = "date"
        return frame
