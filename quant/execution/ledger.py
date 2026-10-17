"""本地账本（ledger）：持仓、现金、委托幂等键与成交历史的事实来源。

实盘里最容易出事的不是策略，而是"账本和券商对不上"：多下了单、少卖了股、
现金算错导致废单。所以这里把账本做成**显式、可落盘、可对账**的对象：

* 现金与持仓按成交逐笔更新，T+1 用 ``available`` 表达（买入当日不可卖）；
* 每次提交委托都记 ``order_keys``，同一交易日重复提交会被风控拦下；
* ``reconcile()`` 拿券商对账单逐字段比对，差异列出来——这是每天收盘后必做的功课。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant.gateway.base import GatewayFill

__all__ = ["Ledger", "Lot"]


@dataclass
class Lot:
    """一只标的的持仓。"""

    shares: int = 0
    available: int = 0
    """可卖股数（T+1：当日买入的要等下一个交易日）。"""

    last_price: float = 0.0
    cost: float = 0.0
    """持仓均价（含费）。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "shares": int(self.shares),
            "available": int(self.available),
            "lastPrice": round(float(self.last_price), 4),
            "cost": round(float(self.cost), 4),
        }

    @property
    def market_value(self) -> float:
        return float(self.shares) * float(self.last_price)

    @property
    def unrealized(self) -> float:
        return (float(self.last_price) - float(self.cost)) * float(self.shares)


class Ledger:
    """可落盘的账户账本。"""

    def __init__(
        self,
        *,
        cash: float = 0.0,
        positions: Mapping[str, Lot] | None = None,
        order_keys: Iterable[str] | None = None,
        trades: list[dict[str, Any]] | None = None,
        last_settle: str = "",
        path: Path | None = None,
    ) -> None:
        self.cash = float(cash)
        self.positions: dict[str, Lot] = {k: v for k, v in (positions or {}).items()}
        self.order_keys: set[str] = set(order_keys or ())
        self.trades: list[dict[str, Any]] = list(trades or [])
        self.last_settle = last_settle
        self.path = Path(path) if path else None

    # -- 持久化 -------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        path = Path(path)
        if not path.is_file():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        positions = {
            symbol: Lot(
                shares=int(raw.get("shares", 0)),
                available=int(raw.get("available", 0)),
                last_price=float(raw.get("lastPrice", 0.0)),
                cost=float(raw.get("cost", 0.0)),
            )
            for symbol, raw in (data.get("positions") or {}).items()
        }
        return cls(
            cash=float(data.get("cash", 0.0)),
            positions=positions,
            order_keys=data.get("orderKeys") or [],
            trades=data.get("trades") or [],
            last_settle=str(data.get("lastSettle") or ""),
            path=path,
        )

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("没有指定账本路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": round(self.cash, 4),
            "positions": {s: lot.to_dict() for s, lot in sorted(self.positions.items())},
            "orderKeys": sorted(self.order_keys),
            "trades": self.trades[-2000:],
            "lastSettle": self.last_settle,
            "updatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.path = target
        return target

    # -- 成交与结算 ---------------------------------------------------------

    def apply_fill(self, fill: GatewayFill) -> dict[str, Any]:
        """落一笔成交，返回成交记录（含已实现盈亏）。"""
        lot = self.positions.setdefault(fill.symbol, Lot())
        shares = int(fill.shares)
        amount = float(fill.price) * shares
        realized = 0.0
        if fill.side == "buy":
            total_cost = lot.cost * lot.shares + amount + float(fill.fee)
            lot.shares += shares
            lot.cost = total_cost / lot.shares if lot.shares else 0.0
            # T+1：当日买入不增加可卖
            self.cash -= amount + float(fill.fee)
        else:
            shares = min(shares, lot.shares)
            realized = (float(fill.price) - lot.cost) * shares - float(fill.fee)
            lot.shares -= shares
            lot.available = max(0, lot.available - shares)
            self.cash += float(fill.price) * shares - float(fill.fee)
            if lot.shares == 0:
                lot.cost = 0.0
        lot.last_price = float(fill.price)
        record = {
            "time": fill.time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": fill.symbol,
            "side": fill.side,
            "shares": shares,
            "price": round(float(fill.price), 4),
            "amount": round(amount, 2),
            "fee": round(float(fill.fee), 2),
            "realizedPnl": round(realized, 2),
            "cashAfter": round(self.cash, 2),
            "clientId": fill.client_id,
        }
        self.trades.append(record)
        return record

    def settle(self, day: date | str | None = None) -> int:
        """新交易日开始：把昨日买入的持仓转成可卖。返回解锁的标的数。"""
        today = _day_str(day)
        unlocked = 0
        for lot in self.positions.values():
            if lot.available < lot.shares:
                unlocked += 1
            lot.available = lot.shares
        self.last_settle = today
        return unlocked

    def settle_if_new_day(self, day: date | str | None = None) -> bool:
        today = _day_str(day)
        if self.last_settle == today:
            return False
        self.settle(today)
        return True

    # -- 估值 ---------------------------------------------------------------

    def mark(self, prices: Mapping[str, float]) -> None:
        for symbol, price in prices.items():
            lot = self.positions.get(symbol)
            if lot is not None and price and float(price) > 0:
                lot.last_price = float(price)

    def market_value(self, prices: Mapping[str, float] | None = None) -> float:
        if prices:
            self.mark(prices)
        return sum(lot.market_value for lot in self.positions.values() if lot.shares > 0)

    def equity(self, prices: Mapping[str, float] | None = None) -> float:
        return self.cash + self.market_value(prices)

    def shares(self) -> dict[str, int]:
        return {s: lot.shares for s, lot in self.positions.items() if lot.shares > 0}

    def available_shares(self) -> dict[str, int]:
        return {s: lot.available for s, lot in self.positions.items() if lot.available > 0}

    def weights(self, prices: Mapping[str, float] | None = None) -> dict[str, float]:
        total = self.equity(prices)
        if total <= 0:
            return {}
        return {s: lot.market_value / total for s, lot in self.positions.items() if lot.shares > 0}

    # -- 幂等 ---------------------------------------------------------------

    def has_order_key(self, key: str) -> bool:
        return key in self.order_keys

    def record_order_keys(self, keys: Iterable[str], *, day: date | str | None = None) -> None:
        stamp = _day_str(day)
        for key in keys:
            self.order_keys.add(f"{stamp}|{key}")

    # -- 对账 ---------------------------------------------------------------

    def reconcile(self, statement: Iterable[Mapping[str, Any]], *, tolerance: int = 0) -> list[dict[str, Any]]:
        """与券商对账单比对。

        ``statement`` 每行形如 ``{"symbol": "600519.SH", "shares": 200, "available": 200, "cash": ...}``，
        差异返回 ``[{"symbol", "field", "ledger", "broker", "diff"}]``。
        """
        diffs: list[dict[str, Any]] = []
        rows = list(statement)
        seen: set[str] = set()
        for row in rows:
            symbol = str(row.get("symbol") or "").strip()
            if not symbol:
                continue
            seen.add(symbol)
            lot = self.positions.get(symbol, Lot())
            for field_name, ledger_value in (("shares", lot.shares), ("available", lot.available)):
                if field_name not in row or row[field_name] in (None, ""):
                    continue
                broker_value = int(float(row[field_name]))
                if abs(broker_value - int(ledger_value)) > tolerance:
                    diffs.append(
                        {
                            "symbol": symbol,
                            "field": field_name,
                            "ledger": int(ledger_value),
                            "broker": broker_value,
                            "diff": broker_value - int(ledger_value),
                        }
                    )
            if "cash" in row and row["cash"] not in (None, ""):
                broker_cash = float(row["cash"])
                if abs(broker_cash - self.cash) > 0.01 + tolerance:
                    diffs.append(
                        {
                            "symbol": symbol,
                            "field": "cash",
                            "ledger": round(self.cash, 2),
                            "broker": round(broker_cash, 2),
                            "diff": round(broker_cash - self.cash, 2),
                        }
                    )
        for symbol, lot in self.positions.items():
            if lot.shares > 0 and symbol not in seen:
                diffs.append(
                    {
                        "symbol": symbol,
                        "field": "shares",
                        "ledger": lot.shares,
                        "broker": 0,
                        "diff": -lot.shares,
                        "note": "对账单里没有这只标的（券商已清仓或漏记）",
                    }
                )
        return diffs

    # -- 快照 ---------------------------------------------------------------

    def snapshot(self, prices: Mapping[str, float] | None = None) -> dict[str, Any]:
        if prices:
            self.mark(prices)
        market_value = self.market_value()
        return {
            "cash": round(self.cash, 2),
            "marketValue": round(market_value, 2),
            "equity": round(self.cash + market_value, 2),
            "positions": {s: lot.to_dict() for s, lot in sorted(self.positions.items()) if lot.shares > 0},
            "unrealized": round(sum(lot.unrealized for lot in self.positions.values()), 2),
            "trades": len(self.trades),
            "lastSettle": self.last_settle,
        }

    def to_text(self, prices: Mapping[str, float] | None = None) -> str:
        snap = self.snapshot(prices)
        lines = [
            f"现金 {snap['cash']:,.2f} · 市值 {snap['marketValue']:,.2f} · 总权益 {snap['equity']:,.2f} · "
            f"浮盈 {snap['unrealized']:,.2f}（成交 {snap['trades']} 笔，结算日 {snap['lastSettle'] or '—'}）"
        ]
        if snap["positions"]:
            lines.append("持仓：")
            for symbol, lot in snap["positions"].items():
                lines.append(
                    f"  {symbol}  {lot['shares']} 股（可卖 {lot['available']}）"
                    f"  成本 {lot['cost']:.3f}  现价 {lot['lastPrice']:.3f}"
                )
        else:
            lines.append("持仓：空仓")
        return "\n".join(lines)


def _day_str(day: date | str | None = None) -> str:
    if day is None:
        return date.today().isoformat()
    if isinstance(day, date):
        return day.isoformat()
    return str(day)
