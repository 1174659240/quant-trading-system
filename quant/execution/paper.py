"""模拟盘 / 半自动实盘：账户状态落盘，下单建议可复核。

设计原则——**不替你偷偷下单**：

* ``plan()`` 只输出「下一交易日应该买/卖多少」的建议单（含价格、金额、费用预估）
* ``apply_fills()`` 让你把真实成交回报录入，账户随之更新（T+1 自动结算）
* ``mark_to_market()`` 用最新收盘价重估，追加净值曲线

状态文件（默认 ``data/paper/``）: ``account.json`` / ``trades.csv`` / ``equity.csv``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..backtest.broker import Broker, Fill, Order
from ..backtest.portfolio import Portfolio, Position
from ..config import BacktestConfig
from ..data.store import DataStore
from ..risk import RiskManager


@dataclass
class PaperTrader:
    store: DataStore
    strategy: object
    config: BacktestConfig = field(default_factory=BacktestConfig)
    state_dir: str | Path = "data/paper"
    risk: RiskManager | None = None

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.account_path = self.state_dir / "account.json"
        self.trades_path = self.state_dir / "trades.csv"
        self.equity_path = self.state_dir / "equity.csv"
        self.broker = Broker(costs=self.config.costs, rules=self.config.rules)
        self.portfolio = self._load_account()

    # ------------------------------------------------------------ 状态读写
    def _load_account(self) -> Portfolio:
        if self.account_path.is_file():
            raw = json.loads(self.account_path.read_text(encoding="utf-8"))
            pf = Portfolio(cash=float(raw.get("cash", self.config.initial_cash)))
            pf.realized_pnl = float(raw.get("realized_pnl", 0.0))
            pf.total_fees = float(raw.get("total_fees", 0.0))
            for symbol, pos in raw.get("positions", {}).items():
                pf.positions[symbol] = Position(
                    symbol=symbol,
                    shares=int(pos.get("shares", 0)),
                    available=int(pos.get("available", 0)),
                    cost_basis=float(pos.get("cost_basis", 0.0)),
                    last_price=float(pos.get("last_price", 0.0)),
                )
            return pf
        return Portfolio(cash=float(self.config.initial_cash))

    def save(self) -> Path:
        payload = {
            "cash": self.portfolio.cash,
            "realized_pnl": self.portfolio.realized_pnl,
            "total_fees": self.portfolio.total_fees,
            "updated": pd.Timestamp.now().isoformat(timespec="seconds"),
            "positions": {
                s: {
                    "shares": p.shares,
                    "available": p.available,
                    "cost_basis": round(p.cost_basis, 4),
                    "last_price": round(p.last_price, 4),
                    "market_value": round(p.market_value, 2),
                }
                for s, p in self.portfolio.positions.items()
                if p.shares > 0
            },
        }
        self.account_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.account_path

    # ------------------------------------------------------------ 取数
    def load_bars(self, symbols: list[str], lookback_days: int = 400) -> dict[str, pd.DataFrame]:
        end = self.config.end
        start = (pd.Timestamp(end) - pd.Timedelta(days=int(lookback_days * 1.6))).date().isoformat()
        return self.store.get_many(symbols, start, end)

    # ------------------------------------------------------------ 下单建议
    def plan(self, symbols: list[str] | None = None, *, as_of: str | None = None) -> pd.DataFrame:
        """给出下一交易日的建议订单表（不改变账户）。"""
        symbols = symbols or list(self.config.universe)
        if not symbols:
            raise ValueError("没有指定标的池（config.universe 为空）")
        bars = self.load_bars(symbols)
        if as_of is not None:
            bars = {s: f[f.index <= pd.Timestamp(as_of)] for s, f in bars.items()}
        weights = self.strategy.generate(bars)
        last_day = max(f.index.max() for f in bars.values())
        targets = {s: float(weights[s].loc[:last_day].iloc[-1]) for s in weights.columns if len(weights[s].loc[:last_day])}
        if self.risk is not None:
            targets = self.risk.trim_targets(targets, self.portfolio.equity)
        orders = self.broker.plan_orders(last_day, targets, self.portfolio, bars, price_field="close")
        rows = []
        for order in orders:
            est_fee = (
                self.config.costs.buy_cost(order.ref_price, order.shares)
                if order.side == "buy"
                else self.config.costs.sell_cost(order.ref_price, order.shares)
            )
            rows.append(
                {
                    "symbol": order.symbol,
                    "side": "买入" if order.side == "buy" else "卖出",
                    "shares": order.shares,
                    "ref_price": round(order.ref_price, 3),
                    "amount": round(order.value, 2),
                    "est_fee": round(est_fee, 2),
                    "reason": order.reason,
                }
            )
        frame = pd.DataFrame(rows)
        if len(frame):
            frame["weight_after"] = frame["symbol"].map(lambda s: round(targets.get(s, 0.0), 4))
        return frame

    # ------------------------------------------------------------ 回报录入
    def apply_fills(self, fills: list[dict] | pd.DataFrame, date: str | None = None) -> Path:
        """录入真实成交：``[{symbol, side, shares, price, fee?}, ...]``。

        先 ``settle()``（把上一交易日买入的股份转为可卖），再落成交——
        这样当日买入的股份在本日仍然不可卖，T+1 与回测口径一致。
        隐含假设：每天最多调用一次（模拟盘本来就是日频节奏）。
        """
        rows = fills.to_dict("records") if isinstance(fills, pd.DataFrame) else list(fills)
        day = pd.Timestamp(date or pd.Timestamp.today().date())
        self.portfolio.settle()
        records = []
        for row in rows:
            symbol = str(row["symbol"])
            side = "buy" if str(row.get("side", "buy")).lower() in {"buy", "b", "买入"} else "sell"
            shares = int(row["shares"])
            price = float(row["price"])
            fee = float(row["fee"]) if row.get("fee") is not None else (
                self.config.costs.buy_cost(price, shares) if side == "buy" else self.config.costs.sell_cost(price, shares)
            )
            delta = shares if side == "buy" else -shares
            realized = self.portfolio.apply_fill(symbol, delta, price, fee, day)
            records.append(
                {"date": day.date(), "symbol": symbol, "side": side, "shares": shares,
                 "price": price, "fee": round(fee, 2), "realized_pnl": round(realized, 2)}
            )
        if records:
            frame = pd.DataFrame(records)
            header = not self.trades_path.is_file()
            frame.to_csv(self.trades_path, mode="a", header=header, index=False, encoding="utf-8-sig")
        self.save()
        return self.trades_path

    # ------------------------------------------------------------ 净值
    def mark_to_market(self, date: str | None = None) -> pd.DataFrame:
        symbols = sorted({*self.config.universe, *self.portfolio.positions})
        prices: dict[str, float] = {}
        for symbol in symbols:
            try:
                frame = self.store.get(symbol)
                prices[symbol] = float(frame["close"].iloc[-1])
            except Exception:
                continue
        self.portfolio.mark_all(prices, pd.Timestamp(date or pd.Timestamp.today().date()))
        row = pd.DataFrame(
            [{
                "date": (date or pd.Timestamp.today().date()),
                "equity": round(self.portfolio.equity, 2),
                "cash": round(self.portfolio.cash, 2),
                "market_value": round(self.portfolio.market_value, 2),
                "positions": sum(1 for p in self.portfolio.positions.values() if p.shares > 0),
            }]
        )
        header = not self.equity_path.is_file()
        row.to_csv(self.equity_path, mode="a", header=header, index=False, encoding="utf-8-sig")
        self.save()
        return row

    # ------------------------------------------------------------ 展示
    def status(self) -> str:
        pf = self.portfolio
        lines = [
            f"现金     : {pf.cash:,.2f}",
            f"市值     : {pf.market_value:,.2f}",
            f"总权益   : {pf.equity:,.2f}",
            f"已实现盈亏: {pf.realized_pnl:,.2f}",
            f"累计费用 : {pf.total_fees:,.2f}",
            "持仓     :",
        ]
        held = [p for p in pf.positions.values() if p.shares > 0]
        if not held:
            lines.append("  （空仓）")
        for pos in held:
            weight = pos.market_value / pf.equity if pf.equity else 0.0
            lines.append(
                f"  {pos.symbol}: {pos.shares}股 成本{pos.cost_basis:.3f} 现价{pos.last_price:.3f} "
                f"浮盈{pos.unrealized_pnl:,.2f} 占比{weight:.2%} 可卖{pos.available}"
            )
        return "\n".join(lines)
