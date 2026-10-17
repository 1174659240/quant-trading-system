"""本地模拟通道：委托立即按给定限价成交，走本地账本，不碰任何外部系统。

用途是"同一套调度把模拟跑通"：策略信号、风控闸门、订单生成、账本更新、对账
全部走真实路径，只有最后的成交是本地记账。切到真实通道时上层代码不用改。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

from quant.config import CostModel
from quant.gateway.base import (
    GatewayAccount,
    GatewayFill,
    GatewayOrder,
    OrderGateway,
    SubmitResult,
)

__all__ = ["PaperGateway"]

_ROOT = Path(__file__).resolve().parents[2]


class PaperGateway(OrderGateway):
    """立即成交的本地模拟通道。"""

    name = "paper"
    supports_auto_submit = True

    def __init__(self, config, *, costs: CostModel | None = None) -> None:
        super().__init__(dry_run=getattr(config, "dry_run", True))
        self.config = config
        self.costs = costs or CostModel()
        ledger = Path(getattr(config, "ledger_path", "data/trade/ledger.json"))
        self.ledger_path = ledger if ledger.is_absolute() else _ROOT / ledger
        state_dir = Path(getattr(config, "state_dir", "data/orders"))
        self.state_dir = state_dir if state_dir.is_absolute() else _ROOT / state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.last_fills: list[GatewayFill] = []

    def _ledger(self):
        from quant.execution.ledger import Ledger

        return Ledger.load(self.ledger_path)

    def query_account(self) -> GatewayAccount:
        ledger = self._ledger()
        return GatewayAccount(
            cash=ledger.cash,
            available_cash=ledger.cash,
            market_value=ledger.market_value(),
            positions=ledger.available_shares(),
        )

    def submit(self, orders: Sequence[GatewayOrder], *, confirm: bool = False) -> SubmitResult:
        self._require_connected()
        orders = [o for o in orders if int(o.shares) > 0]
        if self.dry_run or not confirm:
            return SubmitResult(accepted=list(orders), dry_run=True, channel=self.name)

        ledger = self._ledger()
        fills: list[GatewayFill] = []
        for order in orders:
            price = float(order.price or 0.0)
            if price <= 0:
                raise ValueError(f"{order.symbol} 缺限价，模拟通道不能按市价成交")
            fee = self.costs.buy_cost(price, order.shares) if order.side == "buy" else self.costs.sell_cost(price, order.shares)
            fill = GatewayFill(
                symbol=order.symbol,
                side=order.side,
                shares=int(order.shares),
                price=price,
                fee=round(fee, 2),
                client_id=order.client_id,
            )
            ledger.apply_fill(fill)
            fills.append(fill)
        ledger.save()
        self.last_fills = fills
        return SubmitResult(accepted=list(orders), dry_run=False, channel=self.name, artifacts=[self.ledger_path])

    def fetch_fills(self, day: date | str | None = None) -> list[GatewayFill]:
        return list(self.last_fills)

    def describe(self) -> dict:
        return {**super().describe(), "ledger": str(self.ledger_path), "costs": self.costs.__class__.__name__}
