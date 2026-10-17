"""半自动通道：把委托写成**人能照着下单**的订单文件，再把成交回报读回来。

这是本工程默认的下单方式，理由是它在"能自己下单"和"不会偷偷下单"之间取平衡：

* 产物是 ``data/orders/<日期>-orders.csv``（带限价、股数、金额、预估费用、下单理由），
  你在券商 App/客户端里照着敲一遍即可；
* 委托台账 ``<日期>-orders.jsonl`` 记下幂等键与提交时间，供风控查重与事后对账；
* 成交后用 ``<日期>-fills.csv`` 回报（模板自动生成），账本随之更新。

它不接触券商账号、不需要任何 SDK、不需要本机常驻客户端，因此也最不容易出事故。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from quant.gateway.base import (
    GatewayAccount,
    GatewayError,
    GatewayFill,
    GatewayOrder,
    OrderGateway,
    SubmitResult,
)

__all__ = ["CsvGateway"]

_ROOT = Path(__file__).resolve().parents[2]

_ORDER_FIELDS = ("symbol", "side", "shares", "price", "amount", "est_fee", "orderType", "clientId", "note")
_FILL_FIELDS = ("symbol", "side", "shares", "price", "fee", "time", "orderId", "clientId", "note")


def _today(day: date | str | None = None) -> str:
    if day is None:
        return date.today().isoformat()
    if isinstance(day, date):
        return day.isoformat()
    return str(day)


class CsvGateway(OrderGateway):
    """订单文件通道（半自动）。"""

    name = "csv"
    supports_auto_submit = False

    def __init__(self, config) -> None:
        super().__init__(dry_run=getattr(config, "dry_run", True))
        self.config = config
        state_dir = Path(getattr(config, "state_dir", "data/orders"))
        self.state_dir = state_dir if state_dir.is_absolute() else _ROOT / state_dir
        ledger = Path(getattr(config, "ledger_path", "data/trade/ledger.json"))
        self.ledger_path = ledger if ledger.is_absolute() else _ROOT / ledger
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # -- 路径 ---------------------------------------------------------------

    def day_path(self, day: date | str | None, kind: str, *, suffix: str = "csv") -> Path:
        return self.state_dir / f"{_today(day)}-{kind}.{suffix}"

    # -- 账户 ---------------------------------------------------------------

    def query_account(self) -> GatewayAccount:
        from quant.execution.ledger import Ledger

        ledger = Ledger.load(self.ledger_path)
        return GatewayAccount(
            cash=ledger.cash,
            available_cash=ledger.cash,
            market_value=ledger.market_value(),
            positions=ledger.available_shares(),
        )

    # -- 提交 ---------------------------------------------------------------

    def submit(self, orders: Sequence[GatewayOrder], *, confirm: bool = False) -> SubmitResult:
        self._require_connected()
        orders = [o for o in orders if int(o.shares) > 0]
        if not orders:
            return SubmitResult(dry_run=True, channel=self.name)

        preview = self.dry_run or not confirm
        kind = "orders.preview" if preview else "orders"
        csv_path = self.day_path(None, kind)
        self._write_orders_csv(csv_path, orders)

        artifacts = [csv_path]
        if not preview:
            jsonl = self.day_path(None, "orders", suffix="jsonl")
            with jsonl.open("a", encoding="utf-8") as fh:
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for order in orders:
                    fh.write(json.dumps({**order.to_dict(), "submittedAt": stamp}, ensure_ascii=False) + "\n")
            artifacts.append(jsonl)

        return SubmitResult(
            accepted=list(orders),
            dry_run=preview,
            artifacts=artifacts,
            channel=self.name,
        )

    def _write_orders_csv(self, path: Path, orders: Sequence[GatewayOrder]) -> None:
        from quant.config import CostModel

        costs = CostModel()
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(_ORDER_FIELDS))
            writer.writeheader()
            for order in orders:
                price = float(order.price or 0.0)
                amount = price * int(order.shares)
                fee = costs.buy_cost(price, order.shares) if order.side == "buy" else costs.sell_cost(price, order.shares)
                writer.writerow(
                    {
                        "symbol": order.symbol,
                        "side": "买入" if order.side == "buy" else "卖出",
                        "shares": order.shares,
                        "price": f"{price:.2f}",
                        "amount": f"{amount:.2f}",
                        "est_fee": f"{fee:.2f}",
                        "orderType": order.order_type,
                        "clientId": order.client_id,
                        "note": order.note,
                    }
                )

    # -- 读回 ---------------------------------------------------------------

    def read_orders(self, day: date | str | None = None, *, kind: str = "orders.preview") -> list[GatewayOrder]:
        """把订单文件读回成委托对象。

        半自动流程的关键一环：``plan`` 写 ``<日期>-orders.preview.csv``，你可以手工删改，
        ``confirm`` 再把这个文件读回来提交——即"最终下什么单"由文件决定，不由程序记性决定。
        """
        path = self.day_path(day, kind)
        if not path.is_file():
            return []
        orders: list[GatewayOrder] = []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                symbol = (row.get("symbol") or "").strip()
                side_raw = (row.get("side") or "").strip().lower()
                if not symbol or side_raw not in ("buy", "sell", "买入", "卖出"):
                    continue
                try:
                    shares = int(float(row.get("shares") or 0))
                    price_raw = (row.get("price") or "").strip()
                    price = float(price_raw) if price_raw else None
                except ValueError:
                    continue
                if shares <= 0:
                    continue
                order_type = ((row.get("orderType") or "limit").strip() or "limit")
                orders.append(
                    GatewayOrder(
                        symbol=symbol,
                        side="buy" if side_raw in ("buy", "买入") else "sell",
                        shares=shares,
                        price=price,
                        order_type=order_type,
                        client_id=(row.get("clientId") or "").strip(),
                        note=(row.get("note") or "").strip(),
                    )
                )
        return orders

    # -- 回报 ---------------------------------------------------------------

    def fetch_fills(self, day: date | str | None = None) -> list[GatewayFill]:
        path = self.day_path(day, "fills")
        if not path.is_file():
            return []
        fills: list[GatewayFill] = []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                symbol = (row.get("symbol") or "").strip()
                side = (row.get("side") or "").strip().lower()
                if not symbol or side not in ("buy", "sell", "买入", "卖出"):
                    continue
                try:
                    shares = int(float(row.get("shares") or 0))
                    price = float(row.get("price") or 0)
                except ValueError:
                    continue
                if shares == 0 or price <= 0:
                    continue
                fills.append(
                    GatewayFill(
                        symbol=symbol,
                        side="buy" if side in ("buy", "买入") else "sell",
                        shares=abs(shares),
                        price=price,
                        fee=float(row.get("fee") or 0.0),
                        order_id=(row.get("orderId") or "").strip(),
                        client_id=(row.get("clientId") or "").strip(),
                        time=(row.get("time") or "").strip(),
                    )
                )
        return fills

    def write_fill_template(self, orders: Sequence[GatewayOrder], day: date | str | None = None) -> Path:
        """按建议单生成回报模板（价格/费用留空，成交后填）。"""
        path = self.day_path(day, "fills")
        existing = self.fetch_fills(day)
        done = {(f.symbol, f.side) for f in existing}
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(_FILL_FIELDS))
            writer.writeheader()
            for fill in existing:
                writer.writerow(
                    {
                        "symbol": fill.symbol,
                        "side": "买入" if fill.side == "buy" else "卖出",
                        "shares": fill.shares,
                        "price": f"{fill.price:.2f}",
                        "fee": f"{fill.fee:.2f}",
                        "time": fill.time,
                        "orderId": fill.order_id,
                        "clientId": fill.client_id,
                        "note": "已录入",
                    }
                )
            for order in orders:
                if (order.symbol, order.side) in done:
                    continue
                writer.writerow(
                    {
                        "symbol": order.symbol,
                        "side": "买入" if order.side == "buy" else "卖出",
                        "shares": order.shares,
                        "price": "",
                        "fee": "",
                        "time": "",
                        "orderId": "",
                        "clientId": order.client_id,
                        "note": "成交后填成交价（部分成交就改股数，其余可删）",
                    }
                )
        return path

    # -- 台账 ---------------------------------------------------------------

    def submitted_orders(self, day: date | str | None = None) -> list[dict]:
        path = self.day_path(day, "orders", suffix="jsonl")
        if not path.is_file():
            return []
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def cancel_all(self) -> int:
        """半自动模式没有"未成交委托"可撤；把当日建议单标记作废即可。"""
        path = self.day_path(None, "orders.preview")
        if path.is_file():
            path.rename(path.with_suffix(".cancelled.csv"))
            return 1
        return 0

    def describe(self) -> dict:
        return {
            **super().describe(),
            "stateDir": str(self.state_dir),
            "ledger": str(self.ledger_path),
            "usage": "生成订单文件 → 券商客户端手工下单 → 用 fills.csv 回报成交 → reconcile 对账",
        }


def order_rows(orders: Sequence[GatewayOrder]) -> list[dict]:
    """把委托转成表格行（供报告/CSV 复用）。"""
    return [{**asdict(o), "value": round(o.value, 2)} for o in orders]


def ensure_order_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():  # pragma: no cover - 目录被同名文件占用
        raise GatewayError(f"不是目录：{path}")
    return path
