"""迅投 QMT 通道（券商版客户端 + ``xtquant``）。

**当前状态：适配器已写好，但本机没有 QMT 环境，所以它从未在真实账户上跑过。**
代码路径都做了三重保护，装好 QMT 后先跑 ``--dry-run`` 验证，再真下单：

1. 默认 ``dry_run=True``：只产出建议单文件，不调任何交易接口；
2. 提交必须 ``confirm=True``（命令行 ``--yes``），且必须已 ``connect()``；
3. 逐单检查 ``必填配置``（userdata_mini 目录、资金账号），缺一项就报错而不是猜。

依赖：券商开通的 QMT 权限（多数券商门槛 50 万或协商）+ 常驻的极速交易客户端 +
客户端自带的 ``xtquant`` 包（在 ``bin.x64/Lib/site-packages`` 下，需自行加入 ``PYTHONPATH``）。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Sequence

from quant.config import CostModel
from quant.gateway.base import (
    GatewayAccount,
    GatewayError,
    GatewayFill,
    GatewayOrder,
    NotConnected,
    OrderGateway,
    SubmitResult,
)

__all__ = ["QmtGateway"]

_ROOT = Path(__file__).resolve().parents[2]

_SIDE_TO_XT = {"buy": "STOCK_BUY", "sell": "STOCK_SELL"}
_PRICE_TYPE_LIMIT = "FIX_PRICE"
_PRICE_TYPE_MARKET = "MARKET_PEER_PRICE_FIRST"


def xtquant_available() -> bool:
    """本机能否 ``import xtquant``。"""
    import importlib.util

    try:
        return importlib.util.find_spec("xtquant") is not None
    except Exception:  # noqa: BLE001
        return False


class QmtGateway(OrderGateway):
    """迅投 QMT 交易通道。"""

    name = "qmt"
    supports_auto_submit = True

    def __init__(self, config, *, costs: CostModel | None = None) -> None:
        super().__init__(dry_run=getattr(config, "dry_run", True))
        self.config = config
        self.costs = costs or CostModel()
        state_dir = Path(getattr(config, "state_dir", "data/orders"))
        self.state_dir = state_dir if state_dir.is_absolute() else _ROOT / state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.userdata = str(getattr(config, "qmt_userdata", "") or "")
        self.account_id = str(getattr(config, "qmt_account", "") or "")
        self.account_type = str(getattr(config, "qmt_account_type", "STOCK") or "STOCK")
        self._trader: Any = None
        self._account: Any = None
        self._xtconstant: Any = None
        self._last_error = ""

    # -- 前置条件 -----------------------------------------------------------

    def prerequisites(self) -> list[str]:
        """返回缺失的前置条件（空列表 = 齐了）。doctor 会用。"""
        problems: list[str] = []
        if not xtquant_available():
            problems.append("未安装 xtquant（券商 QMT 客户端自带，需把 bin.x64/Lib/site-packages 加进 PYTHONPATH）")
        if not self.userdata:
            problems.append("未配置 qmt_userdata（QMT 的 userdata_mini 目录）")
        elif not Path(self.userdata).exists():
            problems.append(f"qmt_userdata 路径不存在：{self.userdata}")
        if not self.account_id:
            problems.append("未配置 qmt_account（资金账号）")
        return problems

    # -- 生命周期 -----------------------------------------------------------

    def connect(self) -> None:
        problems = self.prerequisites()
        if problems:
            raise GatewayError("QMT 通道前置条件未满足：" + "；".join(problems))
        from xtquant import xtconstant  # type: ignore
        from xtquant.xttrader import XtQuantTrader  # type: ignore
        from xtquant.xttype import StockAccount  # type: ignore

        session_id = int(getattr(self.config, "qmt_session_id", 0) or 0)
        if session_id <= 0:
            import time as _time

            session_id = int(_time.time()) % 2_000_000_000
        trader = XtQuantTrader(self.userdata, session_id)
        trader.start()
        result = trader.connect()
        if result != 0:
            raise GatewayError(f"QMT connect() 返回 {result}（0 才是成功）；确认客户端已登录且账号有交易权限")
        account = StockAccount(self.account_id, self.account_type)
        if trader.subscribe(account) != 0:
            raise GatewayError("QMT subscribe(account) 失败；确认资金账号与账户类型正确")
        self._trader, self._account, self._xtconstant = trader, account, xtconstant
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        self._trader = None
        self._account = None

    # -- 查询 ---------------------------------------------------------------

    def query_account(self) -> GatewayAccount:
        if self._trader is None:
            raise NotConnected("QMT 未连接")
        asset = self._trader.query_stock_asset(self._account)
        positions = {}
        for pos in self._trader.query_stock_positions(self._account) or []:
            positions[str(pos.stock_code)] = int(getattr(pos, "can_use_volume", 0) or 0)
        return GatewayAccount(
            cash=float(getattr(asset, "cash", 0.0) or 0.0),
            available_cash=float(getattr(asset, "cash", 0.0) or 0.0)
            - float(getattr(asset, "frozen_cash", 0.0) or 0.0),
            market_value=float(getattr(asset, "market_value", 0.0) or 0.0),
            frozen=float(getattr(asset, "frozen_cash", 0.0) or 0.0),
            positions=positions,
        )

    def fetch_fills(self, day: date | str | None = None) -> list[GatewayFill]:
        if self._trader is None:
            raise NotConnected("QMT 未连接")
        fills: list[GatewayFill] = []
        for trade in self._trader.query_stock_trades(self._account) or []:
            side = "buy" if getattr(trade, "order_type", 0) == self._xtconstant.STOCK_BUY else "sell"
            traded = int(getattr(trade, "traded_volume", 0) or 0)
            if traded <= 0:
                continue
            fills.append(
                GatewayFill(
                    symbol=str(trade.stock_code),
                    side=side,
                    shares=traded,
                    price=float(getattr(trade, "traded_price", 0.0) or 0.0),
                    fee=float(getattr(trade, "traded_amount", 0.0) or 0.0) * 0.0,  # 费用以券商流水为准
                    order_id=str(getattr(trade, "order_id", "") or ""),
                    time=str(getattr(trade, "traded_time", "") or ""),
                )
            )
        return fills

    # -- 交易 ---------------------------------------------------------------

    def submit(self, orders: Sequence[GatewayOrder], *, confirm: bool = False) -> SubmitResult:
        self._require_connected()
        orders = [o for o in orders if int(o.shares) > 0]
        if self.dry_run or not confirm:
            from quant.gateway.csv_gateway import CsvGateway

            preview_channel = CsvGateway(self.config)
            preview_channel.connect()
            result = preview_channel.submit(orders, confirm=False)
            result.channel = f"{self.name}(preview)"
            return result

        accepted: list[GatewayOrder] = []
        rejected: list[tuple[GatewayOrder, str]] = []
        artifacts: list[Path] = []
        for order in orders:
            side = _SIDE_TO_XT[order.side]
            price_type = _PRICE_TYPE_LIMIT if order.price else _PRICE_TYPE_MARKET
            price = float(order.price or 0.0)
            order_id = self._trader.order_stock(
                self._account,
                order.symbol,
                getattr(self._xtconstant, side),
                int(order.shares),
                getattr(self._xtconstant, price_type),
                price,
                "ashare-quant",
                order.note or order.client_id or "ashare-quant",
            )
            if order_id and order_id > 0:
                accepted.append(order)
            else:
                self._last_error = f"order_stock 返回 {order_id}"
                rejected.append((order, self._last_error))
        return SubmitResult(accepted=accepted, rejected=rejected, dry_run=False, channel=self.name, artifacts=artifacts)

    def cancel_all(self) -> int:
        if self._trader is None:
            raise NotConnected("QMT 未连接")
        count = 0
        for pending in self._trader.query_stock_orders(self._account) or []:
            status = getattr(pending, "order_status", None)
            if status in (48, 49, 50, 55, 56):  # 已撤/已成/废单等终态
                continue
            if self._trader.cancel_order_stock(self._account, getattr(pending, "order_id", 0)) == 0:
                count += 1
        return count

    def describe(self) -> dict:
        return {
            **super().describe(),
            "userdata": self.userdata or "(未配置)",
            "account": self.account_id or "(未配置)",
            "prerequisites": self.prerequisites(),
            "warning": "本机未验证过真实下单；首次使用务必先 dry-run，再用最小 100 股试单核验回报口径",
        }
