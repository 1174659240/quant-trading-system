"""下单通道抽象层（gateway）。

把"策略想持有什么"变成"券商能接受的委托"，并承担三件事：

1. **统一接口**：无论背后是订单文件（半自动）、本地模拟，还是 QMT/券商 API，
   上层调度与风控只认同一套 ``OrderGateway``；
2. **默认安全**：``dry_run=True`` 是默认值，任何"真下单"都必须显式关掉它，
   并且必须带 ``confirm=True``（对应命令行的 ``--yes``），避免脚本偷偷下单；
3. **可对账**：每笔委托都留 ``client_id``（幂等键）与落盘凭证，事后能跟券商流水逐条对上。

本模块只依赖标准库。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "GatewayAccount",
    "GatewayError",
    "GatewayFill",
    "GatewayOrder",
    "NotConnected",
    "OrderGateway",
    "SubmitResult",
]


class GatewayError(RuntimeError):
    """通道层错误（连接失败、参数非法、券商拒单等）。"""


class NotConnected(GatewayError):
    """尚未 ``connect()`` 就调用了需要连接的方法。"""


@dataclass(frozen=True)
class GatewayOrder:
    """一笔委托。``client_id`` 是本地幂等键（同一交易日重复提交会被风控拦下）。"""

    symbol: str
    side: str
    """``buy`` 或 ``sell``。"""

    shares: int
    price: float | None = None
    """限价；``None`` 表示用对手价/市价（半自动模式下建议填限价）。"""

    order_type: str = "limit"
    client_id: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        side = str(self.side).lower()
        if side not in ("buy", "sell"):
            raise GatewayError(f"side 只能是 buy/sell，收到 {self.side!r}")
        object.__setattr__(self, "side", side)
        if int(self.shares) <= 0:
            raise GatewayError(f"股数必须为正，收到 {self.shares!r}")

    @property
    def value(self) -> float:
        """按限价估算的成交金额（市价单用 0，风控会另行用最新价估算）。"""
        return float(self.price or 0.0) * int(self.shares)

    @property
    def key(self) -> str:
        """幂等键：同标的、同方向、同股数、同价格视为同一笔。"""
        price = "" if self.price is None else f"{float(self.price):.3f}"
        return f"{self.symbol}|{self.side}|{int(self.shares)}|{price}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["value"] = round(self.value, 2)
        return data


@dataclass
class GatewayFill:
    """一笔成交回报（券商流水或人工录入）。"""

    symbol: str
    side: str
    shares: int
    price: float
    fee: float = 0.0
    order_id: str = ""
    client_id: str = ""
    time: str = ""

    @property
    def amount(self) -> float:
        return float(self.price) * int(self.shares)


@dataclass
class GatewayAccount:
    """账户快照（现金/可卖/持仓）。"""

    cash: float = 0.0
    available_cash: float = 0.0
    market_value: float = 0.0
    frozen: float = 0.0
    positions: dict[str, int] = field(default_factory=dict)
    """标的 → **可卖**股数。"""

    def to_dict(self) -> dict:
        return {
            "cash": round(self.cash, 2),
            "availableCash": round(self.available_cash, 2),
            "marketValue": round(self.market_value, 2),
            "frozen": round(self.frozen, 2),
            "totalEquity": round(self.cash + self.market_value, 2),
            "positions": dict(self.positions),
        }


@dataclass
class SubmitResult:
    """一次提交的结果。"""

    accepted: list[GatewayOrder] = field(default_factory=list)
    rejected: list[tuple[GatewayOrder, str]] = field(default_factory=list)
    dry_run: bool = True
    artifacts: list[Path] = field(default_factory=list)
    channel: str = ""

    @property
    def n_accepted(self) -> int:
        return len(self.accepted)

    def summary(self) -> str:
        head = "预演（未提交）" if self.dry_run else "已提交"
        text = f"{head}：{len(self.accepted)} 笔通过，{len(self.rejected)} 笔被拦"
        if self.rejected:
            text += "；拦截原因：" + "；".join(f"{o.symbol} {reason}" for o, reason in self.rejected[:5])
        return text


class OrderGateway(ABC):
    """通道接口。实现类只需关心"怎么把委托送出去"与"怎么读回账户/成交"。"""

    name = "base"
    supports_auto_submit = False
    """``False`` 表示需要人工落地（半自动）；``True`` 表示能程序化直连券商。"""

    def __init__(self, *, dry_run: bool = True) -> None:
        self.dry_run = bool(dry_run)
        self._connected = False

    # -- 生命周期 -----------------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise NotConnected(f"{self.name} 通道尚未 connect()")

    # -- 查询 ---------------------------------------------------------------

    @abstractmethod
    def query_account(self) -> GatewayAccount:
        """返回账户快照（半自动模式下读本地账本，真实通道读券商）。"""

    def fetch_fills(self, day: date | str | None = None) -> list[GatewayFill]:
        """拉取成交回报；不支持拉取的通道返回空列表（改由人工录入）。"""
        return []

    # -- 交易 ---------------------------------------------------------------

    @abstractmethod
    def submit(self, orders: Sequence[GatewayOrder], *, confirm: bool = False) -> SubmitResult:
        """提交委托。

        ``confirm=False`` 时必须走预演（dry-run），不得产生真实委托。
        """

    def cancel_all(self) -> int:
        """撤掉当日全部未成交委托，返回撤单数。不支持则返回 0。"""
        return 0

    # -- 描述 ---------------------------------------------------------------

    def describe(self) -> dict:
        return {
            "name": self.name,
            "autoSubmit": self.supports_auto_submit,
            "dryRun": self.dry_run,
            "connected": self._connected,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试友好
        return f"<{type(self).__name__} name={self.name} dry_run={self.dry_run}>"


def dedupe(orders: Iterable[GatewayOrder]) -> list[GatewayOrder]:
    """同一批里按键去重（保留首次出现），并丢掉 0 股委托。"""
    seen: set[str] = set()
    out: list[GatewayOrder] = []
    for order in orders:
        if int(order.shares) <= 0:
            continue
        if order.key in seen:
            continue
        seen.add(order.key)
        out.append(order)
    return out
