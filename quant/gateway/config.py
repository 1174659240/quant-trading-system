"""通道配置与工厂：``[gateway]`` / ``[guard]`` 两段配置决定"下单怎么走、闸门多紧"。"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from quant.gateway.base import GatewayError, OrderGateway

if TYPE_CHECKING:  # pragma: no cover
    from quant.execution.guard import TradeLimits

__all__ = [
    "DEFAULT_LIMITS",
    "GatewayConfig",
    "TradeLimits",
    "build_gateway",
    "load_gateway_config",
    "load_guard_limits",
]

# 订单级限额的**唯一默认值来源**在 quant.execution.guard.TradeLimits；
# 这里导出成字典，供 doctor 与文档展示（避免两处数字打架）。
def _default_limits() -> dict[str, Any]:
    from quant.execution.guard import TradeLimits

    return asdict(TradeLimits())


DEFAULT_LIMITS: dict[str, Any] = _default_limits()


@dataclass(frozen=True)
class GatewayConfig:
    """``[gateway]`` 段。

    默认就是"最安全"的一档：半自动订单文件 + 预演模式。
    要变成真正下单，必须同时改 ``name``（换成 qmt/paper）与 ``dry_run``（关掉），
    而且在命令行还得再给一个 ``--yes``。
    """

    name: str = "csv"
    dry_run: bool = True
    state_dir: str = "data/orders"
    """订单文件与台账目录。"""

    ledger_path: str = "data/trade/ledger.json"
    """本地账本（持仓/现金的事实来源，靠对账与券商保持一致）。"""

    initial_cash: float = 500_000.0
    """首次运行时账本的初始现金（元）；之后以账本里的现金为准。"""

    confirm_token: str = ""
    """可选：自动化场景下的确认口令（留空则只能用命令行 ``--yes``）。"""

    qmt_userdata: str = ""
    """迅投 QMT 的 ``userdata_mini`` 目录（走 QMT 通道时必填）。"""

    qmt_account: str = ""
    """资金账号。"""

    qmt_account_type: str = "STOCK"
    qmt_session_id: int = 0
    """0 表示自动取一个基于时间的会话号。"""

    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _filter(cls, raw: Mapping[str, Any] | None):
    raw = dict(raw or {})
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{cls.__name__} 不认识的配置项：{sorted(unknown)}")
    return cls(**raw)


def load_gateway_config(path: str | Path | None) -> GatewayConfig:
    """从 TOML 读 ``[gateway]``；文件不存在或缺段时返回默认（半自动 + 预演）。"""
    if path is None:
        return GatewayConfig()
    path = Path(path)
    if not path.is_file():
        return GatewayConfig()
    with path.open("rb") as fh:
        doc = tomllib.load(fh)
    return _filter(GatewayConfig, doc.get("gateway"))


def load_guard_limits(path: str | Path | None) -> "TradeLimits":
    """从 TOML 读 ``[guard]``；缺文件/缺段时返回默认限额。"""
    from quant.execution.guard import TradeLimits, limits_from_mapping

    if path is None:
        return TradeLimits()
    path = Path(path)
    if not path.is_file():
        return TradeLimits()
    with path.open("rb") as fh:
        doc = tomllib.load(fh)
    section = doc.get("guard")
    if not section:
        return TradeLimits()
    return limits_from_mapping(section)


def build_gateway(config: GatewayConfig | None = None, **overrides: Any) -> OrderGateway:
    """按配置造一个通道实例。未知名字直接报错，不做"猜名字"的兜底。"""
    config = config or GatewayConfig()
    if overrides:
        config = replace(config, **overrides)

    name = str(config.name).strip().lower()
    if name in ("csv", "file", "manual", "semi", "半自动"):
        from quant.gateway.csv_gateway import CsvGateway

        return CsvGateway(config)
    if name in ("paper", "sim", "simulate"):
        from quant.gateway.paper_gateway import PaperGateway

        return PaperGateway(config)
    if name in ("qmt", "xtquant", "迅投"):
        from quant.gateway.qmt_gateway import QmtGateway

        return QmtGateway(config)
    raise GatewayError(
        f"未知通道 {config.name!r}；可选：csv（半自动订单文件，默认）、paper（本地模拟）、qmt（迅投 QMT）"
    )
