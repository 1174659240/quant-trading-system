"""下单通道层：``csv``（半自动，默认）/ ``paper``（本地模拟）/ ``qmt``（迅投 QMT）。"""

from __future__ import annotations

from quant.gateway.base import (
    GatewayAccount,
    GatewayError,
    GatewayFill,
    GatewayOrder,
    NotConnected,
    OrderGateway,
    SubmitResult,
    dedupe,
)
from quant.gateway.config import (
    DEFAULT_LIMITS,
    GatewayConfig,
    build_gateway,
    load_gateway_config,
    load_guard_limits,
)
from quant.gateway.csv_gateway import CsvGateway
from quant.gateway.paper_gateway import PaperGateway
from quant.gateway.qmt_gateway import QmtGateway, xtquant_available

__all__ = [
    "DEFAULT_LIMITS",
    "CsvGateway",
    "GatewayAccount",
    "GatewayConfig",
    "GatewayError",
    "GatewayFill",
    "GatewayOrder",
    "NotConnected",
    "OrderGateway",
    "PaperGateway",
    "QmtGateway",
    "SubmitResult",
    "build_gateway",
    "dedupe",
    "load_gateway_config",
    "load_guard_limits",
    "xtquant_available",
]
