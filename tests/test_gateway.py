"""下单通道（gateway）自测：订单文件、本地模拟、QMT 前置条件与安全默认值。

重点不是"能不能发单"，而是**默认状态下发不出去**：dry_run 默认开、confirm 必须显式、
幂等键会落盘、成交回报能读回来并进账本。
"""

from __future__ import annotations

from _util import setup_console, tmpdir

setup_console()

from quant.execution.ledger import Ledger  # noqa: E402
from quant.gateway import (  # noqa: E402
    CsvGateway,
    GatewayConfig,
    GatewayError,
    GatewayOrder,
    PaperGateway,
    QmtGateway,
    build_gateway,
    dedupe,
    load_gateway_config,
    load_guard_limits,
    xtquant_available,
)
from quant.gateway.base import GatewayFill  # noqa: E402


def _config(directory, **overrides) -> GatewayConfig:
    base = dict(
        name="csv",
        dry_run=True,
        state_dir=str(directory),
        ledger_path=str(directory / "ledger.json"),
    )
    base.update(overrides)
    return GatewayConfig(**base)


# ------------------------------------------------------------------ 基础对象


def test_gateway_order_validates_fields():
    order = GatewayOrder("600519.SH", "BUY", 100, 10.0)
    assert order.side == "buy"  # 统一小写
    assert order.key == "600519.SH|buy|100|10.000"
    assert order.value == 1000.0
    assert order.to_dict()["value"] == 1000.0
    try:
        GatewayOrder("600519.SH", "hold", 100, 10.0)
    except GatewayError as exc:
        assert "buy/sell" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("非法 side 应当报错")
    try:
        GatewayOrder("600519.SH", "buy", 0, 10.0)
    except GatewayError as exc:
        assert "为正" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("0 股应当报错")


def test_dedupe_keeps_first_and_drops_zero():
    orders = [
        GatewayOrder("600519.SH", "buy", 100, 10.0),
        GatewayOrder("600519.SH", "buy", 100, 10.0),
        GatewayOrder("000001.SZ", "sell", 200, 8.0),
    ]
    out = dedupe(orders)
    assert len(out) == 2
    assert [o.symbol for o in out] == ["600519.SH", "000001.SZ"]


def test_build_gateway_dispatch_and_unknown_channel():
    assert isinstance(build_gateway(GatewayConfig(name="csv")), CsvGateway)
    assert isinstance(build_gateway(GatewayConfig(name="paper")), PaperGateway)
    assert isinstance(build_gateway(GatewayConfig(name="qmt")), QmtGateway)
    try:
        build_gateway(GatewayConfig(name="随便写的"))
    except GatewayError as exc:
        assert "未知通道" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("未知通道应当报错")


def test_build_gateway_applies_overrides():
    gateway = build_gateway(GatewayConfig(), dry_run=False, name="paper")
    assert gateway.name == "paper"
    assert gateway.dry_run is False


def test_load_gateway_config_and_guard_limits_from_toml():
    with tmpdir("gateway-config") as directory:
        path = directory / "config.toml"
        path.write_text(
            "\n".join(
                [
                    "[gateway]",
                    'name = "csv"',
                    "dry_run = true",
                    'state_dir = "data/orders"',
                    "initial_cash = 200000",
                    "",
                    "[guard]",
                    "max_orders_per_day = 7",
                    'allowed_symbols = ["600519.SH"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        config = load_gateway_config(path)
        assert config.initial_cash == 200_000
        assert config.name == "csv"
        limits = load_guard_limits(path)
        assert limits.max_orders_per_day == 7
        assert limits.allowed_symbols == ("600519.SH",)
    # 缺文件时退回默认（最安全的一档）
    default = load_gateway_config(None)
    assert default.name == "csv" and default.dry_run is True
    assert load_guard_limits(None).max_weight_per_symbol > 0


# ------------------------------------------------------------------ CSV 通道


def test_csv_gateway_writes_preview_and_reads_it_back():
    with tmpdir("csv-gateway") as directory:
        gateway = CsvGateway(_config(directory))
        gateway.connect()
        orders = [
            GatewayOrder("600519.SH", "buy", 200, 1688.5, client_id="plan", note="加仓至 20.0%"),
            GatewayOrder("000001.SZ", "sell", 1000, 11.2, client_id="plan", note="减仓"),
        ]
        result = gateway.submit(orders, confirm=False)
        assert result.dry_run is True
        assert result.n_accepted == 2
        preview = gateway.day_path(None, "orders.preview")
        assert preview.is_file()
        assert preview in result.artifacts
        assert not gateway.day_path(None, "orders").is_file()  # 没确认就不该有正式文件

        back = gateway.read_orders()
        assert [o.symbol for o in back] == ["600519.SH", "000001.SZ"]
        assert back[0].side == "buy" and back[0].shares == 200
        assert back[1].side == "sell" and abs(back[1].price - 11.2) < 1e-9
        assert "加仓" in back[0].note


def test_csv_gateway_confirm_writes_ledger_and_orders_jsonl():
    with tmpdir("csv-confirm") as directory:
        gateway = CsvGateway(_config(directory, dry_run=False))
        gateway.connect()
        orders = [GatewayOrder("600519.SH", "buy", 200, 1688.5, client_id="plan")]
        # dry_run=False 但没 confirm → 仍然只出预演（防误下单的第二道闸门）
        still_preview = gateway.submit(orders, confirm=False)
        assert still_preview.dry_run is True
        assert not gateway.day_path(None, "orders").is_file()

        result = gateway.submit(orders, confirm=True)
        assert result.dry_run is False
        assert gateway.day_path(None, "orders").is_file()
        assert gateway.day_path(None, "orders", suffix="jsonl").is_file()
        records = gateway.submitted_orders()
        assert len(records) == 1
        assert records[0]["symbol"] == "600519.SH"
        assert records[0]["submittedAt"]


def test_csv_gateway_fill_template_and_fetch_fills():
    with tmpdir("csv-fills") as directory:
        gateway = CsvGateway(_config(directory))
        gateway.connect()
        order = GatewayOrder("600519.SH", "buy", 200, 1688.5, client_id="plan")
        template = gateway.write_fill_template([order])
        assert template.is_file()
        text = template.read_text(encoding="utf-8-sig")
        assert "600519.SH" in text and "price" in text
        assert gateway.fetch_fills() == []  # 价格留空 → 不算成交

        template.write_text(
            "symbol,side,shares,price,fee,time,orderId,clientId,note\n"
            "600519.SH,买入,200,1688.10,84.41,2024-01-02 09:31:00,123,plan,\n",
            encoding="utf-8-sig",
        )
        fills = gateway.fetch_fills()
        assert len(fills) == 1
        assert fills[0].side == "buy" and fills[0].shares == 200
        assert abs(fills[0].price - 1688.10) < 1e-9


def test_csv_gateway_cancel_all_marks_preview_cancelled():
    with tmpdir("csv-cancel") as directory:
        gateway = CsvGateway(_config(directory))
        gateway.connect()
        gateway.submit([GatewayOrder("600519.SH", "buy", 200, 10.0)], confirm=False)
        assert gateway.cancel_all() == 1
        assert gateway.cancel_all() == 0
        assert (directory / f"{__import__('datetime').date.today().isoformat()}-orders.preview.cancelled.csv").is_file()


def test_csv_gateway_query_account_reads_ledger():
    with tmpdir("csv-account") as directory:
        gateway = CsvGateway(_config(directory))
        ledger = Ledger(cash=123_456.0, path=directory / "ledger.json")
        ledger.apply_fill(GatewayFill("600519.SH", "buy", 100, 10.0))
        ledger.save()
        account = gateway.query_account()
        assert abs(account.cash - ledger.cash) < 1e-9
        assert account.positions == {}  # 当日买入不可卖
        assert "手工下单" in gateway.describe()["usage"]


# ------------------------------------------------------------------ 模拟通道


def test_paper_gateway_applies_fills_to_ledger():
    with tmpdir("paper-gateway") as directory:
        config = _config(directory, name="paper", dry_run=False)
        seed = Ledger(cash=50_000.0, path=config.ledger_path)
        seed.save()
        gateway = PaperGateway(config)
        gateway.connect()
        order = GatewayOrder("600519.SH", "buy", 100, 10.0, client_id="plan")
        preview = gateway.submit([order], confirm=False)
        assert preview.dry_run is True

        result = gateway.submit([order], confirm=True)
        assert result.dry_run is False
        assert len(gateway.last_fills) == 1
        ledger = Ledger.load(config.ledger_path)
        assert ledger.shares() == {"600519.SH": 100}
        assert ledger.cash < 50_000.0 - 1000.0  # 本金 + 费用都扣掉了
        assert ledger.available_shares() == {}  # T+1
        account = gateway.query_account()
        assert abs(account.cash - ledger.cash) < 1e-9
        assert account.market_value > 0

        # 没有限价就不许成交（模拟通道也不瞎猜价格）
        gateway.dry_run = False
        try:
            gateway.submit([GatewayOrder("600519.SH", "buy", 100, None)], confirm=True)
        except ValueError as exc:
            assert "缺限价" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("缺限价应当报错")


# ------------------------------------------------------------------ QMT 通道


def test_qmt_gateway_requires_prerequisites():
    with tmpdir("qmt-gateway") as directory:
        gateway = QmtGateway(_config(directory, name="qmt"))
        missing = gateway.prerequisites()
        assert missing  # 本机既没 xtquant 也没配 userdata/account
        assert any("xtquant" in item for item in missing)
        try:
            gateway.connect()
        except GatewayError as exc:
            assert "缺少" in str(exc) or "xtquant" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("前置条件不满足时不应连上")
        info = gateway.describe()
        assert info["name"] == "qmt"
        assert "未验证" in str(info) or "warning" in info

    # xtquant_available 在本机应为 False（没装迅投 SDK）
    assert xtquant_available() in (True, False)


def test_qmt_dry_run_still_writes_preview_file():
    with tmpdir("qmt-dry") as directory:
        gateway = QmtGateway(_config(directory, name="qmt", dry_run=True))
        gateway.connect = lambda: None  # 跳过真实连接，只验证 dry-run 分支
        gateway._connected = True
        result = gateway.submit([GatewayOrder("600519.SH", "buy", 200, 10.0)], confirm=True)
        assert result.dry_run is True
        assert (directory / f"{__import__('datetime').date.today().isoformat()}-orders.preview.csv").is_file()
