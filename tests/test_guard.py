"""订单级风控（PreTradeGuard）与账本（Ledger）的自测。

这两块是"能自己下单"背后真正值钱的部分：策略可以换，闸门不能失效。
测试只依赖标准库与工程内的模块，不联网。
"""

from __future__ import annotations

from datetime import date

from _util import setup_console, tmpdir

setup_console()

from quant.config import CostModel, MarketRules  # noqa: E402
from quant.execution.guard import (  # noqa: E402
    GuardContext,
    PreTradeGuard,
    TradeLimits,
    limits_from_mapping,
)
from quant.execution.ledger import Ledger  # noqa: E402
from quant.gateway.base import GatewayFill, GatewayOrder  # noqa: E402


def _context(**overrides) -> GuardContext:
    base = dict(
        cash=100_000.0,
        equity=100_000.0,
        position_value={},
        available={},
        latest={"600519.SH": 10.0},
        prev_close={"600519.SH": 10.0},
        prev_amount={"600519.SH": 500_000_000.0},
    )
    base.update(overrides)
    return GuardContext(**base)


def _guard(**limit_kwargs) -> PreTradeGuard:
    limits = TradeLimits(**limit_kwargs)
    return PreTradeGuard(limits, rules=MarketRules(), costs=CostModel())


def _reasons(batch) -> str:
    return "；".join(reason for _, reason in batch.rejected)


# ---------------------------------------------------------------- 限额解析


def test_limits_from_mapping_parses_and_rejects_unknown():
    limits = limits_from_mapping({"max_orders_per_day": 5, "allowed_symbols": ["600519.SH"]})
    assert limits.max_orders_per_day == 5
    assert limits.allowed_symbols == ("600519.SH",)
    assert limits.max_single_order_value == TradeLimits().max_single_order_value  # 其余走默认
    try:
        limits_from_mapping({"不存在的项": 1})
    except ValueError as exc:
        assert "不认识" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("未知配置项应当报错")


# ---------------------------------------------------------------- 账本


def test_ledger_buy_is_not_sellable_same_day():
    ledger = Ledger(cash=100_000.0)
    ledger.apply_fill(GatewayFill("600519.SH", "buy", 100, 10.0, fee=5.0))
    assert ledger.shares() == {"600519.SH": 100}
    assert ledger.available_shares() == {}  # T+1：当日买入不可卖
    assert abs(ledger.cash - (100_000 - 1000 - 5)) < 1e-9
    unlocked = ledger.settle("2024-01-03")
    assert unlocked == 1
    assert ledger.available_shares() == {"600519.SH": 100}


def test_ledger_sell_realizes_pnl_and_clears_cost():
    ledger = Ledger(cash=100_000.0)
    ledger.apply_fill(GatewayFill("600519.SH", "buy", 200, 10.0))
    ledger.settle("2024-01-03")
    record = ledger.apply_fill(GatewayFill("600519.SH", "sell", 200, 12.0, fee=6.0))
    assert record["realizedPnl"] == (12.0 - 10.0) * 200 - 6.0
    assert ledger.shares() == {}
    assert ledger.positions["600519.SH"].cost == 0.0
    assert abs(ledger.cash - (100_000 - 2000 + 2400 - 6)) < 1e-9


def test_ledger_save_and_load_roundtrip():
    with tmpdir("ledger") as directory:
        path = directory / "ledger.json"
        ledger = Ledger(cash=50_000.0, path=path)
        ledger.apply_fill(GatewayFill("600519.SH", "buy", 100, 10.0, fee=5.0))
        ledger.record_order_keys(["600519.SH|buy|100|10.000"], day="2024-01-02")
        ledger.save()

        back = Ledger.load(path)
        assert back.cash == ledger.cash
        assert back.shares() == {"600519.SH": 100}
        assert back.has_order_key("2024-01-02|600519.SH|buy|100|10.000")
        assert len(back.trades) == 1
        assert back.trades[0]["symbol"] == "600519.SH"


def test_ledger_settle_if_new_day_is_idempotent():
    ledger = Ledger(cash=1000.0)
    ledger.apply_fill(GatewayFill("600519.SH", "buy", 100, 10.0))
    assert ledger.settle_if_new_day("2024-01-03") is True
    assert ledger.settle_if_new_day("2024-01-03") is False
    assert ledger.available_shares() == {"600519.SH": 100}


def test_ledger_valuation_and_weights():
    ledger = Ledger(cash=10_000.0)
    ledger.apply_fill(GatewayFill("600519.SH", "buy", 100, 10.0))
    assert abs(ledger.market_value({"600519.SH": 12.0}) - 1200.0) < 1e-9
    assert abs(ledger.equity({"600519.SH": 12.0}) - (ledger.cash + 1200.0)) < 1e-9
    assert abs(ledger.weights({"600519.SH": 12.0})["600519.SH"] - 1200.0 / ledger.equity()) < 1e-9


def test_ledger_reconcile_reports_differences():
    ledger = Ledger(cash=10_000.0)
    ledger.apply_fill(GatewayFill("600519.SH", "buy", 100, 10.0))
    clean = [{"symbol": "600519.SH", "shares": 100, "available": 0, "cash": ledger.cash}]
    assert ledger.reconcile(clean) == []

    dirty = [{"symbol": "600519.SH", "shares": 200, "available": 100}]
    diffs = ledger.reconcile(dirty)
    fields = {(d["symbol"], d["field"]) for d in diffs}
    assert ("600519.SH", "shares") in fields
    assert ("600519.SH", "available") in fields
    # 券商对账单里没有的持仓 → 也要报出来（可能是券商已清仓）
    missing = ledger.reconcile([{"symbol": "000001.SZ", "shares": 0}])
    assert any(d["note"] for d in missing if d["field"] == "shares")


# ---------------------------------------------------------------- 硬拒绝


def test_guard_kill_switch_blocks_everything():
    batch = _guard().check_batch([GatewayOrder("600519.SH", "buy", 100, 10.0)], _context(kill_switch=True))
    assert not batch.accepted
    assert "急停" in _reasons(batch)


def test_guard_session_only_blocks_out_of_hours():
    guard = _guard(session_only=True)
    batch = guard.check_batch([GatewayOrder("600519.SH", "buy", 100, 10.0)], _context(session_ok=False))
    assert not batch.accepted
    assert "连续竞价" in _reasons(batch)
    # 时段内则放行（股数放大到最小委托额之上）
    ok = guard.check_batch([GatewayOrder("600519.SH", "buy", 1_000, 10.0)], _context(session_ok=True))
    assert len(ok.accepted) == 1
    assert ok.accepted[0].shares == 1_000


def test_guard_rejects_whitelist_blacklist_and_missing_price():
    guard = _guard(allowed_symbols=("000001.SZ",))
    batch = guard.check_batch([GatewayOrder("600519.SH", "buy", 100, 10.0)], _context())
    assert "白名单" in _reasons(batch)

    guard = _guard(banned_symbols=("600519.SH",))
    batch = guard.check_batch([GatewayOrder("600519.SH", "buy", 100, 10.0)], _context())
    assert "黑名单" in _reasons(batch)

    guard = _guard(require_limit_price=True)
    batch = guard.check_batch([GatewayOrder("600519.SH", "buy", 100, None)], _context())
    assert "限价" in _reasons(batch)


def test_guard_rejects_price_deviation():
    guard = _guard(max_price_deviation_pct=0.05)
    batch = guard.check_batch([GatewayOrder("600519.SH", "buy", 100, 10.5)], _context(latest={"600519.SH": 9.0}))
    assert not batch.accepted
    assert "偏离" in _reasons(batch)
    # 5% 以内放行（股数要足以越过最小委托额）
    ok = guard.check_batch([GatewayOrder("600519.SH", "buy", 1_000, 9.4)], _context(latest={"600519.SH": 9.0}))
    assert len(ok.accepted) == 1


def test_guard_rejects_duplicate_order_and_daily_cap():
    order = GatewayOrder("600519.SH", "buy", 100, 10.0)
    key = f"{date.today().isoformat()}|{order.key}"
    batch = _guard().check_batch([order], _context(submitted_keys={key}))
    assert "幂等" in _reasons(batch)

    batch = _guard(max_orders_per_day=1).check_batch([order], _context(orders_today=1))
    assert "上限" in _reasons(batch)


def test_guard_blocks_limit_up_buy_and_limit_down_sell():
    guard = _guard()
    context = _context(latest={"600519.SH": 11.0}, prev_close={"600519.SH": 10.0})
    batch = guard.check_batch([GatewayOrder("600519.SH", "buy", 100, 11.0)], context)
    assert "涨停" in _reasons(batch)

    context = _context(
        latest={"600519.SH": 9.0},
        prev_close={"600519.SH": 10.0},
        available={"600519.SH": 1000},
        position_value={"600519.SH": 9000.0},
    )
    batch = guard.check_batch([GatewayOrder("600519.SH", "sell", 100, 9.0)], context)
    assert "跌停" in _reasons(batch)


# ---------------------------------------------------------------- 缩量


def test_guard_shrinks_buy_to_available_cash():
    guard = _guard(max_single_order_value=1e9, max_weight_per_symbol=1.0, min_cash_ratio=0.0)
    context = _context(cash=50_000.0, equity=50_000.0, prev_amount={})
    batch = guard.check_batch([GatewayOrder("600519.SH", "buy", 100_000, 10.0)], context)
    assert len(batch.accepted) == 1
    final = batch.accepted[0]
    assert final.shares < 100_000
    assert final.shares % 100 == 0
    assert final.shares * 10.0 <= 50_000.0
    assert batch.adjusted and batch.adjusted[0].shrunk


def test_guard_shrinks_by_position_weight():
    guard = _guard(max_weight_per_symbol=0.10, min_cash_ratio=0.0, max_single_order_value=1e9)
    context = _context(cash=100_000.0, equity=100_000.0, prev_amount={})
    batch = guard.check_batch([GatewayOrder("600519.SH", "buy", 5_000, 10.0)], context)
    assert batch.accepted[0].shares == 1_000  # 10% × 100000 / 10
    assert "单票权重上限" in "；".join(batch.decisions[0].reasons)


def test_guard_shrinks_by_turnover_and_participation():
    guard = _guard(max_turnover_of_equity=0.05, min_cash_ratio=0.0, max_single_order_value=1e9, max_weight_per_symbol=1.0)
    batch = guard.check_batch(
        [GatewayOrder("600519.SH", "buy", 5_000, 10.0)],
        _context(cash=100_000.0, equity=100_000.0, prev_amount={}),
    )
    assert batch.accepted[0].shares == 500  # 5% × 100000 / 10
    assert "单日换手" in "；".join(batch.decisions[0].reasons)

    guard = _guard(max_volume_participation=0.01, min_cash_ratio=0.0, max_single_order_value=1e9,
                   max_weight_per_symbol=1.0, max_turnover_of_equity=1.0)
    batch = guard.check_batch(
        [GatewayOrder("600519.SH", "buy", 5_000, 10.0)],
        _context(cash=1_000_000.0, equity=1_000_000.0, prev_amount={"600519.SH": 1_000_000.0}),
    )
    assert batch.accepted[0].shares == 1_000  # 1% × 100 万 / 10
    assert "冲击控制" in "；".join(batch.decisions[0].reasons)


def test_guard_shrinks_sell_to_available_shares():
    guard = _guard(min_order_value=0.0)
    context = _context(
        cash=0.0,
        equity=5_000.0,
        position_value={"600519.SH": 5_000.0},
        available={"600519.SH": 200},
        prev_amount={},
    )
    batch = guard.check_batch([GatewayOrder("600519.SH", "sell", 500, 10.0)], context)
    assert batch.accepted[0].shares == 200
    assert "可卖" in "；".join(batch.decisions[0].reasons)


def test_guard_rejects_below_min_order_value():
    guard = _guard(min_order_value=5_000.0)
    batch = guard.check_batch([GatewayOrder("600519.SH", "buy", 100, 10.0)], _context(prev_amount={}))
    assert not batch.accepted
    assert "最小委托额" in _reasons(batch)


def test_guard_accumulates_state_within_batch():
    guard = _guard(max_weight_per_symbol=0.10, min_cash_ratio=0.0, max_single_order_value=1e9)
    orders = [
        GatewayOrder("600519.SH", "buy", 1_000, 10.0),
        GatewayOrder("600519.SH", "buy", 1_000, 10.0),  # 与第一笔完全同键 → 批内幂等保护
        GatewayOrder("600519.SH", "buy", 2_000, 10.0),  # 单票额度已被第一笔用满 → 缩到 0 股
    ]
    batch = guard.check_batch(orders, _context(cash=1_000_000.0, equity=100_000.0, prev_amount={}))
    assert len(batch.accepted) == 1
    assert batch.accepted[0].shares == 1_000
    assert len(batch.rejected) == 2
    reasons = _reasons(batch)
    assert "幂等" in reasons  # 批内累计了已提交键
    assert "通过 1 笔" in batch.summary()


def test_guard_describe_exposes_limits():
    info = _guard().describe()
    assert info["limits"]["max_weight_per_symbol"] == TradeLimits().max_weight_per_symbol
    assert info["rules"]["lot_size"] == 100
    assert "commission_rate" in info["costs"]
