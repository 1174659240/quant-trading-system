"""实盘调度（半自动）：信号 → 计划 → 人工确认 → 提交 → 回报 → 对账。

这是"可以自己下单"的**安全实现**，也是成熟机构流程在个人账户上的最小可行版本：

    python -m quant trade plan       # 收盘后跑：出建议单，写 data/orders/<日期>-orders.preview.csv
    # 打开 CSV 看一眼（可以删单、改价、改股数），然后
    python -m quant trade confirm --yes
    # 第二天在券商 App 里照着 orders.csv 下单，成交后
    python -m quant trade record --file data/orders/<日期>-fills.csv
    python -m quant trade status     # 看账本
    python -m quant trade reconcile --statement broker.csv   # 与券商对账

四个安全开关，缺一不可：

1. ``[gateway] dry_run = true``（默认）：只出预演文件，永远不进入"已提交"状态；
2. ``confirm`` 必须显式带 ``--yes``（或在配置里放 ``confirm_token`` 并用 ``--token`` 匹配）；
3. **订单级护栏**（``quant.execution.guard``）在写文件之前先过一遍：涨跌停、整手、
   单票权重、单日换手、成交量参与度、幂等查重……拦下来的单不会进文件；
4. ``data/KILL_SWITCH`` 文件一旦存在，所有提交直接拒绝（急停开关）。

**真实下单前请务必读 ``docs/实盘接入与合规.md``**：程序化交易有报备要求，
本工程的默认通道（半自动订单文件）不涉及券商 API，风险最低。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant import plugins  # noqa: E402
from quant.config import BacktestConfig, load_config  # noqa: E402
from quant.data import CsvSource, DataStore, SyntheticSource  # noqa: E402
from quant.execution.guard import GuardContext, PreTradeGuard  # noqa: E402
from quant.execution.ledger import Ledger  # noqa: E402
from quant.gateway import build_gateway, load_gateway_config, load_guard_limits  # noqa: E402
from quant.gateway.base import GatewayError, GatewayFill, GatewayOrder  # noqa: E402

SHANGHAI = timezone(timedelta(hours=8))
AM_OPEN, AM_CLOSE = time(9, 30), time(11, 30)
PM_OPEN, PM_CLOSE = time(13, 0), time(15, 0)
KILL_SWITCH = ROOT / "data" / "KILL_SWITCH"


# --------------------------------------------------------------------------
# 公共装配
# --------------------------------------------------------------------------


def _config_path(path: str | None) -> Path:
    if path and Path(path).is_file():
        return Path(path)
    return ROOT / "config.toml"


def load_cfg(path: str | None) -> BacktestConfig:
    candidate = _config_path(path)
    return load_config(candidate) if candidate.is_file() else BacktestConfig()


def load_strategy_name(path: str | None, fallback: str = "multifactor") -> str:
    """从 [strategy] name 取默认策略名（tomllib 只读一次，失败就退回 fallback）。"""
    candidate = _config_path(path)
    if not candidate.is_file():
        return fallback
    try:
        import tomllib

        with candidate.open("rb") as handle:
            raw = tomllib.load(handle)
    except Exception:
        return fallback
    section = raw.get("strategy") or {}
    name = str(section.get("name") or "").strip()
    return name or fallback


def build_store(args) -> DataStore:
    if args.source == "synthetic":
        source = SyntheticSource(start=args.start or "2020-01-01", periods=int(args.periods or 1200))
    else:
        csv_dir = Path(args.csv_dir or "data/csv")
        if not csv_dir.is_absolute():
            csv_dir = ROOT / csv_dir
        source = CsvSource(csv_dir)
    return DataStore(source)


def load_bars(args, store: DataStore, cfg: BacktestConfig, as_of: str | None) -> dict[str, pd.DataFrame]:
    symbols = list(cfg.universe)
    bars = store.get_many(symbols, start=cfg.start or None, end=as_of or cfg.end or None)
    if not bars:
        raise SystemExit("没有取到任何行情：先用 python -m quant fetch 准备数据")
    return bars


def open_ledger(gateway_cfg, *, capital: float | None = None) -> Ledger:
    ledger = Ledger.load(gateway_cfg.ledger_path)
    if not ledger.trades and ledger.cash == 0.0 and not ledger.positions:
        ledger.cash = float(capital if capital is not None else gateway_cfg.initial_cash)
    return ledger


def session_ok(now: datetime | None = None) -> bool:
    """是否在连续竞价时段（北京时间，粗判，不含节假日）。"""
    now = now or datetime.now(SHANGHAI)
    if now.tzinfo is None:
        now = now.replace(tzinfo=SHANGHAI)
    local = now.astimezone(SHANGHAI)
    if local.weekday() >= 5:
        return False
    return (AM_OPEN <= local.time() <= AM_CLOSE) or (PM_OPEN <= local.time() <= PM_CLOSE)


def latest_row(bars: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, dict[str, dict[str, float]]]:
    """取最后一个交易日，以及当天各标的的 收盘/昨收/成交额。"""
    stamps = [frame.index[-1] for frame in bars.values() if len(frame)]
    if not stamps:
        raise SystemExit("行情是空的")
    day = max(stamps)
    snapshot: dict[str, dict[str, float]] = {}
    for symbol, frame in bars.items():
        if day not in frame.index:
            continue
        row = frame.loc[day]
        snapshot[symbol] = {
            "close": float(row.get("close") or 0.0),
            "prev_close": float(row.get("prev_close") or 0.0),
            "amount": float(row.get("amount") or 0.0),
        }
    return day, snapshot


def build_guard_context(
    ledger: Ledger,
    snapshot: dict[str, dict[str, float]],
    *,
    submitted_today: list[dict] | None = None,
) -> GuardContext:
    prices = {symbol: info["close"] for symbol, info in snapshot.items()}
    shares = ledger.shares()
    position_value = {
        symbol: shares.get(symbol, 0) * prices.get(symbol, 0.0) for symbol in set(shares) | set(prices)
    }
    submitted = submitted_today or []
    return GuardContext(
        cash=ledger.cash,
        equity=ledger.cash + sum(position_value.values()),
        position_value=position_value,
        available=ledger.available_shares(),
        latest=prices,
        prev_close={symbol: info["prev_close"] for symbol, info in snapshot.items()},
        prev_amount={symbol: info["amount"] for symbol, info in snapshot.items()},
        submitted_keys=set(ledger.order_keys),
        orders_today=len(submitted),
        daily_turnover=sum(float(row.get("amount") or 0.0) for row in submitted),
        kill_switch=KILL_SWITCH.exists(),
        session_ok=session_ok(),
    )


def target_orders(
    weights: pd.Series,
    ledger: Ledger,
    snapshot: dict[str, dict[str, float]],
    *,
    rules,
    min_value: float = 0.0,
) -> list[GatewayOrder]:
    """把目标权重转成委托：先卖后买，整手，卖出不超过可卖。"""
    equity = ledger.cash + sum(
        ledger.shares().get(symbol, 0) * info["close"] for symbol, info in snapshot.items()
    )
    lot = rules.lot_size
    current = ledger.shares()
    available = ledger.available_shares()
    sells: list[GatewayOrder] = []
    buys: list[GatewayOrder] = []
    for symbol in sorted(set(weights.index) | set(current)):
        price = float(snapshot.get(symbol, {}).get("close") or 0.0)
        if price <= 0:
            continue
        weight = float(weights.get(symbol, 0.0) or 0.0)
        desired = int(max(weight, 0.0) * equity // price)
        desired -= desired % lot
        held = int(current.get(symbol, 0))
        delta = desired - held
        if delta < 0:
            shares = min(-delta, available.get(symbol, 0))
            shares -= shares % lot
            if shares > 0 and shares * price >= min_value:
                sells.append(
                    GatewayOrder(symbol, "sell", shares, round(price, 2), client_id="plan",
                                 note=f"减仓至 {weight:.1%}")
                )
        elif delta > 0:
            shares = delta - delta % lot
            if shares > 0 and shares * price >= min_value:
                buys.append(
                    GatewayOrder(symbol, "buy", shares, round(price, 2), client_id="plan",
                                 note=f"加仓至 {weight:.1%}")
                )
    return sells + buys


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------


def cmd_plan(args) -> int:
    config_path = _config_path(args.config)
    cfg = load_cfg(args.config)
    gateway_cfg = load_gateway_config(config_path)
    if args.no_dry_run:
        gateway_cfg = replace(gateway_cfg, dry_run=False)
    limits = load_guard_limits(config_path)
    plugins.load_plugins_from_config_file(config_path if config_path.is_file() else None, root=ROOT)

    store = build_store(args)
    bars = load_bars(args, store, cfg, args.date)
    day, snapshot = latest_row(bars)
    strategy_name = args.strategy or load_strategy_name(args.config)
    params = {k: v for k, v in dict(cfg.params).items() if not k.startswith("_")}
    strategy = plugins.build_strategy(strategy_name, **params)
    weights = strategy.generate(bars)
    target = weights.loc[weights.index <= day].iloc[-1] if len(weights) else pd.Series(dtype=float)

    gateway = build_gateway(gateway_cfg)
    gateway.connect()
    ledger = open_ledger(gateway_cfg, capital=args.capital)
    settled = ledger.settle_if_new_day(date.today())

    cost_model = cfg.costs
    rules = cfg.rules
    orders = target_orders(target, ledger, snapshot, rules=rules, min_value=limits.min_order_value)
    guard = PreTradeGuard(limits, rules=rules, costs=cost_model)
    submitted = gateway.submitted_orders() if hasattr(gateway, "submitted_orders") else []
    context = build_guard_context(ledger, snapshot, submitted_today=submitted)
    batch = guard.check_batch(orders, context)

    print(f"信号日期 {day.date()}（策略 {strategy.describe()}）")
    print(f"账户：总权益 {context.equity:,.2f} 元 · 现金 {ledger.cash:,.2f} 元 · 持仓 {len(ledger.shares())} 只"
          f"{'（已按新交易日结算 T+1）' if settled else ''}")
    if batch.accepted:
        print(f"\n建议单 {len(batch.accepted)} 笔：")
        for order in batch.accepted:
            print(f"  {order.side:4} {order.symbol} {order.shares} 股 @ {order.price:.2f} "
                  f"≈ {order.value:,.0f} 元  {order.note}")
    for decision in batch.decisions:
        if decision.ok and decision.shrunk:
            print(f"  [缩量] {decision.order.symbol} {decision.order.shares} → {decision.final_order.shares} 股：" +
                  "；".join(decision.reasons))
    if batch.rejected:
        print(f"\n被护栏拦下 {len(batch.rejected)} 笔：")
        for order, reason in batch.rejected:
            print(f"  {order.side:4} {order.symbol} {order.shares} 股 —— {reason}")
    if not orders:
        print("\n当前没有需要调整的持仓（或全部被拦下）。")

    result = gateway.submit(batch.accepted, confirm=False)
    for path in result.artifacts:
        print(f"\n订单文件：{path}")
    if hasattr(gateway, "write_fill_template") and batch.accepted:
        template = gateway.write_fill_template(batch.accepted)
        print(f"成交回报模板：{template}")
    print("\n下一步：核对/编辑订单文件后执行")
    print("  python -m quant trade confirm --yes")
    if KILL_SWITCH.exists():
        print(f"\n注意：急停开关存在（{KILL_SWITCH}），任何提交都会被拒绝。删除该文件才恢复。")
    return 0


def cmd_confirm(args) -> int:
    config_path = _config_path(args.config)
    gateway_cfg = load_gateway_config(config_path)
    limits = load_guard_limits(config_path)
    _ = limits
    gateway = build_gateway(gateway_cfg)
    gateway.connect()

    if not hasattr(gateway, "read_orders"):
        raise SystemExit(f"通道 {gateway.name} 不支持从订单文件确认，请直接用它自己的提交方式")
    orders = gateway.read_orders(args.date)
    if not orders:
        print("订单文件不存在或没有有效委托，先跑 trade plan。")
        return 1

    confirmed = bool(args.yes)
    if not confirmed and gateway_cfg.confirm_token and args.token:
        confirmed = args.token == gateway_cfg.confirm_token
    if not confirmed:
        print(f"待提交 {len(orders)} 笔（来自 {gateway.day_path(args.date, 'orders.preview')}）：")
        for order in orders:
            print(f"  {order.side:4} {order.symbol} {order.shares} 股 @ {order.price} {order.note}")
        print("\n没有确认：加 --yes 才会提交（这是防止误下单的第二道闸门）。")
        return 1

    if KILL_SWITCH.exists():
        print(f"急停开关存在（{KILL_SWITCH}），拒绝提交。")
        return 1

    result = gateway.submit(orders, confirm=True)
    ledger = open_ledger(gateway_cfg)
    if result.accepted:
        ledger.record_order_keys([order.key for order in result.accepted], day=date.today())
        ledger.save()
    if result.dry_run:
        print(f"已生成订单文件 {len(result.accepted)} 笔（dry-run：不会真实下单，去券商客户端按文件手工委托）")
    else:
        print(f"已提交 {len(result.accepted)} 笔（通道 {gateway.name}）")
    for path in result.artifacts:
        print(f"  {path}")
    if result.accepted:
        print(f"幂等键已记入账本：{gateway_cfg.ledger_path}（同一天同样的单不会被重复提交）")
    return 0


def _parse_fills(text: str | None) -> list[GatewayFill]:
    fills: list[GatewayFill] = []
    if not text:
        return fills
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) < 2:
            raise SystemExit(f"成交格式应为 symbol:股数[:价格[:费用]]，收到 {chunk!r}")
        symbol = parts[0].strip()
        shares = int(float(parts[1]))
        price = float(parts[2]) if len(parts) > 2 and parts[2] else None
        fee = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
        side = "buy" if shares > 0 else "sell"
        fills.append(GatewayFill(symbol=symbol, side=side, shares=abs(shares), price=price or 0.0, fee=fee))
    return fills


def _fills_from_file(path: Path) -> list[GatewayFill]:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    fills: list[GatewayFill] = []
    for _, row in frame.iterrows():
        symbol = str(row.get("symbol") or "").strip()
        side_raw = str(row.get("side") or "").strip().lower()
        if not symbol or side_raw not in ("buy", "sell", "买入", "卖出"):
            continue
        try:
            shares = int(float(row.get("shares") or 0))
            price = float(row.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if shares <= 0 or price <= 0:
            continue
        fills.append(
            GatewayFill(
                symbol=symbol,
                side="buy" if side_raw in ("buy", "买入") else "sell",
                shares=shares,
                price=price,
                fee=float(row.get("fee") or 0.0),
                client_id=str(row.get("clientId") or ""),
                time=str(row.get("time") or ""),
            )
        )
    return fills


def cmd_record(args) -> int:
    config_path = _config_path(args.config)
    gateway_cfg = load_gateway_config(config_path)
    ledger = open_ledger(gateway_cfg)
    ledger.settle_if_new_day(args.date or date.today())

    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = ROOT / path
        fills = _fills_from_file(path)
    else:
        fills = _parse_fills(args.fills)
    if not fills:
        print("没有解析到任何成交记录：用 --fills 或 --file 指定。")
        return 1

    for fill in fills:
        if not fill.price or fill.price <= 0:
            print(f"跳过 {fill.symbol}：缺少成交价（回报文件里要填 price）")
            continue
        record = ledger.apply_fill(fill)
        print(f"录入 {record['side']:4} {record['symbol']} {record['shares']} 股 @ {record['price']:.2f} "
              f"费 {record['fee']:.2f} 已实现 {record['realizedPnl']:,.2f}")
    ledger.save()
    print("\n" + ledger.to_text())
    print(f"账本：{gateway_cfg.ledger_path}")
    return 0


def cmd_status(args) -> int:
    config_path = _config_path(args.config)
    gateway_cfg = load_gateway_config(config_path)
    ledger = open_ledger(gateway_cfg)
    print(ledger.to_text())
    gateway = None
    try:
        gateway = build_gateway(gateway_cfg)
        gateway.connect()
        account = gateway.query_account()
        print(f"\n通道 {gateway.name}：现金 {account.cash:,.2f} · 市值 {account.market_value:,.2f}")
        for key, value in gateway.describe().items():
            print(f"  {key}: {value}")
    except GatewayError as exc:
        print(f"\n通道不可用：{exc}")
    limits = load_guard_limits(config_path)
    print("\n订单护栏：")
    for key, value in limits.__dict__.items():
        print(f"  {key} = {value}")
    print(f"\n急停开关：{'已触发（拒绝一切提交）' if KILL_SWITCH.exists() else '未触发'}"
          f"（{'创建' if not KILL_SWITCH.exists() else '删除'} {KILL_SWITCH} 可切换）")
    if gateway is not None and hasattr(gateway, "submitted_orders"):
        rows = gateway.submitted_orders(args.date)
        if rows:
            print(f"\n今日已提交 {len(rows)} 笔：")
            for row in rows:
                print(f"  {row.get('submittedAt', '')} {row.get('side')} {row.get('symbol')} "
                      f"{row.get('shares')} 股 @ {row.get('price')}")
    return 0


def cmd_reconcile(args) -> int:
    config_path = _config_path(args.config)
    gateway_cfg = load_gateway_config(config_path)
    ledger = open_ledger(gateway_cfg)
    path = Path(args.statement)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        print(f"对账单不存在：{path}")
        return 1
    statement = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig").to_dict("records")
    diffs = ledger.reconcile(statement)
    if not diffs:
        print(f"对账通过：账本与对账单一致（{len(statement)} 行）。")
        return 0
    print(f"发现 {len(diffs)} 处差异：")
    for diff in diffs:
        print(f"  {diff['symbol']} {diff['field']}：账本 {diff['ledger']} vs 券商 {diff['broker']} "
              f"(差 {diff['diff']}) {diff.get('note', '')}")
    print("\n先人工核对，再决定改账本还是查券商回报；不要盲目信任任何一侧。")
    return 1


def cmd_limits(args) -> int:
    config_path = _config_path(args.config)
    limits = load_guard_limits(config_path)
    print("订单级护栏（[guard] 段，未配置则用默认值）：")
    for key, value in limits.__dict__.items():
        print(f"  {key} = {value}")
    print("\n组合级护栏由 [risk] 段的 overlay 提供（max_weight / exposure_cap / vol_target…）。")
    return 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m quant trade", description="半自动实盘调度")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(ROOT / "config.toml"), help="配置文件（默认 config.toml）")
    common.add_argument("--date", default=None, help="指定日期（YYYY-MM-DD）")

    p_plan = sub.add_parser("plan", parents=[common], help="生成建议单与订单文件")
    p_plan.add_argument("--strategy", default=None, help="策略名（默认取 config.toml 的 [strategy] name）")
    p_plan.add_argument("--source", default="csv", choices=["csv", "synthetic"], help="行情来源")
    p_plan.add_argument("--csv-dir", default="data/csv")
    p_plan.add_argument("--start", default=None)
    p_plan.add_argument("--periods", default=1200)
    p_plan.add_argument("--capital", type=float, default=None, help="首次运行的初始现金")
    p_plan.add_argument("--no-dry-run", action="store_true", help="写正式订单文件（默认只出 .preview）")

    p_confirm = sub.add_parser("confirm", parents=[common], help="确认并提交订单文件")
    p_confirm.add_argument("--yes", action="store_true", help="确认提交（必备）")
    p_confirm.add_argument("--token", default=None, help="与配置里的 confirm_token 匹配")

    p_record = sub.add_parser("record", parents=[common], help="录入成交回报")
    p_record.add_argument("--fills", default=None, help="symbol:股数[:价格[:费用]]，多笔用 ; 分隔（卖为负股数）")
    p_record.add_argument("--file", default=None, help="成交回报 CSV（列：symbol,side,shares,price,fee）")

    sub.add_parser("status", parents=[common], help="账本与通道状态")
    p_rec = sub.add_parser("reconcile", parents=[common], help="与券商对账单比对")
    p_rec.add_argument("--statement", required=True, help="对账单 CSV（列：symbol,shares,available[,cash]）")
    sub.add_parser("limits", parents=[common], help="展示订单级护栏")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    handlers = {
        "plan": cmd_plan,
        "confirm": cmd_confirm,
        "record": cmd_record,
        "status": cmd_status,
        "reconcile": cmd_reconcile,
        "limits": cmd_limits,
    }
    try:
        return int(handlers[args.command](args))
    except GatewayError as exc:
        print(f"通道错误：{exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
