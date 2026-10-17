"""模拟盘（纸上交易）——不接券商，只把信号落成可人工执行的清单。

三个子命令（子命令写在前面，选项写在后面）::

    # 1) 出下一交易日的建议单（不落任何成交）
    python -m scripts.paper_trade plan --strategy momentum_rank --cash 1000000

    # 2) 把你实际成交的结果录进去（symbol:股数:价格，卖为负股数；价格可省略＝用当日收盘价）
    python -m scripts.paper_trade apply --source csv --fills "600519.SH:200:1680.5,000001.SZ:-1000"

    # 3) 看账户现状
    python -m scripts.paper_trade status --source csv

设计取舍：成交价按你的填报值（缺省用当日收盘价），费用由 CostModel 估算，
所以纸上账户的净值曲线与回测口径一致，可以直接和回测结果对照。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant import plugins
from quant import strategy as strategy_lib
from quant.config import BacktestConfig, load_config
from quant.data import CsvSource, DataStore, SyntheticSource
from quant.execution import PaperTrader
from quant.risk import RiskLimits, RiskManager

DEFAULT_SYMBOLS = "600519.SH,000001.SZ,300750.SZ,601318.SH"
DEFAULT_STATE = "data/paper"


def load_cfg(path: str) -> BacktestConfig:
    return load_config(path) if Path(path).is_file() else BacktestConfig()


def build_store(args, cfg: BacktestConfig) -> DataStore:
    if args.source == "synthetic":
        source = SyntheticSource(list(cfg.universe))
    elif args.source == "csv":
        source = CsvSource(args.csv_dir)
    else:
        from quant.data.akshare_source import AkshareSource

        source = AkshareSource()
    return DataStore(source, data_dir=cfg.data_dir, cache_dir=None if args.source == "synthetic" else cfg.data_dir)


def make_trader(args) -> PaperTrader:
    plugins.load_plugins_from_config_file(args.config if Path(args.config).is_file() else None, root=ROOT)
    cfg = load_cfg(args.config)
    symbols = [s.strip() for s in (args.symbols or ",".join(cfg.universe) or DEFAULT_SYMBOLS).split(",") if s.strip()]
    cfg = replace(cfg, universe=tuple(symbols))
    if args.cash:
        cfg = replace(cfg, initial_cash=args.cash)
    strategy = plugins.build_strategy(args.strategy, **dict(cfg.params))
    risk = RiskManager(RiskLimits()) if args.risk_limits else None
    return PaperTrader(build_store(args, cfg), strategy, cfg, state_dir=args.state_dir, risk=risk)


def cmd_plan(args) -> int:
    trader = make_trader(args)
    plan = trader.plan(as_of=args.as_of)
    if plan is None or plan.empty:
        print("没有建议单（可能：非调仓日、没有可用数据、或目标权重与现状一致）")
        return 0
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(plan.to_string(index=False))
    out = Path(args.state_dir) / "plan-latest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n建议单已保存 {out}")
    example = ",".join(
        f"{row.symbol}:{int(row.shares) if row.side == '买入' else -int(row.shares)}:{row.ref_price}"
        for row in plan.itertuples()
    )
    print("人工成交后记录：")
    print(f'  python -m scripts.paper_trade apply --source {args.source} --fills "{example}"')
    return 0


def parse_fills(text: str | None, path: str | None) -> list[dict]:
    """``"600519.SH:200:1680.5,000001.SZ:-1000"`` → 记录列表（价格为可选第三段）。"""
    fills: list[dict] = []
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if "side" not in row:
                row = dict(row)
                row["side"] = "buy" if float(row.get("shares", 0)) >= 0 else "sell"
                row["shares"] = abs(float(row["shares"]))
            fills.append(row)
    if text:
        for chunk in text.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split(":")
            if len(parts) < 2:
                raise SystemExit(f'成交格式应为 symbol:股数[:价格]（卖为负股数）：{chunk}')
            symbol, raw_shares = parts[0].strip(), float(parts[1])
            entry: dict = {
                "symbol": symbol,
                "side": "buy" if raw_shares >= 0 else "sell",
                "shares": abs(int(raw_shares)),
            }
            if len(parts) >= 3 and parts[2].strip():
                entry["price"] = float(parts[2])
            fills.append(entry)
    return fills


def fill_prices(trader: PaperTrader, fills: list[dict], date: str | None) -> None:
    """缺少 price 的成交，用该日（或之前最近一日）收盘价补上。"""
    missing = [f for f in fills if f.get("price") is None]
    if not missing:
        return
    symbols = sorted({str(f["symbol"]) for f in missing})
    bars = trader.store.get_many(symbols, trader.config.start, date or trader.config.end)
    for fill in missing:
        frame = bars.get(str(fill["symbol"]))
        if frame is None or frame.empty:
            raise SystemExit(f"无法取到 {fill['symbol']} 的行情来推算成交价，请显式写 symbol:股数:价格")
        if date:
            frame = frame[frame.index <= pd.Timestamp(date)]
        fill["price"] = float(frame["close"].iloc[-1])


def cmd_apply(args) -> int:
    trader = make_trader(args)
    fills = parse_fills(args.fills, args.fills_file)
    if not fills:
        raise SystemExit("没有成交可录（用 --fills 或 --fills-file）")
    fill_prices(trader, fills, args.date)
    date = args.date or None
    if date is None:
        bars = trader.store.get_many(sorted({str(f["symbol"]) for f in fills}), trader.config.start, trader.config.end)
        if bars:
            date = max(f.index.max() for f in bars.values()).date().isoformat()
    trader.apply_fills(fills, date=date)
    trader.mark_to_market(date=date)
    print(f"已录入 {len(fills)} 笔成交（成交日 {date}）")
    return cmd_status(args)


def cmd_status(args) -> int:
    trader = make_trader(args)
    print(f"账户状态目录：{trader.state_dir}")
    print(trader.status())
    equity_file = Path(args.state_dir) / "equity.csv"
    if equity_file.is_file():
        curve = pd.read_csv(equity_file)
        if len(curve):
            col = "equity" if "equity" in curve.columns else curve.columns[-1]
            first, last = float(curve[col].iloc[0]), float(curve[col].iloc[-1])
            print(f"净值曲线  : {len(curve)} 个记录，{first:,.0f} → {last:,.0f}（{last / first - 1:+.2%}）")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="模拟盘")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", default="config.toml")
        sp.add_argument("--strategy", default="momentum_rank")
        sp.add_argument("--source", default="csv", choices=["csv", "synthetic", "akshare"])
        sp.add_argument("--csv-dir", default="data/csv")
        sp.add_argument("--symbols", default=None)
        sp.add_argument("--cash", type=float, default=None, help="首次建账户的资金")
        sp.add_argument("--state-dir", default=DEFAULT_STATE)
        sp.add_argument("--risk-limits", action="store_true", help="启用 RiskManager 权重限额")
        return sp

    sp = common(sub.add_parser("plan", help="出下一交易日建议单"))
    sp.add_argument("--as-of", default=None, help="以该日期为止的数据出信号（默认最新）")

    sp = common(sub.add_parser("apply", help="录入实际成交"))
    sp.add_argument("--date", default=None, help="成交日期（默认最新交易日）")
    sp.add_argument("--fills", default=None, help='例如 "600519.SH:200:1680.5,000001.SZ:-1000"')
    sp.add_argument("--fills-file", default=None, help="JSON 文件：[{symbol, shares, price?}, ...]")

    common(sub.add_parser("status", help="查看账户"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return {"plan": cmd_plan, "apply": cmd_apply, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
