"""Paper 执行校准：每日收盘后运行，记录「假设成交」与「真实可成交」证据。

两种模式
--------
- ``replay``（默认）：用数据湖 processed 层（≤2024-12-31）按日重放，用于**校验管线本身**，
  与回测权重逐月对比；不读 holdout。
- ``live``：拉取当日真实行情（现有 `quant.data.web_source` / akshare），用于**正式 paper 跟踪**。
  live 会读取 2025 年之后的行情与财报（前向运行必需），**只写 `results/growth/paper/`**，
  不得回灌到任何研究结果或参数选择中。

台账（`results/growth/paper/`）
------------------------------
- ``state.json``        当前现金 / 持仓 / 待成交订单 / 最后运行日
- ``ledger.csv``        每日：净值、现金、市值、持仓数、gross、费用、换手
- ``fills-YYYY-MM-DD.csv`` 当日成交明细（含「真实可成交」标记与有效费率）
- ``orders-YYYY-MM-DD.csv`` 当日生成的待成交订单（次日开盘执行）
- ``report.md``         3 个月后可跑的汇总对账

用法
----
    python -X utf8 -m scripts.growth_paper --mode replay --start 2024-01-02 --end 2024-12-31
    python -X utf8 -m scripts.growth_paper --mode live --live-ack
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PAPER = Path("results/growth/paper")
SECTOR_CAP = 0.25
TOP_K = 50
BUFFER = 60
GROSS = 0.95
MAX_WEIGHT = 0.04
LOT = 100
MIN_TRADE = 5000.0
INIT_CASH = 1_000_000.0
COMMISSION_RATE = 0.00025
COMMISSION_MIN = 5.0
STAMP = 0.0005
TRANSFER = 1e-5
SLIP_TIERS = (2.0, 2.5, 3.0, 4.0, 5.0)


# --------------------------------------------------------------------------- #
# 状态
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    path = PAPER / "state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"cash": INIT_CASH, "holdings": {}, "pending": [], "last_date": None,
            "run_id": 0, "mode": "replay"}


def save_state(state: dict) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 行情与可成交性
# --------------------------------------------------------------------------- #
def bar_of(bars: dict[str, pd.DataFrame], symbol: str, day: pd.Timestamp) -> pd.Series | None:
    frame = bars.get(symbol)
    if frame is None or day not in frame.index:
        return None
    return frame.loc[day]


def tradability(row: pd.Series, symbol: str, prev_close: float, day: pd.Timestamp) -> dict:
    """判定当日是否可买/可卖，以及是否一字板。"""
    if row is None or float(row.get("volume", 0.0)) <= 0:
        return {"suspended": True, "limit_up": False, "limit_down": False, "one_word": False}
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    change = close / prev_close - 1.0 if prev_close else 0.0
    cap = 0.20 if (_is_growth_board(symbol) and day >= pd.Timestamp("2020-08-24")) else 0.10
    if str(symbol).startswith("688") and day >= pd.Timestamp("2019-07-22"):
        cap = 0.20
    one_word = abs(high - low) <= close * 1e-4
    return {
        "suspended": False,
        "limit_up": change >= cap - 0.005,
        "limit_down": change <= -cap + 0.005,
        "one_word": bool(one_word),
        "change": change,
    }


def _is_growth_board(symbol: str) -> bool:
    code = str(symbol).split(".")[0]
    return code.startswith(("300", "301"))


def slip_tier(adv: float, adv_cut: list[float]) -> float:
    """按 ADV 分位给滑点（tier1 最流动 2bp … tier5 最差 5bp）。"""
    if not adv_cut or not np.isfinite(adv):
        return SLIP_TIERS[2]
    for tier, cut in enumerate(adv_cut):
        if adv >= cut:
            return SLIP_TIERS[tier]
    return SLIP_TIERS[-1]


# --------------------------------------------------------------------------- #
# 订单与成交
# --------------------------------------------------------------------------- #
def plan_orders(target_weights: pd.Series, equity: float, holdings: dict[str, int],
                open_px: pd.Series) -> list[dict]:
    """按目标权重生成订单（整手、<5000 元忽略）。权重已含 gross，不再二次缩放。"""
    target_value = target_weights * equity
    orders: list[dict] = []
    symbols = set(target_weights.index) | set(holdings)
    for symbol in sorted(symbols):
        price = float(open_px.get(symbol, np.nan))
        if not np.isfinite(price) or price <= 0:
            continue
        weight = float(target_weights.get(symbol, 0.0))
        weight = min(weight, MAX_WEIGHT)
        want_value = target_value.get(symbol, 0.0) if weight > 0 else 0.0
        current = holdings.get(symbol, 0) * price
        delta = want_value - current
        if abs(delta) < MIN_TRADE:
            continue
        shares = int(math.floor(abs(delta) / price / LOT)) * LOT
        if shares <= 0:
            continue
        orders.append({"symbol": symbol, "side": "buy" if delta > 0 else "sell",
                       "shares": shares, "ref_price": price,
                       "amount": shares * price})
    return orders


def fees_of(amount: float, side: str) -> dict:
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    stamp = amount * STAMP if side == "sell" else 0.0
    transfer = amount * TRANSFER
    total = commission + stamp + transfer
    return {"commission": commission, "stamp": stamp, "transfer": transfer, "total": total,
            "effective_bps": total / amount * 1e4 if amount > 0 else 0.0}


def execute(order: dict, row: pd.Series | None, prev_close: float, day: pd.Timestamp,
            slip_bp: float) -> dict:
    flags = tradability(row, order["symbol"], prev_close, day)
    base = dict(order, date=str(day.date()), fillable=False, reason="", fill_price=np.nan,
                filled_shares=0, fees=0.0, effective_bps=np.nan, **flags)
    if flags["suspended"]:
        base["reason"] = "停牌"
        return base
    if order["side"] == "buy" and (flags["limit_up"] or flags["one_word"]):
        base["reason"] = "涨停/一字板买不进"
        return base
    if order["side"] == "sell" and (flags["limit_down"] or flags["one_word"] and flags["limit_down"]):
        base["reason"] = "跌停/一字板卖不掉"
        return base
    price = float(row["open"]) * (1 + slip_bp / 1e4 if order["side"] == "buy" else 1 - slip_bp / 1e4)
    amount = order["shares"] * price
    fee = fees_of(amount, order["side"])
    base.update({"fillable": True, "fill_price": price, "filled_shares": order["shares"],
                 "amount": amount, "fees": fee["total"], "effective_bps": fee["effective_bps"],
                 "commission": fee["commission"], "stamp": fee["stamp"],
                 "transfer": fee["transfer"], "slip_bp": slip_bp})
    return base


# --------------------------------------------------------------------------- #
# 选股（复刻 BufferTopK：老仓位在缓冲带内保留 → 按排名补满 → 门类上限）
# --------------------------------------------------------------------------- #
def select(score_row: pd.Series, eligible_row: pd.Series, sector_row: pd.Series,
           held: list[str]) -> list[str]:
    row = score_row.where(eligible_row.reindex(score_row.index)).dropna()
    row = row.sort_values(ascending=False)
    order = list(row.index)
    rank = {symbol: i for i, symbol in enumerate(order)}
    weight_value = min(GROSS / TOP_K, MAX_WEIGHT)
    counts: dict[str, int] = {}

    def capacity_left(key: str) -> bool:
        return (counts.get(key, 0) + 1) * weight_value <= SECTOR_CAP + 1e-12

    chosen: list[str] = []
    for symbol in held:
        if rank.get(symbol, 10**9) >= BUFFER:
            continue
        key = str(sector_row.get(symbol, "UNKNOWN")) if sector_row is not None else "UNKNOWN"
        if not capacity_left(key):
            continue
        chosen.append(symbol)
        counts[key] = counts.get(key, 0) + 1
        if len(chosen) >= TOP_K:
            break
    if len(chosen) < TOP_K:
        for symbol in order:
            if symbol in chosen:
                continue
            key = str(sector_row.get(symbol, "UNKNOWN")) if sector_row is not None else "UNKNOWN"
            if not capacity_left(key):
                continue
            chosen.append(symbol)
            counts[key] = counts.get(key, 0) + 1
            if len(chosen) >= TOP_K:
                break
    return chosen


def weights_of(chosen: list[str]) -> pd.Series:
    if not chosen:
        return pd.Series(dtype="float64")
    value = min(GROSS / TOP_K, MAX_WEIGHT)
    return pd.Series(value, index=chosen)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def refresh_bars(days: int = 460, *, resume: bool = True) -> Path:
    """拉取最近 ``days`` 天全市场日线到 paper 缓存（live 专用，会读 2025+ 数据）。

    只在**正式 paper 跟踪**时运行；产出的面板永不带回研究流程或参数选择。
    """
    from quant.data import web_source
    from quant.refdata import load_universe

    cache = PAPER / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "bars-live.parquet"
    universe = load_universe()
    codes = [str(s) for s in universe["symbol"].astype(str) if str(s)[:2] not in ("43", "83", "87", "88", "92")]
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    bars, used, failed = web_source.fetch_many(codes, start=start, adjust="hfq", verbose=False)
    rows = []
    for symbol, frame in bars.items():
        frame = frame.copy()
        frame["symbol"] = symbol
        rows.append(frame.reset_index())
    if not rows:
        raise SystemExit("refresh_bars: 一只都没抓到，检查网络/接口")
    frame = pd.concat(rows, ignore_index=True)
    frame.to_parquet(target, compression="zstd")
    print(f"[paper] live 缓存 {len(bars)} 只 / {len(frame)} 行 / 失败 {len(failed)} → {target}", flush=True)
    return target


def live_bars() -> dict:
    """数据湖 ≤2024 行情 + live 缓存 → 合并 bars（live 模式用）。"""
    from scripts.redlow_study import board_exclude, load_bars

    cache = PAPER / "cache" / "bars-live.parquet"
    if not cache.exists():
        raise SystemExit(f"缺 {cache}：先跑 --refresh-bars（会读取 2025+ 行情）")
    lake_bars, _ = load_bars()
    live = pd.read_parquet(cache)
    merged = {symbol: frame for symbol, frame in lake_bars.items()}
    for symbol, frame in live.groupby("symbol", sort=False):
        frame = frame.set_index(pd.to_datetime(frame["date"])).drop(columns=["symbol", "date"])
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if symbol in merged:
            old = merged[symbol]
            both = pd.concat([old[old.index < frame.index[0]], frame])
            merged[symbol] = both[~both.index.duplicated(keep="last")].sort_index()
        else:
            merged[symbol] = frame
    for symbol in board_exclude(merged):
        merged.pop(symbol, None)
    return merged


def run(mode: str, start: str | None, end: str | None, live_ack: bool) -> int:
    from scripts import growth_risk as gr
    from scripts import growth_robust as grb

    PAPER.mkdir(parents=True, exist_ok=True)
    state = load_state()
    state["mode"] = mode

    if mode == "live" and not live_ack:
        raise SystemExit("live 模式会读取 2025+ 行情/财报（前向运行必需），需显式 --live-ack")

    if mode == "replay":
        prepared = grb.prepare("main")
    else:
        prepared = grb.prepare("live", bars=live_bars(), start=start, end=end,
                               cutoff=end, alt_kwargs={"cutoff": end, "strict": False},
                               sector_cap=SECTOR_CAP)

    panel, marks = prepared["panel"], prepared["marks"]
    adv20, illiq20 = gr.liquidity_panels(panel)
    slip_maps, _ = gr.tier_map(marks, prepared["eligible"], adv20, illiq20)

    dates = panel.dates
    lo = pd.Timestamp(start) if start else dates[0]
    hi = pd.Timestamp(end) if end else dates[-1]
    window = dates[(dates >= lo) & (dates <= hi)]
    bar_days = set(window)

    ledger_path = PAPER / "ledger.csv"
    ledger_rows: list[dict] = []
    run_started = time.time()

    for day in window:
        # 1) 结算昨日挂单（当日开盘）
        fills_today: list[dict] = []
        for order in state.get("pending", []):
            symbol = order["symbol"]
            row = bar_of(prepared["bars"], symbol, day)
            prev = float(panel.prev_close.at[day, symbol]) if symbol in panel.symbols else np.nan
            adv = float(adv20.at[day, symbol]) if symbol in panel.symbols else np.nan
            slip_bp = float(order.get("slip_bp", 3.0))
            fill = execute(order, row, prev, day, slip_bp)
            fills_today.append(fill)
            if not fill["fillable"]:
                continue
            amount = fill["amount"]
            if order["side"] == "buy":
                state["cash"] -= amount + fill["fees"]
                state["holdings"][symbol] = state["holdings"].get(symbol, 0) + fill["filled_shares"]
            else:
                state["cash"] += amount - fill["fees"]
                left = state["holdings"].get(symbol, 0) - fill["filled_shares"]
                if left > 0:
                    state["holdings"][symbol] = left
                else:
                    state["holdings"].pop(symbol, None)
        state["pending"] = []
        if fills_today:
            pd.DataFrame(fills_today).to_csv(PAPER / f"fills-{day.date()}.csv",
                                             index=False, encoding="utf-8-sig")

        # 2) 收盘估值
        close = panel.close.loc[day]
        market_value = sum(shares * float(close.get(symbol, 0.0) or 0.0)
                           for symbol, shares in state["holdings"].items())
        equity = state["cash"] + market_value
        fees_today = sum(f["fees"] for f in fills_today if f["fillable"])
        traded = sum(f["amount"] for f in fills_today if f["fillable"])
        ledger_rows.append({"date": str(day.date()), "equity": equity, "cash": state["cash"],
                            "market_value": market_value,
                            "positions": len(state["holdings"]),
                            "gross": market_value / equity if equity else 0.0,
                            "fees": fees_today, "traded": traded})

        # 3) 调仓日：生成次日订单
        is_rebalance = day in set(marks)
        if is_rebalance:
            score = prepared["score"].loc[day]
            eligible = prepared["eligible"].loc[day]
            sector = prepared["sector"].loc[day] if prepared["sector"] is not None else None
            chosen = select(score, eligible, sector, list(state["holdings"]))
            target = weights_of(chosen)
            next_days = dates[dates > day]
            if len(next_days):
                nxt = next_days[0]
                open_px = panel.open.loc[nxt]
                orders = plan_orders(target, equity, state["holdings"], open_px)
                tier_today = slip_maps.get(day, {}) if slip_maps else {}
                for order in orders:
                    order["created"] = str(day.date())
                    order["target_weight"] = float(target.get(order["symbol"], 0.0))
                    order["slip_bp"] = float(tier_today.get(order["symbol"], 3.0))
                state["pending"] = orders
                pd.DataFrame(orders).to_csv(PAPER / f"orders-{day.date()}.csv",
                                            index=False, encoding="utf-8-sig")

        state["last_date"] = str(day.date())

    state["run_id"] = int(state.get("run_id", 0)) + 1
    save_state(state)

    if ledger_rows:
        frame = pd.DataFrame(ledger_rows)
        if ledger_path.exists():
            old = pd.read_csv(ledger_path)
            frame = pd.concat([old[~old["date"].isin(frame["date"])], frame], ignore_index=True)
        frame.sort_values("date").to_csv(ledger_path, index=False, encoding="utf-8-sig")
        last = frame.iloc[-1]
        print(f"[paper] {mode} {frame['date'].iloc[0]} → {frame['date'].iloc[-1]} "
              f"净值 {last['equity']:.0f} 持仓 {int(last['positions'])} "
              f"gross {last['gross']:.3f} ({time.time() - run_started:.0f}s)", flush=True)
    save_state(state)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="成长策略 paper 执行校准")
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--live-ack", action="store_true", help="确认 live 会读取 2025+ 数据")
    parser.add_argument("--refresh-bars", action="store_true",
                        help="拉取最近行情到 paper 缓存（live 前置步骤，会读取 2025+ 数据）")
    parser.add_argument("--reset", action="store_true", help="清空 state.json 重来（不删台账）")
    args = parser.parse_args(argv)
    if args.reset:
        (PAPER / "state.json").unlink(missing_ok=True)
    if args.refresh_bars:
        if not args.live_ack:
            raise SystemExit("--refresh-bars 会读取 2025+ 行情，需同时给 --live-ack")
        refresh_bars()
        return 0
    return run(args.mode, args.start, args.end, args.live_ack)


if __name__ == "__main__":
    raise SystemExit(main())
