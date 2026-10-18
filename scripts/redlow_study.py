"""红利低波纯多头组合回测（任务一）。

口径（写进报告时必须原样保留）
- 股票池：沪深两市（剔除北交所），含已退市股；剔除 ST/*ST、上市不满 365 天、停牌、
  一字涨跌停（``high == low`` 且 ``|日收益| >= 9.5%``）、退市整理期（退市前 30 个交易日）；
- 因子：``div_yield``（近 365 天已除息每股现金分红 / 收盘价，PIT）、``vol_20`` / ``vol_60``
  （20/60 日收益波动率取负）、``turnover_20``（20 日平均换手率取负）；
- 持仓：等权，``top_k`` = 30 / 50 / 100；
- 调仓：月频，月末收盘生成信号，**次一交易日开盘价**成交（引擎 ``shift(1)`` @ open）；
- 成本：默认（佣金 2.5bp + 印花税卖出 5bp + 过户费 0.1bp + 滑点 2bp，往返 0.142%），
  另跑往返 0.30% / 0.50% 两档保守成本（按等额提高单边滑点实现）；
- 基准：**同一股票池、同一可交易掩码**的等权月频组合。

holdout 锁：所有数据与统计一律 ``<= 2024-12-31``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quant import valuation  # noqa: E402
from quant.backtest import Engine  # noqa: E402
from quant.config import BacktestConfig, CostModel  # noqa: E402
from quant.factors import style  # noqa: E402,F401  （导入即注册新因子）
from quant.factors.panel import build_panel  # noqa: E402
from quant.refdata import load_universe  # noqa: E402
from quant.strategy.base import Strategy  # noqa: E402
from quant.strategy.multifactor import MultiFactorStrategy  # noqa: E402
from scripts.factor_market import FIELDS, load_delisted_bars, load_delisting_events, load_layer  # noqa: E402

LAKE = Path("D:/quant-data")
HOLDOUT = pd.Timestamp("2025-01-01")
START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2024-12-31")
MONTHLY = "ME"
STRESS_WINDOWS = [("2024-01-01", "2024-02-29")]


# --------------------------------------------------------------------------- 数据
def load_bars(*, smoke: int | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    frames, stats = load_layer(LAKE, "processed", limit=smoke)
    delisted = load_delisted_bars(LAKE)
    added = 0
    for symbol, frame in delisted.items():
        if symbol in frames:
            continue
        frames[symbol] = frame
        added += 1
    clipped: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        part = frame.loc[(frame.index >= "2014-01-01") & (frame.index < HOLDOUT)]
        part = part.dropna(subset=["open", "high", "low", "close", "volume", "amount", "prev_close"])
        if len(part) >= 60:
            clipped[symbol] = part
    stats = {**stats, "symbols": len(clipped), "delisted_added": added}
    return clipped, stats


def board_exclude(symbols) -> set[str]:
    """北交所代码（4/8 开头的 43/83/87/88/92 段）不进池。"""
    out = set()
    for symbol in symbols:
        code = str(symbol).split(".")[0]
        if code[:2] in {"43", "83", "87", "88", "92"}:
            out.add(symbol)
    return out


def build_block(panel) -> pd.DataFrame:
    """可交易掩码之外的全部剔除项：ST / 次新 / 停牌 / 一字板 / 退市整理期。"""
    index, columns = panel.dates, panel.symbols
    block = pd.DataFrame(False, index=index, columns=columns)

    st = valuation.st_matrix(list(columns))
    block |= st.reindex(index=index, columns=columns).fillna(False).astype(bool)

    universe = load_universe()
    ipo = dict(zip(universe["symbol"].astype(str), pd.to_datetime(universe["ipo_date"], errors="coerce")))
    for position, symbol in enumerate(columns):
        start = ipo.get(str(symbol).split(".")[0])
        if start is None or pd.isna(start):
            continue
        block.iloc[:, position] |= (index - start).days < 365

    block |= panel.volume.fillna(0.0).le(0)
    one_price = (panel.high - panel.low).abs().le(panel.close * 1e-4)
    limit_move = panel.close.div(panel.prev_close).sub(1.0).abs().ge(0.095)
    block |= one_price & limit_move

    events = load_delisting_events(LAKE)
    for position, symbol in enumerate(columns):
        last = events.get(symbol) or events.get(str(symbol).split(".")[0])
        if last is None:
            continue
        cutoff = last - pd.Timedelta(days=45)
        block.iloc[:, position] |= (index >= cutoff) & (index <= last)
    return block


# --------------------------------------------------------------------------- 基准
class EqualWeightPool(Strategy):
    """同一可交易池的等权月频组合（基准列）。

    注意：**不要**用 Engine 跑这个基准——8 万只（这里是 5 千只）等权时单票目标金额
    只有几百元、不足 1 手，`broker.plan_orders` 会全部拒单，净值变成「几乎全现金」
    （实测年化波动 0.64%、Beta 12+）。基准必须用净值解析式算：见 `pool_benchmark`。
    """

    name = "equal-weight-pool"

    def __init__(self, block: pd.DataFrame, *, gross: float = 0.95) -> None:
        super().__init__(gross=gross)
        self.block = block
        self.gross = gross

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = pd.DataFrame({symbol: frame["close"] for symbol, frame in bars.items()}).sort_index()
        eligible = close.notna() & ~self.block.reindex(index=close.index, columns=close.columns).fillna(True)
        count = eligible.sum(axis=1).replace(0, np.nan)
        weights = eligible.astype(float).div(count, axis=0) * self.gross
        weights = weights.fillna(0.0)
        return self.hold_to_rebalance(weights, MONTHLY)


# --------------------------------------------------------------------------- 评估
def cost_presets() -> dict[str, CostModel]:
    base = CostModel()
    return {
        "default": base,
        "conservative_30": CostModel(slippage_bps=9.9),
        "conservative_50": CostModel(slippage_bps=19.9),
    }


def round_trip_bp(costs: CostModel) -> float:
    buy = costs.commission_rate + costs.transfer_fee_rate + costs.slippage_bps / 1e4
    sell = buy + costs.stamp_duty_rate
    return round((buy + sell) * 1e4, 2)


def monthly_returns(daily: pd.DataFrame) -> pd.Series:
    """按自然月复利，含样本第一个月（不丢 2017-01）。"""
    ret = daily["equity"].pct_change().dropna()
    if ret.empty:
        return pd.Series(dtype=float)
    grouped = (1.0 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1.0
    index = pd.to_datetime([f"{year}-{month:02d}-01" for year, month in grouped.index]).to_period("M").to_timestamp("M")
    return pd.Series(grouped.to_numpy(), index=index)


def yearly_returns(daily: pd.DataFrame) -> pd.Series:
    """按自然年复利，含样本第一个年（不丢 2017）。"""
    ret = daily["equity"].pct_change().dropna()
    if ret.empty:
        return pd.Series(dtype=float)
    grouped = (1.0 + ret).groupby(ret.index.year).prod() - 1.0
    index = pd.to_datetime([f"{year}-12-31" for year in grouped.index])
    return pd.Series(grouped.to_numpy(), index=index)


# --------------------------------------------------------------------------- 基准（解析式）
def pool_benchmark(
    panel,
    block: pd.DataFrame,
    delisting: Mapping[str, pd.Timestamp] | None = None,
    *,
    cash: float = 1_000_000.0,
    gross: float = 0.95,
) -> pd.Series:
    """同池等权、月频再平衡的基准净值（解析计算，不走撮合）。

    口径：每月末对「当月首个交易日可交易且未剔除」的股票等权配置，月内按
    等权平均收益（相当于每日再平衡，与月度漂移的差异 <1bp）。退市股在最后
    一个有效交易日按 haircut30 记 −30%（与策略的退市结算假设一致）。
    """
    returns = panel.returns.copy()
    mask = block.reindex(index=returns.index, columns=returns.columns).fillna(True).astype(bool)
    returns = returns.mask(mask)
    if delisting:
        for symbol, _day in delisting.items():
            if symbol not in returns.columns:
                continue
            column = returns[symbol].dropna()
            if len(column):
                returns.loc[column.index[-1], symbol] = -0.30
    months = returns.index.to_period("M")
    pieces = []
    for month in months.unique():
        part = returns.loc[months == month]
        if not len(part):
            continue
        available = part.iloc[0].notna()
        if not available.any():
            continue
        pieces.append(part.loc[:, available].mean(axis=1))
    if not pieces:
        return pd.Series(dtype=float)
    daily = pd.concat(pieces).sort_index()
    equity = cash * (1.0 + daily * gross).cumprod()
    equity.iloc[0] = cash
    return equity


def summarize_benchmark(equity: pd.Series) -> dict:
    ret = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = len(ret) / 244.0
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 else None
    vol = float(ret.std(ddof=1) * np.sqrt(244))
    drawdown = float((equity / equity.cummax() - 1).min())
    month = (1.0 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1.0
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": float(cagr / vol) if vol > 0 and cagr is not None else None,
        "max_drawdown": drawdown,
        "calmar": float(cagr / abs(drawdown)) if drawdown < 0 and cagr is not None else None,
        "win_rate": float((ret > 0).mean()),
        "turnover": 0.0,
        "fees": 0.0,
        "trades": 0,
        "beta": 1.0,
        "alpha": 0.0,
        "info_ratio": 0.0,
        "monthly_win_rate": round(float((month > 0).mean()), 4),
        "month_count": int(len(month)),
        "benchmark_total": 0.0,
        "excess_total": 0.0,
        "excess_ir": 0.0,
        "excess_win_rate": 0.0,
        "avg_turnover": 0.0,
    }


def extra_metrics(daily: pd.DataFrame, benchmark: pd.Series, metrics: dict) -> dict:
    equity = daily["equity"]
    month = monthly_returns(daily)
    bench = benchmark.reindex(equity.index).ffill()
    strategy_ret = equity.pct_change()
    bench_ret = bench.pct_change()
    excess = (strategy_ret - bench_ret).dropna()
    excess_cum = float((1 + strategy_ret).prod() / max((1 + bench_ret).prod(), 1e-12) - 1)
    out = {
        "calmar": metrics.get("calmar"),
        "monthly_win_rate": round(float((month > 0).mean()), 4),
        "month_count": int(len(month)),
        "benchmark_total": round(float(bench.iloc[-1] / bench.iloc[0] - 1), 6),
        "excess_total": round(excess_cum, 6),
        "excess_ir": round(float(excess.mean() / excess.std(ddof=1) * np.sqrt(244)), 3) if excess.std() > 0 else None,
        "excess_win_rate": round(float((excess > 0).mean()), 4),
        "avg_turnover": metrics.get("avg_turnover"),
    }
    return out


def window_stats(daily: pd.DataFrame, start: str, end: str, bars: dict[str, pd.DataFrame], holdings: list[str]) -> dict:
    equity = daily["equity"].loc[start:end]
    if len(equity) < 2:
        return {"window": f"{start}~{end}", "trading_days": 0}
    ret = equity.pct_change().dropna()
    running = equity.cummax()
    drawdown = equity / running - 1.0
    worst_streak, current = 0, 0
    for value in ret:
        current = current + 1 if value < 0 else 0
        worst_streak = max(worst_streak, current)
    # 恢复：区间末之后净值重回区间起点
    after = daily["equity"].loc[pd.Timestamp(end):]
    target = float(equity.iloc[0])
    recovered = after[after >= target]
    recover_days = int(len(after.loc[: recovered.index[0]])) - 1 if len(recovered) else None

    streaks = []
    for symbol in holdings:
        frame = bars.get(symbol)
        if frame is None:
            continue
        part = frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
        if len(part) < 2:
            continue
        ret_s = part["close"].pct_change()
        locked = (part["high"] - part["low"]).abs().le(part["close"] * 1e-4) & ret_s.abs().ge(0.095)
        best, run = 0, 0
        for flag in locked.fillna(False):
            run = run + 1 if flag else 0
            best = max(best, run)
        streaks.append(best)
    return {
        "window": f"{start}~{end}",
        "trading_days": int(len(equity)),
        "window_return": round(float(equity.iloc[-1] / equity.iloc[0] - 1), 6),
        "max_drawdown": round(float(drawdown.min()), 6),
        "worst_day": round(float(ret.min()), 6),
        "worst_streak_down": int(worst_streak),
        "days_below_minus3pct": int((ret < -0.03).sum()),
        "holdings_checked": len(streaks),
        "max_limit_down_streak": int(max(streaks)) if streaks else None,
        "median_limit_down_streak": float(np.median(streaks)) if streaks else None,
        "recover_days": recover_days,
        "recovered": recover_days is not None,
    }


def summarize(result, benchmark_equity: pd.Series, bars, holdings) -> dict:
    metrics = result.report.metrics
    row = {
        "total_return": metrics.get("total_return"),
        "cagr": metrics.get("cagr"),
        "ann_vol": metrics.get("ann_vol"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "calmar": metrics.get("calmar"),
        "win_rate": metrics.get("win_rate"),
        "turnover": metrics.get("avg_turnover"),
        "fees": metrics.get("total_fees"),
        "trades": metrics.get("trade_count"),
        "beta": metrics.get("beta"),
        "alpha": metrics.get("alpha"),
        "info_ratio": metrics.get("info_ratio"),
    }
    row.update(extra_metrics(result.daily, benchmark_equity, metrics))
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="红利低波组合回测")
    parser.add_argument("--smoke", type=int, default=None, help="只读前 N 只（冒烟）")
    parser.add_argument("--out", default="results/redlow")
    parser.add_argument("--topk", default="30,50,100")
    parser.add_argument("--single-topk", type=int, default=50)
    parser.add_argument("--start", default=str(START.date()))
    parser.add_argument("--end", default=str(END.date()))
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    bars, load_stats = load_bars(smoke=args.smoke)
    print(f"bars {len(bars)} 只（含退市 {load_stats['delisted_added']}）", flush=True)
    drop = board_exclude(bars)
    for symbol in drop:
        bars.pop(symbol, None)
    print(f"剔除北交所 {len(drop)} 只 → {len(bars)} 只", flush=True)

    panel = build_panel(bars, fields=FIELDS)
    panel = panel.slice(pd.Timestamp(args.start), pd.Timestamp(args.end))
    print(f"panel {panel.close.shape} {panel.dates[0].date()} ~ {panel.dates[-1].date()}", flush=True)

    block = build_block(panel)
    block = block.reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)
    print(f"剔除占比 {block.to_numpy().mean():.2%}", flush=True)

    config = BacktestConfig(start=args.start, end=args.end, initial_cash=args.cash, benchmark=None)
    presets = cost_presets()
    print("成本档（往返 bp）：", {key: round_trip_bp(value) for key, value in presets.items()}, flush=True)
    # Engine 的日历取自传入 bars（不看 config.start/end）：因子要用 2014–2016 预热，
    # 但净值与统计必须从 START(2017-01-01) 开始，所以权重在长 bars 上生成、回测用裁剪后的 bars。
    bt_bars = {symbol: frame.loc[START:END] for symbol, frame in bars.items() if len(frame)}
    print(f"回测区间 {bt_bars[list(bt_bars)[0]].index[0].date()} ~ {bt_bars[list(bt_bars)[0]].index[-1].date()}", flush=True)

    # 基准：同池等权（解析式计算；用 Engine 撮合会因单票不足 1 手而全拒单）
    delisting_events = load_delisting_events(LAKE)
    bench_equity = pool_benchmark(panel, block, delisting_events, cash=args.cash)
    bench_equity.to_csv(out / "benchmark-equity.csv", encoding="utf-8-sig")
    bench_row = {"config": "benchmark_equal_weight", **summarize_benchmark(bench_equity)}
    print(f"基准：累计 {bench_row['total_return']:.2%} CAGR {bench_row['cagr']:.2%}", flush=True)
    rows = [bench_row]

    def run_config(name: str, factors: str, top_k: int, cost_key: str) -> dict:
        strategy = MultiFactorStrategy(
            factors=factors,
            weighting="equal",
            top_k=top_k,
            rebalance=MONTHLY,
            gross_exposure=0.95,
            max_weight=0.10,
            min_price=2.0,
            min_amount=0.0,
            min_history=120,
            min_universe=100,
            exclude=block,
        )
        cfg = BacktestConfig(start=args.start, end=args.end, initial_cash=args.cash, costs=presets[cost_key],
                             benchmark=None)
        engine = Engine(config=cfg, benchmark_prices=(bench_equity / args.cash).ffill(),
                        delisting=load_delisting_events(LAKE), delisting_mode="haircut30")
        started = time.time()
        weights = strategy.generate(bars)
        result = engine.run(bt_bars, weights, strategy_name=strategy.describe())
        elapsed = time.time() - started
        picks = strategy.last_diagnostics.get("picks", {})
        last_holdings = list(picks.values())[-1] if picks else []
        row = {
            "config": name,
            "factors": factors,
            "top_k": top_k,
            "cost": cost_key,
            "round_trip_bp": round_trip_bp(presets[cost_key]),
            **summarize(result, bench_equity, bars, last_holdings),
        }
        out.mkdir(parents=True, exist_ok=True)
        result.equity.to_csv(out / f"equity-{name}.csv", encoding="utf-8-sig")
        if len(result.trades):
            result.trades.to_csv(out / f"trades-{name}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"equity": result.daily["equity"]}).to_csv(out / f"daily-{name}.csv", encoding="utf-8-sig")
        print(
            f"[{name}] 累计 {row['total_return']:.2%} CAGR {row['cagr']:.2%} 夏普 {row['sharpe']} "
            f"MDD {row['max_drawdown']:.2%} α {row['alpha']:.2%} 超额 {row['excess_total']:.2%} "
            f"换手 {row['turnover']:.1%} 费用 {row['fees']:,.0f}（{elapsed:.0f}s）",
            flush=True,
        )
        return row, result, picks

    results = {}
    for factor in ("div_yield", "vol_20", "vol_60", "turnover_20"):
        name = f"single-{factor}"
        row, result, holdings = run_config(name, factor, args.single_topk, "default")
        rows.append(row)
        results[name] = (result, holdings)

    combo_factors = "div_yield,vol_60,turnover_20"
    for top_k in [int(value) for value in str(args.topk).split(",") if value.strip()]:
        name = f"combo-{top_k}"
        row, result, holdings = run_config(name, combo_factors, top_k, "default")
        rows.append(row)
        results[name] = (result, holdings)
        for cost_key in ("conservative_30", "conservative_50"):
            extra_name = f"combo-{top_k}-{cost_key}"
            extra_row, extra_result, extra_holdings = run_config(extra_name, combo_factors, top_k, cost_key)
            rows.append(extra_row)
            results[extra_name] = (extra_result, extra_holdings)

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")

    # 分年度收益（2017-2024）
    yearly = {}
    for name, (result, _) in results.items():
        yearly[name] = yearly_returns(result.daily).rename(name)
    yearly_table = pd.DataFrame(yearly)
    yearly_table.index = yearly_table.index.year
    yearly_table.to_csv(out / "yearly.csv", encoding="utf-8-sig")

    # 月胜率 / 月度收益明细（组合 30 只）
    monthly = pd.DataFrame({name: monthly_returns(result.daily) for name, (result, _) in results.items()})
    monthly.to_csv(out / "monthly.csv", encoding="utf-8-sig")

    # 压力区间：持仓取「区间开始前最后一次调仓」的组合
    def picks_at(picks: dict, day: pd.Timestamp) -> list[str]:
        keys = sorted(key for key in picks if pd.Timestamp(key) <= day)
        return list(picks[keys[-1]]) if keys else []

    stress_rows = []
    for start, end in STRESS_WINDOWS:
        bench_stress = window_stats(pd.DataFrame({"equity": bench_equity}), start, end, bars, [])
        bench_stress["config"] = "benchmark_equal_weight"
        stress_rows.append(bench_stress)
    for name in ("combo-30", "combo-50", "combo-100"):
        if name not in results:
            continue
        result, picks = results[name]
        for start, end in STRESS_WINDOWS:
            row = window_stats(result.daily, start, end, bars, picks_at(picks, pd.Timestamp(start)))
            row["config"] = name
            stress_rows.append(row)
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(out / "stress.csv", index=False, encoding="utf-8-sig")

    meta = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "seconds": round(time.time() - t0, 1),
        "load_stats": load_stats,
        "panel": {"dates": int(len(panel.dates)), "symbols": int(len(panel.symbols)),
                  "first": str(panel.dates[0].date()), "last": str(panel.dates[-1].date())},
        "block_share": round(float(block.to_numpy().mean()), 4),
        "cost_bp": {key: round_trip_bp(value) for key, value in presets.items()},
        "holdout_locked_from": str(HOLDOUT.date()),
        "stress_windows": STRESS_WINDOWS,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 {meta['seconds']}s → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
