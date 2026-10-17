"""回测引擎测试：A股规则（T+1/涨跌停/整手/费用/现金）、指标、overlay、报表。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import setup_console, tmpdir

setup_console()

from quant.backtest import Broker, Engine, Portfolio, run_backtest
from quant.backtest.metrics import build_report, max_drawdown, total_return
from quant.config import BacktestConfig, CostModel, MarketRules
from quant.data import DataStore, SyntheticSource, make_gbm_frame
from quant.report import write_report
from quant.risk import exposure_cap, max_weight, vol_target
from quant.strategy import BuyAndHold, MaCross, MomentumRank


def bars_fixture(periods: int = 300, symbols=("600519.SH", "000001.SZ")):
    return {s: make_gbm_frame(s, periods=periods, seed=100 + i) for i, s in enumerate(symbols)}


# ------------------------------------------------------------------ 规则
def test_t_plus_one_blocks_same_day_sell():
    bars = bars_fixture(10, ("A.SH",))
    pf = Portfolio(cash=1_000_000.0)
    broker = Broker()
    day0, day1 = bars["A.SH"].index[0], bars["A.SH"].index[1]
    price = float(bars["A.SH"].loc[day0, "open"])

    from quant.backtest.broker import Order

    pf.mark_all({s: float(f.loc[day0, "open"]) for s, f in bars.items()}, day0)
    fills = broker.execute(day0, [Order(day0, "A.SH", "buy", 1000, "test", 1.0, price)], bars, pf)
    assert len(fills) == 1 and pf.shares("A.SH") == 1000
    assert pf.positions["A.SH"].available == 0, "当日买入不可卖"

    pf.mark_all({s: float(f.loc[day0, "open"]) for s, f in bars.items()}, day0)
    broker.execute(day0, [Order(day0, "A.SH", "sell", 1000, "test", 0.0, price)], bars, pf)
    assert pf.shares("A.SH") == 1000, "T+1 当日不应卖出"
    assert any("T+1" in r.reason for r in broker.rejections)

    pf.settle()
    pf.mark_all({s: float(f.loc[day1, "open"]) for s, f in bars.items()}, day1)
    fills2 = broker.execute(day1, [Order(day1, "A.SH", "sell", 1000, "test", 0.0, float(bars["A.SH"].loc[day1, "open"]))], bars, pf)
    assert len(fills2) == 1 and pf.shares("A.SH") == 0, "次日应可卖出"
    print("  ✓ T+1：当日买入不可卖，次日可卖")


def test_limit_up_blocks_buy_and_limit_down_blocks_sell():
    idx = pd.bdate_range("2024-01-01", periods=3, name="date")
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 9.0],
            "high": [10.2, 11.0, 9.2],
            "low": [9.8, 11.0, 9.0],
            "close": [10.0, 11.0, 9.0],
            "volume": [1e6] * 3,
            "prev_close": [10.0, 10.0, 11.0],
        },
        index=idx,
    )
    bars = {"X.SH": frame}
    broker = Broker()
    pf = Portfolio(cash=1_000_000.0)
    from quant.backtest.broker import Order

    # 第二天开盘 11.0 = 10.0 的涨停价 → 不可买
    broker.execute(idx[1], [Order(idx[1], "X.SH", "buy", 1000, "t", 1.0, 11.0)], bars, pf)
    assert pf.shares("X.SH") == 0
    assert any("涨停" in r.reason for r in broker.rejections)

    # 持有 1000 股，第三天开盘 9.0 = 11.0 的跌停价（-18% 但按规则算 11*0.9=9.9，这里直接用超限价）→ 不可卖
    broker2 = Broker()
    pf2 = Portfolio(cash=0.0)
    pf2.positions["X.SH"] = __import__("quant.backtest.portfolio", fromlist=["Position"]).Position("X.SH", shares=1000, available=1000, cost_basis=10.0, last_price=11.0)
    broker2.execute(idx[2], [Order(idx[2], "X.SH", "sell", 1000, "t", 0.0, 9.0)], bars, pf2)
    assert pf2.shares("X.SH") == 1000 and any("跌停" in r.reason for r in broker2.rejections)
    print("  ✓ 涨停买不到、跌停卖不出")


def test_lot_size_and_cash_constraint():
    bars = bars_fixture(10, ("A.SH",))
    day = bars["A.SH"].index[0]
    price = float(bars["A.SH"].loc[day, "open"])
    broker = Broker()
    pf = Portfolio(cash=100_000.0)
    target_weight = 0.5
    equity = pf.equity
    orders = broker.plan_orders(day, {"A.SH": target_weight}, pf, bars, price_field="open")
    assert len(orders) == 1
    assert orders[0].shares % 100 == 0, "买入必须是整手"
    fills = broker.execute(day, orders, bars, pf)
    assert len(fills) == 1
    assert pf.shares("A.SH") % 100 == 0
    invested = pf.shares("A.SH") * fills[0].price
    assert 0.45 * equity < invested < 0.52 * equity, f"目标 50% 仓位实际 {invested / equity:.1%}"

    # 现金不足：只给 5000 元却想买 100 元 × 100 股以上
    poor = Portfolio(cash=5_000.0)
    broker_poor = Broker()
    orders_poor = broker_poor.plan_orders(day, {"A.SH": 1.0}, poor, bars, price_field="open")
    fills_poor = broker_poor.execute(day, orders_poor, bars, poor)
    assert poor.cash >= 0, "现金不允许为负"
    for fill in fills_poor:
        assert fill.shares % 100 == 0
    print("  ✓ 整手约束与现金约束生效（现金%d → 成交%d笔）" % (5_000, len(fills_poor)))


def test_costs_charged():
    bars = bars_fixture(40, ("A.SH",))
    cfg = BacktestConfig(initial_cash=500_000.0, costs=CostModel(slippage_bps=0.0), rules=MarketRules())
    result = run_backtest(bars, BuyAndHold(), cfg)
    assert len(result.trades) >= 1
    buys = result.trades[result.trades["side"] == "buy"]
    assert (buys["fee"] > 0).all(), "买入必须收佣金/过户费"
    assert result.metrics["total_fees"] > 0
    manual = sum(
        max(f["gross"] * cfg.costs.commission_rate, cfg.costs.commission_min)
        + f["gross"] * cfg.costs.transfer_fee_rate
        + (f["gross"] * cfg.costs.stamp_duty_rate if f["side"] == "sell" else 0.0)
        for _, f in result.trades.iterrows()
    )
    assert abs(manual - result.trades["fee"].sum()) < 1.0
    print("  ✓ 佣金/印花税/过户费计入且与手工计算一致")


def test_slippage_cost_sign():
    bars = bars_fixture(40, ("A.SH",))
    cfg = BacktestConfig(initial_cash=500_000.0, costs=CostModel(slippage_bps=10.0))
    result = run_backtest(bars, BuyAndHold(), cfg)
    buys = result.trades[result.trades["side"] == "buy"]
    assert (buys["slippage_cost"] > 0).all(), "买入滑点成本应为正"
    print("  ✓ 滑点按单边基点计价并单独记录")


# ------------------------------------------------------------------ 引擎
def test_engine_no_lookahead_and_equity_accounting():
    bars = bars_fixture(250, ("A.SH",))
    cfg = BacktestConfig(initial_cash=1_000_000.0)
    result = run_backtest(bars, BuyAndHold(), cfg)
    eq = result.equity
    assert eq.iloc[0] == cfg.initial_cash
    # equity = cash + market_value
    daily = result.daily
    diff = (daily["equity"] - (daily["cash"] + daily["market_value"])).abs().max()
    assert diff < 1e-6, f"净值与现金+市值不一致: {diff}"
    # 首日不可能已有持仓（权重 shift 一天）
    first_day = bars["A.SH"].index[0]
    assert result.held.loc[first_day].abs().sum() == 0, "首日不应持仓（信号需滞后一天执行）"
    price_ret = bars["A.SH"]["close"].iloc[-1] / bars["A.SH"]["open"].iloc[1] - 1.0
    strat_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    assert 0 < strat_ret <= price_ret + 0.02, f"策略收益 {strat_ret:.2%} 不应超过标的 {price_ret:.2%}"
    print("  ✓ 引擎无未来函数、净值闭合、收益不超过标的")


def test_rotation_strategy_runs():
    bars = bars_fixture(500, ("600519.SH", "000001.SZ", "300750.SZ"))
    cfg = BacktestConfig(initial_cash=800_000.0)
    result = run_backtest(bars, MomentumRank(lookback=20, top_k=2, rebalance="W-FRI"), cfg)
    assert result.metrics["trade_count"] > 0
    assert result.metrics["max_drawdown"] <= 0
    assert np.isfinite(result.metrics["sharpe"])
    assert (result.held.sum(axis=1) <= 1.0 + 1e-9).all(), "总仓位不得超过 100%"
    assert len(result.rejections) >= 0
    print("  ✓ 轮动策略跑通：%d 笔成交、%d 个未成交记录、最大回撤 %.2f%%"
          % (result.metrics["trade_count"], len(result.rejections), result.metrics["max_drawdown"] * 100))


def test_legacy_monthly_alias_is_not_daily():
    """pandas 3 不再认 "M"；翻译失败会静默退化成每日调仓（成交数暴涨），必须挡住。"""
    from quant.strategy.base import normalize_freq

    assert normalize_freq("M") == "ME"
    assert normalize_freq("Q") == "QE"
    assert normalize_freq("W-FRI") == "W-FRI"

    bars = bars_fixture(300, ("600519.SH", "000001.SZ", "300750.SZ"))
    cfg = BacktestConfig(initial_cash=800_000.0)
    strategy = MomentumRank(lookback=20, top_k=2, rebalance="M")
    index = None
    result = run_backtest(bars, strategy, cfg)
    index = result.equity.index
    mask = strategy.rebalance_mask(index, "M")
    monthly_marks = int(mask.sum())
    # 300 个交易日 ≈ 12 个月；允许每月末落在非交易日时略少
    assert 8 <= monthly_marks <= 16, f"M 应折算为月度调仓，实际调仓日 {monthly_marks} 个"
    assert monthly_marks < len(index) / 5, "M 被当成每日调仓了"

    try:
        strategy.rebalance_mask(index, "不是频率")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("非法频率应当报错，而不是静默退化为每日调仓")
    print(f"  ✓ 旧别名 M→ME：{monthly_marks} 个调仓日（{len(index)} 个交易日）")


def test_min_trade_value_filters_drift_trades():
    """最小调仓金额：低于阈值的小额漂移不再每天下碎单。"""
    import dataclasses

    bars = bars_fixture(300, ("600519.SH", "000001.SZ", "300750.SZ"))
    base_cfg = BacktestConfig(initial_cash=800_000.0)
    strategy = MomentumRank(lookback=20, top_k=2, rebalance="W-FRI")
    loose = run_backtest(bars, strategy, base_cfg)
    strict = run_backtest(bars, strategy, dataclasses.replace(base_cfg, min_trade_value=200_000.0))
    assert strict.metrics["trade_count"] < loose.metrics["trade_count"]
    assert strict.metrics["total_fees"] <= loose.metrics["total_fees"] + 1e-6
    print(f"  ✓ 最小调仓金额生效：成交 {loose.metrics['trade_count']} → {strict.metrics['trade_count']} 笔，"
          f"费用 {loose.metrics['total_fees']:.0f} → {strict.metrics['total_fees']:.0f} 元")


def test_overlays_reduce_exposure():
    bars = bars_fixture(400, ("A.SH", "B.SZ", "C.SZ"))
    cfg = BacktestConfig(initial_cash=1_000_000.0)
    base = run_backtest(bars, MomentumRank(lookback=20, top_k=1, rebalance="W-FRI"), cfg)
    capped = run_backtest(
        bars,
        MomentumRank(lookback=20, top_k=1, rebalance="W-FRI"),
        cfg,
        overlays=[max_weight(0.3), exposure_cap(0.8), vol_target(0.10)],
    )
    assert capped.targets.max().max() <= 0.3 + 1e-9
    assert capped.targets.sum(axis=1).max() <= 0.8 + 1e-9
    assert capped.metrics["avg_exposure"] <= base.metrics["avg_exposure"] + 1e-9
    print("  ✓ overlay 生效：单票≤30%%、总仓≤80%%、波动目标压低暴露(%.1f%%→%.1f%%)"
          % (base.metrics["avg_exposure"] * 100, capped.metrics["avg_exposure"] * 100))


def test_metrics_known_values():
    eq = pd.Series([100.0, 120.0, 90.0, 150.0], index=pd.bdate_range("2024-01-01", periods=4))
    assert abs(total_return(eq) - 0.5) < 1e-12
    assert abs(max_drawdown(eq) + 0.25) < 1e-12
    rep = build_report(eq)
    assert rep.metrics["max_drawdown"] == -0.25
    assert rep.metrics["mdd_peak"] == "2024-01-02" and rep.metrics["mdd_trough"] == "2024-01-03"
    assert "最大回撤" in rep.to_text()
    print("  ✓ 绩效指标与最大回撤区间计算正确")


def test_report_files_written():
    bars = bars_fixture(200, ("A.SH", "B.SZ"))
    result = run_backtest(bars, MaCross(5, 20), BacktestConfig(initial_cash=300_000.0))
    with tmpdir("report") as tmp:
        out = write_report(result, tmp, "t")
        for name in ("equity.csv", "daily.csv", "metrics.json", "report.md", "equity.svg"):
            assert (out / name).is_file(), f"缺少 {name}"
        svg = (out / "equity.svg").read_text(encoding="utf-8")
        assert svg.startswith("<svg") and "polyline" in svg
        md = (out / "report.md").read_text(encoding="utf-8")
        assert "累计收益" in md and "最大回撤" in md
    print("  ✓ 报表落盘：CSV/JSON/Markdown/SVG 齐全")


def test_benchmark_prices_used_for_alpha():
    """给了基准收盘价，就按它算 Alpha；否则会退回「全池等权指数」。"""
    bars = bars_fixture(240, ("A.SH", "B.SZ"))
    calendar = sorted(bars["A.SH"].index)
    # 构造一条与 A 走势相反的基准，确保 alpha/beta 不是等权指数的结果
    closes = bars["A.SH"]["close"]
    benchmark = pd.Series(1000.0 * (1.5 - closes / closes.iloc[0]), index=calendar)
    cfg = BacktestConfig(initial_cash=200_000.0, benchmark="INDEX.SH")

    engine = Engine(config=cfg, benchmark_prices=benchmark)
    result = engine.run_strategy(bars, BuyAndHold())

    expected = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)
    assert abs(result.metrics["benchmark_return"] - expected) < 1e-9, (
        f"基准收益应为 {expected:.4f}，实际 {result.metrics['benchmark_return']:.4f}"
    )
    assert not result.benchmark.dropna().empty
    # 基准没有混进 bars，否则会变成第 3 只可交易标的
    assert set(result.targets.columns) == {"A.SH", "B.SZ"}

    # 不给基准 → 退回等权指数（不同结果），且不报错
    plain = run_backtest(bars, BuyAndHold(), cfg)
    assert abs(plain.metrics["benchmark_return"] - expected) > 1e-9
    print("  ✓ 基准序列按传入的收盘价计算（不再错用全池等权指数）")


def test_delisting_settlement_modes():
    """退市结算：持有到退市日的仓位必须按假设折价变现，三种方式净值严格递减。"""
    bars = bars_fixture(60, ("A.SH", "D.SZ"))
    last_bar = bars["D.SZ"].index[20]
    # D.SZ 只交易 21 天就退市，之后没有行情；A.SH 一直交易
    bars["D.SZ"] = bars["D.SZ"].iloc[:21]
    cfg = BacktestConfig(initial_cash=1_000_000.0, universe=("A.SH", "D.SZ"))

    results = {}
    for mode in ("last", "haircut30", "total_loss"):
        result = run_backtest(bars, BuyAndHold(), cfg, delisting={"D.SZ": last_bar}, delisting_mode=mode)
        results[mode] = result
        assert result.delistings is not None and len(result.delistings) == 1, f"{mode} 应结算 1 笔退市"
        row = result.delistings.iloc[0]
        assert row["symbol"] == "D.SZ"
        assert row["date"] > last_bar, "退市结算必须发生在最后一个交易日之后"
        assert row["factor"] == {"last": 1.0, "haircut30": 0.7, "total_loss": 0.0}[mode]

    final = {mode: float(r.equity.iloc[-1]) for mode, r in results.items()}
    assert final["last"] > final["haircut30"] > final["total_loss"], final
    # 折价额按「结算当日净值」核对：之后组合会拿回笼现金再投资，终值差额会被放大
    settle_date = results["haircut30"].delistings.iloc[0]["date"]
    shares = float(results["haircut30"].delistings.iloc[0]["shares"])
    price = float(results["haircut30"].delistings.iloc[0]["last_price"])
    eq_last = float(results["last"].daily.loc[settle_date, "equity"])
    eq_hair = float(results["haircut30"].daily.loc[settle_date, "equity"])
    assert abs((eq_last - eq_hair) - 0.3 * shares * price) < 1e-6, (eq_last - eq_hair, 0.3 * shares * price)
    plain = run_backtest(bars, BuyAndHold(), cfg)
    assert plain.delistings is not None and len(plain.delistings) == 0

    try:
        run_backtest(bars, BuyAndHold(), cfg, delisting={"D.SZ": last_bar}, delisting_mode="whatever")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("未知退市结算方式应报 ValueError")
    print("  ✓ 退市结算：last > haircut30 > total_loss，折价额等于持仓市值×折扣")


def test_paper_trader_flow():
    from quant.execution import PaperTrader

    bars = {
        s: make_gbm_frame(s, start="2023-01-01", periods=300, seed=200 + i)
        for i, s in enumerate(("600519.SH", "000001.SZ"))
    }

    class FixedSource:
        def load(self, symbol, start=None, end=None):
            return bars[symbol]

        def symbols(self):
            return list(bars)

    store = DataStore(FixedSource())
    cfg = BacktestConfig(
        initial_cash=200_000.0,
        universe=("600519.SH", "000001.SZ"),
        start="2023-01-01",
        end="2024-03-01",
    )
    with tmpdir("paper") as tmp:
        trader = PaperTrader(store=store, strategy=BuyAndHold(), config=cfg, state_dir=tmp)
        plan = trader.plan()
        assert isinstance(plan, pd.DataFrame)
        if len(plan):
            first = plan.iloc[0]
            trader.apply_fills([{"symbol": first["symbol"], "side": "buy", "shares": int(first["shares"]), "price": float(first["ref_price"])}])
            assert trader.portfolio.shares(first["symbol"]) > 0
        trader.mark_to_market()
        status = trader.status()
        assert "总权益" in status
        assert (Path(tmp) / "account.json").is_file()
    print("  ✓ 模拟盘：下单建议 → 成交录入 → 净值落盘闭环")


def run() -> None:
    for fn in (
        test_t_plus_one_blocks_same_day_sell,
        test_limit_up_blocks_buy_and_limit_down_blocks_sell,
        test_lot_size_and_cash_constraint,
        test_costs_charged,
        test_slippage_cost_sign,
        test_engine_no_lookahead_and_equity_accounting,
        test_rotation_strategy_runs,
        test_overlays_reduce_exposure,
        test_metrics_known_values,
        test_report_files_written,
        test_delisting_settlement_modes,
        test_paper_trader_flow,
    ):
        fn()


if __name__ == "__main__":
    run()
