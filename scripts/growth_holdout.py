"""小容量成长股策略 —— holdout 一次性样本外验收（2025-01-02 ~ 2026-10-16）。

红线（写死在代码里，不靠记忆）：

* 这是**一次性**验收。第一次运行成功后写入 ``results/growth/holdout/consumed.json``，
  之后必须显式 ``--force`` 才能再跑；任何重跑都要在报告里留痕。
* 口径 = 冻结的 cap25：QN 十因子 / 四块 40-25-15-20 / 50 只等权 / gross 0.95 /
  单票 4% / 证监会门类 25% / 缓冲带 60 / 月末信号次月首个交易日开盘成交 /
  逐日向目标回归且忽略 <5000 元单 / 分档滑点 + 佣金万 2.5（最低 5 元）+
  印花税卖出 0.05% + 过户费 0.001% / 退市 haircut30 / E 层默认开。
* 与样本内 processed 层拼接：≤2024-12-31 用 processed（只做预热），
  2025-01-02 起用 holdout 层；不重复、不穿越。
* 已知缺口：另类数据（解禁/预告/快报/质押）止于 2024-12-31 → holdout 期 n8/n9 为空，
  aux 块按 a1/a2 归一（在报告里明写）。

用法::

    python -X utf8 -m scripts.growth_holdout --run          # 首次
    python -X utf8 -m scripts.growth_holdout --run --force  # 确认要重跑
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import valuation  # noqa: E402
from quant.backtest import Engine  # noqa: E402
from quant.config import BacktestConfig, CostModel  # noqa: E402
from quant.factors import growth, growth_alt, style  # noqa: E402,F401
from quant.factors.panel import build_panel  # noqa: E402
from scripts import growth_leftside as gl  # noqa: E402
from scripts import growth_risk as gr  # noqa: E402
from scripts.factor_market import FIELDS, load_delisting_events, load_layer  # noqa: E402
from scripts.growth_backtest import SPECS, composite_score, month_ends, sector_panel  # noqa: E402
from scripts.redlow_study import LAKE, board_exclude, build_block, load_bars, pool_benchmark  # noqa: E402
from scripts.risk_system import add_speed, extreme_plan, extreme_trigger  # noqa: E402

HOLD_DIR = LAKE / "holdout"
START = pd.Timestamp("2025-01-02")
END = pd.Timestamp("2026-10-16")
SCORE_START = pd.Timestamp("2024-12-31")  # 边界月末：决定 2025 首个交易日建仓
CUTOFF = END                      # 财务：NOTICE_DATE <= 当日的最新一期
ALT_CUTOFF = pd.Timestamp("2024-12-31")   # 另类数据冻结日
OUT = Path("results/growth/holdout")
CASH = 1_000_000.0
TOP_K = 50
GROSS = 0.95
MAX_WEIGHT = 0.04
SECTOR_CAP = 0.25                 # cap25 冻结值
BUFFER_RANK = 60
MIN_TRADE = 5_000.0
BASE_BPS = 2.0
DROPPED_ALT = ("n8_unlock_60d", "n9_unlock_120d")


# --------------------------------------------------------------------------- 数据
def load_holdout_bars() -> dict[str, pd.DataFrame]:
    """读 holdout 日线（只读这层，列已与 processed 对齐）。"""
    frames, stats = load_layer(LAKE, "holdout")
    print(f"[holdout] {stats.get('files', len(frames))} 只 / {stats.get('rows', '?')} 行", flush=True)
    return frames


def combine_bars(sample: dict[str, pd.DataFrame], hold: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """样本内 + holdout 拼接；同一只股票按日期去重，holdout 覆盖同日期（不会重叠但防御）。"""
    merged: dict[str, pd.DataFrame] = {}
    for symbol, frame in sample.items():
        merged[symbol] = frame
    for symbol, frame in hold.items():
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
        old = merged.get(symbol)
        if old is None:
            merged[symbol] = frame.sort_index()
            continue
        joined = pd.concat([old, frame])
        joined = joined[~joined.index.duplicated(keep="last")].sort_index()
        merged[symbol] = joined
    return merged


def holdout_st_matrix(symbols: list[str]) -> pd.DataFrame:
    """holdout 期 ST 标记（估值层 isST），与样本内 st_matrix 拼成一张完整掩码。"""
    inside = valuation.st_matrix(symbols)
    outside = valuation.st_matrix(symbols, root=valuation.HOLDOUT_DIR)
    merged = pd.concat([inside, outside])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def build(*, verbose: bool = True) -> dict:
    t0 = time.time()
    sample_bars, _ = load_bars()
    hold_bars = load_holdout_bars()
    bars = combine_bars(sample_bars, hold_bars)
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    panel = build_panel(bars, fields=FIELDS)
    if verbose:
        print(f"[panel] {panel.close.shape} {panel.dates[0].date()}~{panel.dates[-1].date()}", flush=True)

    block = build_block(panel).reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)
    st_hold = holdout_st_matrix(list(panel.symbols))
    block |= st_hold.reindex(index=panel.dates, columns=panel.symbols).fillna(False).astype(bool)

    facts = growth.GrowthFactors(panel, cutoff=CUTOFF)
    eligible = growth.eligible_mask(facts, block_mask=block)
    # 边界：2024-12-31 的月末信号决定 2025 年首个交易日的建仓，必须包含它，
    # 否则组合在 holdout 头一个月全程空仓（纯现金），不是策略规则的本意。
    marks = month_ends(panel.dates, SCORE_START, END)

    alt = growth_alt.alt_factors(panel, cutoff=ALT_CUTOFF)
    for name in DROPPED_ALT:                      # 另类数据止于 2024-12-31 → holdout 期按剩余项归一
        frame = alt.get(name)
        if frame is not None:
            frame.loc[frame.index >= START, :] = np.nan
            alt[name] = frame
    sector = sector_panel(marks, list(panel.symbols))
    score = composite_score(panel, facts, alt, SPECS["QN"]["blocks"], eligible, marks)
    if verbose:
        print(f"[factors] marks {len(marks)} eligible {int(eligible.to_numpy().sum(axis=1).mean())}/日 "
              f"{time.time() - t0:.0f}s", flush=True)
    return {"bars": bars, "panel": panel, "block": block, "facts": facts, "eligible": eligible,
            "marks": marks, "alt": alt, "sector": sector, "score": score, "elapsed": time.time() - t0}


def build_exposure(panel) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """E 层（冻结规则）：市场帧 → 触发 → 清仓/递减回补路径。"""
    frame = add_speed(gl.market_frame(panel))
    trigger = extreme_trigger(frame)
    frame["extreme_trigger"] = trigger
    path, events = extreme_plan(frame, trigger)
    return path, frame, events


# --------------------------------------------------------------------------- 基准
def index_benchmark(symbol: str, name: str) -> pd.Series | None:
    """指数收盘（外部行情，仅用于对照；取不到就返回 None 并在报告里标不可得）。

    依次尝试东财（``index_zh_a_hist``）与新浪（``stock_zh_index_daily``）；
    东财当晚实测对本机直接 ``RemoteDisconnected``，新浪通常可兜底。
    """
    cache = OUT / f"index-{symbol}.csv"
    if cache.is_file():
        frame = pd.read_csv(cache, parse_dates=["date"])
        return frame.set_index("date")["close"].sort_index()
    try:
        import akshare as ak
    except Exception as exc:                      # pragma: no cover
        print(f"[index] akshare 不可用：{exc}", flush=True)
        return None

    def _em():
        frame = ak.index_zh_a_hist(symbol=symbol, period="daily",
                                   start_date=SCORE_START.strftime("%Y%m%d"),
                                   end_date=END.strftime("%Y%m%d"))
        frame = frame.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]]
        return frame

    def _sina():
        prefix = "sh" if symbol.startswith(("000", "999")) else "sz"
        frame = ak.stock_zh_index_daily(symbol=f"{prefix}{symbol}")
        frame = frame.rename(columns={"date": "date", "close": "close"})[["date", "close"]]
        return frame

    for label, fetch in (("东财", _em), ("新浪", _sina)):
        try:
            frame = fetch()
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.set_index("date").sort_index()
            frame = frame.loc[:END]
            if frame.empty:
                raise RuntimeError("空表")
            OUT.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache, encoding="utf-8-sig")
            print(f"[index] {name}({symbol}) 取到 {len(frame)} 行（{label}）", flush=True)
            return frame["close"]
        except Exception as exc:
            print(f"[index] {name}({symbol}) {label}失败：{type(exc).__name__}: {exc}", flush=True)
    return None


def yearly_returns(equity: pd.Series) -> pd.Series:
    frame = equity.copy()
    frame.index = pd.to_datetime(frame.index)
    return frame.groupby(frame.index.year).apply(lambda s: float(s.iloc[-1] / s.iloc[0] - 1))


def monthly_returns(equity: pd.Series) -> pd.Series:
    frame = equity.copy()
    frame.index = pd.to_datetime(frame.index)
    return frame.groupby([frame.index.year, frame.index.month]).apply(
        lambda s: float(s.iloc[-1] / s.iloc[0] - 1))


# --------------------------------------------------------------------------- 主流程
def run(*, force: bool = False) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / "consumed.json"
    if marker.is_file() and not force:
        print(f"[拒绝] {marker} 已存在——这是一次性验收，确认要重跑请加 --force（会在报告里留痕）",
              flush=True)
        return 2

    started = time.time()
    prepared = build()
    panel, marks, score, eligible, sector = (prepared["panel"], prepared["marks"], prepared["score"],
                                             prepared["eligible"], prepared["sector"])
    exposure, frame, events = build_exposure(panel)
    frame.to_csv(OUT / "market-frame.csv", encoding="utf-8-sig")
    if len(events):
        events.to_csv(OUT / "events-E.csv", index=False, encoding="utf-8-sig")

    adv20, illiq20 = gr.liquidity_panels(panel)
    slip_maps, tier_rows = gr.tier_map(marks, eligible, adv20, illiq20)
    tier_rows.to_csv(OUT / "liquidity-tiers.csv", index=False, encoding="utf-8-sig")

    base_weights = gr.BufferTopK(score, eligible, sector=sector, sector_cap=SECTOR_CAP,
                                 buffer_rank=BUFFER_RANK).generate(prepared["bars"])
    weights = gr.exposure_to_weights(base_weights, exposure)
    weights.loc[:, :] = weights.fillna(0.0)

    delisting = load_delisting_events(LAKE)          # 只有 ≤2024 的事件；2025+ 退市未知，报告中声明
    bench_equity = pool_benchmark(panel, prepared["block"], delisting, cash=CASH)
    bench_equity.to_csv(OUT / "benchmark-pool.csv", encoding="utf-8-sig")

    bt_bars = {s: data.loc[SCORE_START:END] for s, data in prepared["bars"].items() if len(data)}
    tier_slip: dict[str, float] = {}
    for day in marks:
        tier_slip.update(slip_maps.get(day, {}))

    gr.install_broker(min_trade=MIN_TRADE)
    gr.RiskBroker.ALLOW = None
    gr.RiskBroker.MIN_TRADE = MIN_TRADE
    gr.RiskBroker.SLIP = tier_slip
    gr.RiskBroker.BASE_BPS = BASE_BPS
    config = dict(start=str(START.date()), end=str(END.date()), initial_cash=CASH,
                  benchmark=None, costs=CostModel(slippage_bps=BASE_BPS))
    window_exposure = exposure.loc[START:END]
    print(f"[engine] {len(bt_bars)} 只 / 窗口 {START.date()}~{END.date()}；"
          f"E 层窗口内平均仓位 {window_exposure.mean():.3f}、清仓 {int((window_exposure < 0.05).sum())} 天",
          flush=True)
    result = Engine(config=BacktestConfig(**config), benchmark_prices=None,
                    delisting=delisting, delisting_mode="haircut30").run(
        bt_bars, weights, strategy_name="holdout-cap25")
    # 2024-12-31 只作为「权重生效日」的锚（bt_bars 从它开始，引擎才能 shift(1) 到
    # 2025-01-02 建仓）；净值起点保留 2024-12-31 的 100 万现金，否则首日建仓的
    # 损益会被漏掉（首日收盘 981,647 会被误当成起点，虚增累计收益）。
    result.equity = result.equity.loc[SCORE_START:]
    result.daily = result.daily.loc[result.daily.index >= SCORE_START] if "date" not in result.daily.columns \
        else result.daily.loc[result.daily["date"] >= SCORE_START]

    row = {"spec": "cap25", "label": "冻结口径 QN + E 层",
           **gr.evaluate(result, bench_equity, cash=CASH, slip_label="分档"),
           "ann_turnover_one_way": gr.turnover_stats(weights, marks)["ann_turnover_one_way"],
           "mdd_date": str(gr.drawdown_series(result.equity).idxmin().date())}
    metrics = pd.DataFrame([row])
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    result.equity.rename("equity").to_csv(OUT / "equity.csv", index_label="date", encoding="utf-8-sig")
    result.daily.to_csv(OUT / "daily.csv", index_label="date", encoding="utf-8-sig")
    if len(result.trades):
        result.trades.to_csv(OUT / "trades.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(OUT / "weights.csv.gz", compression="gzip", encoding="utf-8-sig")

    equity = result.equity.copy()
    equity.index = pd.to_datetime(equity.index)
    yearly = yearly_returns(equity)
    monthly = monthly_returns(equity)
    bench_window = bench_equity.loc[START:END].ffill()
    bench_yearly = yearly_returns(bench_window)
    yearly_frame = pd.DataFrame({"strategy": yearly, "pool_benchmark": bench_yearly})
    yearly_frame.to_csv(OUT / "yearly.csv", encoding="utf-8-sig")
    monthly.to_csv(OUT / "monthly.csv", encoding="utf-8-sig")

    indexes: dict[str, pd.Series] = {}
    for code, name in (("000852", "中证1000"), ("399006", "创业板指")):
        series = index_benchmark(code, name)
        if series is not None:
            series = series.loc[START:END]
            indexes[name] = series
            series.to_csv(OUT / f"index-{code}.csv", encoding="utf-8-sig")

    # 指数对照：累计 / 年化
    index_rows = []
    for name, series in indexes.items():
        total = float(series.iloc[-1] / series.iloc[0] - 1)
        years = max(len(series) / 244.0, 1e-9)
        index_rows.append({"index": name, "total_return": total,
                           "cagr": float((1 + total) ** (1 / years) - 1)})
    pd.DataFrame(index_rows).to_csv(OUT / "index-metrics.csv", index=False, encoding="utf-8-sig")

    payload = {"generated": pd.Timestamp.now("UTC").isoformat(), "window": [str(START.date()), str(END.date())],
               "metrics": row, "yearly": {str(k): float(v) for k, v in yearly.items()},
               "bench_yearly": {str(k): float(v) for k, v in bench_yearly.items()},
               "indexes": index_rows, "trigger_days": int(frame["extreme_trigger"].loc[START:END].sum()),
               "elapsed": round(time.time() - started, 1),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.write_text(json.dumps({"first_run": payload["generated"], "force": force,
                                  "script_sha256": payload["script_sha256"]},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: row[k] for k in ("total_return", "cagr", "sharpe", "max_drawdown",
                                          "calmar", "ann_turnover_one_way", "fees_pct_of_equity")},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {time.time() - started:.0f}s → {OUT}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="小盘成长策略 holdout 一次性验收")
    parser.add_argument("--run", action="store_true", help="执行验收（默认只打印说明）")
    parser.add_argument("--force", action="store_true", help="确认重跑已消费的 holdout")
    args = parser.parse_args(argv)
    if not args.run:
        print(__doc__)
        return 0
    return run(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
