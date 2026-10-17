"""全市场因子研究 + 多因子回测（只用 processed 层，holdout 层保持加锁）。

    python -m quant factor-market --limit 300                # 冒烟
    python -m quant factor-market --factors all              # 全市场全因子

口径（2026-10-17 拍板）：
* 复权：前复权 qfq（processed 层已落）；
* 清洗：逐日横截面 MAD 去极值，k=5；停牌行**剔除**出因子计算（``panel.tradable``），
  回测里保留行、但由 tradable 与 broker 规则约束成交；
* ST：本期无免费数据，**不剔除**，报告显式声明；
* 样本：processed 层是「当期存活样本」，含幸存者偏差，收益是乐观上界；
* 日期：processed 层截止 2024-12-31，2025 起为 holdout，本脚本不读。

本脚本只回答两件事：全市场读取性能到底多少、因子在全市场上的 IC/分层是否还成立；
顺带跑一次月度调仓的等权/IC 加权多因子回测作为「策略层」的粗糙基线。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.backtest.engine import Engine  # noqa: E402
from quant.config import BacktestConfig  # noqa: E402
from quant.factors import (  # noqa: E402
    build_panel,
    compute,
    evaluate_factor,
    factor_corr,
    names as factor_names,
    quantile_returns,
    quantile_summary,
)
from quant.report import to_markdown_table  # noqa: E402
from quant.strategy.multifactor import MultiFactorStrategy  # noqa: E402

LAKE = Path("D:/quant-data")
USECOLS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount", "prev_close"]
FIELDS = ("open", "high", "low", "close", "volume", "amount", "prev_close")
SPREAD = "多空(Q最高-Q最低)"


# --------------------------------------------------------------------------- 读取
def load_layer(root: Path, kind: str, limit: int | None = None, format: str = "auto") -> tuple[dict, dict]:
    """读一层的全部标的，返回 ``(frames, stats)``。

    ``format=parquet`` 走列式副本：一次扫描整层再按代码切组，比逐只 read_csv 快一个量级；
    ``auto`` 在有 Parquet 层时用它，否则回退 CSV。
    """
    parquet_dir = Path(root) / kind / "parquet"
    use_parquet = format == "parquet" or (format == "auto" and parquet_dir.exists())
    if use_parquet and parquet_dir.exists() and limit is None:
        return _load_layer_parquet(parquet_dir, kind)
    if format == "parquet":
        raise SystemExit(f"{parquet_dir} 不存在，先跑 `python -m quant lake parquet --kinds {kind}`")
    return _load_layer_csv(root, kind, limit)


def load_delisted_bars(root: Path = LAKE) -> dict[str, pd.DataFrame]:
    """读退市股日线（baostock 后复权），把列名对齐到 processed 层的 FIELDS。"""
    folder = Path(root) / "delisted" / "daily"
    bars: dict[str, pd.DataFrame] = {}
    if not folder.is_dir():
        return bars
    for path in sorted(folder.glob("??????.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame = frame.rename(columns={"preclose": "prev_close"})
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        keep = [column for column in FIELDS if column in frame.columns]
        if len(keep) < len(FIELDS):
            continue
        bars[path.stem] = frame[keep].astype(float)
    return bars


def load_delisting_events(root: Path = LAKE) -> dict[str, pd.Timestamp]:
    """读退市事件表 → {symbol: 最后交易日}（裸代码与带后缀两种写法都放）。"""
    path = Path(root) / "delisted" / "events.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype={"symbol": str})
    if "bars" in frame.columns:
        frame = frame[frame["bars"] > 0]
    mapping: dict[str, pd.Timestamp] = {}
    for row in frame.itertuples():
        code = str(row.symbol).zfill(6)
        stamp = pd.Timestamp(row.last_bar)
        mapping[code] = stamp
        for suffix in (".SH", ".SZ", ".BJ"):
            mapping[code + suffix] = stamp
    return mapping


def _load_layer_parquet(parquet_dir: Path, kind: str) -> tuple[dict, dict]:
    import pyarrow.dataset as ds

    t0 = time.perf_counter()
    table = ds.dataset(str(parquet_dir), format="parquet", partitioning="hive").to_table(
        columns=["symbol", "date", *FIELDS]
    )
    long = table.to_pandas()
    read_seconds = time.perf_counter() - t0
    long["symbol"] = long["symbol"].astype(str)
    long["date"] = pd.to_datetime(long["date"])

    t1 = time.perf_counter()
    frames: dict[str, pd.DataFrame] = {}
    for symbol, part in long.groupby("symbol", sort=False):
        frame = part.drop(columns=["symbol"]).set_index("date").sort_index()
        frame.index = pd.DatetimeIndex(frame.index, name="date")
        frames[symbol] = frame[list(FIELDS)].astype(float)
    parse_seconds = time.perf_counter() - t1
    rows = int(len(long))
    nbytes = sum(p.stat().st_size for p in parquet_dir.rglob("*.parquet"))
    total = read_seconds + parse_seconds
    stats = {
        "kind": kind,
        "format": "parquet",
        "files": len(frames),
        "loaded": len(frames),
        "skipped": [],
        "rows": rows,
        "bytes": int(nbytes),
        "seconds_total": round(total, 2),
        "read_ms": {
            "sum": round(read_seconds, 2),
            "mean": round(read_seconds / max(len(frames), 1) * 1000, 3),
            "p50": None,
            "p95": None,
            "max": None,
        },
        "post_ms": {"mean": round(parse_seconds / max(len(frames), 1) * 1000, 2), "sum": round(parse_seconds, 2)},
        "throughput": {
            "files_per_s": round(len(frames) / max(total, 1e-9), 1),
            "rows_per_s": round(rows / max(total, 1e-9), 0),
            "mb_per_s": round(nbytes / 1024 / 1024 / max(total, 1e-9), 1),
        },
    }
    print(
        f"  列式层：{len(frames)} 只 / {rows:,} 行，一次扫描 {read_seconds:.2f}s + 切组 {parse_seconds:.2f}s"
        f"（{nbytes / 1024 / 1024:.0f} MB）",
        flush=True,
    )
    return frames, stats


def _load_layer_csv(root: Path, kind: str, limit: int | None = None) -> tuple[dict, dict]:
    """逐只读 CSV，返回 ``(frames, stats)``；同时记录纯读取与后处理耗时。"""
    daily = Path(root) / kind / "daily"
    paths = sorted(daily.glob("*.csv"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"{daily} 下没有 CSV")

    frames: dict[str, pd.DataFrame] = {}
    read_times: list[float] = []
    parse_times: list[float] = []
    rows = 0
    nbytes = 0
    skipped: list[tuple[str, str]] = []
    t_all = time.perf_counter()
    for index, path in enumerate(paths, 1):
        nbytes += path.stat().st_size
        t0 = time.perf_counter()
        try:
            frame = pd.read_csv(path, usecols=USECOLS, parse_dates=["date"])
        except Exception as exc:  # 坏文件不阻塞整体
            skipped.append((path.name, str(exc)))
            continue
        read_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        frame = frame.dropna(subset=["date"]).drop_duplicates("date")
        symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame.columns and len(frame) else path.stem
        frame = frame.drop(columns=[c for c in ("symbol",) if c in frame.columns])
        frame = frame.set_index("date").sort_index()
        frame.index = pd.DatetimeIndex(frame.index, name="date")
        frames[symbol] = frame.astype(float)
        parse_times.append(time.perf_counter() - t1)
        rows += len(frame)

        if index % 1000 == 0:
            print(f"  已读 {index}/{len(paths)}（{time.perf_counter() - t_all:.1f}s）", flush=True)
    total = time.perf_counter() - t_all

    rt = np.asarray(read_times) * 1000.0
    pt = np.asarray(parse_times) * 1000.0
    stats = {
        "kind": kind,
        "files": len(paths),
        "loaded": len(frames),
        "skipped": skipped,
        "rows": int(rows),
        "bytes": int(nbytes),
        "seconds_total": round(total, 2),
        "read_ms": {
            "mean": round(float(rt.mean()), 2),
            "p50": round(float(np.percentile(rt, 50)), 2),
            "p95": round(float(np.percentile(rt, 95)), 2),
            "max": round(float(rt.max()), 2),
            "sum": round(float(rt.sum()) / 1000.0, 2),
        },
        "post_ms": {"mean": round(float(pt.mean()), 2), "sum": round(float(pt.sum()) / 1000.0, 2)},
        "throughput": {
            "files_per_s": round(len(paths) / total, 1),
            "rows_per_s": round(rows / total, 0),
            "mb_per_s": round(nbytes / 1024 / 1024 / total, 1),
        },
    }
    return frames, stats


# --------------------------------------------------------------------------- 因子
def summarize_factor(name: str, result: dict, horizon: int) -> dict:
    ic = result["ic"]
    decay = result["decay"]
    qsum = result["quantile_summary"]
    spread_mean = float(qsum.loc[SPREAD, "mean"]) if SPREAD in qsum.index else float("nan")
    spread_win = float(qsum.loc[SPREAD, "positive_rate"]) if SPREAD in qsum.index else float("nan")
    row = {
        "factor": name,
        "coverage": round(float(result["coverage"]), 1),
        "ic_mean": round(float(ic["ic_mean"]), 4),
        "ic_std": round(float(ic["ic_std"]), 4),
        "ic_ir": round(float(ic["ic_ir"]), 3),
        "t_stat": round(float(ic["t_stat"]), 2),
        "positive_rate": round(float(ic["positive_rate"]), 3),
    }
    for days in (1, horizon):
        row[f"ic_{days}d"] = round(float(decay.get(days, float("nan"))), 4)
    row["spread"] = spread_mean
    row["spread_win"] = spread_win
    return row


def run_factor_study(panel, wanted: list[str], *, horizon: int, quantiles: int, min_count: int,
                     block: pd.DataFrame | None = None):
    rows: list[dict] = []
    frames: dict[str, pd.DataFrame] = {}
    timings: dict[str, float] = {}
    for name in wanted:
        t0 = time.perf_counter()
        frame = compute(name, panel)
        frame = frame.where(panel.tradable)  # 停牌/无量：不参与横截面
        if block is not None and not block.empty:
            frame = frame.where(~block.reindex(index=frame.index, columns=frame.columns).fillna(False))
        timings[f"compute.{name}"] = round(time.perf_counter() - t0, 2)
        t1 = time.perf_counter()
        result = evaluate_factor(
            frame, panel, name=name, horizons=(1, horizon), quantiles=quantiles, min_count=min_count
        )
        timings[f"evaluate.{name}"] = round(time.perf_counter() - t1, 2)
        rows.append(summarize_factor(name, result, horizon))
        frames[name] = frame
        print(
            f"  {name:<22} 覆盖 {rows[-1]['coverage']:>7.1f}  IC {rows[-1]['ic_mean']:+.4f}  "
            f"IR {rows[-1]['ic_ir']:+.3f}  多空 {rows[-1]['spread']:+.4%}",
            flush=True,
        )
    return rows, frames, timings


# --------------------------------------------------------------------------- 分层 / 去冗余 / 网格
def layer_study(panel, frames: dict, rows: list[dict], *, horizon: int = 5, quantiles: int = 5,
                min_count: int = 50, top_n: int = 6) -> dict[str, pd.DataFrame]:
    """Q1–Q5 分层：按因子横截面分位建组，看未来 ``horizon`` 日的等权收益（Alphalens 口径）。

    成熟项目对照：Alphalens 的 ``mean_return_by_quantile`` 用前瞻收益按分位分组取均值，
    本函数同样口径；差别是我们不做行业中性（缺行业 PIT 数据），只用 ``tradable`` 过滤停牌。
    """
    forward = panel.forward_returns(int(horizon))
    ranked = sorted(rows, key=lambda r: abs(float(r["ic_ir"])), reverse=True)[:top_n]
    tables: dict[str, pd.DataFrame] = {}
    for row in ranked:
        name = row["factor"]
        if name not in frames:
            continue
        qret = quantile_returns(frames[name], forward, quantiles=quantiles, min_count=min_count)
        summary = quantile_summary(qret, periods_per_year=252.0 / max(int(horizon), 1))
        summary["ic_ir"] = float(row["ic_ir"])
        summary["ic_mean"] = float(row["ic_mean"])
        tables[name] = summary
    return tables


def _avg_rank_corr(a: pd.DataFrame, b: pd.DataFrame, *, sample: int = 120, min_names: int = 100) -> float:
    """两个因子逐日横截面秩相关的均值（采样日期，避免全量代价）。"""
    dates = a.index.intersection(b.index)[:: max(len(a.index) // sample, 1)]
    values: list[float] = []
    for day in dates:
        x, y = a.loc[day], b.loc[day]
        mask = x.notna() & y.notna()
        if int(mask.sum()) < min_names:
            continue
        corr = x[mask].rank().corr(y[mask].rank())
        if corr == corr:
            values.append(float(corr))
    return float(np.mean(values)) if values else 0.0


def dedup_factors(frames: dict, rows: list[dict], *, threshold: float = 0.7) -> tuple[list[str], pd.DataFrame]:
    """按 |IC_IR| 从高到低贪心保留因子，剔除与已保留因子平均秩相关超阈值的冗余项。"""
    order = sorted(rows, key=lambda r: abs(float(r["ic_ir"])), reverse=True)
    kept: list[str] = []
    kept_rows: dict[str, dict] = {}
    records: list[dict] = []
    for row in order:
        name = row["factor"]
        if name not in frames:
            continue
        if not kept:
            kept.append(name)
            kept_rows[name] = row
            records.append({"factor": name, "action": "keep", "max_abs_corr": 0.0, "vs": "", "ic_ir": row["ic_ir"]})
            continue
        pairs = [(k, _avg_rank_corr(frames[name], frames[k])) for k in kept]
        worst_name, worst = max(pairs, key=lambda item: abs(item[1]))
        if abs(worst) >= threshold:
            records.append({"factor": name, "action": "drop", "max_abs_corr": round(abs(worst), 3),
                            "vs": worst_name, "ic_ir": row["ic_ir"]})
        else:
            kept.append(name)
            kept_rows[name] = row
            records.append({"factor": name, "action": "keep", "max_abs_corr": round(abs(worst), 3),
                            "vs": worst_name, "ic_ir": row["ic_ir"]})
    return kept, pd.DataFrame(records)


def run_grid(bars: dict, wanted: list[str], args, out_dir: Path) -> pd.DataFrame:
    """调仓频率 × 持仓数的网格：每个组合跑一次回测，输出收益/夏普/回撤/换手/费用。"""
    freqs = [s.strip() for s in str(args.grid).split(",") if s.strip()]
    topks = [int(s) for s in str(args.grid_topk).split(",") if s.strip()] or [args.top_k]
    rows: list[dict] = []
    for freq in freqs:
        for top_k in topks:
            print(f"== 网格：rebalance={freq} top_k={top_k} ==", flush=True)
            local = argparse.Namespace(**vars(args))
            local.rebalance = freq
            local.top_k = top_k
            out = run_backtest(bars, wanted, local, out_dir / f"grid-{freq}-{top_k}")
            metrics = dict(out["metrics"])
            row = {"rebalance": freq, "top_k": top_k, "seconds": out["seconds"],
                   "rebalances": out["rebalances"], "avg_holdings": out["avg_holdings"],
                   "total_return": metrics.get("累计收益"), "cagr": metrics.get("年化收益（CAGR）"),
                   "sharpe": metrics.get("夏普"), "max_drawdown": metrics.get("最大回撤"),
                   "turnover": metrics.get("日均换手"), "fees": metrics.get("费用合计"),
                   "trades": metrics.get("成交笔数"), "alpha": metrics.get("Beta / Alpha"),
                   "benchmark": metrics.get("基准累计收益")}
            rows.append(row)
            print(f"   收益 {row['total_return']} 夏普 {row['sharpe']} 回撤 {row['max_drawdown']}"
                  f" 换手 {row['turnover']} 费用 {row['fees']}", flush=True)
            gc.collect()
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "grid.csv", index=False, encoding="utf-8-sig")
    return table


# --------------------------------------------------------------------------- 回测
def metrics_rows(result) -> list[tuple[str, str]]:
    m = result.metrics
    report = result.report
    def pct(key, digits=2):
        value = m.get(key)
        return "" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{float(value) * 100:.{digits}f}%"
    def num(key, fmt="{:.3f}"):
        value = m.get(key)
        return "" if value is None or (isinstance(value, float) and np.isnan(value)) else fmt.format(float(value))
    rows = [
        ("累计收益", pct("total_return")),
        ("年化收益（CAGR）", pct("cagr")),
        ("年化波动", pct("ann_vol")),
        ("夏普", num("sharpe")),
        ("索提诺", num("sortino")),
        ("最大回撤", pct("max_drawdown")),
        ("卡玛", num("calmar")),
        ("胜率（日）", pct("win_rate")),
        ("日均换手", pct("avg_turnover", 3)),
        ("费用合计", f"{float(m.get('total_fees') or 0):,.0f} 元"),
        ("成交笔数", f"{int(m.get('trade_count') or 0)}"),
        ("Beta / Alpha", f"{num('beta')} / {pct('alpha')}"),
        ("信息比率", num("info_ratio")),
        ("基准累计收益", pct("benchmark_return")),
    ]
    return rows


def run_backtest(bars: dict, wanted: list[str], args, out_dir: Path) -> dict:
    strategy = MultiFactorStrategy(
        factors=",".join(wanted),
        weighting=args.weighting,
        ic_lookback=250,
        ic_min_periods=120,
        weight_refresh=3,
        top_k=args.top_k,
        rebalance=args.rebalance,
        gross_exposure=0.95,
        max_weight=min(0.10, 2.0 / max(args.top_k, 2)),
        winsorize_method="mad",
        winsorize_n=5.0,
        standardize=True,
        min_price=2.0,
        min_amount=args.min_amount,
        min_history=120,
        min_universe=100,
        exclude=getattr(args, "st_mask", None),
    )
    config = BacktestConfig(
        start=args.start,
        end=args.end,
        initial_cash=args.cash,
        benchmark=None,
        min_trade_value=args.min_trade_value,
    )
    # 基准：全池等权买入持有（每只按首个有效收盘价归一后横截面取均值），用于 Beta/Alpha/IR
    close = pd.DataFrame({symbol: frame["close"] for symbol, frame in bars.items()})
    close = close.loc[(close.index >= pd.Timestamp(args.start)) & (close.index <= pd.Timestamp(args.end))]
    base = close.apply(lambda series: series.dropna().iloc[0] if series.notna().any() else np.nan)
    benchmark = close.div(base.where(base > 0), axis=1).where(close > 0).mean(axis=1, skipna=True)
    engine = Engine(
        config=config,
        benchmark_prices=benchmark,
        delisting=load_delisting_events() if getattr(args, "include_delisted", False) else None,
        delisting_mode=getattr(args, "delisting_mode", "haircut30"),
    )
    t0 = time.perf_counter()
    result = engine.run_strategy(bars, strategy)
    elapsed = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(out_dir / "equity.csv", encoding="utf-8-sig")
    if len(result.trades):
        result.trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")

    picks = strategy.last_diagnostics.get("picks", {})
    weight_log = strategy.last_diagnostics.get("factor_weights_by_date", {})
    holdings = {day: len(names) for day, names in picks.items()}
    return {
        "seconds": round(elapsed, 2),
        "metrics": metrics_rows(result),
        "strategy": strategy.describe(),
        "rebalances": len(picks),
        "avg_holdings": round(float(np.mean(list(holdings.values()))) if holdings else 0.0, 1),
        "last_holdings": list(picks.values())[-1] if picks else [],
        "weight_log_samples": dict(list(weight_log.items())[:2]),
        "report": result.report,
        "result": result,
    }


# --------------------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m quant factor-market", description="全市场因子研究 + 回测")
    parser.add_argument("--lake", default=str(LAKE))
    parser.add_argument("--kind", default="processed", choices=["processed", "raw", "holdout"])
    parser.add_argument("--format", default="auto", choices=["auto", "csv", "parquet"],
                        help="读取格式；auto 在有 Parquet 层时用列式副本")
    parser.add_argument("--limit", type=int, default=0, help="只读前 N 只（冒烟用）")
    parser.add_argument("--factors", default="all", help="all 或逗号分隔")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--layer-horizon", type=int, default=5,
                        help="Q1–Q5 分层表用的前瞻天数（默认 5 日，对应周频）")
    parser.add_argument("--dedup-threshold", type=float, default=0.7,
                        help="因子去冗余阈值：与已保留因子的平均秩相关绝对值超过它就剔除，默认 0.7")
    parser.add_argument("--grid", default="", help="调仓频率网格，逗号分隔，如 M,2W,W")
    parser.add_argument("--grid-topk", default="", help="持仓数网格，逗号分隔，如 20,30,50")
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--rebalance", default="M")
    parser.add_argument("--weighting", default="ic", choices=["equal", "ic", "ir", "manual"])
    parser.add_argument("--min-amount", type=float, default=2e7)
    parser.add_argument("--cash", type=float, default=10_000_000.0)
    parser.add_argument("--min-trade-value", type=float, default=5_000.0,
                        help="单笔调仓最小金额（元），低于它的零碎调整跳过，默认 5000")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--include-delisted", action="store_true",
                        help="把 256 只退市股补进票池（消除幸存者偏差；data/delisted 需已落盘）")
    parser.add_argument("--delisting-mode", default="haircut30", choices=["last", "haircut30", "total_loss"],
                        help="退市结算假设，仅 --include-delisted 时生效")
    parser.add_argument("--fundamental", action="store_true",
                        help="注册基本面/估值因子（需要 valuation + fundamentals 落盘）")
    parser.add_argument("--exclude-st", action="store_true",
                        help="按估值层的逐日 isST 剔除 ST 股（选股与 IC 都剔除）")
    parser.add_argument("--out", default=str(ROOT / "results" / "factor-market"))
    args = parser.parse_args(argv)

    if args.kind == "holdout":
        raise SystemExit("holdout 层是最终校核集，本脚本不读；如需解锁请显式改代码")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== 读取 {args.kind} 层：{args.lake} ==", flush=True)
    frames, load_stats = load_layer(Path(args.lake), args.kind, args.limit or None, format=args.format)
    if args.fundamental:
        from quant.factors import fundamental

        registered = fundamental.register()
        print(f"== 基本面因子已注册：{len(registered)} 个（{', '.join(registered)}） ==", flush=True)
    if args.include_delisted:
        dead = load_delisted_bars(Path(args.lake))
        added = [symbol for symbol in dead if symbol not in frames]
        frames.update({symbol: dead[symbol] for symbol in added})
        load_stats["delisted_added"] = len(added)
        print(f"== 补进退市股 {len(added)} 只（票池 {len(frames)} 只） ==", flush=True)
    # bars 也要裁到 [start, end]：退市股带来 2014–2015 与 2025–2026 的日期，
    # 而 MultiFactorStrategy 会自己从 bars 建面板 → 不裁的话策略在只有退市股交易的
    # 年份里也会下单（实测 2014-08 就开出第一笔，2014-2015 牛市把收益拉高了 60pp）。
    lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end)
    clipped = {}
    for symbol, frame in frames.items():
        window = frame.loc[(frame.index >= lo) & (frame.index <= hi)]
        if len(window) >= 2:
            clipped[symbol] = window
    dropped = len(frames) - len(clipped)
    if dropped:
        print(f"== bars 裁到 {args.start} ~ {args.end}：{len(clipped)} 只保留，{dropped} 只区间内无数据 ==", flush=True)
    frames = clipped
    print(json.dumps(load_stats, ensure_ascii=False, indent=2), flush=True)

    print("== 构建面板 ==", flush=True)
    t0 = time.perf_counter()
    panel = build_panel(frames, fields=FIELDS)
    panel_seconds = round(time.perf_counter() - t0, 2)
    # 面板裁到 [start, end]：补进退市股后日期并集会扩到 2014–2026，
    # 那些只有退市股在交易的横截面会把 IC 和分层一起污染（曾踩到）。
    window = (panel.dates >= pd.Timestamp(args.start)) & (panel.dates <= pd.Timestamp(args.end))
    if not window.all():
        clipped = panel.dates[window]
        panel = type(panel)({key: value.loc[clipped] for key, value in panel.fields.items()})
        print(f"  面板按 {args.start} ~ {args.end} 裁剪：{int((~window).sum())} 天不在窗口内", flush=True)
    panel_rows = sum(int(frame.memory_usage(deep=True).sum()) for frame in panel.fields.values())
    print(f"  面板 {len(panel.dates)} 天 × {len(panel.symbols)} 只，耗时 {panel_seconds}s，"
          f"内存约 {panel_rows / 1024**2:.0f} MB", flush=True)

    wanted = factor_names() if args.factors == "all" else [s.strip() for s in args.factors.split(",") if s.strip()]
    unknown = [name for name in wanted if name not in factor_names()]
    if unknown:
        raise SystemExit(f"未知因子：{unknown}；可选 {factor_names()}")

    st_mask = None
    if args.exclude_st:
        from quant.valuation import st_matrix

        t0 = time.perf_counter()
        st_mask = st_matrix()
        args.st_mask = st_mask
        if st_mask.empty:
            print("== --exclude-st：估值层没有 isST 数据，本次不剔除 ==", flush=True)
        else:
            st_mask = st_mask.reindex(index=panel.dates, columns=panel.symbols).fillna(False).astype(bool)
            args.st_mask = st_mask
            share = float(st_mask.to_numpy().mean())
            print(
                f"== 剔除 ST：{len(st_mask)} 天 × {st_mask.shape[1]} 只，ST 占比 {share * 100:.2f}%"
                f"（{round(time.perf_counter() - t0, 1)}s） ==",
                flush=True,
            )

    print(f"== 因子研究（{len(wanted)} 个） ==", flush=True)
    rows, factor_frames, timings = run_factor_study(
        panel, wanted, horizon=args.horizon, quantiles=args.quantiles, min_count=args.min_count,
        block=st_mask,
    )

    corr = None
    if 1 < len(factor_frames) <= 10:
        t0 = time.perf_counter()
        corr = factor_corr(factor_frames)
        timings["factor_corr"] = round(time.perf_counter() - t0, 2)

    table = pd.DataFrame(rows).sort_values("ic_ir", ascending=False).reset_index(drop=True)
    table.to_csv(out_dir / "factor-summary.csv", index=False, encoding="utf-8-sig")
    if corr is not None:
        corr.to_csv(out_dir / "factor-corr.csv", encoding="utf-8-sig")

    print("== Q1–Q5 分层（前瞻 %d 日） ==" % args.layer_horizon, flush=True)
    t0 = time.perf_counter()
    layers = layer_study(panel, factor_frames, rows, horizon=args.layer_horizon,
                         quantiles=args.quantiles, min_count=args.min_count)
    timings["layers"] = round(time.perf_counter() - t0, 2)
    if layers:
        merged = pd.concat({name: frame for name, frame in layers.items()}, names=["factor", "bucket"])
        merged.to_csv(out_dir / "quantile.csv", encoding="utf-8-sig")

    print("== 因子去冗余（阈值 %.2f） ==" % args.dedup_threshold, flush=True)
    t0 = time.perf_counter()
    kept, dedup = dedup_factors(factor_frames, rows, threshold=args.dedup_threshold)
    timings["dedup"] = round(time.perf_counter() - t0, 2)
    dedup.to_csv(out_dir / "factor-dedup.csv", index=False, encoding="utf-8-sig")
    print(f"  保留 {len(kept)}/{len(rows)} 个：{', '.join(kept)}", flush=True)

    del factor_frames
    gc.collect()

    backtest: dict | None = None
    grid: pd.DataFrame | None = None
    if not args.skip_backtest:
        print("== 多因子回测 ==", flush=True)
        backtest = run_backtest(frames, wanted, args, out_dir)
        print(f"  完成，{backtest['seconds']}s", flush=True)
    if args.grid:
        print("== 调仓频率/持仓数网格 ==", flush=True)
        grid = run_grid(frames, wanted, args, out_dir)

    timings["read"] = load_stats["seconds_total"]
    timings["panel"] = panel_seconds
    (out_dir / "timings.json").write_text(json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Markdown 报告 ----------------
    lines: list[str] = ["# 全市场因子研究（processed 层）", ""]
    fmt = load_stats.get("format", "csv")
    label = "parquet" if fmt == "parquet" else "daily"
    lines.append(f"- 数据：`{args.lake}\\{args.kind}\\{label}`，{load_stats['loaded']} 只 / {load_stats['rows']:,} 行，"
                 f"{args.start} ~ {args.end}")
    delisted_added = int(load_stats.get("delisted_added") or 0)
    if delisted_added:
        lines.append(
            f"- 样本口径：**已补进 {delisted_added} 只退市股**（票池 {len(frames)} 只），退市持仓按 "
            f"`{args.delisting_mode}` 结算（last=按最后价 / haircut30=折价30% / total_loss=全损）；"
            f"ST 未剔除（无免费数据）；停牌行在因子计算中剔除、回测中保留但不参与成交"
        )
    else:
        lines.append(f"- 样本口径：**当期存活样本（含幸存者偏差，收益是乐观上界）**；ST 未剔除（无免费数据）；"
                     f"停牌行在因子计算中剔除、回测中保留但不参与成交")
    lines.append(f"- 清洗：逐日 MAD 去极值 k=5，标准化；IC 用秩相关，min_count={args.min_count}")
    lines.append("- 基准：全池等权买入持有（首个有效收盘价归一后等权），用于 Beta / Alpha / 信息比率")
    lines.append(f"- 调仓：{args.rebalance}（旧别名经 `normalize_freq` 折算成 pandas 3 可认的频率，如 M→ME）；"
                 f"小额调仓阈值 {args.min_trade_value:,.0f} 元，低于它的漂移不再下碎单")
    lines.append("- 口径修正：早期版本把 pandas 3 不认的 `M` 静默回退成每日调仓（成交数与费用被放大），"
                 "本报告数字均为修正后重跑，与旧报告不可直接比较")
    lines.append("- holdout（2025-01-01 起）未读取；结论仅代表样本内")
    lines.append("")

    lines.append("## 读取性能")
    lines.append("")
    rd = load_stats["read_ms"]
    lines.append(f"- 文件数 {load_stats['files']}，成功 {load_stats['loaded']}，跳过 {len(load_stats['skipped'])}；"
                 f"总耗时 **{load_stats['seconds_total']}s**（{'列式一次扫描' if fmt == 'parquet' else '纯 read_csv'} 合计 {rd['sum']}s）")
    if fmt == "parquet":
        lines.append(f"- 列式读取：整层一次扫描后按代码切组，等价单只平均 {rd['mean']}ms（CSV 路径见 --format csv）")
    else:
        lines.append(f"- 单文件 read_csv：平均 {rd['mean']}ms / P50 {rd['p50']}ms / P95 {rd['p95']}ms / 最大 {rd['max']}ms")
    lines.append(f"- 后处理（建索引/排序/转 float）合计 {load_stats['post_ms']['sum']}s，平均 {load_stats['post_ms']['mean']}ms")
    lines.append(f"- 吞吐：{load_stats['throughput']['files_per_s']} 文件/s，{load_stats['throughput']['rows_per_s']:,} 行/s，"
                 f"{load_stats['throughput']['mb_per_s']} MB/s（{load_stats['bytes'] / 1024**2:.0f} MB）")
    lines.append(f"- 面板：{len(panel.dates)} 天 × {len(panel.symbols)} 只，构建 {panel_seconds}s，约 {panel_rows / 1024**2:.0f} MB")
    lines.append("")

    lines.append(f"## 因子摘要（IC 观察 1 日；衰减看 {args.horizon} 日）")
    lines.append("")
    lines.append(to_markdown_table(table, floatfmt=".4f", index=False))
    lines.append("")
    if args.horizon in table.columns and "ic_1d" in table.columns:
        lines.append(f"> IC 均值 > 0.02 且 IR > 0.3 才算有信息；全市场 N≈{len(panel.symbols)}，"
                     f"IC 标准误约 ±{1 / np.sqrt(max(len(panel.dates), 1)):.3f}，小 IC 要结合 t 值看。")
        lines.append("")

    if corr is not None:
        lines.append("## 因子相关性（秩相关）")
        lines.append("")
        lines.append(to_markdown_table(corr, floatfmt="+.3f"))
        lines.append("")

    if layers:
        lines.append(f"## Q1–Q5 分层收益（前瞻 {args.layer_horizon} 日）")
        lines.append("")
        lines.append("> 口径同 Alphalens 的 `mean_return_by_quantile`：逐日按因子横截面分位建 5 组，"
                     "取各组未来 N 日等权收益；这里不做行业中性（缺行业 PIT 数据），停牌已从横截面剔除。")
        lines.append("")
        for name, frame in layers.items():
            show = frame.copy()
            show["days"] = show["days"].astype(int)
            for col in ("mean", "annual"):
                show[col] = (show[col] * 100).round(2).astype(str) + "%"
            show["positive_rate"] = (show["positive_rate"] * 100).round(1).astype(str) + "%"
            show = show[["mean", "annual", "positive_rate", "days"]]
            lines.append(f"### {name}（IC_IR {frame['ic_ir'].iloc[0]:+.3f}）")
            lines.append("")
            lines.append(to_markdown_table(show))
            lines.append("")

    if dedup is not None and len(dedup):
        lines.append(f"## 因子去冗余（平均秩相关阈值 {args.dedup_threshold:.2f}）")
        lines.append("")
        lines.append("> 从 |IC_IR| 最高的因子开始贪心保留；与已保留因子平均秩相关超阈值的判为冗余。"
                     "这一步只算统计冗余，不代表冗余因子没用（信息可能体现在别的维度）。")
        lines.append("")
        lines.append(to_markdown_table(dedup, floatfmt="+.3f"))
        lines.append("")

    if grid is not None and len(grid):
        lines.append("## 调仓频率 × 持仓数网格")
        lines.append("")
        lines.append(to_markdown_table(grid))
        lines.append("")
        lines.append("> 网格是样本内结果，不能用于选参数上线；它只回答「成本与信号衰减的平衡点大致在哪」。")
        lines.append("")

    if backtest is not None:
        lines.append(f"## 多因子回测（{args.rebalance} 调仓）")
        lines.append("")
        lines.append(f"- 策略：{backtest['strategy']}")
        lines.append(f"- 调仓次数 {backtest['rebalances']}，平均持仓 {backtest['avg_holdings']} 只，"
                     f"最新一期持仓示例：{', '.join(backtest['last_holdings'][:10])}")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("| --- | --- |")
        for key, value in backtest["metrics"]:
            lines.append(f"| {key} | {value} |")
        lines.append("")
        lines.append("> 基准是「全池等权买入持有」，不是沪深300：两者都含幸存者偏差，但策略受限于流动性/持仓数，"
                     "基准代表「同池躺平」的机会成本；回测未计涨跌停排队等滑点外成本。")
        lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```")
    lines.append(f"python -m quant factor-market --limit {args.limit} --factors {args.factors} "
                 f"--top-k {args.top_k} --rebalance {args.rebalance} --weighting {args.weighting} "
                 f"--format {args.format}")
    lines.append("```")
    (out_dir / "factor-market.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告写入 {out_dir}", flush=True)
    if backtest is not None:
        for key, value in backtest["metrics"]:
            print(f"  {key:<14} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
