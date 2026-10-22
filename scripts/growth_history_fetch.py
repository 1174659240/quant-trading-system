"""补 2007–2016 历史日线（含当时退市股）与历史财务代理，落到独立目录、不改动现有数据湖。

设计要点
--------
- 数据源：baostock（免费）。**只写** ``D:/quant-data/growth-history/``，现有 ``processed/``、``holdout/``、
  ``fundamentals/``、``alt/`` 一个字节都不动（本脚本不持有任何写句柄到那些目录）。
- 分区与列名对齐 ``D:/quant-data/processed/parquet/year=*/data.parquet``：
  ``symbol``（6 位裸码）/ ``date``（object, YYYY-MM-DD）/ open, high, low, close, volume, amount,
  prev_close, adj_factor, suspended, tradable。
- 断点续跑：``raw/daily/<symbol>.csv.gz`` 存在且非空即跳过；失败清单落 ``meta/failures.json``，
  ``--retry-failed`` 只重跑失败项。
- 限速与重试：每个进程内串行请求（baostock 单连接），失败按 1/2/4 秒退避重试 3 次。
- 财务：``query_profit_data``（ROE/毛利/净利/营收/总股本）与 ``query_growth_data``（净利同比）逐股逐季，
  全部带 ``pubDate``，PIT 安全；覆盖 2007Q1–2016Q4（2013 起与现有 fundamentals 有重叠，可交叉校验）。

用法
----
    python -m scripts.growth_history_fetch --print-plan
    python -m scripts.growth_history_fetch --only bars --workers 8 --resume
    python -m scripts.growth_history_fetch --only fund --workers 8 --resume
    python -m scripts.growth_history_fetch --only consolidate
    python -m scripts.growth_history_fetch --only audit
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("D:/quant-data/growth-history")
START = "2007-01-01"
END = "2016-12-31"
BAR_START = "2007-01-01"
BAR_END = "2016-12-31"
FUND_QUARTERS = [(year, quarter) for year in range(2007, 2017) for quarter in (1, 2, 3, 4)]
BAR_FIELDS = "date,open,high,low,close,preclose,volume,amount,tradestatus,pctChg,isST"


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    @property
    def raw(self) -> Path:
        return self.root / "raw" / "daily"

    @property
    def parquet(self) -> Path:
        return self.root / "parquet"

    @property
    def fundamental(self) -> Path:
        return self.root / "fundamentals"

    @property
    def meta(self) -> Path:
        return self.root / "meta"

    @property
    def log(self) -> Path:
        return self.root / "fetch-log.jsonl"

    def ensure(self) -> None:
        for folder in (self.raw, self.parquet, self.fundamental, self.meta):
            folder.mkdir(parents=True, exist_ok=True)


PATHS = Paths()
WORKER_STATE: dict[str, Any] = {}


def _login() -> None:
    import baostock as bs

    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {result.error_msg}")
    WORKER_STATE["bs"] = bs
    WORKER_STATE["consecutive_failures"] = 0


def _init_worker() -> None:
    _login()


def _relogin() -> None:
    bs = WORKER_STATE.get("bs")
    try:
        if bs is not None:
            bs.logout()
    except Exception:  # noqa: BLE001
        pass
    time.sleep(1.0)
    _login()


def _with_retry(fn, attempts: int = 3):
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            WORKER_STATE["consecutive_failures"] = int(WORKER_STATE.get("consecutive_failures", 0)) + 1
            time.sleep(1.0 * (2**attempt))
            if WORKER_STATE["consecutive_failures"] >= 8:
                _relogin()
                WORKER_STATE["consecutive_failures"] = 0
    raise RuntimeError(f"重试 {attempts} 次仍失败: {last}")


def _rows(cursor) -> pd.DataFrame:
    """把 baostock 游标转成 DataFrame；**错误必须抛出**，否则断连会被误判成「没有数据」。"""
    if cursor.error_code != "0":
        raise RuntimeError(f"baostock 返回错误 {cursor.error_code}: {cursor.error_msg}")
    data = []
    while cursor.next():
        data.append(cursor.get_row_data())
    if cursor.error_code != "0":
        raise RuntimeError(f"baostock 游标中途出错 {cursor.error_code}: {cursor.error_msg}")
    return pd.DataFrame(data, columns=cursor.fields)


def normalize_code(code: str) -> str:
    """``sh.600000`` → ``600000``（与 processed/parquet 的 symbol 一致）。"""
    return code.split(".")[-1]


def load_universe() -> pd.DataFrame:
    """baostock 全部股票（type=1），裁到 2007–2016 有交易的标的。"""
    cache = PATHS.meta / "universe-history.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import baostock as bs

    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(result.error_msg)
    try:
        frame = _rows(bs.query_stock_basic())
    finally:
        bs.logout()
    frame = frame[frame["type"] == "1"].copy()
    frame["symbol"] = frame["code"].map(normalize_code)
    frame["ipo_date"] = pd.to_datetime(frame["ipoDate"], errors="coerce")
    frame["out_date"] = pd.to_datetime(frame["outDate"], errors="coerce")
    listed_during = frame["ipo_date"].notna() & (frame["ipo_date"] <= pd.Timestamp(END))
    alive = frame["out_date"].isna() | (frame["out_date"] >= pd.Timestamp(START))
    frame = frame[listed_during & alive].copy()
    frame["status"] = frame["status"].map({"1": "listed", "0": "delisted"})
    frame = frame[["code", "symbol", "code_name", "ipo_date", "out_date", "status"]].sort_values("symbol")
    PATHS.meta.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def fetch_calendar() -> pd.DataFrame:
    path = PATHS.meta / "calendar.csv"
    if path.exists():
        return pd.read_csv(path)
    import baostock as bs

    bs.login()
    try:
        frame = _rows(bs.query_trade_dates(start_date=START, end_date=END))
    finally:
        bs.logout()
    frame["is_trading_day"] = frame["is_trading_day"].astype(int)
    frame = frame[frame["is_trading_day"] == 1][["calendar_date"]].rename(columns={"calendar_date": "date"})
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def fetch_bars_for(code: str) -> dict[str, Any]:
    """拉一只股票 2007–2016 的日线（不复权）+ 复权因子，写 ``raw/daily/<symbol>.csv.gz``。"""
    bs = WORKER_STATE["bs"]
    symbol = normalize_code(code)
    target = PATHS.raw / f"{symbol}.csv.gz"
    started = time.time()
    if target.exists() and target.stat().st_size > 200:
        return {"code": code, "symbol": symbol, "status": "skip", "rows": 0, "seconds": 0.0}

    bars = _with_retry(lambda: _rows(bs.query_history_k_data_plus(
        code, BAR_FIELDS, start_date=BAR_START, end_date=BAR_END, frequency="d", adjustflag="3")))
    if bars.empty:
        # 断连也会表现为空表：重连后再试一次，仍空才算真的没有数据
        _relogin()
        bs = WORKER_STATE["bs"]
        bars = _with_retry(lambda: _rows(bs.query_history_k_data_plus(
            code, BAR_FIELDS, start_date=BAR_START, end_date=BAR_END, frequency="d", adjustflag="3")))
    if bars.empty:
        return {"code": code, "symbol": symbol, "status": "empty", "rows": 0, "seconds": time.time() - started}

    factors = _with_retry(lambda: _rows(bs.query_adjust_factor(code=code, start_date=BAR_START, end_date=BAR_END)))
    bars = bars.rename(columns={"preclose": "prev_close"})
    for column in ("open", "high", "low", "close", "prev_close", "volume", "amount"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars["suspended"] = (pd.to_numeric(bars["tradestatus"], errors="coerce").fillna(0) == 0).astype("int8")
    bars["tradable"] = (1 - bars["suspended"]).astype("int8")

    if factors.empty:
        bars["adj_factor"] = 1.0
    else:
        factors["date"] = pd.to_datetime(factors["dividOperateDate"], errors="coerce")
        factors = factors.dropna(subset=["date"]).sort_values("date")
        factor_series = pd.to_numeric(factors["backAdjustFactor"], errors="coerce")
        merged = pd.merge_asof(
            bars.assign(_date=pd.to_datetime(bars["date"])).sort_values("_date"),
            pd.DataFrame({"_date": factors["date"].to_numpy(), "adj_factor": factor_series.to_numpy()}),
            on="_date", direction="backward",
        )
        # 因子表只在除权日给一行 → 首个除权日之前用最早的因子回填
        merged["adj_factor"] = merged["adj_factor"].bfill().fillna(1.0)
        bars = merged.drop(columns=["_date"])

    bars["symbol"] = symbol
    keep = ["symbol", "date", "open", "high", "low", "close", "volume", "amount",
            "prev_close", "adj_factor", "suspended", "tradable"]
    bars = bars[keep]
    with gzip.open(target, "wt", encoding="utf-8-sig", newline="") as handle:
        bars.to_csv(handle, index=False)
    return {"code": code, "symbol": symbol, "status": "ok", "rows": int(len(bars)),
            "seconds": round(time.time() - started, 1)}


def fetch_fund_for(code: str) -> dict[str, Any]:
    """拉一只股票 2007–2016 逐季的利润表代理与增速（都带 pubDate）。"""
    bs = WORKER_STATE["bs"]
    symbol = normalize_code(code)
    target = PATHS.fundamental / f"{symbol}.parquet"
    started = time.time()
    if target.exists() and target.stat().st_size > 200:
        return {"code": code, "symbol": symbol, "status": "skip", "rows": 0, "seconds": 0.0}

    profit_rows: list[dict[str, Any]] = []
    growth_rows: list[dict[str, Any]] = []
    for year, quarter in FUND_QUARTERS:
        for api, bucket in (("query_profit_data", profit_rows), ("query_growth_data", growth_rows)):
            cursor = _with_retry(lambda api=api, year=year, quarter=quarter: getattr(bs, api)(
                code=code, year=year, quarter=quarter))
            frame = _rows(cursor)
            if not frame.empty:
                bucket.append(frame.iloc[0].to_dict())
    if not profit_rows and not growth_rows:
        return {"code": code, "symbol": symbol, "status": "empty", "rows": 0, "seconds": time.time() - started}

    profit = pd.DataFrame(profit_rows).drop(columns=["code"], errors="ignore")
    growth = pd.DataFrame(growth_rows).drop(columns=["code"], errors="ignore")
    merged = profit.merge(growth, on=["pubDate", "statDate"], how="outer", suffixes=("_profit", "_growth"))
    merged.insert(0, "symbol", symbol)
    merged.to_parquet(target, index=False)
    return {"code": code, "symbol": symbol, "status": "ok", "rows": int(len(merged)),
            "seconds": round(time.time() - started, 1)}


def run_sequential(task: str, codes: list[str], limit: int | None = None) -> list[dict[str, Any]]:
    """单进程串行跑一个分片（``--shard``）。baostock 的全局连接在多进程池里会崩，
    因此改用「多个独立进程 + 每个进程串行」的编排，崩溃只影响单个分片。"""
    PATHS.ensure()
    worker = fetch_bars_for if task == "bars" else fetch_fund_for
    todo = codes[:limit] if limit else codes
    _init_worker()
    results: list[dict[str, Any]] = []
    started = time.time()
    for done, code in enumerate(todo, start=1):
        try:
            record = worker(code)
        except Exception as exc:  # noqa: BLE001
            record = {"code": code, "symbol": normalize_code(code), "status": "fail",
                      "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=2)}
            try:
                _relogin()
            except Exception:  # noqa: BLE001
                pass
        results.append(record)
        if done % 25 == 0 or done == len(todo):
            elapsed = time.time() - started
            print(f"[{task}] {done}/{len(todo)} {(done / max(elapsed, 1e-9)):.2f}/s "
                  f"剩余 {(len(todo) - done) / max(done / max(elapsed, 1e-9), 1e-9) / 60:.1f} 分钟", flush=True)
    with PATHS.log.open("a", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps({"ts": pd.Timestamp.now().isoformat(timespec="seconds"), "task": task, **record},
                                    ensure_ascii=False, default=str) + "\n")
    failures = [record for record in results if record["status"] == "fail"]
    path = PATHS.meta / f"failures-{task}.json"
    if failures:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        path.write_text(json.dumps(existing + failures, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    return results


def run_pool(task: str, codes: list[str], workers: int, limit: int | None = None) -> list[dict[str, Any]]:
    PATHS.ensure()
    worker = fetch_bars_for if task == "bars" else fetch_fund_for
    todo = codes[:limit] if limit else codes
    results: list[dict[str, Any]] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(worker, code): code for code in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001
                record = {"code": code, "symbol": normalize_code(code), "status": "fail",
                          "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=2)}
            results.append(record)
            if done % 50 == 0 or done == len(todo):
                ok = sum(1 for item in results if item["status"] in {"ok", "skip"})
                elapsed = time.time() - started
                rate = done / max(elapsed, 1e-9)
                print(f"[{task}] {done}/{len(todo)} ok={ok} {rate:.1f}/s 预计剩余 "
                      f"{(len(todo) - done) / max(rate, 1e-9) / 60:.1f} 分钟", flush=True)
    with PATHS.log.open("a", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps({"ts": pd.Timestamp.now().isoformat(timespec="seconds"), "task": task, **record},
                                    ensure_ascii=False, default=str) + "\n")
    failures = [record for record in results if record["status"] == "fail"]
    (PATHS.meta / f"failures-{task}.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return results


def consolidate() -> dict[str, Any]:
    """把 raw/daily/*.csv.gz 汇总成 ``parquet/year=YYYY/data.parquet``（列与 processed 对齐）。"""
    PATHS.ensure()
    files = sorted(PATHS.raw.glob("*.csv.gz"))
    if not files:
        raise SystemExit("没有 raw/daily 数据可汇总")
    frames = []
    for path in files:
        frame = pd.read_csv(path, dtype={"symbol": "string"}, encoding="utf-8-sig")
        frames.append(frame)
    whole = pd.concat(frames, ignore_index=True)
    whole["date"] = whole["date"].astype(str)
    whole["year"] = whole["date"].str[:4].astype(int)
    summary: dict[str, Any] = {"files": len(files), "rows": int(len(whole)), "years": {}}
    for year, chunk in whole.groupby("year"):
        folder = PATHS.parquet / f"year={year}"
        folder.mkdir(parents=True, exist_ok=True)
        ordered = chunk.drop(columns=["year"]).sort_values(["symbol", "date"])
        ordered.to_parquet(folder / "data.parquet", index=False, compression="zstd")
        summary["years"][int(year)] = {"rows": int(len(ordered)), "symbols": int(ordered["symbol"].nunique())}
    summary["symbols"] = int(whole["symbol"].nunique())
    summary["date_min"] = str(whole["date"].min())
    summary["date_max"] = str(whole["date"].max())
    (PATHS.meta / "manifest.json").write_text(
        json.dumps({"generated": pd.Timestamp.now().isoformat(timespec="seconds"), **summary},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def audit() -> dict[str, Any]:
    """完整性检查：交易日数、行数、缺口、退市股、关键区间。"""
    report: dict[str, Any] = {}
    calendar = pd.read_csv(PATHS.meta / "calendar.csv")["date"].astype(str)
    report["trade_days"] = int(len(calendar))
    by_year: dict[str, Any] = {}
    for year in range(2007, 2017):
        path = PATHS.parquet / f"year={year}" / "data.parquet"
        if not path.exists():
            by_year[str(year)] = {"missing": True}
            continue
        frame = pd.read_parquet(path)
        days = frame["date"].nunique()
        expected = int(calendar[calendar.str.startswith(str(year))].shape[0])
        by_year[str(year)] = {
            "rows": int(len(frame)),
            "symbols": int(frame["symbol"].nunique()),
            "trade_days": int(days),
            "expected_days": expected,
            "missing_days": expected - int(days),
        }
    report["by_year"] = by_year

    universe = load_universe()
    delisted = universe[universe["status"] == "delisted"]
    present = set()
    for year in range(2007, 2017):
        path = PATHS.parquet / f"year={year}" / "data.parquet"
        if path.exists():
            present |= set(pd.read_parquet(path, columns=["symbol"])["symbol"].unique())
    report["universe"] = int(len(universe))
    report["delisted_universe"] = int(len(delisted))
    report["delisted_with_bars"] = int(delisted["symbol"].isin(present).sum())
    report["symbols_with_bars"] = len(present)
    report["coverage"] = round(len(present) / max(len(universe), 1), 4)

    for label, window in (("2008", ("2008-01-01", "2008-12-31")), ("2015crash", ("2015-06-01", "2015-08-31"))):
        frames = []
        for year in sorted({int(window[0][:4]), int(window[1][:4])}):
            path = PATHS.parquet / f"year={year}" / "data.parquet"
            if path.exists():
                chunk = pd.read_parquet(path)
                frames.append(chunk[(chunk["date"] >= window[0]) & (chunk["date"] <= window[1])])
        if not frames:
            report[label] = {"missing": True}
            continue
        window_frame = pd.concat(frames, ignore_index=True)
        days = sorted(window_frame["date"].unique())
        diag = {"days": len(days), "rows": int(len(window_frame)), "first": days[0], "last": days[-1],
                "expected_days": int(((calendar >= window[0]) & (calendar <= window[1])).sum())}
        # 逐日覆盖：任一交易日的股票数不应低于当期股票数的 50%
        per_day = window_frame.groupby("date")["symbol"].nunique()
        diag["min_symbols_per_day"] = int(per_day.min())
        diag["max_gap_days"] = int(pd.Series(pd.to_datetime(days)).diff().dt.days.max())
        report[label] = diag
    (PATHS.meta / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="bars", choices=["bars", "fund", "consolidate", "audit", "all"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--shard", type=int, default=0, help="分片序号（0 起）")
    parser.add_argument("--shards", type=int, default=1, help="分片总数；>1 时改用串行模式")
    args = parser.parse_args(argv)

    PATHS.ensure()
    universe = load_universe()
    codes = universe["code"].tolist()
    calendar = fetch_calendar()
    print(f"[plan] 目录 {PATHS.root}", flush=True)
    print(f"[plan] 宇宙 {len(codes)} 只（退市 {(universe['status'] == 'delisted').sum()} 只）", flush=True)
    print(f"[plan] 交易日 {len(calendar)}（{calendar['date'].iloc[0]} ~ {calendar['date'].iloc[-1]}）", flush=True)
    print(f"[plan] 输出 raw/daily/<symbol>.csv.gz、parquet/year=YYYY/data.parquet、fundamentals/<symbol>.parquet",
          flush=True)
    if args.print_plan:
        return 0

    if args.retry_failed:
        path = PATHS.meta / "failures-bars.json"
        if path.exists():
            codes = [item["code"] for item in json.loads(path.read_text(encoding="utf-8"))]
            print(f"[plan] 重试失败清单 {len(codes)} 只", flush=True)

    if args.only in {"bars", "all"}:
        if args.shards > 1:
            codes = codes[args.shard :: args.shards]
            print(f"[shard {args.shard}/{args.shards}] {len(codes)} 只", flush=True)
            results = run_sequential("bars", codes, args.limit)
        else:
            results = run_pool("bars", codes, args.workers, args.limit)
        print(f"[bars] ok={sum(1 for r in results if r['status'] == 'ok')} "
              f"skip={sum(1 for r in results if r['status'] == 'skip')} "
              f"fail={sum(1 for r in results if r['status'] == 'fail')}", flush=True)
    if args.only in {"fund", "all"}:
        if args.shards > 1:
            codes = codes[args.shard :: args.shards]
            print(f"[shard {args.shard}/{args.shards}] {len(codes)} 只", flush=True)
            results = run_sequential("fund", codes, args.limit)
        else:
            results = run_pool("fund", codes, args.workers, args.limit)
        print(f"[fund] ok={sum(1 for r in results if r['status'] == 'ok')} "
              f"skip={sum(1 for r in results if r['status'] == 'skip')} "
              f"fail={sum(1 for r in results if r['status'] == 'fail')}", flush=True)
    if args.only in {"consolidate", "all"}:
        print("[consolidate]", json.dumps(consolidate(), ensure_ascii=False), flush=True)
    if args.only in {"audit", "all"}:
        print("[audit]", json.dumps(audit(), ensure_ascii=False, indent=1)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
