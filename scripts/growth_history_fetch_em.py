"""补采 2007–2016 历史数据（新浪为主、腾讯兜底），替代被拉黑的 baostock。

背景（2026-10-20 实测）：
* baostock 本机 IP 被列入黑名单（``10001011``），且其财务接口不带原始披露日；
* 东财 ``stock_zh_a_hist`` 间歇性 ``RemoteDisconnected``，不可作为唯一依赖。

本脚本口径：
* 日线：新浪 ``stock_zh_a_daily``（不复权 OHLCV + 成交额，单位=股/元，与数据湖一致）+
  ``hfq-factor`` 复权因子 → 组装成湖里的 ``close × adj_factor = 后复权`` 口径；
  退市股新浪不提供，自动回退腾讯（hfq + 不复权两条序列，因子=比值）。
* 财务：``quant.fundamentals.period_frame``，与 ``D:/quant-data/fundamentals`` **完全同源**
  （同一 ``NOTICE_DATE``），补 2006-03-31~2012-12-31 共 28 期；2013 期起仍用冻结数据湖。

只写 ``D:/quant-data/growth-history/``，不触碰任何已冻结文件。所有行严格 ≤2016-12-31。

用法::

    python -m scripts.growth_history_fetch_em --task all --resume
    python -m scripts.growth_history_fetch_em --task bars --limit 5      # 冒烟
    python -m scripts.growth_history_fetch_em --task audit
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant import fundamentals as fund

ROOT = Path("D:/quant-data/growth-history")
BAR_START = "2007-01-04"
BAR_END = "2016-12-30"
FUND_PERIODS = [
    f"{year}-{month}" for year in range(2006, 2013) for month in ("03-31", "06-30", "09-30", "12-31")
]
FUND_TABLES = ("profit", "balance", "cashflow", "indicator")
BAR_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount",
               "prev_close", "adj_factor", "suspended", "tradable"]
SOURCES = Path("D:/quant-data")
UNIVERSE = SOURCES / "ref" / "universe.csv"


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

    def ensure(self) -> None:
        for path in (self.raw, self.parquet, self.fundamental, self.meta):
            path.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


def normalize_code(code: str) -> str:
    text = str(code).strip().upper()
    for noise in ("SH.", "SZ.", ".SH", ".SZ"):
        text = text.replace(noise, "")
    return text.zfill(6)


def exchange_prefix(code: str) -> str:
    return "sh" if code.startswith(("5", "6", "9")) else "sz"


def load_universe() -> pd.DataFrame:
    frame = pd.read_csv(UNIVERSE, encoding="utf-8-sig", dtype={"symbol": str})
    frame["code"] = frame["symbol"].map(normalize_code)
    frame["ipo_date"] = pd.to_datetime(frame["ipo_date"], errors="coerce")
    frame["out_date"] = pd.to_datetime(frame["out_date"], errors="coerce")
    frame = frame[frame["board"].isin(["主板", "创业板", "科创板"])].drop_duplicates("code")
    listed = frame["ipo_date"].notna() & (frame["ipo_date"] <= pd.Timestamp(BAR_END))
    return frame[listed].sort_values("code").reset_index(drop=True)


def _retry(fn, attempts: int = 4, *, sleep: float = 1.2):
    last: Exception | None = None
    for index in range(attempts):
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 - 网络接口异常类型不稳定
            last = error
            time.sleep(sleep * (index + 1))
    raise RuntimeError(f"重试 {attempts} 次仍失败: {last}") from last


@functools.lru_cache(maxsize=1)
def delisted_codes() -> frozenset[str]:
    """退市股（``status == 0``）：新浪对它们直接返回空 body，必须走腾讯。"""
    frame = pd.read_csv(UNIVERSE, encoding="utf-8-sig", dtype={"symbol": str})
    frame = frame[frame["status"].astype(str).isin(["0", "退市", "已退市"])]
    return frozenset(frame["symbol"].map(normalize_code))


def install_http_defaults() -> None:
    """akshare 内部不设 timeout、也不带 UA → 会偶发 RemoteDisconnected 或无限挂起。"""
    import socket

    socket.setdefaulttimeout(25.0)
    import requests

    original = requests.Session.request

    def wrapper(self, method, url, **kwargs):  # noqa: ANN001
        kwargs.setdefault("timeout", 30)
        headers = dict(kwargs.get("headers") or {})
        headers.setdefault("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        headers.setdefault("Referer", "https://finance.sina.com.cn/")
        kwargs["headers"] = headers
        return original(self, method, url, **kwargs)

    requests.Session.request = wrapper


# ---------------------------------------------------------------- 日线


def _sina_raw(code: str) -> tuple[pd.DataFrame, pd.Series]:
    """新浪：不复权 OHLCV+成交额（股/元）与后复权因子。"""
    import akshare as ak

    symbol = f"{exchange_prefix(code)}{code}"
    frame = _retry(lambda: ak.stock_zh_a_daily(symbol=symbol, start_date="20070101",
                                               end_date="20161231", adjust=""), attempts=2)
    if frame is None or frame.empty:
        raise ValueError("新浪无数据")
    factors = _retry(lambda: ak.stock_zh_a_daily(symbol=symbol, start_date="20000101",
                                                 end_date="20161231", adjust="hfq-factor"))
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").astype("datetime64[ns]")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if factors is None or factors.empty:
        series = pd.Series(1.0, index=frame["date"])
    else:
        factors["date"] = pd.to_datetime(factors["date"], errors="coerce").astype("datetime64[ns]")
        factors = factors.dropna(subset=["date"]).sort_values("date")
        merged = pd.merge_asof(frame[["date"]], factors, on="date", direction="backward")
        series = pd.Series(merged["hfq_factor"].to_numpy(), index=frame["date"]).bfill().fillna(1.0)
    return frame, series


def _tencent_frames(code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """腾讯：后复权 + 不复权两条序列（退市股兜底）。"""
    from quant.data.web_source import fetch_tencent

    symbol = f"{exchange_prefix(code)}{code}"
    hfq = _retry(lambda: fetch_tencent(symbol, start="2007-01-01", end="2016-12-30", adjust="hfq",
                                       timeout=8, max_pages=6), attempts=2)
    raw = _retry(lambda: fetch_tencent(symbol, start="2007-01-01", end="2016-12-30", adjust="none",
                                       timeout=8, max_pages=6), attempts=2)
    for frame in (hfq, raw):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").astype("datetime64[ns]")
    return hfq.dropna(subset=["date"]).sort_values("date"), raw.dropna(subset=["date"]).sort_values("date")


def _assemble(frame: pd.DataFrame, factors: pd.Series, calendar: pd.DatetimeIndex,
              code: str, *, amount: pd.Series | None = None) -> pd.DataFrame:
    if "date" in frame.columns:
        frame = frame.set_index("date")
    grid = pd.DataFrame(index=pd.DatetimeIndex(calendar).astype("datetime64[ns]"))
    merged = grid.join(frame[["open", "high", "low", "close", "volume"]], how="left")
    if amount is not None:
        merged["amount"] = amount.reindex(merged.index)
    else:
        merged["amount"] = np.nan
    last_real = merged["close"].dropna().index.max()
    merged["suspended"] = merged["close"].isna().astype("int8")
    merged["close"] = merged["close"].ffill()
    for column in ("open", "high", "low"):
        merged[column] = merged[column].fillna(merged["close"])
    merged["volume"] = merged["volume"].fillna(0.0)
    merged["amount"] = merged["amount"].fillna(0.0)
    merged["adj_factor"] = factors.reindex(merged.index).ffill().bfill().fillna(1.0)
    merged["prev_close"] = merged["close"].shift(1)
    merged["tradable"] = (1 - merged["suspended"]).astype("int8")
    merged["symbol"] = code
    merged = merged.reset_index().rename(columns={"index": "date"})
    # 退市股在退市日之后不应继续补行（否则会变成永远停牌的「僵尸」持仓）
    cut = pd.Timestamp(BAR_END) if last_real is pd.NaT else min(pd.Timestamp(BAR_END), last_real)
    merged = merged[merged["date"] <= cut].dropna(subset=["prev_close"])
    return merged[BAR_COLUMNS]


def fetch_bars_for(code: str, calendar: pd.DatetimeIndex) -> dict[str, Any]:
    symbol = normalize_code(code)
    target = PATHS.raw / f"{symbol}.csv.gz"
    started = time.time()
    if target.exists() and target.stat().st_size > 200:
        return {"code": symbol, "status": "skip", "rows": 0, "seconds": 0.0}

    source = "sina"
    try:
        if symbol in delisted_codes():
            raise LookupError("退市股跳过新浪")
        frame, factors = _sina_raw(symbol)
        bars = _assemble(frame, factors, calendar, symbol, amount=frame.set_index("date")["amount"])
    except Exception:  # noqa: BLE001 - 退市股新浪没有，回退腾讯
        source = "tencent"
        hfq, raw = _tencent_frames(symbol)
        if hfq.empty or raw.empty:
            return {"code": symbol, "status": "empty", "rows": 0,
                    "seconds": round(time.time() - started, 1)}
        merged = pd.merge_asof(hfq[["date", "close"]].rename(columns={"close": "close_hfq"}),
                               raw[["date", "close", "volume"]], on="date", direction="backward")
        merged["factor"] = (merged["close_hfq"] / merged["close"]).replace([np.inf, -np.inf], np.nan)
        factors = pd.Series(merged["factor"].to_numpy(), index=merged["date"]).bfill().fillna(1.0)
        raw_indexed = raw.set_index("date")
        bars = _assemble(raw_indexed[["open", "high", "low", "close", "volume"]], factors,
                         calendar, symbol,
                         amount=raw_indexed["close"] * raw_indexed["volume"])

    with gzip.open(target, "wt", encoding="utf-8-sig", newline="") as handle:
        bars.to_csv(handle, index=False)
    return {"code": symbol, "source": source, "status": "ok", "rows": int(len(bars)),
            "first": str(bars["date"].min())[:10], "last": str(bars["date"].max())[:10],
            "seconds": round(time.time() - started, 1)}


def trading_calendar() -> pd.DatetimeIndex:
    """交易日历：新浪 000001 的交易日（不再依赖东财）。"""
    path = PATHS.meta / "calendar.csv"
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["date"], encoding="utf-8-sig")
        return pd.DatetimeIndex(frame["date"])
    import akshare as ak

    frame = _retry(lambda: ak.stock_zh_a_daily(symbol="sz000001", start_date="20070101",
                                               end_date="20161231", adjust=""))
    days = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).dropna().unique()))
    pd.DataFrame({"date": days}).to_csv(path, index=False, encoding="utf-8-sig")
    return days


# ---------------------------------------------------------------- 财务


def fetch_fund_period(table: str, period: str) -> dict[str, Any]:
    target = PATHS.fundamental / table
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{period}.parquet"
    started = time.time()
    if path.exists() and path.stat().st_size > 200:
        return {"table": table, "period": period, "status": "skip", "rows": 0, "seconds": 0.0}
    frame = _retry(lambda: fund.period_frame(table, period, verbose=False))
    if frame is None or frame.empty:
        return {"table": table, "period": period, "status": "empty", "rows": 0,
                "seconds": round(time.time() - started, 1)}
    frame.to_parquet(path, index=False, compression="zstd")
    notice = pd.to_datetime(frame.get("NOTICE_DATE"), errors="coerce") if "NOTICE_DATE" in frame else None
    return {"table": table, "period": period, "status": "ok", "rows": int(len(frame)),
            "notice": int(notice.notna().sum()) if notice is not None else 0,
            "seconds": round(time.time() - started, 1)}


# ---------------------------------------------------------------- 固化与审计


def consolidate() -> dict[str, Any]:
    files = sorted(PATHS.raw.glob("*.csv.gz"))
    if not files:
        return {"files": 0, "rows": 0}
    frames = []
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return {"files": len(files), "rows": 0}
    all_bars = pd.concat(frames, ignore_index=True)
    all_bars["date"] = pd.to_datetime(all_bars["date"], errors="coerce")
    all_bars = all_bars.dropna(subset=["date"])
    total = 0
    for year, part in all_bars.groupby(all_bars["date"].dt.year):
        directory = PATHS.parquet / f"year={year}"
        directory.mkdir(parents=True, exist_ok=True)
        part.sort_values(["date", "symbol"]).to_parquet(directory / "data.parquet",
                                                        index=False, compression="zstd")
        total += len(part)
    return {"files": len(files), "rows": total, "years": int(all_bars["date"].dt.year.nunique())}


def audit() -> dict[str, Any]:
    files = sorted(PATHS.raw.glob("*.csv.gz"))
    rows = 0
    first, last = None, None
    codes: list[str] = []
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
        if frame.empty:
            continue
        rows += len(frame)
        days = pd.to_datetime(frame["date"], errors="coerce")
        low, high = days.min(), days.max()
        first = low if first is None or low < first else first
        last = high if last is None or high > last else last
        codes.append(frame["symbol"].iloc[0])
    profit_dir = PATHS.fundamental / "profit"
    periods = sorted(p.stem for p in profit_dir.glob("*.parquet")) if profit_dir.exists() else []
    leaked = last is not None and last > pd.Timestamp(BAR_END)
    return {"files": len(files), "rows": rows,
            "first": str(first)[:10] if first is not None else None,
            "last": str(last)[:10] if last is not None else None,
            "codes": len(set(codes)), "fund_periods": len(periods),
            "fund_first": periods[0] if periods else None, "fund_last": periods[-1] if periods else None,
            "leaked_after_2016": bool(leaked)}


def run(task: str, limit: int | None = None, *, sleep: float = 0.3,
        shard: tuple[int, int] | None = None) -> int:
    install_http_defaults()
    PATHS.ensure()
    log_path = PATHS.meta / "fetch-log.jsonl"
    results: list[dict[str, Any]] = []
    started = time.time()
    if task in ("bars", "all"):
        calendar = trading_calendar()
        print(f"[calendar] {len(calendar)} 个交易日 {calendar[0].date()} ~ {calendar[-1].date()}", flush=True)
        codes = load_universe()["code"].tolist()
        if shard is not None:
            index, total = shard
            codes = codes[index::total]
            print(f"[shard] {index + 1}/{total} → {len(codes)} 只", flush=True)
        if limit:
            codes = codes[:limit]
        for index, code in enumerate(codes, 1):
            try:
                row = fetch_bars_for(code, calendar)
            except Exception as error:  # noqa: BLE001
                row = {"code": normalize_code(code), "status": "error", "error": str(error)[:200]}
            results.append(row)
            if row["status"] == "ok" and index % 10 == 0:
                print(f"[bars] {index}/{len(codes)} {row.get('source')} {row['rows']} 行 "
                      f"{row.get('first')}~{row.get('last')}", flush=True)
            if index % 200 == 0:
                with log_path.open("a", encoding="utf-8") as handle:
                    for item in results[-200:]:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            time.sleep(sleep)
    if task in ("fund", "all"):
        for table in FUND_TABLES:
            for period in FUND_PERIODS:
                try:
                    row = fetch_fund_period(table, period)
                except Exception as error:  # noqa: BLE001
                    row = {"table": table, "period": period, "status": "error", "error": str(error)[:200]}
                results.append(row)
                if row["status"] == "ok":
                    print(f"[fund] {table} {period} {row['rows']} 行 notice {row['notice']}", flush=True)
                time.sleep(sleep / 2)
    if task in ("bars", "all") and shard is None:
        print(f"[parquet] {consolidate()}", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit_result = audit()
    (PATHS.meta / "audit.json").write_text(json.dumps(audit_result, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"[done] {counts} 用时 {time.time() - started:.1f}s", flush=True)
    print(json.dumps(audit_result, ensure_ascii=False), flush=True)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="补采 2007-2016 历史数据（新浪/腾讯）")
    parser.add_argument("--task", choices=["bars", "fund", "all", "audit"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 只（冒烟）")
    parser.add_argument("--sleep", type=float, default=0.3, help="每只之间休眠秒数（防限流）")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="分片并发：只跑第 I 片（1-based），与其他进程互补互不重叠")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.task == "audit":
        print(json.dumps(audit(), ensure_ascii=False, indent=2))
        return 0
    shard = None
    if args.shard:
        index_text, total_text = args.shard.split("/")
        shard = (int(index_text) - 1, int(total_text))
    return run(args.task, args.limit, sleep=args.sleep, shard=shard)


if __name__ == "__main__":
    raise SystemExit(main())
