"""补齐 2007–2016 历史日线的缺口股（新浪 JSON 长历史 + 后复权因子）。

背景：``scripts/growth_history_fetch_em.py`` 用 ``ak.stock_zh_a_daily`` 拉新浪日线，但该接口
对退市/长期停牌股返回 ``JSONDecodeError``；腾讯当日返回 501、东财 RemoteDisconnected、
baostock 账号被拉黑。实测新浪另一条路径
``CN_MarketData.getKLineData?datalen=10000`` 能返回**上市以来全历史**（含退市股），
配 ``stock_zh_a_daily(adjust="hfq-factor")`` 的因子变更点即可补齐。

本脚本只写 ``D:/quant-data/growth-history/`` 下的新文件，**不触碰**任何既有冻结层。

用法：
    python -m scripts.growth_history_gapfill              # 补全部缺口
    python -m scripts.growth_history_gapfill --limit 10   # 先试 10 只
    python -m scripts.growth_history_gapfill --consolidate
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd
import requests

from scripts import growth_history_fetch_em as em

HIST = Path("D:/quant-data/growth-history")
RAW = HIST / "raw" / "daily"
URL = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://finance.sina.com.cn/",
}
START = pd.Timestamp("2007-01-04")
END = pd.Timestamp("2016-12-30")


def existing_codes() -> set[str]:
    return {path.name[: -len(".csv.gz")] for path in RAW.glob("*.csv.gz") if path.stat().st_size > 200}


def missing_codes() -> list[str]:
    universe = pd.read_parquet(HIST / "meta" / "universe-history.parquet")
    bare = universe["code"].map(lambda value: str(value).split(".")[-1].zfill(6))
    have = existing_codes()
    seen: list[str] = []
    for code in bare:
        if code not in have and code not in seen:
            seen.append(code)
    return seen


def _retry(fn, attempts: int = 4, sleep: float = 1.5):
    last: Exception | None = None
    for index in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(sleep * (index + 1))
    raise RuntimeError(f"重试 {attempts} 次仍失败: {last}")


def sina_history(code: str) -> pd.DataFrame:
    """新浪 JSON 长历史：字段 day/open/high/low/close/volume（无成交额、无 preclose）。"""
    symbol = f"{em.exchange_prefix(code)}{code}"

    def call():
        response = requests.get(URL, params={"symbol": symbol, "scale": "240", "ma": "no",
                                             "datalen": "10000"},
                                headers=HEADERS, timeout=25)
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"返回非列表: {str(payload)[:60]}")
        return pd.DataFrame(payload)

    frame = _retry(call)
    if frame.empty:
        raise ValueError("新浪长历史为空")
    frame["date"] = pd.to_datetime(frame["day"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    return frame[(frame["date"] >= START) & (frame["date"] <= END)]


def hfq_factors(code: str, dates: pd.DatetimeIndex) -> pd.Series:
    symbol = f"{em.exchange_prefix(code)}{code}"

    def call():
        return ak.stock_zh_a_daily(symbol=symbol, adjust="hfq-factor")

    try:
        frame = _retry(call, attempts=2)
    except Exception:  # noqa: BLE001
        return pd.Series(1.0, index=dates)
    if frame is None or frame.empty:
        return pd.Series(1.0, index=dates)
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    merged = pd.merge_asof(pd.DataFrame({"date": dates}), frame, on="date", direction="backward")
    series = pd.Series(merged["hfq_factor"].to_numpy(), index=dates).bfill().fillna(1.0)
    return series.astype("float64")


def fetch_one(code: str, calendar: pd.DatetimeIndex) -> dict[str, Any]:
    target = RAW / f"{code}.csv.gz"
    started = time.time()
    if target.exists() and target.stat().st_size > 200:
        return {"code": code, "status": "skip", "rows": 0, "seconds": 0.0}
    try:
        frame = sina_history(code)
    except Exception as exc:  # noqa: BLE001
        return {"code": code, "status": "fail", "error": f"{type(exc).__name__}: {exc}"[:200],
                "seconds": round(time.time() - started, 1)}
    if frame.empty:
        return {"code": code, "status": "empty", "rows": 0, "seconds": round(time.time() - started, 1)}
    indexed = frame.set_index("date")[["open", "high", "low", "close", "volume"]]
    factors = hfq_factors(code, pd.DatetimeIndex(indexed.index))
    amount = (indexed["close"] * indexed["volume"]).astype("float64")
    bars = em._assemble(indexed, factors, calendar, code, amount=amount)
    if bars.empty:
        return {"code": code, "status": "empty", "rows": 0, "seconds": round(time.time() - started, 1)}
    with gzip.open(target, "wt", encoding="utf-8-sig", newline="") as handle:
        bars.to_csv(handle, index=False)
    return {"code": code, "status": "ok", "rows": int(len(bars)),
            "first": str(bars["date"].min())[:10], "last": str(bars["date"].max())[:10],
            "factor_at_first": float(bars["adj_factor"].iloc[0]),
            "seconds": round(time.time() - started, 1)}


def run(limit: int | None, *, sleep: float) -> int:
    codes = missing_codes()
    if limit:
        codes = codes[:limit]
    calendar = em.trading_calendar()
    print(f"[gapfill] 缺口 {len(codes)} 只，交易日 {len(calendar)}", flush=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, code in enumerate(codes, 1):
        row = fetch_one(code, calendar)
        results.append(row)
        if row["status"] == "fail":
            failures.append(row)
        if index % 10 == 0 or row["status"] != "skip":
            print(f"[{index}/{len(codes)}] {code} {row['status']} rows={row.get('rows', 0)} "
                  f"{row.get('first', '')}~{row.get('last', '')} ({row.get('seconds')}s)", flush=True)
        time.sleep(sleep)

    ok = [row for row in results if row["status"] == "ok"]
    empty = [row for row in results if row["status"] == "empty"]
    summary = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "requested": len(codes),
        "ok": len(ok),
        "empty": len(empty),
        "fail": len(failures),
        "rows": int(sum(row.get("rows", 0) for row in ok)),
        "failures": failures,
    }
    (HIST / "meta" / "gapfill-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[gapfill] ok={summary['ok']} empty={summary['empty']} fail={summary['fail']} "
          f"rows={summary['rows']}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--consolidate", action="store_true", help="补完后重建 parquet 年分区并做完整性审计")
    args = parser.parse_args(argv)
    code = run(args.limit, sleep=args.sleep)
    if args.consolidate:
        print("[consolidate]", em.consolidate(), flush=True)
        print("[audit]", json.dumps(em.audit(), ensure_ascii=False, default=str), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
