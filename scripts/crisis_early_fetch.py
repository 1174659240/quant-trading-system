"""早期危机段日线补采（任务一）：baostock，落 ``D:/quant-data/early/``，与现有日线不混写。

样本：
    2001-2005  互联网泡沫后慢熊 + 股权分置改革（主段）
    1997-1999  亚洲金融危机（可选段，只用于市场帧）

纪律：只写 ``D:/quant-data/early/``；不读写 processed / holdout / fundamentals。
价格用 baostock ``adjustflag=1``（后复权），``adj_factor`` 记 1.0、``prev_close`` 用后复权收盘前值，
保证收益率自洽（与 ``processed`` 的"不复权 + adj_factor"表示不同，见 manifest 的 note）。

用法：
    python -m scripts.crisis_early_fetch --plan
    python -m scripts.crisis_early_fetch --sample 2001-2005
    python -m scripts.crisis_early_fetch --sample all
    python -m scripts.crisis_early_fetch --sample 2001-2005 --consolidate
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

ROOT = Path("D:/quant-data/early")
BASIC_CACHE = ROOT / "meta" / "universe-basic.csv"
FIELDS = "date,open,high,low,close,volume,amount,tradestatus,pctChg,isST"
INDICES = {"sh.000001": "000001", "sz.399001": "399001"}

SAMPLES = {
    "2001-2005": ("2001-01-01", "2005-12-31"),
    "1997-1999": ("1997-01-01", "1999-12-31"),
}


# --------------------------------------------------------------------------- #
# baostock 会话
# --------------------------------------------------------------------------- #
def login() -> None:
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"baostock login 失败 {result.error_code}: {result.error_msg}")


def logout() -> None:
    try:
        bs.logout()
    except Exception:  # noqa: BLE001
        pass


def _rows(cursor) -> list[list[str]]:
    rows = []
    while cursor.error_code == "0" and cursor.next():
        rows.append(cursor.get_row_data())
    if cursor.error_code != "0":
        raise RuntimeError(f"baostock 查询错误 {cursor.error_code}: {cursor.error_msg}")
    return rows


def stock_basic() -> pd.DataFrame:
    """全市场证券列表（缓存），只保留 type==1 的股票。"""
    BASIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if BASIC_CACHE.is_file():
        return pd.read_csv(BASIC_CACHE)
    cursor = bs.query_stock_basic()
    rows = _rows(cursor)
    frame = pd.DataFrame(rows, columns=["code", "name", "ipo_date", "out_date", "type", "status"])
    frame = frame[frame["type"] == "1"].reset_index(drop=True)
    frame.to_csv(BASIC_CACHE, index=False, encoding="utf-8-sig")
    return frame


def codes_for(sample: str, *, order: str = "stride") -> pd.DataFrame:
    start, end = SAMPLES[sample]
    basic = stock_basic()
    ipo = pd.to_datetime(basic["ipo_date"], errors="coerce")
    out = pd.to_datetime(basic["out_date"], errors="coerce")
    keep = (ipo.notna()) & (ipo <= pd.Timestamp(end))
    keep &= out.isna() | (out >= pd.Timestamp(start))
    frame = basic[keep].copy()
    frame["symbol"] = frame["code"].str.split(".").str[1]
    frame = frame.sort_values("code").reset_index(drop=True)
    if order == "stride":
        # 按代码后三位交错，使任一时刻已采样本跨交易所/板块均衡（断点续跑时尤其重要）
        frame = frame.assign(_key=frame["symbol"].str[-3:]).sort_values(
            ["_key", "symbol"]).drop(columns="_key").reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------- #
# 抓取
# --------------------------------------------------------------------------- #
def _frame(code: str, start: str, end: str) -> pd.DataFrame:
    cursor = bs.query_history_k_data_plus(
        code, FIELDS, start_date=start, end_date=end, frequency="d", adjustflag="1")
    rows = _rows(cursor)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=FIELDS.split(","))
    frame = frame.rename(columns={"tradestatus": "trade_status", "pctChg": "pct_chg", "isST": "is_st"})
    for column in ("open", "high", "low", "close", "volume", "amount", "pct_chg"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date")
    frame["prev_close"] = frame["close"].shift(1)
    frame["adj_factor"] = 1.0
    suspended = frame["trade_status"].astype(str).str.strip() != "1"
    frame["suspended"] = suspended.astype("int8")
    frame["tradable"] = (~suspended).astype("int8")
    frame["volume"] = frame["volume"].fillna(0.0)
    frame["amount"] = frame["amount"].fillna(0.0)
    out = pd.DataFrame({
        "symbol": code.split(".")[1],
        "date": frame["date"].dt.strftime("%Y-%m-%d"),
        "open": frame["open"], "high": frame["high"], "low": frame["low"], "close": frame["close"],
        "volume": frame["volume"], "amount": frame["amount"],
        "prev_close": frame["prev_close"],
        "adj_factor": frame["adj_factor"],
        "suspended": frame["suspended"], "tradable": frame["tradable"],
        "pct_chg": frame["pct_chg"], "is_st": frame["is_st"].astype("int8"),
    })
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def fetch_sample(sample: str, *, limit: int | None = None) -> dict:
    start, end = SAMPLES[sample]
    out = ROOT / "bars" / sample
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    universe = codes_for(sample)
    if limit:
        universe = universe.head(limit)
    failures: list[dict] = []
    fetched = skipped = 0
    started = time.time()
    print(f"[{sample}] 宇宙 {len(universe)} 只，{start} ~ {end}", flush=True)

    for position, row in enumerate(universe.itertuples(), start=1):
        code, symbol = row.code, row.symbol
        target = raw / f"{symbol}.csv.gz"
        if target.is_file() and target.stat().st_size > 200:
            skipped += 1
            continue
        for attempt in (1, 2, 3):
            try:
                frame = _frame(code, start, end)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    failures.append({"code": code, "error": f"{type(exc).__name__}: {exc}"[:200]})
                    frame = None
                else:
                    time.sleep(2.0 * attempt)
                    login()
        if frame is None:
            continue
        if frame.empty:
            failures.append({"code": code, "error": "空表"})
            continue
        frame.to_csv(target, index=False, compression="gzip", encoding="utf-8-sig")
        fetched += 1
        if position % 100 == 0:
            rate = position / max(time.time() - started, 1e-9) * 60
            print(f"  {position}/{len(universe)}  {rate:.0f} 只/分  "
                  f"新增 {fetched} 跳过 {skipped} 失败 {len(failures)}", flush=True)
        if position % 200 == 0:
            logout()
            login()

    payload = {
        "sample": sample, "start": start, "end": end,
        "universe": int(len(universe)), "fetched": fetched, "skipped": skipped,
        "failures": failures, "elapsed": round(time.time() - started, 1),
    }
    (ROOT / "meta").mkdir(parents=True, exist_ok=True)
    (ROOT / "meta" / f"failures-{sample}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{sample}] 完成：新增 {fetched} / 跳过 {skipped} / 失败 {len(failures)}，"
          f"{payload['elapsed']}s", flush=True)
    return payload


def fetch_indices(sample: str) -> None:
    start, end = SAMPLES[sample]
    target_dir = ROOT / "indices" / sample      # 按样本分目录，避免后采的样本覆盖前一个
    target_dir.mkdir(parents=True, exist_ok=True)
    for code, symbol in INDICES.items():
        cursor = bs.query_history_k_data_plus(
            code, "date,open,high,low,close,volume,amount",
            start_date=start, end_date=end, frequency="d", adjustflag="3")
        rows = _rows(cursor)
        if not rows:
            continue
        frame = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        for column in ("open", "high", "low", "close", "volume", "amount"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.to_csv(target_dir / f"{symbol}.csv", index=False, encoding="utf-8-sig")
        print(f"  [index] {code} {len(frame)} 行", flush=True)


def consolidate(sample: str) -> dict:
    start, end = SAMPLES[sample]
    raw = ROOT / "bars" / sample / "raw"
    files = sorted(raw.glob("*.csv.gz"))
    if not files:
        print(f"[{sample}] 无 raw 文件", flush=True)
        return {}
    frames = []
    for path in files:
        frame = pd.read_csv(path, dtype={"symbol": str})
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data[(data["date"] >= pd.Timestamp(start)) & (data["date"] <= pd.Timestamp(end))]
    out = ROOT / "bars" / sample / "parquet"
    if out.exists():
        for old in out.glob("year=*"):
            if old.is_dir():
                shutil.rmtree(old)      # Windows 上目录不能用 unlink
            else:
                old.unlink()
    data["year"] = data["date"].dt.year
    for year, part in data.groupby("year"):
        target = out / f"year={year}"
        target.mkdir(parents=True, exist_ok=True)
        part.drop(columns=["year"]).to_parquet(target / "data.parquet", compression="zstd", index=False)
    symbols = data.groupby("symbol")["date"].agg(["min", "max", "count"]).reset_index()
    symbols.columns = ["symbol", "first_date", "last_date", "rows"]
    meta = ROOT / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    symbols.to_csv(meta / f"symbols-{sample}.csv", index=False, encoding="utf-8-sig")
    calendar = pd.Series(sorted(data["date"].unique()))
    pd.DataFrame({"date": calendar.dt.strftime("%Y-%m-%d")}).to_csv(
        meta / f"calendar-{sample}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "sample": sample, "start": start, "end": end,
        "rows": int(len(data)), "symbols": int(data["symbol"].nunique()),
        "days": int(data["date"].nunique()),
        "first": str(data["date"].min().date()), "last": str(data["date"].max().date()),
        "note": "adjustflag=1 后复权价；adj_factor 记 1.0；prev_close 为后复权前值",
    }
    (meta / f"manifest-{sample}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{sample}] 合并层 {summary['rows']} 行 / {summary['symbols']} 只 / {summary['days']} 天", flush=True)
    return summary


def plan() -> None:
    for sample in SAMPLES:
        universe = codes_for(sample)
        print(f"{sample}: 宇宙 {len(universe)} 只，{SAMPLES[sample][0]} ~ {SAMPLES[sample][1]}")
    basic = stock_basic()
    print(f"全市场证券 {len(basic)} 行（type==1），缓存 {BASIC_CACHE}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="早期危机段日线补采（baostock）")
    parser.add_argument("--sample", choices=[*SAMPLES, "all"], default="2001-2005")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args(argv)

    login()
    try:
        if args.plan:
            plan()
            return 0
        samples = list(SAMPLES) if args.sample == "all" else [args.sample]
        for sample in samples:
            fetch_indices(sample)
            if not args.index_only:
                fetch_sample(sample, limit=args.limit)
            if args.consolidate:
                consolidate(sample)
    finally:
        logout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
