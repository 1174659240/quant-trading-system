"""三类独立数据采集：股东户数（东财）、融资融券（交易所）、个股资金流（探测）。

只做采集与落盘，不做任何因子/回测；不读取、不触碰 ``D:/quant-data/holdout``
（该目录有 ``.LOCKED``，本脚本没有任何指向它的路径）。

产物（全部新建，不覆盖现有 results）::

    D:/quant-data/alt/raw/gdhs/<YYYYMMDD>.csv.gz          每个报告期一张全市场表
    D:/quant-data/alt/raw/margin/sse/<YYYYMMDD>.csv.gz    沪市当日个股两融明细
    D:/quant-data/alt/raw/margin/szse/<YYYYMMDD>.csv.gz   深市当日个股两融明细
    D:/quant-data/alt/collect-log.jsonl                   每次调用的结果（成功/失败/行数/耗时）

用法::

    python -u scripts/alt_data_collect.py --probe          # 只探测接口，不写数据
    python -u scripts/alt_data_collect.py --gdhs           # 股东户数（约 2 分钟）
    python -u scripts/alt_data_collect.py --margin         # 两融抽样日（约 25 分钟）
    python -u scripts/alt_data_collect.py --margin --limit-dates 6
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

LAKE = Path("D:/quant-data")
ALT = LAKE / "alt"
RAW = ALT / "raw"
LOG = ALT / "collect-log.jsonl"

# 采集窗口：2016 起（2016 年为因子预热）；2025-01-01 之后一律不采。
# 股东户数只取到 2024-09-30：2024-12-31 那份的公告日期在 2025 年，属于 2025 年公开信息，
# 按纪律不采（它本来也只会被 2025 年之后的回测用到）。
GDHS_START, GDHS_END = "2016-03-31", "2024-09-30"
MARGIN_START, MARGIN_END = "2016-12-01", "2024-12-31"
# 两融只在「月末 + 月末前 5/20 个交易日」采样：够算 5 日/20 日环比，又不用拉 1960 天
MARGIN_OFFSETS = (0, 5, 20)


# ---------------------------------------------------------------------------
def calendar() -> pd.DatetimeIndex:
    """交易日历（只读 meta/calendar.csv）。"""
    frame = pd.read_csv(LAKE / "meta" / "calendar.csv")
    col = "date" if "date" in frame.columns else frame.columns[0]
    return pd.DatetimeIndex(pd.to_datetime(frame[col])).sort_values()


def log(record: dict) -> None:
    ALT.mkdir(parents=True, exist_ok=True)
    record["ts"] = datetime.now().isoformat(timespec="seconds")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")


def _fetch(name: str, fn, *, retries: int = 3):
    """带重试的抓取，返回 (frame_or_None, meta)。"""
    err = ""
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            frame = fn()
            return frame, {"ok": True, "rows": int(len(frame)), "secs": round(time.time() - t0, 2),
                           "attempt": attempt}
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"[:200]
            time.sleep(1.5 * attempt)
    return None, {"ok": False, "error": err}


# ---------------------------------------------------------------------------
def quarters(start: str, end: str) -> list[str]:
    out = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            stamp = f"{year}{month:02d}{day:02d}"
            if start.replace("-", "") <= stamp <= end.replace("-", ""):
                out.append(stamp)
    return out


def collect_gdhs() -> None:
    import akshare as ak

    target = RAW / "gdhs"
    periods = quarters(GDHS_START, GDHS_END)
    print(f"[gdhs] 报告期 {len(periods)} 个：{periods[0]} ~ {periods[-1]}", flush=True)
    for i, period in enumerate(periods, 1):
        path = target / f"{period}.csv.gz"
        if path.exists():
            print(f"[gdhs] {i}/{len(periods)} {period} 已存在，跳过", flush=True)
            continue
        frame, meta = _fetch("gdhs", lambda p=period: ak.stock_zh_a_gdhs(symbol=p))
        if frame is None or not len(frame):
            print(f"[gdhs] {i}/{len(periods)} {period} 失败 {meta}", flush=True)
            log({"kind": "gdhs", "key": period, **meta})
            continue
        _save(frame, path)
        log({"kind": "gdhs", "key": period, "cols": list(frame.columns), **meta})
        print(f"[gdhs] {i}/{len(periods)} {period} rows={len(frame)} {meta['secs']}s", flush=True)


def margin_dates() -> list[pd.Timestamp]:
    cal = calendar()
    cal = cal[(cal >= pd.Timestamp(MARGIN_START)) & (cal <= pd.Timestamp(MARGIN_END))]
    picks: set[pd.Timestamp] = set()
    for _, group in cal.to_series().groupby([cal.year, cal.month]):
        month_end = group.iloc[-1]
        pos = cal.get_loc(month_end)
        for off in MARGIN_OFFSETS:
            if pos - off >= 0 and cal[pos - off] >= pd.Timestamp(MARGIN_START):
                picks.add(cal[pos - off])
    return sorted(picks)


def collect_margin(limit_dates: int | None = None) -> None:
    import akshare as ak

    dates = margin_dates()
    if limit_dates:
        dates = dates[-limit_dates:]
    print(f"[margin] 抽样日 {len(dates)} 个：{dates[0].date()} ~ {dates[-1].date()}", flush=True)
    for i, day in enumerate(dates, 1):
        stamp = day.strftime("%Y%m%d")
        for exch, fn in (("sse", ak.stock_margin_detail_sse), ("szse", ak.stock_margin_detail_szse)):
            path = RAW / "margin" / exch / f"{stamp}.csv.gz"
            if path.exists():
                continue
            frame, meta = _fetch(f"margin-{exch}", lambda d=stamp, f=fn: f(date=d))
            if frame is None or not len(frame):
                print(f"[margin] {i}/{len(dates)} {exch} {stamp} 失败 {meta.get('error', '')[:80]}", flush=True)
                log({"kind": f"margin-{exch}", "key": stamp, **meta})
                continue
            _save(frame, path)
            log({"kind": f"margin-{exch}", "key": stamp, "cols": list(frame.columns), **meta})
        if i % 10 == 0 or i == len(dates):
            print(f"[margin] {i}/{len(dates)} {stamp}", flush=True)


# ---------------------------------------------------------------------------
def probe() -> None:
    """接口可用性探测（只打印，不落数据）。"""
    import akshare as ak
    import requests

    print("akshare", ak.__version__, flush=True)

    frame, meta = _fetch("gdhs-20231229", lambda: ak.stock_zh_a_gdhs(symbol="20231229"))
    print("[probe] gdhs 全市场(报告期)", meta, list(frame.columns)[:6] if frame is not None else "", flush=True)

    frame, meta = _fetch("gdhs-detail", lambda: ak.stock_zh_a_gdhs_detail_em(symbol="000001"))
    if frame is not None:
        det = frame.copy()
        det["股东户数公告日期"] = pd.to_datetime(det["股东户数公告日期"], errors="coerce")
        gap = (det["股东户数公告日期"] - pd.to_datetime(det["股东户数统计截止日"])).dt.days
        print(f"[probe] gdhs 单股明细 rows={len(det)} 报告期 {det['股东户数统计截止日'].min()} ~ "
              f"{det['股东户数统计截止日'].max()} 公告滞后中位数 {gap.median():.0f} 天", flush=True)
    print("[probe] gdhs 明细", meta, flush=True)

    frame, meta = _fetch("margin-sse", lambda: ak.stock_margin_detail_sse(date="20240102"))
    print("[probe] 两融 沪市", meta, list(frame.columns) if frame is not None else "", flush=True)

    frame, meta = _fetch("margin-szse", lambda: ak.stock_margin_detail_szse(date="20240102"))
    print("[probe] 两融 深市", meta, list(frame.columns) if frame is not None else "", flush=True)

    # 个股资金流：akshare 包装版 + 裸接口，各试两次，确认服务端历史深度
    frame, meta = _fetch("fflow-ak", lambda: ak.stock_individual_fund_flow(stock="000001", market="sz"), retries=2)
    print("[probe] 资金流 akshare", meta, flush=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
               "Referer": "https://data.eastmoney.com/"}
    params = {"lmt": "0", "klt": "101", "secid": "0.000001",
              "fields1": "f1,f2,f3,f7",
              "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
              "ut": "b2884a393a59ad64002292a3e90d46a5"}
    try:
        resp = requests.get("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
                            params=params, headers=headers, timeout=10)
        klines = (resp.json().get("data") or {}).get("klines") or []
        print(f"[probe] 资金流 裸接口 200 rows={len(klines)} first={klines[0][:11] if klines else ''} "
              f"last={klines[-1][:11] if klines else ''}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[probe] 资金流 裸接口 FAIL", type(exc).__name__, str(exc)[:120], flush=True)


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三类独立数据采集（股东户数/两融/资金流探测）")
    parser.add_argument("--probe", action="store_true", help="只探测接口")
    parser.add_argument("--gdhs", action="store_true", help="采集股东户数")
    parser.add_argument("--margin", action="store_true", help="采集两融抽样日")
    parser.add_argument("--limit-dates", type=int, default=None, help="两融只跑最后 N 个抽样日（冒烟）")
    parser.add_argument("--all", action="store_true", help="gdhs + margin")
    args = parser.parse_args(argv)

    ALT.mkdir(parents=True, exist_ok=True)
    if args.probe:
        probe()
    if args.gdhs or args.all:
        collect_gdhs()
    if args.margin or args.all:
        collect_margin(limit_dates=args.limit_dates)
    if not any((args.probe, args.gdhs, args.margin, args.all)):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
