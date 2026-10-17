"""退市股日线 + 退市事件表（幸存者偏差的关键补丁）。

为什么单独一个模块：爬取层的 universe 来自新浪实时列表（5576 只），**只含 2 只退市股**，
所以 337 只退市股在原始层里根本没有日线。没有它们，任何回测都自动剔除了"最惨的那批"。

数据来源 baostock（唯一覆盖退市股的免费源，实测退市后还能查到在职期间完整日线）。
落 ``D:/quant-data/delisted/daily/<代码>.parquet``（adjustflag=1 后复权）。

退市收益三种假设（写进报告，回测跑三版）：
- ``last``：按最后交易日收盘价卖出（乐观，隐含"能在退市前跑掉"）
- ``haircut30``：退市时再打 -30%（老三板流动性折价的经验值）
- ``total_loss``：按 0 计（-100%，最保守）

事件表字段：symbol/out_date/last_bar/gap_days/last_close_hfq/bars/board，供回测按版本注入。
"""

from __future__ import annotations

import json
import socket
import time
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from .refdata import bs_frame, ensure_bs, load_universe

DELISTED_DIR = Path("D:/quant-data/delisted")
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg"
ADJUST = "1"  # 后复权：退市股的长期收益必须复权，否则除权日会被当成暴跌
EVENTS_CSV = DELISTED_DIR / "events.csv"


def path_for(symbol: str) -> Path:
    return DELISTED_DIR / "daily" / f"{str(symbol).zfill(6)}.parquet"


def _log(record: dict) -> None:
    DELISTED_DIR.mkdir(parents=True, exist_ok=True)
    with (DELISTED_DIR / "fetch_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["symbol", "date"])
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    for column in ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "tradestatus" in frame.columns:
        frame["tradestatus"] = pd.to_numeric(frame["tradestatus"], errors="coerce").fillna(0).astype("int8")
    frame["symbol"] = str(symbol).zfill(6)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return frame


def fetch_symbol(symbol: str, *, start: str = "2014-01-01", end: str | None = None, retries: int = 3):
    """默认只抓 2014 起：更早的退市股对 2014+ 的回测无意义，且老股票日线极长会拖慢整体。"""
    import baostock as bs

    code = str(symbol).split(".")[0].strip().zfill(6)
    exchange = "sh" if code.startswith(("6", "9")) else "sz"
    end = end or date.today().isoformat()
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            rs = bs.query_history_k_data_plus(
                f"{exchange}.{code}", FIELDS, start_date=start, end_date=end, frequency="d", adjustflag=ADJUST
            )
            if rs.error_code != "0":
                last_error = f"{rs.error_code}: {rs.error_msg}"
            else:
                return clean(bs_frame(rs), code), attempt
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            ensure_bs(force=True)
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"{code}: {last_error}")


def targets(as_of: str | None = None) -> pd.DataFrame:
    """universe 里已退市的股票（不含在途退市）。"""
    uni = load_universe()
    as_of = as_of or date.today().isoformat()
    mask = (uni["status"] == 0) & (uni["out_date"].astype(str) != "") & (uni["out_date"].astype(str) <= as_of)
    return uni.loc[mask].copy()


def build(*, limit: int | None = None, resume: bool = True, pause: float = 0.05, verbose: bool = True) -> dict:
    """抓退市股日线；文件存在即跳过。"""
    uni = targets()
    symbols = uni["symbol"].astype(str).tolist()
    todo = [s for s in symbols if not (resume and path_for(s).exists())]
    if limit:
        todo = todo[:limit]
    DELISTED_DIR.mkdir(parents=True, exist_ok=True)
    st = ensure_bs()
    if not st["ok"]:
        raise RuntimeError(f"baostock 登录失败 {st['msg']}")

    socket.setdefaulttimeout(180.0)
    done = 0
    rows = 0
    empty: list[str] = []
    failures: dict[str, str] = {}
    t_all = time.perf_counter()
    for idx, symbol in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            frame, attempts = fetch_symbol(symbol)
        except RuntimeError as exc:
            failures[symbol] = str(exc)
            continue
        if len(frame) == 0:
            empty.append(symbol)
            continue
        path = path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
        done += 1
        rows += len(frame)
        _log(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "symbol": symbol,
                "rows": int(len(frame)),
                "first": frame["date"].iloc[0].strftime("%Y-%m-%d"),
                "last": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
                "attempts": attempts,
                "seconds": round(time.perf_counter() - t0, 2),
            }
        )
        if verbose and idx % 25 == 0:
            elapsed = time.perf_counter() - t_all
            rate = idx / elapsed if elapsed else 0
            print(f"  退市股 {idx}/{len(todo)} ok={done} empty={len(empty)} fail={len(failures)} {rate:.2f}/s", flush=True)
        time.sleep(pause)

    summary = {
        "targets": len(symbols),
        "todo": len(todo),
        "written": done,
        "rows": rows,
        "empty": len(empty),
        "empty_sample": empty[:20],
        "failed": len(failures),
        "failures": failures,
        "seconds": round(time.perf_counter() - t_all, 1),
        "dir": str(DELISTED_DIR),
    }
    (DELISTED_DIR / "failures.json").write_text(
        json.dumps({"failures": failures, "empty": empty}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def load(symbol: str) -> pd.DataFrame:
    path = path_for(symbol)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def build_events() -> dict:
    """由 universe 的退市日 + 已抓日线生成事件表（退市前最后交易日/最后收盘）。"""
    uni = targets()
    rows = []
    for _, item in uni.iterrows():
        symbol = str(item["symbol"]).zfill(6)
        frame = load(symbol)
        out_date = str(item["out_date"])[:10]
        record = {
            "symbol": symbol,
            "code_name": item.get("code_name", ""),
            "board": item.get("board", ""),
            "out_date": out_date,
            "bars": int(len(frame)),
            "last_bar": "",
            "last_close_hfq": float("nan"),
            "first_bar": "",
            "gap_days": pd.NA,
        }
        if len(frame):
            record["first_bar"] = str(frame["date"].iloc[0])[:10]
            record["last_bar"] = str(frame["date"].iloc[-1])[:10]
            close = pd.to_numeric(frame["close"], errors="coerce").dropna()
            if len(close):
                record["last_close_hfq"] = float(close.iloc[-1])
            record["gap_days"] = (pd.Timestamp(out_date) - pd.Timestamp(record["last_bar"])).days
        rows.append(record)
    events = pd.DataFrame(rows)
    if len(events):
        events = events.sort_values("out_date").reset_index(drop=True)
        DELISTED_DIR.mkdir(parents=True, exist_ok=True)
        events.to_csv(EVENTS_CSV, index=False, encoding="utf-8")
    have = events[events["bars"] > 0] if len(events) else events
    summary = {
        "targets": int(len(events)),
        "with_bars": int(len(have)),
        "without_bars": int(len(events) - len(have)) if len(events) else 0,
        "rows": int(events["bars"].sum()) if len(events) else 0,
        "gap_median": float(pd.to_numeric(have["gap_days"], errors="coerce").median()) if len(have) else None,
        "gap_p90": float(pd.to_numeric(have["gap_days"], errors="coerce").quantile(0.9)) if len(have) else None,
        "by_year": events["out_date"].str[:4].value_counts().sort_index().to_dict() if len(events) else {},
        "written": str(EVENTS_CSV),
    }
    return summary


def delisting_return(symbol: str, mode: str = "last") -> float:
    """退市处置收益（相对最后收盘）：last=0，haircut30=-0.30，total_loss=-1.0。"""
    if mode == "last":
        return 0.0
    if mode == "haircut30":
        return -0.30
    if mode == "total_loss":
        return -1.0
    raise ValueError(f"未知退市处置假设：{mode}")
