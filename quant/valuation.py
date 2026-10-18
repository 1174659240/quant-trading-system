"""估值与 ST 因子快照：baostock 日频 peTTM / pbMRQ / psTTM / pcfNcfTTM / turn / isST。

设计要点：
- 逐股票落 ``D:/quant-data/valuation/<symbol>.parquet``，可断点续跑（文件存在即跳过）；
- 每次抓取写 ``fetch_log.jsonl``，失败代码进 ``failures.json``；
- 数值列统一 ``to_numeric``（baostock 缺失返回空串），``isST``/``tradestatus`` 保持整数标记；
- 北交所在 baostock 无覆盖，抓到 0 行会记入 skipped 而不是 failure。

口径声明（写进任何用它的报告）：
1. baostock 的 peTTM/pbMRQ 是**当日收盘后**的滚动值，回测里只能用 t 日及以前的值；
2. ``isST`` 是当日是否 ST/*ST，属于事后标记但当日可知；
3. 退市股在退市日后无行情，估值序列自然截断。
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from .refdata import REF_DIR, bs_frame, ensure_bs, load_universe

VALUATION_DIR = Path("D:/quant-data/valuation")
FIELDS = "date,code,close,preclose,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
NUMERIC = ("close", "preclose", "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM")
FLAGS = ("tradestatus", "isST")
DEFAULT_START = "2014-01-01"


def path_for(symbol: str) -> Path:
    return VALUATION_DIR / f"{str(symbol).zfill(6)}.parquet"


def _append_log(record: dict, *, suffix: str = "") -> None:
    VALUATION_DIR.mkdir(parents=True, exist_ok=True)
    path = VALUATION_DIR / f"fetch_log{suffix}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log(suffix: str = "") -> list[dict]:
    path = VALUATION_DIR / f"fetch_log{suffix}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def clean(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把 baostock 的字符串表转成契约表：date 升序、数值列 float、标记列 int。"""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["symbol", "date", *NUMERIC, *FLAGS])
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in FLAGS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int8")
    frame["symbol"] = str(symbol).zfill(6)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return frame[["symbol", "date", *NUMERIC, *FLAGS]]


def fetch_symbol(symbol: str, *, start: str = DEFAULT_START, end: str | None = None, retries: int = 3):
    """单只股票的估值序列，返回 (frame, attempts)；异常/空表交给上层记账。"""
    import baostock as bs

    code = str(symbol).split(".")[0].strip().zfill(6)
    exchange = "sh" if code.startswith(("6", "9")) else "sz"
    bs_code = f"{exchange}.{code}"
    end = end or date.today().isoformat()
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, FIELDS, start_date=start, end_date=end, frequency="d", adjustflag="3"
            )
            if rs.error_code != "0":
                last_error = f"{rs.error_code}: {rs.error_msg}"
            else:
                return clean(bs_frame(rs), code), attempt
        except Exception as exc:  # noqa: BLE001 - 记录后重试
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            ensure_bs(force=True)
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"{code}: {last_error}")


def build(
    symbols: Sequence[str] | None = None,
    *,
    start: str = DEFAULT_START,
    end: str | None = None,
    pause: float = 0.10,
    limit: int | None = None,
    resume: bool = True,
    every: int = 50,
    verbose: bool = True,
    shard: tuple[int, int] | None = None,
) -> dict:
    """批量落估值序列。文件已存在即跳过（resume），失败进 failures.json。

    ``shard=(i, n)`` 表示只处理第 i 片（0 起），用于多进程并行：每片写自己的日志，
    文件级 resume 保证分片之间不会重复抓同一只。
    """
    if symbols is None:
        uni = load_universe()
        symbols = uni["symbol"].astype(str).tolist()
    symbols = [str(s).zfill(6) for s in symbols]
    if shard is not None:
        i, n = shard
        if not 0 <= i < n:
            raise ValueError(f"分片越界：{shard}")
        symbols = [s for pos, s in enumerate(symbols) if pos % n == i]
    todo = [s for s in symbols if not (resume and path_for(s).exists())]
    if limit:
        todo = todo[:limit]

    VALUATION_DIR.mkdir(parents=True, exist_ok=True)
    log_suffix = "" if shard is None else f".w{shard[0]}"
    st = ensure_bs()
    if not st["ok"]:
        raise RuntimeError(f"baostock 登录失败 {st['msg']}")

    done = skipped = 0
    rows = 0
    failures: dict[str, str] = {}
    empty: list[str] = []
    t_all = time.perf_counter()
    for idx, symbol in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            frame, attempts = fetch_symbol(symbol, start=start, end=end)
        except RuntimeError as exc:
            failures[symbol] = str(exc)
            continue
        if len(frame) == 0:
            empty.append(symbol)
            skipped += 1
            continue
        path = path_for(symbol)
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
        done += 1
        rows += len(frame)
        _append_log(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "symbol": symbol,
                "rows": int(len(frame)),
                "first": frame["date"].iloc[0].strftime("%Y-%m-%d"),
                "last": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
                "attempts": attempts,
                "seconds": round(time.perf_counter() - t0, 3),
                "bytes": path.stat().st_size,
                "fields": FIELDS,
            },
            suffix=log_suffix,
        )
        if verbose and idx % every == 0:
            elapsed = time.perf_counter() - t_all
            rate = idx / elapsed if elapsed else 0
            eta = (len(todo) - idx) / rate / 60 if rate else 0
            print(
                f"  估值 {idx}/{len(todo)} ok={done} empty={skipped} fail={len(failures)} "
                f"{rate:.2f}/s 预计剩余 {eta:.1f} 分钟",
                flush=True,
            )
        time.sleep(pause)

    summary = {
        "start": start,
        "end": end or date.today().isoformat(),
        "requested": len(symbols),
        "todo": len(todo),
        "written": done,
        "rows": rows,
        "empty": len(empty),
        "failed": len(failures),
        "empty_sample": empty[:20],
        "failures": failures,
        "seconds": round(time.perf_counter() - t_all, 1),
        "dir": str(VALUATION_DIR),
        "shard": None if shard is None else {"index": shard[0], "total": shard[1]},
    }
    (VALUATION_DIR / f"failures{log_suffix}.json").write_text(
        json.dumps({"failures": failures, "empty": empty}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def st_matrix(symbols: Sequence[str] | None = None) -> pd.DataFrame:
    """``isST`` 布尔矩阵（index=交易日，columns=6 位代码，True=当日 ST）。

    只读一列，不加载价量，所以比 :func:`load` 快；缺失日保持 NaN（转 bool 为 False）。
    """
    wanted = None if symbols is None else {str(s).zfill(6) for s in symbols}
    series: dict[str, pd.Series] = {}
    for path in sorted(VALUATION_DIR.glob("??????.parquet")):
        if wanted is not None and path.stem not in wanted:
            continue
        one = pd.read_parquet(path, columns=["date", "isST"])
        if len(one) == 0:
            continue
        series[path.stem] = pd.Series(
            one["isST"].to_numpy(), index=pd.to_datetime(one["date"]), dtype="float64"
        )
    if not series:
        return pd.DataFrame(dtype=bool)
    wide = pd.DataFrame(series).sort_index().ffill()
    return wide.fillna(0.0).astype(int).astype(bool)


def build_turn_panel(
    out: Path | None = None,
    *,
    start: str = "2014-01-01",
    cutoff: str = "2024-12-31",
) -> dict:
    """把逐股 ``turn`` 列透视成「交易日 × 代码」宽表并落盘（任务一/二的换手族因子用）。

    只读 ``date/symbol/turn`` 三列，12.4M 行约 2 秒扫描 + 6 秒透视。
    ``cutoff`` 默认 2024-12-31：holdout（2025 起）不进入任何因子面板。
    """
    target = Path(out) if out is not None else REF_DIR / "turn_panel.parquet"
    files = sorted(VALUATION_DIR.glob("??????.parquet"))
    frames = [pd.read_parquet(path, columns=["date", "symbol", "turn"]) for path in files]
    if not frames:
        raise RuntimeError(f"{VALUATION_DIR} 下没有估值文件，先跑 `python -m quant lake valuation`")
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= pd.Timestamp(start)) & (raw["date"] <= pd.Timestamp(cutoff))]
    wide = (
        raw.pivot(index="date", columns="symbol", values="turn")
        .sort_index()
        .sort_index(axis=1)
        .reset_index()
    )
    value_cols = [c for c in wide.columns if c != "date"]
    wide[value_cols] = wide[value_cols].astype("float32")
    target.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(target, index=False)
    return {
        "path": str(target),
        "rows": int(len(wide)),
        "symbols": int(wide.shape[1] - 1),
        "first": str(wide["date"].iloc[0])[:10] if len(wide) else None,
        "last": str(wide["date"].iloc[-1])[:10] if len(wide) else None,
        "mb": round(target.stat().st_size / 1024 / 1024, 1),
    }


def load(symbols: Sequence[str] | None = None) -> pd.DataFrame:
    """读取估值面板（按 symbol 过滤）；只读已落盘文件。"""
    wanted = None if symbols is None else {str(s).zfill(6) for s in symbols}
    frames = []
    for path in sorted(VALUATION_DIR.glob("??????.parquet")):
        if wanted is not None and path.stem not in wanted:
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", *NUMERIC, *FLAGS])
    return pd.concat(frames, ignore_index=True)


def snapshot(day: str) -> pd.DataFrame:
    """取某一交易日的横截面（没有该日数据的股票不出现）。"""
    frame = load()
    if len(frame) == 0:
        return frame
    ts = pd.Timestamp(day)
    return frame.loc[frame["date"] == ts].reset_index(drop=True)


def coverage() -> dict:
    """已落盘覆盖度：股票数、行数、日期范围、ST 占比。"""
    files = sorted(VALUATION_DIR.glob("??????.parquet"))
    total_rows = 0
    st_rows = 0
    first = ""
    last = ""
    for path in files:
        one = pd.read_parquet(path, columns=["date", "isST"])
        total_rows += len(one)
        st_rows += int(one["isST"].sum())
        if len(one):
            lo, hi = one["date"].min(), one["date"].max()
            first = min(first, str(lo)[:10]) if first else str(lo)[:10]
            last = max(last, str(hi)[:10]) if last else str(hi)[:10]
    return {
        "symbols": len(files),
        "rows": total_rows,
        "st_rows": st_rows,
        "st_ratio": round(st_rows / total_rows, 4) if total_rows else 0.0,
        "first": first,
        "last": last,
        "dir": str(VALUATION_DIR),
        "universe": int(len(load_universe())),
    }
