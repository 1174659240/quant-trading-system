"""采集业绩预告 / 业绩快报 / 限售解禁 / 股权质押四类数据（阶段 A）。

设计约束
--------
- 严格 ≤ 2024-12-31：报告期、解禁月、质押快照日都不越界；越界记录在合并层按可见日丢弃。
- 只新建目录与文件，不改动既有数据湖任何文件：写入前对既有层做指纹，结束后重新计算并比对。
- 断点续跑：每个采集单位落一个 ``.csv.gz``，已存在且可读即跳过；失败记 ``growth-collect-failures.json``。
- 原始层保留东财中文列名（保真），合并层统一成英文列名（供因子代码使用）。

用法
----
    python -m scripts.growth_alt_collect --only all
    python -m scripts.growth_alt_collect --only pledge --resume
    python -m scripts.growth_alt_collect --only forecast --retry-failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

LAKE = Path("D:/quant-data")
ALT = LAKE / "alt"
RAW = ALT / "raw"
CUTOFF = pd.Timestamp("2024-12-31")
START = pd.Timestamp("2016-01-01")

DATASETS = {
    "forecast": {"dir": RAW / "forecast", "parquet": ALT / "forecast.parquet"},
    "express": {"dir": RAW / "express", "parquet": ALT / "express.parquet"},
    "unlock": {"dir": RAW / "unlock", "parquet": ALT / "unlock.parquet"},
    "pledge": {"dir": RAW / "pledge", "parquet": ALT / "pledge.parquet"},
}
LOG_PATH = ALT / "growth-collect-log.jsonl"
MANIFEST_PATH = ALT / "growth-manifest.json"
FAIL_PATH = ALT / "growth-collect-failures.json"
COVERAGE_PATH = ALT / "growth-coverage.json"

# 既有层只读校验对象（不写入）
GUARD_DIRS = [RAW / "gdhs", RAW / "margin", LAKE / "events"]
GUARD_FILES = [ALT / "collect-log.jsonl"]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def log(event: dict[str, Any]) -> None:
    """append 一行 jsonl（不覆盖既有日志）。"""
    event = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
    ALT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def lake_fingerprint() -> dict[str, Any]:
    """对既有层做 (路径, 大小, mtime) 摘要，用于证明未被改动。"""
    rows: list[str] = []
    items = 0
    for folder in GUARD_DIRS:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                stat = path.stat()
                rows.append(f"{path.relative_to(LAKE)}|{stat.st_size}|{stat.st_mtime_ns}")
                items += 1
    for path in GUARD_FILES:
        if path.exists():
            stat = path.stat()
            rows.append(f"{path.relative_to(LAKE)}|{stat.st_size}|{stat.st_mtime_ns}|{sha256_file(path)}")
            items += 1
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return {"files": items, "digest": digest}


def retry(fn: Callable[[], Any], *, tries: int = 4, base: float = 2.0) -> Any:
    """指数退避重试；最后一次仍失败则抛出。"""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < tries:
                time.sleep(base * (2**attempt))
    assert last is not None
    raise last


def quarters(start: str = "2016-03-31", end: str = "2024-12-31") -> list[str]:
    out = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for md in ("03-31", "06-30", "09-30", "12-31"):
            day = f"{year}-{md}"
            if start <= day <= end:
                out.append(day)
    return out


def months(start: str = "2016-01", end: str = "2024-12") -> list[str]:
    out = []
    year, month = int(start[:4]), int(start[5:7])
    while f"{year:04d}-{month:02d}" <= end:
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def unit_path(dataset: str, key: str) -> Path:
    return DATASETS[dataset]["dir"] / f"{key}.csv.gz"


def save_unit(dataset: str, key: str, frame: pd.DataFrame) -> Path:
    path = unit_path(dataset, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip", encoding="utf-8")
    return path


def read_unit(dataset: str, key: str) -> pd.DataFrame:
    return pd.read_csv(unit_path(dataset, key), dtype=str, keep_default_na=False, na_values=[""])


def load_failures() -> dict[str, Any]:
    if FAIL_PATH.exists():
        return json.loads(FAIL_PATH.read_text(encoding="utf-8"))
    return {}


def save_failures(failures: dict[str, Any]) -> None:
    FAIL_PATH.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_failure(dataset: str, key: str, exc: Exception) -> None:
    failures = load_failures()
    bucket = failures.setdefault(dataset, {})
    bucket[key] = {"error": f"{type(exc).__name__}: {exc}"[:300], "when": datetime.now().isoformat(timespec="seconds")}
    save_failures(failures)


def clear_failure(dataset: str, key: str) -> None:
    failures = load_failures()
    bucket = failures.get(dataset)
    if bucket and key in bucket:
        bucket.pop(key)
        save_failures(failures)


# --------------------------------------------------------------------------- #
# 四个采集器
# --------------------------------------------------------------------------- #
def _akshare():
    import akshare as ak  # 延迟导入：避免 --print-plan 时也加载

    _install_http_defaults()
    return ak


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _install_http_defaults(timeout: float = 30.0) -> None:
    """给所有 requests 调用兜底 timeout 与 UA：akshare 内部调用不带 timeout，网络卡住会永久阻塞。"""
    import requests

    if getattr(requests.Session, "_growth_patched", False):
        return
    original = requests.Session.request

    def request(self, method, url, **kwargs):  # noqa: ANN001
        kwargs.setdefault("timeout", timeout)
        headers = dict(kwargs.pop("headers", None) or {})
        headers.setdefault("User-Agent", UA)
        headers.setdefault("Referer", "https://data.eastmoney.com/")
        kwargs["headers"] = headers
        return original(self, method, url, **kwargs)

    requests.Session.request = request
    requests.Session._growth_patched = True  # type: ignore[attr-defined]


def collect_forecast(resume: bool, retry_failed: bool) -> dict[str, Any]:
    ak = _akshare()
    done, skipped, failed = 0, 0, 0
    for period in quarters():
        path = unit_path("forecast", period)
        if resume and path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        attempt_keys = load_failures().get("forecast", {})
        if retry_failed and period not in attempt_keys:
            skipped += 1
            continue
        try:
            frame = retry(lambda: ak.stock_yjyg_em(date=period.replace("-", "")))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            mark_failure("forecast", period, exc)
            print(f"[forecast] {period} FAILED {type(exc).__name__}")
            continue
        frame = frame.dropna(how="all")
        if frame.empty:
            failed += 1
            mark_failure("forecast", period, RuntimeError("empty frame"))
            print(f"[forecast] {period} EMPTY")
            continue
        save_unit("forecast", period, frame)
        clear_failure("forecast", period)
        done += 1
        print(f"[forecast] {period} rows={len(frame)}")
        log({"stage": "A", "dataset": "forecast", "key": period, "rows": int(len(frame)), "status": "ok"})
    return {"dataset": "forecast", "written": done, "skipped": skipped, "failed": failed}


def collect_express(resume: bool, retry_failed: bool) -> dict[str, Any]:
    ak = _akshare()
    done, skipped, failed = 0, 0, 0
    for period in quarters():
        path = unit_path("express", period)
        if resume and path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        attempt_keys = load_failures().get("express", {})
        if retry_failed and period not in attempt_keys:
            skipped += 1
            continue
        try:
            frame = retry(lambda: ak.stock_yjkb_em(date=period.replace("-", "")))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            mark_failure("express", period, exc)
            print(f"[express] {period} FAILED {type(exc).__name__}")
            continue
        frame = frame.dropna(how="all")
        if frame.empty:
            failed += 1
            mark_failure("express", period, RuntimeError("empty frame"))
            print(f"[express] {period} EMPTY")
            continue
        save_unit("express", period, frame)
        clear_failure("express", period)
        done += 1
        print(f"[express] {period} rows={len(frame)}")
        log({"stage": "A", "dataset": "express", "key": period, "rows": int(len(frame)), "status": "ok"})
    return {"dataset": "express", "written": done, "skipped": skipped, "failed": failed}


def collect_unlock(resume: bool, retry_failed: bool) -> dict[str, Any]:
    ak = _akshare()
    done, skipped, failed = 0, 0, 0
    for month in months():
        path = unit_path("unlock", month)
        if resume and path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        attempt_keys = load_failures().get("unlock", {})
        if retry_failed and month not in attempt_keys:
            skipped += 1
            continue
        first = pd.Timestamp(f"{month}-01")
        last = first + pd.offsets.MonthEnd(0)
        start, end = first.strftime("%Y%m%d"), last.strftime("%Y%m%d")
        try:
            frame = retry(
                lambda: ak.stock_restricted_release_detail_em(start_date=start, end_date=end)
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            mark_failure("unlock", month, exc)
            print(f"[unlock] {month} FAILED {type(exc).__name__}")
            continue
        frame = frame.dropna(how="all")
        if frame.empty:
            # 有些月份确实没有解禁：落一个空表并记 ok，避免每次重跑
            save_unit("unlock", month, frame)
            done += 1
            print(f"[unlock] {month} rows=0")
            log({"stage": "A", "dataset": "unlock", "key": month, "rows": 0, "status": "ok-empty"})
            continue
        save_unit("unlock", month, frame)
        clear_failure("unlock", month)
        done += 1
        print(f"[unlock] {month} rows={len(frame)}")
        log({"stage": "A", "dataset": "unlock", "key": month, "rows": int(len(frame)), "status": "ok"})
    return {"dataset": "unlock", "written": done, "skipped": skipped, "failed": failed}


def pledge_calendar() -> list[str]:
    """用东财质押市场概况的周频快照日当日历，取 2016-01-01 ~ 2024-12-31。"""
    ak = _akshare()
    profile = retry(ak.stock_gpzy_profile_em)
    days = pd.to_datetime(profile["交易日期"], errors="coerce").dropna()
    days = days[(days >= START) & (days <= CUTOFF)]
    return sorted({d.strftime("%Y-%m-%d") for d in days})


def collect_pledge(resume: bool, retry_failed: bool) -> dict[str, Any]:
    ak = _akshare()
    try:
        calendar = pledge_calendar()
    except Exception as exc:  # noqa: BLE001
        print(f"[pledge] calendar FAILED {type(exc).__name__}: {exc}")
        return {"dataset": "pledge", "written": 0, "skipped": 0, "failed": 1}
    print(f"[pledge] calendar days={len(calendar)} ({calendar[0]} .. {calendar[-1]})")
    done, skipped, failed = 0, 0, 0
    for day in calendar:
        path = unit_path("pledge", day)
        if resume and path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        attempt_keys = load_failures().get("pledge", {})
        if retry_failed and day not in attempt_keys:
            skipped += 1
            continue
        try:
            frame = retry(lambda: ak.stock_gpzy_pledge_ratio_em(date=day.replace("-", "")))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            mark_failure("pledge", day, exc)
            print(f"[pledge] {day} FAILED {type(exc).__name__}")
            continue
        frame = frame.dropna(how="all")
        if frame.empty:
            failed += 1
            mark_failure("pledge", day, RuntimeError("empty frame"))
            continue
        save_unit("pledge", day, frame)
        clear_failure("pledge", day)
        done += 1
        if done % 20 == 0:
            print(f"[pledge] {day} rows={len(frame)} (done={done} skipped={skipped})")
        log({"stage": "A", "dataset": "pledge", "key": day, "rows": int(len(frame)), "status": "ok"})
    return {"dataset": "pledge", "written": done, "skipped": skipped, "failed": failed}


# --------------------------------------------------------------------------- #
# 合并层（英文列名）
# --------------------------------------------------------------------------- #
def _codes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.zfill(6)


def _num(series: pd.Series) -> pd.Series:
    """东财数值列偶带千分位逗号或百分号，先清洗再转数。"""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("--", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _unit_frames(dataset: str) -> list[tuple[str, pd.DataFrame]]:
    folder = DATASETS[dataset]["dir"]
    if not folder.exists():
        return []
    out = []
    for path in sorted(folder.glob("*.csv.gz")):
        key = path.name[: -len(".csv.gz")]
        out.append((key, read_unit(dataset, key)))
    return out


def build_forecast() -> pd.DataFrame:
    frames = []
    for key, frame in _unit_frames("forecast"):
        part = pd.DataFrame(
            {
                "code": _codes(frame["股票代码"]),
                "name": frame["股票简称"],
                "report_period": pd.Timestamp(key),
                "notice_date": pd.to_datetime(frame["公告日期"], errors="coerce"),
                "indicator": frame["预测指标"],
                "value": _num(frame["预测数值"]),
                "change_pct": _num(frame["业绩变动幅度"]),
                "last_year": _num(frame["上年同期值"]),
                "forecast_type": frame["预告类型"],
                "change_text": frame["业绩变动"],
                "reason": frame["业绩变动原因"],
            }
        )
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["notice_date"].notna() & (out["notice_date"] <= CUTOFF)]
    return out.sort_values(["notice_date", "code", "indicator"]).reset_index(drop=True)


def build_express() -> pd.DataFrame:
    frames = []
    for key, frame in _unit_frames("express"):
        part = pd.DataFrame(
            {
                "code": _codes(frame["股票代码"]),
                "name": frame["股票简称"],
                "report_period": pd.Timestamp(key),
                "notice_date": pd.to_datetime(frame["公告日期"], errors="coerce"),
                "eps": _num(frame["每股收益"]),
                "revenue": _num(frame["营业收入-营业收入"]),
                "revenue_last": _num(frame["营业收入-去年同期"]),
                "revenue_yoy": _num(frame["营业收入-同比增长"]),
                "revenue_qoq": _num(frame["营业收入-季度环比增长"]),
                "net_profit": _num(frame["净利润-净利润"]),
                "net_profit_last": _num(frame["净利润-去年同期"]),
                "net_profit_yoy": _num(frame["净利润-同比增长"]),
                "net_profit_qoq": _num(frame["净利润-季度环比增长"]),
                "bps": _num(frame["每股净资产"]),
                "roe": _num(frame["净资产收益率"]),
                "industry": frame["所处行业"],
            }
        )
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["notice_date"].notna() & (out["notice_date"] <= CUTOFF)]
    return out.sort_values(["notice_date", "code"]).reset_index(drop=True)


def build_unlock() -> pd.DataFrame:
    frames = []
    for key, frame in _unit_frames("unlock"):
        if frame.empty:
            continue
        part = pd.DataFrame(
            {
                "code": _codes(frame["股票代码"]),
                "name": frame["股票简称"],
                "unlock_date": pd.to_datetime(frame["解禁时间"], errors="coerce"),
                "share_type": frame["限售股类型"],
                "unlock_shares": _num(frame["解禁数量"]),
                "actual_shares": _num(frame["实际解禁数量"]),
                "actual_value": _num(frame["实际解禁市值"]),
                "pct_float_mcap": _num(frame["占解禁前流通市值比例"]),
                "prev_close": _num(frame["解禁前一交易日收盘价"]),
                "ret_20d_before": _num(frame["解禁前20日涨跌幅"]),
                "ret_20d_after": _num(frame["解禁后20日涨跌幅"]),
            }
        )
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["unlock_date"].notna() & (out["unlock_date"] <= CUTOFF)]
    return out.sort_values(["unlock_date", "code"]).reset_index(drop=True)


def build_pledge() -> pd.DataFrame:
    frames = []
    for key, frame in _unit_frames("pledge"):
        part = pd.DataFrame(
            {
                "code": _codes(frame["股票代码"]),
                "name": frame["股票简称"],
                "as_of": pd.to_datetime(frame["交易日期"], errors="coerce"),
                "industry": frame["所属行业"],
                "pledge_ratio": _num(frame["质押比例"]),
                "pledge_shares": _num(frame["质押股数"]),
                "pledge_value": _num(frame["质押市值"]),
                "pledge_count": _num(frame["质押笔数"]),
                "unrestricted_shares": _num(frame["无限售股质押数"]),
                "restricted_shares": _num(frame["限售股质押数"]),
                "ret_1y": _num(frame["近一年涨跌幅"]),
                "industry_code": frame["行业代码"] if "行业代码" in frame.columns else frame.get("所属行业代码"),
            }
        )
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["as_of"].notna() & (out["as_of"] <= CUTOFF)]
    return out.sort_values(["as_of", "code"]).reset_index(drop=True)


BUILDERS = {
    "forecast": build_forecast,
    "express": build_express,
    "unlock": build_unlock,
    "pledge": build_pledge,
}


def consolidate(only: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in only:
        builder = BUILDERS[name]
        frame = builder()
        if frame.empty:
            print(f"[consolidate] {name}: EMPTY")
            result[name] = {"rows": 0}
            continue
        path = DATASETS[name]["parquet"]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")
        result[name] = {
            "rows": int(len(frame)),
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        print(f"[consolidate] {name}: rows={len(frame)} -> {path}")
    return result


# --------------------------------------------------------------------------- #
# 覆盖报告
# --------------------------------------------------------------------------- #
def _coverage(frame: pd.DataFrame, date_col: str, code_col: str = "code") -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    yearly = frame.assign(_year=dates.dt.year).groupby("_year").agg(
        rows=(code_col, "size"), stocks=(code_col, "nunique")
    )
    return {
        "rows": int(len(frame)),
        "codes": int(frame[code_col].nunique()),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "missing_rate": {
            col: round(float(frame[col].isna().mean()), 4)
            for col in frame.columns
            if col not in {"name", "change_text", "reason", "industry", "share_type", "indicator", "forecast_type"}
        },
        "by_year": {int(k): {"rows": int(v["rows"]), "stocks": int(v["stocks"])} for k, v in yearly.iterrows()},
    }


def coverage_report() -> dict[str, Any]:
    out: dict[str, Any] = {}
    plan = {
        "forecast": ("notice_date", "report_period"),
        "express": ("notice_date", "report_period"),
        "unlock": ("unlock_date", "unlock_date"),
        "pledge": ("as_of", "as_of"),
    }
    for name, (date_col, extra) in plan.items():
        path = DATASETS[name]["parquet"]
        if not path.exists():
            out[name] = {"rows": 0, "note": "parquet missing"}
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            out[name] = {"rows": 0}
            continue
        entry = _coverage(frame, date_col)
        # 额外的可见性分布：按"报告期"看公告日滞后天数
        if name == "forecast":
            lag = (pd.to_datetime(frame["notice_date"]) - pd.to_datetime(frame["report_period"])).dt.days
            entry["notice_lag_days"] = {
                "p10": int(lag.quantile(0.1)),
                "p50": int(lag.quantile(0.5)),
                "p90": int(lag.quantile(0.9)),
            }
            entry["indicators"] = frame["indicator"].value_counts().head(8).to_dict()
            entry["forecast_types"] = frame["forecast_type"].value_counts().to_dict()
        out[name] = entry
    return out


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def print_plan() -> None:
    print("既有层指纹（只读）:", json.dumps(lake_fingerprint(), ensure_ascii=False))
    for name, spec in DATASETS.items():
        print(f"{name}: dir={spec['dir']} parquet={spec['parquet']}")
    print("quarters:", len(quarters()), quarters()[0], "..", quarters()[-1])
    print("months:", len(months()), months()[0], "..", months()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="采集成长策略所需的三类补充数据")
    parser.add_argument("--only", default="all", choices=["all", *DATASETS.keys()])
    parser.add_argument("--resume", action="store_true", help="已有文件跳过")
    parser.add_argument("--retry-failed", action="store_true", help="只补失败清单里的单位")
    parser.add_argument("--skip-consolidate", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args(argv)

    if args.print_plan:
        print_plan()
        return 0

    ALT.mkdir(parents=True, exist_ok=True)
    before = lake_fingerprint()
    log({"stage": "A", "event": "start", "only": args.only, "guard_before": before})

    targets = list(DATASETS) if args.only == "all" else [args.only]
    collectors = {
        "forecast": collect_forecast,
        "express": collect_express,
        "unlock": collect_unlock,
        "pledge": collect_pledge,
    }
    summary = []
    for name in targets:
        t0 = time.time()
        print(f"=== {name} start ===")
        result = collectors[name](args.resume, args.retry_failed)
        result["seconds"] = round(time.time() - t0, 1)
        summary.append(result)
        print(f"=== {name} done in {result['seconds']}s: {result} ===")
        log({"stage": "A", "event": "dataset_done", **result})

    consolidated: dict[str, Any] = {}
    if not args.skip_consolidate:
        consolidated = consolidate(targets)

    coverage = coverage_report()
    COVERAGE_PATH.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    after = lake_fingerprint()
    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "cutoff": str(CUTOFF.date()),
        "collect": summary,
        "consolidated": consolidated,
        "coverage": coverage,
        "guard_before": before,
        "guard_after": after,
        "guard_unchanged": before["digest"] == after["digest"],
        "failures": load_failures(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log({"stage": "A", "event": "finish", "guard_unchanged": manifest["guard_unchanged"], "collect": summary})

    print(json.dumps({"collect": summary, "guard_unchanged": manifest["guard_unchanged"]}, ensure_ascii=False))
    if not manifest["guard_unchanged"]:
        print("!! 既有层指纹变化：请立即核对", before, after)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
