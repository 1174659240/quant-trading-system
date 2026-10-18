"""主要财务指标（东财 RPT_F10_FINANCE_MAINFINADATA）：金融业可用的 ROE/EPS/BPS/营收/利润 + 股本。

为什么要单独一张 ``mainfin``：东财通用三表（``RPT_F10_FINANCE_GBALANCE`` /
``GCASHFLOW``）**整张不含金融业**——直查 ``SECURITY_CODE="000001"`` 的资产负债表
与现金流量表都是 0 行，银行的 ROE、EPS、BPS、营收、利润在通用三表里拿不到。
``MAINFINADATA`` 对全部公司（含银行/保险/证券/北交所）都齐全。

数据来源与派生关系：
    ``quant.fundamentals`` 抓的 ``indicator`` 表就是同一个 reportName 的 31 列投影，
    已经 55 期全量落盘（``D:/quant-data/fundamentals/indicator/``）。本模块把它
    按 ``fundamentals/mainfin/`` 再落一层，好处是下游因子只依赖 mainfin，不必知道
    indicator 的存在；坏处是多一份 ~200MB 的副本。**不重复抓网络**。

口径与偏差（任何用到它的回测报告都要照抄）：
1. 每报告期只保留修订后的一行（东财无版本号）→ 存在轻微后视偏差；
2. ``ROEJQ`` 是**加权 ROE**，年报口径为全年，季报/中报为年初至今累计，跨期不可直接比；
3. ``BPS × TOTAL_SHARE ≈ 归母净资产``，可与 ``GBALANCE.TOTAL_PARENT_EQUITY`` 对拍；
4. ``TOTAL_SHARE`` / ``A_FREE_SHARE``（总股本 / 流通 A 股）是报告期末快照，
   季度内的送转、增发要等下一期才反映；
5. 报告期早于上市日或晚于退市日的行，由下游按 universe 区间裁剪。
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from . import fundamentals

MAINFIN_DIR = fundamentals.FUND_DIR / "mainfin"

# 23 个业务列 + 8 个键列；全部来自 MAINFINADATA，金融业与非金融业同表同义
MAINFIN_COLUMNS = list(fundamentals.COLUMNS["indicator"])

# 金融业三张核心字段：原先在通用三表里整列为空，现在必须非空
BANK_PROBE = {
    "000001": "平安银行",
    "600036": "招商银行",
    "601318": "中国平安",
    "600030": "中信证券",
    "601601": "中国太保",
}


def path_for(report_date: str) -> Path:
    return MAINFIN_DIR / f"{report_date}.parquet"


def build(
    *,
    start: str = "2013-03-31",
    end: str | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """把 ``fundamentals/indicator`` 逐期投影成 ``fundamentals/mainfin``。"""
    days = fundamentals.periods(start, end)
    MAINFIN_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    skipped = 0
    rows = 0
    missing: list[str] = []
    t_all = time.perf_counter()

    for idx, day in enumerate(days, 1):
        src = fundamentals.path_for("indicator", day)
        dst = path_for(day)
        if dst.exists() and not force:
            skipped += 1
            continue
        if not src.exists():
            missing.append(day)
            continue
        frame = fundamentals.load("indicator", day)
        cols = [c for c in MAINFIN_COLUMNS if c in frame.columns]
        frame = frame[cols]
        tmp = dst.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(dst)
        done += 1
        rows += len(frame)
        if verbose:
            print(f"  mainfin {idx}/{len(days)} {day} {len(frame)} 行", flush=True)

    summary = {
        "dir": str(MAINFIN_DIR),
        "periods": len(days),
        "written": done,
        "skipped": skipped,
        "rows": rows,
        "missing_source": missing,
        "seconds": round(time.perf_counter() - t_all, 1),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    (MAINFIN_DIR / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def load(report_date: str) -> pd.DataFrame:
    return pd.read_parquet(path_for(report_date))


def load_all(*, start: str | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted(MAINFIN_DIR.glob("????-??-??.parquet")):
        if start and path.stem < start:
            continue
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def coverage() -> dict:
    files = sorted(MAINFIN_DIR.glob("????-??-??.parquet"))
    if not files:
        return {"periods": 0, "rows": 0, "mb": 0.0}
    rows = sum(len(pd.read_parquet(p, columns=["SECURITY_CODE"])) for p in files)
    return {
        "periods": len(files),
        "first": files[0].stem,
        "last": files[-1].stem,
        "rows": int(rows),
        "mb": round(sum(p.stat().st_size for p in files) / 1024 / 1024, 1),
    }


def verify(codes: Sequence[str] | None = None, periods: Sequence[str] | None = None) -> pd.DataFrame:
    """抽查金融股的 ROE/EPS/BPS/营收/归母，返回长表（供 doctor / 文档引用）。"""
    codes = list(codes or BANK_PROBE)
    periods = list(periods or [p.stem for p in sorted(MAINFIN_DIR.glob("????-??-??.parquet"))][-4:])
    frames = []
    for day in periods:
        frame = load(day)
        hit = frame[frame["SECURITY_CODE"].astype(str).isin(codes)].copy()
        if hit.empty:
            continue
        hit = hit.assign(report_date=day)
        frames.append(hit)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    keep = [
        "report_date", "SECURITY_CODE", "SECURITY_NAME_ABBR", "ORG_TYPE",
        "ROEJQ", "EPSJB", "BPS", "TOTALOPERATEREVE", "PARENTNETPROFIT",
        "TOTAL_SHARE", "A_FREE_SHARE", "NOTICE_DATE",
    ]
    return out[[c for c in keep if c in out.columns]]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="quant.mainfin", description="主要财务指标表（金融股兜底）")
    parser.add_argument("--start", default="2013-03-31")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true", help="只打印金融股抽查结果")
    args = parser.parse_args(argv)

    if args.verify:
        frame = verify()
        print(frame.to_string(index=False) if not frame.empty else "（无数据，先跑 build）")
        return 0
    summary = build(start=args.start, end=args.end, force=args.force)
    print(json.dumps({k: v for k, v in summary.items() if k != "missing_source"}, ensure_ascii=False))
    if summary["missing_source"]:
        print(f"  缺源报告期 {len(summary['missing_source'])} 个：{summary['missing_source'][:5]}")
    frame = verify()
    if not frame.empty:
        print("\n金融股抽查：")
        print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
