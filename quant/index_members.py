"""指数成分快照（沪深300/中证500/中证1000/上证50/科创50/创业板指）。

残酷现实（第三节探针实测）：akshare 与中证指数官网**都没有按日期取历史成分的接口**，
只能拿到"当前成分"，所以任何"当时是否成分股"的筛选都带幸存者偏差。

这里能做两步补救：
1. 中证系接口自带 ``日期``（快照日），落盘时把 as_of 记下来，将来多次运行可攒出成分变动史；
2. ``index_stock_cons`` 给每个成分的 ``纳入日期``，可用作"何时进指数"的下界——
   但**没有剔除日期**，所以不能反推"当时不在指数里"。

用法::

    python -m quant lake index            # 全部指数快照
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from .refdata import REF_DIR

INDEX_CSV = REF_DIR / "index_members.csv"

INDICES = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "000016": "上证50",
    "000688": "科创50",
    "399006": "创业板指",
}


def _ak():
    try:
        import akshare as ak  # 可选依赖，只有抓指数成分时才需要
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise RuntimeError("缺少 akshare：pip install akshare（本系统其余部分不依赖它）") from exc
    return ak


def snapshot(code: str, name: str) -> pd.DataFrame:
    """单个指数的当前成分。失败抛 RuntimeError，由 build 记账。"""
    ak = _ak()
    errors = []
    try:
        raw = ak.index_stock_cons_csindex(symbol=code)
        if len(raw):
            frame = pd.DataFrame(
                {
                    "index_code": code,
                    "index_name": name,
                    "symbol": raw["成分券代码"].astype(str).str.zfill(6),
                    "stock_name": raw["成分券名称"],
                    "as_of": raw["日期"].astype(str).str[:10],
                    "in_date": "",
                    "weight": pd.NA,
                    "source": "csindex",
                }
            )
            return frame
        errors.append("csindex 空表")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"csindex: {type(exc).__name__} {exc}")

    try:
        raw = ak.index_stock_cons(symbol=code)
        if len(raw):
            frame = pd.DataFrame(
                {
                    "index_code": code,
                    "index_name": name,
                    "symbol": raw["品种代码"].astype(str).str.zfill(6),
                    "stock_name": raw["品种名称"],
                    "as_of": date.today().isoformat(),
                    "in_date": raw["纳入日期"].astype(str).str[:10],
                    "weight": pd.NA,
                    "source": "sina",
                }
            )
            return frame
        errors.append("sina 空表")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina: {type(exc).__name__} {exc}")
    raise RuntimeError(f"{code} {name}: {' / '.join(errors)}")


def build(codes: Sequence[str] | None = None, *, merge: bool = True) -> dict:
    """抓指数成分快照并落盘；``merge=True`` 时与既有文件合并（同一指数同一快照日覆盖）。"""
    wanted = list(codes) if codes else list(INDICES)
    frames = []
    failures: dict[str, str] = {}
    for code in wanted:
        name = INDICES.get(code, code)
        try:
            frames.append(snapshot(code, name))
            print(f"  {code} {name}: {len(frames[-1])} 只", flush=True)
        except RuntimeError as exc:
            failures[code] = str(exc)
            print(f"  ✗ {code} {name}: {exc}", flush=True)

    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merge and INDEX_CSV.exists() and len(fresh):
        old = pd.read_csv(INDEX_CSV, dtype={"symbol": str, "as_of": str, "in_date": str, "index_code": str})
        old["weight"] = pd.to_numeric(old.get("weight"), errors="coerce")
        key = ["index_code", "as_of"]
        old = old.merge(
            fresh[key].drop_duplicates().assign(_new=1), on=key, how="left"
        )
        old = old[old["_new"].isna()].drop(columns="_new")
        combined = pd.concat([old, fresh], ignore_index=True)
    else:
        combined = fresh
    combined = combined.drop_duplicates(["index_code", "symbol", "as_of"], keep="last")

    if len(combined):
        combined.to_csv(INDEX_CSV, index=False, encoding="utf-8")
    summary = {
        "as_of": date.today().isoformat(),
        "indices": {
            code: {
                "name": INDICES.get(code, code),
                "members": int((combined["index_code"] == code).sum()) if len(combined) else 0,
            }
            for code in wanted
        },
        "rows": int(len(combined)),
        "failures": failures,
        "written": str(INDEX_CSV),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return summary


def load() -> pd.DataFrame:
    return pd.read_csv(INDEX_CSV, dtype={"symbol": str, "as_of": str, "in_date": str, "index_code": str})


def members(code: str, day: str | None = None) -> set[str]:
    """某指数在 ``day`` 当天可用的成分（取 <= day 的最近一版快照）。"""
    frame = load()
    frame = frame[frame["index_code"] == str(code)]
    if day:
        avail = sorted(d for d in frame["as_of"].unique() if d <= day)
        if not avail:
            return set()
        frame = frame[frame["as_of"] == avail[-1]]
    return set(frame["symbol"])


def write_manifest(payload: dict) -> Path:
    path = REF_DIR / "manifest.json"
    merged = {}
    if path.exists():
        try:
            merged = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged["index_members"] = payload
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
