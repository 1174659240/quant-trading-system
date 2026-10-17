"""baostock 探针：逐条实测 10 项能力，产出 results/probe-baostock.{md,json}。

设计参考（成熟项目，不重复造轮子）：
- baostock 官方 Python API 与其自带示例（字段名、ResultData 迭代方式）；
- akshare 的 A 股历史接口作为第二来源对照（本探针只做交叉验证，不引入依赖）；
- qlib 的 PIT 处理思路：财务数据必须按 pubDate（公告日）而非 statDate（报告期）可用；
- Alphalens 的分层收益口径（forward return + quantile），本探针不涉及，仅记录用于后续因子研究。

用法：
    python -m scripts.probe_baostock                 # 全量探针
    python -m scripts.probe_baostock --only 1,2,3    # 只跑指定项
    python -m scripts.probe_baostock --out results/probe-baostock

注意：本脚本只读、只写 results/ 与 _diag/；不写任何 D:\\quant-data 目录。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import pandas as pd  # noqa: E402

DEFAULT_OUT = PROJECT / "results" / "probe-baostock"

# 退市 / 长期停牌 / 更名样本：用来验证「完整生命周期」与 PIT 宇宙
DELISTED = ["sh.600001", "sz.000003", "sh.600087", "sz.000033", "sz.000405"]
# ST 摘帽 / 戴帽样本（含已退市），验证 isST 历史覆盖
ST_CODES = ["sh.600087", "sz.000033", "sh.600145", "sz.002260", "sz.000405"]

HIST_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
    "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)


def to_frame(rs) -> pd.DataFrame:
    if rs.error_code != "0":
        raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return (time.perf_counter() - t0) * 1000, value


# --------------------------------------------------------------------------
# 1. 连通性与限速
# --------------------------------------------------------------------------
def probe_connectivity(limit: int = 60) -> dict:
    import baostock as bs

    login_ms, lg = timed(bs.login)
    if lg.error_code != "0":
        return {"ok": False, "login_ms": round(login_ms, 1), "error": f"{lg.error_code}: {lg.error_msg}"}

    lat = []
    errors = []
    codes = ["sh.600519", "sz.000001", "sz.300750", "sh.601318", "sz.000858", "sh.600036"]
    for i in range(limit):
        code = codes[i % len(codes)]
        try:
            ms, rs = timed(
                lambda: bs.query_history_k_data_plus(
                    code, "date,close", start_date="2024-01-02", end_date="2024-03-01", frequency="d", adjustflag="3"
                )
            )
            if rs.error_code != "0":
                errors.append(rs.error_code)
            else:
                lat.append(ms)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    # 连续 login / logout 稳定性
    relogin = []
    for _ in range(3):
        bs.logout()
        ms, lg2 = timed(bs.login)
        relogin.append({"ms": round(ms, 1), "code": lg2.error_code})

    # 全市场清单一次拉取
    ms_basic, rs_basic = timed(bs.query_stock_basic)
    try:
        basic = to_frame(rs_basic)
        basic_n = len(basic)
    except Exception:  # noqa: BLE001
        basic_n = -1

    return {
        "ok": True,
        "login_ms": round(login_ms, 1),
        "relogin": relogin,
        "calls": len(lat),
        "latency_ms": {
            "p50": round(statistics.median(lat), 1) if lat else None,
            "p95": round(sorted(lat)[int(len(lat) * 0.95) - 1], 1) if lat else None,
            "max": round(max(lat), 1) if lat else None,
        },
        "errors": len(errors),
        "error_samples": errors[:5],
        "stock_basic_rows": basic_n,
        "stock_basic_ms": round(ms_basic, 1),
        "server_time": lg.msg if getattr(lg, "msg", None) else "",
    }


# --------------------------------------------------------------------------
# 2. 退市股 / PIT 宇宙
# --------------------------------------------------------------------------
def probe_delisted() -> dict:
    import baostock as bs

    rs = bs.query_stock_basic()
    basic = to_frame(rs)
    for col in ("ipoDate", "outDate", "status", "type"):
        if col in basic.columns:
            basic[col] = basic[col].astype(str).str.strip()

    by_status = basic["status"].value_counts().to_dict() if "status" in basic else {}
    stocks = basic[basic["type"] == "1"] if "type" in basic else basic
    delisted = stocks[stocks["status"] == "0"] if "status" in stocks else stocks.iloc[0:0]

    # PIT 宇宙：历史上某日全部证券
    pit = {}
    for day in ("2010-01-05", "2015-01-05", "2020-01-06", "2024-01-02"):
        try:
            rs2 = bs.query_all_stock(day=day)
            f2 = to_frame(rs2)
            pit[day] = {"n": len(f2), "codes": set(f2["code"]) if "code" in f2.columns else set()}
        except Exception as exc:  # noqa: BLE001
            pit[day] = {"n": -1, "error": f"{type(exc).__name__}: {exc}", "codes": set()}

    samples = []
    for code in DELISTED:
        item = {"code": code}
        row = basic[basic["code"] == code]
        if len(row):
            item.update(
                {k: row.iloc[0].get(k) for k in ("code_name", "ipoDate", "outDate", "status", "type")}
            )
        try:
            rs3 = bs.query_history_k_data_plus(
                code, HIST_FIELDS, start_date="1990-01-01", end_date="", frequency="d", adjustflag="3"
            )
            f3 = to_frame(rs3)
            item["rows"] = len(f3)
            if len(f3):
                item["first"] = f3["date"].iloc[0]
                item["last"] = f3["date"].iloc[-1]
                item["span_years"] = round(
                    (datetime.fromisoformat(f3["date"].iloc[-1]) - datetime.fromisoformat(f3["date"].iloc[0])).days
                    / 365.25,
                    1,
                )
                # 上市前 ipoDate 与首根是否一致
                if item.get("ipoDate") and item["ipoDate"] not in ("", "None"):
                    item["ipo_gap_days"] = round(
                        (datetime.fromisoformat(f3["date"].iloc[0]) - datetime.fromisoformat(item["ipoDate"])).days, 1
                    )
        except Exception as exc:  # noqa: BLE001
            item["error"] = f"{type(exc).__name__}: {exc}"
        samples.append(item)

    # 历史日里「当时存在、现在退市」的样本是否在宇宙中
    pit_hits = []
    if len(delisted):
        for day, info in pit.items():
            if info["n"] > 0 and len(info["codes"]):
                hit = [c for c in delisted["code"].tolist()[:200] if c in info["codes"]]
                pit_hits.append({"day": day, "delisted_in_pit": len(hit), "sample": hit[:5]})

    return {
        "basic_rows": len(basic),
        "status_counts": by_status,
        "delisted_n": len(delisted),
        "delisted_sample": delisted.head(5).to_dict("records") if len(delisted) else [],
        "pit": {k: {"n": v["n"], "error": v.get("error")} for k, v in pit.items()},
        "pit_hits": pit_hits,
        "samples": samples,
    }


# --------------------------------------------------------------------------
# 3. ST 字段历史覆盖
# --------------------------------------------------------------------------
def probe_st() -> dict:
    import baostock as bs

    out = []
    for code in ST_CODES:
        item = {"code": code}
        try:
            rs = bs.query_history_k_data_plus(
                code, "date,close,volume,tradestatus,isST", start_date="2005-01-01", end_date="", frequency="d"
            )
            f = to_frame(rs)
            item["rows"] = len(f)
            if len(f):
                f["isST"] = f["isST"].astype(str).str.strip()
                item["first"] = f["date"].iloc[0]
                item["last"] = f["date"].iloc[-1]
                item["st_days"] = int((f["isST"] == "1").sum())
                item["st_ratio"] = round(item["st_days"] / len(f), 3)
                st_dates = f.loc[f["isST"] == "1", "date"]
                if len(st_dates):
                    item["st_first"] = st_dates.iloc[0]
                    item["st_last"] = st_dates.iloc[-1]
                # ST 区间是否连续块
                item["st_blocks"] = int((f["isST"].ne(f["isST"].shift()) & (f["isST"] == "1")).sum())
                item["suspend_days"] = int((f["tradestatus"].astype(str) == "0").sum())
        except Exception as exc:  # noqa: BLE001
            item["error"] = f"{type(exc).__name__}: {exc}"
        out.append(item)
    return {"samples": out}


# --------------------------------------------------------------------------
# 4. 估值字段
# --------------------------------------------------------------------------
def probe_valuation(codes: list[str] | None = None) -> dict:
    import baostock as bs

    codes = codes or ["sh.600519", "sz.000001", "sz.300750", "sh.601318", "sz.002415"]
    out = []
    for code in codes:
        item = {"code": code}
        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,close,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                start_date="2016-01-04",
                end_date="2024-12-31",
                frequency="d",
            )
            f = to_frame(rs)
            item["rows"] = len(f)
            for col in ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"):
                s = f[col].astype(str).str.strip()
                empty = s.isin(["", "None", "nan"])
                num = pd.to_numeric(s, errors="coerce")
                item[col] = {
                    "empty_pct": round(float(empty.mean()) * 100, 2) if len(s) else None,
                    "neg_pct": round(float((num < 0).mean()) * 100, 2) if len(s) else None,
                    "p01": round(float(num.quantile(0.01)), 2) if num.notna().any() else None,
                    "p99": round(float(num.quantile(0.99)), 2) if num.notna().any() else None,
                    "max": round(float(num.max()), 2) if num.notna().any() else None,
                }
        except Exception as exc:  # noqa: BLE001
            item["error"] = f"{type(exc).__name__}: {exc}"
        out.append(item)
    return {"samples": out}


# --------------------------------------------------------------------------
# 5. 财务：pubDate / 修订
# --------------------------------------------------------------------------
def probe_financials(codes: list[str] | None = None, years: tuple[int, ...] = (2022, 2023, 2024)) -> dict:
    import baostock as bs

    codes = codes or [
        "sh.600519", "sz.000001", "sz.300750", "sh.601318", "sz.002415",
        "sh.600036", "sz.000858", "sh.601166", "sz.002594", "sh.600276",
        "sz.000333", "sh.600030", "sz.002304", "sh.601012", "sh.600887",
        "sz.000651", "sh.601899", "sz.300059", "sh.600309", "sz.002714",
    ]
    apis = {
        "profit": bs.query_profit_data,
        "operation": bs.query_operation_data,
        "growth": bs.query_growth_data,
        "balance": bs.query_balance_data,
        "cash_flow": bs.query_cash_flow_data,
    }
    summary = {}
    pub_lag = []
    dup_stat = []
    field_sets = {}
    for name, fn in apis.items():
        rows_total, pub_ok, missing_pub = 0, 0, 0
        fields_seen = None
        for code in codes:
            for year in years:
                try:
                    rs = fn(code=code, year=year, quarter=4)
                    if rs.error_code != "0":
                        continue
                    f = to_frame(rs)
                    if not len(f):
                        continue
                    fields_seen = fields_seen or list(f.columns)
                    rows_total += len(f)
                    if "pubDate" in f.columns:
                        pub = f["pubDate"].astype(str).str.strip()
                        pub_ok += int((pub != "").sum())
                        missing_pub += int((pub == "").sum())
                        if "statDate" in f.columns:
                            for _, r in f.iterrows():
                                try:
                                    p = datetime.fromisoformat(str(r["pubDate"]))
                                    s = datetime.fromisoformat(str(r["statDate"]))
                                    pub_lag.append((p - s).days)
                                except Exception:  # noqa: BLE001
                                    pass
                                if str(r.get("statDate", "")).strip() and str(r.get("pubDate", "")).strip():
                                    pass
                        # 同一 statDate 是否出现多行（修订版本）
                        if "statDate" in f.columns:
                            vc = f["statDate"].astype(str).str.strip().value_counts()
                            if (vc > 1).any():
                                dup_stat.append({"api": name, "code": code, "year": year,
                                                 "statDate": vc[vc > 1].index.tolist()})
                except Exception:  # noqa: BLE001
                    continue
        summary[name] = {
            "rows": rows_total,
            "fields": fields_seen,
            "pubDate_filled": pub_ok,
            "pubDate_missing": missing_pub,
        }
    lag = pd.Series(pub_lag, dtype="float64")
    return {
        "apis": summary,
        "pub_lag_days": {
            "n": int(len(lag)),
            "min": int(lag.min()) if len(lag) else None,
            "p50": round(float(lag.median()), 1) if len(lag) else None,
            "p95": round(float(lag.quantile(0.95)), 1) if len(lag) else None,
            "max": int(lag.max()) if len(lag) else None,
        } if len(lag) else {"n": 0},
        "dup_statDate_samples": dup_stat[:5],
    }


# --------------------------------------------------------------------------
# 6. 行业
# --------------------------------------------------------------------------
def probe_industry() -> dict:
    import baostock as bs

    rs = bs.query_stock_industry()
    f = to_frame(rs)
    out = {"rows": len(f), "fields": list(f.columns)}
    if "updateDate" in f.columns:
        u = f["updateDate"].astype(str).str.strip()
        out["updateDate_unique"] = u.nunique()
        out["updateDate_minmax"] = [u.min(), u.max()]
        out["updateDate_top"] = u.value_counts().head(5).to_dict()
    if "industry" in f.columns:
        out["industry_top"] = f["industry"].value_counts().head(10).to_dict()
    # 历史日期查询是否返回不同快照
    hist = {}
    for day in ("2016-01-05", "2020-01-06", "2024-01-02"):
        try:
            rs2 = bs.query_stock_industry(date=day)
            f2 = to_frame(rs2)
            hist[day] = {"rows": len(f2), "same_as_current": bool(len(f2) == len(f) and set(f2["code"]) == set(f["code"]))}
            if len(f2) and "industry" in f2.columns and len(f):
                merged = f[["code", "industry"]].merge(f2[["code", "industry"]], on="code", suffixes=("", "_hist"))
                diff = (merged["industry"] != merged["industry_hist"]).sum()
                hist[day]["industry_diff"] = int(diff)
        except Exception as exc:  # noqa: BLE001
            hist[day] = {"error": f"{type(exc).__name__}: {exc}"}
    out["history"] = hist
    return out


# --------------------------------------------------------------------------
# 7. 股本（季频是否含 totalShare / liqaShare）
# --------------------------------------------------------------------------
def probe_shares() -> dict:
    import baostock as bs

    out = {}
    for name, fields in (
        ("balance", "code,pubDate,statDate,currentRatio,quickRatio,cashRatio,YOYLiability,liabilityToAsset,assetToEquity"),
        ("growth", "code,pubDate,statDate,YOYEquity,YOYAsset,YOYNI,YOYEPSBasic,YOYPNI"),
        ("operation", "code,pubDate,statDate,NRTurnRatio,INVTurnRatio,CATurnRatio,AssetTurnRatio"),
    ):
        try:
            fn = {"balance": bs.query_balance_data, "growth": bs.query_growth_data,
                  "operation": bs.query_operation_data}[name]
            f = to_frame(fn(code="sh.600519", year=2024, quarter=4))
            out[f"{name}_fields"] = list(f.columns)
        except Exception as exc:  # noqa: BLE001
            out[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    # 日线里是否带股本字段
    for fields in ("date,close,totalShare,liqaShare", "date,close,turn"):
        try:
            rs = bs.query_history_k_data_plus("sh.600519", fields, start_date="2024-12-02",
                                              end_date="2024-12-31", frequency="d")
            if rs.error_code != "0":
                out[f"hist[{fields}]"] = f"error {rs.error_code}: {rs.error_msg}"
            else:
                f = to_frame(rs)
                out[f"hist[{fields}]"] = {"rows": len(f), "fields": list(f.columns)}
        except Exception as exc:  # noqa: BLE001
            out[f"hist[{fields}]"] = f"{type(exc).__name__}: {exc}"
    # 复权因子接口（与湖 adj_factor 对照的候选）
    try:
        rs = bs.query_adjust_factor(code="sh.600519", start_date="2016-01-01", end_date="2024-12-31")
        f = to_frame(rs)
        out["adjust_factor"] = {"rows": len(f), "fields": list(f.columns), "head": f.head(3).to_dict("records")}
    except Exception as exc:  # noqa: BLE001
        out["adjust_factor_error"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------
# 8. 指数成分
# --------------------------------------------------------------------------
def probe_index() -> dict:
    import baostock as bs

    out = {}
    for name, fn in (("hs300", bs.query_hs300_stocks), ("zz500", bs.query_zz500_stocks), ("sz50", bs.query_sz50_stocks)):
        snaps = {}
        prev = None
        for day in ("2016-01-05", "2018-01-02", "2020-01-06", "2022-01-04", "2024-01-02"):
            try:
                rs = fn(date=day)
                f = to_frame(rs)
                codes = set(f["code"]) if "code" in f.columns else set()
                churn = None
                if prev is not None and codes:
                    churn = len(codes ^ prev)
                snaps[day] = {"n": len(codes), "churn_vs_prev": churn}
                prev = codes
            except Exception as exc:  # noqa: BLE001
                snaps[day] = {"error": f"{type(exc).__name__}: {exc}"}
        out[name] = snaps
    return out


# --------------------------------------------------------------------------
# 9. 北交所覆盖
# --------------------------------------------------------------------------
def probe_bj() -> dict:
    import baostock as bs

    rs = bs.query_stock_basic()
    f = to_frame(rs)
    codes = f["code"].astype(str) if "code" in f.columns else pd.Series(dtype=str)
    bj = codes[codes.str.startswith(("bj.", "sh.92", "sz.92"))]
    prefixes = codes.str.split(".").str[0].value_counts().to_dict() if len(codes) else {}
    detail = []
    for code in ("bj.920002", "bj.830799", "sh.920002"):
        item = {"code": code}
        try:
            rs2 = bs.query_history_k_data_plus(code, "date,close,volume", start_date="2023-01-01",
                                               end_date="", frequency="d")
            if rs2.error_code == "0":
                f2 = to_frame(rs2)
                item.update({"rows": len(f2), "first": f2["date"].iloc[0] if len(f2) else None,
                             "last": f2["date"].iloc[-1] if len(f2) else None})
            else:
                item["error"] = f"{rs2.error_code}: {rs2.error_msg}"
        except Exception as exc:  # noqa: BLE001
            item["error"] = f"{type(exc).__name__}: {exc}"
        detail.append(item)
    return {"prefix_counts": prefixes, "bj_codes": bj.head(10).tolist(), "probe": detail}


# --------------------------------------------------------------------------
# 10. 与现有湖交叉验证
# --------------------------------------------------------------------------
def probe_cross(codes: list[str] | None = None) -> dict:
    import baostock as bs
    from quant.lake import Lake

    codes = codes or ["sh.600519", "sz.000001", "sz.300750"]
    lake = Lake(root=Path(r"D:\quant-data"))
    out = []
    for code in codes:
        item = {"code": code}
        sym = code.split(".")[1]
        try:
            rs = bs.query_history_k_data_plus(
                code, "date,close,volume,amount,tradestatus,adjustflag",
                start_date="2016-01-04", end_date="2024-12-31", frequency="d", adjustflag="3",
            )
            b = to_frame(rs)
            b["date"] = b["date"].astype(str)
            raw = lake.read(sym, kind="raw")
            raw = raw.reset_index()
            raw["date"] = raw["date"].astype(str).str[:10]
            raw = raw[(raw["date"] >= "2016-01-04") & (raw["date"] <= "2024-12-31")].copy()
            m = raw.merge(b, on="date", suffixes=("_lake", "_bs"))
            item["overlap_days"] = len(m)
            item["lake_days"] = len(raw)
            item["bs_days"] = len(b)
            if len(m):
                close_diff = (m["close_lake"].astype(float) - m["close_bs"].astype(float)).abs()
                item["close_diff_pct"] = round(float((close_diff / m["close_bs"].astype(float)).mean()) * 100, 4)
                item["close_mismatch_1pct"] = int((close_diff / m["close_bs"].astype(float) > 0.01).sum())
                vol = (m["volume_lake"].astype(float) - m["volume_bs"].astype(float)).abs()
                item["volume_mismatch_pct"] = round(float((vol / m["volume_bs"].astype(float).replace(0, 1)).mean()) * 100, 3)
                item["suspended_lake"] = int((m["suspended_lake"].astype(int) == 1).sum()) if "suspended_lake" in m else None
                item["bs_tradestatus0"] = int((m["tradestatus"].astype(str) == "0").sum())
            # 前复权收益率对照
            rs2 = bs.query_history_k_data_plus(
                code, "date,close,pctChg", start_date="2016-01-04", end_date="2024-12-31",
                frequency="d", adjustflag="2",
            )
            bq = to_frame(rs2)
            bq["date"] = bq["date"].astype(str)
            proc = lake.read(sym, kind="processed")
            proc = proc.reset_index()
            proc["date"] = proc["date"].astype(str).str[:10]
            proc = proc[(proc["date"] >= "2016-01-04") & (proc["date"] <= "2024-12-31")].copy()
            mq = proc.merge(bq, on="date", suffixes=("_lake", "_bs"))
            if len(mq) > 10:
                r_lake = mq["close_lake"].astype(float).pct_change().dropna()
                r_bs = mq["close_bs"].astype(float).pct_change().dropna()
                item["qfq_ret_corr"] = round(float(r_lake.corr(r_bs)), 6)
                item["qfq_ret_maxdiff_pct"] = round(float((r_lake - r_bs).abs().max()) * 100, 4)
                item["qfq_ret_mae_pct"] = round(float((r_lake - r_bs).abs().mean()) * 100, 4)
        except Exception as exc:  # noqa: BLE001
            item["error"] = f"{type(exc).__name__}: {exc}"
        out.append(item)
    return {"samples": out}


PROBES = {
    1: ("连通性与限速", probe_connectivity),
    2: ("退市股与 PIT 宇宙", probe_delisted),
    3: ("ST 历史覆盖", probe_st),
    4: ("估值字段", probe_valuation),
    5: ("财务 pubDate / 修订", probe_financials),
    6: ("行业分类", probe_industry),
    7: ("股本口径", probe_shares),
    8: ("指数成分", probe_index),
    9: ("北交所", probe_bj),
    10: ("与湖交叉验证", probe_cross),
}


def render_md(results: dict, elapsed: dict) -> str:
    lines = [
        "# baostock 探针报告",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"- baostock 版本：{results.get('_version', '?')}",
        "- 口径：全部为实测，未写入 `D:\\quant-data`；脚本 `scripts/probe_baostock.py`。",
        "",
    ]
    for key in sorted((k for k in results if not str(k).startswith("_")), key=str):
        name, _ = PROBES[int(key)]
        body = results[key]
        lines += [f"## {key}. {name}", "", "```json", json.dumps(body, ensure_ascii=False, indent=2, default=str), "```", ""]
    lines += ["## 耗时", ""]
    lines += [f"- {PROBES[k][0]}：{v:.1f}s" for k, v in elapsed.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="baostock 探针（只读）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", default="", help="逗号分隔的探针编号，默认全部")
    args = ap.parse_args(argv)

    try:
        import baostock as bs
    except ImportError:
        print("未安装 baostock。安装：python -m pip install baostock")
        return 2

    bs.login()  # 单项重跑时也必须先登录（探针 1 自己会做 login/logout 稳定性测试）
    wanted = [int(x) for x in args.only.split(",") if x.strip()] or sorted(PROBES)
    results = {"_version": getattr(bs, "__version__", "?")}
    elapsed = {}
    for idx in wanted:
        name, fn = PROBES[idx]
        print(f"[{idx}] {name} ...", flush=True)
        t0 = time.perf_counter()
        try:
            results[idx] = fn()
        except Exception as exc:  # noqa: BLE001
            results[idx] = {"fatal": f"{type(exc).__name__}: {exc}"}
        elapsed[idx] = time.perf_counter() - t0
        print(f"    done in {elapsed[idx]:.1f}s", flush=True)
    try:
        bs.logout()
    except Exception:  # noqa: BLE001
        pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(render_md(results, elapsed), encoding="utf-8")
    print(f"写出：{out.with_suffix('.md')} / {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
