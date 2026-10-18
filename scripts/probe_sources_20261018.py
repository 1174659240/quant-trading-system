"""数据源/插件兼容性自检（2026-10-18）。

客户端更新后，把所有用到的外部接口逐个实测一遍，产出
``results/probe-sources-20261018.{json,md}``。判据只有一条：**能不能拿到本次研究需要的字段**，
不看文档、不看版本号。

分组：
  A 环境     Python / pandas / numpy / requests / akshare / baostock 版本
  B baostock universe / 日线 / 行业 PIT / 股本（liqaShare）
  C 东财数据中心 三表 + MAINFINADATA（每张取 1 页，看返回是否仍有目标列）
  D akshare 封装 东财批量薄表 / 申万 / 财报预约 / 指数成分 / 退市名单
  E 行情源       东财 push2his / 腾讯 / 新浪 / 新浪全市场列表
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "probe-sources-20261018"
RECORDS: list[dict] = []


def timeout_call(fn, seconds: float = 30.0, label: str = ""):
    """守护线程 + join：akshare 有几个接口没有超时参数，会直接挂死。"""
    box: dict = {}

    def runner():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
            box["trace"] = traceback.format_exc()

    thread = threading.Thread(target=runner, daemon=True, name=f"probe-{label}")
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"{label} 超过 {seconds:.0f}s 未返回")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def record(group: str, name: str, fn, *, seconds: float = 30.0, note: str = "") -> object:
    started = time.time()
    try:
        value = timeout_call(fn, seconds, name)
        item = {
            "group": group,
            "name": name,
            "ok": True,
            "seconds": round(time.time() - started, 2),
            "note": note,
        }
        if hasattr(value, "shape"):
            item["rows"], item["cols"] = int(value.shape[0]), int(value.shape[1])
            item["columns"] = [str(c) for c in list(value.columns)[:40]]
        elif isinstance(value, dict):
            item["keys"] = list(value)[:30]
            item["summary"] = {k: value[k] for k in list(value)[:12] if isinstance(value[k], (int, float, str))}
        RECORDS.append(item)
        print(f"[ok  {item['seconds']:6.2f}s] {group} {name} {item.get('rows', '')}", flush=True)
        return value
    except BaseException as exc:  # noqa: BLE001
        item = {
            "group": group,
            "name": name,
            "ok": False,
            "seconds": round(time.time() - started, 2),
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "note": note,
        }
        RECORDS.append(item)
        print(f"[FAIL       ] {group} {name}: {item['error']}", flush=True)
        return None


def section_a() -> None:
    import numpy as np
    import pandas as pd
    import requests

    versions = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "requests": requests.__version__,
    }
    try:
        import baostock as bs

        versions["baostock"] = getattr(bs, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        versions["baostock"] = f"import failed: {exc}"
    try:
        import akshare as ak

        versions["akshare"] = getattr(ak, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        versions["akshare"] = f"import failed: {exc}"
    RECORDS.append({"group": "A 环境", "name": "versions", "ok": True, "seconds": 0.0, "versions": versions})
    print(f"[ok        ] A 环境 {versions}", flush=True)


def section_b() -> None:
    import baostock as bs

    import quant.refdata as refdata

    def login():
        return refdata.ensure_bs(force=True)

    record("B baostock", "登录", login, seconds=40)

    def basic():
        rs = bs.query_stock_basic()
        frame = refdata.bs_frame(rs)
        return frame

    basic_frame = record("B baostock", "query_stock_basic", basic, seconds=90)
    if basic_frame is not None and len(basic_frame):
        alive = basic_frame[basic_frame["status"].astype(str) == "1"]
        RECORDS[-1]["summary"] = {
            "rows": len(basic_frame),
            "type1_alive": int((alive["type"].astype(str) == "1").sum()),
            "columns": ",".join(basic_frame.columns),
        }

    def kline():
        rs = bs.query_history_k_data_plus(
            "sh.600519",
            "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
            start_date="2024-01-01",
            end_date="2024-03-01",
            frequency="d",
            adjustflag="3",
        )
        return refdata.bs_frame(rs)

    kline_frame = record("B baostock", "日线 600519", kline, seconds=60)
    if kline_frame is not None and len(kline_frame):
        RECORDS[-1]["summary"] = {"last": str(kline_frame["date"].iloc[-1]), "close": str(kline_frame["close"].iloc[-1])}

    def industry():
        rs = bs.query_stock_industry(date="2024-12-31")
        return refdata.bs_frame(rs)

    industry_frame = record("B baostock", "行业 PIT 2024-12-31", industry, seconds=120)
    if industry_frame is not None and len(industry_frame):
        RECORDS[-1]["summary"] = {
            "rows": len(industry_frame),
            "columns": ",".join(industry_frame.columns),
            "update_dates": int(industry_frame["updateDate"].nunique()) if "updateDate" in industry_frame else -1,
        }

    def shares():
        rs = bs.query_profit_data(code="sh.600519", year=2024, quarter=4)
        return refdata.bs_frame(rs)

    shares_frame = record("B baostock", "query_profit_data（liqaShare/totalShare）", shares, seconds=60)
    if shares_frame is not None and len(shares_frame):
        RECORDS[-1]["summary"] = {"columns": ",".join(shares_frame.columns), "rows": len(shares_frame)}


def section_c() -> None:
    import quant.fundamentals as fund

    for table in ("profit", "balance", "cashflow", "indicator"):
        report = fund.REPORTS[table]
        wanted = ",".join(fund.COLUMNS[table])

        def fetch(report=report, wanted=wanted):
            frame, pages = fund.page(report, "(REPORT_DATE='2024-12-31')", wanted, 1)
            return frame, pages

        got = record("C 东财数据中心", f"{table}（{report}）1 页", fetch, seconds=60)
        if got is not None:
            frame, pages = got
            missing = [c for c in fund.COLUMNS[table] if c not in frame.columns]
            RECORDS[-1]["summary"] = {
                "page_rows": len(frame),
                "pages": pages,
                "missing_columns": ",".join(missing) if missing else "无",
            }


def section_d() -> None:
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        RECORDS.append({"group": "D akshare", "name": "import", "ok": False, "seconds": 0.0, "error": str(exc)})
        return

    thin = {
        "stock_lrb_em": lambda: ak.stock_lrb_em(date="20241231"),
        "stock_zcfz_em": lambda: ak.stock_zcfz_em(date="20241231"),
        "stock_xjll_em": lambda: ak.stock_xjll_em(date="20241231"),
    }
    for name, fn in thin.items():
        frame = record("D akshare", name, fn, seconds=90, note="薄表摘要口径，只用来确认接口没坏")
        if frame is not None:
            RECORDS[-1]["columns"] = [str(c) for c in list(frame.columns)]

    record("D akshare", "sw_index_first_info", lambda: ak.sw_index_first_info(), seconds=90)
    record("D akshare", "sw_index_second_info", lambda: ak.sw_index_second_info(), seconds=90)
    component = record(
        "D akshare", "index_component_sw(801010)", lambda: ak.index_component_sw(symbol="801010"), seconds=90
    )
    if component is not None:
        RECORDS[-1]["columns"] = [str(c) for c in list(component.columns)]
    record(
        "D akshare",
        "stock_report_disclosure(2024年报)",
        lambda: ak.stock_report_disclosure(period="2024年报", market="沪深京"),
        seconds=120,
    )
    cons = record(
        "D akshare", "index_stock_cons_csindex(000300)", lambda: ak.index_stock_cons_csindex(symbol="000300"), seconds=90
    )
    if cons is not None and len(cons):
        date_col = [c for c in cons.columns if "日期" in str(c)]
        RECORDS[-1]["summary"] = {
            "rows": len(cons),
            "date_col": date_col[0] if date_col else "无",
            "dates": int(cons[date_col[0]].nunique()) if date_col else -1,
        }
    record("D akshare", "stock_info_sh_delist", lambda: ak.stock_info_sh_delist(), seconds=40)
    record(
        "D akshare",
        "stock_info_sz_delist(终止上市公司)",
        lambda: ak.stock_info_sz_delist(symbol="终止上市公司"),
        seconds=40,
    )


def section_e() -> None:
    import importlib

    web = importlib.import_module("quant.data.web_source")

    calls = [
        ("东财 push2his 600519", lambda: web.fetch_eastmoney("600519.SH", "2024-01-01", "2024-01-31", adjust="none"), 25),
        ("腾讯 ifzq 600519", lambda: web.fetch_tencent("600519.SH", "2024-01-01", "2024-01-31"), 40),
        ("新浪 getKLineData 600519", lambda: web.fetch_sina("600519.SH"), 25),
    ]
    for name, fn, seconds in calls:
        before = len(RECORDS)
        frame = record("E 行情源", name, fn, seconds=seconds)
        if frame is not None and len(frame):
            RECORDS[before]["summary"] = {
                "rows": len(frame),
                "first": str(frame["date"].iloc[0]),
                "last": str(frame["date"].iloc[-1]),
            }
    symbols = record("E 行情源", "新浪全市场列表（截 200）", lambda: web.list_symbols(limit=200), seconds=90)
    if symbols is not None:
        RECORDS[-1]["summary"] = {"rows": len(symbols), "first": symbols[0] if symbols else ""}


def render(records: list[dict]) -> str:
    lines = [
        "# 数据源 / 插件兼容性自检（2026-10-18）",
        "",
        "判据：**本次研究需要的字段能不能真的拿到**，不看文档与版本号。",
        "每条都带耗时；超时统一用守护线程包装（akshare 部分接口没有 timeout 参数）。",
        "",
        "| 分组 | 接口 | 结果 | 耗时 | 行/列 | 说明 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in records:
        if item["name"] == "versions":
            continue
        if item.get("ok"):
            status = "可用"
            detail = item.get("note", "")
            if "summary" in item:
                detail = (detail + " " + json.dumps(item["summary"], ensure_ascii=False)).strip()
            elif "columns" in item:
                detail = (detail + " " + ",".join(item["columns"][:8])).strip()
            size = f"{item.get('rows', '-')} / {item.get('cols', '-')}"
        else:
            status = "**失败**"
            detail = item.get("error", "")
            size = "-"
        lines.append(
            f"| {item['group']} | `{item['name']}` | {status} | {item.get('seconds', 0):.2f}s | {size} | {detail[:200]} |"
        )
    versions = next((item for item in records if item["name"] == "versions"), None)
    if versions:
        lines += ["", "## 环境版本", ""]
        for key, value in versions["versions"].items():
            lines.append(f"- {key}: `{value}`")
    failed = [item for item in records if not item.get("ok") and item["name"] != "versions"]
    lines += ["", f"## 结论", "", f"- 共 {len(records) - 1} 项，失败 {len(failed)} 项。"]
    if failed:
        for item in failed:
            lines.append(f"- ❌ `{item['name']}`：{item.get('error', '')[:160]}")
    else:
        lines.append("- 全部接口可用，无需替换。")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for section in (section_a, section_b, section_c, section_d, section_e):
        try:
            section()
        except BaseException as exc:  # noqa: BLE001
            RECORDS.append(
                {
                    "group": section.__name__,
                    "name": "section",
                    "ok": False,
                    "seconds": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    json_path = OUT.with_suffix(".json")
    md_path = OUT.with_suffix(".md")
    json_path.write_text(json.dumps(RECORDS, ensure_ascii=False, indent=1), encoding="utf-8")
    md_path.write_text(render(RECORDS), encoding="utf-8")
    print(f"\n写出 {json_path} / {md_path}", flush=True)
    return 0 if all(item.get("ok") for item in RECORDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
