"""阶段 1 探针：确认 2025-01-01 ~ 2026-10-16 各慢数据层接口可用、字段一致、条数正常。

只读探针，不写数据湖。产物：``results/holdout-probe.json``、``results/holdout-probe.md``。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pandas as pd

from quant import dividends, fundamentals, valuation
from quant.factors.panel import build_panel

OUT_JSON = Path("results/holdout-probe.json")
OUT_MD = Path("results/holdout-probe.md")
START, END = "2025-01-01", "2026-10-16"
RECORDS: list[dict] = []


def timeout_call(fn, seconds: float = 60.0, label: str = ""):
    box: dict = {}

    def runner():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"{label or fn} 超过 {seconds}s")
    if "error" in box:
        raise RuntimeError(box["error"])
    return box.get("value")


def record(group: str, name: str, fn, seconds: float = 60.0, note: str = "") -> None:
    t0 = time.time()
    item = {"group": group, "name": name, "note": note}
    try:
        value = timeout_call(fn, seconds, name)
        item.update({"ok": True, "seconds": round(time.time() - t0, 1)})
        if isinstance(value, dict):
            item.update(value)
        else:
            item["rows"] = int(len(value))
            if isinstance(value, pd.DataFrame) and len(value):
                item["columns"] = list(map(str, value.columns))[:40]
                for column in ("date", "report_date", "ex_date", "available_date", "event_date"):
                    if column in value.columns:
                        dates = pd.to_datetime(value[column], errors="coerce").dropna()
                        if len(dates):
                            item[f"{column}_range"] = [str(dates.min().date()), str(dates.max().date())]
                            break
    except BaseException as exc:  # noqa: BLE001
        item.update({"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(exc).__name__}: {exc}"})
    RECORDS.append(item)
    flag = "✓" if item["ok"] else "✗"
    print(f"{flag} {group}/{name}: {item.get('rows', '')} {item.get('error', '')} ({item['seconds']}s)", flush=True)


def probe_fundamentals() -> None:
    for table in ("profit", "balance", "cashflow", "indicator"):
        for period in ("2025-03-31", "2025-12-31", "2026-09-30"):
            report = fundamentals.REPORTS[table]
            flt = f"(REPORT_DATE='{period}')" + fundamentals.TYPE_FILTER
            record(
                f"fundamentals/{table}",
                period,
                lambda report=report, flt=flt: fundamentals.page(report, flt, fundamentals.COLUMNS[table], 1)[0],
                seconds=90,
            )


def probe_dividends() -> None:
    for period in ("2025-06-30", "2025-12-31", "2026-06-30"):
        record("dividends", period, lambda period=period: dividends.fetch_period(period), seconds=90)


def probe_valuation() -> None:
    record("valuation", "600519.SH 2025+", lambda: valuation.fetch_symbol("600519", start=START, end=END), seconds=90)
    record("valuation", "000001.SZ 2025+", lambda: valuation.fetch_symbol("000001", start=START, end=END), seconds=90)
    record("valuation", "退市股 600087 2025+", lambda: valuation.fetch_symbol("600087", start=START, end=END), seconds=90)


def probe_events() -> None:
    from quant import events

    for name in ("lhb", "lhb_inst", "repurchase", "hold", "fhps", "st"):
        record(
            f"events/{name}",
            "holdout 行数",
            lambda name=name: events.load(name, allow_holdout=True, start=pd.Timestamp(START)),
            seconds=60,
        )


def render() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"start": START, "end": END, "probes": RECORDS}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [f"# 阶段 1 探针（{START} ~ {END}）", "", "| 分组 | 名称 | 结果 | 行数 | 区间 | 耗时 |", "|---|---|---|---|---|---|"]
    for item in RECORDS:
        span = item.get("date_range") or item.get("report_date_range") or item.get("ex_date_range") or item.get("available_date_range") or ""
        lines.append(
            f"| {item['group']} | {item['name']} | {'✓' if item['ok'] else '✗ ' + item.get('error', '')} | "
            f"{item.get('rows', '')} | {span} | {item['seconds']}s |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print(f"探针 {START} ~ {END}", flush=True)
    timeout_call(lambda: valuation.ensure_bs(), 60, "baostock login")
    probe_valuation()
    probe_fundamentals()
    probe_dividends()
    probe_events()
    render()
    bad = [item for item in RECORDS if not item["ok"]]
    print(f"完成：{len(RECORDS)} 项，失败 {len(bad)} 项 → {OUT_MD}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
