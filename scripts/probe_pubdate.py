"""财务公告日（pubDate）对照：baostock 的 pubDate 是否等于交易所真实公告日。

用户要求的第 5 项抽查：20 只股票 × 2023 年年报，拿东财公告接口里的公告日期做对照。
输出 results/probe-pubdate.{md,json}；只读，不写数据湖。
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "probe-pubdate"

CODES = [
    "600519", "000001", "300750", "601318", "002415",
    "600036", "000858", "601166", "002594", "600276",
    "000333", "600030", "002304", "601012", "600887",
    "000651", "601899", "300059", "600309", "002714",
]

UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}


def fetch_announcements(code: str, pages: int = 5) -> list[dict]:
    rows: list[dict] = []
    for page in range(1, pages + 1):
        query = urllib.parse.urlencode(
            {
                "sr": "-1",
                "page_size": "100",
                "page_index": str(page),
                "ann_type": "A",
                "client_source": "web",
                "stock_list": code,
                "f_node": "0",
                "s_node": "0",
            }
        )
        req = urllib.request.Request(
            "https://np-anotice-stock.eastmoney.com/api/security/ann?" + query, headers=UA
        )
        payload = json.loads(urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace"))
        items = payload.get("data", {}).get("list", [])
        if not items:
            break
        rows.extend(items)
        if len(items) < 100:
            break
        time.sleep(0.2)
    return rows


def main() -> int:
    import baostock as bs

    bs.login()
    details = []
    for code in CODES:
        prefixed = ("sh." if code.startswith(("6", "9")) else "sz.") + code
        item: dict = {"code": code}
        try:
            rs = bs.query_profit_data(code=prefixed, year=2023, quarter=4)
            fields = rs.fields
            rows = []
            while rs.next():
                rows.append(dict(zip(fields, rs.get_row_data())))
            if rows:
                item["bs_pubDate"] = rows[0].get("pubDate", "")
                item["bs_statDate"] = rows[0].get("statDate", "")
        except Exception as exc:  # noqa: BLE001
            item["bs_error"] = f"{type(exc).__name__}: {exc}"

        try:
            anns = fetch_announcements(code)
            annual = [
                a for a in anns
                if re.search(r"2023\s*年年度报告", a.get("title", ""))
                and "摘要" not in a.get("title", "")
                and "英文" not in a.get("title", "")
                and "摘录" not in a.get("title", "")
            ]
            corrections = [a for a in anns if "更正" in a.get("title", "") and "2023" in a.get("title", "")]
            if annual:
                annual.sort(key=lambda a: a.get("notice_date", ""))
                ann_date = annual[0]["notice_date"][:10]
                item["ann_date"] = ann_date
                item["ann_title"] = annual[0]["title"][:60]
                if item.get("bs_pubDate"):
                    d1 = datetime.fromisoformat(item["bs_pubDate"][:10])
                    d2 = datetime.fromisoformat(ann_date)
                    item["diff_days"] = (d2 - d1).days
            item["ann_total"] = len(anns)
            item["corrections_2023"] = len(corrections)
        except Exception as exc:  # noqa: BLE001
            item["ann_error"] = f"{type(exc).__name__}: {exc}"
        details.append(item)
        print(f"{code}: pubDate={item.get('bs_pubDate')} ann={item.get('ann_date')} diff={item.get('diff_days')}", flush=True)
    try:
        bs.logout()
    except Exception:  # noqa: BLE001
        pass

    diffs = [d["diff_days"] for d in details if isinstance(d.get("diff_days"), int)]
    summary = {
        "n": len(details),
        "matched": sum(1 for d in diffs if d == 0),
        "within_1d": sum(1 for d in diffs if abs(d) <= 1),
        "diffs": diffs,
        "missing_ann": sum(1 for d in details if "ann_date" not in d),
        "missing_pub": sum(1 for d in details if "bs_pubDate" not in d),
        "corrections_total": sum(d.get("corrections_2023", 0) for d in details),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# pubDate 对照交易所公告（2023 年报，20 只）",
        "",
        f"- 完全一致 {summary['matched']}/{len(diffs)}，±1 天内 {summary['within_1d']}/{len(diffs)}，"
        f"缺公告 {summary['missing_ann']}，缺 pubDate {summary['missing_pub']}，2023 年更正公告 {summary['corrections_total']} 条。",
        "",
        "| 代码 | baostock pubDate | 东财公告日 | 差(天) | 公告标题 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for d in details:
        lines.append(
            f"| {d['code']} | {d.get('bs_pubDate', '')} | {d.get('ann_date', '')} | {d.get('diff_days', '')} | "
            f"{(d.get('ann_title') or d.get('ann_error') or d.get('bs_error') or '')[:50]} |"
        )
    lines += ["", "说明：公告标题匹配「2023年年度报告」且排除摘要/英文/摘录；公告日取首条。", ""]
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print("写出", OUT.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
