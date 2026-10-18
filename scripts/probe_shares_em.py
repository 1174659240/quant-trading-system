from __future__ import annotations

import json
import time
from pathlib import Path

import requests

URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
HEAD = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
COLS = "SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,ORG_CODE,END_DATE,CHANGE_REASON,TOTAL_SHARES,LISTED_A_SHARES,LIMITED_SHARES,H_FREE_SHARE,B_FREE_SHARE,SECURITY_TYPE_CODE"
FLT = '(SECURITY_TYPE_CODE in ("058001001","058001002"))(END_DATE>=\'2014-01-01\')'

out: dict = {}
t0 = time.time()
resp = requests.get(URL, params={"reportName": "RPT_F10_EH_EQUITY", "columns": COLS, "pageSize": "500",
                                 "pageNumber": "1", "source": "HSF10", "client": "PC", "filter": FLT},
                    headers=HEAD, timeout=45)
payload = resp.json()
res = payload.get("result") or {}
data = res.get("data") or []
out["page1"] = {"code": payload.get("code"), "message": payload.get("message"),
                "rows": len(data), "pages": res.get("pages"), "count": res.get("count"),
                "seconds": round(time.time() - t0, 2)}
out["head"] = [{k: r.get(k) for k in ("SECURITY_CODE", "END_DATE", "CHANGE_REASON", "TOTAL_SHARES", "LISTED_A_SHARES")} for r in data[:3]]
if res.get("pages"):
    total = int(res["pages"])
    rows = len(data)
    t1 = time.time()
    for page in (2, total):
        chunk = requests.get(URL, params={"reportName": "RPT_F10_EH_EQUITY", "columns": COLS, "pageSize": "500",
                                          "pageNumber": str(page), "source": "HSF10", "client": "PC", "filter": FLT},
                             headers=HEAD, timeout=45).json()
        got = (chunk.get("result") or {}).get("data") or []
        rows += len(got)
        out[f"page{page}"] = {"rows": len(got), "first": got[0]["SECURITY_CODE"] if got else None}
    out["sample_seconds"] = round(time.time() - t1, 2)
    out["est_total_rows"] = rows
    out["est_full_seconds"] = round(out["sample_seconds"] / 2 * total, 1)

Path("results/probe-shares-em.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2)[:1800])
