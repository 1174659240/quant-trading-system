from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import akshare as ak

out: dict = {}

# 1) 巨潮资讯-股本变动（按股票，一次拿全历史）
for symbol in ("600519", "000001", "605369"):
    t0 = time.time()
    try:
        df = ak.stock_share_change_cninfo(symbol=symbol, start_date="20140101", end_date="20261018")
        out[f"cninfo_{symbol}"] = {
            "seconds": round(time.time() - t0, 2),
            "rows": int(len(df)),
            "columns": list(df.columns)[:20],
            "head": df.head(3).astype(str).to_dict("records") if len(df) else [],
        }
    except Exception as exc:  # noqa: BLE001
        out[f"cninfo_{symbol}"] = {"error": f"{type(exc).__name__}: {exc}"}

# 2) 东财数据中心：股本结构/股本变动 reportName 探测
import requests

URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
candidates = [
    "RPT_F10_EH_EQUITY",
    "RPT_F10_EH_EQUITYSTRUCTURE",
    "RPT_F10_EH_SHARECHANGE",
    "RPT_F10_EH_SHARESTRUCTURE",
    "RPT_F10_EH_EQUITY_CHANGE",
]
for name in candidates:
    try:
        r = requests.get(
            URL,
            params={"reportName": name, "columns": "ALL", "pageSize": "3", "pageNumber": "1",
                    "source": "HSF10", "client": "PC", "filter": '(SECURITY_CODE="600519")'},
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"},
            timeout=25,
        )
        payload = r.json()
        res = payload.get("result") or {}
        data = res.get("data") or []
        out[f"em_{name}"] = {
            "code": payload.get("code"),
            "message": payload.get("message"),
            "rows": len(data),
            "columns": list(data[0].keys())[:30] if data else [],
            "first": data[0] if data else None,
        }
    except Exception as exc:  # noqa: BLE001
        out[f"em_{name}"] = {"error": f"{type(exc).__name__}: {exc}"}

Path("results/probe-shares-pit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: (v if "error" in v else {kk: vv for kk, vv in v.items() if kk != "head"}) for k, v in out.items()}, ensure_ascii=False, indent=2)[:3000])
