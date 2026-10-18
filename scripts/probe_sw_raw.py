from __future__ import annotations

import json
import time
from pathlib import Path

import requests

URL = "https://www.swsresearch.com/institute-sw/api/index_publish/details/component_stocks/"
HEAD = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Referer": "https://www.swsresearch.com/institute_sw/allIndex/releasedIndex"}

out = {}
for code in ("801010", "801080"):
    for attempt in range(3):
        t0 = time.time()
        try:
            r = requests.get(URL, params={"swindexcode": code, "page": "1", "page_size": "10000"}, headers=HEAD, timeout=20, verify=False)
            text = r.text[:200]
            data = r.json()
            results = (data.get("data") or {}).get("results") or []
            out[f"{code}_{attempt}"] = {
                "status": r.status_code,
                "seconds": round(time.time() - t0, 2),
                "keys": list(data.keys())[:6],
                "n_results": len(results),
                "first": results[0] if results else None,
                "head": text,
            }
        except Exception as exc:  # noqa: BLE001
            out[f"{code}_{attempt}"] = {"error": f"{type(exc).__name__}: {exc}"}
        time.sleep(1.5)

Path("results/probe-sw-raw.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2)[:2500])
