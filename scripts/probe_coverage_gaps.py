from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant import refdata

val = Path("D:/quant-data/valuation")
files = [p.stem for p in val.glob("??????.parquet")]
bj = [s for s in files if s.startswith(("43", "83", "87", "88", "92"))]
uni = refdata.load_universe()
bj_uni = uni[uni["board"] == "北交所"]
crawl = Path("D:/quant-data/meta/universe.csv")
crawl_codes = set(pd.read_csv(crawl, dtype=str)["symbol"].str.zfill(6)) if crawl.exists() else set()

out = {
    "valuation_symbols": len(files),
    "valuation_bj": len(bj),
    "valuation_bj_sample": bj[:5],
    "universe_bj": int(len(bj_uni)),
    "crawl_not_in_valuation_sample": sorted(crawl_codes - set(files))[:8],
    "crawl_not_in_valuation": len(crawl_codes - set(files)),
    "delisted_events": int(len(pd.read_csv("D:/quant-data/delisted/events.csv", dtype={"symbol": str}))),
    "delisted_with_bars": int((pd.read_csv("D:/quant-data/delisted/events.csv", dtype={"symbol": str})["bars"] > 0).sum()),
}
Path("results/probe-coverage-gaps.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
