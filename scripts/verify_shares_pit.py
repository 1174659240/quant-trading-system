from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant import refdata

f = refdata.load_shares()
checks = {}
for day in ("2018-12-31", "2020-12-31", "2024-06-28", "2026-10-17"):
    snap = refdata.shares_at(f, day).set_index("symbol")
    for sym in ("600519", "000001", "605369", "601577", "688981"):
        key = f"{sym}@{day}"
        checks[key] = None if sym not in snap.index else {
            "total": float(snap.loc[sym, "total_share"]),
            "float": None if pd.isna(snap.loc[sym, "float_share"]) else float(snap.loc[sym, "float_share"]),
            "since": str(snap.loc[sym, "change_date"]),
        }

# 历史变动条数分布
per_symbol = f[f.is_real_code == 1].groupby("symbol").size()
out = {
    "checks": checks,
    "symbols": int(f.symbol.nunique()),
    "real_symbols": int(f[f.is_real_code == 1].symbol.nunique()),
    "rows": int(len(f)),
    "per_symbol": per_symbol.describe().round(2).to_dict(),
    "future_dated": int((f.change_date > "2026-10-18").sum()),
    "rows_before_2015": int((f.change_date < "2015-01-01").sum()),
}
Path("results/probe-shares-verify.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
