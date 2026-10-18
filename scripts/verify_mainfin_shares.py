from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import quant.fundamentals as F
import quant.mainfin as M
from quant import refdata

out: dict = {}

shares = refdata.load_shares()
uni = refdata.load_universe()
sym = set(shares["symbol"])
out["shares_rows"] = int(len(shares))
out["shares_symbols"] = int(shares["symbol"].nunique())
out["shares_periods"] = int(shares["report_date"].nunique())
out["in_universe"] = int(len(sym & set(uni["symbol"])))
out["not_in_universe"] = int(len(sym - set(uni["symbol"])))
out["exchange_counts"] = shares["exchange"].value_counts().to_dict()
bogus = sorted(s for s in sym - set(uni["symbol"]) if not s.isdigit() or s.startswith("A2"))
out["bogus_codes_sample"] = bogus[:10]
out["bogus_codes_count"] = len(bogus)

# 内部一致性：BPS × TOTAL_SHARE ≈ 归母净资产（资产负债表覆盖非金融业）
day = "2024-12-31"
mf = M.load(day)
bal = F.load("balance", day)
mf = mf.assign(symbol=mf["SECURITY_CODE"].astype(str).str.zfill(6))
bal = bal.assign(symbol=bal["SECURITY_CODE"].astype(str).str.zfill(6))
m = mf[["symbol", "BPS", "TOTAL_SHARE", "PARENTNETPROFIT", "ROEJQ"]].copy()
b = bal[["symbol", "TOTAL_PARENT_EQUITY", "TOTAL_ASSETS", "GOODWILL"]].copy()
j = m.merge(b, on="symbol", how="inner")
for col in ("BPS", "TOTAL_SHARE", "TOTAL_PARENT_EQUITY"):
    j[col] = pd.to_numeric(j[col], errors="coerce")
j = j.dropna(subset=["BPS", "TOTAL_SHARE", "TOTAL_PARENT_EQUITY"])
j = j[j["TOTAL_PARENT_EQUITY"] > 0]
j["implied"] = j["BPS"] * j["TOTAL_SHARE"]
j["rel"] = (j["implied"] - j["TOTAL_PARENT_EQUITY"]).abs() / j["TOTAL_PARENT_EQUITY"]
rel = j["rel"].dropna()
out["bps_share_vs_equity"] = {
    "n": int(rel.size),
    "median": round(float(rel.median()), 6),
    "p90": round(float(rel.quantile(0.9)), 6),
    "within_1pct": int((rel <= 0.01).sum()),
    "within_5pct": int((rel <= 0.05).sum()),
}

# 金融股：通用三表为空，mainfin 必须非空
fin = mf[mf["ORG_TYPE"].astype(str).isin(["银行", "保险", "证券"])]
out["financials"] = {
    "n": int(len(fin)),
    "roe_notnull": int(pd.to_numeric(fin["ROEJQ"], errors="coerce").notna().sum()),
    "eps_notnull": int(pd.to_numeric(fin["EPSJB"], errors="coerce").notna().sum()),
    "bps_notnull": int(pd.to_numeric(fin["BPS"], errors="coerce").notna().sum()),
    "revenue_notnull": int(pd.to_numeric(fin["TOTALOPERATEREVE"], errors="coerce").notna().sum()),
    "share_notnull": int(pd.to_numeric(fin["TOTAL_SHARE"], errors="coerce").notna().sum()),
    "in_balance": int(fin["symbol"].isin(set(bal["symbol"])).sum()),
    "in_cashflow": int(fin["symbol"].isin(set(F.load("cashflow", day)["SECURITY_CODE"].astype(str).str.zfill(6))).sum()),
}
out["period_rows"] = {p.stem: int(len(pd.read_parquet(p, columns=["SECURITY_CODE"]))) for p in sorted(M.MAINFIN_DIR.glob("????-??-??.parquet"))}

Path("results/probe-mainfin-shares.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "period_rows"}, ensure_ascii=False, indent=2))
