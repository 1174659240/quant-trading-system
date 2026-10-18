"""step2 前置探测：东财 push2his 抖动、baostock 全市场股本、申万二级成分可得性。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

out: dict = {}

# 1) 东财 push2his 连试 3 次
import quant.data.web_source as web  # noqa: E402

tries = []
for i in range(3):
    t0 = time.time()
    try:
        frame = web.fetch_eastmoney("600519.SH", "2024-01-01", "2024-01-31", adjust="none")
        tries.append({"i": i, "ok": True, "rows": len(frame), "s": round(time.time() - t0, 2)})
    except Exception as exc:  # noqa: BLE001
        tries.append({"i": i, "ok": False, "error": f"{type(exc).__name__}: {exc}", "s": round(time.time() - t0, 2)})
    time.sleep(2)
out["eastmoney_push2his"] = tries

# 2) baostock query_profit_data 不传 code：是不是一次给全市场
import baostock as bs  # noqa: E402

lg = bs.login()
t0 = time.time()
rs = bs.query_profit_data(code="", year=2024, quarter=4)
rows = []
while rs.next():
    rows.append(rs.get_row_data())
frame = pd.DataFrame(rows, columns=rs.fields)
out["profit_all"] = {
    "error": f"{rs.error_code}: {rs.error_msg}",
    "rows": int(len(frame)),
    "cols": list(frame.columns),
    "seconds": round(time.time() - t0, 2),
}
if len(frame):
    sample = frame[frame["code"].isin(["sh.600519", "sz.000001", "sh.601398", "bj.920819"])]
    out["profit_all_sample"] = sample.to_dict(orient="records")
    out["profit_all_unique"] = int(frame["code"].nunique())

# 3) 申万二级有没有成分接口
import akshare as ak  # noqa: E402

second = ak.sw_index_second_info()
out["sw_second_rows"] = int(len(second))
code2 = str(second.iloc[0]["行业代码"])
try:
    part = ak.index_component_sw(symbol=code2)
    out["sw_second_component"] = {"code": code2, "rows": int(len(part)), "cols": list(part.columns)}
except Exception as exc:  # noqa: BLE001
    out["sw_second_component"] = {"code": code2, "error": f"{type(exc).__name__}: {exc}"}

# 4) 一级成分逐行业取，合计与 31 个行业的成份个数对不对得上
first = ak.sw_index_first_info()
total = 0
missing = []
for row in first.itertuples():
    code = str(row.行业代码)
    try:
        part = ak.index_component_sw(symbol=code)
        total += len(part)
        if len(part) == 0:
            missing.append(code)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{code}: {exc}")
out["sw_first"] = {
    "industries": int(len(first)),
    "sum_declared": int(first["成份个数"].astype(int).sum()),
    "sum_fetched": int(total),
    "missing": missing[:10],
}

Path(ROOT / "results" / "probe-step2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
