"""QVIX 前瞻恐慌分量（任务三）：只作辅助确认，不单独触发。

- 数据：``ak.index_option_50etf_qvix()``（50ETF 期权隐含波动率指数，2015-02-09 起）。
- 2015-02-09 之前为 NaN，视为「不提供信息」，不影响其它条件。
- 判据（事前，见 ``docs/极端风险清仓预注册.md`` §5）：
  1) 与已实现波动分位 ``vol_pct`` 的秩相关；
  2) 把「QVIX 分位 ≥ 0.95 且 5 日急升 ≥ 20%」加进触发条件，看是否 (a) 让 2015 更早触发、
     (b) 破坏 2018/2022/2024 的精确性；
  3) 秩相关 > 0.85 且对触发时点无改善 → 判定无增量、不纳入。

用法：
    python -m scripts.risk_qvix
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import risk_system as rs

OUT = Path("results/risk-system")
QVIX_FILE = OUT / "qvix.csv"

WINDOWS = {
    "history": {"2008": ("2008-01-01", "2008-12-31"),
                "2015": ("2015-06-01", "2015-09-30")},
    "main": {"2018": ("2018-01-01", "2018-12-31"),
             "2022": ("2022-01-01", "2022-12-31"),
             "2024": ("2024-01-01", "2024-02-29")},
}


def fetch_qvix() -> pd.DataFrame:
    if QVIX_FILE.is_file():
        frame = pd.read_csv(QVIX_FILE, parse_dates=["date"])
        return frame.set_index("date").sort_index()
    import akshare as ak

    raw = ak.index_option_50etf_qvix()
    raw["date"] = pd.to_datetime(raw["date"])
    frame = raw.set_index("date").sort_index()[["open", "high", "low", "close"]]
    OUT.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(QVIX_FILE, index=False, encoding="utf-8-sig")
    return frame


def main() -> int:
    frames = {}
    for sample in ("history", "main"):
        path = OUT / f"market-frame-{sample}.csv"
        if not path.is_file():
            print(f"[skip] 缺少 {path}")
            continue
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frames[sample] = frame

    qvix = fetch_qvix()
    close = qvix["close"]
    qvix_pct = close.rolling(250, min_periods=60).rank(pct=True)
    qvix_rise = close / close.shift(5) - 1.0

    rows, payload = [], {}
    for sample, frame in frames.items():
        merged = frame.copy()
        merged["qvix"] = close.reindex(merged.index).ffill()
        merged["qvix_pct"] = qvix_pct.reindex(merged.index).ffill()
        merged["qvix_rise"] = qvix_rise.reindex(merged.index).ffill()
        base_trig = rs.extreme_trigger(merged)
        # 辅助确认：QVIX 分位 ≥0.95 且 5 日急升 ≥20%（NaN = 不提供信息，视为通过）
        have = merged["qvix"].notna()
        cond = ((merged["qvix_pct"] >= 0.95) & (merged["qvix_rise"] >= 0.20)).fillna(False)
        qvix_gate = pd.Series(np.where(have, cond, True), index=merged.index)   # 无数据=不提供信息
        need = (
            (merged["index"] < merged["ma200"])
            & (merged["dd250"] <= -rs.EXT_DD)
            & (merged["breadth"] <= rs.EXT_BREADTH)
            & ((merged["r20"] <= rs.EXT_R20) | (merged["r60"] <= rs.EXT_R60))
        ).fillna(False)
        variant = (need & qvix_gate).rolling(rs.EXT_CONFIRM, min_periods=rs.EXT_CONFIRM).sum().ge(rs.EXT_CONFIRM)

        for label, (start, end) in WINDOWS[sample].items():
            part = merged.loc[start:end]
            keep = part["qvix"].notna()
            if keep.sum() > 10:
                a = part.loc[keep, "qvix_pct"].rank()
                b = part.loc[keep, "vol_pct"].rank()
                rho = float(a.corr(b))          # 秩上的 Pearson（避免 scipy 依赖）
            else:
                rho = float("nan")
            rows.append({
                "sample": sample, "window": label, "days": int(len(part)),
                "qvix_available": int(keep.sum()),
                "qvix_max": round(float(part["qvix"].max()), 2) if keep.any() else None,
                "qvix_pct_max": round(float(part["qvix_pct"].max()), 3) if keep.any() else None,
                "qvix_rise_max": round(float(part["qvix_rise"].max()), 3) if keep.any() else None,
                "rho_with_vol_pct": round(float(rho), 3) if np.isfinite(rho) else None,
                "trigger_base": int(base_trig.loc[start:end].sum()),
                "trigger_with_qvix": int(variant.loc[start:end].sum()),
            })
            payload.setdefault(label, {})["base_first"] = str(
                base_trig.loc[start:end][base_trig.loc[start:end]].index[:1].to_list())
            payload[label]["variant_first"] = str(
                variant.loc[start:end][variant.loc[start:end]].index[:1].to_list())

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "qvix-increment.csv", index=False, encoding="utf-8-sig")
    overall = {
        "corr_all": round(float(qvix_pct.corr(pd.Series(dtype=float))), 3) if False else None,
        "qvix_start": str(close.index[0].date()),
        "qvix_end": str(close.index[-1].date()),
        "rows": int(len(close)),
    }
    (OUT / "qvix-increment.json").write_text(
        json.dumps({"overall": overall, "table": rows, "first_days": payload},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = ["# QVIX 前瞻恐慌分量：增量检验", "",
             f"数据：50ETF QVIX，{overall['qvix_start']} ~ {overall['qvix_end']}，{overall['rows']} 行。",
             "用法：只作辅助确认（QVIX 分位 ≥0.95 且 5 日急升 ≥20%），不单独触发；2015-02-09 前 NaN。", "",
             table.to_markdown(index=False), ""]
    for label, item in payload.items():
        if item["base_first"] != "[]" or item["variant_first"] != "[]":
            lines.append(f"- **{label}**：基线首次触发 {item['base_first']}；加 QVIX 后 {item['variant_first']}")
    (OUT / "qvix-increment.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
