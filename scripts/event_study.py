"""事件研究（CAR）：以 PIT 可见日为 t0，算相对同池等权的异常收益。

输出 ``results/event-study/``：
- ``car.csv``：事件 × 分组 × 窗口的累计异常收益（均值/中位/t 值/样本量）；
- ``car-curve.csv``：t0−10 ~ t0+60 的累计异常收益曲线；
- ``distribution.csv``：事件按年/月分布与扎堆度；
- ``meta.json``：口径与参数。

全部只读 ``<= 2024-12-31``（holdout 锁）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant import events
from quant.factors.panel import build_panel
from scripts.redlow_study import HOLDOUT, LAKE, load_bars

WINDOWS = {"pre_10": (-10, -1), "pre_5": (-5, -1), "post_1": (0, 1), "post_5": (0, 5),
           "post_10": (0, 10), "post_20": (0, 20), "post_60": (0, 60),
           # 可交易口径：从 t0 收盘（=可见）之后起算，t+1 收盘 ~ t+k 收盘
           "tradable_1": (1, 1), "tradable_5": (1, 5), "tradable_20": (1, 20), "tradable_60": (1, 60)}
CURVE = list(range(-10, 61))


def groups() -> dict[str, pd.DataFrame]:
    """事件 → 带 ``group`` 列的表。"""
    out: dict[str, pd.DataFrame] = {}

    lhb = events.load("lhb")
    out["龙虎榜"] = lhb.assign(group=lhb["direction"])

    inst = events.load("lhb_inst")
    out["龙虎榜机构"] = inst.assign(group=inst["direction"])

    repurchase = events.load("repurchase").copy()
    amount = repurchase["plan_amount_high"].fillna(repurchase["plan_amount_low"])
    rank = amount.rank(pct=True, method="average")
    repurchase["group"] = np.where(rank >= 2 / 3, "大额回购", np.where(rank <= 1 / 3, "小额回购", "中额回购"))
    out["回购"] = repurchase

    hold = events.load("hold")
    out["增减持"] = hold.assign(group=hold["direction"])

    st = events.load("st")
    out["ST翻转"] = st.assign(group=st["direction"])

    fhps = events.load("fhps")
    fhps = fhps.assign(group=np.where(fhps["is_high_transfer"] > 0, "高送转", "现金分红预案"))
    out["送转分红"] = fhps
    return out


def car_table(excess: pd.DataFrame, events_frame: pd.DataFrame, label: str) -> tuple[list[dict], list[dict]]:
    index = excess.index
    dates = pd.DatetimeIndex(events_frame["available_date"])
    rows = pd.Index(excess.columns).get_indexer(events_frame["symbol"].astype(str).str.zfill(6))
    position = index.searchsorted(dates.values, side="left")
    values = excess.to_numpy(dtype="float64")
    n_days = len(index)

    car_rows, curve_rows = [], []
    for group, part in events_frame.groupby("group"):
        picked_rows, picked_cols = [], []
        for pos, col in zip(position[part.index], rows[part.index]):
            if col < 0 or pos < 0 or pos >= n_days:
                continue
            if pos - 10 < 0 or pos + 60 >= n_days:
                continue
            picked_rows.append(pos)
            picked_cols.append(col)
        if len(picked_rows) < 20:
            continue
        # 用相对位置取窗口，避免把日历天当成交易日
        offsets = np.arange(-10, 61)
        base = np.array(picked_rows)
        sample = values[base[:, None] + offsets[None, :], np.asarray(picked_cols)[:, None]]
        sample = np.nan_to_num(sample, nan=0.0)
        curve = np.nanmean(np.cumsum(sample, axis=1), axis=0)
        for offset, value in zip(offsets, curve):
            curve_rows.append({"event": label, "group": group, "offset": int(offset), "car": float(value)})
        for name, (begin, end) in WINDOWS.items():
            lo, hi = begin + 10, end + 10
            window = sample[:, lo:hi + 1].sum(axis=1)
            valid = window[np.isfinite(window)]
            if len(valid) < 20:
                continue
            std = valid.std(ddof=1)
            car_rows.append({
                "event": label,
                "group": group,
                "window": name,
                "n": int(len(valid)),
                "mean_car": round(float(valid.mean()), 6),
                "median_car": round(float(np.median(valid)), 6),
                "t_stat": round(float(valid.mean() / std * np.sqrt(len(valid))), 3) if std > 0 else None,
                "positive_share": round(float((valid > 0).mean()), 4),
            })
    return car_rows, curve_rows


def distribution(frame: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    part = frame.dropna(subset=["event_date"]).copy()
    part["year"] = pd.DatetimeIndex(part["event_date"]).year
    part["month"] = pd.DatetimeIndex(part["event_date"]).to_period("M").astype(str)
    for year, group in part.groupby("year"):
        rows.append({
            "event": label, "bucket": "year", "key": str(year), "events": int(len(group)),
            "symbols": int(group["symbol"].nunique()),
        })
    month_counts = part["month"].value_counts()
    if len(month_counts):
        rows.append({
            "event": label, "bucket": "concentration", "key": "max_month_share",
            "events": int(len(part)), "symbols": int(part["symbol"].nunique()),
            "value": round(float(month_counts.iloc[0] / len(part)), 4),
            "detail": str(month_counts.index[0]),
        })
    # 同一天最多多少只（事件扎堆）
    day_counts = part.groupby("event_date")["symbol"].nunique()
    if len(day_counts):
        rows.append({
            "event": label, "bucket": "concentration", "key": "max_same_day_symbols",
            "events": int(day_counts.max()),
            "symbols": int(part["symbol"].nunique()),
            "detail": str(day_counts.idxmax()),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="事件研究 CAR")
    parser.add_argument("--out", default="results/event-study")
    parser.add_argument("--smoke", type=int, default=None)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bars, stats = load_bars(smoke=args.smoke)
    panel = build_panel(bars)
    print(f"面板 {len(panel.dates)} 天 × {len(panel.symbols)} 只 ({stats})", flush=True)

    returns = panel.returns
    pool = returns.mean(axis=1)
    excess = returns.sub(pool, axis=0).clip(-0.25, 0.25)

    car_rows, curve_rows, dist_rows = [], [], []
    for label, frame in groups().items():
        frame = frame[frame["available_date"] <= pd.Timestamp("2024-12-31")].reset_index(drop=True)
        rows, curve = car_table(excess, frame, label)
        car_rows += rows
        curve_rows += curve
        dist_rows += distribution(frame, label)
        print(f"  {label}: {len(frame)} 事件 → {len(rows)} 行 CAR", flush=True)

    pd.DataFrame(car_rows).to_csv(out / "car.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve_rows).to_csv(out / "car-curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dist_rows).to_csv(out / "distribution.csv", index=False, encoding="utf-8-sig")
    meta = {
        "panel_days": int(len(panel.dates)),
        "panel_symbols": int(len(panel.symbols)),
        "start": str(panel.dates[0].date()),
        "end": str(panel.dates[-1].date()),
        "excess": "个股日收益 - 同日全池等权收益（截尾 ±25%）",
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "seconds": round(time.time() - t0, 1),
        "holdout": "只读 <= 2024-12-31",
        "bars_stats": stats,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 {meta['seconds']}s → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
