"""前沿因子评估：月度 IC / 分层 / 与全部存量因子的正交性。

口径：2017-01-03 ~ 2024-12-31（holdout 锁死）、月末截面、前瞻 20 日收益、剔 ST；
所有相关系数都是「逐月横截面秩相关」的时间序列均值。

产物 ``results/frontier/``：``ic.csv``、``yearly-ic.csv``、``quantiles.csv``、
``corr.csv``（长表）、``same-source.csv``（|ρ|>=0.7 的对）、``meta.json``。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import quant.factors.events as event_factors
import quant.factors.fundamental  # noqa: F401  导入即注册
import quant.factors.frontier as frontier
import quant.factors.style  # noqa: F401  导入即注册
from quant.factors import library
from quant.factors.evaluate import ic_summary, rank_ic
from quant.factors.panel import build_panel
from scripts.factor_market import FIELDS
from scripts.factor_monthly_ic import month_end_dates
from scripts.redlow_study import END, START, board_exclude, build_block, load_bars

FRONTIER = [
    "overnight_ret_20",
    "intraday_ret_20",
    "high_52w",
    "residual_momentum",
    "downside_beta",
    "ivol_250",
    "fip_momentum",
]


def _ranks(frame: pd.DataFrame, dates: pd.Index) -> dict[pd.Timestamp, pd.Series]:
    out: dict[pd.Timestamp, pd.Series] = {}
    for day in dates:
        row = frame.loc[day].dropna() if day in frame.index else pd.Series(dtype=float)
        if len(row) >= 30:
            out[day] = row.rank(pct=True)
    return out


def _pair_corr(a: dict, b: dict) -> float:
    vals = []
    for day in a.keys() & b.keys():
        x, y = a[day].align(b[day], join="inner")
        if len(x) >= 30:
            vals.append(float(np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="前沿因子 IC / 分层 / 正交性")
    parser.add_argument("--out", default="results/frontier")
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    bars, _ = load_bars()
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    panel = build_panel(bars, fields=FIELDS).slice(pd.Timestamp(START), pd.Timestamp(END))
    block = build_block(panel).reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)
    dates = month_end_dates(panel.dates)
    forward = panel.forward_returns(args.horizon)
    fwd_rank = _ranks(forward, dates)
    print(f"panel {panel.close.shape} 月末 {len(dates)} 个", flush=True)

    frontier.register()
    event_factors.register()
    frames: dict[str, pd.DataFrame] = {}
    ic_rows, yearly_rows, q_rows = [], [], []
    frontier_ranks: dict[str, dict] = {}

    for name in FRONTIER:
        frame = library.compute(name, panel)
        frames[name] = frame
        ranks = _ranks(frame, dates)
        frontier_ranks[name] = ranks
        ic = rank_ic(frame, forward, min_count=30).reindex(dates).dropna()
        summary = ic_summary(ic, periods_per_year=12)
        year = pd.Series(ic.index.year, index=ic.index)
        yearly = ic.groupby(year).mean()
        ic_rows.append(
            {
                "factor": name,
                "months": summary.get("n", 0),
                "ic_mean": summary.get("ic_mean", np.nan),
                "ic_ir": summary.get("ic_ir", np.nan),
                "t_stat": summary.get("t_stat", np.nan),
                "positive_rate": summary.get("positive_rate", np.nan),
                "positive_years": int((yearly > 0).sum()),
                "years": int(len(yearly)),
                "worst_year": float(yearly.min()) if len(yearly) else np.nan,
                "coverage": float(frame.notna().mean().mean()),
            }
        )
        for y, v in yearly.items():
            yearly_rows.append({"factor": name, "year": int(y), "ic": float(v)})
        # 分层：按横截面秩等分 5 组，看前瞻 20 日均值
        for q in range(1, 6):
            vals = []
            for day in dates:
                ser = frame.loc[day].dropna()
                ret = forward.loc[day]
                joined = pd.concat([ser.rename("f"), ret.rename("r")], axis=1).dropna()
                if len(joined) < 30:
                    continue
                order = joined["f"].sort_values().index
                parts = np.array_split(order, 5)
                vals.append(float(joined.loc[parts[q - 1], "r"].mean()))
            q_rows.append({"factor": name, "quantile": q, "mean_forward_20d": float(np.mean(vals)) if vals else np.nan})
        print(f"[{name}] IC {summary.get('ic_mean', float('nan')):+.4f} IR {summary.get('ic_ir', float('nan')):+.3f}", flush=True)

    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(out / "ic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly_rows).to_csv(out / "yearly-ic.csv", index=False, encoding="utf-8-sig")
    q_df = pd.DataFrame(q_rows)
    pivot = q_df.pivot(index="factor", columns="quantile", values="mean_forward_20d")
    monotonic = {}
    for name in pivot.index:
        row = pivot.loc[name].to_numpy(dtype=float)
        if np.isnan(row).any():
            monotonic[name] = np.nan
            continue
        sign = np.sign(row[4] - row[0])
        steppiness = int(np.sum(np.diff(row) * (1 if sign >= 0 else -1) > 0))
        monotonic[name] = steppiness / 4.0
    pivot["monotonic"] = pd.Series(monotonic)
    pivot.to_csv(out / "quantiles.csv", encoding="utf-8-sig")

    # 正交性：与全部存量因子的逐月横截面秩相关均值
    peers = [n for n in library.names() if n not in FRONTIER]
    corr_rows = []
    for peer in peers:
        try:
            frame = library.compute(peer, panel)
        except Exception as exc:  # noqa: BLE001 - 个别因子缺数据不应中断
            print(f"跳过 {peer}: {exc}", flush=True)
            continue
        ranks = _ranks(frame, dates)
        for name in FRONTIER:
            corr_rows.append({"factor": name, "peer": peer, "corr": _pair_corr(frontier_ranks[name], ranks)})
    corr_df = pd.DataFrame(corr_rows)
    corr_df["abs_corr"] = corr_df["corr"].abs()
    corr_df.to_csv(out / "corr.csv", index=False, encoding="utf-8-sig")
    same = corr_df[corr_df["abs_corr"] >= 0.7].sort_values("abs_corr", ascending=False)
    same.to_csv(out / "same-source.csv", index=False, encoding="utf-8-sig")
    top = corr_df.sort_values("abs_corr", ascending=False).groupby("factor").head(3)
    summary_corr = top.pivot_table(index="factor", columns="peer", values="corr", aggfunc="first")

    meta = {
        "start": START,
        "end": END,
        "horizon": args.horizon,
        "months": len(dates),
        "holdout": "锁定 2025-01-01 起，未读",
        "peers": len(peers),
        "same_source_pairs": int(len(same)),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(ic_df.to_string(index=False), flush=True)
    print("最强相关（每个前沿因子前 3）：", flush=True)
    print(summary_corr.to_string(), flush=True)
    print(f"完成 {time.time()-t0:.0f}s → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
