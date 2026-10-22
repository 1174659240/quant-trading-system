"""多危机泛化测试（任务三）：冻结规则在样本外危机段的表现。

只读已生成的市场帧（``results/crisis-generalization/*/market-frame-*.csv``）与早期补采数据
（``D:/quant-data/early/``），不重新拟合任何阈值。

用法：
    python -m scripts.crisis_generalization --market          # 主/历史样本市场帧
    python -m scripts.crisis_generalization --early 2001-2005  # 早期样本（补采完成后）
    python -m scripts.crisis_generalization --report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.factors.panel import build_panel
from scripts import growth_leftside as gl
from scripts.factor_market import FIELDS
from scripts.risk_system import add_speed, extreme_plan, extreme_trigger

ROOT = Path("results/crisis-generalization")
EARLY = Path("D:/quant-data/early")

WINDOWS = {
    "main": (
        ("2017-01-01", "2017-12-31", "2017 结构市"),
        ("2018-01-01", "2018-12-31", "2018 中跌（预期不触发）"),
        ("2020-02-01", "2020-04-30", "2020-03 疫情 V 型"),
        ("2022-01-01", "2022-12-31", "2022 中跌（预期不触发）"),
        ("2024-01-01", "2024-02-29", "2024-01/02 微盘流动性危机"),
    ),
    "history": (
        ("2008-01-01", "2008-12-31", "2008 金融危机"),
        ("2015-06-01", "2015-12-31", "2015 股灾"),
        ("2015-12-15", "2016-02-29", "2016-01 熔断"),
    ),
    "early-2001-2005": (
        ("2001-01-01", "2005-12-31", "2001-2005 慢熊（2245→998）"),
        ("2001-06-14", "2001-10-22", "2001 国有股减持急跌（附加）"),
        ("2004-01-01", "2004-12-31", "2004 宏观调控（附加）"),
    ),
    "early-1997-1999": (
        ("1997-05-01", "1997-12-31", "1997 亚洲金融危机"),
        ("1998-06-01", "1998-12-31", "1998 外需冲击（附加）"),
    ),
}


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame


def market_frame_for(sample: str) -> pd.DataFrame:
    path = ROOT / sample / f"market-frame-{sample}.csv"
    if path.is_file():
        return _load_frame(path)
    raise SystemExit(f"缺少市场帧 {path}")


def early_frame(sample: str) -> pd.DataFrame:
    """从补采的早期数据构造市场帧（只用价量，不依赖财务）。

    优先读 raw/*.csv.gz（采集中途也能出帧），没有再退回 parquet 分区。
    """
    raw_files = sorted((EARLY / "bars" / sample / "raw").glob("*.csv.gz"))
    parts = sorted((EARLY / "bars" / sample / "parquet").glob("year=*/data.parquet"))
    if raw_files:
        data = pd.concat([pd.read_csv(p, dtype={"symbol": str}) for p in raw_files], ignore_index=True)
    elif parts:
        data = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        raise SystemExit(f"缺少早期数据 {EARLY / 'bars' / sample}")
    data["date"] = pd.to_datetime(data["date"])
    bars = {symbol: part.sort_values("date").set_index("date")
            for symbol, part in data.groupby("symbol")}
    panel = build_panel(bars, fields=FIELDS)
    frame = add_speed(gl.market_frame(panel))
    frame["extreme_trigger"] = extreme_trigger(frame)
    out = ROOT / "early"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"market-frame-{sample}.csv", encoding="utf-8-sig")
    print(f"[early] {sample} 面板 {panel.close.shape}，触发 {int(frame['extreme_trigger'].sum())} 天", flush=True)
    return frame


def _daily_returns(frame: pd.DataFrame) -> pd.Series:
    if "market_ret" in frame.columns:
        return frame["market_ret"].astype(float)
    return frame["index"].pct_change()


def _path_stats(returns: pd.Series) -> tuple[float, float]:
    if returns.empty:
        return float("nan"), float("nan")
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(equity.iloc[-1] - 1.0), float(drawdown.min())


def _rebound_failures(events: pd.DataFrame, frame: pd.DataFrame, *, horizon: int = 20,
                      low_window: int = 15) -> tuple[list[dict], list[dict]]:
    """回补后被打回：某档回补后 ``horizon`` 个交易日内出现新的 ``low_window`` 日新低。

    另附一个更严的口径（回补后 60 日内指数再跌 ≥10%），用于交叉检查。
    """
    if events.empty:
        return [], []
    index = frame["index"]
    low = index.rolling(low_window, min_periods=low_window).min()
    tranches = events[events["kind"] == "tranche"].copy()
    new_low_hits: list[dict] = []
    deep_hits: list[dict] = []
    for row in tranches.itertuples():
        stamp = pd.Timestamp(row.date)
        if stamp not in index.index:
            continue
        after = index.loc[stamp:].iloc[: horizon + 1]
        low_after = low.loc[stamp:].iloc[: horizon + 1]
        hits = after[after <= low_after]
        if len(hits) > 1:  # 第 0 个是回补当日
            first = hits.index[1]
            new_low_hits.append({"episode": int(row.episode), "tranche": str(stamp.date()),
                                 "new_low": str(first.date()),
                                 "exposure_after": float(row.exposure_after)})
        deep = index.loc[stamp:].iloc[:61]
        if len(deep) >= 5:
            worst = float(deep.min() / float(index.loc[stamp]) - 1.0)
            if worst <= -0.10:
                deep_hits.append({"episode": int(row.episode), "tranche": str(stamp.date()),
                                  "worst_60d": round(worst, 4)})
    return new_low_hits, deep_hits


def window_table(sample: str, frame: pd.DataFrame, exposure: pd.Series,
                 events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    returns = _daily_returns(frame)
    strat_returns = returns * exposure.shift(1).fillna(1.0)
    for start, end, label in WINDOWS.get(sample, ()):
        segment = frame.loc[start:end]
        if segment.empty:
            continue
        triggers = segment["extreme_trigger"].astype(bool)
        trigger_days = int(triggers.sum())
        first = str(triggers[triggers].index[0].date()) if trigger_days else "—"
        ret_a, mdd_a = _path_stats(returns.loc[start:end])
        ret_e, mdd_e = _path_stats(strat_returns.loc[start:end])
        window_events = events[(events["date"] >= start) & (events["date"] <= end)] if len(events) else events
        new_low_hits, deep_hits = _rebound_failures(window_events, frame)
        rows.append({
            "sample": sample, "window": label, "start": start, "end": end,
            "trigger_days": trigger_days, "first_trigger": first,
            "ret_A_market": round(ret_a, 4), "ret_E_market": round(ret_e, 4),
            "mdd_A_market": round(mdd_a, 4), "mdd_E_market": round(mdd_e, 4),
            "exposure_min": round(float(exposure.loc[start:end].min()), 3),
            "exposure_avg": round(float(exposure.loc[start:end].mean()), 3),
            "events": len(window_events),
            "rebound_fail": len(new_low_hits),
            "fail_detail": json.dumps(new_low_hits, ensure_ascii=False) if new_low_hits else "",
            "rebound_deep": len(deep_hits),
            "deep_detail": json.dumps(deep_hits, ensure_ascii=False) if deep_hits else "",
        })
    return pd.DataFrame(rows)


def run_market() -> int:
    tables = []
    summary = {}
    for sample in ("main", "history"):
        try:
            frame = market_frame_for(sample)
        except SystemExit as exc:
            print(exc, flush=True)
            continue
        exposure, events = extreme_plan(frame, frame["extreme_trigger"].astype(bool))
        exposure.to_csv(ROOT / sample / f"exposure-E-{sample}.csv", encoding="utf-8-sig")
        table = window_table(sample, frame, exposure, events)
        tables.append(table)
        summary[sample] = {"trigger_days": int(frame["extreme_trigger"].sum()),
                           "first": str(frame.index[0].date()), "last": str(frame.index[-1].date())}
        print(f"[{sample}] 触发 {summary[sample]['trigger_days']} 天", flush=True)
    if tables:
        out = pd.concat(tables, ignore_index=True)
        out.to_csv(ROOT / "windows-market.csv", index=False, encoding="utf-8-sig")
        print(out[["sample", "window", "trigger_days", "first_trigger", "ret_A_market",
                   "ret_E_market", "mdd_A_market", "mdd_E_market", "rebound_fail"]].to_string(index=False),
              flush=True)
    (ROOT / "summary-market.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def run_early(sample: str) -> int:
    frame = early_frame(sample)
    exposure, events = extreme_plan(frame, frame["extreme_trigger"].astype(bool))
    label = f"early-{sample}"
    events.to_csv(ROOT / "early" / f"events-E-{sample}.csv", index=False, encoding="utf-8-sig")
    table = window_table(label, frame, exposure, events)
    path = ROOT / "early" / f"windows-{sample}.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(table[["window", "trigger_days", "first_trigger", "ret_A_market", "ret_E_market",
                 "mdd_A_market", "mdd_E_market", "rebound_fail"]].to_string(index=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多危机泛化测试")
    parser.add_argument("--market", action="store_true")
    parser.add_argument("--early", default=None, choices=["2001-2005", "1997-1999"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    if args.all or args.market:
        run_market()
    for sample in (["2001-2005", "1997-1999"] if args.all else ([args.early] if args.early else [])):
        if (EARLY / "bars" / sample).is_dir():
            run_early(sample)
        else:
            print(f"[early] {sample} 数据未就绪", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
