"""幸存者偏差量化：把 256 只退市股补进票池，看因子 IC / 分层收益会不会变。

为什么必须做：原始行情来自新浪实时列表（5576 只），**当期在市**才有；
历史回测里"每期可交易股票池"如果只含活着的股票，等于自动剔除了最惨的一批，
动量、反转、低波这类因子会被系统性低估风险。

做法：同一套因子分别在两个面板上评估——
* ``live``：processed/daily（5278 只在市股票）
* ``full``：live + delisted/daily（256 只退市股，后复权日线，退市后无数据）

输出 ``results/survivorship-bias.md``：每个因子在两套票池上的 IC 均值/IC IR/多空收益。

    python -m quant factor-bias
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.data import CsvSource, DataStore  # noqa: E402
from quant.factors import build_panel, compute, evaluate_factor  # noqa: E402
from quant.report import to_markdown_table  # noqa: E402

LIVE_DIR = Path("D:/quant-data/processed/daily")
DELISTED_DIR = Path("D:/quant-data/delisted/daily")
SPREAD_ROW = "多空(Q最高-Q最低)"
FIELDS = ("open", "high", "low", "close", "volume", "amount", "prev_close")


def load_live() -> dict[str, pd.DataFrame]:
    store = DataStore(CsvSource(str(LIVE_DIR)))
    return store.get_many(None)


def load_delisted() -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    if not DELISTED_DIR.is_dir():
        return bars
    for path in sorted(DELISTED_DIR.glob("??????.parquet")):
        frame = pd.read_parquet(path)
        if len(frame) == 0:
            continue
        frame = frame.rename(columns={"preclose": "prev_close"})
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        keep = [c for c in FIELDS if c in frame.columns]
        bars[path.stem] = frame[keep]
    return bars


def evaluate(factor: str, panel, horizon: int, quantiles: int) -> dict:
    frame = compute(factor, panel)
    result = evaluate_factor(frame, panel, name=factor, horizons=(1, horizon), quantiles=quantiles)
    summary = result["ic"]
    qsum = result["quantile_summary"]
    spread = float(qsum.loc[SPREAD_ROW, "mean"]) if SPREAD_ROW in qsum.index else float("nan")
    return {
        "coverage": round(float(result["coverage"]), 1),
        "ic_mean": round(float(summary["ic_mean"]), 4),
        "ic_ir": round(float(summary["ic_ir"]), 3),
        "t_stat": round(float(summary["t_stat"]), 2),
        "positive_rate": round(float(summary["positive_rate"]), 3),
        "spread": round(spread * 100, 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m quant factor-bias", description="幸存者偏差量化")
    parser.add_argument(
        "--factors",
        default="momentum,reversal,low_vol,downside_vol,turnover_abn,amihud,max_ret",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    live = load_live()
    dead = load_delisted()
    print(f"live {len(live)} 只 · delisted {len(dead)} 只", flush=True)

    all_bars = dict(live)
    for symbol, frame in dead.items():
        if symbol not in all_bars:
            all_bars[symbol] = frame

    panel_all = build_panel(all_bars)
    live_symbols = [s for s in live if s in panel_all.symbols]
    panel_live = type(panel_all)({key: value[live_symbols] for key, value in panel_all.fields.items()})

    # 只在**两个面板都有行情的日期**上比较：退市股日线起于 2014、在市股票起于 2016，
    # 若不裁剪，2014-2015 与 2025 之后的横截面里只剩退市股，IC 会被噪声吞掉（曾踩到）。
    common = panel_live.dates.intersection(panel_all.dates)
    panel_live = type(panel_all)({key: value.loc[common] for key, value in panel_live.fields.items()})
    panel_all = type(panel_all)({key: value.loc[common] for key, value in panel_all.fields.items()})
    print(
        f"面板：live {len(panel_live.symbols)} 只 / full {len(panel_all.symbols)} 只，"
        f"共同区间 {common[0].date()} ~ {common[-1].date()}（{len(common)} 个交易日）",
        flush=True,
    )

    rows = []
    wanted = [item.strip() for item in args.factors.split(",") if item.strip()]
    for name in wanted:
        left = evaluate(name, panel_live, args.horizon, args.quantiles)
        print(f"  {name} live 完成 IC {left['ic_mean']:+.4f}", flush=True)
        right = evaluate(name, panel_all, args.horizon, args.quantiles)
        print(f"  {name} full 完成 IC {right['ic_mean']:+.4f}", flush=True)
        rows.append(
            {
                "factor": name,
                "ic_live": left["ic_mean"],
                "ic_full": right["ic_mean"],
                "ic_diff": round(right["ic_mean"] - left["ic_mean"], 4),
                "ir_live": left["ic_ir"],
                "ir_full": right["ic_ir"],
                "spread_live": left["spread"],
                "spread_full": right["spread"],
                "coverage_live": left["coverage"],
                "coverage_full": right["coverage"],
            }
        )

    table = pd.DataFrame(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "survivorship-bias.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 幸存者偏差量化：补进退市股后因子表现怎么变",
        "",
        f"- 在市票池：{len(panel_live.symbols)} 只；全票池：{len(panel_all.symbols)} 只（多 {len(panel_all.symbols) - len(panel_live.symbols)} 只退市股）",
        f"- 区间：{common[0].date()} ~ {common[-1].date()}（{len(common)} 个交易日）；IC 观察期 {args.horizon} 日；分层 {args.quantiles} 组",
        "- `ic_diff = full - live`：负值表示只用在市票池会**高估**该因子",
        "",
        to_markdown_table(table, floatfmt="+.4f", index=False),
        "",
        "> 退市股日线为 baostock 后复权，退市后无数据；回测里必须按 `delisted/events.csv` 的",
        "> `last_close_hfq` 结算，并跑 `last / -30% / -100%` 三种处置假设。",
    ]
    (out_dir / "survivorship-bias.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + table.to_string(index=False))
    print(f"\n报告已写入 {out_dir / 'survivorship-bias.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
