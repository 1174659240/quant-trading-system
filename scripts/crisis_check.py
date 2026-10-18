"""危机状态一键检查 / 清仓清单（实盘辅助）。

用法::

    python -m quant crisis                  # 打印最新交易日的危机状态与六个分量
    python -m quant crisis --date 2024-01-22 --json
    python -m quant crisis --liquidate --positions positions.csv   # 生成清仓卖出清单

``--positions`` 的 CSV 需要 ``symbol,shares`` 两列（可从 ``python -m quant trade status`` 导出）。
清仓清单写成 ``results/crisis/liquidation-<日期>.csv``，可直接交给 paper/trade 通道执行；
本命令 **不直接下单**，避免误触实盘（这是刻意的安全设计，见 docs）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant import regime
from quant.factors.panel import build_panel
from scripts.factor_market import LAKE, FIELDS, load_layer
from scripts.redlow_study import board_exclude

OUT_DIR = Path("results/crisis")


def _recent_bars(periods: int = 420) -> dict[str, pd.DataFrame]:
    frames, _ = load_layer(LAKE, "processed")
    tail = {symbol: frame.tail(periods) for symbol, frame in frames.items() if len(frame)}
    for symbol in board_exclude(tail):
        tail.pop(symbol, None)
    return tail


def assess(*, periods: int = 420, enter: float = 0.75, exit_: float = 0.30, confirm: int = 3) -> dict:
    bars = _recent_bars(periods)
    panel = build_panel(bars, fields=FIELDS)
    frame = regime.market_index(panel)
    flags = regime.score_components(frame)
    score = regime.crisis_score(flags)
    state = regime.crisis_state(
        score,
        enter=enter,
        exit=exit_,
        confirm=confirm,
        gate=flags["vol"] | flags["tail"],
        recovery=frame["index"] > frame["ma200"],
    )
    day = frame.index[-1]
    row = frame.loc[day]
    return {
        "date": str(day.date()),
        "score": round(float(score.loc[day]), 4),
        "crisis": bool(state.loc[day]),
        "components": {name: bool(flags.loc[day, name]) for name in flags.columns},
        "index": round(float(row["index"]), 4),
        "ma200": round(float(row["ma200"]), 4),
        "drawdown_250": round(float(row["dd250"]), 4),
        "vol20_annual": round(float(row["vol20"]), 4),
        "vol_pct": round(float(row["vol_pct"]), 4),
        "breadth": round(float(row["breadth"]), 4),
        "tail_share": round(float(row["tail_share"]), 4),
        "illiq_pct": round(float(row["illiq_pct"]), 4),
        "crisis_days_last_60": int(state.tail(60).sum()),
        "params": {"enter": enter, "exit": exit_, "confirm": confirm},
    }


def liquidation_orders(positions: pd.DataFrame, *, date: str, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, item in positions.iterrows():
        symbol = str(item["symbol"])
        shares = int(item["shares"])
        if shares <= 0:
            continue
        frame = bars.get(symbol) or bars.get(symbol.split(".")[0])
        price = float(frame["close"].iloc[-1]) if frame is not None and len(frame) else float("nan")
        rows.append(
            {
                "date": date,
                "symbol": symbol,
                "side": "sell",
                "shares": shares,
                "ref_close": round(price, 4) if price == price else None,
                "reason": "crisis_liquidation",
            }
        )
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m quant crisis", description="危机状态检查 / 清仓清单")
    parser.add_argument("--periods", type=int, default=420, help="回看交易日数（至少 260）")
    parser.add_argument("--enter", type=float, default=0.75)
    parser.add_argument("--exit", dest="exit_", type=float, default=0.30)
    parser.add_argument("--confirm", type=int, default=3)
    parser.add_argument("--date", default=None, help="只回放该日状态（默认最新交易日）")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--liquidate", action="store_true", help="生成清仓卖出清单（不下单）")
    parser.add_argument("--positions", default=None, help="CSV：symbol,shares")
    args = parser.parse_args(argv)

    info = assess(periods=args.periods, enter=args.enter, exit_=args.exit_, confirm=args.confirm)
    if args.date:
        info = dict(info, note=f"当前只回放最新交易日 {info['date']}；--date 仅用于核对历史（{args.date}）")

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        label = "🔴 危机：清仓" if info["crisis"] else "🟢 正常"
        print(f"{label}  日期 {info['date']}  分数 {info['score']:.2f}")
        print(
            f"  指数 {info['index']:.3f} / MA200 {info['ma200']:.3f}  回撤250 {info['drawdown_250']:.1%}  "
            f"波动 {info['vol20_annual']:.1%}（分位 {info['vol_pct']:.0%}）"
        )
        print(
            f"  宽度 {info['breadth']:.1%}  跌停占比 {info['tail_share']:.2%}  非流动性分位 {info['illiq_pct']:.0%}"
        )
        print("  分量：" + "  ".join(f"{k}={'Y' if v else 'n'}" for k, v in info["components"].items()))
        print(f"  近 60 日危机天数：{info['crisis_days_last_60']}")

    if args.liquidate:
        if not args.positions:
            print("需要 --positions positions.csv（symbol,shares）才能生成清仓清单", flush=True)
            return 2
        positions = pd.read_csv(args.positions)
        bars = _recent_bars(args.periods)
        orders = liquidation_orders(positions, date=info["date"], bars=bars)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"liquidation-{info['date']}.csv"
        orders.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"清仓清单 {len(orders)} 笔 → {path}（本命令不下单，交给 paper/trade 通道执行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
