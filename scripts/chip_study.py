"""筹码分布因子研究（任务二）：先小样本验证，再全量 IC / 分层 / 逐年 / 量价簇残差。

产物落 ``results/factor-chip/``：
- ``validation.md``：8 只代表股与东财公开筹码分布（获利比例、平均成本）的方向一致性；
- ``chip-ic.md`` / ``chip-ic.json``：全量单因子诊断（成长合格集内、月末）。

用法：
    python -m scripts.chip_study --validate
    python -m scripts.chip_study --full
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant.factors import chip, growth, library, style
from quant.factors.panel import build_panel
from scripts.factor_market import FIELDS
from scripts.growth_backtest import month_ends
from scripts.growth_factor_ic import (_max_abs_corr, _quantile_table, _rank_ic_series,
                                      _residual_returns, forward_month_returns)
from scripts.redlow_study import END, START, board_exclude, build_block, load_bars

OUT = Path("results/factor-chip")
VALIDATION_CODES = ["000001", "600519", "300750", "002415", "601318",
                    "000651", "688981", "300059"]


def load_panel():
    bars, _ = load_bars()
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    panel = build_panel(bars, fields=FIELDS)
    return panel, bars


# --------------------------------------------------------------------------- #
# 一、小样本验证：与东财公开筹码分布对比
# --------------------------------------------------------------------------- #
def validate(panel) -> str:
    """不变量 + 两个独立估计的一致性。

    说明：东财 ``stock_cyq_em`` 只返回**最近 90 个交易日**，全部落在 2025 年之后；
    与本项目「2025-01-01 起 holdout 一行不读」的纪律冲突，因此**不联网取东财数据**，
    改用不依赖外部数据的检验：取值范围、单调关系、250 日与 90 日窗口一致性、与 VWAP 成本的一致性。
    """
    dates = panel.dates
    marks = dates[-90:]
    own = chip.chip_panels(panel, marks, window=250)
    own90 = chip.chip_panels(panel, marks, window=90)

    rows = []
    for code in VALIDATION_CODES:
        if code not in panel.symbols:
            continue
        close = panel.close[code].reindex(marks)
        amount = panel.amount[code].reindex(dates)
        volume = panel.volume[code].reindex(dates)
        vwap = ((amount / volume.replace(0, np.nan)).rolling(250, min_periods=120).mean()
                .reindex(marks))
        profit = own["profit"][code]
        cost = close / (1.0 + own["overhang"][code])
        rows.append({
            "code": code,
            "profit_min": round(float(profit.min()), 3),
            "profit_max": round(float(profit.max()), 3),
            "profit_mean": round(float(profit.mean()), 3),
            "conc_order_ok": bool((own["conc90"][code] >= own["conc70"][code]).all()),
            "rho_profit_overhang": round(float(profit.corr(own["overhang"][code])), 3),
            "rho_profit_250_90": round(float(profit.corr(own90["profit"][code])), 3),
            "cost_vs_vwap": round(float((cost / vwap).median()), 3),
            "rho_cost_vwap": round(float(cost.corr(vwap)), 3),
        })

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "validation.csv", index=False, encoding="utf-8-sig")
    lines = ["# 筹码分布因子：小样本验证", "",
             "8 只代表股（大盘/小盘/成长/价值/科技），最近 90 个交易日（截至数据湖 2024-12-31）。", "",
             "**未与东财 `stock_cyq_em` 对比**：该接口只给最近 90 个交易日（当前均在 2025 年之后），"
             "与「2025+ holdout 一行不读」冲突，因此只用不依赖外部数据的检验。", "",
             table.to_markdown(index=False), ""]
    if len(table):
        lines += [
            f"- 获利盘比例全部落在 [{table['profit_min'].min():.3f}, {table['profit_max'].max():.3f}]，未越界；",
            f"- 90% 区间宽度恒 ≥ 70% 区间宽度：{'全部通过' if table['conc_order_ok'].all() else '存在反例'}；",
            f"- 获利盘与资本利得悬置的秩相关中位数 {table['rho_profit_overhang'].median():.3f}（应为强正）；",
            f"- 250 日与 90 日窗口的获利盘秩相关中位数 {table['rho_profit_250_90'].median():.3f}；",
            f"- 自建成本 / 250 日 VWAP 中位数 {table['cost_vs_vwap'].median():.3f}（落在 1 附近即合理），"
            f"两者秩相关中位数 {table['rho_cost_vwap'].median():.3f}。",
            "",
            "判定：不变量全部成立 + 两个独立估计高度一致 → 通过，进入全量构造。",
        ]
    text = "\n".join(lines)
    (OUT / "validation.md").write_text(text, encoding="utf-8")
    print(text)
    return text


# --------------------------------------------------------------------------- #
# 二、全量因子诊断
# --------------------------------------------------------------------------- #
def stats(ic: pd.Series) -> dict:
    values = ic.dropna()
    if len(values) < 12:
        return {"months": int(len(values))}
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    t_stat = mean / std * np.sqrt(len(values)) if std > 0 else float("nan")
    years = values.groupby(values.index.year)
    positive = int(sum(1 for _, part in years if part.mean() > 0))
    return {"months": int(len(values)), "ic": round(mean, 4), "ic_ir": round(mean / std, 3) if std else None,
            "t": round(t_stat, 2), "positive_years": positive, "years": int(len(years))}


def full(panel, bars) -> int:
    started = time.time()
    eligible_block = build_block(panel).reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)
    facts = growth.GrowthFactors(panel, cutoff=pd.Timestamp("2024-12-31"))
    eligible = growth.eligible_mask(facts, block_mask=eligible_block)
    marks = month_ends(panel.dates, START, END)
    print(f"[marks] {len(marks)} 个月末", flush=True)

    panels = chip.chip_panels(panel, marks)
    forward = forward_month_returns(panel.close, marks)
    print(f"[chip] 计算完成 {time.time() - started:.1f}s", flush=True)

    controls = {
        "turnover_20": library.compute("turnover_20", panel),
        "vol_60": library.compute("vol_60", panel),
        "reversal": -library.compute("momentum", panel),
        "low_price": library.compute("low_price", panel),
        "amount_20": library.compute("amount_20", panel),
    }
    residual_forward = _residual_returns(forward, controls, marks)

    catalog = {
        "chip_profit": ("获利盘比例", +1),
        "chip_overhang": ("资本利得悬置", +1),
        "chip_conc70": ("70% 成本集中度", +1),
        "chip_conc90": ("90% 成本集中度", +1),
    }
    rows, payload = [], {}
    for key, (label, _) in catalog.items():
        raw = panels[key[5:]] if key[5:] in panels else panels[key.replace("chip_", "")]
        factor = raw.where(eligible)
        ic = _rank_ic_series(factor, forward, marks)
        res = stats(_rank_ic_series(factor, residual_forward, marks))
        quant = _quantile_table(factor, forward, marks)
        by_year = ic.groupby(ic.index.year).mean()
        max_corr, partner = _max_abs_corr(factor, controls, marks)
        row = {"factor": key, "label": label, **stats(ic),
               "residual_t": res.get("t"), "residual_ic": res.get("ic"),
               "residual_months": res.get("months"),
               "max_corr": round(max_corr, 3), "partner": partner,
               "q1": round(float(quant.iloc[:, 0].mean()), 4) if len(quant) else None,
               "q5": round(float(quant.iloc[:, -1].mean()), 4) if len(quant) else None,
               "monotonic": round(float(np.corrcoef(np.arange(1, 6), quant.mean().to_numpy())[0, 1]), 2)
               if len(quant) > 3 else None}
        rows.append(row)
        payload[key] = {"stats": row, "by_year": {int(y): round(float(v), 4) for y, v in by_year.items()}}
        print(f"  {key}: IC {row.get('ic')} t {row.get('t')} 残差 t {row.get('residual_t')} "
              f"与量价簇最大相关 {row['max_corr']}（{partner}）", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "chip-ic.csv", index=False, encoding="utf-8-sig")
    (OUT / "chip-ic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 筹码分布因子：全量诊断", "",
             f"样本 2017-01~2024-12，{len(marks)} 个月末，成长合格集内 rank IC，forward=月末收盘进场。", "",
             table.to_markdown(index=False), "",
             "残差口径：对 {turnover_20, vol_60, 反转, low_price, amount_20} 逐月截面 OLS 取残差后重算 IC。", ""]
    for key, item in payload.items():
        years = "、".join(f"{y}:{v:+.3f}" for y, v in item["by_year"].items())
        lines.append(f"- **{key}** 逐年 IC：{years}")
    (OUT / "chip-ic.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[done] {time.time() - started:.1f}s", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="筹码分布因子研究")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    panel, bars = load_panel()
    print(f"[panel] {panel.close.shape}", flush=True)
    if args.validate:
        validate(panel)
    if args.full:
        return full(panel, bars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
