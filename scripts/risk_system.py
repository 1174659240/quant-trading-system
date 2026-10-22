"""极端风险清仓系统（任务一）+ 递减式回补：通用市场级仓位系数。

规则冻结在 ``docs/极端风险清仓预注册.md``，本模块常量必须与之一致。
只新增，不改 ``quant/`` 与既有脚本；产物落 ``results/risk-system/``。

用法：
    python -m scripts.risk_system --sample main  --specs A,B,E,C5k
    python -m scripts.risk_system --sample history --specs A,B,E,C5k
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtest.engine import Engine
from quant.config import BacktestConfig, CostModel
from quant.factors import growth, growth_alt
from quant.factors.panel import build_panel
from scripts import growth_leftside as gl
from scripts import growth_risk as gr
from scripts.factor_market import FIELDS, load_delisting_events
from scripts.growth_backtest import SPECS, composite_score, month_ends, sector_panel, turnover_stats
from scripts.redlow_study import END, LAKE, START, board_exclude, build_block, load_bars, pool_benchmark

OUT = Path("results/risk-system")

# ---- 预注册阈值（docs/极端风险清仓预注册.md §2） --------------------------- #
EXT_DD = 0.20            # 回撤：dd250 <= -20%
EXT_BREADTH = 0.12       # 广度：站上 MA50 占比 <= 12%
EXT_R20 = -0.35          # 速度：20 日累计 <= -35%
EXT_R60 = -0.45          # 速度：60 日累计 <= -45%
EXT_CONFIRM = 3          # 连续 3 日确认
QUIET_DAYS = 15          # 企稳：15 日不创新低
VOL_CAP = 0.97           # 企稳：波动分位 < 0.97
VOL_DROP = 0.10          # 或较本段峰值回落 0.10
FRACTIONS = (0.5, 1 / 3, 1 / 6)   # 递减三档（占剩余缺口的比例）
STEP_DAYS = 20           # 两档最小间隔（交易日）
GATE_LOW_DAYS = 5        # 闸门：近 5 日新低
GATE_ILLIQ_PCT = 0.95    # 闸门：非流动性分位
GATE_ILLIQ_LOOKBACK = 20
GRADIENT_EXPOSURE = 0.5  # B 规格危机期仓位
C5K_EXPOSURE = 0.0       # 既有 crisis5k：清仓
C5K_RAMP = 10            # 既有 crisis5k：10 日线性回补（对照，不改）
# ---- 任务二 QVIX 回补侧确认（docs/QVIX回补与多危机泛化预注册.md §2） -------- #
QVIX_PCT_CAP = 0.90      # QVIX 分位（500 日滚动）必须 < 0.90
QVIX_DROP_CAP = 0.70     # QVIX 必须从本段峰值回落 >= 30%
QVIX_CACHE = Path("results/q-vix/qvix.csv")


# --------------------------------------------------------------------------- #
# 触发与仓位路径
# --------------------------------------------------------------------------- #
def add_speed(frame: pd.DataFrame) -> pd.DataFrame:
    """给 market_frame 补 20/60 日累计涨跌（简单求和，与诊断脚本一致）。"""
    out = frame.copy()
    out["r20"] = out["market_ret"].rolling(20, min_periods=20).sum()
    out["r60"] = out["market_ret"].rolling(60, min_periods=60).sum()
    return out


def extreme_trigger(frame: pd.DataFrame) -> pd.Series:
    """§2.1：趋势 + 回撤 + 广度 + 速度，连续 3 日确认。"""
    need = (
        (frame["index"] < frame["ma200"])
        & (frame["dd250"] <= -EXT_DD)
        & (frame["breadth"] <= SENS["breadth"])
        & ((frame["r20"] <= SENS["r20"]) | (frame["r60"] <= SENS["r60"]))
    ).fillna(False)
    return need.rolling(EXT_CONFIRM, min_periods=EXT_CONFIRM).sum().ge(EXT_CONFIRM)


# 敏感性分析时临时覆盖用（默认等于预注册阈值）
SENS = {"breadth": EXT_BREADTH, "r20": EXT_R20, "r60": EXT_R60}

# 预注册 §4：广度 0.10/0.15、速度整体收紧/放宽到 0.30/0.40（r60 按 45/35 同比例缩放）
SENS_CONFIGS = (
    ("breadth_0.10", {"breadth": 0.10}),
    ("breadth_0.15", {"breadth": 0.15}),
    ("speed_0.30", {"r20": -0.30, "r60": -0.30 * 45 / 35}),
    ("speed_0.40", {"r20": -0.40, "r60": -0.40 * 45 / 35}),
)


def _gates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    index = frame["index"]
    new_low = (index <= index.rolling(QUIET_DAYS, min_periods=QUIET_DAYS).min()).fillna(False)
    quiet = (new_low.rolling(QUIET_DAYS, min_periods=1).sum().fillna(0) == 0).to_numpy()
    recent_low = (new_low.rolling(GATE_LOW_DAYS, min_periods=1).sum().fillna(0) > 0).to_numpy()
    liq_bad = (
        (frame["illiq_pct"] > GATE_ILLIQ_PCT)
        & (frame["illiq_20"] > frame["illiq_20"].shift(GATE_ILLIQ_LOOKBACK))
    ).fillna(False).to_numpy()
    return quiet, (recent_low | liq_bad)


def extreme_plan(frame: pd.DataFrame, trigger: pd.Series, *,
                 qvix: pd.Series | None = None,
                 qvix_pct: pd.Series | None = None,
                 qvix_confirm: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """§2.2–2.3：触发后清仓、企稳后递减 3 档回补、闸门暂停、再触发重置。

    ``qvix_confirm=True`` 时，每一档回补额外要求 QVIX 已从本段峰值回落 ≥ ``QVIX_DROP``
    且 QVIX 分位 < ``QVIX_PCT_CAP``（任务二：回补侧确认，不参与触发）。
    """
    index = frame.index
    quiet, gate = _gates(frame)
    vol = frame["vol_pct"].to_numpy(dtype=float)
    trig = trigger.reindex(index).fillna(False).to_numpy(dtype=bool)
    idx = frame["index"].to_numpy(dtype=float)
    qv = (qvix.reindex(index).to_numpy(dtype=float) if qvix is not None
          else np.full(len(index), np.nan))
    qp = (qvix_pct.reindex(index).to_numpy(dtype=float) if qvix_pct is not None
          else np.full(len(index), np.nan))

    def qvix_ready(position: int, peak: float) -> bool:
        if not qvix_confirm:
            return True
        value = qv[position]
        if not np.isfinite(value):
            return True  # 2015 年前无 QVIX，不做要求
        pct = qp[position] if np.isfinite(qp[position]) else 1.0
        return bool(peak > 0 and value <= QVIX_DROP_CAP * peak and pct < QVIX_PCT_CAP)

    exposure = np.ones(len(index))
    current = 1.0
    in_crisis = False
    peak_vol = 0.0
    qvix_peak = 0.0
    episode = 0
    ramp: dict | None = None
    last_step = -10**9
    paused = False
    events: list[dict] = []

    for position in range(len(index)):
        stale = trig[position]
        if stale and not in_crisis:
            in_crisis = True
            episode += 1
            current = 0.0
            ramp = None
            peak_vol = vol[position] if np.isfinite(vol[position]) else 0.0
            qvix_peak = qv[position] if np.isfinite(qv[position]) else 0.0
            paused = False
            events.append({"date": index[position], "kind": "clear", "episode": episode,
                           "exposure_after": 0.0, "index": idx[position],
                           "note": "极端触发，清仓"})
        elif in_crisis:
            if stale and ramp is not None:
                # 回补途中再次触发 → 重置为清仓
                episode += 1
                current = 0.0
                ramp = None
                paused = False
                qvix_peak = qv[position] if np.isfinite(qv[position]) else 0.0
                events.append({"date": index[position], "kind": "reset", "episode": episode,
                               "exposure_after": 0.0, "index": idx[position],
                               "note": "回补中再次触发，重新清仓"})
            else:
                if np.isfinite(vol[position]):
                    peak_vol = max(peak_vol, vol[position])
                if np.isfinite(qv[position]):
                    qvix_peak = max(qvix_peak, qv[position])
                vol_now = vol[position]
                stable = bool(quiet[position] and np.isfinite(vol_now)
                              and (vol_now < VOL_CAP or vol_now <= peak_vol - VOL_DROP))
                if stable:
                    if ramp is None:
                        ramp = {"gap": 1.0 - current, "done": 0}
                        last_step = position - STEP_DAYS
                        events.append({"date": index[position], "kind": "ramp_start",
                                       "episode": episode, "exposure_after": round(current, 4),
                                       "index": idx[position], "note": "企稳确认，开始分批回补"})
                    spacing = (position - last_step) >= STEP_DAYS
                    if gate[position] or not spacing or not qvix_ready(position, qvix_peak):
                        if not paused:
                            paused = True
                            if not qvix_ready(position, qvix_peak):
                                note = "QVIX 未确认（未回落 30% 或分位仍极端）"
                            else:
                                note = "新低未消" if gate[position] else "间隔不足"
                            events.append({"date": index[position], "kind": "gate_pause",
                                           "episode": episode, "exposure_after": round(current, 4),
                                           "index": idx[position], "note": note})
                    else:
                        before = current
                        current = min(1.0, before + ramp["gap"] * FRACTIONS[ramp["done"]])
                        ramp["done"] += 1
                        last_step = position
                        paused = False
                        events.append({"date": index[position], "kind": "tranche", "episode": episode,
                                       "exposure_before": round(before, 4),
                                       "exposure_after": round(current, 4),
                                       "index": idx[position],
                                       "note": f"第 {ramp['done']}/3 档（缺口 {ramp['gap']:.2f}）"})
                        if ramp["done"] >= 3 or current >= 1.0 - 1e-9:
                            current = 1.0
                            in_crisis = False
                            ramp = None
        exposure[position] = current

    return pd.Series(exposure, index=index), pd.DataFrame(events)


def gradient_plan(score: pd.Series, state: pd.Series,
                  frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """B 规格：危机期 0.5，解除后按 §2.3 递减 3 档回补（取代 10 日线性）。"""
    index = score.index
    quiet, gate = _gates(frame)
    vol = frame["vol_pct"].reindex(index).to_numpy(dtype=float)
    flags = state.reindex(index).fillna(False).to_numpy(dtype=bool)
    scores = score.to_numpy(dtype=float)
    idx = frame["index"].reindex(index).to_numpy(dtype=float)

    exposure = np.ones(len(index))
    current = 1.0
    in_crisis = False
    peak_vol = 0.0
    episode = 0
    ramp: dict | None = None
    last_step = -10**9
    events: list[dict] = []

    for position in range(len(index)):
        if flags[position]:
            if not in_crisis:
                in_crisis = True
                episode += 1
                current = GRADIENT_EXPOSURE
                ramp = None
                peak_vol = vol[position] if np.isfinite(vol[position]) else 0.0
                last_step = position - STEP_DAYS
                events.append({"date": index[position], "kind": "crisis", "episode": episode,
                               "exposure_after": current, "index": idx[position],
                               "note": "梯度降仓至 0.5"})
            else:
                if np.isfinite(vol[position]):
                    peak_vol = max(peak_vol, vol[position])
        else:
            if in_crisis:
                in_crisis = False
                ramp = {"gap": 1.0 - current, "done": 0}
                last_step = position - STEP_DAYS
                events.append({"date": index[position], "kind": "ramp_start", "episode": episode,
                               "exposure_after": round(current, 4), "index": idx[position],
                               "note": "危机解除，开始递减回补"})
            if ramp is not None and ramp["done"] < 3:
                stable = bool(quiet[position] and np.isfinite(vol[position])
                              and (vol[position] < VOL_CAP or vol[position] <= peak_vol - VOL_DROP))
                spacing = (position - last_step) >= STEP_DAYS
                if stable and spacing and not gate[position]:
                    before = current
                    current = min(1.0, before + ramp["gap"] * FRACTIONS[ramp["done"]])
                    ramp["done"] += 1
                    last_step = position
                    events.append({"date": index[position], "kind": "tranche", "episode": episode,
                                   "exposure_before": round(before, 4),
                                   "exposure_after": round(current, 4), "index": idx[position],
                                   "note": f"第 {ramp['done']}/3 档"})
                    if ramp["done"] >= 3 or current >= 1.0 - 1e-9:
                        current = 1.0
                        ramp = None
            elif ramp is None:
                soft = 0.45 <= scores[position] < gr.CRISIS_ENTER
                current = gr.CRISIS_SOFT_EXPOSURE if soft else 1.0
        exposure[position] = current

    return pd.Series(exposure, index=index), pd.DataFrame(events)


def binary_plan(state: pd.Series, *, exposure: float = C5K_EXPOSURE,
                ramp_days: int = C5K_RAMP) -> pd.Series:
    """既有 crisis5k 口径：危机期 exposure，解除后 ramp_days 线性回补（对照，不改）。"""
    flags = state.fillna(False).to_numpy(dtype=bool)
    values = np.ones(len(flags))
    for position, flag in enumerate(flags):
        if flag:
            values[position] = exposure
    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        while index < len(flags) and flags[index]:
            index += 1
        for step in range(ramp_days):
            pos = index + step
            if pos >= len(flags):
                break
            values[pos] = min(values[pos], exposure + (1.0 - exposure) * (step + 1) / ramp_days)
    return pd.Series(values, index=state.index)


def exposure_to_weights(weights: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    frame = exposure.reindex(weights.index).ffill().fillna(1.0)
    return weights.mul(frame, axis=0).fillna(0.0)


# 默认市场级风险层（2026-10-21 用户拍板：极端清仓 E 对所有策略默认开启）
DEFAULT_LAYER = "extreme"


def default_layer(panel) -> tuple[pd.Series, pd.DataFrame]:
    """返回 ``(全局仓位系数 0~1, 事件明细)``；所有策略的默认层就是它。

    用法见 ``docs/小盘成长策略设计与数据核对.md`` §5.6：
    ``weights = exposure_to_weights(strategy_weights, default_layer(panel)[0])``。
    适用边界（只覆盖 2008 型快速深跌；慢熊/熔断/V 型浅跌不触发）同样写在该节。
    """
    frame = add_speed(gl.market_frame(panel))
    trigger = extreme_trigger(frame)
    return extreme_plan(frame, trigger)


def load_qvix(cache: Path = QVIX_CACHE) -> pd.Series:
    """50ETF QVIX 日线收盘（2015-02-09 起）；缓存到 ``results/q-vix/qvix.csv``。"""
    if cache.is_file():
        frame = pd.read_csv(cache)
    else:
        import akshare as ak
        frame = ak.index_option_50etf_qvix()
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False, encoding="utf-8-sig")
    columns = {str(c).strip().lower(): c for c in frame.columns}
    date_col = columns.get("date", frame.columns[0])
    close_col = columns.get("close", frame.columns[-1])
    series = pd.Series(pd.to_numeric(frame[close_col], errors="coerce").to_numpy(),
                       index=pd.to_datetime(frame[date_col])).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    print(f"[qvix] {len(series)} 行 {series.index.min().date()} ~ {series.index.max().date()}", flush=True)
    return series


def qvix_percentile(series: pd.Series, window: int = 500, min_periods: int = 120) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


def monthly_qvix_gate(path: pd.Series, qvix: pd.Series, qvix_pct: pd.Series) -> pd.Series:
    """在月末生效点上追加 QVIX 确认：加仓档若 QVIX 未回落则不生效（顺延）。

    必要性：仓位只在月末调仓日进入组合权重（``BufferTopK`` 的权重是月末快照），
    逐日层面的判断会空转；因此把确认放在真正能成交的月末采样点上。
    """
    marks = pd.Series(path.index, index=path.index).groupby(
        [path.index.year, path.index.month]).last()
    qv = qvix.reindex(path.index).ffill()
    qp = qvix_pct.reindex(path.index).ffill()
    sampled = path.copy()
    previous = 1.0
    episode_start = path.index[0]
    for mark in marks:
        target = float(path.loc[mark])
        if target <= 1e-9 and previous > 1e-9:
            episode_start = mark
        peak = float(qv.loc[episode_start:mark].max()) if mark >= episode_start else float("nan")
        value = float(qv.loc[mark])
        pct = float(qp.loc[mark])
        if target > previous + 1e-9 and np.isfinite(value):
            ready = (np.isfinite(peak) and peak > 0 and value <= QVIX_DROP_CAP * peak
                     and pct < QVIX_PCT_CAP)
            if not ready:
                target = previous
        sampled.loc[mark] = target
        previous = target
    # 月末之间保持上一档（与实际权重快照一致）
    return sampled.reindex(path.index).ffill()


# --------------------------------------------------------------------------- #
# 回测
# --------------------------------------------------------------------------- #
@dataclass
class Spec:
    name: str
    label: str
    plan: str            # none | gradient | extreme | binary
    extra: dict = field(default_factory=dict)


SPEC_LIST = (
    Spec("A", "无风控", "none"),
    Spec("B", "梯度降仓 0.5 + 递减回补", "gradient"),
    Spec("E", "极端清仓 + 递减回补", "extreme"),
    Spec("EQ", "极端清仓 + QVIX 回补确认", "extreme_qvix"),
    Spec("C5k", "既有 crisis5k（0.75 二元 + 10 日线性）", "binary"),
)


def run(sample: str, specs: tuple[Spec, ...], out: Path, *, sensitivity: bool = False,
        sector_cap_override: float | None = None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if sample == "history":
        bars = gl.load_history_bars()
        start, end, cutoff = gl.HIST_START, gl.HIST_END, gl.HIST_CUTOFF
        alt_kwargs = {"cutoff": gl.HIST_CUTOFF, "strict": False}
        fund_extra = (gl.HIST / "fundamentals",)
        sector_cap = 1.0
    else:
        bars, _ = load_bars()
        start, end, cutoff = START, END, pd.Timestamp("2024-12-31")
        alt_kwargs = {}
        fund_extra = ()
        sector_cap = gr.SECTOR_CAP

    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    panel = build_panel(bars, fields=FIELDS)
    block = build_block(panel).reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)
    print(f"[panel] {sample} {panel.close.shape}", flush=True)

    facts = growth.GrowthFactors(panel, cutoff=cutoff, extra_dirs=fund_extra)
    eligible = growth.eligible_mask(facts, block_mask=block)
    marks = month_ends(panel.dates, start, end)
    alt = growth_alt.alt_factors(panel, **alt_kwargs)
    sector = None if sample == "history" else sector_panel(marks, list(panel.symbols))
    if sector_cap_override is not None:
        sector_cap = float(sector_cap_override)
        print(f"[sector] 门类上限覆盖为 {sector_cap:.0%}", flush=True)
    score = composite_score(panel, facts, alt, SPECS["QN"]["blocks"], eligible, marks)

    adv20, illiq20 = gr.liquidity_panels(panel)
    slip_maps, tier_rows = gr.tier_map(marks, eligible, adv20, illiq20)
    tier_rows.to_csv(out / f"liquidity-tiers-{sample}.csv", index=False, encoding="utf-8-sig")

    base_frame, crisis_score, crisis_state, _ = gr.crisis_inputs(panel)
    frame = add_speed(gl.market_frame(panel))
    frame["score"] = crisis_score.reindex(frame.index)
    frame["crisis"] = crisis_state.reindex(frame.index).fillna(False)
    trigger = extreme_trigger(frame)
    frame["extreme_trigger"] = trigger
    frame.to_csv(out / f"market-frame-{sample}.csv", encoding="utf-8-sig")

    plans: dict[str, pd.Series] = {"none": pd.Series(1.0, index=frame.index)}
    events_all: dict[str, pd.DataFrame] = {}
    ext_path, ext_events = extreme_plan(frame, trigger)
    plans["extreme"] = ext_path
    events_all["E"] = ext_events
    qvix_raw = load_qvix()
    qvix_series = qvix_raw.reindex(frame.index).ffill()
    qvix_pct = qvix_percentile(qvix_raw).reindex(frame.index).ffill()
    extq_path, extq_events = extreme_plan(frame, trigger, qvix=qvix_series,
                                          qvix_pct=qvix_pct, qvix_confirm=True)
    extq_path = monthly_qvix_gate(extq_path, qvix_series, qvix_pct)
    plans["extreme_qvix"] = extq_path
    events_all["EQ"] = extq_events
    grad_path, grad_events = gradient_plan(crisis_score, crisis_state, frame)
    plans["gradient"] = grad_path
    events_all["B"] = grad_events
    plans["binary"] = binary_plan(crisis_state.reindex(frame.index))

    summary = {}
    plan_event_key = {"extreme": "E", "gradient": "B", "extreme_qvix": "EQ"}
    for name, path in plans.items():
        events = events_all.get(plan_event_key.get(name, name), pd.DataFrame())
        summary[name] = {
            "days_cleared": int((path < 0.05).sum()),
            "days_partial": int(((path >= 0.05) & (path < 0.99)).sum()),
            "avg": round(float(path.mean()), 4),
            "min": round(float(path.min()), 4),
            "tranches": int((events.get("kind", pd.Series(dtype=str)) == "tranche").sum())
            if len(events) else None,
        }
    for name, events in events_all.items():
        events.to_csv(out / f"events-{name}-{sample}.csv", index=False, encoding="utf-8-sig")
    print(f"[risk] 触发 {int(trigger.sum())} 天；E 平均仓位 {plans['extreme'].mean():.3f}；"
          f"B {plans['gradient'].mean():.3f}；C5k {plans['binary'].mean():.3f}", flush=True)

    delisting = (gl.history_delistings(panel, end) if sample == "history"
                 else load_delisting_events(LAKE))
    bench_equity = pool_benchmark(panel, block, delisting, cash=1_000_000.0)
    bench_equity.to_csv(out / f"benchmark-{sample}.csv", encoding="utf-8-sig")
    bt_bars = {symbol: data.loc[start:end] for symbol, data in bars.items() if len(data)}
    gr.install_broker(min_trade=5000.0)

    base_weights = gr.BufferTopK(score, eligible, sector=sector, sector_cap=sector_cap,
                                 buffer_rank=gr.BUFFER_RANK).generate(bars)
    tier_slip: dict[str, float] = {}
    for day in marks:
        tier_slip.update(slip_maps.get(day, {}))

    config = dict(start=str(start.date()), end=str(end.date()), initial_cash=1_000_000.0,
                  benchmark=None, costs=CostModel(slippage_bps=gr.BASE_BPS))

    rows = []
    for spec in specs:
        weights = exposure_to_weights(base_weights, plans[spec.plan])
        gr.RiskBroker.ALLOW = None
        gr.RiskBroker.MIN_TRADE = 5000.0
        gr.RiskBroker.SLIP = tier_slip
        gr.RiskBroker.BASE_BPS = gr.BASE_BPS
        print(f"[{spec.name}] 回测…（平均仓位 {weights.abs().sum(axis=1).mean() / 0.95:.3f}）", flush=True)
        result = Engine(config=BacktestConfig(**config), benchmark_prices=None, delisting=delisting,
                        delisting_mode="haircut30").run(bt_bars, weights, strategy_name=f"risk-{spec.name}")
        row = {"sample": sample, "spec": spec.name, "label": spec.label,
               **gr.evaluate(result, bench_equity, cash=1_000_000.0, slip_label="分档"),
               **{f"exposure_{k}": v for k, v in summary[spec.plan].items()}}
        row["ann_turnover_one_way"] = turnover_stats(weights, marks)["ann_turnover_one_way"]
        row["mdd_date"] = str(gr.drawdown_series(result.equity).idxmin().date())
        rows.append(row)
        result.equity.to_csv(out / f"equity-{spec.name}-{sample}.csv", encoding="utf-8-sig")
        result.daily.to_csv(out / f"daily-{spec.name}-{sample}.csv", encoding="utf-8-sig")
        if len(result.trades):
            result.trades.to_csv(out / f"trades-{spec.name}-{sample}.csv", index=False, encoding="utf-8-sig")
        print(f"  → 累计 {row['total_return']:.2%} 年化 {row['cagr']:.2%} 夏普 {row['sharpe']:.3f} "
              f"MDD {row['max_drawdown']:.2%} 换手 {row['ann_turnover_one_way']:.0%} "
              f"暴露 {row['avg_exposure']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out / f"metrics-{sample}.csv", index=False, encoding="utf-8-sig")

    sens_rows = []
    if sensitivity:
        for name, override in SENS_CONFIGS:
            SENS.update({"breadth": EXT_BREADTH, "r20": EXT_R20, "r60": EXT_R60})
            SENS.update(override)
            trig_s = extreme_trigger(frame)
            path_s, _ = extreme_plan(frame, trig_s)
            weights_s = exposure_to_weights(base_weights, path_s)
            print(f"[sens {name}] 触发 {int(trig_s.sum())} 天，平均仓位 {path_s.mean():.3f}", flush=True)
            res_s = Engine(config=BacktestConfig(**config), benchmark_prices=None, delisting=delisting,
                           delisting_mode="haircut30").run(bt_bars, weights_s, strategy_name=f"sens-{name}")
            row_s = {"sample": sample, "spec": name, "label": f"E 敏感性 {name}",
                     **gr.evaluate(res_s, bench_equity, cash=1_000_000.0, slip_label="分档"),
                     "trigger_days": int(trig_s.sum())}
            sens_rows.append(row_s)
        for key, value in {"breadth": EXT_BREADTH, "r20": EXT_R20, "r60": EXT_R60}.items():
            SENS[key] = value
        sens = pd.DataFrame(sens_rows)
        sens.to_csv(out / f"sensitivity-{sample}.csv", index=False, encoding="utf-8-sig")
        print(sens[["spec", "cagr", "sharpe", "max_drawdown", "trigger_days"]].to_string(index=False),
              flush=True)

    (out / f"summary-{sample}.json").write_text(
        json.dumps({"sample": sample, "trigger_days": int(trigger.sum()),
                    "plans": summary, "elapsed": round(time.time() - started, 1)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(table[["spec", "cagr", "sharpe", "max_drawdown", "calmar",
                 "ann_turnover_one_way", "avg_exposure"]].to_string(index=False), flush=True)
    print(f"[done] {out} {time.time() - started:.1f}s", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="极端风险清仓系统回测")
    parser.add_argument("--sample", choices=["main", "history"], default="main")
    parser.add_argument("--specs", default="A,B,E,C5k")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sensitivity", action="store_true", help="历史样本上跑 4 组敏感性（预注册 §4）")
    parser.add_argument("--sector-cap", type=float, default=None,
                        help="覆盖证监会门类单行业上限（默认用 growth_backtest.SECTOR_CAP=0.30）")
    args = parser.parse_args(argv)
    wanted = {name.strip().upper() for name in args.specs.split(",")}
    specs = tuple(spec for spec in SPEC_LIST if spec.name.upper() in wanted)
    return run(args.sample, specs, args.out, sensitivity=args.sensitivity,
               sector_cap_override=args.sector_cap)


if __name__ == "__main__":
    raise SystemExit(main())
