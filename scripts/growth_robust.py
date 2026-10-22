"""阶段 4：稳健性与过拟合检验（严格档，规则冻结，只检验不调参）。

检验对象（用户 m04147 指定）：
    QN 因子集 + 逐日交易且忽略 <5000 元单 + 分档滑点 + 极端清仓层 E（默认层）
    主样本 2017-01-01 ~ 2024-12-31；历史样本 2008-2016 作补充。

七项检验：DSR / PBO(CSCV) / 置换 / Stationary Bootstrap / Walk-Forward / IS-OOS / 参数敏感性。
所有数字写入 ``results/growth/robustness/``，结论写入 ``results/growth/robustness.md``。

纪律：
- 不改 ``quant/``；不覆盖既有结果目录；2025-01-01 起数据一行不读（所有读取都带 cutoff 断言）。
- 本轮只检验，不调参、不改阈值；参数敏感性只报告"是否孤立尖峰"，不据此改规格。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path("results/growth/robustness")
MAIN_RISK = Path("results/growth/stage3/risk")
STAGE3 = Path("results/growth/stage3")
LEFTSIDE = Path("results/growth/leftside")
HIST_DAILY = Path("results/crisis-generalization/history-v2/daily-E-history.csv")
HOLDOUT = pd.Timestamp("2025-01-01")
START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2024-12-31")
TRADING_DAYS = 244
NORMAL = NormalDist()
EULER = 0.5772156649015329

# 主规格 = legacy5k_slip（QN + 逐日重置 + 忽略<5000元单 + 分档滑点）；
# 极端清仓 E 在主样本窗口内 exposure 恒为 1.0（零触发），故与 A 逐日相同，
# 这一点在 main_is_e_noop() 里用落盘的 daily-A/E 文件复核，不靠口头声明。
MAIN_DAILY = MAIN_RISK / "daily-legacy5k_slip.csv"

# 可插拔的权重构造器：阶段 4.5 用它把「软权重 + 成本感知」版本接进同一套检验
# 签名 builder(prepared, spec) -> 权重矩阵；None 表示用默认 BufferTopK。
BUILDER: dict = {"fn": None}


def set_root(path) -> None:
    global ROOT
    ROOT = Path(path)


def set_builder(fn) -> None:
    BUILDER["fn"] = fn


def _build_weights(prepared: dict, spec: dict) -> pd.DataFrame:
    """按 spec 构造目标权重；默认 top-k + 缓冲带 60（阶段 3 口径）。

    ``spec["score"]`` 可覆盖打分矩阵（WFA 用拟合出的块权重重新合成）。
    """
    if BUILDER["fn"] is not None:
        return BUILDER["fn"](prepared, spec)
    from scripts import growth_risk as gr
    score = spec.get("score")
    if score is None:
        score = prepared["score"]
    return gr.BufferTopK(score, prepared["eligible"], sector=prepared["sector"],
                         sector_cap=spec.get("sector_cap", prepared["sector_cap"]),
                         top_k=spec.get("top_k", 50),
                         max_weight=spec.get("max_weight", 0.04),
                         buffer_rank=gr.BUFFER_RANK).generate(prepared["bars"])


# ---------------------------------------------------------------- 基础工具
def _assert_no_holdout(dates: pd.Index) -> None:
    latest = pd.Timestamp(pd.Index(dates).max())
    if latest >= HOLDOUT:
        raise AssertionError(f"读到 2025+ 数据：{latest}")


def load_daily(path: Path) -> pd.Series:
    """读回测日频净值 → 日收益序列（index=date）。"""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    _assert_no_holdout(frame["date"])
    series = frame.set_index("date")["equity"].astype(float)
    return series


def to_returns(equity: pd.Series) -> pd.Series:
    returns = equity.pct_change()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def monthly_returns(equity: pd.Series) -> pd.Series:
    month_end = equity.groupby([equity.index.year, equity.index.month]).last()
    values = month_end.pct_change().dropna()
    values.index = pd.Index([pd.Timestamp(year=y, month=m, day=1) for y, m in month_end.index[1:]])
    return values


def stats_block(returns: pd.Series, periods_per_year: int = TRADING_DAYS,
                rf: float = 0.02) -> dict:
    """与 ``quant.backtest.metrics.sharpe`` 同口径：(mean×244 − rf) / vol。"""
    if len(returns) < 3:
        return {"n": int(len(returns))}
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    vol = std * math.sqrt(periods_per_year)
    sharpe = (mean * periods_per_year - rf) / vol if vol > 0 else float("nan")
    equity = (1 + returns).cumprod()
    years = len(returns) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")
    peak = equity.cummax()
    mdd = float((equity / peak - 1).min())
    return {"n": int(len(returns)), "total": float(equity.iloc[-1] - 1), "cagr": cagr,
            "vol": vol, "sharpe": sharpe, "mdd": mdd,
            "calmar": cagr / abs(mdd) if mdd < 0 else float("nan")}


def _sharpe_from_sums(n: float, s1: float, s2: float) -> float:
    if n <= 0:
        return float("nan")
    mean = s1 / n
    var = s2 / n - mean * mean
    if var <= 0:
        return float("nan")
    return mean / math.sqrt(var)


def _percentile_ci(values: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def main_is_e_noop() -> dict:
    """复核：主样本内 E 层与无风控逐日相同（零触发）。"""
    a = load_daily(Path("results/crisis-generalization/main/daily-A-main.csv"))
    e = load_daily(Path("results/crisis-generalization/main/daily-E-main.csv"))
    same = bool(np.allclose(a.to_numpy(), e.to_numpy(), rtol=0, atol=1e-6))
    return {"identical": same, "max_abs_diff": float(np.max(np.abs(a.to_numpy() - e.to_numpy())))}


# ---------------------------------------------------------------- 任务一：试验台账
def build_ledger() -> dict:
    """从各阶段落盘产物与文档recap出试验台账；N 只增不减。"""
    ic = json.loads(Path("results/growth/factor-ic.json").read_text(encoding="utf-8"))
    registry = json.loads(Path("results/growth/old-factors-registry.json").read_text(encoding="utf-8"))
    metrics_main = pd.read_csv(STAGE3 / "metrics.csv")
    metrics_risk = pd.concat([
        pd.read_csv(MAIN_RISK / "metrics-stage31.csv"),
        pd.read_csv(MAIN_RISK / "metrics-stage32.csv"),
    ], ignore_index=True)
    leftside_main = pd.read_csv(LEFTSIDE / "metrics.csv")
    leftside_hist = pd.read_csv(Path("results/growth/leftside-history-v2/metrics.csv"))
    risk_main = pd.read_csv(Path("results/risk-system/metrics-main.csv"))
    risk_hist = pd.read_csv(Path("results/risk-system/metrics-history.csv"))

    factors = {
        "old": int(registry["N_old"]),
        "new_core": int(registry["N_new"]),
        "alt": int(registry["N_alt"]),
        "tested_stage_c": int(len(ic["factors"])),
        "note": "阶段 C 单因子诊断 93 个（旧 66 + 新核心 15 + 补充 13 − 已删/未算合计差异见 factor-ic.json）",
    }
    specs = sorted(metrics_main["spec"].astype(str).unique().tolist())
    risk_variants = sorted(set(metrics_risk["spec"].astype(str)) | set(risk_main["spec"].astype(str)))
    leftside_variants = sorted(set(leftside_main["spec"].astype(str)))
    grid = {
        "extreme_calibration": 192,
        "note": "阶段 3.5 极端清仓阈值标定网格（dd/tail/breadth/illiq/k/confirm 组合）",
    }
    trials = {
        "factor_screening": factors,
        "portfolio_specs": {"count": len(specs), "items": specs},
        "risk_trade_variants": {"count": len(metrics_risk), "items": risk_variants},
        "leftside_main": {"count": len(leftside_main), "items": leftside_variants},
        "leftside_history": {"count": len(leftside_hist), "items": sorted(leftside_hist["spec"].astype(str))},
        "risk_system_main": {"count": len(risk_main), "items": sorted(risk_main["spec"].astype(str))},
        "risk_system_history": {"count": len(risk_hist), "items": sorted(risk_hist["spec"].astype(str))},
        "calibration_grid": grid,
    }
    # N 的三档口径（只增不减；最终判定用最严的 N_all）
    n_specs = len(specs)                                   # 4
    n_strategy = (n_specs + len(metrics_risk) + len(leftside_main)
                  + len(leftside_hist) + len(risk_main))   # 策略级配置
    n_all = (factors["tested_stage_c"] + n_strategy + grid["extreme_calibration"]
             + len(risk_hist))
    ledger = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "objective": "QN + 逐日交易(忽略<5000元) + 分档滑点 + 极端清仓 E 默认层，主样本 2017-2024",
        "trials": trials,
        "N": {"N_spec_only": n_specs, "N_strategy_level": n_strategy, "N_all": n_all},
        "policy": "N 只增不减；本轮检验不产生新配置（敏感性扫描的 12 个邻域点单独登记，不计入选优 N）",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "trial-ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return ledger


# ---------------------------------------------------------------- 检验 1：DSR
def deflated_sharpe(returns: pd.Series, n_trials: int, *, periods: int = TRADING_DAYS,
                    rf: float = 0.02) -> dict:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio（SR 用超无风险口径，同 metrics）。"""
    r = returns.to_numpy(dtype=float)
    t = r.size
    excess = r - rf / periods
    sr = float(np.mean(excess) / np.std(excess, ddof=1))   # 单期 SR
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt()) + 3.0             # 原始峰度
    var_sr = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (t - 1)
    if var_sr <= 0:
        return {"sr": sr, "t": t, "skew": skew, "kurt": kurt, "sr0": float("nan"),
                "dsr": float("nan")}
    sr0 = math.sqrt(var_sr) * ((1 - EULER) * NORMAL.inv_cdf(1 - 1.0 / n_trials)
                               + EULER * NORMAL.inv_cdf(1 - 1.0 / (n_trials * math.e)))
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(1 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    return {"sr_period": sr, "sr_ann": sr * math.sqrt(periods), "t": t, "skew": skew,
            "kurt": kurt, "sr0_ann": sr0 * math.sqrt(periods),
            "dsr": float(NORMAL.cdf(z)), "n_trials": n_trials,
            "pass": bool(NORMAL.cdf(z) >= 0.95)}


# ---------------------------------------------------------------- 检验 4：bootstrap
def stationary_bootstrap(returns: pd.Series, *, reps: int = 1000, block: int = 6,
                         periods: int = 12, seed: int = 20261021) -> dict:
    """Politis-Romano 平稳 bootstrap（几何块长，均值 block 期）。"""
    rng = np.random.default_rng(seed)
    values = returns.to_numpy(dtype=float)
    n = values.size
    p = 1.0 / block
    sharpes = np.empty(reps)
    cagrs = np.empty(reps)
    for i in range(reps):
        idx = np.empty(n, dtype=int)
        pos = 0
        while pos < n:
            start = rng.integers(0, n)
            length = rng.geometric(p)
            length = min(length, n - pos)
            idx[pos:pos + length] = (start + np.arange(length)) % n
            pos += length
        sample = values[idx]
        std = sample.std(ddof=1)
        if std > 0:
            sharpes[i] = (sample.mean() * periods - 0.02) / (std * math.sqrt(periods))
        else:
            sharpes[i] = np.nan
        cagrs[i] = float(np.prod(1 + sample) ** (periods / n) - 1)
    lo_s, hi_s = _percentile_ci(sharpes)
    lo_c, hi_c = _percentile_ci(cagrs)
    return {"reps": reps, "block_months": block, "n": int(n),
            "sharpe_mean": float(np.nanmean(sharpes)), "sharpe_ci": [lo_s, hi_s],
            "cagr_mean": float(np.nanmean(cagrs)), "cagr_ci": [lo_c, hi_c],
            "p_sharpe_le_0": float(np.mean(sharpes <= 0)),
            "pass": bool(lo_s > 0)}


# ---------------------------------------------------------------- 检验 6：IS/OOS
def is_oos(returns: pd.Series, *, split: str = "2020-12-31") -> dict:
    """IS/OOS 落差：默认 2017-2020 vs 2021-2024；判据是 OOS≥IS 的 50%（不是绝对落差）。"""
    is_part = returns[returns.index <= split]
    oos_part = returns[returns.index > split]
    if len(is_part) < 3 or len(oos_part) < 3:
        return {"is": stats_block(is_part), "oos": stats_block(oos_part),
                "cagr_ratio": float("nan"), "sharpe_ratio": float("nan"), "pass": False,
                "note": "样本不足，无法切分"}
    a = stats_block(is_part)
    b = stats_block(oos_part)
    cagr_a, cagr_b = a.get("cagr"), b.get("cagr")
    sharpe_a, sharpe_b = a.get("sharpe"), b.get("sharpe")
    ratio = (cagr_b / cagr_a) if cagr_a and abs(cagr_a) > 1e-9 else float("nan")
    return {"is": a, "oos": b, "cagr_ratio": float(ratio),
            "sharpe_ratio": float(sharpe_b / sharpe_a) if sharpe_a else float("nan"),
            "pass": bool(ratio >= 0.5)}


# ---------------------------------------------------------------- 检验 2：PBO
def _config_monthly() -> pd.DataFrame:
    """收集同区间、同成本口径的配置月收益矩阵（用于 CSCV）。"""
    paths: list[tuple[str, Path]] = []
    for spec in ("P", "Q", "QN", "M"):
        paths.append((f"spec-{spec}", STAGE3 / f"daily-{spec}-default.csv"))
    for name in ("legacy", "legacy5k", "legacy5k_slip", "base", "buf60", "buf60_slip",
                 "vt25", "ddb", "crisis", "gradient", "crisis5k", "gradient5k"):
        paths.append((f"risk-{name}", MAIN_RISK / f"daily-{name}.csv"))
    for name in ("A", "B", "C", "C15", "C25"):
        paths.append((f"left-{name}", LEFTSIDE / f"daily-{name}.csv"))
    series = {}
    for label, path in paths:
        if path.is_file():
            series[label] = monthly_returns(load_daily(path))
    frame = pd.DataFrame(series).sort_index()
    return frame.dropna(how="any")


def pbo_cscv(returns: pd.DataFrame, *, splits: int = 16) -> dict:
    """Bailey et al. (2016) CSCV。块长用 ``np.array_split`` 均分（96 个月能被 16 整除时为 6）。"""
    values = returns.to_numpy(dtype=float)
    t, n = values.shape
    chunks = np.array_split(np.arange(t), splits)
    sums = np.array([values[idx].sum(axis=0) for idx in chunks])          # (S, N)
    sums2 = np.array([(values[idx] ** 2).sum(axis=0) for idx in chunks])  # (S, N)
    sizes = np.array([len(idx) for idx in chunks], dtype=float)
    half = splits // 2
    omegas: list[float] = []
    for combo in itertools.combinations(range(splits), half):
        mask = np.zeros(splits, dtype=bool)
        mask[list(combo)] = True
        n_is = float(sizes[mask].sum())
        n_oos = float(sizes[~mask].sum())
        is_sr = np.array([_sharpe_from_sums(n_is, sums[mask, j].sum(), sums2[mask, j].sum())
                          for j in range(n)])
        oos_sr = np.array([_sharpe_from_sums(n_oos, sums[~mask, j].sum(), sums2[~mask, j].sum())
                           for j in range(n)])
        if not np.isfinite(is_sr).any() or not np.isfinite(oos_sr).all():
            continue
        best = int(np.nanargmax(is_sr))
        rank = int(pd.Series(oos_sr).rank(method="average").iloc[best])
        omegas.append(rank / (n + 1))
    omegas_arr = np.array(omegas)
    return {"splits": splits, "combinations": int(len(omegas_arr)), "configs": int(n),
            "months": int(t), "block_sizes": sizes.astype(int).tolist(),
            "pbo": float(np.mean(omegas_arr < 0.5)),
            "omega_median": float(np.median(omegas_arr)),
            "pass": bool(np.mean(omegas_arr < 0.5) <= 0.5)}


# ---------------------------------------------------------------- 主入口（轻量部分）
def run_fast(*, daily_path=None, tag: str = "") -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    daily_file = Path(daily_path) if daily_path is not None else MAIN_DAILY
    equity = load_daily(daily_file)
    daily = to_returns(equity)
    monthly = monthly_returns(equity)
    ledger = build_ledger()
    e_noop = main_is_e_noop() if daily_path is None else {"note": "v2 版本不适用（结构已变）"}

    result: dict = {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "main_spec": {"daily_file": str(daily_file), "e_layer_noop": e_noop,
                      "window": [str(daily.index.min().date()), str(daily.index.max().date())],
                      "days": int(len(daily)),
                      "stats": stats_block(daily)},
        "ledger": ledger["N"],
        "dsr": {key: deflated_sharpe(daily, n) for key, n in ledger["N"].items()},
        "bootstrap": {"main_monthly": stationary_bootstrap(monthly)},
        "is_oos": is_oos(daily),
    }

    configs = _config_monthly()
    configs.to_csv(ROOT / "pbo-monthly-returns.csv", encoding="utf-8-sig")
    result["pbo"] = pbo_cscv(configs)

    # 历史样本补充（E 层在 2008-2016 有真实触发，故用 daily-E-history）
    if HIST_DAILY.is_file():
        hist_equity = load_daily(HIST_DAILY)
        hist_daily = to_returns(hist_equity)
        hist_monthly = monthly_returns(hist_equity)
        result["history"] = {
            "daily_file": str(HIST_DAILY),
            "window": [str(hist_daily.index.min().date()), str(hist_daily.index.max().date())],
            "stats": stats_block(hist_daily),
            "bootstrap": stationary_bootstrap(hist_monthly),
            "is_oos": is_oos(hist_daily, split="2012-12-31"),
        }

    (ROOT / f"tests-fast{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k in ("ledger", "dsr", "is_oos")},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"[fast{tag}] {time.time() - started:.1f}s", flush=True)
    return result


# ---------------------------------------------------------------- 组合层准备（重）
def prepare(sample: str = "main", *, bars: dict | None = None, start=None, end=None,
            cutoff=None, alt_kwargs: dict | None = None, sector_cap: float | None = None,
            fund_extra: tuple = ()) -> dict:
    """构建面板/因子/合格集/打分/基准；主样本与历史样本同构。

    ``bars`` 等可选参数供 paper 的 live 模式注入「当日已刷新」的行情面板；
    不传时行为与既有版本完全一致。
    """
    from quant.factors import growth, growth_alt
    from quant.factors.panel import build_panel
    from scripts import growth_leftside as gl
    from scripts import growth_risk as gr
    from scripts.factor_market import FIELDS, load_delisting_events
    from scripts.growth_backtest import (SPECS, composite_score, month_ends, sector_panel)
    from scripts.redlow_study import (LAKE, START, END, board_exclude, build_block,
                                      load_bars, pool_benchmark)

    t0 = time.time()
    live = bars is not None
    if sample == "history":
        bars = gl.load_history_bars()
        start, end, cutoff = gl.HIST_START, gl.HIST_END, gl.HIST_CUTOFF
        alt_kwargs = {"cutoff": gl.HIST_CUTOFF, "strict": False}
        fund_extra = (gl.HIST / "fundamentals",)
        sector_cap = 1.0
    elif not live:
        bars, _ = load_bars()
        start, end, cutoff = START, END, pd.Timestamp("2024-12-31")
        alt_kwargs = {}
        fund_extra = ()
        sector_cap = 0.30
    else:
        start = pd.Timestamp(start) if start is not None else pd.Timestamp("2023-01-01")
        end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
        cutoff = pd.Timestamp(cutoff) if cutoff is not None else end
        alt_kwargs = dict(alt_kwargs or {})
        sector_cap = 1.0 if sector_cap is None else sector_cap
    for symbol in board_exclude(bars):
        bars.pop(symbol, None)
    panel = build_panel(bars, fields=FIELDS)
    if not live:
        _assert_no_holdout(panel.dates)

    block = build_block(panel).reindex(index=panel.dates, columns=panel.symbols).fillna(True).astype(bool)
    facts = growth.GrowthFactors(panel, cutoff=cutoff, extra_dirs=fund_extra)
    eligible = growth.eligible_mask(facts, block_mask=block)
    marks = month_ends(panel.dates, start, end)
    alt = growth_alt.alt_factors(panel, **alt_kwargs)
    sector = None if sample == "history" else sector_panel(marks, list(panel.symbols))
    blocks = SPECS["QN"]["blocks"]
    score = composite_score(panel, facts, alt, blocks, eligible, marks)
    if live:
        delisting = None
        bench_equity = None
        bt_bars = {s: data.loc[start:end] for s, data in bars.items() if len(data)}
    else:
        delisting = (gl.history_delistings(panel, end) if sample == "history"
                     else load_delisting_events(LAKE))
        bench_equity = pool_benchmark(panel, block, delisting, cash=1_000_000.0)
        bt_bars = {s: data.loc[start:end] for s, data in bars.items() if len(data)}
    print(f"[prepare:{sample}] panel {panel.close.shape} marks {len(marks)} "
          f"eligible {int(eligible.to_numpy().sum(axis=1).mean())}/日 {time.time() - t0:.0f}s", flush=True)
    return {"sample": sample, "panel": panel, "bars": bars, "bt_bars": bt_bars, "block": block,
            "facts": facts, "eligible": eligible, "marks": marks, "alt": alt, "sector": sector,
            "sector_cap": sector_cap, "score": score, "delisting": delisting,
            "bench": bench_equity, "start": start, "end": end, "blocks": blocks}


def _zrank(values: np.ndarray) -> np.ndarray:
    """截面 rank 后 z-score；NaN 保持 NaN。"""
    series = pd.Series(values, dtype="float64")
    ranked = series.rank(pct=True)
    z = (ranked - ranked.mean()) / ranked.std(ddof=0)
    return np.asarray(z, dtype="float64")


def forward_month_returns(close: pd.DataFrame, marks: pd.DatetimeIndex) -> pd.DataFrame:
    """月末收盘 → 下一个月末收盘的收益（与阶段 C 的 IC 口径一致）。"""
    marks = pd.DatetimeIndex(sorted(set(marks) & set(close.index)))
    prices = close.loc[marks]
    fwd = prices.shift(-1) / prices - 1
    return fwd.iloc[:-1]


def permutation_test(score: pd.DataFrame, close: pd.DataFrame, marks: pd.DatetimeIndex,
                     eligible: pd.DataFrame, *, reps: int = 1000,
                     seed: int = 20261021) -> dict:
    """截面「股票-因子值」配对随机置换，得到均值 IC 的零分布。"""
    rng = np.random.default_rng(seed)
    fwd = forward_month_returns(close, marks)
    xs, ys, used = [], [], []
    for day in fwd.index:
        if day not in score.index:
            continue
        s = score.loc[day].reindex(close.columns).to_numpy(dtype="float64")
        f = fwd.loc[day].reindex(close.columns).to_numpy(dtype="float64")
        mask = np.isfinite(s) & np.isfinite(f)
        if mask.sum() < 50:
            continue
        xs.append(_zrank(s[mask]))
        ys.append(_zrank(f[mask]))
        used.append(day)
    if len(xs) < 12:
        return {"error": "有效月份不足", "months": len(xs)}

    def mean_ic(y_arrays) -> float:
        vals = [float(np.nanmean(x * y)) for x, y in zip(xs, y_arrays)]
        return float(np.nanmean(vals))

    actual = mean_ic(ys)
    null = np.empty(reps)
    lengths = [len(y) for y in ys]
    for i in range(reps):
        shuffled = [rng.permutation(y) for y in ys]
        null[i] = mean_ic(shuffled)
    p = float((1 + np.sum(null >= actual)) / (1 + reps))
    return {"months": len(used), "reps": reps, "actual_mean_ic": actual,
            "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "null_p99": float(np.percentile(null, 99)),
            "p_value": p, "pass": bool(actual >= np.percentile(null, 95)),
            "mean_names": float(np.mean(lengths))}


def analytic_monthly(score: pd.DataFrame, close: pd.DataFrame, marks: pd.DatetimeIndex, *,
                     top_k: int = 50, gross: float = 0.95, cost_bp: float = 14.2) -> pd.Series:
    """解析式月频 top-k 等权（用于 WFA 的 IS/OOS 同口径比较，含往返成本）。"""
    fwd = forward_month_returns(close, marks)
    rows = {}
    prev = set()
    for day in fwd.index:
        if day not in score.index:
            continue
        s = score.loc[day].reindex(close.columns)
        f = fwd.loc[day].reindex(close.columns)
        valid = s.notna() & f.notna()
        if valid.sum() < top_k:
            continue
        picked = s[valid].nlargest(top_k).index
        weight = gross / top_k
        gross_ret = float((f[picked] * weight).sum())
        turnover = len(set(picked) - prev) * weight
        cost = turnover * cost_bp / 1e4
        rows[day] = gross_ret - cost
        prev = set(picked)
    series = pd.Series(rows).sort_index()
    return series


def fit_block_weights(block_scores: dict[str, pd.DataFrame], close: pd.DataFrame,
                      marks: pd.DatetimeIndex, *, window: int = 36) -> pd.DataFrame:
    """滚动 window 个月拟合块权重：w_b ∝ max(0, 过去 window 月块打分的均值 rank IC)。"""
    fwd = forward_month_returns(close, marks)
    names = list(block_scores)
    ic_by_block = {}
    for name in names:
        frame = block_scores[name]
        ics = []
        for day in fwd.index:
            if day not in frame.index:
                continue
            s = frame.loc[day].reindex(close.columns).to_numpy(dtype="float64")
            f = fwd.loc[day].reindex(close.columns).to_numpy(dtype="float64")
            mask = np.isfinite(s) & np.isfinite(f)
            if mask.sum() < 50:
                continue
            ics.append((day, float(np.corrcoef(pd.Series(s[mask]).rank(),
                                               pd.Series(f[mask]).rank())[0, 1])))
        ic_by_block[name] = pd.Series(dict(ics)).sort_index()
    weights = {}
    for i, day in enumerate(marks):
        start = day - pd.DateOffset(months=window)
        row = {}
        for name in names:
            series = ic_by_block[name]
            window_ic = series[(series.index >= start) & (series.index < day)]
            row[name] = max(0.0, float(window_ic.mean())) if len(window_ic) >= 12 else 0.0
        total = sum(row.values())
        if total <= 0:
            row = {name: 1.0 / len(names) for name in names}
        else:
            row = {name: value / total for name, value in row.items()}
        weights[day] = row
    return pd.DataFrame(weights).T


def build_tv_score(prepared: dict, block_weights: pd.DataFrame) -> pd.DataFrame:
    """用逐月块权重拼出时变打分矩阵（只保留 marks 行）。"""
    from scripts.growth_backtest import composite_score
    blocks = prepared["blocks"]
    single = {}
    for name, members in blocks.items():
        single[name] = composite_score(prepared["panel"], prepared["facts"], prepared["alt"],
                                       {name: members}, prepared["eligible"], prepared["marks"],
                                       min_names=20)
    rows = {}
    for day in block_weights.index:
        if day not in prepared["marks"]:
            continue
        total = None
        for name in blocks:
            frame = single[name]
            if day not in frame.index:
                continue
            part = frame.loc[day] * float(block_weights.loc[day, name])
            total = part if total is None else total.add(part, fill_value=0.0)
        if total is not None:
            rows[day] = total
    return pd.DataFrame(rows).T.reindex(columns=prepared["panel"].symbols)


def run_engine(prepared: dict, weights: pd.DataFrame, *, name: str, min_trade: float = 5000.0,
               slip_map: dict | None = None, baseline_bps: float = 2.0) -> dict:
    """真实 Engine 回测（与主规格同口径：分档滑点、<5000 元单忽略、haircut30）。"""
    from quant.backtest.engine import Engine
    from quant.config import BacktestConfig, CostModel
    from scripts import growth_risk as gr

    config = dict(start=str(prepared["start"].date()), end=str(prepared["end"].date()),
                  initial_cash=1_000_000.0, benchmark=None,
                  costs=CostModel(slippage_bps=baseline_bps))
    gr.install_broker(min_trade=min_trade)
    gr.RiskBroker.MIN_TRADE = min_trade
    gr.RiskBroker.SLIP = slip_map or {}
    gr.RiskBroker.BASE_BPS = baseline_bps
    result = Engine(config=BacktestConfig(**config), benchmark_prices=None,
                    delisting=prepared["delisting"], delisting_mode="haircut30").run(
        prepared["bt_bars"], weights, strategy_name=name)
    row = gr.evaluate(result, prepared["bench"], cash=1_000_000.0, slip_label="分档")
    row["name"] = name
    row["mdd_date"] = str(gr.drawdown_series(result.equity).idxmin().date())
    return {"row": row, "daily": result.daily}


def sensitivity_scan(prepared: dict, *, tag: str = "main", runs: list[dict] | None = None) -> pd.DataFrame:
    """参数敏感性：邻域扫描（不据此改规格，只看是否孤立尖峰）。"""
    from scripts import growth_risk as gr
    BufferTopK = gr.BufferTopK

    marks = prepared["marks"]
    adv20, illiq20 = gr.liquidity_panels(prepared["panel"])
    slip_maps, tier_rows = gr.tier_map(marks, prepared["eligible"], adv20, illiq20)
    tier_slip: dict[str, float] = {}
    for day in marks:
        tier_slip.update(slip_maps.get(day, {}))

    base = {"top_k": 50, "max_weight": 0.04, "sector_cap": prepared["sector_cap"],
            "min_trade": 5000.0, "slip": "tiered"}
    if runs is None:
        runs = []
        for value in (30, 40, 60, 70):
            runs.append({**base, "top_k": value})
        for value in (0.03, 0.05):
            runs.append({**base, "max_weight": value})
        for value in (0.25, 1.0):
            if prepared["sample"] == "main":
                runs.append({**base, "sector_cap": value})
        for value in (0.0, 10000.0):
            runs.append({**base, "min_trade": value})
        runs.append({**base, "slip": "fixed"})

    rows = []
    base_weights = None
    for spec in [base, *runs]:
        weights = _build_weights(prepared, spec)
        if spec is base:
            base_weights = weights
        use_slip = tier_slip if spec["slip"] == "tiered" else None
        outcome = run_engine(prepared, weights, name=f"sens-{spec}", min_trade=spec["min_trade"],
                             slip_map=use_slip)
        row = dict(spec)
        row.update({k: outcome["row"].get(k) for k in
                    ("total_return", "cagr", "sharpe", "max_drawdown", "calmar",
                     "ann_turnover_one_way", "fees_pct_of_equity")})
        row["sample"] = prepared["sample"]
        rows.append(row)
        print(f"[sens] {spec} → cagr {row['cagr']:.2%} sharpe {row['sharpe']:.3f} "
              f"mdd {row['max_drawdown']:.2%}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / f"sensitivity-{tag}.csv", index=False, encoding="utf-8-sig")
    return frame


def wfa(prepared: dict, *, first_year: int = 2020, tag: str = "main") -> dict:
    """Walk-forward：滚动 36 月拟合块权重 → 下一年验证（引擎 + 解析式双口径）。"""
    from scripts import growth_risk as gr
    BufferTopK = gr.BufferTopK

    blocks = prepared["blocks"]
    single = {name: composite_score_single(prepared, name) for name in blocks}
    weights = fit_block_weights(single, prepared["panel"].close, prepared["marks"], window=36)
    weights.to_csv(ROOT / f"wfa-block-weights-{tag}.csv", encoding="utf-8-sig")
    tv_score = build_tv_score(prepared, weights)

    valid_marks = prepared["marks"][prepared["marks"].year >= first_year]
    tv = tv_score.loc[tv_score.index >= pd.Timestamp(f"{first_year}-01-01")]
    adv20, illiq20 = gr.liquidity_panels(prepared["panel"])
    slip_maps, _ = gr.tier_map(valid_marks, prepared["eligible"], adv20, illiq20)
    tier_slip: dict[str, float] = {}
    for day in valid_marks:
        tier_slip.update(slip_maps.get(day, {}))
    strategy_weights = _build_weights(prepared, {"score": tv})
    engine_prep = dict(prepared)
    engine_prep["start"] = pd.Timestamp(f"{first_year}-01-01")
    outcome = run_engine(engine_prep, strategy_weights, name=f"wfa-{tag}", slip_map=tier_slip)
    daily = outcome["daily"]
    if "date" in daily.columns:
        equity = daily.set_index(pd.to_datetime(daily["date"]))["equity"].astype(float)
    else:
        equity = daily["equity"].astype(float).copy()
        equity.index = pd.to_datetime(equity.index)
    yearly = {}
    for year, part in equity.groupby(equity.index.year):
        if int(year) < first_year:
            continue
        returns = part.pct_change().dropna()
        yearly[int(year)] = stats_block(returns)
    # 解析式同口径：拟合窗口（IS，36 个月）与验证年（OOS）
    analytic = analytic_monthly(tv_score, prepared["panel"].close, prepared["marks"])
    rows = []
    for year in sorted(yearly):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        oos = analytic[(analytic.index >= year_start) & (analytic.index <= year_end)]
        is_win = analytic[(analytic.index >= year_start - pd.DateOffset(years=3)) &
                          (analytic.index < year_start)]
        oos_ret = float((1 + oos).prod() - 1) if len(oos) else float("nan")
        is_ret = float((1 + is_win).prod() - 1) if len(is_win) else float("nan")
        is_ann = float((1 + is_ret) ** (12 / len(is_win)) - 1) if len(is_win) >= 12 else float("nan")
        wfe = (oos_ret / is_ann) if (is_ann > 0 and np.isfinite(is_ann)) else float("nan")
        rows.append({"year": year, "oos_engine": yearly[year].get("total"),
                     "oos_engine_sharpe": yearly[year].get("sharpe"),
                     "oos_analytic": oos_ret, "is_analytic": is_ret, "is_annualized": is_ann,
                     "wfe": wfe})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / f"wfa-{tag}.csv", index=False, encoding="utf-8-sig")
    positive = int((table["oos_engine"] > 0).sum())
    valid_wfe = table["wfe"].dropna()
    return {"first_year": first_year, "years": int(len(table)), "positive_years": positive,
            "table": table.to_dict(orient="records"),
            "median_wfe": float(valid_wfe.median()) if len(valid_wfe) else float("nan"),
            "wfe_valid_windows": int(len(valid_wfe)),
            "pass": bool(positive > len(table) / 2)}


def composite_score_single(prepared: dict, name: str) -> pd.DataFrame:
    from scripts.growth_backtest import composite_score
    members = {name: prepared["blocks"][name]}
    return composite_score(prepared["panel"], prepared["facts"], prepared["alt"], members,
                           prepared["eligible"], prepared["marks"], min_names=20)


def run_heavy(sample: str = "main") -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    prepared = prepare(sample)
    out: dict = {"sample": sample, "generated": pd.Timestamp.now("UTC").isoformat()}
    path = ROOT / f"tests-heavy-{sample}.json"

    def flush() -> None:
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    try:
        out["permutation"] = permutation_test(prepared["score"], prepared["panel"].close,
                                              prepared["marks"], prepared["eligible"])
        print(f"[perm] ic={out['permutation'].get('actual_mean_ic'):.4f} "
              f"p={out['permutation'].get('p_value'):.4f}", flush=True)
    except Exception as exc:                     # noqa: BLE001 - 单项失败不拖垮整轮
        out["permutation"] = {"error": repr(exc)}
    flush()

    try:
        table = sensitivity_scan(prepared, tag=sample)
        out["sensitivity"] = {"rows": table.to_dict(orient="records")}
    except Exception as exc:                     # noqa: BLE001
        out["sensitivity"] = {"error": repr(exc)}
    flush()

    try:
        out["wfa"] = wfa(prepared, tag=sample)
    except Exception as exc:                     # noqa: BLE001
        out["wfa"] = {"error": repr(exc)}
    flush()
    print(f"[heavy:{sample}] done", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="阶段 4 稳健性检验")
    parser.add_argument("--fast", action="store_true", help="跑轻量检验（DSR/bootstrap/IS-OOS/PBO/台账）")
    parser.add_argument("--heavy", choices=["main", "history"], help="跑重检验（置换/敏感性/WFA）")
    args = parser.parse_args()
    if args.fast:
        run_fast()
    if args.heavy:
        run_heavy(args.heavy)
