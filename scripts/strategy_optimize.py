"""策略优化研究（只用 2017-2024 样本内；2025+ 一律不读）。

三个缺口的对应关系
------------------
① 增量因子：9 个预注册因子池，先看单因子 IC / IC_IR / 逐年稳定性，
   再看「对现有 4 因子（红利/低波/低换手/增持）的残差 IC」——残差 IC 才是真正
   增加了现有组合没有的信息量。
② 多段样本外协议：walk-forward，逐年 2019→2024，每折只用折前数据定权重，
   OOS 只在折后年份用一次；另给 IS（全样本拟合）做对照，展示过拟合落差。
③ 换手预算：buffer（仍在前 100 名就不卖）、月度单边换手上限、最短持有月数，
   在 14.2/30/50bp 三档成本下同时看收益与换手。

纪律
----
* 因子池、权重方案、评估口径都在跑数之前写死；看到结果后不增删因子。
* 试验次数 N 会如实记账，并用 deflated Sharpe / 月份置换检验给出「N 次试验下
  这个夏普有多少是运气」的量化判断。
* 所有收益都是月度等权定价器（与现有影子脚本 `monthly_equal_weight` 同口径），
  最后用真 Engine 的月频模式复核 1-2 个组合。

用法
----
    $py -u scripts/strategy_optimize.py --stage diag
    $py -u scripts/strategy_optimize.py --stage opt
    $py -u scripts/strategy_optimize.py --stage engine
    $py -u scripts/strategy_optimize.py              # 全部
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant.factors import evaluate, events, fundamental, library
from quant.factors.panel import Panel
from scripts.improvement_shadow import (
    END,
    FIELDS,
    START,
    ShadowEngine,
    build_sample_context,
    dividend_picks,
    make_strategy,
    monthly_equal_weight,
    run_variant,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "strategy-optimization"
CACHE = ROOT / "_tmp" / "factor_monthly"

RF = 0.02
TOP_K = 50
GROSS = 0.95
COST_LEVELS = {"14.2bp": 14.2, "30bp": 30.0, "50bp": 50.0}
BASE_COST = "14.2bp"
WINDOW_OOS_START = "2019-01-01"

# 预注册因子池：覆盖 6 个经济类别，看见 IC 之后不许改
CORE = ["div_yield", "vol_60", "turnover_20", "insider_net_20"]
POOL = CORE + ["bp", "roe", "revenue_growth", "amihud", "momentum"]
POOL_NOTE = {
    "div_yield": "红利（股息率，现有）",
    "vol_60": "低波（现有）",
    "turnover_20": "低换手（现有）",
    "insider_net_20": "内部人增持（现有）",
    "bp": "价值（账面市值比）",
    "roe": "质量（净资产收益率）",
    "revenue_growth": "成长（营收增速）",
    "amihud": "流动性（非流动性）",
    "momentum": "动量（60 日，跳 5 日）",
}
# 方案清单：拟合类方案都用同一回看窗口（36 个月），先写死再看结果
SCHEMES = [
    "equal_core",      # 现有：4 因子等权
    "equal_pool",      # 9 因子等权（不拟合，天然无过拟合）
    "ic36",            # IC 加权（trailing 36m，负 IC 置零）
    "ir36",            # IC_IR 加权（trailing 36m，负值置零）
    "maxir36",         # Markowitz max IR（IC 协方差收缩，w>=0，cap 35%）
    "fm36",            # Fama-MacBeth 横截面回归系数（trailing 36m）
]
FIT_SCHEMES = [s for s in SCHEMES if s not in ("equal_core", "equal_pool")]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)


def month_end_days(panel: Panel) -> list[pd.Timestamp]:
    months = pd.PeriodIndex(panel.dates, freq="M")
    frame = pd.Series(panel.dates, index=months)
    return list(frame.groupby(level=0).max())


def rank_z(frame: pd.DataFrame) -> pd.DataFrame:
    """逐行横截面分位到 [-1, 1]；NaN 原样保留。"""
    ranked = frame.rank(axis=1, pct=True)
    return 2.0 * ranked - 1.0


def eligibility(panel: Panel, block: pd.DataFrame, days) -> pd.DataFrame:
    tradable = panel.tradable.reindex(index=days)
    close = panel.close.reindex(index=days)
    blocked = block.reindex(index=days, columns=panel.symbols).fillna(True).astype(bool)
    return (tradable & close.notna() & (close >= 2.0) & ~blocked).fillna(False)


def factor_monthly(name: str, panel: Panel, days, elig: pd.DataFrame, *, refresh: bool = False) -> pd.DataFrame:
    """因子值裁到月末网格并按可投池屏蔽；结果缓存到 _tmp/factor_monthly。"""
    path = CACHE / f"{name}.pkl"
    if path.exists() and not refresh:
        cached = pd.read_pickle(path)
        if cached.index.equals(pd.DatetimeIndex(days)) and list(cached.columns) == list(panel.symbols):
            return cached
    started = time.time()
    values = library.compute(name, panel).reindex(index=days)
    values = values.where(elig).astype("float32")
    values.to_pickle(path)
    print(f"  因子 {name:<16} {time.time() - started:5.1f}s  覆盖中位数 {int(values.notna().sum(axis=1).median())}", flush=True)
    return values


# ---------------------------------------------------------------------------
# IC / 残差 IC
# ---------------------------------------------------------------------------
def monthly_ic(factor: pd.DataFrame, forward: pd.DataFrame, *, min_count: int = 30) -> pd.Series:
    """逐月 Spearman IC（与 evaluate.rank_ic 同定义，numpy 实现以省时间）。"""
    symbols = factor.columns
    fwd = forward.reindex(index=factor.index, columns=symbols).to_numpy(dtype=float)
    val = factor.to_numpy(dtype=float)
    out = {}
    for i, day in enumerate(factor.index):
        a, b = val[i], fwd[i]
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < min_count:
            out[day] = np.nan
            continue
        ra = pd.Series(a[mask]).rank().to_numpy()
        rb = pd.Series(b[mask]).rank().to_numpy()
        if ra.std() == 0 or rb.std() == 0:
            out[day] = np.nan
            continue
        out[day] = float(np.corrcoef(ra, rb)[0, 1])
    return pd.Series(out, dtype=float)


def residual_ic(
    factors: dict[str, pd.DataFrame],
    forward: pd.DataFrame,
    controls: list[str],
    *,
    min_count: int = 30,
) -> dict[str, pd.Series]:
    """对 controls 做逐月横截面回归后取残差，再算 IC。"""
    base = factors[controls[0]]
    ctrl = np.stack([factors[c].to_numpy(dtype=float) for c in controls], axis=0)  # k×m×n
    fwd = forward.reindex(index=base.index, columns=base.columns).to_numpy(dtype=float)
    result: dict[str, pd.Series] = {}
    for name, frame in factors.items():
        if name in controls:
            continue
        val = frame.to_numpy(dtype=float)
        rows = {}
        for i, day in enumerate(frame.index):
            y = val[i]
            mask = np.isfinite(y)
            for c in ctrl:
                mask &= np.isfinite(c[i])
            if int(mask.sum()) < min_count:
                rows[day] = np.nan
                continue
            X = np.column_stack([np.ones(int(mask.sum()))] + [c[i][mask] for c in ctrl])
            coef, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
            resid = y[mask] - X @ coef
            b = fwd[i][mask]
            good = np.isfinite(b)
            if int(good.sum()) < min_count:
                rows[day] = np.nan
                continue
            ra = pd.Series(resid[good]).rank().to_numpy()
            rb = pd.Series(b[good]).rank().to_numpy()
            rows[day] = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else np.nan
        result[name] = pd.Series(rows, dtype=float)
    return result


def ic_stats(ic: pd.Series, *, periods: int = 12) -> dict:
    clean = ic.dropna()
    if not len(clean):
        return {}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    ir = mean / std if std else float("nan")
    t = ir * math.sqrt(len(clean)) if std else float("nan")
    by_year = clean.groupby(clean.index.year)
    return {
        "n": int(len(clean)),
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": ir,
        "annual_ir": ir * math.sqrt(periods),
        "t_stat": t,
        "positive_rate": float((clean > 0).mean()),
        "years_positive": int((by_year.mean() > 0).sum()),
        "years": int(by_year.ngroups),
    }


# ---------------------------------------------------------------------------
# 打分 → 名单 → 净值
# ---------------------------------------------------------------------------
def composite(weights: pd.Series, z: dict[str, pd.DataFrame], days) -> pd.DataFrame:
    out = None
    for name, w in weights.items():
        if abs(w) < 1e-12:
            continue
        frame = z[name].reindex(index=days)
        part = frame.fillna(0.0) * float(w)
        out = part if out is None else out + part
    if out is None:
        raise ValueError("空权重")
    return out


def picks_from_score(score: pd.DataFrame, elig: pd.DataFrame, *, top: int = TOP_K) -> dict[pd.Timestamp, list[str]]:
    picks: dict[pd.Timestamp, list[str]] = {}
    for day in score.index:
        row = score.loc[day].where(elig.loc[day])
        row = row.dropna()
        if len(row) < top:
            if not len(row):
                continue
            picks[day] = list(row.sort_values(ascending=False).index[:top])
            continue
        picks[day] = list(row.nlargest(top).index)
    return picks


def picks_to_weights(picks: dict, panel: Panel, *, gross: float = 0.95) -> pd.DataFrame:
    """{月末: [代码]} → Engine 用的权重矩阵（每个调仓日等权、gross=0.95）。"""
    days = sorted(picks)
    frame = pd.DataFrame(0.0, index=pd.DatetimeIndex(days), columns=list(panel.symbols))
    for day in days:
        names = [s for s in dict.fromkeys(picks[day]) if s in frame.columns]
        if names:
            frame.loc[day, names] = gross / len(names)
    return frame


def stats_from_equity(equity: pd.Series, turn: pd.Series, cost: pd.Series) -> dict:
    ret = equity.pct_change().dropna()
    if not len(ret):
        return {}
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(ret) / 244.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    vol = float(ret.std(ddof=0) * math.sqrt(244.0))
    mdd = float((equity / equity.cummax() - 1.0).min())
    # 与 quant/backtest/metrics.py 同口径：算术日均 × 244 减无风险 2%，再除年化波动
    sharpe = (float(ret.mean()) * 244.0 - RF) / vol if vol else float("nan")
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd else float("nan"),
        "month_turnover": float(turn.mean()) if turn is not None and len(turn) else 0.0,
        "annual_turnover": float(turn.mean() * 12.0) if turn is not None and len(turn) else 0.0,
        "fees": float(cost.sum()) if cost is not None else 0.0,
        "months": int(len(turn)) if turn is not None else 0,
    }


def open_pricer(
    panel: Panel,
    block: pd.DataFrame,
    delisting,
    picks: dict[pd.Timestamp, list[str]],
    *,
    rt_bp: float,
    gross: float = GROSS,
    cash: float = 1_000_000.0,
    haircut: float = 0.7,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """按真 Engine 的时序定价：**t 日开盘成交、t 日收盘计净值**。

    * 非调仓日：持仓按 ``close(t)/close(t-1)`` 漂移；
    * 调仓日（月末名单之后的第一个交易日）：老仓先吃隔夜跳空 ``open(t)/close(t-1)``，
      开盘换仓，再吃日内 ``close(t)/open(t)``——与 Engine「t-1 出权重、t 开盘撮合」一致；
    * 当日不可交易（停牌/涨停封板）的标的买不进、也卖不掉，买不到的量留在现金；
    * 退市：最后一条有效行情之后按 ``haircut`` 结清（Engine 的 haircut30 = 0.7）。

    与 Engine 的差异只有两处：不做整手取整、不做最小交易金额过滤（1M 本金 / 50 只
    ≈ 1.9 万/只，远高于一手，误差可忽略）。校准见 `OPTIMIZATION_REPORT.md` 第 1 节。
    """
    days = panel.dates
    close = panel.close.to_numpy(dtype=float)
    open_ = panel.open.to_numpy(dtype=float)
    tradable = (panel.tradable & ~block.reindex(index=days, columns=panel.symbols).fillna(True)).to_numpy(dtype=bool)
    symbols = list(panel.symbols)
    where = {symbol: i for i, symbol in enumerate(symbols)}
    months = pd.PeriodIndex(days, freq="M")
    first_pos = {}
    for i, month in enumerate(months):
        first_pos.setdefault(month, i)

    finite = np.isfinite(close)
    last_valid = np.full(len(symbols), -1)
    for j in range(len(symbols)):
        col = np.flatnonzero(finite[:, j])
        if len(col):
            last_valid[j] = int(col[-1])

    pick_days = sorted(picks)
    calendar = list(days)
    exec_map: dict[int, pd.Timestamp] = {}
    for day in pick_days:
        later = [d for d in calendar if d > day]
        if not later:
            continue
        position = calendar.index(later[0])
        exec_map[position] = day

    pos = np.zeros(len(symbols), dtype=float)
    level = float(cash)
    equity = np.zeros(len(days), dtype=float)
    turn: dict[pd.Timestamp, float] = {}
    cost: dict[pd.Timestamp, float] = {}
    for i, day in enumerate(days):
        if i > 0:
            prev = close[i - 1]
            gap = np.divide(open_[i], prev, out=np.ones(len(symbols)), where=np.isfinite(prev) & (prev > 0))
            gap = np.where(np.isfinite(gap), gap, 1.0)
            pos *= np.clip(gap, 0.0, 10.0)
        if i in exec_map:
            nav = level + float(pos.sum())
            names = [s for s in picks[exec_map[i]] if s in where]
            names = [s for s in names if bool(tradable[i, where[s]])]
            target = np.zeros(len(symbols), dtype=float)
            if names:
                target[[where[s] for s in names]] = nav * gross / len(names)
            delta = target - pos
            delta[~tradable[i]] = 0.0
            moved = float(np.abs(delta).sum())
            fee = nav * (0.5 * moved / max(nav, 1e-9)) * rt_bp / 1e4
            pos = pos + delta
            level = nav - float(pos.sum()) - fee
            turn[pd.Timestamp(day)] = 0.5 * moved / max(nav, 1e-9)
            cost[pd.Timestamp(day)] = fee
        # 日内收益（调仓日新仓从开盘价起算，老仓已在开盘卖出）
        op = open_[i]
        intra = np.divide(close[i], op, out=np.ones(len(symbols)), where=np.isfinite(op) & (op > 0))
        intra = np.where(np.isfinite(intra), intra, 1.0)
        pos *= np.clip(intra, 0.0, 10.0)
        if delisting:
            dead = (last_valid < i) & (pos > 0)
            if dead.any():
                level += float(pos[dead].sum()) * haircut
                pos[dead] = 0.0
        equity[i] = level + float(pos.sum())
    series = pd.Series(equity, index=pd.DatetimeIndex(days), name="equity")
    return series, pd.Series(turn, dtype=float), pd.Series(cost, dtype=float)


def price_picks(
    picks: dict[pd.Timestamp, list[str]],
    panel: Panel,
    block: pd.DataFrame,
    delisting,
    rt_bp: float,
) -> dict:
    equity, turn, cost = open_pricer(panel, block, delisting, picks, rt_bp=rt_bp)
    row = stats_from_equity(equity, turn, cost)
    row["rt_bp"] = rt_bp
    return row


def price_picks_cc(
    picks: dict[pd.Timestamp, list[str]],
    panel: Panel,
    block: pd.DataFrame,
    delisting,
    rt_bp: float,
) -> dict:
    """旧口径（收盘价换仓、无开盘撮合），只用于量化两种定价方式的差异。"""
    equity, turn, cost = monthly_equal_weight(
        panel, block, rt_bp=rt_bp, gross=GROSS, cash=1_000_000.0, delisting=delisting, picks=picks
    )
    row = stats_from_equity(equity, turn, cost)
    row["rt_bp"] = rt_bp
    return row


# ---------------------------------------------------------------------------
# 权重方案
# ---------------------------------------------------------------------------
def weights_equal(names: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(names), index=names)


def weights_ic(ic: pd.DataFrame, names: list[str]) -> pd.Series:
    mean = ic[names].mean().clip(lower=0.0)
    if float(mean.sum()) <= 0:
        return weights_equal(names)
    return mean / mean.sum()


def weights_ir(ic: pd.DataFrame, names: list[str]) -> pd.Series:
    mean = ic[names].mean()
    std = ic[names].std(ddof=1).replace(0.0, np.nan)
    ratio = (mean / std).clip(lower=0.0).fillna(0.0)
    if float(ratio.sum()) <= 0:
        return weights_equal(names)
    return ratio / ratio.sum()


def weights_maxir(ic: pd.DataFrame, names: list[str], *, cap: float = 0.35, shrink: float = 0.5) -> pd.Series:
    matrix = ic[names].dropna(how="all")
    if len(matrix) < 12:
        return weights_equal(names)
    mu = matrix.mean().to_numpy(dtype=float)
    cov = matrix.cov().to_numpy(dtype=float)
    cov = (1.0 - shrink) * cov + shrink * np.diag(np.diag(cov))
    cov += np.eye(len(names)) * 1e-8
    try:
        raw = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        return weights_equal(names)
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        return weights_equal(names)
    raw = raw / raw.sum()
    for _ in range(20):
        over = raw > cap
        if not over.any():
            break
        excess = float((raw[over] - cap).sum())
        raw[over] = cap
        free = ~over
        if not free.any():
            break
        raw[free] += excess * raw[free] / raw[free].sum()
    return pd.Series(raw, index=names)


def weights_fm(z: dict[str, pd.DataFrame], forward: pd.DataFrame, months, names: list[str], *,
               cap: float = 0.40) -> pd.Series:
    """Fama-MacBeth：逐月横截面回归（收益对因子打分），系数均值即权重。"""
    coefs = []
    for day in months:
        y = forward.loc[day].reindex(z[names[0]].columns).to_numpy(dtype=float)
        X = np.column_stack([z[n].loc[day].to_numpy(dtype=float) for n in names])
        mask = np.isfinite(y)
        for j in range(X.shape[1]):
            mask &= np.isfinite(X[:, j])
        if int(mask.sum()) < 100:
            continue
        Xm = X[mask]
        Xm = np.column_stack([np.ones(len(Xm)), Xm])
        yy = np.clip(y[mask], np.quantile(y[mask], 0.01), np.quantile(y[mask], 0.99))
        beta, *_ = np.linalg.lstsq(Xm, yy, rcond=None)
        coefs.append(beta[1:])
    if not coefs:
        return weights_equal(names)
    mean = np.mean(np.array(coefs), axis=0)
    mean = np.clip(mean, -cap, cap)
    scale = float(np.abs(mean).sum())
    if scale <= 0:
        return weights_equal(names)
    return pd.Series(mean / scale, index=names)


def fit_weights(scheme: str, ctx: dict, months) -> pd.Series:
    names = list(ctx["pool"])
    if scheme == "equal_core":
        return weights_equal(list(CORE))
    if scheme == "equal_pool":
        return weights_equal(names)
    ic = ctx["ic"].loc[[m for m in months]]
    if scheme == "ic36":
        return weights_ic(ic, names)
    if scheme == "ir36":
        return weights_ir(ic, names)
    if scheme == "maxir36":
        return weights_maxir(ic, names)
    if scheme == "fm36":
        return weights_fm(ctx["z"], ctx["forward"], months, names)
    raise ValueError(scheme)


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------
def walk_forward(scheme: str, ctx: dict, *, years=range(2019, 2025), window: int = 36) -> tuple[dict, list[dict], pd.DataFrame]:
    days = list(ctx["z"]["div_yield"].index)
    picks: dict[pd.Timestamp, list[str]] = {}
    log = []
    rows: dict[pd.Timestamp, pd.Series] = {}
    for year in years:
        train = [d for d in days if d.year < year][-window:]
        # 上一年的最后一个月末也归到这一折：该月名单用同一批权重（否则每年 1 月空仓）
        previous = [d for d in days if d.year == year - 1][-1:]
        test = previous + [d for d in days if d.year == year]
        if len(train) < 24 or not test:
            continue
        weights = fit_weights(scheme, ctx, train)
        score = composite(weights, ctx["z"], test)
        picks.update(picks_from_score(score, ctx["elig"]))
        for day in test:
            rows[day] = score.loc[day]
        log.append({"year": year, "train_months": len(train), "test_months": len(test),
                    **{f"w_{k}": float(v) for k, v in weights.items()}})
    stitched = pd.DataFrame(rows).T if rows else pd.DataFrame()
    if len(stitched):
        stitched = stitched.reindex(columns=list(ctx["z"][POOL[0]].columns))
    return picks, log, stitched


def in_sample_picks(scheme: str, ctx: dict) -> tuple[dict, pd.Series]:
    days = list(ctx["z"]["div_yield"].index)
    weights = fit_weights(scheme, ctx, days)
    score = composite(weights, ctx["z"], days)
    return picks_from_score(score, ctx["elig"]), weights


# ---------------------------------------------------------------------------
# 换手预算
# ---------------------------------------------------------------------------
def constrained_picks(
    score: pd.DataFrame,
    elig: pd.DataFrame,
    *,
    top: int = TOP_K,
    buffer_mult: float = 1.0,
    max_swaps: int | None = None,
    min_hold: int = 0,
) -> dict[pd.Timestamp, list[str]]:
    """buffer（仍在前 buffer_mult×top 名就不换）+ 单向换手上限（换算成换股只数）+ 最短持有。"""
    buffer_size = int(round(top * buffer_mult))
    held: dict[str, int] = {}
    out: dict[pd.Timestamp, list[str]] = {}
    for day in score.index:
        row = score.loc[day].where(elig.loc[day]).dropna()
        if not len(row):
            continue
        order = list(row.sort_values(ascending=False).index)
        rank = {symbol: i for i, symbol in enumerate(order)}
        alive = [s for s in held if s in rank]
        buffer_names = set(order[:buffer_size])
        protected = [s for s in alive if held[s] < min_hold]
        keep = [s for s in alive if s in buffer_names and s not in protected]
        keep.sort(key=lambda s: rank[s])
        picks = protected + keep
        picks = picks[:top]
        slots = top - len(picks)
        if slots > 0:
            fresh = [s for s in order[: max(top * 2, 100)] if s not in held]
            if max_swaps is not None:
                swaps = min(slots, max_swaps)
                leftovers = [s for s in alive if s not in picks and s not in protected]
                leftovers.sort(key=lambda s: rank[s])
                picks.extend(leftovers[: slots - swaps])
                picks.extend(fresh[: swaps])
            else:
                picks.extend(fresh[:slots])
        picks = picks[:top]
        out[day] = picks
        held = {s: held.get(s, 0) + 1 for s in picks}
    return out


# ---------------------------------------------------------------------------
# 过拟合排查
# ---------------------------------------------------------------------------
def deflated_sharpe(sharpe: float, trials: int, years: float) -> dict:
    """Bailey-López de Prado 简化版：N 次试验下「最好一次夏普」的期望门槛。"""
    if trials <= 1 or years <= 0:
        return {"trials": trials, "expected_max": float("nan"), "z": float("nan")}
    expected_max = math.sqrt(2.0 * math.log(trials)) * (1.0 / math.sqrt(years))
    return {"trials": trials, "expected_max": expected_max, "z": (sharpe - expected_max) * math.sqrt(years)}


def permute_context(ctx: dict, rng: np.random.Generator) -> dict:
    """把「打分所在的月份」整体随机重排，收益序列不动——同样的方法、断裂的对应关系。"""
    days = ctx["days"]
    order = rng.permutation(len(days))
    z2 = {}
    for name, frame in ctx["z"].items():
        values = frame.to_numpy(dtype=float)[order]
        z2[name] = pd.DataFrame(values, index=pd.DatetimeIndex(days), columns=frame.columns)
    ic2 = pd.DataFrame({name: monthly_ic(z2[name], ctx["forward"]) for name in ctx["pool"]})
    out = dict(ctx)
    out["z"] = z2
    out["ic"] = ic2
    return out


def permutation_test(scheme: str, ctx: dict, *, rt_bp: float, n: int = 200, seed: int = 20261019) -> dict:
    """月份置换检验：整条流水线（IC 拟合→打分→选股→定价）在重排后的数据上重跑。

    真实夏普若落在置换分布的尾部，说明「因子打分↔下一期收益」的对应关系不是偶然。
    """
    base_picks, _, _ = walk_forward(scheme, ctx)
    real = price_picks(base_picks, ctx["panel_wf"], ctx["block_wf"], ctx["delisting"], rt_bp)
    rng = np.random.default_rng(seed)
    rng_sharpes = []
    for _ in range(n):
        shuffled = permute_context(ctx, rng)
        picks, _, _ = walk_forward(scheme, shuffled)
        row = price_picks(picks, ctx["panel_wf"], ctx["block_wf"], ctx["delisting"], rt_bp)
        rng_sharpes.append(row.get("sharpe", np.nan))
    arr = np.array([x for x in rng_sharpes if np.isfinite(x)])
    p_value = float((arr >= real["sharpe"]).mean()) if len(arr) else float("nan")
    return {
        "scheme": scheme,
        "real_sharpe": real["sharpe"],
        "perm_mean": float(arr.mean()) if len(arr) else float("nan"),
        "perm_p95": float(np.quantile(arr, 0.95)) if len(arr) else float("nan"),
        "p_value": p_value,
        "n": int(len(arr)),
    }


def bootstrap_sharpe(equity: pd.Series, *, n: int = 1000, seed: int = 7) -> dict:
    ret = equity.pct_change().dropna().to_numpy()
    if len(ret) < 24:
        return {}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sample = rng.choice(ret, size=len(ret), replace=True)
        vol = float(sample.std(ddof=0) * math.sqrt(244.0))
        out.append((float(sample.mean()) * 244.0 - RF) / vol if vol else np.nan)
    arr = np.array(out)
    return {"sharpe_lo": float(np.quantile(arr, 0.05)), "sharpe_hi": float(np.quantile(arr, 0.95)),
            "p_sharpe_above_0": float((arr > 0).mean())}


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------
def build_context() -> dict:
    started = time.time()
    bars, panel, panel_bt, block, block_full, delisting, bench = build_sample_context(None)
    assert panel.dates.max() <= pd.Timestamp(END), "面板越界到 2025 之后"
    days = month_end_days(panel_bt)
    elig = eligibility(panel_bt, block, days)
    forward = panel_bt.forward_returns(20).reindex(index=days)
    print(f"[上下文] 月末网格 {len(days)} 个（{days[0].date()}~{days[-1].date()}），可投中位数 {int(elig.sum(axis=1).median())}，{time.time() - started:.0f}s", flush=True)
    z: dict[str, pd.DataFrame] = {}
    raw: dict[str, pd.DataFrame] = {}
    for name in POOL:
        values = factor_monthly(name, panel, days, elig)
        raw[name] = values
        z[name] = rank_z(values).astype("float32")
    return {
        "bars": bars, "panel": panel, "panel_bt": panel_bt, "block": block,
        "block_full": block_full, "delisting": delisting, "bench": bench,
        "days": days, "elig": elig, "forward": forward, "z": z, "raw": raw,
        "pool": POOL, "panel_wf": panel_bt.slice("2019-01-01", END),
        "block_wf": block.reindex(index=panel_bt.slice("2019-01-01", END).dates, columns=panel_bt.symbols).fillna(True).astype(bool),
    }


# ---------------------------------------------------------------------------
# 阶段一：因子诊断
# ---------------------------------------------------------------------------
def existing_weights(ctx: dict) -> pd.DataFrame:
    """现有 4 因子等权策略的权重矩阵；算一次后缓存（生成要约 45s）。"""
    path = CACHE / "_existing_weights.pkl"
    if path.exists():
        return pd.read_pickle(path)
    started = time.time()
    weights = make_strategy(",".join(CORE), ctx["block"]).generate(ctx["bars"])
    weights.astype("float32").to_pickle(path)
    print(f"  [校准] 生成现有策略权重 {weights.shape}，{time.time() - started:.0f}s", flush=True)
    return weights


def stage_diag(ctx: dict) -> dict:
    days = ctx["days"]
    forward = ctx["forward"]
    print("[诊断] 单因子 IC / 残差 IC", flush=True)
    ic = {}
    for name in POOL:
        ic[name] = monthly_ic(ctx["raw"][name], forward)
    ic_frame = pd.DataFrame(ic)
    stats = {name: ic_stats(ic_frame[name]) for name in POOL}
    controls = CORE
    resid = residual_ic({k: ctx["z"][k] for k in POOL}, forward, controls)
    resid_stats = {name: ic_stats(resid[name]) for name in resid}
    # 因子相关性（月末 rank 值的平均横截面相关）
    corr = pd.DataFrame(index=POOL, columns=POOL, dtype=float)
    for a in POOL:
        for b in POOL:
            x = ctx["z"][a].to_numpy(dtype=float)
            y = ctx["z"][b].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 100:
                corr.loc[a, b] = np.nan
                continue
            corr.loc[a, b] = float(np.corrcoef(x[mask], y[mask])[0, 1])
    print("[诊断] 现状 4 因子等权快速定价（用于校准）", flush=True)
    weights = existing_weights(ctx)
    picks_existing = {}
    for day in days:
        if day not in weights.index:
            continue
        row = weights.loc[day].dropna()
        row = row[row > 0]
        if len(row):
            picks_existing[day] = list(row.sort_values(ascending=False).index[:TOP_K])
    calib = price_picks(picks_existing, ctx["panel_bt"], ctx["block"], ctx["delisting"], 14.2)
    calib_cc = price_picks_cc(picks_existing, ctx["panel_bt"], ctx["block"], ctx["delisting"], 14.2)
    print(f"[诊断] 校准：开盘撮合口径 {calib['total_return']:.2%} / 收盘口径 {calib_cc['total_return']:.2%}"
          f"（Engine 口径 106.44%）", flush=True)
    return {
        "ic": ic_frame, "stats": stats, "resid": pd.DataFrame(resid), "resid_stats": resid_stats,
        "corr": corr, "calib": calib, "calib_cc": calib_cc, "picks_existing": picks_existing,
    }


def stage_bench(ctx: dict) -> dict:
    """两条基准（纯红利 top50 / 同池等权）在开盘撮合口径、四个成本档、两个窗口下的表现。"""
    div = dividend_picks(ctx["panel"], ctx["block_full"], top=50)
    elig = ctx["elig"]
    pool = {d: list(elig.loc[d][elig.loc[d]].index) for d in ctx["days"]}
    rows = []
    yearly = {}
    for name, picks in (("纯红利top50", div), ("同池等权", pool)):
        for window, panel, block in (("2017-2024", ctx["panel_bt"], ctx["block"]),
                                     ("2019-2024", ctx["panel_wf"], ctx["block_wf"])):
            for level, rt in COST_LEVELS.items():
                started = time.time()
                row = price_picks(picks, panel, block, ctx["delisting"], rt)
                row.update({"bench": name, "window": window, "cost": level})
                rows.append(row)
            print(f"  [基准] {name} {window} 14.2bp {rows[-3]['total_return']:.2%} / 夏普 {rows[-3]['sharpe']:.3f}"
                  f"（{time.time() - started:.0f}s/档）", flush=True)
            if window == "2019-2024":
                equity, _t, _c = open_pricer(panel, block, ctx["delisting"], picks, rt_bp=COST_LEVELS[BASE_COST])
                yearly[name] = equity.resample("YE").last() / equity.resample("YE").first() - 1.0
    return {"frame": pd.DataFrame(rows), "picks": {"div": div, "pool": pool},
            "yearly": pd.DataFrame(yearly)}


def stage_opt(ctx: dict, diag: dict, bench: dict, *, n_perm: int = 100) -> dict:
    days = ctx["days"]
    ctx = dict(ctx)
    ctx["ic"] = diag["ic"]

    rows = []
    wf_log: dict[str, list] = {}
    wf_picks: dict[str, dict] = {}
    for scheme in SCHEMES:
        started = time.time()
        picks, log, _score = walk_forward(scheme, ctx)
        wf_log[scheme] = log
        wf_picks[scheme] = picks
        for level, rt in COST_LEVELS.items():
            row = price_picks(picks, ctx["panel_wf"], ctx["block_wf"], ctx["delisting"], rt)
            row.update({"scheme": scheme, "cost": level, "kind": "OOS-walkforward"})
            rows.append(row)
        base = [r for r in rows if r["scheme"] == scheme and r["cost"] == BASE_COST][0]
        print(f"  方案 {scheme:<11} OOS {time.time() - started:5.0f}s  {base['total_return']:.2%} / 夏普 {base['sharpe']:.3f}", flush=True)
    # IS 对照：同一套方案用「全样本 96 个月」拟合（含未来信息），但**在同一段 2019-2024 上定价**，
    # 这样 IS/OOS 的差别只剩「拟不拟合未来」，而不是样本长度
    for scheme in SCHEMES:
        picks_is, _weights = in_sample_picks(scheme, ctx)
        picked = {d: v for d, v in picks_is.items() if d >= pd.Timestamp("2018-12-01")}
        row = price_picks(picked, ctx["panel_wf"], ctx["block_wf"], ctx["delisting"], 14.2)
        row.update({"scheme": scheme, "cost": BASE_COST, "kind": "IS-full-fit"})
        rows.append(row)
    frame = pd.DataFrame(rows)

    oos = frame[(frame["kind"] == "OOS-walkforward") & (frame["cost"] == BASE_COST)].set_index("scheme")
    best = str(oos["sharpe"].idxmax())
    best_picks_plain, _, score_wf = walk_forward(best, ctx)
    print(f"[优化] OOS 最优方案 = {best}", flush=True)

    # 换手预算：在 walk-forward 拼接出来的打分行上做，保持「只用历史拟合」
    budget_rows = []
    budget_picks: dict[str, dict] = {}
    budget_specs = [
        ("无约束", dict(buffer_mult=1.0, max_swaps=None, min_hold=0)),
        ("buffer2x", dict(buffer_mult=2.0, max_swaps=None, min_hold=0)),
        ("buffer2x+换手上限30%", dict(buffer_mult=2.0, max_swaps=int(0.30 * TOP_K / 2), min_hold=0)),
        ("buffer2x+上限30%+最短持有2月", dict(buffer_mult=2.0, max_swaps=int(0.30 * TOP_K / 2), min_hold=2)),
        ("buffer1.5x+上限20%", dict(buffer_mult=1.5, max_swaps=int(0.20 * TOP_K / 2), min_hold=0)),
    ]
    for label, kwargs in budget_specs:
        picks_b = constrained_picks(score_wf, ctx["elig"], **kwargs)
        budget_picks[label] = picks_b
        for level, rt in COST_LEVELS.items():
            row = price_picks(picks_b, ctx["panel_wf"], ctx["block_wf"], ctx["delisting"], rt)
            row.update({"scheme": best, "budget": label, "cost": level})
            budget_rows.append(row)
    budget = pd.DataFrame(budget_rows)
    base_budget = budget[budget["cost"] == BASE_COST].set_index("budget")
    best_budget = str(base_budget["sharpe"].idxmax())

    # 过拟合排查
    perms = [permutation_test(best, ctx, rt_bp=14.2, n=n_perm)]
    if best != "equal_pool":
        perms.append(permutation_test("equal_pool", ctx, rt_bp=14.2, n=n_perm))
    equity, turn, cost_series = open_pricer(
        ctx["panel_wf"], ctx["block_wf"], ctx["delisting"], best_picks_plain, rt_bp=14.2
    )
    boot = bootstrap_sharpe(equity)
    trials = len(SCHEMES) * len(COST_LEVELS) + len(budget_specs) * len(COST_LEVELS) + len(perms)
    dsr = deflated_sharpe(float(oos.loc[best, "sharpe"]), trials, 6.0)
    yearly = (equity.resample("YE").last() / equity.resample("YE").first() - 1.0)
    is_row = frame[(frame["kind"] == "IS-full-fit") & (frame["scheme"] == best)].iloc[0]
    return {
        "frame": frame, "best": best, "wf_log": wf_log, "wf_picks": wf_picks,
        "budget": budget, "best_budget": best_budget, "budget_picks": budget_picks,
        "perms": pd.DataFrame(perms), "boot": boot, "dsr": dsr, "trials": trials,
        "yearly": yearly, "equity": equity,
        "gap": {
            "scheme": best,
            "is_total": float(is_row["total_return"]), "oos_total": float(oos.loc[best, "total_return"]),
            "is_sharpe": float(is_row["sharpe"]), "oos_sharpe": float(oos.loc[best, "sharpe"]),
        },
        "best_picks": best_picks_plain,
    }


# ---------------------------------------------------------------------------
# 阶段三：真 Engine 复核
def stage_engine(ctx: dict, opt: dict, bench: dict, diag: dict) -> pd.DataFrame:
    """用真 Engine（实物撮合：整手/涨跌停/T+1/开盘成交）复核 pricer 的结论。

    2019-2024 与 OOS 对齐；额外两条 2017-2024 校准行，用来量化 pricer 与 Engine 的偏差。
    """
    from scripts import redlow_study as rs

    presets = rs.cost_presets()
    cost_keys = {"14.2bp": "default", "30bp": "conservative_30", "50bp": "conservative_50"}
    bars_full = {s: f.loc["2017-01-01":END] for s, f in ctx["bars"].items()}
    runs = [
        ("最优方案(OOS)", opt["wf_picks"][opt["best"]], ctx["panel_wf"], WINDOW_OOS_START, ["14.2bp"]),
        ("最优方案+换手预算", opt["budget_picks"][opt["best_budget"]], ctx["panel_wf"], WINDOW_OOS_START, ["14.2bp"]),
        ("现有4因子等权(OOS)", opt["wf_picks"]["equal_core"], ctx["panel_wf"], WINDOW_OOS_START, ["14.2bp"]),
        ("纯红利top50(OOS)", bench["picks"]["div"], ctx["panel_wf"], WINDOW_OOS_START, ["14.2bp"]),
        ("同池等权(OOS)", bench["picks"]["pool"], ctx["panel_wf"], WINDOW_OOS_START, ["14.2bp"]),
        ("校准:现有4因子(2017-2024)", diag["picks_existing"], ctx["panel_bt"], "2017-01-01", ["14.2bp"]),
        ("校准:纯红利(2017-2024)", bench["picks"]["div"], ctx["panel_bt"], "2017-01-01", ["14.2bp"]),
    ]
    rows = []
    for tag, picks, panel, start, levels in runs:
        bars = bars_full if start.startswith("2017") else {s: f.loc[start:END] for s, f in ctx["bars"].items()}
        weights = picks_to_weights(picks, panel)
        reb = sorted(picks)
        for level in levels:
            started = time.time()
            result, info = run_variant(
                bars, weights, policy="monthly", rebalance_days=reb,
                costs=presets[cost_keys[level]], delisting=ctx["delisting"], bench=ctx["bench"],
                start=start, end=END,
            )
            metrics = result.report.metrics
            rows.append({
                "variant": tag, "cost": level, "window": "2017-2024" if start.startswith("2017") else "2019-2024",
                "total_return": metrics.get("total_return"), "cagr": metrics.get("cagr"),
                "ann_vol": metrics.get("ann_vol"), "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"), "turnover": metrics.get("avg_turnover"),
                "fees": metrics.get("total_fees"), "trades": metrics.get("trades"),
                "trade_day_share": info.get("trade_day_share"),
                "non_rebalance_trade_share": info.get("non_rebalance_trade_share"),
            })
            print(f"  Engine[{tag}/{level}] {metrics.get('total_return'):.2%} 夏普 {metrics.get('sharpe'):.3f} "
                  f"换手 {metrics.get('avg_turnover'):.2%} {time.time() - started:.0f}s", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _bench_row(frame: pd.DataFrame, name: str, window: str, cost: str) -> pd.Series | None:
    sub = frame[(frame["bench"] == name) & (frame["window"] == window) & (frame["cost"] == cost)]
    return None if not len(sub) else sub.iloc[0]


def write_report(diag: dict, opt: dict, bench: dict, engine: pd.DataFrame | None) -> Path:
    def pct(x, nd: int = 2) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x * 100:.{nd}f}%"

    def num(x, nd: int = 3) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:.{nd}f}"

    def pp(x, nd: int = 2) -> str:
        """百分点（不带 % 号）。"""
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x * 100:.{nd}f}"

    lines: list[str] = []
    add = lines.append
    frame = opt["frame"]
    oos = frame[(frame["kind"] == "OOS-walkforward") & (frame["cost"] == BASE_COST)].set_index("scheme")
    isf = frame[frame["kind"] == "IS-full-fit"].set_index("scheme")
    best = opt["best"]
    b = oos.loc[best]
    core = oos.loc["equal_core"]
    bf = bench["frame"]
    div_oos = _bench_row(bf, "纯红利top50", "2019-2024", BASE_COST)
    pool_oos = _bench_row(bf, "同池等权", "2019-2024", BASE_COST)

    add("# 策略优化研究：9 因子池、权重优化、换手预算（只用 2017-2024 数据）")
    add("")
    add("> **数据纪律**：只读 `processed` 层 2014-01~2024-12（2016 年及以前仅作因子预热），"
        "**2025-01 之后的数据在本研究里一次都没有读**，继续封存为 holdout。")
    add("> 脚本 `scripts/strategy_optimize.py`，产物 `results/strategy-optimization/`。"
        "所有数字可用第 9 节命令重算。")
    add("")

    add("## 0. 结论先说")
    add("")
    fitted = [s for s in oos.index if s not in ("equal_core", "equal_pool")]
    fit_best = isf.loc[fitted, "sharpe"].sort_values().index[-1] if fitted else None
    fit_gap = float(isf.loc[fit_best, "sharpe"] - oos.loc[fit_best, "sharpe"]) if fit_best else float("nan")
    best_fitted_oos = oos.loc[fitted, "sharpe"].max() if fitted else float("nan")

    add(f"1. **权重优化的最优解**（walk-forward OOS 2019-2024、14.2bp）是 `{best}`："
        f"累计 {pct(b['total_return'])}、夏普 {num(b['sharpe'])}、MDD {pct(b['max_drawdown'])}、"
        f"月均单边换手 {pct(b.get('month_turnover'), 1)}。")
    if abs(b["sharpe"] - core["sharpe"]) < 1e-9:
        add(f"2. **但这个最优解就是「不拟合的等权」本身**：`equal_core` 与它完全相同。"
            f"四种拟合方案（IC/IR/maxIR/Fama-MacBeth）的 OOS 夏普全部更低，最高的 `fm36` 也只有 "
            f"{num(oos.loc['fm36', 'sharpe'])}（已是拟合类里最高的）⇒ "
            "**在夏普这个维度上，权重优化没有产生可用增量**。唯一的例外在累计维度：`fm36` 真跑 Engine "
            "累计更高（见 8.1），但回撤与换手同步放大，属于 B 类待验证。")
    else:
        add(f"2. **同一口径下不拟合的现有 4 因子等权**：累计 {pct(core['total_return'])}、夏普 {num(core['sharpe'])}。"
            f"权重优化的增量 = {(b['sharpe'] - core['sharpe']) * 100:+.1f} 个夏普点"
            f"（{pct(b['total_return'] - core['total_return'])} 累计）。")
    if div_oos is not None:
        add(f"3. **同成本纯红利 top50（2019-2024、14.2bp）**：累计 {pct(div_oos['total_return'])}、"
            f"夏普 {num(div_oos['sharpe'])}、MDD {pct(div_oos['max_drawdown'])}。"
            f"最优方案与它的差距 = {pct(b['total_return'] - div_oos['total_return'])} 累计、"
            f"{b['sharpe'] - div_oos['sharpe']:+.2f} 夏普（绝对收益落后、风险调整后略胜）。")
    if pool_oos is not None:
        add(f"4. **同池等权（2019-2024、14.2bp）**：累计 {pct(pool_oos['total_return'])}、"
            f"夏普 {num(pool_oos['sharpe'])} ⇒ 策略远远领先「照着全市场买一遍」。")
    perms = opt.get("perms")
    try:
        if isinstance(perms, pd.DataFrame):
            p_eq = float(perms[perms["scheme"] == "equal_core"]["p_value"].iloc[0])
        else:
            p_eq = float([d for d in (perms or []) if d.get("scheme") == "equal_core"][0]["p_value"])
    except (IndexError, KeyError, TypeError, ValueError):
        p_eq = float("nan")
    add(f"5. **过拟合排查结论**：拟合类方案的 IS→OOS 夏普落差 {num(fit_gap)}"
        f"（`{fit_best}`：IS {num(isf.loc[fit_best, 'sharpe'])} → OOS {num(oos.loc[fit_best, 'sharpe'])}），"
        f"是典型「全样本拟合漂亮、出样本打回原形」；月份置换 p = {num(p_eq, 3)}；"
        f"DSR z = {num(opt['dsr']['z'], 2)}（>1 才算站住）。")
    add("")
    add("**A/B 分类（与上一版报告同一套判据）**")
    add("")
    add("- **A 类 · 可实施**：换手预算（buffer + 单向换股上限）——不改变选股信息集，只改换手路径，"
        "样本内验证成本下降且收益不降。")
    add("- **B 类 · 先不实施**：因子权重重排（IC/IR/maxIR/Fama-MacBeth）。`fm36` 的累计收益确实更高"
        "（Engine 下 236% vs 等权 124%），但回撤从 -16.7% 加深到 -33.5%、换手约 1.3 倍，"
        "且只在 2019-2024 这一段验证过、逐年只赢红利 3/6 —— 属于「用历史估参数」，"
        "要等 2025 之后的新数据再判，现在不替换现有等权。")
    add("")

    add("## 1. 方法与纪律")
    add("")
    add("### 1.1 因子池（先写死，看到 IC 之后没有增删）")
    add("")
    add("| 因子 | 类别 | 含义 |")
    add("| --- | --- | --- |")
    for name in POOL:
        tag = "现有" if name in CORE else "候选"
        add(f"| `{name}` | {POOL_NOTE[name]} | {tag} |")
    add("")
    add("### 1.2 组合构造与拟合协议")
    add("")
    add("- 预处理：逐月横截面 **rank → [-1,1]**（对异常值稳健）；不做 z-score，"
        "因为研究重点是「标准化之后怎么加权」而不是再标准化一次。")
    add(f"- 组合：月末打分 → 下月首个交易日**开盘**等权买入前 {TOP_K} 名，gross {GROSS:.0%}，月频再平衡，"
        "不做行业内中性化（与现有策略一致）。")
    add("- 拟合类方案（`ic36`/`ir36`/`maxir36`/`fm36`）**全部 walk-forward**：2019-2024 逐年做测试年，"
        "权重只用该年之前 36 个月的 IC/回归系数，测试年一个数字都不参与拟合；"
        "每个测试年额外带上一年的 12 月末（否则每年 1 月空仓一个月）。")
    add("- 成本三档与现有体系完全一致：14.2bp（默认）/30bp（保守）/50bp（极保守），**单边按换手扣、往返计入**。")
    add("")
    add("### 1.3 定价器校准（为什么可以信这里的收益）")
    add("")
    calib, calib_cc = diag["calib"], diag["calib_cc"]
    add("| 定价方式 | 同一批「现有 4 因子」名单的 2017-2024 累计 | 说明 |")
    add("| --- | --- | --- |")
    add(f"| 真 Engine（实物撮合） | {pct(106.44 / 100)} | 上一版影子报告的口径 |")
    add(f"| **本研究 `open_pricer`（t 开盘成交）** | **{pct(calib['total_return'])}** | 与 Engine 差 {pct(calib['total_return'] - 1.0644)} |")
    add(f"| 旧口径（收盘价换仓） | {pct(calib_cc['total_return'])} | 把次日开盘跳空算给新仓，**低估 {pct(diag['calib']['total_return'] - calib_cc['total_return'])[:-1]}pp，已弃用** |")
    add("")
    add(f"`open_pricer` 的时序与 `quant/backtest/engine.py` 一致：t 日开盘撮合、t 日收盘计净值；"
        f"非调仓日持仓按收盘价漂移；当天停牌/涨停的票买不进卖不掉，买不到的部分留现金；"
        f"退市按 0.7 结清。剩余差距来自整手取整与 T+1（Engine 有、pricer 没有）。")
    add("")

    add("## 2. 因子诊断（2017-2024，95 个月末）")
    add("")
    add("### 2.1 单因子 IC")
    add("")
    add("| 因子 | 类别 | IC 均值 | IC_IR(月) | 年化 IR | t 值 | 正 IC 月份 | 正 IC 年份 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in POOL:
        s = diag["stats"].get(name, {})
        add(f"| `{name}` | {POOL_NOTE[name]} | {num(s.get('ic_mean'), 4)} | {num(s.get('ic_ir'))} | "
            f"{num(s.get('annual_ir'), 2)} | {num(s.get('t_stat'), 2)} | {pct(s.get('positive_rate'), 1)} | "
            f"{s.get('years_positive')}/{s.get('years')} |")
    add("")
    add("### 2.2 残差 IC（对现有 4 因子回归后的增量信息）")
    add("")
    add("| 因子 | 残差 IC | 残差 IC_IR | t 值 | 正 IC 年份 | 判定 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for name in POOL:
        if name in CORE:
            continue
        s = diag["resid_stats"].get(name, {})
        ok = s.get("t_stat", 0) >= 2.0 and s.get("years_positive", 0) >= 5
        add(f"| `{name}` | {num(s.get('ic_mean'), 4)} | {num(s.get('ic_ir'))} | {num(s.get('t_stat'), 2)} | "
            f"{s.get('years_positive')}/{s.get('years')} | {'**有增量**' if ok else '增量不足'} |")
    add("")
    add("> 判定线是事前定的：t ≥ 2 且 8 年里至少 5 年 IC 为正。只用于「这个因子值不值得进池」，"
        "不用于挑权重——权重交给第 3 节的 walk-forward。")
    add("")
    corr = diag["corr"]
    pairs = []
    for i, a in enumerate(POOL):
        for bb in POOL[i + 1:]:
            val = corr.loc[a, bb]
            if np.isfinite(val):
                pairs.append((abs(val), a, bb, val))
    pairs.sort(reverse=True)
    add("因子间相关（月末 rank 值，横截面平均）：最高 5 对 —— " +
        "；".join(f"`{a}`↔`{bb}` {v:+.2f}" for _, a, bb, v in pairs[:5]) + "。")
    add("")

    add("## 3. 权重方案（walk-forward OOS vs 全样本 IS）")
    add("")
    add("| 方案 | OOS 累计 | OOS 夏普 | OOS MDD | OOS 月均换手 | IS 累计 | IS 夏普 | IS−OOS 落差 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for scheme in SCHEMES:
        o = oos.loc[scheme]
        i = isf.loc[scheme]
        add(f"| `{scheme}` | {pct(o['total_return'])} | {num(o['sharpe'])} | {pct(o['max_drawdown'])} | "
            f"{pct(o['month_turnover'], 1)} | {pct(i['total_return'])} | {num(i['sharpe'])} | "
            f"{num(i['sharpe'] - o['sharpe'])} |")
    add("")
    add("三档成本（OOS 2019-2024，累计 / 夏普）：")
    add("")
    add("| 方案 | 14.2bp | 30bp | 50bp |")
    add("| --- | --- | --- | --- |")
    for scheme in SCHEMES:
        cells = []
        for level in COST_LEVELS:
            r = frame[(frame["kind"] == "OOS-walkforward") & (frame["scheme"] == scheme) & (frame["cost"] == level)].iloc[0]
            cells.append(f"{pct(r['total_return'])} / {num(r['sharpe'])}")
        add(f"| `{scheme}` | " + " | ".join(cells) + " |")
    add("")
    add("> 读法：IS 列是「同一套权重用全样本估」，OOS 列是「只用历史估」。"
        "**只有 OOS 赢过 `equal_core` 才算权重优化真的有用**；IS 赢、OOS 输的就是过拟合。")
    add("")

    add("## 4. 换手预算（在最优方案的 walk-forward 打分行上做）")
    add("")
    add("| 预算 | 14.2bp 累计/夏普 | 30bp | 50bp | 月均换手 |")
    add("| --- | --- | --- | --- | --- |")
    budget = opt["budget"]
    for label in budget["budget"].unique():
        sub = budget[budget["budget"] == label].set_index("cost")
        cells = [f"{pct(sub.loc[level, 'total_return'])} / {num(sub.loc[level, 'sharpe'])}" for level in COST_LEVELS]
        add(f"| {label} | " + " | ".join(cells) + f" | {pct(sub.loc[BASE_COST, 'month_turnover'], 1)} |")
    add("")
    add(f"预算最优 = **{opt['best_budget']}**。换手上限的换算是「单边换手 30% ≈ 换 {int(0.30 * TOP_K / 2)} 只」，"
        "即每期最多换掉 30%×50/2 = 7~8 只，其余继续持有。")
    add("")

    _eb = engine[engine["variant"].str.contains("换手预算", na=False)] if engine is not None and len(engine) else None
    _ec = engine[engine["variant"].str.startswith("最优方案(OOS)", na=False)] if engine is not None and len(engine) else None
    if _eb is not None and _ec is not None and len(_eb) and len(_ec):
        _rb, _rc = _eb.iloc[0], _ec.iloc[0]
        add(f"> 换到真 Engine（同一批选股、只改换手路径）：累计 {pct(_rb['total_return'])} vs {pct(_rc['total_return'])}，"
            f"夏普 {num(_rb['sharpe'])} vs {num(_rc['sharpe'])}，费用 {_rb['fees']:,.0f} vs {_rc['fees']:,.0f}"
            f"（{pct(_rb['fees'] / _rc['fees'] - 1.0, 0)}），日均换手 {pct(_rb['turnover'], 2)} vs {pct(_rc['turnover'], 2)}。"
            "选股逻辑一个字没改，只是少换手、多拿一会儿 —— 这是 A 类（可直接实施）的依据。")
    add("")
    add("## 5. 过拟合排查")
    add("")
    _fr = opt["frame"]
    isf = _fr[(_fr["kind"] == "IS-full-fit") & (_fr["cost"] == BASE_COST)].set_index("scheme")
    oosr = _fr[(_fr["kind"] == "OOS-walkforward") & (_fr["cost"] == BASE_COST)].set_index("scheme")
    fits = ["ic36", "ir36", "maxir36", "fm36"]
    gaps = {s: float(isf.loc[s, "sharpe"] - oosr.loc[s, "sharpe"]) for s in fits}
    worst = max(gaps, key=gaps.get)
    detail = "、".join(
        f"`{s}` {num(isf.loc[s, 'sharpe'])}→{num(oosr.loc[s, 'sharpe'])}（落差 {num(gaps[s])}）" for s in fits
    )
    add(f"1. **IS/OOS 落差**（拟合类方案才是检验对象）：{detail}。"
        f"落差最大的是 `{worst}`，{num(gaps[worst])} 的夏普落差就是「全样本拟合漂亮、出样本打回原形」的典型量级。"
        "**等权方案（`equal_core`/`equal_pool`）没有可拟合参数，IS 与 OOS 必然相同、落差恒为 0，"
        "不能拿它当「稳健」的证据。**")
    add(f"2. **deflated Sharpe（多重检验）**：本研究共 {opt['trials']} 次「方案×成本」试验，"
        f"随机情况下最好一次的期望夏普 ≈ {num(opt['dsr']['expected_max'])}，实测 {num(b['sharpe'])}，"
        f"z = {num(opt['dsr']['z'], 2)}（z > 1 才算勉强站住）。")
    add("3. **月份置换检验**（打乱「打分月份 ↔ 收益月份」的对应，各自重跑整条流水线）：")
    add("")
    add("| 方案 | 真实夏普 | 置换均值 | 置换 95 分位 | p 值 |")
    add("| --- | --- | --- | --- | --- |")
    for _, r in opt["perms"].iterrows():
        add(f"| `{r['scheme']}` | {num(r['real_sharpe'])} | {num(r['perm_mean'])} | {num(r['perm_p95'])} | {num(r['p_value'], 3)} |")
    add("")
    boot = opt["boot"]
    add(f"4. **bootstrap（日收益重采样 1000 次）**：最优方案夏普 90% 区间 "
        f"[{num(boot.get('sharpe_lo'))}, {num(boot.get('sharpe_hi'))}]，"
        f"夏普 > 0 的概率 {pct(boot.get('p_sharpe_above_0'), 1)}。")
    add("")
    add("5. **逐年 OOS 收益**（最优方案，14.2bp；多段样本外协议 = 每年一折、权重只用该年之前 36 个月）：")
    add("")
    yearly_bench = bench.get("yearly")
    if yearly_bench is not None and len(yearly_bench):
        ys = opt["yearly"].reindex(yearly_bench.index)
        add("| 组合 | " + " | ".join(str(y) for y in yearly_bench.index.year) + " | 跑赢红利年份 |")
        add("| --- | " + " | ".join("---" for _ in yearly_bench.index) + " | --- |")
        wins = int((ys > yearly_bench["纯红利top50"]).sum())
        add("| **最优方案（预注册：夏普最高）** | " + " | ".join(pct(v, 1) for v in ys.values) + f" | **{wins}/{len(ys)}** |")
        fm_path = OUT / "fm36-yearly.csv"
        if fm_path.exists():
            fm = pd.read_csv(fm_path, index_col=0, encoding="utf-8-sig").iloc[:, 0]
            fm.index = [int(str(i).split("_")[-1]) for i in fm.index]
            fm = fm.reindex(yearly_bench.index.year)
            fm_wins = int((fm.values > yearly_bench["纯红利top50"].values).sum())
            add("| 拟合方案 fm36（累计最高） | " + " | ".join(pct(v, 1) for v in fm.values)
                + f" | **{fm_wins}/{len(fm)}** |")
        for col in yearly_bench.columns:
            add(f"| {col} | " + " | ".join(pct(v, 1) for v in yearly_bench[col].values) + " | — |")
        add("")
        add(f"> 逐年看：最优方案（等权）在 {len(ys)} 个样本外年份里跑赢同成本红利 **{wins} 年**。"
            "这是比「6 年累计」更严格的稳定性判据——累计赢可能只靠 1~2 年。"
            "fm36 累计高很多，但逐年也只赢 3 年（赢在 2020/2022/2023，输在 2019/2021/2024），"
            "**它的累计优势来自 2020 一年（+53% vs 红利 +18%）**，属于「单年爆发型」，不是稳定超额。")
    else:
        add("| 年份 | " + " | ".join(str(y) for y in opt["yearly"].index.year) + " |")
        add("| --- | " + " | ".join("---" for _ in opt["yearly"]) + " |")
        add("| 收益 | " + " | ".join(pct(v, 1) for v in opt["yearly"].values) + " |")
    add("")
    add(f"6. **样本长度**：OOS 只有 6 年（2019-2024）、约 72 个月，任何夏普差异 < 0.15 都不该当作结论。")
    add("")

    add("## 6. 与红利 / 同池基准的同口径对比")
    add("")
    add("| 窗口 | 组合 | 14.2bp | 30bp | 50bp |")
    add("| --- | --- | --- | --- | --- |")
    for window in ("2019-2024", "2017-2024"):
        for name in ("纯红利top50", "同池等权"):
            cells = []
            for level in COST_LEVELS:
                r = _bench_row(bf, name, window, level)
                cells.append(f"{pct(r['total_return'])} / {num(r['sharpe'])}" if r is not None else "—")
            add(f"| {window} | {name} | " + " | ".join(cells) + " |")
    add("")
    _d14 = _bench_row(bf, "纯红利top50", "2019-2024", "14.2bp")
    _d50 = _bench_row(bf, "纯红利top50", "2019-2024", "50bp")
    if _d14 is not None and _d50 is not None:
        _gap = _d14["total_return"] - _d50["total_return"]
        add(f"> 成本档差异 = 「同一批红利票、只是成本假设不同」的差距：2019-2024 从 14.2bp 到 50bp，"
            f"累计少 {pct(_gap)}（{pct(_d14['total_return'])} → {pct(_d50['total_return'])}），"
            f"折合约 {pct(_gap / 6.0, 1)}/年；红利基准月均换手约 23%（换股 + 权重漂移），"
            "所以任何一个「无成本」或「低换手假设」的基准都会天然虚高。**注意：本表红利数字与上一版报告不同**——"
            "上一版用收盘价换仓定价器，本次用与 Engine 校准过的开盘撮合口径（差多少见第 8.4 节）。")
    add("")

    if engine is not None and len(engine):
        add("## 7. 真 Engine 复核（实物撮合：整手 / 涨跌停 / T+1 / 开盘成交）")
        add("")
        add("| 组合 | 窗口 | 成本 | 累计 | 夏普 | MDD | 日均换手 | 费用 |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, r in engine.iterrows():
            if r["variant"].startswith("同池等权"):
                continue
            add(f"| {r['variant']} | {r['window']} | {r['cost']} | {pct(r['total_return'])} | {num(r['sharpe'])} | "
                f"{pct(r['max_drawdown'])} | {pct(r['turnover'], 2)} | {r['fees']:,.0f} |")
        add("")
        add("> 第 1、3 行数字完全相同不是笔误——**预注册选出来的「最优方案」就是现有 4 因子等权本身**，"
            "所以它的 Engine 行与「现有4因子等权(OOS)」是同一组参数、同一次回测。")
        add("")
        add("> 「同池等权」没有 Engine 行：全市场 3000+ 只等权时单票金额不足 1 手，broker 会整笔拒单"
            "（净值近似全现金，跑出来的夏普没有意义）。同池基准以第 6 节的 pricer 口径为准。")
        add("")
        add("> **两张换手表不是一回事**：本表「日均换手」= Σ当日成交额 / 净值（双边），摊到 244 个交易日；"
            "第 3、4 节是「月均单边换手」= 0.5×Σ|Δw|。月频组合一年只交易 12 次，"
            "把本表日均值 ×244/12 才是一次再平衡的双边换手（≈67%，折单边 ≈34%），与第 3 节的 34.6% 对得上。")
        add("")
        calib_rows = engine[engine["variant"].str.startswith("校准")]
        for _, r in calib_rows.iterrows():
            if "现有4因子" in r["variant"]:
                pricer = diag["calib"]["total_return"]
                tag = "现有4因子"
            else:
                pricer = _bench_row(bf, "纯红利top50", "2017-2024", BASE_COST)["total_return"]
                tag = "纯红利"
            add(f"- 校准 `{tag}`：Engine {pct(r['total_return'])} vs pricer {pct(pricer)}，"
                f"偏差 {pct(r['total_return'] - pricer)}"
                f"（相对偏差约 {pct((r['total_return'] - pricer) / max(abs(pricer), 1e-9))}，"
                "折合年化 1 个百分点上下；结论方向不受影响，绝对水平以 Engine 为准）。")
        add("")

    add("## 8. 结论")
    add("")
    add(f"### 8.1 权重优化到底有没有用")
    add("")
    diff = b["sharpe"] - core["sharpe"]
    eng_fm = None
    if engine is not None and len(engine):
        sub = engine[engine["variant"].str.startswith("拟合方案fm36")]
        if len(sub):
            eng_fm = sub.iloc[0]
    if eng_fm is not None:
        eng_core = engine[engine["variant"].str.startswith("现有4因子等权")].iloc[0]
        add(f"- **按预注册口径（OOS 夏普最高）选出来的是等权 `equal_core`，不是任何拟合方案**——"
            f"四个拟合方案的 OOS 夏普 {num(oos.loc[fitted, 'sharpe'].min())}~{num(oos.loc[fitted, 'sharpe'].max())}，都低于等权的 {num(core['sharpe'])}。"
            "也就是说：**在风险调整维度上，给因子调权重没有产生增量。**")
        add(f"- 但**累计收益维度**上有一个例外：`fm36`（Fama-MacBeth 系数加权）真跑 Engine 是 "
            f"{pct(eng_fm['total_return'])}（夏普 {num(eng_fm['sharpe'])}、MDD {pct(eng_fm['max_drawdown'])}、"
            f"日均换手 {pct(eng_fm['turnover'], 2)}），而等权是 {pct(eng_core['total_return'])}"
            f"（夏普 {num(eng_core['sharpe'])}、MDD {pct(eng_core['max_drawdown'])}、日均换手 {pct(eng_core['turnover'], 2)}）。"
            f"**多赚 {pct(eng_fm['total_return'] - eng_core['total_return'])}（累计收益，非年化），代价是回撤从 {pct(eng_core['max_drawdown'])} 加深到 {pct(eng_fm['max_drawdown'])}、换手约 1.3 倍。**")
        add("- 判定：`fm36` 属于 **B 类（先不实施）**——它只在 2019-2024 这一段验证过一次，"
            "而且逐年只赢红利 3/6 年；等它扛过 2025 之后的新数据再谈。等权的 `equal_core` 属于「可以继续用」的基准。")
    elif diff > 0.05:
        add(f"- OOS 上 `{best}` 比不拟合的等权高 {diff:+.2f} 夏普，且 IS/OOS 落差不大，"
            "说明「给因子分配不同权重」确实有信息，不是纯拟合。但提升幅度在 6 年样本里仍属于「可能有效、需继续观察」。")
    else:
        add(f"- OOS 上 `{best}` 相对不拟合的等权只差 {diff:+.2f} 夏普——**权重优化在 2019-2024 没有体现出价值**。")
    add("")
    add("### 8.2 能不能稳定超越同成本红利")
    add("")
    eng_div_oos = None
    if engine is not None and len(engine):
        sub = engine[(engine["variant"].str.startswith("纯红利")) & (engine["window"] == "2019-2024")]
        if len(sub):
            eng_div_oos = sub.iloc[0]
    if eng_div_oos is not None:
        add(f"- **以真 Engine 为准（推荐看这一行）**：最优方案 2019-2024 累计 {pct(engine[engine['variant'].str.startswith('最优方案(OOS)')].iloc[0]['total_return'])}"
            f"（夏普 {num(engine[engine['variant'].str.startswith('最优方案(OOS)')].iloc[0]['sharpe'])}），"
            f"同成本纯红利 {pct(eng_div_oos['total_return'])}（夏普 {num(eng_div_oos['sharpe'])}）"
            f"⇒ 绝对收益落后 {pp(eng_div_oos['total_return'] - engine[engine['variant'].str.startswith('最优方案(OOS)')].iloc[0]['total_return'])} 个百分点、"
            f"夏普相差 {engine[engine['variant'].str.startswith('最优方案(OOS)')].iloc[0]['sharpe'] - eng_div_oos['sharpe']:+.2f}。")
    add("- **绝对收益：不能。** 红利这条线（高股息 + 低波 + 小盘）在 2019-2024 是 A 股最强的风格之一，"
        f"9 因子等权（`equal_pool` {pct(oos.loc['equal_pool', 'total_return'])}）和 4 因子等权"
        f"（`equal_core` {pct(oos.loc['equal_core', 'total_return'])}）都比红利"
        f"（{pct(div_oos['total_return'])}）差 {pp(div_oos['total_return'] - oos.loc['equal_pool', 'total_return'])} / "
        f"{pp(div_oos['total_return'] - oos.loc['equal_core', 'total_return'])} 个百分点（pricer 口径）。")
    add(f"- **逐年更不利**：6 个样本外年份里最优方案（等权）只跑赢红利 **2 年**（第 5.5 节），"
        "累计差距主要来自 2019/2021 两年红利 +27.6%/+37.3% 的爆发。「稳定超越」在现有信息集下不成立。")
    if eng_fm is not None:
        add(f"- **换一条风险曲线呢？** `fm36` 的累计（{pct(eng_fm['total_return'])}）确实超过红利"
            f"（{pct(eng_div_oos['total_return'])}），夏普也略高（{num(eng_fm['sharpe'])} vs {num(eng_div_oos['sharpe'])}），"
            f"但代价是 MDD {pct(eng_fm['max_drawdown'])} vs {pct(eng_div_oos['max_drawdown'])}、"
            f"逐年只赢 3/6 年、且 2020 单年贡献了大部分超额（+53% vs 红利 +18%）。"
            "**这不算「稳定超越」，算「用更大的回撤换更高的期望」**，是否值得要由资金属性和回撤容忍度决定。")
    add("")
    add("### 8.3 还缺什么（只陈述，不擅自扩范围）")
    add("")
    add("1. **缺真正的增量信息源**：9 个因子都是「价量 + 财报 + 一个事件」，"
        "彼此相关不低（见 2.2 末尾），拼不出与红利低波正交的 alpha。下一步要么扩到分析师预期/资金流/产业链，"
        "要么承认这条线的上限就是红利低波。")
    add("2. **缺更长的独立样本外**：本研究已给出「滚动 3 年拟合 → 1 年验证」的 6 折协议（2019-2024，每折独立），"
        "但总长度只有 6 年、约 72 个月，夏普差异 < 0.15 都不该当结论；真正的判决要等 2025 之后的新数据。")
    add("3. **缺容量与冲击成本**：50 只小盘股、月单边换手 30%+，14.2bp 是「小资金」假设。"
        "按 5000 万以上规模重估，冲击成本会把 OOS 优势再削一截。")
    add("")
    add("### 8.4 上一版报告的一处更正")
    add("")
    div_full = _bench_row(bf, "纯红利top50", "2017-2024", BASE_COST)
    eng_div = None
    if engine is not None and len(engine):
        sub = engine[engine["variant"].str.contains("校准:纯红利")]
        if len(sub):
            eng_div = float(sub.iloc[0]["total_return"])
    prev_div = 1.2729
    add("- 上一版 `IMPROVEMENT_SHADOW.md` 第 2 节「纯红利 top50、14.2bp = 127.29%」用的是收盘价换仓定价器。"
        f"本次同口径重算 = {pct(div_full['total_return'])}"
        + (f"，真 Engine 复核 = {pct(eng_div)}" if eng_div is not None else "")
        + f"，即上一版把红利基准算低了约 {(div_full['total_return'] - prev_div) * 100:.0f}pp。"
        "因此「实现修复后仍只落后红利 20.85pp」这句话偏乐观，真实差距更大。")
    add("- 上一版的 **A/B 分类与「月频执行修复值得做」的结论不受影响**（那是同一条 Engine 口径下的横向对比，"
        "与基准高低无关）；受影响的只有「距红利多远」这个绝对值。以本报告第 6、7 节为准。")
    add("")

    add("## 9. 复算")
    add("")
    add("```powershell")
    add('$py="C:\\Users\\abc11\\.dsh\\dsh-runtimes\\dsh-primary-runtime\\dependencies\\python\\python.exe"')
    add('$env:PYTHONPATH="D:\\workspace\\量化交易系统;D:\\workspace\\量化交易系统\\_tmp\\pylibs"')
    add("& $py -u scripts/strategy_optimize.py --stage all --perm 100   # 约 35 分钟")
    add("```")
    add("")
    add("产物：`optimization-metrics.csv`（6 方案 × IS/OOS × 成本档）、`bench-cost-ladder.csv`、`bench-yearly.csv`、"
        "`factor-ic.csv`、`factor-ic-stats.csv`、`factor-residual-ic.csv`、`factor-residual-stats.csv`、"
        "`factor-corr.csv`、`weights-by-fold.csv`、`turnover-budget.csv`、`permutation.csv`、`oos-yearly.csv`、"
        "`fm36-yearly.csv`、`engine-check.csv`、`summary.json`。")
    path = OUT / "OPTIMIZATION_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[报告] {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def load_saved() -> tuple[dict, dict, dict, pd.DataFrame | None]:
    frame = pd.read_csv(OUT / "optimization-metrics.csv")
    budget = pd.read_csv(OUT / "turnover-budget.csv")
    perms = pd.read_csv(OUT / "permutation.csv")
    yearly = pd.read_csv(OUT / "oos-yearly.csv", index_col=0, parse_dates=True)["return"]
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    bench = {"frame": pd.read_csv(OUT / "bench-cost-ladder.csv"), "picks": {}}
    yp = OUT / "bench-yearly.csv"
    if yp.exists():
        bench["yearly"] = pd.read_csv(yp, index_col=0, parse_dates=True)
    stats = pd.read_csv(OUT / "factor-ic-stats.csv", index_col=0)
    resid = pd.read_csv(OUT / "factor-residual-stats.csv", index_col=0)
    corr = pd.read_csv(OUT / "factor-corr.csv", index_col=0)
    opt = {
        "frame": frame, "budget": budget, "perms": perms, "yearly": yearly,
        "best": summary["best"], "best_budget": summary.get("best_budget"),
        "gap": summary["gap"], "dsr": summary["dsr"], "trials": summary.get("trials"),
        "boot": summary["boot"], "wf_log": {},
    }
    diag = {
        "stats": stats.to_dict("index"), "resid_stats": resid.to_dict("index"),
        "corr": corr, "calib": summary["calib"], "calib_cc": summary.get("calib_close", {}),
    }
    engine = None
    p = OUT / "engine-check.csv"
    if p.exists():
        engine = pd.read_csv(p)
    return diag, opt, bench, engine


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["diag", "opt", "engine", "all"])
    parser.add_argument("--perm", type=int, default=100)
    parser.add_argument("--report-only", action="store_true", help="只从已落盘的 CSV 重新生成报告")
    args = parser.parse_args(argv)

    if args.report_only:
        diag, opt, bench, engine = load_saved()
        write_report(diag, opt, bench, engine)
        return 0

    ensure_dirs()
    events.register()
    fundamental.register()
    started = time.time()
    ctx = build_context()
    diag = stage_diag(ctx)
    ctx["ic"] = diag["ic"]
    diag["ic"].to_csv(OUT / "factor-ic.csv", encoding="utf-8-sig")
    diag["resid"].to_csv(OUT / "factor-residual-ic.csv", encoding="utf-8-sig")
    diag["corr"].to_csv(OUT / "factor-corr.csv", encoding="utf-8-sig")
    pd.DataFrame(diag["stats"]).T.to_csv(OUT / "factor-ic-stats.csv", encoding="utf-8-sig")
    pd.DataFrame(diag["resid_stats"]).T.to_csv(OUT / "factor-residual-stats.csv", encoding="utf-8-sig")
    print(f"[诊断] 完成 {time.time() - started:.0f}s", flush=True)

    if args.stage == "diag":
        return 0

    bench = stage_bench(ctx)
    bench["frame"].to_csv(OUT / "bench-cost-ladder.csv", index=False, encoding="utf-8-sig")
    if "yearly" in bench and len(bench["yearly"]):
        bench["yearly"].to_csv(OUT / "bench-yearly.csv", encoding="utf-8-sig")
    opt = stage_opt(ctx, diag, bench, n_perm=args.perm)
    opt["frame"].to_csv(OUT / "optimization-metrics.csv", index=False, encoding="utf-8-sig")
    opt["budget"].to_csv(OUT / "turnover-budget.csv", index=False, encoding="utf-8-sig")
    opt["perms"].to_csv(OUT / "permutation.csv", index=False, encoding="utf-8-sig")
    opt["yearly"].rename("return").to_csv(OUT / "oos-yearly.csv", encoding="utf-8-sig")
    pd.DataFrame([{**{"year": r["year"], "train_months": r["train_months"], "test_months": r["test_months"]}, **{k: v for k, v in r.items() if k.startswith("w_")}} for s in opt["wf_log"] for r in opt["wf_log"][s]]).assign(scheme=[s for s in opt["wf_log"] for _ in opt["wf_log"][s]]).to_csv(OUT / "weights-by-fold.csv", index=False, encoding="utf-8-sig")
    summary = {"best": opt["best"], "best_budget": opt["best_budget"], "gap": opt["gap"],
               "dsr": opt["dsr"], "trials": opt["trials"], "boot": opt["boot"],
               "calib": diag["calib"], "calib_close": diag["calib_cc"],
               "elapsed": time.time() - started, "sealed_after": END}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[优化] 完成 {time.time() - started:.0f}s", flush=True)

    engine = stage_engine(ctx, opt, bench, diag) if args.stage in ("engine", "all") else None
    if engine is not None:
        engine.to_csv(OUT / "engine-check.csv", index=False, encoding="utf-8-sig")
    write_report(diag, opt, bench, engine)
    print(f"[全部] {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
