"""多因子选股策略（日线低频）。

流程与券商/公募的主流做法一致：

1. **股票池**：只用当日可交易（有量有价、非停牌）、价格与成交额达标、
   上市时间够长的标的——这一步比因子本身更能决定成败（别把次新股和僵尸股算进去）；
2. **因子**：从 :mod:`quant.factors.library` 取多个量价因子（`值越大越看多`）；
3. **预处理**：去极值 →（可选）风格中性化 → 标准化；
4. **合成**：等权 / 手工权重 / IC 或 IC_IR 加权（**滚动窗口，只用历史数据**，
   见 ``weighting`` 的说明）；
5. **选股**：按打分取前 ``top_k``，带**缓冲区**（``buffer_rank``）：
   已在持仓里的只要没掉出前 ``buffer_rank`` 就继续持有，避免为了名次微小变化
   来回换手——低频多因子的超额收益经常被交易成本吃掉；
6. **赋权**：等权或按打分加权，单票权重上限 ``max_weight``，总仓位 ``gross_exposure``。

**关于 IC 加权不能有前视**：``weighting='ic'/'ir'`` 时，每个调仓日只用
「截至上一交易日已能观测到的因子值与已实现的未来收益」估计 IC，
每 ``weight_refresh`` 个调仓日刷新一次；样本不足 ``ic_min_periods`` 就退回等权。
这是本策略与"拿全样本 IC 调参"最大的区别。
"""

from __future__ import annotations

import warnings

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant.factors.composite import equal_weights
from quant.factors.library import compute
from quant.factors.panel import Panel, build_panel
from quant.factors.preprocess import preprocess
from quant.strategy.base import Strategy

__all__ = ["MultiFactorStrategy"]


def _row_spearman(left: pd.DataFrame, right: pd.DataFrame, min_count: int = 5) -> pd.Series:
    """逐日横截面秩相关（numpy 向量化）。

    ``left``/``right`` 必须**已经按日求过秩**（``rank(axis=1)``）。等价于
    逐日用 ``rank_ic``，但把"排秩"从「每个因子 × 每次刷新」降到「每因子一次」，
    并且归约走 numpy（DataFrame 的 ``sum(axis=1)`` 在大宽表上会反复转置），
    多因子回测里这是数量级的差别（5000 只 × 250 日窗口）。
    """
    l = left.to_numpy(dtype=np.float64, na_value=np.nan)
    r = right.to_numpy(dtype=np.float64, na_value=np.nan)
    mask = np.isfinite(l) & np.isfinite(r)
    n = mask.sum(axis=1)
    l = np.where(mask, l, np.nan)
    r = np.where(mask, r, np.nan)
    # 全空行（当日无有效截面）nanmean 会警告；结果本就是 NaN，静音即可
    all_nan = n == 0
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lc = l - np.nanmean(l, axis=1, keepdims=True)
        rc = r - np.nanmean(r, axis=1, keepdims=True)
    lc[all_nan] = 0.0
    rc[all_nan] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.nansum(lc * rc, axis=1)
        den = np.sqrt(np.nansum(lc * lc, axis=1) * np.nansum(rc * rc, axis=1))
        out = cov / np.where(den == 0, np.nan, den)
    out[n < max(min_count, 2)] = np.nan
    return pd.Series(out, index=left.index)


def _parse_factors(factors: Any) -> dict[str, dict[str, Any]]:
    """把 ``"momentum,low_vol"`` / ``["momentum"]`` / ``{"momentum": {"lookback": 60}}`` 统一成字典。"""
    if factors is None:
        return {}
    if isinstance(factors, str):
        items = [part.strip() for part in factors.split(",") if part.strip()]
        return {name: {} for name in items}
    if isinstance(factors, Mapping):
        return {str(name): dict(params or {}) for name, params in factors.items()}
    if isinstance(factors, Sequence):
        return {str(name): {} for name in factors}
    raise TypeError(f"无法解析 factors：{factors!r}")


def _parse_factor_weights(weights: Any, names: list[str]) -> dict[str, float] | None:
    if weights in (None, ""):
        return None
    if isinstance(weights, Mapping):
        return {str(k): float(v) for k, v in weights.items()}
    if isinstance(weights, str):
        values = [float(part) for part in weights.split(",") if part.strip()]
        if len(values) != len(names):
            raise ValueError(f"factor_weights 有 {len(values)} 个值，但因子有 {len(names)} 个")
        return {name: value for name, value in zip(names, values)}
    if isinstance(weights, Sequence):
        values = [float(v) for v in weights]
        if len(values) != len(names):
            raise ValueError("factor_weights 长度必须与 factors 一致")
        return dict(zip(names, values))
    raise TypeError(f"无法解析 factor_weights：{weights!r}")


class MultiFactorStrategy(Strategy):
    """日线低频多因子选股（默认等权合成、按周调仓、持仓带缓冲区）。"""

    name = "multifactor"

    def __init__(
        self,
        *,
        factors: Any = "momentum,low_vol,reversal",
        factor_weights: Any = None,
        weighting: str = "equal",
        ic_lookback: int = 250,
        ic_min_periods: int = 60,
        weight_refresh: int = 4,
        top_k: int = 20,
        buffer_rank: int | None = None,
        rebalance: str | None = "W-FRI",
        gross_exposure: float = 0.95,
        max_weight: float | None = 0.10,
        weight_scheme: str = "equal",
        winsorize_method: str | None = "mad",
        winsorize_n: float = 3.0,
        standardize: bool = True,
        rank: bool = False,
        neutralize_by: str | None = None,
        min_price: float = 2.0,
        min_amount: float = 0.0,
        min_history: int = 120,
        min_universe: int = 10,
    ) -> None:
        specs = _parse_factors(factors)
        if not specs:
            raise ValueError("至少需要一个因子")
        weight_map = _parse_factor_weights(factor_weights, list(specs))
        if weighting not in ("equal", "manual", "ic", "ir"):
            raise ValueError("weighting 只能是 equal / manual / ic / ir")
        if weight_scheme not in ("equal", "score"):
            raise ValueError("weight_scheme 只能是 equal / score")
        super().__init__(
            factors=",".join(specs),
            weighting=weighting,
            top_k=int(top_k),
            rebalance=rebalance,
            weight_scheme=weight_scheme,
        )
        self.factor_specs = specs
        self.factor_weights = weight_map
        self.weighting = weighting
        self.ic_lookback = int(ic_lookback)
        self.ic_min_periods = int(ic_min_periods)
        self.weight_refresh = max(1, int(weight_refresh))
        self.top_k = max(1, int(top_k))
        self.buffer_rank = int(buffer_rank) if buffer_rank else max(self.top_k * 2, self.top_k + 5)
        self.rebalance = rebalance
        self.gross_exposure = float(gross_exposure)
        self.max_weight = float(max_weight) if max_weight else None
        self.weight_scheme = weight_scheme
        self.winsorize_method = winsorize_method
        self.winsorize_n = float(winsorize_n)
        self.standardize = bool(standardize)
        self.rank = bool(rank)
        self.neutralize_by = neutralize_by or None
        self.min_price = float(min_price)
        self.min_amount = float(min_amount)
        self.min_history = int(min_history)
        self.min_universe = int(min_universe)
        self.last_diagnostics: dict[str, Any] = {}

    # -- 主流程 -------------------------------------------------------------

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        panel = build_panel(bars)
        if len(panel.dates) == 0:
            raise ValueError("bars 为空，无法生成多因子权重")
        raw = self._build_factors(panel)
        eligible = self._eligible(panel)
        mask = self.rebalance_mask(panel.dates, self.rebalance)
        rebalance_dates = [d for d in panel.dates if bool(mask.loc[d])]
        if not rebalance_dates:
            rebalance_dates = list(panel.dates)

        score_rows: list[pd.Series] = []
        used_dates: list[pd.Timestamp] = []
        weight_log: dict[str, dict[str, float]] = {}
        if self.weighting == "manual" and self.factor_weights:
            current_weights = {name: float(self.factor_weights.get(name, 0.0)) for name in raw}
        else:
            current_weights = equal_weights(raw.keys())
        forward = panel.forward_returns(1)
        # IC 加权要把因子逐日排秩做秩相关；排秩与刷新日无关，先整体排一次，
        # 否则每个因子 × 每次刷新都要重排 250×5000 的窗口（实测慢一个量级）。
        ranked: dict[str, pd.DataFrame] | None = None
        forward_rank: pd.DataFrame | None = None
        if self.weighting in ("ic", "ir"):
            ranked = {name: frame.rank(axis=1).astype("float32") for name, frame in raw.items()}
            forward_rank = forward.rank(axis=1).astype("float32")

        for index, day in enumerate(rebalance_dates):
            if self.weighting in ("ic", "ir") and (index % self.weight_refresh == 0 or not weight_log):
                current_weights = self._estimate_weights(
                    raw, forward, day, ranked=ranked, forward_rank=forward_rank
                )
                weight_log[str(day.date())] = current_weights
            row = None
            for name, frame in raw.items():
                weight = float(current_weights.get(name, 0.0))
                if weight == 0.0:
                    continue
                part = frame.loc[day] * weight
                row = part if row is None else row.add(part, fill_value=0.0)
            if row is None:
                continue
            score_rows.append(row)
            used_dates.append(day)

        if not score_rows:
            empty = pd.DataFrame(0.0, index=panel.dates, columns=panel.symbols)
            self.last_diagnostics = {"picks": {}, "reason": "没有任何可用打分"}
            return self.clip_weights(empty)

        scores = pd.DataFrame(score_rows, index=pd.DatetimeIndex(used_dates))
        picks, weights_rows, notes = self._select(scores, eligible, used_dates)
        targets = pd.DataFrame(0.0, index=panel.dates, columns=panel.symbols)
        if weights_rows:
            row_index = pd.DatetimeIndex([day for day, _ in weights_rows])
            row_values = [dict(weights) for _, weights in weights_rows]
            picked = pd.DataFrame(row_values, index=row_index)
            picked = picked.reindex(columns=panel.symbols).fillna(0.0)
            targets.loc[picked.index, picked.columns] = picked.to_numpy()
        targets = self.hold_to_rebalance(targets, self.rebalance)
        self.last_diagnostics = {
            "factors": list(self.factor_specs),
            "weights": current_weights,
            "factor_weights_by_date": weight_log,
            "picks": picks,
            "notes": notes,
            "rebalance_dates": [str(getattr(d, "date", lambda: d)()) for d in used_dates],
        }
        return self.clip_weights(targets)

    # -- 因子计算 -----------------------------------------------------------

    def _build_factors(self, panel: Panel) -> dict[str, pd.DataFrame]:
        exposure = None
        if self.neutralize_by:
            if self.neutralize_by in self.factor_specs:
                exposure = preprocess(
                    compute(self.neutralize_by, panel, **self.factor_specs[self.neutralize_by]),
                    winsorize_method=self.winsorize_method,
                    winsorize_n=self.winsorize_n,
                    standardize_=True,
                )
            else:
                exposure = preprocess(
                    compute(self.neutralize_by, panel),
                    winsorize_method=self.winsorize_method,
                    winsorize_n=self.winsorize_n,
                    standardize_=True,
                )
        out: dict[str, pd.DataFrame] = {}
        for name, params in self.factor_specs.items():
            frame = compute(name, panel, **params)
            out[name] = preprocess(
                frame,
                winsorize_method=self.winsorize_method,
                winsorize_n=self.winsorize_n,
                exposure=None if name == self.neutralize_by else exposure,
                standardize_=self.standardize,
                rank=self.rank,
                min_count=max(5, min(20, len(panel.symbols) // 2)),
            )
        return out

    # -- 股票池 -------------------------------------------------------------

    def _eligible(self, panel: Panel) -> pd.DataFrame:
        """当日可交易的股票池掩码（停牌、价格/成交额不达标、上市不足一并剔除）。"""
        mask = panel.tradable.copy()
        if self.min_price > 0:
            mask &= panel.close >= self.min_price
        if self.min_amount > 0:
            mask &= panel.amount >= self.min_amount
        if self.min_history > 1:
            history = panel.close.notna().cumsum()
            mask &= history >= self.min_history
        return mask

    # -- 因子权重（只用历史信息） -------------------------------------------

    def _estimate_weights(
        self,
        raw: Mapping[str, pd.DataFrame],
        forward: pd.DataFrame,
        day: pd.Timestamp,
        *,
        ranked: Mapping[str, pd.DataFrame] | None = None,
        forward_rank: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        if self.weighting == "manual" and self.factor_weights:
            return dict(self.factor_weights)
        if self.weighting == "equal":
            return equal_weights(raw.keys())
        known = forward.index[forward.index < day]
        window = known[-self.ic_lookback :]
        if len(window) < self.ic_min_periods:
            return equal_weights(raw.keys())
        fwd_rank = forward_rank.loc[window] if forward_rank is not None else forward.loc[window].rank(axis=1)
        raw_scores: dict[str, float] = {}
        for name, frame in raw.items():
            window_rank = ranked[name].loc[window] if ranked is not None else frame.loc[window].rank(axis=1)
            series = _row_spearman(window_rank, fwd_rank)
            if series.notna().sum() == 0:
                raw_scores[name] = 0.0
                continue
            mean = float(series.mean())
            std = float(series.std(ddof=1)) if series.notna().sum() > 1 else 0.0
            if self.weighting == "ir" and np.isfinite(std) and std > 0:
                value = mean / std
            else:
                value = mean
            if not np.isfinite(value) or value <= 0:
                raw_scores[name] = 0.0
            else:
                raw_scores[name] = float(value)
        total = sum(raw_scores.values())
        if total <= 0:
            return equal_weights(raw.keys())
        return {name: value / total for name, value in raw_scores.items()}

    # -- 选股与赋权 ---------------------------------------------------------

    def _select(
        self,
        scores: pd.DataFrame,
        eligible: pd.DataFrame,
        dates: Sequence[pd.Timestamp],
    ) -> tuple[dict[str, list[str]], list[tuple[pd.Timestamp, dict[str, float]]], list[str]]:
        held: list[str] = []
        pick_log: dict[str, list[str]] = {}
        weight_rows: list[tuple[pd.Timestamp, dict[str, float]]] = []
        notes: list[str] = []
        for day in dates:
            if day not in eligible.index:
                continue
            ok = eligible.loc[day]
            row = scores.loc[day].where(ok).dropna().sort_values(ascending=False)
            ranked = list(row.index)
            if len(ranked) < max(self.min_universe, 1):
                notes.append(f"{day.date()} 合格标的只有 {len(ranked)} 只，本期空仓")
                pick_log[str(day.date())] = []
                weight_rows.append((day, {}))
                held = []
                continue
            buffer_set = set(ranked[: self.buffer_rank])
            keep = [symbol for symbol in held if symbol in buffer_set]
            need = self.top_k - len(keep)
            if need > 0:
                chosen = set(keep)
                for symbol in ranked:
                    if need <= 0:
                        break
                    if symbol in chosen:
                        continue
                    keep.append(symbol)
                    chosen.add(symbol)
                    need -= 1
            picks = keep[: self.top_k]
            held = picks
            weights = self._weights_for(picks, row)
            pick_log[str(day.date())] = picks
            weight_rows.append((day, weights))
        return pick_log, weight_rows, notes

    def _weights_for(self, picks: Sequence[str], scores: pd.Series) -> dict[str, float]:
        if not picks:
            return {}
        if self.weight_scheme == "score":
            values = scores.reindex(picks).astype(float)
            shifted = values - values.min() + 1e-6
            raw = (shifted / shifted.sum()).to_dict()
        else:
            raw = {symbol: 1.0 / len(picks) for symbol in picks}
        scaled = {symbol: weight * self.gross_exposure for symbol, weight in raw.items()}
        return self._cap_weights(scaled)

    def _cap_weights(self, weights: Mapping[str, float]) -> dict[str, float]:
        """单票权重上限：超限的部分按比例分给未触顶的标的（最多迭代 20 轮）。"""
        if not self.max_weight or not weights:
            return dict(weights)
        capped = dict(weights)
        cap = self.max_weight
        for _ in range(20):
            over = {symbol: value for symbol, value in capped.items() if value > cap + 1e-9}
            if not over:
                break
            excess = sum(value - cap for value in over.values())
            for symbol in over:
                capped[symbol] = cap
            room = {symbol: cap - value for symbol, value in capped.items() if value < cap - 1e-9}
            if not room or excess <= 0:
                break
            total_room = sum(room.values())
            for symbol, space in room.items():
                capped[symbol] += excess * (space / total_room)
        return capped
