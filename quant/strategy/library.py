"""内置策略库。都在 A 股日线上可直接跑，参数少、逻辑透明。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import CompositeStrategy, Strategy
from .multifactor import MultiFactorStrategy


def _close_matrix(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({s: f["close"] for s, f in bars.items()}).sort_index()


def _col(bars: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    return pd.DataFrame({s: f[field] for s, f in bars.items() if field in f.columns}).sort_index()


class BuyAndHold(Strategy):
    """等权买入并持有（基准）。"""

    name = "buy_and_hold"

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = _close_matrix(bars)
        n = max(len(close.columns), 1)
        w = pd.DataFrame(1.0 / n, index=close.index, columns=close.columns)
        w.iloc[0] = 1.0 / n
        return self.clip_weights(w)


class MaCross(Strategy):
    """双均线趋势：快线在慢线上方持有。"""

    name = "ma_cross"

    def __init__(self, fast: int = 20, slow: int = 60, rebalance: str | None = None, **params) -> None:
        super().__init__(fast=fast, slow=slow, rebalance=rebalance, **params)
        self.fast, self.slow, self.rebalance_freq = int(fast), int(slow), rebalance

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = _close_matrix(bars)
        held = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for s in close.columns:
            c = close[s]
            held[s] = (ind.sma(c, self.fast) > ind.sma(c, self.slow)).astype(float)
        held = held.where(close.notna(), 0.0)
        count = held.sum(axis=1).replace(0.0, np.nan)
        w = held.div(count, axis=0).fillna(0.0)  # 等权分配在“有信号”的标的之间
        return self.clip_weights(self.hold_to_rebalance(w, self.rebalance_freq))


class MacdTrend(Strategy):
    """MACD 金叉持有、死叉离场。"""

    name = "macd_trend"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, **params) -> None:
        super().__init__(fast=fast, slow=slow, signal=signal, **params)
        self.fast, self.slow, self.signal = int(fast), int(slow), int(signal)

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = _close_matrix(bars)
        held = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for s in close.columns:
            m = ind.macd(close[s], self.fast, self.slow, self.signal)
            raw = (m["dif"] > m["dea"]).astype(float)
            held[s] = raw.where(m["dea"].notna(), 0.0)
        count = held.sum(axis=1).replace(0.0, np.nan)
        return self.clip_weights(held.div(count, axis=0).fillna(0.0))


class RsiReversion(Strategy):
    """RSI 超卖买入、超买卖出（均值回复）。"""

    name = "rsi_reversion"

    def __init__(self, period: int = 14, lower: float = 30.0, upper: float = 70.0, **params) -> None:
        super().__init__(period=period, lower=lower, upper=upper, **params)
        self.period, self.lower, self.upper = int(period), float(lower), float(upper)

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = _close_matrix(bars)
        state = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for s in close.columns:
            r = ind.rsi(close[s], self.period)
            holding = 0.0
            col = np.zeros(len(r))
            for i, value in enumerate(r.to_numpy()):
                if np.isnan(value):
                    col[i] = 0.0
                    continue
                if holding == 0.0 and value < self.lower:
                    holding = 1.0
                elif holding == 1.0 and value > self.upper:
                    holding = 0.0
                col[i] = holding
            state[s] = col
        count = state.sum(axis=1).replace(0.0, np.nan)
        return self.clip_weights(state.div(count, axis=0).fillna(0.0))


class BollingerReversion(Strategy):
    """跌破下轨买入，回到中轨卖出。"""

    name = "boll_reversion"

    def __init__(self, period: int = 20, k: float = 2.0, **params) -> None:
        super().__init__(period=period, k=k, **params)
        self.period, self.k = int(period), float(k)

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = _close_matrix(bars)
        state = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for s in close.columns:
            b = ind.bollinger(close[s], self.period, self.k)
            c = close[s]
            holding = 0.0
            col = np.zeros(len(c))
            for i in range(len(c)):
                if np.isnan(b["lower"].iloc[i]):
                    continue
                if holding == 0.0 and c.iloc[i] < b["lower"].iloc[i]:
                    holding = 1.0
                elif holding == 1.0 and c.iloc[i] >= b["mid"].iloc[i]:
                    holding = 0.0
                col[i] = holding
            state[s] = col
        count = state.sum(axis=1).replace(0.0, np.nan)
        return self.clip_weights(state.div(count, axis=0).fillna(0.0))


class DonchianBreakout(Strategy):
    """突破 N 日高点买入，跌破 M 日低点离场（海龟简化版）。"""

    name = "donchian"

    def __init__(self, entry: int = 20, exit: int = 10, **params) -> None:
        super().__init__(entry=entry, exit=exit, **params)
        self.entry, self.exit = int(entry), int(exit)

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = _col(bars, "high")
        low = _col(bars, "low")
        state = pd.DataFrame(0.0, index=high.index, columns=high.columns)
        for s in high.columns:
            h, l = high[s], low[s]
            up = h.rolling(self.entry, min_periods=self.entry).max().shift(1)
            dn = l.rolling(self.exit, min_periods=self.exit).min().shift(1)
            holding = 0.0
            col = np.zeros(len(h))
            for i in range(len(h)):
                if holding == 0.0 and not np.isnan(up.iloc[i]) and h.iloc[i] > up.iloc[i]:
                    holding = 1.0
                elif holding == 1.0 and not np.isnan(dn.iloc[i]) and l.iloc[i] < dn.iloc[i]:
                    holding = 0.0
                col[i] = holding
            state[s] = col
        count = state.sum(axis=1).replace(0.0, np.nan)
        return self.clip_weights(state.div(count, axis=0).fillna(0.0))


class MomentumRank(Strategy):
    """动量轮动：按 N 日收益排序，持有最强的 K 只，等权。

    两个口径要分清：

    * 默认（``concentrate=False``）——每只分到 ``1/top_k`` 的仓位，**信号不足时差额留现金**。
      例如 top_k=2 但只有 1 只有正动量，则拿 50% 仓位 + 50% 现金；
      一只都没有正动量就空仓。这是「只在有把握时出手」的口径。
    * ``concentrate=True``——选中的标的等分满仓（1 只就 100%）。回撤会明显更大。
    """

    name = "momentum_rank"

    def __init__(
        self,
        lookback: int = 60,
        top_k: int = 2,
        rebalance: str = "W-FRI",
        skip: int = 0,
        vol_adjust: bool = False,
        concentrate: bool = False,
        **params,
    ) -> None:
        super().__init__(
            lookback=lookback, top_k=top_k, rebalance=rebalance, skip=skip,
            vol_adjust=vol_adjust, concentrate=concentrate, **params,
        )
        self.lookback, self.top_k = int(lookback), int(top_k)
        self.rebalance_freq, self.skip = rebalance, int(skip)
        self.vol_adjust = bool(vol_adjust)
        self.concentrate = bool(concentrate)

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = _close_matrix(bars)
        score = close.shift(self.skip) / close.shift(self.lookback + self.skip) - 1.0
        weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        k = min(self.top_k, len(close.columns))
        if k <= 0:
            return weights
        for dt, row in score.iterrows():
            valid = row.dropna()
            if valid.empty:
                continue
            picks = valid.nlargest(k)
            picks = picks[picks > 0.0]  # 只做正动量
            if picks.empty:
                continue
            # 组合总仓位：concentrate 时满仓，否则按「选中只数 / 名义只数」给，差额留现金
            target_total = 1.0 if self.concentrate else len(picks) / k
            if self.vol_adjust:
                vols = {}
                for s in picks.index:
                    c = close[s].loc[:dt].tail(20)
                    ret = c.pct_change().dropna()
                    vols[s] = float(ret.std(ddof=1)) if len(ret) > 1 else np.nan
                inv = pd.Series({s: 1.0 / v for s, v in vols.items() if v and v > 0}).replace([np.inf], np.nan).dropna()
                if not inv.empty:
                    weights.loc[dt, inv.index] = ((inv / inv.sum()) * target_total).values
                    continue
            weights.loc[dt, picks.index] = target_total / len(picks)
        return self.clip_weights(self.hold_to_rebalance(weights, self.rebalance_freq))


class FactorScore(Strategy):
    """多因子打分选股：动量 / 反转 / 低波动 / 量能，加权后取前 K 名。"""

    name = "factor_score"

    def __init__(
        self,
        top_k: int = 3,
        rebalance: str = "W-FRI",
        momentum_window: int = 60,
        reversal_window: int = 5,
        vol_window: int = 20,
        weights: dict[str, float] | None = None,
        **params,
    ) -> None:
        super().__init__(top_k=top_k, rebalance=rebalance, weights=weights, **params)
        self.top_k, self.rebalance_freq = int(top_k), rebalance
        self.momentum_window, self.reversal_window, self.vol_window = int(momentum_window), int(reversal_window), int(vol_window)
        self.factor_weights = weights or {"momentum": 1.0, "reversal": 0.5, "low_vol": 0.5, "volume": 0.3}

    def _z(self, frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1)
        sd = frame.std(axis=1, ddof=1).replace(0.0, np.nan)
        return frame.sub(mean, axis=0).div(sd, axis=0)

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = _close_matrix(bars)
        volume = _col(bars, "volume")
        factors: dict[str, pd.DataFrame] = {
            "momentum": self._z(close / close.shift(self.momentum_window) - 1.0),
            "reversal": -self._z(close / close.shift(self.reversal_window) - 1.0),
            "low_vol": -self._z(close.pct_change().rolling(self.vol_window, min_periods=self.vol_window).std(ddof=1)),
        }
        if not volume.empty:
            factors["volume"] = self._z(volume.rolling(self.vol_window, min_periods=self.vol_window).mean())

        score = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        total_w = 0.0
        for name, weight in self.factor_weights.items():
            if name in factors and weight:
                score = score.add(factors[name].reindex_like(score) * weight, fill_value=0.0)
                total_w += abs(weight)
        if total_w:
            score = score / total_w

        weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        k = min(self.top_k, len(close.columns))
        for dt, row in score.iterrows():
            valid = row.dropna()
            if len(valid) < 1:
                continue
            picks = valid.nlargest(k)
            picks = picks[picks > 0.0]
            if picks.empty:
                continue
            weights.loc[dt, picks.index] = 1.0 / len(picks)
        return self.clip_weights(self.hold_to_rebalance(weights, self.rebalance_freq))


class TrendFilteredMomentum(Strategy):
    """大盘择时 + 个股动量：指数在均线上才持仓，否则空仓（A股最常见的降回撤手段）。"""

    name = "trend_filtered_momentum"

    def __init__(
        self,
        lookback: int = 60,
        top_k: int = 2,
        rebalance: str = "W-FRI",
        regime_window: int = 60,
        regime_symbol: str | None = None,
        **params,
    ) -> None:
        super().__init__(lookback=lookback, top_k=top_k, rebalance=rebalance, regime_window=regime_window, **params)
        self.inner = MomentumRank(lookback=lookback, top_k=top_k, rebalance=rebalance)
        self.regime_window = int(regime_window)
        self.regime_symbol = regime_symbol

    def generate(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        weights = self.inner.generate(bars)
        close = _close_matrix(bars)
        ref = self.regime_symbol if self.regime_symbol in close.columns else None
        if ref is None:
            # 用全池等权指数当择时基准
            market = close.pct_change().mean(axis=1).add(1.0).cumprod()
        else:
            market = close[ref]
        ma = ind.sma(market, self.regime_window)
        risk_on = (market > ma).reindex(weights.index).fillna(False).astype(float)
        return self.clip_weights(weights.mul(risk_on, axis=0))


BUILTIN: dict[str, type[Strategy]] = {
    "buy_and_hold": BuyAndHold,
    "ma_cross": MaCross,
    "macd_trend": MacdTrend,
    "rsi_reversion": RsiReversion,
    "boll_reversion": BollingerReversion,
    "donchian": DonchianBreakout,
    "momentum_rank": MomentumRank,
    "factor_score": FactorScore,
    "trend_filtered_momentum": TrendFilteredMomentum,
    "multifactor": MultiFactorStrategy,
}


def build(name: str, **params) -> Strategy:
    """按名字建策略，供命令行/配置用。"""
    key = name.strip().lower()
    if key not in BUILTIN:
        raise KeyError(f"未知策略 {name}；可选：{', '.join(sorted(BUILTIN))}")
    return BUILTIN[key](**params)


__all__ = [
    "BUILTIN",
    "BollingerReversion",
    "BuyAndHold",
    "CompositeStrategy",
    "DonchianBreakout",
    "FactorScore",
    "MaCross",
    "MacdTrend",
    "MomentumRank",
    "MultiFactorStrategy",
    "RsiReversion",
    "Strategy",
    "TrendFilteredMomentum",
    "build",
]
