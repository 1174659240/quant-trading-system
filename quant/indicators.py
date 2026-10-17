"""技术指标库：全部输入输出都是 ``pandas`` 序列，便于向量化与单元测试。

约定：输入 ``pd.Series``（价格/成交量），输出同索引的新序列；第一个不可用
的位置是 ``NaN``，不做隐式填充——信号层自己决定怎么处理预热期。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 244


# ---------------------------------------------------------------- 均线族
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period, min_periods=period).apply(
        lambda w: float(np.dot(w, weights) / weights.sum()), raw=True
    )


def rolling_std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).std(ddof=1)


# ---------------------------------------------------------------- 动量/振荡
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI。"""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss != 0.0, 100.0)


def roc(series: pd.Series, period: int = 20) -> pd.Series:
    return series.pct_change(period)


def momentum(series: pd.Series, period: int = 20) -> pd.Series:
    return series / series.shift(period) - 1.0


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    dif = ema(series, fast) - ema(series, slow)
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": (dif - dea) * 2.0})


def kdj(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9, k_period: int = 3, d_period: int = 3
) -> pd.DataFrame:
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    span = (highest - lowest).replace(0.0, np.nan)
    rsv = (close - lowest) / span * 100.0
    rsv = rsv.fillna(50.0)
    k = rsv.ewm(alpha=1.0 / k_period, adjust=False).mean()
    d = k.ewm(alpha=1.0 / d_period, adjust=False).mean()
    return pd.DataFrame({"k": k, "d": d, "j": 3.0 * k - 2.0 * d})


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period, min_periods=period).mean()
    md = (tp - ma).abs().rolling(period, min_periods=period).mean()
    return (tp - ma) / (0.015 * md.replace(0.0, np.nan))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    span = (highest - lowest).replace(0.0, np.nan)
    return -100.0 * (highest - close) / span


# ---------------------------------------------------------------- 波动/通道
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def bollinger(series: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(series, period)
    sd = rolling_std(series, period)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid.replace(0.0, np.nan)
    pctb = (series - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "width": width, "pctb": pctb})


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2.0})


def realized_vol(series: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    ret = series.pct_change()
    vol = ret.rolling(period, min_periods=period).std(ddof=1)
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period, min_periods=period).mean()
    sd = rolling_std(series, period)
    return (series - mean) / sd.replace(0.0, np.nan)


# ---------------------------------------------------------------- 量能
def volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(period, min_periods=period).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    flow = tp * volume
    up = flow.where(tp.diff() > 0, 0.0).rolling(period, min_periods=period).sum()
    down = flow.where(tp.diff() < 0, 0.0).rolling(period, min_periods=period).sum()
    ratio = up / down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + ratio)


# ---------------------------------------------------------------- 组合工具
def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """fast 上穿 slow → True。"""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def resample_ohlcv(frame: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    """日线聚合成周/月线（open 取首、high 取最大、close 取末、volume/amount 求和）。"""
    agg: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "amount" in frame.columns:
        agg["amount"] = "sum"
    out = frame.resample(rule).agg(agg).dropna(subset=["close"])
    out["prev_close"] = out["close"].shift(1)
    if len(out):
        out.iloc[0, out.columns.get_loc("prev_close")] = out.iloc[0]["open"]
    out.index.name = "date"
    return out


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    """给一份 OHLCV 表附上常用指标列（列名带参数后缀，不覆盖原列）。"""
    close, high, low, volume = frame["close"], frame["high"], frame["low"], frame["volume"]
    out = frame.copy()
    out["ma5"] = sma(close, 5)
    out["ma10"] = sma(close, 10)
    out["ma20"] = sma(close, 20)
    out["ma60"] = sma(close, 60)
    out["ema12"] = ema(close, 12)
    out["ema26"] = ema(close, 26)
    macd_df = macd(close)
    out["macd_dif"], out["macd_dea"], out["macd_hist"] = macd_df["dif"], macd_df["dea"], macd_df["hist"]
    out["rsi14"] = rsi(close, 14)
    boll = bollinger(close)
    out["boll_mid"], out["boll_up"], out["boll_low"], out["boll_width"], out["boll_pctb"] = (
        boll["mid"], boll["upper"], boll["lower"], boll["width"], boll["pctb"],
    )
    out["atr14"] = atr(high, low, close, 14)
    out["vol20"] = realized_vol(close, 20)
    kdj_df = kdj(high, low, close)
    out["kdj_k"], out["kdj_d"], out["kdj_j"] = kdj_df["k"], kdj_df["d"], kdj_df["j"]
    out["vol_ma20"] = volume_ma(volume, 20)
    out["obv"] = obv(close, volume)
    out["ret1"] = close.pct_change()
    out["ret5"] = close.pct_change(5)
    out["ret20"] = close.pct_change(20)
    return out
