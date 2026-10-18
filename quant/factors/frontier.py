"""前沿（国外文献）因子：只用日线 OHLCV 可实现的部分。

逐条对应文献（详见 ``docs/风险清仓与前沿因子.md``）：

- ``overnight_ret_20``：Lou, Polk & Skouras (2019, JFE) "A Tug of War: Overnight versus
  Intraday Expected Returns"——隔夜收益由散户/被动资金主导，隔夜收益越高、未来收益越低。
- ``intraday_ret_20``：同上，日内收益由机构主导，方向相反。
- ``high_52w``：George & Hwang (2004, JF)——距 52 周最高价越近，未来收益越高（锚定效应）。
- ``residual_momentum``：Blitz, Huij & Martens (2011, JEF)——剔除市场 beta 后的 12-1 动量。
- ``downside_beta``：Ang, Chen & Xing (2006, RFS)——下行 beta 越高，预期收益越低。
- ``ivol_250``：Ang, Hodrick, Xing & Zhang (2006, JF)——特质波动率与收益负相关。
- ``fip_momentum``：Da, Gurun & Warachka (2014, RFS) "Frog in the Pan"——信息越连续
  （ID 越低）动量越强；用 ``mom × (1 - ID)`` 做连续化处理。

所有因子都遵守库约定：``compute(name, panel) -> DataFrame``，``sign`` 统一成「越大越看多」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.factors.library import register_factor

TRADING_DAYS = 244


def _market_returns(panel) -> pd.Series:
    return panel.returns.where(panel.tradable.reindex_like(panel.returns).fillna(False)).mean(axis=1)


def _beta(returns: pd.DataFrame, market: pd.Series, window: int = 250, min_periods: int = 120) -> pd.DataFrame:
    mkt = market.reindex(returns.index)
    var = mkt.rolling(window, min_periods=min_periods).var()
    cov = returns.rolling(window, min_periods=min_periods).cov(mkt)
    return cov.div(var.replace(0, np.nan), axis=0)


def overnight_ret_20(panel, *, window: int = 20) -> pd.DataFrame:
    overnight = np.log(panel.open / panel.close.shift(1))
    return overnight.rolling(window, min_periods=max(5, window // 2)).mean()


def intraday_ret_20(panel, *, window: int = 20) -> pd.DataFrame:
    intraday = np.log(panel.close / panel.open)
    return intraday.rolling(window, min_periods=max(5, window // 2)).mean()


def high_52w(panel, *, window: int = 250) -> pd.DataFrame:
    peak = panel.high.rolling(window, min_periods=max(100, window // 2)).max()
    return panel.close / peak.replace(0, np.nan)


def residual_momentum(panel, *, window: int = 250, skip: int = 20) -> pd.DataFrame:
    mom = panel.close.shift(skip) / panel.close.shift(window) - 1.0
    mkt = _market_returns(panel)
    mkt_mom = mkt.rolling(window, min_periods=120).sum()  # 近似：市场同窗口累计收益
    beta = _beta(panel.returns, mkt, window)
    return mom - beta.mul(mkt_mom, axis=0)


def downside_beta(panel, *, window: int = 250, min_periods: int = 30) -> pd.DataFrame:
    mkt = _market_returns(panel)
    down = mkt < 0
    mkt_down = mkt.where(down)
    ret_down = panel.returns.where(down, axis=0)
    var = mkt_down.rolling(window, min_periods=min_periods).var()
    cov = ret_down.rolling(window, min_periods=min_periods).cov(mkt_down)
    return cov.div(var.replace(0, np.nan), axis=0)


def ivol_250(panel, *, window: int = 250) -> pd.DataFrame:
    mkt = _market_returns(panel)
    beta = _beta(panel.returns, mkt, window)
    resid = panel.returns - beta.mul(mkt, axis=0)
    return resid.rolling(window, min_periods=120).std(ddof=1) * np.sqrt(TRADING_DAYS)


def fip_momentum(panel, *, window: int = 250, skip: int = 20) -> pd.DataFrame:
    """``mom × (1 - ID)``：ID = sign(mom) ×（跌日占比 - 涨日占比），越负越「连续」。"""
    rets = panel.returns
    mom = panel.close.shift(skip) / panel.close.shift(window) - 1.0
    pos = (rets > 0).rolling(window, min_periods=120).mean()
    neg = (rets < 0).rolling(window, min_periods=120).mean()
    discreteness = np.sign(mom) * (neg - pos)
    return mom * (1.0 - discreteness)


def register() -> None:
    register_factor(
        "overnight_ret_20",
        overnight_ret_20,
        sign=-1,
        category="前沿",
        description="过去 20 日平均隔夜收益（Lou-Polk-Skouras 2019，取负）",
    )
    register_factor(
        "intraday_ret_20",
        intraday_ret_20,
        sign=-1,
        category="前沿",
        description="过去 20 日平均日内收益（A 股实测为反转，故取负；Lou-Polk-Skouras 2019 的方向在此不成立）",
    )
    register_factor(
        "high_52w", high_52w, sign=1, category="前沿", description="收盘价距 52 周最高价的比例（George-Hwang 2004）"
    )
    register_factor(
        "residual_momentum",
        residual_momentum,
        sign=1,
        category="前沿",
        description="12-1 动量减去 beta×市场动量（Blitz et al. 2011）",
    )
    register_factor(
        "downside_beta",
        downside_beta,
        sign=-1,
        category="前沿",
        description="下行 beta（Ang-Chen-Xing 2006，取负）",
    )
    register_factor(
        "ivol_250", ivol_250, sign=-1, category="前沿", description="250 日特质波动率（Ang et al. 2006，取负）"
    )
    register_factor(
        "fip_momentum",
        fip_momentum,
        sign=1,
        category="前沿",
        description="连续信息动量 mom×(1-ID)（Da-Gurun-Warachka 2014）",
    )
