"""因子预处理：去极值 → 中性化 → 标准化。

这三步是券商/公募多因子流水线的标准动作，顺序不能乱：

1. **去极值**（winsorize）：日线因子里总有脏数据（一字板、复权跳变），
   用 MAD 或分位数把两端压住，否则一个异常值就能主导当天的横截面排序；
2. **中性化**（neutralize）：把因子里能被「风格暴露」解释的部分回归掉，
   留下纯残差。没有行业分类与市值数据时，可以用量价因子（如波动、流动性）
   当暴露的代理——这是**近似**，接上真实行业/市值数据后应替换；
3. **标准化**（standardize）：横截面 z-score，让不同量纲的因子可加权相加。

所有操作都是**逐日横截面**（`axis=1`），不含任何跨期信息，天然无前视。
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

__all__ = ["fill_missing", "neutralize", "preprocess", "rank_normalize", "standardize", "winsorize"]


def winsorize(
    frame: pd.DataFrame,
    *,
    method: str = "mad",
    n: float = 3.0,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """逐日横截面去极值。``method='mad'``（中位数±n×1.4826×MAD）或 ``'quantile'``。"""
    if method not in ("mad", "quantile"):
        raise ValueError("method 只能是 'mad' 或 'quantile'")
    if method == "mad":
        median = frame.median(axis=1)
        mad = (frame.sub(median, axis=0)).abs().median(axis=1)
        spread = (mad * 1.4826).replace(0, np.nan)
        lower = median - n * spread
        upper = median + n * spread
    else:
        lower = frame.quantile(lower_quantile, axis=1)
        upper = frame.quantile(upper_quantile, axis=1)
    clipped = frame.clip(lower=lower, upper=upper, axis=0)
    # 极值被压住的格点在 MAD=0（如整行同值）时会变成 NaN，退回原值
    return clipped.where(clipped.notna(), frame)


def standardize(frame: pd.DataFrame, *, ddof: int = 0, min_count: int = 2) -> pd.DataFrame:
    """逐日横截面 z-score。"""
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=ddof).replace(0, np.nan)
    enough = frame.notna().sum(axis=1) >= min_count
    z = frame.sub(mean, axis=0).div(std, axis=0)
    return z.where(enough, np.nan)


def rank_normalize(frame: pd.DataFrame, *, min_count: int = 2) -> pd.DataFrame:
    """逐日横截面转分位：线性映射到 [-1, 1]（对异常值最稳健）。"""
    ranked = frame.rank(axis=1, pct=True)
    centered = 2.0 * ranked - 1.0
    enough = frame.notna().sum(axis=1) >= min_count
    return centered.where(enough, np.nan)


def neutralize(
    frame: pd.DataFrame,
    exposure: pd.DataFrame | Mapping[str, pd.DataFrame] | None,
    *,
    method: str = "ols",
    buckets: int = 5,
    min_count: int = 10,
) -> pd.DataFrame:
    """对风格暴露做逐日横截面中性化。

    * ``method='ols'``：对暴露（可多列）做最小二乘回归，取残差；
    * ``method='bucket'``：按暴露分 ``buckets`` 组，各组内减去组内中位数
      （等价于「行业中性」的简化版，稳健但信息损失略大）。

    ``exposure=None`` 时原样返回。
    """
    if exposure is None:
        return frame
    expo = exposure if isinstance(exposure, pd.DataFrame) else pd.concat(exposure, axis=1)
    expo = expo.reindex(index=frame.index, columns=frame.columns)
    out = frame.copy()
    for date in frame.index:
        row = frame.loc[date]
        valid = row.notna()
        if valid.sum() < min_count:
            continue
        if method == "ols":
            ex = expo.loc[date].reindex(row.index)[valid]
            usable = ex.notna().all(axis=1) if isinstance(ex, pd.DataFrame) else ex.notna()
            index = row.index[valid][usable]
            if len(index) < min_count:
                continue
            y = row.reindex(index).to_numpy(dtype=float)
            ex_use = expo.loc[date].reindex(index)
            matrix = ex_use.to_numpy(dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            design = np.column_stack([np.ones(len(index)), matrix])
            try:
                beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            out.loc[date, index] = y - design @ beta
        elif method == "bucket":
            ex = expo.loc[date].reindex(row.index)[valid]
            usable = ex.notna() if not isinstance(ex, pd.DataFrame) else ex.iloc[:, 0].notna()
            index = row.index[valid][usable]
            values = row.reindex(index)
            ex_values = ex.reindex(index)
            if isinstance(ex_values, pd.DataFrame):
                ex_values = ex_values.iloc[:, 0]
            if len(index) < buckets * 2:
                continue
            try:
                groups = pd.qcut(ex_values.rank(method="first"), buckets, labels=False)
            except ValueError:
                continue
            medians = values.groupby(groups).transform("median")
            out.loc[date, index] = values - medians
        else:
            raise ValueError("method 只能是 'ols' 或 'bucket'")
    return out


def fill_missing(frame: pd.DataFrame, *, limit: int = 5) -> pd.DataFrame:
    """按标的（列）向前填补有限天数：停牌期间因子沿用最近有效值。"""
    return frame.ffill(limit=limit)


def preprocess(
    frame: pd.DataFrame,
    *,
    winsorize_method: str | None = "mad",
    winsorize_n: float = 3.0,
    exposure: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
    neutralize_method: str = "ols",
    standardize_: bool = True,
    rank: bool = False,
    fill_limit: int = 0,
    min_count: int = 5,
) -> pd.DataFrame:
    """完整预处理流水线：填补 → 去极值 → 中性化 → 标准化（或分位化）。

    每一步都可以关掉；``min_count`` 是横截面最少有效标的数，不够就整行置 NaN
    （样本太少的横截面排序没有意义）。
    """
    out = frame.copy()
    if fill_limit:
        out = fill_missing(out, limit=fill_limit)
    enough = out.notna().sum(axis=1) >= min_count
    if winsorize_method:
        out = winsorize(out, method=winsorize_method, n=winsorize_n)
    if exposure is not None:
        out = neutralize(out, exposure, method=neutralize_method, min_count=min_count)
    if rank:
        out = rank_normalize(out, min_count=min_count)
    elif standardize_:
        out = standardize(out, min_count=min_count)
    return out.where(enough, np.nan)


def apply_chain(
    frame: pd.DataFrame,
    steps: Iterable[tuple[str, Mapping[str, object]]],
) -> pd.DataFrame:
    """按 ``[("winsorize", {...}), ("standardize", {})]`` 这样的步骤链处理（插件/配置用）。"""
    out = frame
    for name, kwargs in steps:
        fn = {
            "winsorize": winsorize,
            "standardize": standardize,
            "rank": rank_normalize,
            "neutralize": neutralize,
            "fill": fill_missing,
            "preprocess": preprocess,
        }.get(name)
        if fn is None:
            raise KeyError(f"未知的预处理步骤 {name!r}")
        out = fn(out, **dict(kwargs))
    return out
