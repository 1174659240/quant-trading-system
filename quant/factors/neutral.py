"""行业 + 市值中性化（Frisch–Waugh–Lovell 两步法）。

目标回归：

    y = α + Σ 行业哑变量 + β · ln(总市值) + ε

直接对 84 个行业哑变量做最小二乘，每天要解一个 5000×85 的方程，24 个因子 × 2187 天
会把回测时间放大一个量级。FWL 定理给了等价的两步做法：

1. **行业组内去均值**：等价于回归掉全部行业哑变量（含截距）；
2. **对同样去均值后的 ln(市值) 回归**，取残差。

两步法只需 ``np.bincount`` 求组内均值 + 一次标量回归，实测 24 个因子 × 2187 天 ≈ 数秒。
**数学上等价于一次性全回归**，不是近似。

三个必须守住的规矩：

* 组内去均值前必须先按「当日可计算」过滤，否则停牌/缺数据的格子会把组均值拉偏；
* 行业标签缺失（``NaN``）单独成一组（代码 0），不能直接丢掉——丢掉等于按行业分位筛选；
* 全市场市值同向变动时 ``β·ln(市值)`` 那一项对排序影响很小，但**不能省**：市值因子（``size``）
  如果自己不做市值中性化，等于拿「市值」解释「市值」。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["codes_from_labels", "residualize", "residualize_many"]


def codes_from_labels(labels: pd.DataFrame, *, fill: int = 0) -> pd.DataFrame:
    """把行业标签矩阵（日期 × 代码，值为字符串）转成整数代码矩阵，缺失归入 0 组。

    整个矩阵共用一个映射（同一天内同一行业代码相同，跨天也稳定），
    避免逐日 ``factorize`` 造成代码漂移。
    """
    flat = labels.to_numpy(dtype=object)
    values, codes = np.unique(
        np.where(pd.isna(flat), "", flat.astype(str)), return_inverse=True
    )
    codes = codes.reshape(flat.shape) + (1 if values[0] == "" else 0)
    out = pd.DataFrame(codes, index=labels.index, columns=labels.columns)
    # 缺失行业归 0；其余行业从 1 开始（0 保留给「未知」）
    out = out.where(~pd.isna(flat), fill)
    return out.astype("int32")


def _group_demean(values: np.ndarray, groups: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """按组去均值（values 可以是一列，也可以是 n×k 矩阵）。

    ``counts`` 里可能出现 0（某个组在本批样本里完全缺失），用 ``where`` 挡住，
    否则会算出一堆 ``inf`` 触发告警（残差本身用不到那些格子）。
    """
    if values.ndim == 1:
        sums = np.bincount(groups, weights=values, minlength=len(counts))
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        return values - means[groups]
    out = np.empty_like(values)
    for column in range(values.shape[1]):
        sums = np.bincount(groups, weights=values[:, column], minlength=len(counts))
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        out[:, column] = values[:, column] - means[groups]
    return out


def residualize(
    frame: pd.DataFrame,
    *,
    codes: pd.DataFrame,
    size: pd.DataFrame,
    dates: Sequence[pd.Timestamp] | None = None,
    min_count: int = 30,
) -> pd.DataFrame:
    """单个因子的行业 + 市值中性化（逐日横截面），返回残差矩阵。"""
    out = residualize_many({"_": frame}, codes=codes, size=size, dates=dates, min_count=min_count)
    return out["_"]


def residualize_many(
    frames: Mapping[str, pd.DataFrame],
    *,
    codes: pd.DataFrame,
    size: pd.DataFrame,
    dates: Sequence[pd.Timestamp] | None = None,
    min_count: int = 30,
) -> dict[str, pd.DataFrame]:
    """多个因子一次性中性化（共用同一份行业/市值暴露，省掉重复的组均值计算）。

    ``frames`` 里的每个矩阵、``codes``、``size`` 必须同 index/columns；
    返回同形状的残差矩阵，未处理的日期为 ``NaN``。
    """
    names = list(frames)
    if not names:
        return {}
    reference = frames[names[0]]
    codes = codes.reindex(index=reference.index, columns=reference.columns)
    size = size.reindex(index=reference.index, columns=reference.columns)
    day_list = list(dates) if dates is not None else list(reference.index)
    group_matrix = codes.to_numpy(dtype=np.int32, na_value=0)
    size_matrix = size.to_numpy(dtype=np.float64, na_value=np.nan)
    arrays = {name: frames[name].to_numpy(dtype=np.float64, na_value=np.nan) for name in names}
    result = {name: np.full(reference.shape, np.nan, dtype=np.float32) for name in names}
    positions = {date: pos for pos, date in enumerate(reference.index)}

    for date in day_list:
        pos = positions.get(date)
        if pos is None:
            continue
        groups = group_matrix[pos]
        ln_size = size_matrix[pos]
        base_valid = np.isfinite(ln_size)
        if base_valid.sum() < min_count:
            continue
        groups_valid = groups[base_valid]
        uniq = np.unique(groups_valid)
        if len(uniq) < 2 and not np.isfinite(ln_size[base_valid]).all():
            continue
        # 组内去均值只需要在有数据的子集上做：先把「全部因子都缺失」的行也滤掉
        stacked = np.column_stack([arrays[name][pos] for name in names])
        row_valid = base_valid & np.isfinite(stacked).any(axis=1)
        if row_valid.sum() < min_count:
            continue
        idx = np.flatnonzero(row_valid)
        sub_groups = groups[idx]
        sub_size = ln_size[idx]
        counts = np.bincount(sub_groups, minlength=int(sub_groups.max()) + 1)
        # 行业去均值：核心里是「用同一套组均值」，每个因子列单独算
        size_demeaned = _group_demean(sub_size, sub_groups, counts)
        denom = float(np.dot(size_demeaned, size_demeaned))
        for order, name in enumerate(names):
            column = stacked[idx, order]
            valid = np.isfinite(column)
            if valid.sum() < min_count:
                continue
            y = column[valid]
            g = sub_groups[valid]
            sub_counts = np.bincount(g, minlength=len(counts))
            y_demeaned = _group_demean(y, g, sub_counts)
            if denom > 0:
                # size 的组内残差要按同一批有效样本重算，否则 β 会被缺失值稀释
                s = _group_demean(sub_size[valid], g, sub_counts)
                s_norm = float(np.dot(s, s))
                beta = float(np.dot(s, y_demeaned) / s_norm) if s_norm > 0 else 0.0
                residual = y_demeaned - beta * s
            else:
                residual = y_demeaned
            target = result[name][pos]
            target[idx[valid]] = residual.astype(np.float32)
    return {name: pd.DataFrame(result[name], index=reference.index, columns=reference.columns) for name in names}
