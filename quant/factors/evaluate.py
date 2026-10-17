"""因子评估：IC、IC 衰减、分层收益、因子相关性。

判断一个因子能不能用，看的不是"回测赚钱"，而是这几件事：

* **IC（信息系数）**：因子值与未来收益的横截面秩相关。均值大、稳定性高（IR 高）才有用；
* **IC 衰减**：IC 随持有期拉长怎么衰减——决定调仓频率（日线低频通常看 1/5/10/20 日）；
* **分层收益**：把股票按因子分成 5 层，看收益是否单调。单调性比 IC 更直观，
  也是"这个因子是不是只靠极端组"的照妖镜；
* **因子相关性**：相关性过高的两个因子等于一个，合成前必须先看这张表。

注意 ``forward_returns`` 含前视，只允许出现在评估里，**绝不能**回流到信号链路。
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant.factors.panel import Panel

__all__ = [
    "evaluate_factor",
    "factor_corr",
    "ic_decay",
    "ic_summary",
    "pearson_ic",
    "quantile_returns",
    "quantile_summary",
    "rank_ic",
]

TRADING_DAYS = 244


def _cross_sectional_corr(left: pd.DataFrame, right: pd.DataFrame, *, rank: bool) -> pd.Series:
    a = left.rank(axis=1) if rank else left
    b = right.rank(axis=1) if rank else right
    valid = a.notna() & b.notna()
    a = a.where(valid)
    b = b.where(valid)
    am = a.sub(a.mean(axis=1), axis=0)
    bm = b.sub(b.mean(axis=1), axis=0)
    num = (am * bm).sum(axis=1, skipna=True)
    den = np.sqrt((am**2).sum(axis=1, skipna=True) * (bm**2).sum(axis=1, skipna=True))
    return num / den.replace(0, np.nan)


def rank_ic(factor: pd.DataFrame, forward_returns: pd.DataFrame, *, min_count: int = 5) -> pd.Series:
    """秩相关 IC（Spearman）。"""
    valid = factor.notna() & forward_returns.notna()
    counts = valid.sum(axis=1)
    ic = _cross_sectional_corr(factor, forward_returns, rank=True)
    ic = ic.where(counts >= min_count)
    ic.name = "rank_ic"
    return ic


def pearson_ic(factor: pd.DataFrame, forward_returns: pd.DataFrame, *, min_count: int = 5) -> pd.Series:
    """普通相关 IC（Pearson，对极值敏感）。"""
    valid = factor.notna() & forward_returns.notna()
    counts = valid.sum(axis=1)
    ic = _cross_sectional_corr(factor, forward_returns, rank=False)
    ic = ic.where(counts >= min_count)
    ic.name = "pearson_ic"
    return ic


def ic_summary(ic: pd.Series, *, periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
    """IC 序列的汇总统计（业内常用口径）。"""
    series = pd.Series(ic).dropna()
    n = int(len(series))
    if n == 0:
        return {
            "n": 0,
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
            "t_stat": float("nan"),
            "positive_rate": float("nan"),
            "abs_mean": float("nan"),
            "annual_ir": float("nan"),
        }
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if n > 1 else float("nan")
    ir = mean / std if std and not np.isnan(std) else float("nan")
    return {
        "n": n,
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": ir,
        "t_stat": ir * np.sqrt(n) if not np.isnan(ir) else float("nan"),
        "positive_rate": float((series > 0).mean()),
        "abs_mean": float(series.abs().mean()),
        "annual_ir": ir * np.sqrt(periods_per_year) if not np.isnan(ir) else float("nan"),
    }


def ic_decay(
    factor: pd.DataFrame,
    panel: Panel,
    *,
    horizons: Sequence[int] = (1, 2, 3, 5, 10, 20),
    min_count: int = 5,
) -> pd.Series:
    """IC 衰减曲线：不同未来持有期下的 IC 均值。"""
    rows: dict[int, float] = {}
    for horizon in horizons:
        ic = rank_ic(factor, panel.forward_returns(int(horizon)), min_count=min_count)
        rows[int(horizon)] = float(pd.Series(ic).dropna().mean()) if len(ic.dropna()) else float("nan")
    out = pd.Series(rows, name="ic_mean")
    out.index.name = "horizon"
    return out


def quantile_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    quantiles: int = 5,
    min_count: int | None = None,
) -> pd.DataFrame:
    """分层收益：按因子值分组（Q1 最低 → Q{quantiles} 最高），返回每日各组等权收益。"""
    if quantiles < 2:
        raise ValueError("quantiles 至少为 2")
    floor = min_count if min_count is not None else max(quantiles * 2, 10)
    ranked = factor.rank(axis=1, pct=True)
    bucket = np.ceil(ranked * quantiles)
    bucket = bucket.clip(1, quantiles)
    valid = factor.notna() & forward_returns.notna()
    counts = valid.sum(axis=1)
    columns: dict[int, pd.Series] = {}
    for b in range(1, quantiles + 1):
        mask = (bucket == b) & valid
        columns[b] = forward_returns.where(mask).mean(axis=1)
    out = pd.DataFrame(columns)
    out = out.where(counts >= floor)
    out.columns = [f"Q{i}" for i in range(1, quantiles + 1)]
    out.index.name = "date"
    return out


def quantile_summary(qret: pd.DataFrame, *, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """分层收益汇总：均值、年化、胜率、样本天数。"""
    rows = []
    for column in qret.columns:
        series = qret[column].dropna()
        mean = float(series.mean()) if len(series) else float("nan")
        rows.append(
            {
                "bucket": column,
                "mean": mean,
                "annual": (1.0 + mean) ** periods_per_year - 1.0 if not np.isnan(mean) else float("nan"),
                "positive_rate": float((series > 0).mean()) if len(series) else float("nan"),
                "days": int(len(series)),
            }
        )
    summary = pd.DataFrame(rows).set_index("bucket")
    top, bottom = qret.columns[-1], qret.columns[0]
    summary.loc["多空(Q最高-Q最低)", "mean"] = float((qret[top] - qret[bottom]).mean())
    summary.loc["多空(Q最高-Q最低)", "positive_rate"] = float(((qret[top] - qret[bottom]) > 0).mean())
    summary.loc["多空(Q最高-Q最低)", "days"] = int((qret[top] - qret[bottom]).notna().sum())
    return summary


def factor_corr(factors: Mapping[str, pd.DataFrame], *, rank: bool = True) -> pd.DataFrame:
    """因子间的平均横截面相关（默认秩相关）：合成前用它剔除冗余因子。"""
    names = list(factors)
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            daily = _cross_sectional_corr(factors[left_name], factors[right_name], rank=rank).dropna()
            value = float(daily.mean()) if len(daily) else float("nan")
            matrix.loc[left_name, right_name] = value
            matrix.loc[right_name, left_name] = value
    return matrix


def evaluate_factor(
    factor: pd.DataFrame,
    panel: Panel,
    *,
    name: str = "factor",
    horizons: Iterable[int] = (1, 2, 5, 10, 20),
    quantiles: int = 5,
    min_count: int = 5,
) -> dict:
    """一次性给出 IC、衰减、分层、覆盖度——因子入库前的体检。"""
    horizon_list = [int(h) for h in horizons]
    ic = rank_ic(factor, panel.forward_returns(1), min_count=min_count)
    qret = quantile_returns(factor, panel.forward_returns(1), quantiles=quantiles, min_count=min_count)
    decay = ic_decay(factor, panel, horizons=horizon_list, min_count=min_count)
    coverage = float(factor.notna().sum(axis=1).mean())
    return {
        "name": name,
        "ic": ic_summary(ic),
        "ic_series": ic,
        "decay": decay,
        "quantiles": qret,
        "quantile_summary": quantile_summary(qret),
        "coverage": coverage,
    }


def evaluate_text(result: Mapping) -> str:
    """把 ``evaluate_factor`` 的结果打印成人看的文本。"""
    ic = result.get("ic", {})
    lines = [
        f"因子 {result.get('name', '?')}：覆盖度 {result.get('coverage', float('nan')):.1f} 只/日，"
        f"IC 均值 {ic.get('ic_mean', float('nan')):.4f}，IC IR {ic.get('ic_ir', float('nan')):.3f}，"
        f"t 值 {ic.get('t_stat', float('nan')):.2f}，IC>0 占比 {ic.get('positive_rate', float('nan')):.1%}"
    ]
    decay = result.get("decay")
    if decay is not None and len(decay):
        parts = "，".join(f"{int(h)}日 {v:.4f}" for h, v in decay.items())
        lines.append(f"  IC 衰减：{parts}")
    summary = result.get("quantile_summary")
    if summary is not None and len(summary):
        parts = "，".join(f"{idx} {row['mean']:.4%}" for idx, row in summary.iterrows())
        lines.append(f"  分层收益：{parts}")
    return "\n".join(lines)
