"""阶段 B：业绩预告/快报、限售解禁、股权质押因子（N 系列）——独立模块，**不注册**进共享 ``FACTORS``。

数据来源（阶段 A 采集，见 ``scripts/growth_alt_collect.py`` 与 ``docs/小盘成长策略-进度日志.md``）：
``D:/quant-data/alt/{forecast,express,unlock,pledge}.parquet``，全部 ≤ 2024-12-31。

PIT 口径（与采集脚本一致，写死）
--------------------------------
- 预告/快报：``notice_date``（公告日期）为可见日，公告日当天收盘即生效。
- 解禁：``unlock_date`` 为事件日；因子只用 **t 之后** 的解禁计划（时间表提前公开），数量用计划
  ``unlock_shares``，价格用 t 日收盘价。合并层里的 ``ret_20d_after`` 是未来收益，**禁止使用**。
- 质押：``as_of``（周五快照）为可见日，两次快照之间用前值 ffill。

因子清单（13 个，事前登记；口径见进度日志 §阶段 B）
----------------------------------------------------
N1 ``fc_sue``           预告归母中值对季节随机游走标准化
N2 ``fc_yoy``           预告归母同比（正基数才有效）
N3 ``fc_type``          预告类型序数
N5 ``fc_express_sp``    快报净利 vs 预告中值
N6 ``fc_express_yoy``   快报净利同比
N7 ``fc_revision``      快报净利 vs 去年同期（基线超预期）
N8 ``unlock_60d``       未来 60 日解禁规模 / 流通市值（取负）
N9 ``unlock_120d``      未来 120 日解禁规模 / 流通市值（取负）
N10 ``unlock_past20d``  过去 20 日已解禁规模 / 流通市值（取负）
N11 ``pledge_ratio``    质押比例（取负）
N12 ``pledge_chg13w``   质押比例 13 周变化（取负）
N13 ``pledge_count``    质押笔数（取负，排雷诊断）
N14 ``fc_roa``          预告归母中值 / 总资产（按 Q 年化，取正）

``N4 fc_width``（预告区间宽度）**不构造**：akshare 的 ``预测数值`` 已是上下限中值，宽度须直连东财原始接口，
本阶段不做（记录在进度日志的"事前调整"里，无任何 IC 依据）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from quant.factors import style
from quant.factors.growth import GrowthFactors
from quant.factors.panel import Panel

__all__ = [
    "ALT_DIR",
    "FORBIDDEN_COLUMNS",
    "FACTOR_META",
    "AltGrowthFactors",
    "alt_factors",
    "factor_catalog",
    "load_alt",
]

ALT_DIR = Path("D:/quant-data/alt")
CUTOFF = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")
FORECAST_PERIOD_CAP = pd.Timestamp("2024-09-30")
"""预告/快报只用报告期 ≤ 该日的记录：年报与一季报预告多在次年 1–4 月披露，落在 2025+ 会被 PIT 丢弃。"""

FORBIDDEN_COLUMNS = ("ret_20d_after", "ret_20d_before", "ret_1y")
"""解禁表里的「解禁后20日涨跌幅」= 未来收益，任何情况下不得进入因子。"""

DATE_COLUMNS = ("notice_date", "report_period", "unlock_date", "as_of")

# 预告类型序数（11 类，东财口径）
TYPE_SCORE = {
    "预增": 1.0,
    "扭亏": 1.0,
    "略增": 0.5,
    "续盈": 0.25,
    "减亏": 0.25,
    "不确定": 0.0,
    "略减": -0.5,
    "预减": -1.0,
    "首亏": -1.0,
    "续亏": -1.0,
    "增亏": -1.0,
}

PARENT_INDICATOR = "归属于上市公司股东的净利润"
DEDUCT_INDICATOR = "扣除非经常性损益后的净利润"

FACTOR_META: dict[str, dict[str, object]] = {
    "n1_fc_sue": {"source": "forecast", "raw_direction": 1, "desc": "预告归母 SUE（季节随机游走标准化）"},
    "n2_fc_yoy": {"source": "forecast", "raw_direction": 1, "desc": "预告归母同比（正基数）"},
    "n3_fc_type": {"source": "forecast", "raw_direction": 1, "desc": "预告类型序数"},
    "n5_fc_express_sp": {"source": "forecast+express", "raw_direction": 1, "desc": "快报净利 vs 预告中值"},
    "n6_fc_express_yoy": {"source": "express", "raw_direction": 1, "desc": "快报净利同比"},
    "n7_fc_revision": {"source": "express", "raw_direction": 1, "desc": "快报净利 vs 去年同期"},
    "n8_unlock_60d": {"source": "unlock", "raw_direction": -1, "desc": "未来60日解禁/流通市值"},
    "n9_unlock_120d": {"source": "unlock", "raw_direction": -1, "desc": "未来120日解禁/流通市值"},
    "n10_unlock_past20d": {"source": "unlock", "raw_direction": -1, "desc": "过去20日解禁/流通市值"},
    "n11_pledge_ratio": {"source": "pledge", "raw_direction": -1, "desc": "质押比例"},
    "n12_pledge_chg13w": {"source": "pledge", "raw_direction": -1, "desc": "质押比例13周变化"},
    "n13_pledge_count": {"source": "pledge", "raw_direction": -1, "desc": "质押笔数"},
    "n14_fc_roa": {"source": "forecast+balance", "raw_direction": 1, "desc": "预告归母/总资产（年化）"},
}


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def load_alt(
    name: str,
    *,
    alt_dir: Path = ALT_DIR,
    cutoff: pd.Timestamp = CUTOFF,
    columns: Iterable[str] | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """读合并层 parquet，并断言没有任何日期越过 ``cutoff``（holdout 隔离）。

    ``strict=False`` 时不报错、而是**丢弃**越过 cutoff 的行——历史样本（cutoff=2016-12-31）
    需要读同一个合并层，越界行属于主样本而非 holdout，丢弃即可。
    """
    path = Path(alt_dir) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"缺少合并层文件：{path}（先运行 scripts.growth_alt_collect）")
    use = None if columns is None else [c for c in columns if c not in FORBIDDEN_COLUMNS]
    frame = pd.read_parquet(path, columns=use)
    for column in DATE_COLUMNS:
        if column in frame.columns:
            stamps = pd.to_datetime(frame[column], errors="coerce")
            if (stamps > cutoff).any():
                if strict:
                    raise ValueError(f"{name}.{column} 存在 > {cutoff.date()} 的记录，违反 holdout 纪律")
                frame = frame.loc[stamps <= cutoff].reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------- #
# 面板散列
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Context:
    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    ipo_days: tuple[pd.Timestamp | None, ...]


def _ipo_days(symbols: Iterable[str], override: Mapping[str, object] | None = None) -> tuple[pd.Timestamp | None, ...]:
    if override is not None:
        table = {str(k).split(".")[0]: pd.Timestamp(v) for k, v in override.items()}
        return tuple(table.get(str(s).split(".")[0]) for s in symbols)
    from quant import refdata

    universe = refdata.load_universe()
    table = dict(
        zip(
            universe["symbol"].astype(str).str.zfill(6),
            pd.to_datetime(universe["ipo_date"], errors="coerce"),
        )
    )
    out: list[pd.Timestamp | None] = []
    for symbol in symbols:
        stamp = table.get(str(symbol).split(".")[0])
        out.append(None if stamp is None or pd.isna(stamp) else pd.Timestamp(stamp))
    return tuple(out)


def _scatter(values: pd.DataFrame, notice: pd.DataFrame, ctx: _Context) -> pd.DataFrame:
    """「期间 × 代码」的值按可见日散列到交易日，并 ffill；上市前的格子置 NaN。"""
    if values.empty:
        return pd.DataFrame(np.nan, index=ctx.dates, columns=list(ctx.symbols), dtype="float32")
    codes = {str(code): i for i, code in enumerate(values.columns)}
    notice_frame = notice.reindex(index=values.index, columns=values.columns)
    notice_np = notice_frame.to_numpy(dtype="datetime64[ns]")
    values_np = values.to_numpy(dtype=np.float32)
    day_np = ctx.dates.to_numpy(dtype="datetime64[ns]")
    out = np.full((len(ctx.dates), len(ctx.symbols)), np.nan, dtype=np.float32)
    for position, symbol in enumerate(ctx.symbols):
        column = codes.get(str(symbol).split(".")[0])
        if column is None:
            continue
        avail = notice_np[:, column]
        series = values_np[:, column]
        ok = ~pd.isna(avail) & ~np.isnan(series)
        if not ok.any():
            continue
        order = np.argsort(avail[ok], kind="stable")
        when = avail[ok][order]
        vals = series[ok][order]
        index = np.searchsorted(when, day_np, side="right") - 1
        good = index >= 0
        ipo = ctx.ipo_days[position]
        if ipo is not None:
            good &= day_np >= np.datetime64(ipo, "ns")
        out[good, position] = vals[index[good]]
    return pd.DataFrame(out, index=ctx.dates, columns=list(ctx.symbols))


def _window_sum(events: pd.DataFrame, ctx: _Context, *, days: int, forward: bool) -> pd.DataFrame:
    """按标的对「解禁事件」做滚动窗口求和：forward=True 取 (t, t+days]，False 取 [t-days, t)。"""
    out = np.zeros((len(ctx.dates), len(ctx.symbols)), dtype=np.float32)
    if events.empty:
        return pd.DataFrame(out, index=ctx.dates, columns=list(ctx.symbols))
    day_np = ctx.dates.to_numpy(dtype="datetime64[ns]")
    span = np.timedelta64(days, "D")
    grouped = {str(code).split(".")[0]: g for code, g in events.groupby("code")}
    for position, symbol in enumerate(ctx.symbols):
        frame = grouped.get(str(symbol).split(".")[0])
        if frame is None or frame.empty:
            continue
        stamps = pd.to_datetime(frame["unlock_date"]).to_numpy(dtype="datetime64[ns]")
        order = np.argsort(stamps, kind="stable")
        stamps = stamps[order]
        shares = frame["unlock_shares"].to_numpy(dtype=np.float64)[order]
        shares = np.nan_to_num(shares, nan=0.0)
        cumulative = np.concatenate([[0.0], np.cumsum(shares)])
        if forward:
            hi = np.searchsorted(stamps, day_np + span, side="right")
            lo = np.searchsorted(stamps, day_np, side="right")
        else:
            hi = np.searchsorted(stamps, day_np, side="left")
            lo = np.searchsorted(stamps, day_np - span, side="left")
        total = cumulative[hi] - cumulative[lo]
        ipo = ctx.ipo_days[position]
        good = np.ones(len(day_np), dtype=bool)
        if ipo is not None:
            good &= day_np >= np.datetime64(ipo, "ns")
        out[:, position] = np.where(good, total, np.nan)
    return pd.DataFrame(out, index=ctx.dates, columns=list(ctx.symbols))


def _pivot(long: pd.DataFrame, value: str, *, agg: str = "last") -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    return long.pivot_table(index="report_period", columns="code", values=value, aggfunc=agg)


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class AltGrowthFactors:
    """N1–N14 因子；全部返回 ``index=交易日, columns=标的`` 的 float32 宽表，值越大越看多。"""

    def __init__(
        self,
        panel: Panel,
        *,
        alt_dir: Path = ALT_DIR,
        cutoff: pd.Timestamp = CUTOFF,
        growth: GrowthFactors | None = None,
        ipo_dates: Mapping[str, object] | None = None,
        forecast_cap: pd.Timestamp | None = FORECAST_PERIOD_CAP,
        strict: bool = True,
    ) -> None:
        self.panel = panel
        self.alt_dir = Path(alt_dir)
        self.cutoff = pd.Timestamp(cutoff)
        self.strict = bool(strict)
        self.forecast_cap = forecast_cap
        self._growth = growth
        self._ipo_override = dict(ipo_dates) if ipo_dates is not None else None
        self._ctx: _Context | None = None
        self._tables: dict[str, pd.DataFrame] = {}
        self._cache: dict[str, pd.DataFrame] = {}

    # -- 基础 -------------------------------------------------------------- #
    @property
    def ctx(self) -> _Context:
        if self._ctx is None:
            self._ctx = _Context(
                self.panel.dates,
                tuple(self.panel.symbols),
                _ipo_days(self.panel.symbols, self._ipo_override),
            )
        return self._ctx

    @property
    def growth(self) -> GrowthFactors:
        if self._growth is None:
            self._growth = GrowthFactors(self.panel, ipo_dates=self._ipo_override)
        return self._growth

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            self._tables[name] = load_alt(name, alt_dir=self.alt_dir, cutoff=self.cutoff,
                                           strict=self.strict)
        return self._tables[name]

    def _cached(self, name: str, builder) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = builder().reindex(index=self.panel.dates, columns=self.panel.symbols)
        return self._cache[name]

    # -- 长表 -------------------------------------------------------------- #
    def forecast_long(self) -> pd.DataFrame:
        """每股每期一行（归母净利润），按公告日取最新一条。"""
        frame = self.table("forecast")
        keep = frame[frame["indicator"].astype(str).str.strip() == PARENT_INDICATOR].copy()
        keep["report_period"] = pd.to_datetime(keep["report_period"], errors="coerce")
        keep["notice_date"] = pd.to_datetime(keep["notice_date"], errors="coerce")
        if self.forecast_cap is not None:
            keep = keep[keep["report_period"] <= self.forecast_cap]
        keep["type_score"] = keep["forecast_type"].map(TYPE_SCORE).astype("float32")
        keep = keep.sort_values("notice_date").drop_duplicates(["report_period", "code"], keep="last")
        return keep.reset_index(drop=True)

    def express_long(self) -> pd.DataFrame:
        frame = self.table("express").copy()
        frame["report_period"] = pd.to_datetime(frame["report_period"], errors="coerce")
        frame["notice_date"] = pd.to_datetime(frame["notice_date"], errors="coerce")
        if self.forecast_cap is not None:
            frame = frame[frame["report_period"] <= self.forecast_cap]
        frame["net_profit_yoy_calc"] = np.where(
            frame["net_profit_last"] > 0,
            frame["net_profit"] / frame["net_profit_last"] - 1.0,
            np.nan,
        )
        frame = frame.sort_values("notice_date").drop_duplicates(["report_period", "code"], keep="last")
        return frame.reset_index(drop=True)

    def unlock_events(self) -> pd.DataFrame:
        frame = self.table("unlock")
        if frame.empty:
            return frame
        events = frame.copy()
        events["code"] = events["code"].astype(str).str.zfill(6)
        events["unlock_shares"] = events["unlock_shares"].fillna(0.0)
        grouped = (
            events.groupby(["code", "unlock_date"], as_index=False)
            .agg(unlock_shares=("unlock_shares", "sum"), rows=("unlock_shares", "size"))
            .sort_values("unlock_date")
        )
        return grouped.reset_index(drop=True)

    def pledge_long(self) -> pd.DataFrame:
        frame = self.table("pledge")
        if frame.empty:
            return frame
        out = frame.sort_values("as_of").drop_duplicates(["as_of", "code"], keep="last")
        return out.reset_index(drop=True)

    # -- 预告 / 快报因子 ---------------------------------------------------- #
    def _forecast_wide(self, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        long = self.forecast_long()
        if long.empty:
            return pd.DataFrame(), pd.DataFrame()
        values = _pivot(long, column)
        notice = _pivot(long, "notice_date", agg="max").reindex(index=values.index, columns=values.columns)
        return values, notice

    def _forecast_factor(self, column: str) -> pd.DataFrame:
        values, notice = self._forecast_wide(column)
        if values.empty:
            return pd.DataFrame(np.nan, index=self.panel.dates, columns=self.panel.symbols, dtype="float32")
        return _scatter(values.astype("float64"), notice, self.ctx)

    def fc_sue(self) -> pd.DataFrame:
        """(F − A_{t−4}) / σ，σ 只用 t−4 及更早的季节差分（全部已披露，无前视）。"""
        long = self.forecast_long()
        if long.empty:
            return pd.DataFrame(np.nan, index=self.panel.dates, columns=self.panel.symbols, dtype="float32")

        def build() -> pd.DataFrame:
            growth = self.growth
            quarter = growth.quarter("profit", "PARENT_NETPROFIT")
            base = quarter.shift(4).stack().rename("base")
            delta = (quarter - quarter.shift(4)).shift(4)
            sigma = delta.rolling(8, min_periods=4).std().stack().rename("sigma")
            grid = pd.concat([base, sigma], axis=1).reset_index()
            grid.columns = ["report_period", "code", "base", "sigma"]
            merged = long.merge(grid, on=["report_period", "code"], how="left")
            sue = (merged["value"] - merged["base"]) / merged["sigma"].replace(0.0, np.nan)
            merged["n1_fc_sue"] = sue.replace([np.inf, -np.inf], np.nan)
            values = _pivot(merged, "n1_fc_sue")
            notice = _pivot(merged, "notice_date", agg="max").reindex(
                index=values.index, columns=values.columns
            )
            return _scatter(values.astype("float64"), notice, self.ctx)

        return self._cached("n1_fc_sue_v", build)

    def fc_yoy(self) -> pd.DataFrame:
        long = self.forecast_long()
        if long.empty:
            return pd.DataFrame(np.nan, index=self.panel.dates, columns=self.panel.symbols, dtype="float32")
        frame = long.copy()
        frame["n2_fc_yoy"] = np.where(
            frame["last_year"] > 0,
            frame["value"] / frame["last_year"] - 1.0,
            np.nan,
        )
        values = _pivot(frame, "n2_fc_yoy")
        notice = _pivot(frame, "notice_date", agg="max").reindex(index=values.index, columns=values.columns)
        return self._cached(
            "n2_fc_yoy_v", lambda: _scatter(values.astype("float64"), notice, self.ctx)
        )

    def fc_type(self) -> pd.DataFrame:
        long = self.forecast_long()
        if long.empty:
            return pd.DataFrame(np.nan, index=self.panel.dates, columns=self.panel.symbols, dtype="float32")
        values = _pivot(long, "type_score")
        notice = _pivot(long, "notice_date", agg="max").reindex(index=values.index, columns=values.columns)
        return self._cached("n3_fc_type_v", lambda: _scatter(values, notice, self.ctx))

    def fc_roa(self) -> pd.DataFrame:
        """预告归母中值 / 总资产，按报告期进度年化（Q1×4、H1×2、9M×4/3、年报×1）。"""
        long = self.forecast_long()
        if long.empty:
            return pd.DataFrame(np.nan, index=self.panel.dates, columns=self.panel.symbols, dtype="float32")

        def build() -> pd.DataFrame:
            assets = self.growth.total_assets()
            stacked = assets.stack().rename("total_assets").reset_index()
            stacked.columns = ["report_period", "code", "total_assets"]
            merged = long.merge(stacked, on=["report_period", "code"], how="left")
            month = pd.to_datetime(merged["report_period"]).dt.month
            annualize = month.map({3: 4.0, 6: 2.0, 9: 4.0 / 3.0, 12: 1.0}).astype("float64")
            merged["n14_fc_roa"] = merged["value"] * annualize / merged["total_assets"].replace(0.0, np.nan)
            values = _pivot(merged, "n14_fc_roa")
            notice = _pivot(merged, "notice_date", agg="max").reindex(
                index=values.index, columns=values.columns
            )
            return _scatter(values.astype("float64"), notice, self.ctx)

        return self._cached("n14_fc_roa_v", build)

    def fc_express_yoy(self) -> pd.DataFrame:
        """快报净利同比（自算，正基期）：东财 `净利润-同比增长` 与自算值口径不一致，只自算。"""
        long = self.express_long()
        if long.empty:
            return pd.DataFrame(np.nan, index=self.panel.dates, columns=self.panel.symbols, dtype="float32")
        frame = long.assign(n6_fc_express_yoy=long["net_profit_yoy_calc"])
        values = _pivot(frame, "n6_fc_express_yoy")
        notice = _pivot(frame, "notice_date", agg="max").reindex(index=values.index, columns=values.columns)
        return self._cached("n6_fc_express_yoy_v", lambda: _scatter(values.astype("float64"), notice, self.ctx))

    def fc_revision(self) -> pd.DataFrame:
        """快报 vs 预告的方向性修正：sign(快报净利 − 预告中值)，对负基数稳健。"""
        express = self.express_long()
        forecast = self.forecast_long()
        if express.empty or forecast.empty:
            return pd.DataFrame(np.nan, index=self.panel.dates, columns=self.panel.symbols, dtype="float32")

        def build() -> pd.DataFrame:
            merged = express.merge(
                forecast[["report_period", "code", "value"]].rename(columns={"value": "fc_value"}),
                on=["report_period", "code"],
                how="inner",
            )
            diff = merged["net_profit"] - merged["fc_value"]
            merged["n7"] = np.sign(diff).where(diff.notna() & merged["fc_value"].notna())
            merged["notice"] = pd.to_datetime(merged["notice_date"])
            values = _pivot(merged, "n7")
            notice = _pivot(merged, "notice", agg="max").reindex(index=values.index, columns=values.columns)
            return _scatter(values.astype("float64"), notice, self.ctx)

        return self._cached("n7_v", build)

    def fc_express_sp(self) -> pd.DataFrame:
        """快报净利 / 预告中值 − 1（两者都在 => 快报公告日可见）。"""
        express = self.express_long()
        forecast = self.forecast_long()
        if express.empty or forecast.empty:
            return pd.DataFrame(np.nan, index=self.panel.dates, columns=self.panel.symbols, dtype="float32")

        def build() -> pd.DataFrame:
            merged = express.merge(
                forecast[["report_period", "code", "value"]].rename(columns={"value": "fc_value"}),
                on=["report_period", "code"],
                how="inner",
            )
            merged["n5"] = np.where(
                merged["fc_value"].abs() > 0,
                merged["net_profit"] / merged["fc_value"] - 1.0,
                np.nan,
            )
            merged["notice"] = pd.to_datetime(merged["notice_date"])
            values = _pivot(merged, "n5")
            notice = _pivot(merged, "notice", agg="max").reindex(index=values.index, columns=values.columns)
            return _scatter(values.astype("float64"), notice, self.ctx)

        return self._cached("n5_v", build)

    # -- 解禁因子 ---------------------------------------------------------- #
    def _float_mcap(self) -> pd.DataFrame:
        return style.float_mcap_panel(self.panel)

    def unlock_density(self, *, days: int, forward: bool) -> pd.DataFrame:
        events = self.unlock_events()
        shares = _window_sum(events, self.ctx, days=days, forward=forward)
        close = self.panel.close.reindex(index=self.panel.dates, columns=self.panel.symbols)
        mcap = self._float_mcap().reindex(index=self.panel.dates, columns=self.panel.symbols)
        value = shares.fillna(0.0) * close / mcap.replace(0.0, np.nan)
        value = value.where(shares.notna())
        # 数据起点之前「无事件」与「没采到」不可区分，一律置 NaN 防止把缺失当零风险
        if not events.empty:
            first = pd.to_datetime(events["unlock_date"]).min()
            value.loc[value.index < first, :] = np.nan
        return value

    def unlock_60d(self) -> pd.DataFrame:
        return self._cached("n8", lambda: -self.unlock_density(days=60, forward=True))

    def unlock_120d(self) -> pd.DataFrame:
        return self._cached("n9", lambda: -self.unlock_density(days=120, forward=True))

    def unlock_past20d(self) -> pd.DataFrame:
        return self._cached("n10", lambda: -self.unlock_density(days=20, forward=False))

    # -- 质押因子 ---------------------------------------------------------- #
    def _pledge_scatter(self, column: str, *, fill_missing_zero: bool) -> pd.DataFrame:
        long = self.pledge_long()
        if long.empty:
            return pd.DataFrame(np.nan, index=self.panel.dates, columns=self.panel.symbols, dtype="float32")
        values = long.pivot_table(index="as_of", columns="code", values=column, aggfunc="last")
        notice = pd.DataFrame(
            np.repeat(values.index.to_numpy()[:, None], values.shape[1], axis=1),
            index=values.index,
            columns=values.columns,
        )
        scattered = _scatter(values.astype("float64"), notice, self.ctx)
        if not fill_missing_zero:
            return scattered
        # 缺失 ≈ 无质押：首个快照之后置 0，之前保持 NaN
        seen = scattered.notna().cummax()
        return scattered.ffill().fillna(0.0).where(seen, np.nan)

    def pledge_ratio(self) -> pd.DataFrame:
        return self._cached("n11", lambda: -self._pledge_scatter("pledge_ratio", fill_missing_zero=True))

    def pledge_chg13w(self) -> pd.DataFrame:
        def build() -> pd.DataFrame:
            long = self.pledge_long()
            if long.empty:
                return pd.DataFrame(np.nan, index=self.panel.dates, columns=self.panel.symbols, dtype="float32")
            values = long.pivot_table(index="as_of", columns="code", values="pledge_ratio", aggfunc="last")
            change = values - values.shift(13)
            notice = pd.DataFrame(
                np.repeat(change.index.to_numpy()[:, None], change.shape[1], axis=1),
                index=change.index,
                columns=change.columns,
            )
            return -_scatter(change.astype("float64"), notice, self.ctx)

        return self._cached("n12", build)

    def pledge_count(self) -> pd.DataFrame:
        return self._cached("n13", lambda: -self._pledge_scatter("pledge_count", fill_missing_zero=True))

    # -- 汇总 -------------------------------------------------------------- #
    def all_factors(self) -> dict[str, pd.DataFrame]:
        return {
            "n1_fc_sue": self.fc_sue(),
            "n2_fc_yoy": self.fc_yoy(),
            "n3_fc_type": self.fc_type(),
            "n5_fc_express_sp": self.fc_express_sp(),
            "n6_fc_express_yoy": self.fc_express_yoy(),
            "n7_fc_revision": self.fc_revision(),
            "n8_unlock_60d": self.unlock_60d(),
            "n9_unlock_120d": self.unlock_120d(),
            "n10_unlock_past20d": self.unlock_past20d(),
            "n11_pledge_ratio": self.pledge_ratio(),
            "n12_pledge_chg13w": self.pledge_chg13w(),
            "n13_pledge_count": self.pledge_count(),
            "n14_fc_roa": self.fc_roa(),
        }


_CACHE: dict[str, dict[str, pd.DataFrame]] = {}


def _cache_key(panel: Panel, alt_dir: Path, cap: pd.Timestamp | None) -> str:
    dates = panel.dates
    return f"{Path(alt_dir)}|{cap}|{len(panel.symbols)}|{dates[0]}..{dates[-1]}|{hash(tuple(panel.symbols))}"


def alt_factors(panel: Panel, **kwargs) -> dict[str, pd.DataFrame]:
    """带进程内缓存的入口（同一面板重复调用只算一次）。"""
    alt_dir = Path(kwargs.get("alt_dir", ALT_DIR))
    cap = kwargs.get("forecast_cap", FORECAST_PERIOD_CAP)
    key = _cache_key(panel, alt_dir, cap)
    if key not in _CACHE:
        _CACHE[key] = AltGrowthFactors(panel, **kwargs).all_factors()
    return _CACHE[key]


def factor_catalog() -> pd.DataFrame:
    rows = [{"factor": name, **meta} for name, meta in FACTOR_META.items()]
    return pd.DataFrame(rows)
