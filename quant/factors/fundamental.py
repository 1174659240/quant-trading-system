"""基本面因子：估值（EP/BP/SP/CFP）+ 质量（ROE/毛利/净利）+ 现金流/成长/杠杆。

数据来自本工程已经落盘的两个真实数据源：

* ``D:/quant-data/valuation/<代码>.parquet``：baostock 日频 peTTM / pbMRQ / psTTM / pcfNcfTTM / isST；
* ``D:/quant-data/fundamentals/{profit,balance,cashflow}/<报告期>.parquet``：东财三表全字段。

**防前视的三条硬规矩**（都不许绕过）：

1. 财务数据的可用日 = ``NOTICE_DATE``（公告日），不是 ``REPORT_DATE``；缺失时用
   ``REPORT_DATE + 110 天`` 兜底。因子值在被公告之前一律是 NaN。
2. 季报/中报是**年初至今累计**，先按年差分出单季，再滚动 4 季求和得 TTM；
   季报不全（缺季）时 TTM 记 NaN，不插值。
3. 因子值在公告日**当天收盘后**才可用，本实现按 ``searchsorted(avail, side="left")``
   把值放到"公告日当天或之后第一个交易日"，与"当天收盘后可知"一致（不早于公告日）。

退市/停牌的处理交给 ``Panel.tradable`` 与 universe 区间裁剪，本模块只产出原始因子值。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from quant.factors.library import register_factor
from quant.factors.panel import Panel

__all__ = [
    "availability_grid",
    "fundamental_wide",
    "register",
]

TTM_FALLBACK_DAYS = 110
# 三表里参与 TTM 的流量项（累计值 → 单季 → TTM）
FLOW_COLUMNS = {
    "profit": ["TOTAL_OPERATE_INCOME", "OPERATE_COST", "PARENT_NETPROFIT", "NETPROFIT", "TOTAL_OPERATE_COST"],
    "cashflow": ["NETCASH_OPERATE", "CONSTRUCT_LONG_ASSET"],
}
# 资产负债表的存量项（时点值，不需要差分）
STOCK_COLUMNS = [
    "TOTAL_ASSETS",
    "TOTAL_LIABILITIES",
    "TOTAL_PARENT_EQUITY",
    "TOTAL_EQUITY",
    "GOODWILL",
    "INVENTORY",
    "ACCOUNTS_RECE",
    "SHORT_LOAN",
    "MONETARYFUNDS",
]


def _quarter_frame(table: str, columns: list[str]) -> pd.DataFrame:
    """把某张表的所有报告期拼成长表，只留需要的列。"""
    from quant import fundamentals as fd

    frames = []
    for path in sorted((fd.FUND_DIR / table).glob("????-??-??.parquet")):
        frame = pd.read_parquet(path)
        keep = ["SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE", *[c for c in columns if c in frame.columns]]
        frames.append(frame[keep])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["SECURITY_CODE"] = out["SECURITY_CODE"].astype(str).str.zfill(6)
    out["REPORT_DATE"] = pd.to_datetime(out["REPORT_DATE"], errors="coerce")
    out["NOTICE_DATE"] = pd.to_datetime(out["NOTICE_DATE"], errors="coerce")
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


# 主要财务指标表里需要的字段（金融业的三表不全，靠这张兜底）
INDICATOR_COLUMNS = [
    "BPS",
    "TOTAL_SHARE",
    "A_FREE_SHARE",
    "ROEJQ",
    "XSMLL",
    "XSJLL",
    "ZCFZL",
    "TOTALOPERATEREVE",
    "PARENTNETPROFIT",
    "TOTALOPERATEREVETZ",
    "PARENTNETPROFITTZ",
]


@lru_cache(maxsize=1)
def _indicator() -> dict[str, pd.DataFrame]:
    """主要财务指标的「报告期 × 代码」宽表字典；三表缺失的格子用它补。"""
    long = _quarter_frame("indicator", INDICATOR_COLUMNS)
    return {column: _wide_from_long(long, column) for column in INDICATOR_COLUMNS if column in long.columns}


def _equity() -> pd.DataFrame:
    """归母净资产：优先用资产负债表，缺失（金融业）时用 BPS × 总股本。

    注意必须对**并集**做 reindex：银行的资产负债表整张缺失，若按资产负债表
    的列取并集之前就 reindex，银行会被直接过滤掉（曾经踩到的坑）。
    """
    _, stocks = _ttm_table()
    base = stocks["TOTAL_PARENT_EQUITY"]
    ind = _indicator()
    if "BPS" not in ind or "TOTAL_SHARE" not in ind:
        return base
    proxy = ind["BPS"] * ind["TOTAL_SHARE"]
    index = base.index.union(proxy.index)
    columns = base.columns.union(proxy.columns)
    base = base.reindex(index=index, columns=columns)
    proxy = proxy.reindex(index=index, columns=columns)
    return base.combine_first(proxy)


def _wide_from_long(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    wide = frame.pivot_table(index="REPORT_DATE", columns="SECURITY_CODE", values=value, aggfunc="last")
    return wide.sort_index()


def _single_quarter(wide: pd.DataFrame) -> pd.DataFrame:
    """累计值 → 单季值：Q1 原样，其余减上一季；跨年不差分（Q1 本身就是单季）。"""
    diff = wide.diff()
    is_q1 = wide.index.month == 3
    diff.loc[is_q1, :] = wide.loc[is_q1, :]
    return diff


@lru_cache(maxsize=1)
def _ttm_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (TTM 流量宽表 dict 拆成两块, 存量宽表)，索引为报告期、列为代码。"""
    flows: dict[str, pd.DataFrame] = {}
    for table, columns in FLOW_COLUMNS.items():
        long = _quarter_frame(table, columns)
        for column in columns:
            if column not in long.columns:
                continue
            wide = _wide_from_long(long, column)
            flows[column] = _single_quarter(wide).rolling(4, min_periods=4).sum()
    balance = _quarter_frame("balance", STOCK_COLUMNS)
    stocks = {column: _wide_from_long(balance, column) for column in STOCK_COLUMNS if column in balance.columns}
    return flows, stocks


@lru_cache(maxsize=1)
def _availability() -> pd.DataFrame:
    """每个 (报告期, 代码) 的公告日宽表；取三表里最晚的一个（三表齐了才算完整）。"""
    latest = None
    for table in ("profit", "balance", "cashflow"):
        long = _quarter_frame(table, [])
        notice = _wide_from_long(long, "NOTICE_DATE")
        latest = notice if latest is None else pd.concat([latest, notice]).groupby(level=0).max()
    if latest is None:
        return pd.DataFrame()
    fallback = latest.copy()
    idx = fallback.index.to_series()
    for column in fallback.columns:
        missing = fallback[column].isna()
        if missing.any():
            offset = idx.loc[missing] + pd.Timedelta(days=TTM_FALLBACK_DAYS)
            fallback.loc[missing, column] = offset.values
    return fallback


def _scatter_ffill(values: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """把「报告期 × 代码」的宽表按公告日散到「交易日 × 代码」，只向后填充。"""
    avail = _availability().reindex(index=values.index, columns=values.columns)
    mat = np.full((len(dates), len(symbols)), np.nan, dtype="float32")
    symbol_index = pd.Index(symbols)
    for pos, report_date in enumerate(values.index):
        row = values.iloc[pos]
        has = row.notna()
        if not has.any():
            continue
        codes = row.index[has]
        when = avail.iloc[pos].reindex(codes)
        drop = when.isna()
        codes = codes[~drop]
        when = when[~drop]
        if len(codes) == 0:
            continue
        target = dates.searchsorted(pd.DatetimeIndex(when), side="left")
        cols = symbol_index.get_indexer(codes)
        ok = (target < len(dates)) & (cols >= 0)
        mat[target[ok], cols[ok]] = row.reindex(codes).to_numpy(dtype="float32")[ok]
    wide = pd.DataFrame(mat, index=dates, columns=symbols)
    return wide.ffill()


def fundamental_wide(name: str, panel: Panel) -> pd.DataFrame:
    """按名字给出「交易日 × 代码」的基本面原始值宽表（未做符号调整）。

    支持的 name：``ep``/``bp``/``sp``/``cfp``/``roe``/``gross_margin``/``net_margin``/
    ``accruals``/``asset_growth``/``revenue_growth``/``debt_ratio``/``goodwill_ratio``。
    """
    dates = panel.dates
    symbols = list(panel.symbols)
    flows, stocks = _ttm_table()

    def flow(column: str) -> pd.DataFrame:
        return _scatter_ffill(flows[column], dates, symbols)

    def stock(column: str) -> pd.DataFrame:
        return _scatter_ffill(stocks[column], dates, symbols)

    if name in {"ep", "bp", "sp", "cfp"}:
        field_name = {"ep": "peTTM", "bp": "pbMRQ", "sp": "psTTM", "cfp": "pcfNcfTTM"}[name]
        return _valuation_wide(field_name, dates, symbols).pipe(_inverse)

    if name == "roe":
        return flow("PARENT_NETPROFIT") / _scatter_ffill(_equity(), dates, symbols)
    if name == "gross_margin":
        revenue = flow("TOTAL_OPERATE_INCOME")
        margin = (revenue - flow("OPERATE_COST")) / revenue.replace(0, np.nan)
        ind = _indicator()
        if "XSMLL" in ind:
            proxy = _scatter_ffill(ind["XSMLL"].reindex(index=flows["TOTAL_OPERATE_INCOME"].index) / 100.0, dates, symbols)
            margin = margin.fillna(proxy)
        return margin
    if name == "net_margin":
        revenue = flow("TOTAL_OPERATE_INCOME")
        margin = flow("PARENT_NETPROFIT") / revenue.replace(0, np.nan)
        ind = _indicator()
        if "XSJLL" in ind:
            proxy = _scatter_ffill(ind["XSJLL"].reindex(index=flows["TOTAL_OPERATE_INCOME"].index) / 100.0, dates, symbols)
            margin = margin.fillna(proxy)
        return margin
    if name == "accruals":
        assets = stock("TOTAL_ASSETS")
        return (flow("NETPROFIT") - flow("NETCASH_OPERATE")) / assets.replace(0, np.nan)
    if name == "asset_growth":
        assets = stocks["TOTAL_ASSETS"]
        return _growth(assets, dates, symbols)
    if name == "revenue_growth":
        growth = _growth(flows["TOTAL_OPERATE_INCOME"], dates, symbols)
        ind = _indicator()
        if "TOTALOPERATEREVETZ" in ind:
            proxy = _scatter_ffill(
                ind["TOTALOPERATEREVETZ"].reindex(index=flows["TOTAL_OPERATE_INCOME"].index) / 100.0, dates, symbols
            )
            growth = growth.fillna(proxy)
        return growth
    if name == "debt_ratio":
        assets = stocks["TOTAL_ASSETS"]
        ratio = stocks["TOTAL_LIABILITIES"] / assets.replace(0, np.nan)
        ind = _indicator()
        if "ZCFZL" in ind:
            proxy = ind["ZCFZL"] / 100.0
            index = ratio.index.union(proxy.index)
            columns = ratio.columns.union(proxy.columns)
            ratio = ratio.reindex(index=index, columns=columns)
            proxy = proxy.reindex(index=index, columns=columns)
            ratio = ratio.combine_first(proxy)
        return _scatter_ffill(ratio, dates, symbols)
    if name == "goodwill_ratio":
        return stock("GOODWILL").fillna(0.0) / stock("TOTAL_ASSETS").replace(0, np.nan)
    if name == "size":
        shares = _scatter_ffill(_indicator()["TOTAL_SHARE"], dates, symbols)
        return np.log(panel.close * shares)
    raise KeyError(f"未知基本面因子 {name!r}")


def _growth(wide: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """同比增长：与 4 个报告期前比（宽表按报告期排好，shift(4) 就是去年同期）。"""
    base = wide.shift(4)
    growth = wide / base.replace(0, np.nan) - 1.0
    return _scatter_ffill(growth, dates, symbols)


@lru_cache(maxsize=8)
def _valuation_field(field_name: str) -> pd.DataFrame:
    """估值字段的「交易日 × 代码」矩阵（原始 per-symbol parquet 拼装，按字段缓存）。"""
    from quant import valuation as vd

    series_map: dict[str, pd.Series] = {}
    for path in sorted(vd.VALUATION_DIR.glob("??????.parquet")):
        one = pd.read_parquet(path, columns=["date", field_name])
        if len(one) == 0:
            continue
        series_map[path.stem] = pd.Series(one[field_name].to_numpy(), index=pd.to_datetime(one["date"]))
    if not series_map:
        return pd.DataFrame(dtype="float32")
    wide = pd.DataFrame(series_map).sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide.astype("float32")


def _valuation_wide(field_name: str, dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    wide = _valuation_field(field_name)
    if wide.empty:
        return pd.DataFrame(index=dates, columns=symbols, dtype="float32")
    return wide.reindex(index=dates, columns=symbols).ffill()


def _inverse(frame: pd.DataFrame) -> pd.DataFrame:
    """PE→EP 这类取倒数；PE<=0（亏损）时 EP 记 NaN，避免“越亏越便宜”。"""
    return (1.0 / frame.where(frame > 0)).astype("float32")


def availability_grid(panel: Panel) -> pd.DataFrame:
    """每格是否「财务数据已公告」（True=可用）。用于在回测里剔除财报空窗期的票。"""
    ann = _availability()
    values = ann.notna().astype(float)
    return _scatter_ffill(values, panel.dates, list(panel.symbols)) > 0


# --------------------------------------------------------------------------
# 注册到因子库
# --------------------------------------------------------------------------

_SPECS = [
    ("ep", "positive", "盈利收益率 1/PE(TTM)：价值因子，越大越便宜（亏损记 NaN）"),
    ("bp", "positive", "账面市值比 1/PB(MRQ)：价值因子"),
    ("sp", "positive", "营收市值比 1/PS(TTM)：价值因子，对亏损股也有效"),
    ("cfp", "positive", "现金流市值比 1/PCF(TTM)：现金牛"),
    ("roe", "positive", "TTM 归母净利 / 归母净资产：质量"),
    ("gross_margin", "positive", "TTM 毛利率：定价权"),
    ("net_margin", "positive", "TTM 归母净利率"),
    ("accruals", "negative", "应计 (净利-经营现金流)/总资产：越高越差（Sloan 应计异象）"),
    ("asset_growth", "negative", "总资产同比增速：扩张越猛后续越差（投资因子）"),
    ("revenue_growth", "positive", "营收 TTM 同比：成长"),
    ("debt_ratio", "negative", "资产负债率：杠杆越高越脆弱"),
    ("goodwill_ratio", "negative", "商誉/总资产：减值风险"),
    ("size", "negative", "ln(总市值)：规模因子（小市值溢价，故方向取负）"),
]


def register() -> list[str]:
    """把基本面因子注册进 ``FACTORS``（可重复调用）；返回注册的因子名。

    统一约定：注册进库的因子「值越大越看多」，所以负面因子靠 ``sign=-1`` 在
    ``library.compute`` 里翻转，因子函数本身返回原始经济学值。
    """
    from quant.factors import library

    for name, direction, description in _SPECS:
        if name in library.FACTORS:
            continue
        sign = -1 if direction == "negative" else 1
        library.register_factor(
            name,
            lambda panel, _name=name: fundamental_wide(_name, panel),
            direction=direction,
            category="基本面",
            description=description,
            sign=sign,
        )
    return [name for name, _, _ in _SPECS]
