"""小盘成长策略专用因子（G / E / A 系列）——独立模块，**不注册**进共享 ``FACTORS``。

设计依据见 ``docs/小盘成长策略设计与数据核对.md`` §3（2026-10-19 冻结，含 Q1–Q8 拍板）。
与现有 ``quant/factors/fundamental.py`` 的四处关键差异：

1. **自带单季/TTM 换算**，覆盖 ``DEDUCT_PARENT_NETPROFIT``（扣非）与 ``RESEARCH_EXPENSE``（研发），
   旧模块的 ``FLOW_COLUMNS`` 两者都没有，等于没有可用的扣非增速。
2. **只读报告期 ≤ 2024-12-31 的财报文件**（``REPORT_CUTOFF``）。旧模块用 ``glob`` 会把 2025+ 报告期
   一起读进来，违反 holdout「一行不读」的纪律。
3. **增长率要求两端基期为正**（Q2 拍板）：负基期记 NaN，不使用现成的 ``KCFJCXSYJLRTZ`` 一类字段
   ——实测 18.2% 的样本基期扣非 ≤ 0，那些「增速」是垃圾。
4. **上市日掩码**：财务表里含招股书披露的上市前季度（2017-12 期里有 869 只当时未上市），
   散列到交易日时必须按 ``ipo_date`` 截断，否则前视。

所有因子统一「值越大越看多」（与 ``library.py`` 约定一致），需要反向的已在函数内乘 -1，
并在 ``FACTOR_META`` 的 ``direction`` 里记录原始方向。

面板约定：``panel.symbols`` 是 6 位裸码（数据湖 processed 层现状；带 ``.SZ/.SH`` 后缀也能对齐，
内部一律按裸码匹配），财报里同样是 6 位裸码。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from quant.factors import library, style
from quant.factors.panel import Panel

__all__ = [
    "BLOCKS",
    "BLOCK_WEIGHTS",
    "FACTOR_META",
    "GrowthFactors",
    "eligible_mask",
    "factor_catalog",
    "growth_factors",
    "growth_score",
    "level_yoy",
    "single_quarter",
    "ttm",
]

FUND_DIR = Path("D:/quant-data/fundamentals")
REPORT_CUTOFF = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")
FALLBACK_DAYS = 110
"""``NOTICE_DATE`` 缺失时的兜底延后天数。实测 6 位代码上缺失率 0，实际从不触发。"""

TRADING_DAYS = 244
MIN_NAMES = 20
"""截面有效样本少于该数时不计算因子（月频里通常有 1,000+ 只，只防极端日期）。"""

# 各表需要的字段（只读这些列，省内存）
COLUMNS: dict[str, tuple[str, ...]] = {
    "profit": (
        "DEDUCT_PARENT_NETPROFIT",
        "PARENT_NETPROFIT",
        "RESEARCH_EXPENSE",
        "OPERATE_INCOME",
        "OPERATE_COST",
    ),
    "balance": (
        "GOODWILL",
        "TOTAL_ASSETS",
        "TOTAL_LIABILITIES",
        "TOTAL_PARENT_EQUITY",
        "ACCOUNTS_RECE",
        "INVENTORY",
        "CONTRACT_LIAB",
        "ADVANCE_RECEIVABLES",
    ),
    "cashflow": ("NETCASH_OPERATE",),
    "indicator": ("ROIC",),
}

# 因子元数据：direction 记录原始方向，sign 是已施加的符号（统一成大数看多）
FACTOR_META: dict[str, dict] = {
    "g1_deduct_yoy": dict(block="growth", direction="positive", sign=1, cn="扣非净利润 TTM 同比", source="profit"),
    "g4_eav": dict(block="growth", direction="positive", sign=1, cn="盈余加速 EAV（单季 ROA 的同比变化之差）", source="profit+balance"),
    "e1_contract_yoy": dict(block="growth", direction="positive", sign=1, cn="合同负债+预收款 同比", source="balance"),
    "g5_stability": dict(block="quality", direction="negative", sign=-1, cn="盈利稳定性（8 单季 ROA 标准差，取负）", source="profit+balance"),
    "g8_gm_yoy": dict(block="quality", direction="positive", sign=1, cn="单季毛利率同比变化", source="profit"),
    "g9_roic": dict(block="quality", direction="positive", sign=1, cn="ROIC", source="indicator"),
    "e2_wc_gap": dict(block="quality", direction="negative", sign=-1, cn="营运资本扩张差（取负）", source="balance+profit"),
    "e6_nonrec": dict(block="quality", direction="negative", sign=-1, cn="非经常性损益依赖度（取负）", source="profit"),
    "g6_rd_intensity": dict(block="rnd", direction="positive", sign=1, cn="研发强度（TTM 研发/TTM 营收）", source="profit"),
    "e3_rd_delta": dict(block="rnd", direction="positive", sign=1, cn="研发强度变化（同比差）", source="profit"),
    "a1_reversal": dict(block="aux", direction="negative", sign=-1, cn="短期反转（60 日跳 5 日动量取负）", source="close"),
    "a2_amihud": dict(block="aux", direction="positive", sign=1, cn="非流动性 amihud", source="close+amount"),
    "a3_small": dict(block="aux", direction="negative", sign=-1, cn="小市值（-ln 总市值）", source="close+shares"),
    # 诊断项：只报告 IC，不进合成
    "g2_parent_yoy": dict(block="diag", direction="positive", sign=1, cn="归母净利润 TTM 同比（对照）", source="profit"),
    "g10_revenue_yoy": dict(block="diag", direction="positive", sign=1, cn="营收 TTM 同比（对照）", source="profit"),
}

BLOCK_WEIGHTS: dict[str, float] = {"growth": 0.40, "quality": 0.25, "rnd": 0.15, "aux": 0.20}

BLOCKS: dict[str, tuple[str, ...]] = {
    "growth": ("g1_deduct_yoy", "g4_eav", "e1_contract_yoy"),
    "quality": ("g5_stability", "g8_gm_yoy", "g9_roic", "e2_wc_gap", "e6_nonrec"),
    "rnd": ("g6_rd_intensity", "e3_rd_delta"),
    "aux": ("a1_reversal", "a2_amihud", "a3_small"),
}

_CACHE: dict[str, "GrowthFactors"] = {}


# --------------------------------------------------------------------------- #
# 纯函数：单季 / TTM / 同比
# --------------------------------------------------------------------------- #


def _quarter_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """把报告期对齐到完整季度网格（缺期补空行），供 ``diff`` 使用。"""
    if len(index) == 0:
        return index
    start = pd.Timestamp(index.min()).to_period("Q")
    end = pd.Timestamp(index.max()).to_period("Q")
    return pd.period_range(start, end, freq="Q").to_timestamp(how="end").normalize()


def single_quarter(wide: pd.DataFrame) -> pd.DataFrame:
    """累计值（YTD）→ 单季值。Q1 原样，Q2–Q4 减去上一季；缺上季则记 NaN。"""
    if wide.empty:
        return wide
    index = pd.DatetimeIndex(pd.to_datetime(wide.index))
    grid = _quarter_grid(index)
    frame = wide.copy()
    frame.index = index
    frame = frame.reindex(grid)
    prev = frame.shift(1)
    out = frame - prev
    first = pd.DatetimeIndex(frame.index).month == 3
    out.loc[first, :] = frame.loc[first, :]
    out.index = frame.index
    return out


def ttm(single: pd.DataFrame) -> pd.DataFrame:
    """单季 → TTM（滚动 4 季求和，四季齐备才算）。"""
    return single.rolling(4, min_periods=4).sum()


def level_yoy(wide: pd.DataFrame, *, periods: int = 4, positive: bool = True) -> pd.DataFrame:
    """同比：``x_t / x_{t-periods} - 1``；``positive=True`` 时要求两端 > 0（Q2 拍板）。"""
    base = wide.shift(periods)
    if positive:
        ok = (wide > 0) & (base > 0)
    else:
        ok = wide.notna() & base.notna() & (base != 0)
    return (wide / base - 1.0).where(ok)


def _mask_before(frame: pd.DataFrame, start: str) -> pd.DataFrame:
    """把 ``start`` 之前的报告期整行抹成 NaN（口径断层的处理）。"""
    keep = frame.index >= pd.Timestamp(start)
    out = frame.copy()
    out.loc[~keep, :] = np.nan
    return out


def _diff_yoy(wide: pd.DataFrame, *, periods: int = 4) -> pd.DataFrame:
    """差值同比（不做除法，天然免疫负基数）：``x_t - x_{t-periods}``。"""
    return wide - wide.shift(periods)


# --------------------------------------------------------------------------- #
# 财报读取与 PIT 散列
# --------------------------------------------------------------------------- #


def _period_files(
    table: str,
    fund_dir: Path,
    cutoff: pd.Timestamp,
    extra_dirs: Iterable[Path] = (),
) -> list[tuple[pd.Timestamp, Path]]:
    """枚举报告期文件；``extra_dirs`` 用于拼接历史补采层（同报告期以主目录为准）。"""
    out: dict[pd.Timestamp, Path] = {}
    roots = [Path(fund_dir), *(Path(item) for item in extra_dirs)]
    for index, root in enumerate(roots):
        folder = root / table
        if not folder.is_dir():
            if index == 0:
                raise FileNotFoundError(f"找不到财报目录 {folder}")
            continue
        for path in sorted(folder.glob("????-??-??.parquet")):
            try:
                stamp = pd.Timestamp(path.stem)
            except ValueError:
                continue
            if stamp > cutoff:  # holdout 隔离：报告期 > cutoff 的文件一个都不打开
                continue
            if stamp not in out:
                out[stamp] = path
    if not out:
        raise FileNotFoundError(f"{fund_dir} 在 {cutoff.date()} 之前没有报告期文件")
    return sorted(out.items())


def load_table(
    table: str,
    *,
    fund_dir: Path = FUND_DIR,
    cutoff: pd.Timestamp = REPORT_CUTOFF,
    columns: Iterable[str] | None = None,
    extra_dirs: Iterable[Path] = (),
) -> pd.DataFrame:
    """读一张财报表的长表：``code, report_date, notice_date, <字段...>``。

    - 只保留 6 位真实代码（财务表 8.4% 的行是 ``A#####`` 一类伪代码）；
    - ``NOTICE_DATE`` 缺失时按报告期 +110 天兜底；
    - 同一 (code, report_date) 重复时保留最后一行。
    """
    wanted = tuple(columns if columns is not None else COLUMNS[table])
    frames: list[pd.DataFrame] = []
    for _stamp, path in _period_files(table, fund_dir, cutoff, extra_dirs):
        frame = pd.read_parquet(path)
        use = [column for column in ["SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE", *wanted] if column in frame.columns]
        frames.append(frame[use])
    raw = pd.concat(frames, ignore_index=True)
    raw["code"] = raw["SECURITY_CODE"].astype(str).str.zfill(6)
    raw = raw[raw["code"].str.fullmatch(r"\d{6}")]
    raw["report_date"] = pd.to_datetime(raw["REPORT_DATE"])
    notice = pd.to_datetime(raw["NOTICE_DATE"], errors="coerce")
    raw["notice_date"] = notice.fillna(raw["report_date"] + pd.Timedelta(days=FALLBACK_DAYS))
    raw = raw.drop_duplicates(subset=["code", "report_date"], keep="last")
    keep = ["code", "report_date", "notice_date", *wanted]
    out = raw[keep].copy()
    for column in wanted:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["report_date", "code"]).reset_index(drop=True)


def _to_wide(long: pd.DataFrame, column: str) -> pd.DataFrame:
    return long.pivot(index="report_date", columns="code", values=column).sort_index()


def _notice_wide(long: pd.DataFrame) -> pd.DataFrame:
    return long.pivot(index="report_date", columns="code", values="notice_date").sort_index()


def _max_notice(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """两张报告期表的公告日取较晚者（跨表组合因子用）。"""
    index = left.index.union(right.index)
    columns = left.columns.union(right.columns)
    a = left.reindex(index=index, columns=columns).astype("datetime64[ns]")
    b = right.reindex(index=index, columns=columns).astype("datetime64[ns]")
    # a.where(a.notna(), b)：a 有公告日用 a，否则落回 b（NaT 一定 and 到 b 上）
    return a.where(a.notna(), b)


@dataclass(frozen=True)
class _Context:
    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    ipo_days: tuple[pd.Timestamp | None, ...]


def _scatter(
    values: pd.DataFrame,
    notice: pd.DataFrame,
    ctx: _Context,
) -> pd.DataFrame:
    """把「报告期 × 代码」的值按公告日散列到交易日，并 ffill；上市前的格子置 NaN。

    公告日当天收盘即生效（``searchsorted(side='right') - 1``），与交易日对齐后不再前移
    ——调用方（策略）本来就在 T 日收盘算信号、T+1 开盘成交。
    """
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


def _ipo_days(
    symbols: Iterable[str],
    override: Mapping[str, object] | None = None,
) -> tuple[pd.Timestamp | None, ...]:
    if override is not None:
        table = {str(key).split(".")[0]: pd.Timestamp(value) for key, value in override.items()}
        return tuple(table.get(str(symbol).split(".")[0]) for symbol in symbols)
    from quant import refdata

    universe = refdata.load_universe()
    table = dict(zip(universe["symbol"].astype(str).str.zfill(6), pd.to_datetime(universe["ipo_date"], errors="coerce")))
    out: list[pd.Timestamp | None] = []
    for symbol in symbols:
        stamp = table.get(str(symbol).split(".")[0])
        out.append(None if stamp is None or pd.isna(stamp) else pd.Timestamp(stamp))
    return tuple(out)



# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #


class GrowthFactors:
    """把财报读成「报告期宽表」→ 算单季/TTM/同比 → 按公告日散列到交易日。

    所有 ``panel()`` 方法返回 ``index=交易日, columns=标的`` 的 float32 宽表；
    第一次访问某张表时才读盘，同一进程内按 (日期, 标的) 键缓存。
    """

    def __init__(
        self,
        panel: Panel,
        *,
        fund_dir: Path = FUND_DIR,
        cutoff: pd.Timestamp = REPORT_CUTOFF,
        ipo_dates: Mapping[str, object] | None = None,
    extra_dirs: Iterable[Path] = (),
    ) -> None:
        self.panel = panel
        self.fund_dir = Path(fund_dir)
        self.cutoff = pd.Timestamp(cutoff)
        self.extra_dirs = tuple(Path(item) for item in (extra_dirs or ()))
        self.ipo_dates = dict(ipo_dates) if ipo_dates is not None else None
        self._long: dict[str, pd.DataFrame] = {}
        self._wide: dict[tuple[str, str], pd.DataFrame] = {}
        self._notice: dict[str, pd.DataFrame] = {}
        self._scattered: dict[str, pd.DataFrame] = {}
        self._ctx: _Context | None = None

    # -- 基础访问 ---------------------------------------------------------- #

    @property
    def ctx(self) -> _Context:
        if self._ctx is None:
            self._ctx = _Context(
                self.panel.dates,
                tuple(self.panel.symbols),
                _ipo_days(self.panel.symbols, self.ipo_dates),
            )
        return self._ctx

    def long(self, table: str) -> pd.DataFrame:
        if table not in self._long:
            self._long[table] = load_table(table, fund_dir=self.fund_dir, cutoff=self.cutoff,
                                        extra_dirs=self.extra_dirs)
        return self._long[table]

    def notice(self, table: str) -> pd.DataFrame:
        if table not in self._notice:
            self._notice[table] = _notice_wide(self.long(table))
        return self._notice[table]

    def wide(self, table: str, column: str) -> pd.DataFrame:
        key = (table, column)
        if key not in self._wide:
            self._wide[key] = _to_wide(self.long(table), column)
        return self._wide[key]

    def quarter(self, table: str, column: str) -> pd.DataFrame:
        """单季值（利润表/现金流量表用；资产负债表直接取 ``wide``）。"""
        return single_quarter(self.wide(table, column))

    def ttm(self, table: str, column: str) -> pd.DataFrame:
        return ttm(self.quarter(table, column))

    # -- 派生报告期指标 ---------------------------------------------------- #

    def roa_quarter(self) -> pd.DataFrame:
        """单季扣非 / 当季总资产（不做年化；分母用期末值，避免分母含未来）。"""
        assets = self.wide("balance", "TOTAL_ASSETS").replace(0, np.nan)
        return self.quarter("profit", "DEDUCT_PARENT_NETPROFIT") / assets

    def gross_margin_quarter(self) -> pd.DataFrame:
        income = self.quarter("profit", "OPERATE_INCOME")
        cost = self.quarter("profit", "OPERATE_COST")
        return ((income - cost) / income.where(income > 0)).where(income > 0)

    def rd_intensity(self) -> pd.DataFrame:
        revenue = self.ttm("profit", "OPERATE_INCOME").where(lambda x: x > 0)
        return self.ttm("profit", "RESEARCH_EXPENSE") / revenue

    def contract_advance(self) -> pd.DataFrame:
        """合同负债 + 预收款（CAS14 新准则 2020 年落地，两字段覆盖期互补，必须相加）。"""
        return self.wide("balance", "CONTRACT_LIAB").fillna(0) + self.wide("balance", "ADVANCE_RECEIVABLES").fillna(0)

    def working_capital(self) -> pd.DataFrame:
        return self.wide("balance", "ACCOUNTS_RECE") + self.wide("balance", "INVENTORY")

    def nonrec_ratio(self) -> pd.DataFrame:
        """非经常性损益依赖度 =(归母 TTM − 扣非 TTM)/|归母 TTM|。"""
        parent = self.ttm("profit", "PARENT_NETPROFIT")
        deduct = self.ttm("profit", "DEDUCT_PARENT_NETPROFIT")
        return (parent - deduct) / parent.abs().replace(0, np.nan)

    # -- 因子 --------------------------------------------------------------- #

    def _cached(self, name: str, builder) -> pd.DataFrame:
        if name not in self._scattered:
            self._scattered[name] = builder()
        return self._scattered[name]

    def g1_deduct_yoy(self) -> pd.DataFrame:
        """扣非净利润 TTM 同比（两端 > 0）。"""

        def build() -> pd.DataFrame:
            yoy = level_yoy(self.ttm("profit", "DEDUCT_PARENT_NETPROFIT"))
            return _scatter(yoy, self.notice("profit"), self.ctx)

        return self._cached("g1_deduct_yoy", build)

    def g2_parent_yoy(self) -> pd.DataFrame:
        """归母净利润 TTM 同比（诊断对照，不进合成）。"""

        def build() -> pd.DataFrame:
            yoy = level_yoy(self.ttm("profit", "PARENT_NETPROFIT"))
            return _scatter(yoy, self.notice("profit"), self.ctx)

        return self._cached("g2_parent_yoy", build)

    def g10_revenue_yoy(self) -> pd.DataFrame:
        """营收 TTM 同比（诊断对照，不进合成）。"""

        def build() -> pd.DataFrame:
            yoy = level_yoy(self.ttm("profit", "OPERATE_INCOME"))
            return _scatter(yoy, self.notice("profit"), self.ctx)

        return self._cached("g10_revenue_yoy", build)

    def g4_eav(self) -> pd.DataFrame:
        """盈余加速 EAV = ΔROA_t − ΔROA_{t−1}，ΔROA_t = roa_q_t − roa_q_{t−4}。

        用单季 + 总资产标准化（文献原式的分母是 |单季归母|，负基数会爆炸）。
        """

        def build() -> pd.DataFrame:
            roa = self.roa_quarter()
            delta = roa - roa.shift(4)
            eav = delta - delta.shift(1)
            notice = _max_notice(self.notice("profit"), self.notice("balance"))
            return _scatter(eav, notice, self.ctx)

        return self._cached("g4_eav", build)

    def e1_contract_yoy(self) -> pd.DataFrame:
        """合同负债 + 预收款 同比（营收前瞻量，两端 > 0）。"""

        def build() -> pd.DataFrame:
            yoy = level_yoy(self.contract_advance())
            return _scatter(yoy, self.notice("balance"), self.ctx)

        return self._cached("e1_contract_yoy", build)

    def g5_stability(self) -> pd.DataFrame:
        """盈利稳定性：最近 8 个单季 ROA 的标准差，取负（越稳越大）。"""

        def build() -> pd.DataFrame:
            roa = self.roa_quarter()
            vol = roa.rolling(8, min_periods=8).std()
            notice = _max_notice(self.notice("profit"), self.notice("balance"))
            return _scatter(-vol, notice, self.ctx)

        return self._cached("g5_stability", build)

    def g8_gm_yoy(self) -> pd.DataFrame:
        """单季毛利率同比变化；2021-03 之前一律 NaN（2020 CAS14 重分类，口径不可比）。"""

        def build() -> pd.DataFrame:
            gm = self.gross_margin_quarter()
            change = gm - gm.shift(4)
            change = _mask_before(change, "2021-03-31")
            return _scatter(change, self.notice("profit"), self.ctx)

        return self._cached("g8_gm_yoy", build)

    def g9_roic(self) -> pd.DataFrame:
        """ROIC（indicator 表现成字段，含金融业；金融业由排雷 R7 剔除）。"""

        def build() -> pd.DataFrame:
            return _scatter(self.wide("indicator", "ROIC"), self.notice("indicator"), self.ctx)

        return self._cached("g9_roic", build)

    def e2_wc_gap(self) -> pd.DataFrame:
        """营运资本扩张差 =(应收+存货) 增速 − 营收增速，取负（扩张越快越差）。"""

        def build() -> pd.DataFrame:
            wc = self.working_capital()
            g_wc = level_yoy(wc)
            g_rev = level_yoy(self.ttm("profit", "OPERATE_INCOME"))
            notice = _max_notice(self.notice("balance"), self.notice("profit"))
            return _scatter(-(g_wc - g_rev), notice, self.ctx)

        return self._cached("e2_wc_gap", build)

    def e6_nonrec(self) -> pd.DataFrame:
        """非经常性损益依赖度，取负（越依赖非经常性越差）。"""

        def build() -> pd.DataFrame:
            ratio = self.nonrec_ratio()
            return _scatter(-ratio, self.notice("profit"), self.ctx)

        return self._cached("e6_nonrec", build)

    def g6_rd_intensity(self) -> pd.DataFrame:
        """研发强度 = TTM 研发 / TTM 营收；2018-01 之前 NaN（研发覆盖 2017Q3 才有 76%）。"""

        def build() -> pd.DataFrame:
            intensity = self.rd_intensity()
            intensity = _mask_before(intensity, "2018-01-01")
            return _scatter(intensity, self.notice("profit"), self.ctx)

        return self._cached("g6_rd_intensity", build)

    def e3_rd_delta(self) -> pd.DataFrame:
        """研发强度变化（同比差）；2018-01 之前 NaN。"""

        def build() -> pd.DataFrame:
            intensity = self.rd_intensity()
            delta = intensity - intensity.shift(4)
            delta = _mask_before(delta, "2018-01-01")
            return _scatter(delta, self.notice("profit"), self.ctx)

        return self._cached("e3_rd_delta", build)

    def a1_reversal(self) -> pd.DataFrame:
        """短期反转：60 日跳 5 日动量取负（现有库实测 IC 为负，反向用）。"""

        def build() -> pd.DataFrame:
            return -library.momentum(self.panel, lookback=60, skip=5)

        return self._cached("a1_reversal", build)

    def a2_amihud(self) -> pd.DataFrame:
        """非流动性（正向：越不流动越看多）。"""

        def build() -> pd.DataFrame:
            return library.amihud(self.panel, window=20)

        return self._cached("a2_amihud", build)

    def a3_small(self) -> pd.DataFrame:
        """小市值：-ln(总市值)，PIT 股本（不是 MAINFINADATA.TOTAL_SHARE）。"""

        def build() -> pd.DataFrame:
            mcap = style.mcap_panel(self.panel)
            return -np.log(mcap.where(mcap > 0))

        return self._cached("a3_small", build)

    # -- 门槛与排雷 --------------------------------------------------------- #

    def g3_gate(self) -> pd.DataFrame:
        """连续正增长门槛：最近 4 个单季里「扣非单季同比 > 0（基期 > 0）」≥ 2 季。"""

        def build() -> pd.DataFrame:
            single = self.quarter("profit", "DEDUCT_PARENT_NETPROFIT")
            base = single.shift(4)
            good = (single > 0) & (base > 0) & ((single / base - 1.0) > 0)
            count = good.astype("float32").rolling(4, min_periods=4).sum()
            passed = (count >= 2).astype("float32")
            passed = passed.where(count.notna())
            return _scatter(passed, self.notice("profit"), self.ctx).eq(1)

        return self._cached("g3_gate", build)

    def deduct_ttm(self, *, signed: bool = False) -> pd.DataFrame:
        """扣非 TTM（排雷 R5、R8 用）。``signed=True`` 时保留负值。"""

        def build() -> pd.DataFrame:
            values = self.ttm("profit", "DEDUCT_PARENT_NETPROFIT")
            return _scatter(values, self.notice("profit"), self.ctx)

        name = "deduct_ttm" if not signed else "deduct_ttm_signed"
        return self._cached(name, build)

    def ocf_ttm(self) -> pd.DataFrame:
        def build() -> pd.DataFrame:
            return _scatter(self.ttm("cashflow", "NETCASH_OPERATE"), self.notice("cashflow"), self.ctx)

        return self._cached("ocf_ttm", build)

    def total_assets(self) -> pd.DataFrame:
        def build() -> pd.DataFrame:
            return _scatter(self.wide("balance", "TOTAL_ASSETS"), self.notice("balance"), self.ctx)

        return self._cached("total_assets", build)

    def goodwill_ratio(self) -> pd.DataFrame:
        """商誉 / 归母净资产；商誉缺失≈无商誉，按 0 处理（§1.7）。"""

        def build() -> pd.DataFrame:
            goodwill = self.wide("balance", "GOODWILL").fillna(0.0)
            equity = self.wide("balance", "TOTAL_PARENT_EQUITY").where(lambda x: x > 0)
            return _scatter(goodwill / equity, self.notice("balance"), self.ctx)

        return self._cached("goodwill_ratio", build)

    def debt_ratio(self) -> pd.DataFrame:
        def build() -> pd.DataFrame:
            assets = self.wide("balance", "TOTAL_ASSETS").where(lambda x: x > 0)
            return _scatter(self.wide("balance", "TOTAL_LIABILITIES") / assets, self.notice("balance"), self.ctx)

        return self._cached("debt_ratio", build)

    def nonrec_level(self) -> pd.DataFrame:
        """非经常性占比原始值（R9 用，正数表示靠非经常性撑利润）。"""

        def build() -> pd.DataFrame:
            return _scatter(self.nonrec_ratio(), self.notice("profit"), self.ctx)

        return self._cached("nonrec_level", build)

    def listing_days(self) -> pd.DataFrame:
        """上市天数（交易日近似用自然日；R2 的门槛在策略层做）。"""
        ctx = self.ctx
        out = np.full((len(ctx.dates), len(ctx.symbols)), np.nan, dtype="float32")
        days = ctx.dates.to_numpy(dtype="datetime64[D]")
        for position, ipo in enumerate(ctx.ipo_days):
            if ipo is None:
                continue
            out[:, position] = (days - np.datetime64(ipo, "D")).astype("float32")
        return pd.DataFrame(out, index=ctx.dates, columns=list(ctx.symbols))

    # -- 汇总 --------------------------------------------------------------- #

    def all_factors(self) -> dict[str, pd.DataFrame]:
        """13 个 A 类因子 + 2 个诊断因子（诊断项的 key 以 ``diag_`` 开头，不进合成）。"""
        out = {name: getattr(self, name)() for name in BLOCKS["growth"] + BLOCKS["quality"] + BLOCKS["rnd"] + BLOCKS["aux"]}
        out["diag_g2_parent_yoy"] = self.g2_parent_yoy()
        out["diag_g10_revenue_yoy"] = self.g10_revenue_yoy()
        return out


# --------------------------------------------------------------------------- #
# 合格集与合成分数
# --------------------------------------------------------------------------- #


def eligible_mask(
    factors: GrowthFactors,
    *,
    st_mask: pd.DataFrame | None = None,
    block_mask: pd.DataFrame | None = None,
    min_listing_days: int = 365,
    goodwill_cap: float = 0.30,
    debt_cap: float = 0.70,
    nonrec_cap: float = 0.50,
) -> pd.DataFrame:
    """「成长合格集」布尔掩码：R1–R9 + G3 门槛 + 次新/停牌/上市前。

    R10（市值地板）按 Q5 **默认不启用**，需要时在调用方另外叠加。
    金融业（R7 的一部分）不在这里判：银行/券商/保险的 ``OPERATE_COST`` 为空，
    毛利率与研发因子自然 NaN，且总资产口径不同；策略层用行业表整体剔除。
    """
    deduct = factors.deduct_ttm(signed=True)
    ocf = factors.ocf_ttm()
    assets = factors.total_assets()
    mask = factors.panel.tradable.copy()
    mask &= deduct.notna() & (deduct > 0)  # R5 扣非 TTM > 0
    mask &= factors.listing_days() >= min_listing_days  # R2 次新
    mask &= factors.g3_gate()  # G3 连续正增长门槛
    mask &= factors.goodwill_ratio().fillna(0) <= goodwill_cap  # R6
    mask &= factors.debt_ratio().fillna(1.0) <= debt_cap  # R7
    cash_ok = (ocf / deduct.replace(0, np.nan) >= 0)
    asset_ok = ((deduct - ocf) / assets.replace(0, np.nan) >= -0.10)
    mask &= (cash_ok | asset_ok)  # R8 经营现金流背离
    mask &= factors.nonrec_level().fillna(0) <= nonrec_cap  # R9
    if st_mask is not None:
        mask &= ~st_mask.reindex_like(mask).fillna(False).astype(bool)
    if block_mask is not None:
        mask &= ~block_mask.reindex_like(mask).fillna(False).astype(bool)
    return mask.fillna(False).astype(bool)


def _rank_pct(values: pd.DataFrame, eligible: pd.DataFrame | None, *, winsor: float = 0.01, min_names: int = MIN_NAMES) -> pd.DataFrame:
    """截面 rank 百分位（只在合格集内排），先按 1%/99% 缩尾。"""
    frame = values.where(eligible) if eligible is not None else values
    if winsor and winsor > 0:
        lower = frame.quantile(winsor, axis=1)
        upper = frame.quantile(1 - winsor, axis=1)
        frame = frame.clip(lower=lower, upper=upper, axis=0)
    count = frame.notna().sum(axis=1)
    ranked = frame.rank(axis=1, pct=True)
    return ranked.where(count >= min_names)


def growth_score(
    panel: Panel,
    *,
    factors: GrowthFactors | None = None,
    eligible: pd.DataFrame | None = None,
    blocks: Mapping[str, tuple[str, ...]] = BLOCKS,
    weights: Mapping[str, float] = BLOCK_WEIGHTS,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """分块等权合成分数（Q6：成长 40 / 质量 25 / 研发 15 / 辅助 20）。

    - 每个因子先在合格集内做 rank_pct（winsorize 1%/99%）；
    - 块内对**有值**的因子等权平均；
    - 块间按权重加权，某块当天全空（如 2017 年研发块）就按剩余块的权重重新归一
      ——Q1 拍板的处理方式，等价于「该块暂时不存在」，而不是给中性分。
    """
    gf = factors or GrowthFactors(panel)
    scores: dict[str, pd.DataFrame] = {}
    for block, names in blocks.items():
        ranks = []
        for name in names:
            values = getattr(gf, name)()
            ranks.append(_rank_pct(values, eligible, min_names=min_names))
        if not ranks:
            continue
        stack = np.stack([frame.to_numpy(dtype="float32") for frame in ranks], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            block_score = np.nanmean(stack, axis=0)
        scores[block] = pd.DataFrame(block_score, index=panel.dates, columns=panel.symbols)
    if not scores:
        return pd.DataFrame(np.nan, index=panel.dates, columns=panel.symbols)
    names = list(scores)
    blocks_np = np.stack([scores[name].to_numpy(dtype="float32") for name in names], axis=0)
    weight_np = np.array([weights.get(name, 0.0) for name in names], dtype="float32")[:, None, None]
    available = ~np.isnan(blocks_np)
    weight_sum = np.where(available, weight_np, 0.0).sum(axis=0)
    weighted = np.nansum(np.where(available, blocks_np * weight_np, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = np.where(weight_sum > 0, weighted / weight_sum, np.nan)
    return pd.DataFrame(total, index=panel.dates, columns=panel.symbols).astype("float32")


def growth_factors(panel: Panel, **kwargs) -> dict[str, pd.DataFrame]:
    """一次算齐 13 个 A 类因子（+2 个诊断项，key 前缀 ``diag_``）。"""
    cached = _CACHE.get(_cache_key(panel))
    gf = cached or GrowthFactors(panel, **kwargs)
    _CACHE[_cache_key(panel)] = gf
    return gf.all_factors()


def _cache_key(panel: Panel) -> str:
    dates = panel.dates
    symbols = panel.symbols
    if len(dates) == 0 or not symbols:
        return "empty"
    return f"{dates[0]}~{dates[-1]}|{len(dates)}|{symbols[0]}~{symbols[-1]}|{len(symbols)}"


def factor_catalog() -> pd.DataFrame:
    rows = [
        {
            "factor": name,
            "block": meta["block"],
            "direction": meta["direction"],
            "sign": meta["sign"],
            "description": meta["cn"],
            "source": meta["source"],
        }
        for name, meta in FACTOR_META.items()
    ]
    return pd.DataFrame(rows)
