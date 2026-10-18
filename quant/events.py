"""事件层：龙虎榜 / 回购 / 大股东增减持 / ST 翻转 / 高送转分红预案。

设计要点（PIT 是硬约束）
- 每张表都有 **事件发生日 ``event_date``** 与 **可见日 ``available_date``**：
  - 龙虎榜：上榜日收盘后可见 → ``available_date = event_date``（因子在 t 日收盘算、t+1 开盘才成交）；
  - 回购：以「最新公告日期」为可见日（不用后续进度回填）；
  - 增减持：接口只给变动日，公告滞后 1—2 天 → ``available_date = 变动日 + 2 个交易日``；
  - ST 翻转：翻转日收盘后可见 → ``available_date = event_date``；
  - 分红预案：``available_date = 预案公告日 PLAN_NOTICE_DATE``。
- **落盘全量**（2014 起抓到最新），但 ``load()`` 默认裁到 2024-12-31（holdout 锁），
  只有显式 ``allow_holdout=True`` 才返回 2025 起的行（供下一轮最终验证，本轮代码不调用）。
- 龙虎榜的「上榜后 N 日涨跌幅」属于未来信息，只作为原始字段留存，**不进任何因子**。
- 落盘目录 ``D:/quant-data/events/``，每事件一张 parquet + 一份 ``events_meta.json``。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS_DIR = Path("D:/quant-data/events")
META = EVENTS_DIR / "events_meta.json"
HOLDOUT_DIR = Path("D:/quant-data/holdout/events")
HOLDOUT_START = pd.Timestamp("2025-01-01")
CUTOFF = pd.Timestamp("2024-12-31")
START = pd.Timestamp("2014-01-01")

TABLES = ("lhb", "lhb_inst", "repurchase", "hold", "st", "fhps")
HOLDOUT_FILES = {name: HOLDOUT_DIR / f"{name}.parquet" for name in TABLES}
FILES = {name: EVENTS_DIR / f"{name}.parquet" for name in TABLES}

# 龙虎榜上榜原因归类（用于「方向/强度」分组，规则可复现）
LHB_REASON_RULES = (
    ("涨幅", ("涨幅", "涨幅偏离", "涨停")),
    ("跌幅", ("跌幅", "跌幅偏离", "跌停")),
    ("振幅", ("振幅", "振幅值")),
    ("换手", ("换手率",)),
    ("其他", ()),
)


def _ak():
    import akshare as ak  # 延迟导入：只有抓取时才需要

    return ak


# ---------------------------------------------------------------------------
# 通达日历
# ---------------------------------------------------------------------------


def _trading_days() -> pd.DatetimeIndex:
    from quant import refdata

    return refdata.trade_days()


def shift_trading_days(days: pd.Series, steps: int) -> pd.Series:
    """把「自然日」按交易日历向后推 ``steps`` 个交易日（找不到则保留 NaN）。"""
    calendar = _trading_days()
    if steps == 0:
        return pd.to_datetime(days)
    values = pd.to_datetime(days)
    index = calendar.searchsorted(values.values, side="left") + steps
    valid = (index >= 0) & (index < len(calendar))
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    out.loc[valid] = calendar[np.asarray(index)[valid]]
    return out


def _write(name: str, frame: pd.DataFrame) -> dict:
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = FILES[name]
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return {"rows": int(len(frame)), "path": str(path), "bytes": int(path.stat().st_size)}


def _update_meta(name: str, info: dict) -> None:
    meta = json.loads(META.read_text(encoding="utf-8")) if META.is_file() else {}
    meta[name] = info
    meta["updated_at"] = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. 龙虎榜（明细 + 机构口径）
# ---------------------------------------------------------------------------


def _lhb_reason_group(reason: str) -> str:
    text = str(reason)
    for label, keys in LHB_REASON_RULES:
        if not keys:
            return label
        if any(key in text for key in keys):
            return label
    return "其他"


def _chunks(start: pd.Timestamp, end: pd.Timestamp, *, years: int = 1) -> list[tuple[str, str]]:
    out = []
    begin = pd.Timestamp(start)
    while begin <= end:
        stop = min(begin + pd.DateOffset(years=years) - pd.Timedelta(days=1), end)
        out.append((begin.strftime("%Y%m%d"), stop.strftime("%Y%m%d")))
        begin = stop + pd.Timedelta(days=1)
    return out


def _lhb_detail_chunk(start: str, end: str) -> pd.DataFrame:
    ak = _ak()
    frame = pd.DataFrame(ak.stock_lhb_detail_em(start_date=start, end_date=end))
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "代码": "symbol",
            "名称": "code_name",
            "上榜日": "event_date",
            "解读": "interpretation",
            "收盘价": "close",
            "涨跌幅": "pct_chg",
            "龙虎榜净买额": "net_buy",
            "龙虎榜买入额": "buy_amount",
            "龙虎榜卖出额": "sell_amount",
            "龙虎榜成交额": "lhb_amount",
            "市场总成交额": "market_amount",
            "净买额占总成交比": "net_buy_ratio",
            "成交额占总成交比": "amount_ratio",
            "换手率": "turnover",
            "流通市值": "float_mcap",
            "上榜原因": "reason",
            "上榜后1日": "post_1d",
            "上榜后2日": "post_2d",
            "上榜后5日": "post_5d",
            "上榜后10日": "post_10d",
        }
    )
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame["available_date"] = frame["event_date"]  # 收盘后可见
    for column in ("close", "pct_chg", "net_buy", "buy_amount", "sell_amount", "lhb_amount",
                   "market_amount", "net_buy_ratio", "amount_ratio", "turnover", "float_mcap",
                   "post_1d", "post_2d", "post_5d", "post_10d"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["reason_group"] = frame["reason"].map(_lhb_reason_group)
    frame["direction"] = frame["net_buy"].map(lambda x: "净买" if pd.notna(x) and x > 0 else "净卖")
    return frame.drop(columns=["序号"], errors="ignore")


def _lhb_inst_chunk(start: str, end: str) -> pd.DataFrame:
    ak = _ak()
    frame = pd.DataFrame(ak.stock_lhb_jgmmtj_em(start_date=start, end_date=end))
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "代码": "symbol",
            "名称": "code_name",
            "上榜日期": "event_date",
            "收盘价": "close",
            "涨跌幅": "pct_chg",
            "买方机构数": "buy_inst_count",
            "卖方机构数": "sell_inst_count",
            "机构买入总额": "inst_buy",
            "机构卖出总额": "inst_sell",
            "机构买入净额": "inst_net",
            "市场总成交额": "market_amount",
            "机构净买额占总成交额比": "inst_net_ratio",
            "换手率": "turnover",
            "流通市值": "float_mcap",
            "上榜原因": "reason",
        }
    )
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame["available_date"] = frame["event_date"]
    for column in ("close", "pct_chg", "buy_inst_count", "sell_inst_count", "inst_buy",
                   "inst_sell", "inst_net", "market_amount", "inst_net_ratio", "turnover",
                   "float_mcap"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["reason_group"] = frame["reason"].map(_lhb_reason_group)
    frame["direction"] = frame["inst_net"].map(lambda x: "机构净买" if pd.notna(x) and x > 0 else "机构净卖")
    return frame.drop(columns=["序号"], errors="ignore")


def fetch_lhb(*, start: str = "2014-01-01", end: str | None = None, pause: float = 0.2,
              resume: bool = True) -> dict:
    """按年抓龙虎榜明细与机构统计；已有覆盖的年份直接跳过（断点续跑）。"""
    end_day = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    existing = {}
    for name in ("lhb", "lhb_inst"):
        if resume and FILES[name].is_file():
            frame = pd.read_parquet(FILES[name])
            frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
            existing[name] = frame
        else:
            existing[name] = pd.DataFrame()
    covered = {
        name: (set(frame["event_date"].dropna().dt.year.unique().tolist()) if len(frame) else set())
        for name, frame in existing.items()
    }
    log = []
    t_all = time.time()
    for begin, stop in _chunks(pd.Timestamp(start), end_day):
        year = int(begin[:4])
        if year in covered["lhb"] and year in covered["lhb_inst"]:
            continue
        t0 = time.time()
        try:
            detail = _lhb_detail_chunk(begin, stop)
            inst = _lhb_inst_chunk(begin, stop)
        except Exception as exc:  # noqa: BLE001
            log.append({"chunk": begin, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {begin}: {type(exc).__name__}: {exc}", flush=True)
            continue
        frames = {"lhb": detail, "lhb_inst": inst}
        for name, frame in frames.items():
            if frame.empty:
                continue
            merged = pd.concat([existing[name], frame], ignore_index=True)
            merged = merged.drop_duplicates(["symbol", "event_date", "reason"], keep="last")
            merged = merged.sort_values(["event_date", "symbol"]).reset_index(drop=True)
            existing[name] = merged
            covered[name].add(year)
        item = {
            "chunk": begin,
            "rows_lhb": int(len(detail)),
            "rows_inst": int(len(inst)),
            "seconds": round(time.time() - t0, 1),
        }
        log.append(item)
        print(f"  {begin}: lhb {len(detail)} / inst {len(inst)} / {item['seconds']}s", flush=True)
        time.sleep(pause)
    out = {}
    for name, frame in existing.items():
        if frame.empty:
            continue
        # 未来信息字段只留存，不参与计算
        out[name] = _write(name, frame)
        out[name]["years"] = sorted(frame["event_date"].dropna().dt.year.unique().tolist())
        out[name]["seconds"] = round(time.time() - t_all, 1)
        out[name]["log"] = log
        out[name]["source"] = "akshare stock_lhb_detail_em / stock_lhb_jgmmtj_em"
        _update_meta(name, {k: v for k, v in out[name].items() if k != "log"})
    return out


# ---------------------------------------------------------------------------
# 2. 回购
# ---------------------------------------------------------------------------


def fetch_repurchase(*, pause: float = 0.0) -> dict:
    ak = _ak()
    frame = pd.DataFrame(ak.stock_repurchase_em())
    frame = frame.rename(
        columns={
            "股票代码": "symbol",
            "股票简称": "code_name",
            "最新价": "price",
            "计划回购价格区间": "plan_price",
            "计划回购数量区间-下限": "plan_shares_low",
            "计划回购数量区间-上限": "plan_shares_high",
            "占公告前一日总股本比例-下限": "plan_ratio_low",
            "占公告前一日总股本比例-上限": "plan_ratio_high",
            "计划回购金额区间-下限": "plan_amount_low",
            "计划回购金额区间-上限": "plan_amount_high",
            "回购起始时间": "start_date",
            "实施进度": "progress",
            "已回购股份价格区间-下限": "done_price_low",
            "已回购股份价格区间-上限": "done_price_high",
            "已回购股份数量": "done_shares",
            "已回购金额": "done_amount",
            "最新公告日期": "notice_date",
        }
    )
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["notice_date"] = pd.to_datetime(frame["notice_date"], errors="coerce")
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    # PIT：可见日 = 最新公告日期（不用后续进度回填）；事件发生日同公告日
    frame["event_date"] = frame["notice_date"]
    frame["available_date"] = frame["notice_date"]
    for column in ("price", "plan_price", "plan_shares_low", "plan_shares_high",
                   "plan_ratio_low", "plan_ratio_high", "plan_amount_low", "plan_amount_high",
                   "done_price_low", "done_price_high", "done_shares", "done_amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["buffer"] = frame["plan_price"] / frame["price"] - 1.0  # 计划回购价相对现价溢价
    frame["direction"] = "回购"
    frame = frame.drop(columns=["序号"], errors="ignore")
    frame = frame.drop_duplicates(["symbol", "notice_date"], keep="last")
    frame = frame.sort_values(["event_date", "symbol"]).reset_index(drop=True)
    info = _write("repurchase", frame)
    info.update({
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start": str(frame["event_date"].min().date()) if len(frame) else None,
        "end": str(frame["event_date"].max().date()) if len(frame) else None,
        "source": "akshare stock_repurchase_em",
        "pit": "available_date = 最新公告日期",
        "log": [],
    })
    _update_meta("repurchase", {k: v for k, v in info.items() if k != "log"})
    time.sleep(pause)
    return info


# ---------------------------------------------------------------------------
# 3. 大股东增减持
# ---------------------------------------------------------------------------


def fetch_hold(*, pause: float = 0.0) -> dict:
    ak = _ak()
    frame = pd.DataFrame(ak.stock_hold_management_detail_em())
    frame = frame.rename(
        columns={
            "日期": "event_date",
            "代码": "symbol",
            "名称": "code_name",
            "变动人": "holder",
            "变动股数": "shares",
            "成交均价": "price",
            "变动金额": "amount",
            "变动原因": "reason",
            "变动比例": "ratio",
            "变动后持股数": "hold_after",
            "持股种类": "share_type",
            "董监高人员姓名": "insider",
            "职务": "role",
            "变动人与董监高的关系": "relation",
            "开始时持有": "hold_before",
            "结束后持有": "hold_after_raw",
        }
    )
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    # PIT：公告滞后 1—2 天，可用日一律按变动日 + 2 个交易日
    frame["available_date"] = shift_trading_days(frame["event_date"], 2)
    for column in ("shares", "price", "amount", "ratio", "hold_before", "hold_after", "hold_after_raw"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["direction"] = np.where(frame["shares"] > 0, "增持", "减持")
    frame["net_shares"] = frame["shares"]
    frame = frame.drop(columns=["序号"], errors="ignore")
    frame = frame.drop_duplicates(["symbol", "event_date", "holder", "shares", "amount"], keep="last")
    frame = frame.sort_values(["event_date", "symbol"]).reset_index(drop=True)
    info = _write("hold", frame)
    info.update({
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start": str(frame["event_date"].min().date()) if len(frame) else None,
        "end": str(frame["event_date"].max().date()) if len(frame) else None,
        "source": "akshare stock_hold_management_detail_em",
        "pit": "available_date = 变动日 + 2 个交易日",
        "log": [],
    })
    _update_meta("hold", {k: v for k, v in info.items() if k != "log"})
    time.sleep(pause)
    return info


# ---------------------------------------------------------------------------
# 4. ST 翻转（本地 valuation 层，不联网）
# ---------------------------------------------------------------------------


def build_st(*, start: str = "2014-01-01", include_holdout: bool = False) -> dict:
    """从 ``valuation.st_matrix()`` 的逐日 isST 推导戴帽/摘帽日。

    默认只读 ``<= 2024-12-31``（holdout 锁）；``include_holdout=True`` 时把
    ``holdout/valuation`` 一起读进来，翻转事件写 ``holdout/events/st.parquet``（不改样本内文件）。
    翻转日收盘后可见，故 ``available_date = event_date``。
    """
    from quant import valuation

    if include_holdout:
        base = valuation.st_matrix()
        extra = valuation.st_matrix(root=valuation.HOLDOUT_DIR)
        both = base.combine_first(extra)
        matrix = both.loc[both.index >= pd.Timestamp(start)].sort_index()
        matrix = matrix.fillna(False).astype(bool)
    else:
        matrix = valuation.st_matrix()
        matrix = matrix.loc[(matrix.index >= pd.Timestamp(start)) & (matrix.index <= CUTOFF)]
        matrix = matrix.sort_index()
    # 注意：bool 列 shift 后变成 object，``~False`` 是 -1（真值），
    # 必须先 fillna 再 astype(bool)，否则「每天都是翻转日」。
    prev = matrix.shift(1).fillna(False).astype(bool)
    became = matrix & ~prev
    left = ~matrix & prev
    # 首日没有前一日状态，不能算翻转（否则 2014-01-02 全部 ST 会被记成戴帽）
    became.iloc[0] = False
    left.iloc[0] = False
    rows = []
    for direction, mask in (("戴帽", became), ("摘帽", left)):
        stacked = mask.stack()
        stacked = stacked[stacked]
        for (day, symbol), _ in stacked.items():
            rows.append({"symbol": str(symbol).zfill(6), "event_date": day, "direction": direction})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"rows": 0}
    frame["available_date"] = frame["event_date"]
    frame = frame.sort_values(["event_date", "symbol"]).reset_index(drop=True)
    frame["is_st"] = frame["direction"].eq("戴帽").astype(int)
    if include_holdout:
        frame = frame[frame["available_date"] >= HOLDOUT_START].reset_index(drop=True)
        HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
        target = HOLDOUT_FILES["st"]
        tmp = target.with_suffix(".tmp.parquet")
        frame.to_parquet(tmp, index=False)
        tmp.replace(target)
        info = {"path": str(target), "rows": int(len(frame))}
    else:
        info = _write("st", frame)
    info.update({
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start": str(frame["event_date"].min().date()),
        "end": str(frame["event_date"].max().date()),
        "source": "本机 valuation 层 isST 逐日翻转",
        "pit": "翻转日收盘后可见",
        "log": [],
    })
    _update_meta("st", {k: v for k, v in info.items() if k != "log"})
    return info


# ---------------------------------------------------------------------------
# 5. 高送转 / 分红预案
# ---------------------------------------------------------------------------


def fhps_periods(start: str = "2013-12-31") -> list[str]:
    days = pd.date_range(start=start, end=pd.Timestamp.today().normalize(), freq="QE")
    return [day.strftime("%Y%m%d") for day in days]


def fetch_fhps(*, start: str = "2013-12-31", pause: float = 0.05, resume: bool = True) -> dict:
    ak = _ak()
    existing = pd.read_parquet(FILES["fhps"]) if resume and FILES["fhps"].is_file() else pd.DataFrame()
    have = set(pd.to_datetime(existing["report_date"]).dt.strftime("%Y%m%d")) if len(existing) else set()
    frames = [existing] if len(existing) else []
    log = []
    t_all = time.time()
    for period in fhps_periods(start):
        if period in have:
            continue
        t0 = time.time()
        try:
            frame = pd.DataFrame(ak.stock_fhps_em(date=period))
        except Exception as exc:  # noqa: BLE001
            log.append({"period": period, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if frame.empty:
            log.append({"period": period, "rows": 0})
            time.sleep(pause)
            continue
        frame = frame.rename(
            columns={
                "代码": "symbol",
                "名称": "code_name",
                "送转股份-送转总比例": "transfer_ratio",
                "送转股份-送转比例": "bonus_ratio",
                "送转股份-转股比例": "convert_ratio",
                "现金分红-现金分红比例": "div_ratio",
                "现金分红-股息率": "div_yield",
                "每股收益": "eps",
                "每股净资产": "bps",
                "总股本": "total_shares",
                "预案公告日": "plan_notice",
                "股权登记日": "record_date",
                "除权除息日": "ex_date",
                "方案进度": "progress",
                "最新公告日期": "latest_notice",
            }
        )
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        frame["report_date"] = pd.to_datetime(period)
        for column in ("plan_notice", "record_date", "ex_date", "latest_notice"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        for column in ("transfer_ratio", "bonus_ratio", "convert_ratio", "div_ratio",
                       "div_yield", "eps", "bps", "total_shares"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)
        log.append({"period": period, "rows": int(len(frame)), "seconds": round(time.time() - t0, 1)})
        print(f"  {period}: {len(frame)} 行 / {log[-1]['seconds']}s", flush=True)
        time.sleep(pause)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out["event_date"] = out["plan_notice"]
        out["available_date"] = out["plan_notice"]  # 预案公告日
        out["is_transfer"] = (out["transfer_ratio"].fillna(0) > 0).astype(int)
        out["is_high_transfer"] = (out["transfer_ratio"].fillna(0) >= 5).astype(int)  # 每 10 股送转 ≥5
        out = out.drop_duplicates(["symbol", "report_date"], keep="last")
        out = out.sort_values(["event_date", "symbol"]).reset_index(drop=True)
    info = _write("fhps", out)
    info.update({
        "rows": int(len(out)),
        "symbols": int(out["symbol"].nunique()) if len(out) else 0,
        "start": str(pd.to_datetime(out["event_date"]).min().date()) if len(out) else None,
        "end": str(pd.to_datetime(out["event_date"]).max().date()) if len(out) else None,
        "seconds": round(time.time() - t_all, 1),
        "source": "akshare stock_fhps_em",
        "pit": "available_date = 预案公告日",
        "log": log,
    })
    _update_meta("fhps", {k: v for k, v in info.items() if k != "log"})
    return info


# ---------------------------------------------------------------------------
# 读取（默认 holdout 锁）
# ---------------------------------------------------------------------------


def load(name: str, *, allow_holdout: bool = False, start: str | None = None) -> pd.DataFrame:  # noqa: D401
    """读事件表；默认裁到 2024-12-31，``allow_holdout=True`` 才返回 2025 起的数据。"""
    if name not in FILES:
        raise KeyError(f"未知事件表：{name}（可选 {TABLES}）")
    frame = pd.read_parquet(FILES[name])
    if allow_holdout and HOLDOUT_FILES[name].is_file():
        extra = pd.read_parquet(HOLDOUT_FILES[name])
        frame = pd.concat([frame, extra], ignore_index=True).drop_duplicates()
    for column in ("event_date", "available_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if not allow_holdout:
        keep = frame["available_date"].isna() | (frame["available_date"] <= CUTOFF)
        frame = frame[keep]
        if "event_date" in frame:
            frame = frame[frame["event_date"] <= CUTOFF]
    if start:
        frame = frame[frame["event_date"] >= pd.Timestamp(start)]
    return frame.reset_index(drop=True)


def summary() -> pd.DataFrame:
    rows = []
    for name in TABLES:
        path = FILES[name]
        if not path.is_file():
            rows.append({"event": name, "rows": 0, "exists": False})
            continue
        frame = pd.read_parquet(path, columns=["event_date", "available_date", "symbol"])
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
        frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
        sample = load(name)
        rows.append({
            "event": name,
            "exists": True,
            "rows": int(len(frame)),
            "symbols": int(frame["symbol"].nunique()),
            "start": str(frame["event_date"].min().date()) if frame["event_date"].notna().any() else None,
            "end": str(frame["event_date"].max().date()) if frame["event_date"].notna().any() else None,
            "pit_missing": int(frame["available_date"].isna().sum()),
            "sample_rows": int(len(sample)),
            "bytes": int(path.stat().st_size),
        })
    return pd.DataFrame(rows)
