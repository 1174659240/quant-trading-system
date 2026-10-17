"""参考数据层：universe 区间表、行业 PIT 快照。

设计约束（由 ``results/probe-akshare-em.md`` 的实测结论确定）：

1. universe 以 baostock ``query_stock_basic`` 为权威：type=1 共 5563 只，在市 5224、
   退市 339；交易所摘牌表只作交叉确证（交集 333 只）。
2. B 股（沪 900xxx / 深 200xxx）保留行但 ``is_b_share=1``，默认股票池剔除。
3. 4 只「代码终止但主体存续」建 successor 映射；outDate 晚于基准日的 2 只
   （601059 / 601198，2026-10-19）标 ``is_pending_delist=1``，基准日仍算在市。
4. 行业用 baostock ``query_stock_industry(date)``，一次调用返回该日全市场，
   粒度年度（实测 600733 / 002027 均在同一年度切换分类）→ 按年末交易日各存一版。
5. 落盘位置 ``D:/quant-data/ref/``，parquet + csv 双写，附 manifest（asOf/行数/来源）。
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REF_DIR = Path("D:/quant-data/ref")
CALENDAR = Path("D:/quant-data/meta/calendar.csv")
UNIVERSE_CSV = Path("D:/quant-data/meta/universe.csv")

# 逐条人工核对过的「代码终止但主体存续」（交易所摘牌表里没有这些记录）
SUCCESSOR = {
    "600849": ("601607", "上药转换换股吸收合并入上海医药"),
    "000022": ("001872", "深赤湾A换股吸收合并入招商港口"),
    "000043": ("001914", "中航善达更名换代码为招商积余"),
    "300114": ("302132", "中航电测更名换代码为中航成飞"),
}

_BS_STATE: dict = {"ok": None}


def ensure_bs(*, force: bool = False):
    """幂等登录 baostock；重复 login 会打断已有会话，force=True 时才重新登录。"""
    import baostock as bs

    if force:
        try:
            bs.logout()
        except Exception:  # noqa: BLE001 - 退出失败不影响重连
            pass
        _BS_STATE["ok"] = None
    if _BS_STATE["ok"] is None:
        lg = bs.login()
        _BS_STATE["ok"] = lg.error_code == "0"
        _BS_STATE["msg"] = f"{lg.error_code}: {lg.error_msg}"
    return _BS_STATE


def bs_frame(rs) -> pd.DataFrame:
    if rs.error_code != "0":
        raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def board_of(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith("900") or c.startswith("200"):
        return "B股"
    if c.startswith(("688", "689")):
        return "科创板"
    if c.startswith(("300", "301", "302")):
        return "创业板"
    if c.startswith(("43", "83", "87", "88", "92")):
        return "北交所"
    if c.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "主板"
    return "其他"


def trade_days() -> pd.DatetimeIndex:
    df = pd.read_csv(CALENDAR, dtype=str)
    return pd.DatetimeIndex(pd.to_datetime(df["date"]))


def year_end_trading_days(start_year: int = 2014) -> list[str]:
    """每个自然年的最后一个交易日，作为该年度行业快照的 as_of。"""
    days = trade_days()
    out = []
    for year in range(start_year, days.max().year + 1):
        part = days[days.year == year]
        if len(part):
            out.append(part[-1].strftime("%Y-%m-%d"))
    return out


def _atomic(frame: pd.DataFrame, base: Path) -> dict:
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    parquet_path = base.with_suffix(".parquet")
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    frame.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
    tmp_csv.replace(csv_path)
    written = {"csv": str(csv_path)}
    try:
        tmp_pq = parquet_path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp_pq, index=False)
        tmp_pq.replace(parquet_path)
        written["parquet"] = str(parquet_path)
    except Exception as exc:  # noqa: BLE001 - parquet 可选，缺 pyarrow 不阻塞
        written["parquet_error"] = f"{type(exc).__name__}: {exc}"
    return written


def build_universe(*, as_of: str | None = None) -> dict:
    """universe 区间表：每个代码一行，含上市日/退市日/板块/状态/继任代码。"""
    import baostock as bs

    st = ensure_bs()
    if not st["ok"]:
        raise RuntimeError(f"baostock 登录失败 {st['msg']}")
    as_of = as_of or date.today().isoformat()

    df = bs_frame(bs.query_stock_basic(code=""))
    df["type"] = df["type"].astype(str)
    df["status"] = df["status"].astype(str)
    df = df[df["type"] == "1"].copy()

    out = pd.DataFrame(
        {
            "symbol": df["code"].str.split(".").str[-1].str.zfill(6),
            "exchange": df["code"].str.split(".").str[0].str.upper(),
            "code_name": df["code_name"].astype(str),
            "ipo_date": df["ipoDate"].replace("", pd.NA),
            "out_date": df["outDate"].replace("", pd.NA),
            "status": df["status"].astype(int),
        }
    )
    out["board"] = out["symbol"].map(board_of)
    out["is_b_share"] = out["board"].eq("B股").astype(int)
    has_out = out["out_date"].fillna("").astype(str)
    # 「在途退市」：outDate 已公告但晚于基准日，基准日仍可交易（如 601059 / 601198 的 2026-10-19）
    out["is_pending_delist"] = ((out["status"] == 0) & (has_out > as_of)).astype(int)
    out["is_delisted"] = ((out["status"] == 0) & (has_out != "") & (has_out <= as_of)).astype(int)
    out["successor"] = out["symbol"].map(lambda s: SUCCESSOR.get(s, (None, None))[0])
    out["note"] = out["symbol"].map(lambda s: SUCCESSOR.get(s, (None, None))[1])
    # 统一成 object + ""，否则 None 会让列升成 float，落盘出现 1872.0
    for col in ("ipo_date", "out_date", "successor", "note"):
        out[col] = out[col].fillna("").astype(str)
    out["as_of"] = as_of

    # 与爬取用代码表对账：meta/universe.csv 只是「在市可抓」清单，退市股不在其中
    crawl = set()
    if UNIVERSE_CSV.exists():
        u = pd.read_csv(UNIVERSE_CSV, dtype=str)
        crawl = {str(x).zfill(6) for x in u["symbol"]}
    out["in_crawl_universe"] = out["symbol"].isin(crawl).astype(int)

    out = out.sort_values(["symbol"]).reset_index(drop=True)
    written = _atomic(out, REF_DIR / "universe")
    summary = {
        "as_of": as_of,
        "rows": int(len(out)),
        "live": int((out["status"] == 1).sum()),
        "delisted": int(out["is_delisted"].sum()),
        "pending_delist": int(out["is_pending_delist"].sum()),
        "b_share": int(out["is_b_share"].sum()),
        "with_successor": int(out["successor"].notna().sum()),
        "in_crawl_universe": int(out["in_crawl_universe"].sum()),
        "board_counts": out["board"].value_counts().to_dict(),
        "written": written,
    }
    return summary


def _industry_once(bs, day: str, min_rows: int) -> tuple[pd.DataFrame, int]:
    """单日行业快照。baostock 会在流中断时返回 error_code=0 但行数偏少，
    因此用「行数下限 + 代码唯一」判定完整性，不合格就重新登录重试。"""
    attempts = 0
    last = pd.DataFrame()
    for attempt in (1, 2, 3):
        attempts = attempt
        if attempt > 1:
            ensure_bs(force=True)
            time.sleep(2.0)
        try:
            one = bs_frame(bs.query_stock_industry(date=day))
        except Exception as exc:  # noqa: BLE001 - 记录后重试
            print(f"  [{day}] 第 {attempt} 次异常：{exc}", flush=True)
            continue
        last = one
        if not len(one):
            continue
        ok = len(one) >= min_rows and one["code"].is_unique
        if ok:
            return one, attempts
        print(
            f"  [{day}] 第 {attempt} 次不完整：{len(one)} 行（下限 {min_rows}），重试",
            flush=True,
        )
    return last, attempts


def build_industry(*, start_year: int = 2014, as_of: str | None = None) -> dict:
    """行业 PIT 快照：每个年末交易日一版，长表（symbol, as_of, industry, ...）。"""
    import socket

    import baostock as bs

    # 单日 5000 行实测约 85–100s；超时设太小会在流中途断开并静默截断
    socket.setdefaulttimeout(300.0)
    st = ensure_bs()
    if not st["ok"]:
        raise RuntimeError(f"baostock 登录失败 {st['msg']}")
    as_of = as_of or date.today().isoformat()
    dates = [d for d in year_end_trading_days(start_year) if d <= as_of]
    # as_of 与最近一版同年且相隔很近时不再追加，避免落一份内容完全相同的重复版本
    if dates and (date.fromisoformat(as_of) - date.fromisoformat(dates[-1])).days <= 30:
        dates[-1] = as_of
    elif not dates or dates[-1] != as_of:
        dates.append(as_of)

    frames = []
    elapsed = []
    floor = 2000  # 2014 年末实测 2666 行，之后单调增长
    for idx, d in enumerate(dates, 1):
        t0 = time.perf_counter()
        one, tries = _industry_once(bs, d, floor)
        if not len(one):
            elapsed.append({"as_of": d, "rows": 0, "attempts": tries})
            print(f"  行业 {idx}/{len(dates)} {d} → 0 行", flush=True)
            continue
        one = one.copy()
        one["symbol"] = one["code"].str.split(".").str[-1].str.zfill(6)
        one["exchange"] = one["code"].str.split(".").str[0].str.upper()
        one["as_of"] = d
        one["industry"] = one["industry"].fillna("").astype(str)
        one = one.rename(columns={"code_name": "code_name", "updateDate": "classify_date"})
        frames.append(one[["symbol", "exchange", "code_name", "as_of", "industry", "industryClassification", "classify_date"]])
        secs = round(time.perf_counter() - t0, 2)
        elapsed.append({"as_of": d, "rows": int(len(one)), "s": secs, "attempts": tries})
        floor = max(2000, int(len(one) * 0.98))  # 下一年不应明显少于本年
        print(f"  行业 {idx}/{len(dates)} {d} → {len(one)} 行 / {secs}s / {tries} 次", flush=True)
        time.sleep(0.05)

    if not frames:
        raise RuntimeError("行业快照为空")
    out = pd.concat(frames, ignore_index=True).sort_values(["as_of", "symbol"]).reset_index(drop=True)
    written = _atomic(out, REF_DIR / "industry")
    per_date = out.groupby("as_of").size().to_dict()
    return {
        "as_of": as_of,
        "rows": int(len(out)),
        "dates": len(per_date),
        "per_date": {str(k): int(v) for k, v in per_date.items()},
        "empty_industry_by_date": {
            str(k): int((g["industry"] == "").sum()) for k, g in out.groupby("as_of")
        },
        "live_covered": int(((out["as_of"] == dates[-1]) & (out["industry"] != "")).sum()),
        "written": written,
        "timings": elapsed,
    }


def write_manifest(payload: dict) -> Path:
    """合并写入：只跑 universe 或只跑 industry 时，保留另一节的历史记录。"""
    path = REF_DIR / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict = {}
    if path.exists():
        try:
            merged = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update({k: v for k, v in payload.items() if k != "manifest"})
    merged["generated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_universe() -> pd.DataFrame:
    return pd.read_csv(REF_DIR / "universe.csv", dtype={"symbol": str, "successor": str})


def load_industry() -> pd.DataFrame:
    return pd.read_csv(
        REF_DIR / "industry.csv", dtype={"symbol": str, "as_of": str, "classify_date": str}
    )


def industry_asof(frame: pd.DataFrame, day: str) -> pd.DataFrame:
    """取 <= day 的最近一版行业快照（年度 PIT 语义）。"""
    av = sorted(d for d in frame["as_of"].unique() if d <= day)
    if not av:
        raise KeyError(f"{day} 之前没有行业快照")
    return frame[frame["as_of"] == av[-1]]
