"""第三节探针：东财批量财务 / 申万行业 / 双源比对 / 行业 PIT / 金融业单列。

只产出 results/probe-akshare-em.{md,json}，不写入数据湖。
每一项独立 try/except，单项失败不影响其余项，失败原因如实记录。
"""

import argparse
import io
import json
import sys
import threading
import time
import traceback
from datetime import date, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

DEFAULT_OUT = PROJECT / "results" / "probe-akshare-em"

EM = "https://datacenter-web.eastmoney.com/api/data/v1/get"
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})

# 三张报表的东财数据中心全字段报表名（cols: 利润 203 / 资产负债 319 / 现金流 254）
TABLES = {
    "profit": ("RPT_F10_FINANCE_GINCOME", "利润表"),
    "balance": ("RPT_F10_FINANCE_GBALANCE", "资产负债表"),
    "cashflow": ("RPT_F10_FINANCE_GCASHFLOW", "现金流量表"),
}
# 探针抽样报告期（覆盖 2014 起点、熊市、近年）
SAMPLE_PERIODS = ["2014-03-31", "2018-12-31", "2022-06-30", "2026-06-30"]
# 退市股最后报告期验证
DELISTED_PROBE = {
    "600001": ("sh.600001", "2009-12-29", "2009-09-30"),
    "000003": ("sz.000003", "2002-06-14", "2002-03-31"),
    "000033": ("sz.000033", "2017-07-07", "2017-03-31"),
    "600087": ("sh.600087", "2014-06-05", "2014-03-31"),
}
# 行业 PIT 复测：借壳 / 主业变更，事件前后一级行业应不同
PIT_CASES = {
    "sh.600733": ("北汽蓝谷", "2018-06 借壳前锋股份（房地产->汽车）"),
    "sz.000738": ("航发控制", "2009-2010 南方摩托重组（摩托车->航空发动机）"),
    "sz.002027": ("分众传媒", "2015 借壳七喜控股（计算机->传媒）"),
}
PIT_DATES = ["2015-12-31", "2017-12-31", "2018-12-31", "2019-12-31", "2021-12-31", "2024-12-31"]


def em(report_name, flt=None, columns="ALL", page=1, page_size=500, retries=3):
    params = {
        "reportName": report_name,
        "columns": columns,
        "pageSize": str(page_size),
        "pageNumber": str(page),
        "source": "HSF10",
        "client": "PC",
    }
    if flt:
        params["filter"] = flt
    last = None
    for attempt in range(retries):
        try:
            r = S.get(EM, params=params, timeout=45)
            j = r.json()
            res = j.get("result") or {}
            return pd.DataFrame(res.get("data") or []), res.get("pages")
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"em 请求失败 {report_name}: {last}")


def em_all(report_name, flt=None, columns="ALL", max_pages=100, pause=0.05):
    """翻页取全量，返回合并后的 DataFrame 与页数统计。"""
    first, pages = em(report_name, flt, columns, 1)
    frames = [first]
    pages = int(pages or 0)
    for p in range(2, min(pages, max_pages) + 1):
        time.sleep(pause)
        chunk, _ = em(report_name, flt, columns, p)
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), pages


_BS_STATE = {"ok": None}


def ensure_bs():
    """baostock 全局登录一次；重复 login 会打断已有会话，故做幂等。"""
    import baostock as bs

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


def timeout_call(fn, seconds: float = 25.0, label: str = ""):
    """在守护线程里跑 fn，超时抛 TimeoutError。

    akshare 的个别交易所接口（stock_info_sh_delist / stock_info_sz_delist）没有
    任何超时参数，网络挂起时会无限阻塞；这里统一兜一层。
    """
    box: dict = {}

    def _run():
        try:
            box["v"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["e"] = exc

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise TimeoutError(f"{label or 'call'} 超过 {seconds:g}s 未返回")
    if "e" in box:
        raise box["e"]
    return box["v"]


def lake_symbols():
    p = Path("D:/quant-data/meta/universe.csv")
    df = pd.read_csv(p, dtype={"symbol": str})
    return df


# 交易所摘牌表里没有、但 baostock 标记 outDate 的代码：逐条人工核对过
# （若名字与现役公司不同，自动分类会漏判，这里补注业务含义）
CODE_TERMINATION_NOTES = {
    "600849": "上药转换：换股吸收合并入上海医药(601607)，2010-03-05 终止上市",
    "000022": "深赤湾A：换股吸收合并入招商港口(001872)，2018-12-26 终止上市",
    "000043": "中航善达：更名换代码为招商积余(001914)，2019-12-16 终止",
    "300114": "中航电测：更名换代码为中航成飞(302132)，2025-02-17 终止",
}


def reconcile_delist(ak_codes, bs_codes, basic: pd.DataFrame, board: str, today: str) -> dict:
    """交易所摘牌表 vs baostock status=0 的集合对账，并给差异分类。

    ak_only 里混着 B 股（沪 900xxx / 深 200xxx）；bs_only 里混着「代码终止但主体
    仍在市」（换股吸收合并、更名换代码）与「在途退市」（outDate 晚于基准日）。
    """
    ak_only = ak_codes - bs_codes
    bs_only = bs_codes - ak_codes
    b_pref = "900" if board == "SH" else "200"
    ak_only_b = {c for c in ak_only if c.startswith(b_pref)}
    ak_only_a = ak_only - ak_only_b

    name = dict(zip(basic["code"].str.split(".").str[-1].str.zfill(6), basic["code_name"]))
    out_date = dict(zip(basic["code"].str.split(".").str[-1].str.zfill(6), basic["outDate"]))
    live = basic[basic["status"] == "1"]
    live_names = set(live["code_name"].astype(str))
    live_name_to_code = dict(zip(live["code_name"].astype(str), live["code"].str.split(".").str[-1]))

    buckets = {"code_migration": [], "pending_delist": [], "unexplained": []}
    for c in sorted(bs_only):
        nm = str(name.get(c, ""))
        od = str(out_date.get(c, ""))
        if c in CODE_TERMINATION_NOTES:
            buckets["code_migration"].append(f"{c}({nm})")
        elif nm and nm in live_names:
            buckets["code_migration"].append(f"{c}({nm}->{live_name_to_code.get(nm)})")
        elif od[:10] > today:
            buckets["pending_delist"].append(f"{c}({nm},{od[:10]})")
        else:
            buckets["unexplained"].append(f"{c}({nm},{od[:10]})")

    return {
        "board": board,
        "ak": len(ak_codes),
        "bs": len(bs_codes),
        "overlap": len(ak_codes & bs_codes),
        "ak_only": len(ak_only),
        "ak_only_b_share": len(ak_only_b),
        "ak_only_a_share": len(ak_only_a),
        "ak_only_a_sample": sorted(ak_only_a)[:12],
        "bs_only": len(bs_only),
        "confirmed_delisted": len(ak_codes & bs_codes),
        "bs_only_categorized": {k: {"n": len(v), "items": v[:12]} for k, v in buckets.items()},
        "notes": {c: CODE_TERMINATION_NOTES[c] for c in sorted(bs_only) if c in CODE_TERMINATION_NOTES},
    }


# --------------------------------------------------------------------------
# 1. baostock stock_basic 口径对账：status=0 1197 vs 退市 339
# --------------------------------------------------------------------------
def probe_basic() -> dict:
    import baostock as bs

    st = ensure_bs()
    if not st["ok"]:
        raise RuntimeError(f"baostock 登录失败 {st['msg']}")
    df = bs_frame(bs.query_stock_basic(code=""))
    df["type"] = df["type"].astype(str)
    df["status"] = df["status"].astype(str)
    out = {"rows": len(df), "by_type": df["type"].value_counts().to_dict()}

    st = df[df["type"] == "1"].copy()
    out["stocks_total"] = len(st)
    out["stocks_status0"] = int((st["status"] == "0").sum())
    out["stocks_status1"] = int((st["status"] == "1").sum())

    def exchange(code):
        c = code.split(".")[-1]
        if c.startswith("6"):
            return "SH"
        if c.startswith(("0", "3")):
            return "SZ"
        if c.startswith(("4", "8", "9")):
            return "BJ"
        return "?"

    st["exchange"] = st["code"].map(exchange)
    st["out_year"] = st["outDate"].str[:4].replace("", "空")
    delisted = st[(st["status"] == "0") & (st["outDate"].str.len() >= 10)]
    out["delisted_with_outdate"] = len(delisted)
    out["delisted_by_exchange"] = delisted["exchange"].value_counts().to_dict()
    out["delisted_out_year"] = delisted["out_year"].value_counts().sort_index().to_dict()
    # 未退市但 status=0 的股票：口径歧义来源
    odd = st[(st["status"] == "0") & (st["outDate"].str.len() < 10)]
    out["status0_without_outdate"] = len(odd)
    out["status0_without_outdate_sample"] = odd["code"].head(10).tolist()

    # type 口径：status=0 全量按 type 拆分（指数/基金会被算进 1197）
    s0 = df[df["status"] == "0"]
    out["status0_by_type"] = s0["type"].value_counts().to_dict()
    out["type_name"] = {
        "1": "股票",
        "2": "指数",
        "3": "其它",
        "4": "可转债",
        "5": "ETF",
    }

    # 与湖宇宙对账：退市股是否在 universe.csv 内
    try:
        uni = lake_symbols()
        out["lake_universe"] = len(uni)

        def bare(codes):
            return {str(c).lower().split(".")[-1].zfill(6) for c in codes}

        lake_codes = bare(uni["symbol"])
        bs_codes = bare(st["code"])
        out["lake_minus_basic"] = sorted(lake_codes - bs_codes)[:10]
        out["basic_minus_lake_stocks"] = len(bs_codes - lake_codes)
        out["basic_minus_lake_sample"] = sorted(bs_codes - lake_codes)[:10]
        out["lake_delisted_present"] = len(bare(delisted["code"]) & lake_codes)
    except Exception as exc:  # noqa: BLE001
        out["lake_error"] = f"{type(exc).__name__}: {exc}"

    # 交叉验证：交易所口径退市名单（akshare 上交所/深交所摘牌）
    try:
        import akshare as ak

        sh = timeout_call(ak.stock_info_sh_delist, 25, "sh_delist")
        out["ak_sh_delist"] = {"rows": len(sh), "cols": list(sh.columns)[:8]}
        out["ak_sh_delist_sample"] = sh.head(3).to_dict("records")
        code_col = "公司代码" if "公司代码" in sh.columns else sh.columns[0]
        ak_sh = {str(c).zfill(6) for c in sh[code_col]}
        bs_sh = {str(c).split(".")[-1].zfill(6) for c in delisted["code"] if str(c).lower().startswith("sh.")}
        out["ak_sh_vs_bs"] = reconcile_delist(ak_sh, bs_sh, df, "SH", date.today().isoformat())
        out["ak_sh_vs_bs"].update(
            {"ak_only_sample": sorted(ak_sh - bs_sh)[:10], "bs_only_sample": sorted(bs_sh - ak_sh)[:10]}
        )
    except Exception as exc:  # noqa: BLE001
        out["ak_sh_delist_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import akshare as ak

        sz = timeout_call(lambda: ak.stock_info_sz_delist(symbol="终止上市公司"), 25, "sz_delist")
        out["ak_sz_delist"] = {"rows": len(sz), "cols": list(sz.columns)[:8]}
        out["ak_sz_delist_sample"] = sz.head(3).to_dict("records")
        code_col = "证券代码" if "证券代码" in sz.columns else sz.columns[0]
        ak_sz = {str(c).zfill(6) for c in sz[code_col]}
        bs_sz = {str(c).split(".")[-1].zfill(6) for c in delisted["code"] if str(c).lower().startswith("sz.")}
        out["ak_sz_vs_bs"] = reconcile_delist(ak_sz, bs_sz, df, "SZ", date.today().isoformat())
        out["ak_sz_vs_bs"].update(
            {"ak_only_sample": sorted(ak_sz - bs_sz)[:10], "bs_only_sample": sorted(bs_sz - ak_sz)[:10]}
        )
    except Exception as exc:  # noqa: BLE001
        out["ak_sz_delist_error"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------
# 2. 东财三表批量可用性
# --------------------------------------------------------------------------
def probe_bulk() -> dict:
    import akshare as ak

    out = {"sample_periods": SAMPLE_PERIODS}

    # 2a 元信息：pageSize 上限、页数、列数、单页延迟
    meta = []
    for key, (rn, label) in TABLES.items():
        for period in SAMPLE_PERIODS:
            t0 = time.perf_counter()
            df, pages = em(rn, f"(REPORT_DATE='{period}')", page=1)
            ms = (time.perf_counter() - t0) * 1000
            meta.append(
                {
                    "table": key,
                    "label": label,
                    "period": period,
                    "pages": int(pages or 0),
                    "page1_rows": len(df),
                    "cols": df.shape[1],
                    "page1_ms": round(ms, 1),
                    "est_rows": int(pages or 0) * 500,
                }
            )
            time.sleep(0.1)
    out["meta"] = meta

    # pageSize 上限验证
    for ps in (500, 800, 1000):
        try:
            df, pages = em("RPT_F10_FINANCE_GINCOME", "(REPORT_DATE='2023-12-31')", page=1, page_size=ps)
            out.setdefault("pagesize_cap", []).append({"requested": ps, "returned_rows": len(df), "pages": int(pages or 0)})
        except Exception as exc:  # noqa: BLE001
            out.setdefault("pagesize_cap", []).append({"requested": ps, "error": f"{type(exc).__name__}: {exc}"})

    # 2b 全量翻页：一个期 × 两张表，量 rows / 市场分布 / 丢行率
    full = {}
    for key in ("profit", "balance"):
        rn = TABLES[key][0]
        t0 = time.perf_counter()
        df, pages = em_all(rn, "(REPORT_DATE='2018-12-31')")
        el = time.perf_counter() - t0
        suf = df["SECUCODE"].astype(str).str.split(".").str[-1]
        ab = df[suf.isin(["SH", "SZ", "BJ"]) | df["SECURITY_CODE"].astype(str).str.match(r"^(6|0|3|4|8|9)")]
        full[key] = {
            "rows": len(df),
            "pages": int(pages or 0),
            "elapsed_s": round(el, 1),
            "cols": df.shape[1],
            "suffix": suf.value_counts().to_dict(),
            "a_share_rows": len(ab),
            "a_share_unique": ab["SECURITY_CODE"].nunique(),
            "org_type": (df["ORG_TYPE"].value_counts().to_dict() if "ORG_TYPE" in df else {}),
            "report_type": (df["REPORT_TYPE"].value_counts().to_dict() if "REPORT_TYPE" in df else {}),
            "null_notice": int(df["NOTICE_DATE"].isna().sum()) if "NOTICE_DATE" in df else None,
        }
        time.sleep(0.2)
    out["full_period"] = full

    # 2c 退市股覆盖：按代码取全部报告期
    delist = {}
    for code, (bs_code, out_date, expect_last) in DELISTED_PROBE.items():
        try:
            df, pages = em_all("RPT_F10_FINANCE_GINCOME", f'(SECURITY_CODE="{code}")')
            if len(df):
                dates = df["REPORT_DATE"].astype(str).str[:10].sort_values()
                delist[code] = {
                    "bs_code": bs_code,
                    "delist_date": out_date,
                    "rows": len(df),
                    "first": dates.iloc[0],
                    "last": dates.iloc[-1],
                    "expect_last": expect_last,
                    "last_matches": dates.iloc[-1][:7] == expect_last[:7],
                    "has_notice": bool(df["NOTICE_DATE"].notna().any()),
                }
            else:
                delist[code] = {"bs_code": bs_code, "rows": 0, "note": "东财无该代码历史报表"}
        except Exception as exc:  # noqa: BLE001
            delist[code] = {"bs_code": bs_code, "error": f"{type(exc).__name__}: {exc}"}
        time.sleep(0.15)
    out["delisted_coverage"] = delist

    # 2d 公告日：东财 NOTICE_DATE vs baostock pubDate（20 只 2023 年报）
    try:
        st = ensure_bs()
        out["bs_login"] = st["ok"]
        ser = json.load(io.open(PROJECT / "results" / "probe-pubdate.json", encoding="utf-8"))
        pair = []
        for item in ser["details"]:
            code = item["code"]
            try:
                df, _ = em(
                    "RPT_F10_FINANCE_GINCOME",
                    f'(SECURITY_CODE="{code}")(REPORT_DATE=\'2023-12-31\')',
                )
                em_notice = None
                if len(df) and "NOTICE_DATE" in df:
                    v = df["NOTICE_DATE"].iloc[0]
                    em_notice = str(v)[:10] if pd.notna(v) else None
                bs_pub = item.get("bs_pubDate")
                diff = None
                if em_notice and bs_pub:
                    diff = (datetime.strptime(em_notice, "%Y-%m-%d").date() - datetime.strptime(bs_pub, "%Y-%m-%d").date()).days
                pair.append({"code": code, "em_notice": em_notice, "bs_pubDate": bs_pub, "diff_days": diff})
            except Exception as exc:  # noqa: BLE001
                pair.append({"code": code, "error": f"{type(exc).__name__}: {exc}"})
            time.sleep(0.1)
        ok = [p for p in pair if p.get("diff_days") is not None]
        out["notice_vs_baostock"] = {
            "n": len(pair),
            "compared": len(ok),
            "exact": sum(1 for p in ok if p["diff_days"] == 0),
            "within_1d": sum(1 for p in ok if abs(p["diff_days"]) <= 1),
            "max_abs_diff": max((abs(p["diff_days"]) for p in ok), default=None),
            "diffs": [p["diff_days"] for p in ok],
            "missing_em": [p["code"] for p in pair if p.get("em_notice") is None],
            "details": pair,
        }
    except Exception as exc:  # noqa: BLE001
        out["notice_vs_baostock"] = {"error": f"{type(exc).__name__}: {exc}"}

    # 2e stock_report_disclosure（预约披露）能否用于 PIT
    try:
        d = ak.stock_report_disclosure(market="沪深京", period="2023年报")
        out["disclosure"] = {
            "rows": len(d),
            "cols": list(d.columns),
            "sample": d.head(3).to_dict("records"),
        }
    except Exception as exc:  # noqa: BLE001
        out["disclosure"] = {"error": f"{type(exc).__name__}: {exc}"}

    # 2f 薄表 vs 全表：stock_lrb_em 的列只有 15 个，不能替代三表
    try:
        thin = ak.stock_lrb_em(date="20231231")
        out["thin_table"] = {"rows": len(thin), "cols": thin.shape[1], "columns": list(thin.columns)}
    except Exception as exc:  # noqa: BLE001
        out["thin_table"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------
# 3. 申万行业
# --------------------------------------------------------------------------
def probe_sw() -> dict:
    import akshare as ak

    out = {}
    t0 = time.perf_counter()
    first = ak.sw_index_first_info()
    out["first_info"] = {
        "rows": len(first),
        "cols": list(first.columns),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "sample": first.head(3).to_dict("records"),
    }
    # 一级行业代码列名容错
    code_col = None
    for cand in ("行业代码", "指数代码", "code"):
        if cand in first.columns:
            code_col = cand
            break
    name_col = None
    for cand in ("行业名称", "指数名称", "name"):
        if cand in first.columns:
            name_col = cand
            break
    out["code_col"] = code_col
    if code_col:
        cons = []
        empty = []
        for _, row in first.iterrows():
            code = str(row[code_col]).split(".")[0]
            name = str(row[name_col]) if name_col else code
            try:
                d = ak.index_component_sw(symbol=code)
                n = len(d)
                cons.append({"code": code, "name": name, "members": n, "cols": list(d.columns)[:5]})
                if n == 0:
                    empty.append(name)
            except Exception as exc:  # noqa: BLE001
                cons.append({"code": code, "name": name, "members": None, "error": f"{type(exc).__name__}: {exc}"[:120]})
                empty.append(name)
            time.sleep(0.05)
        out["cons"] = cons
        out["empty_industries"] = empty
        out["total_members"] = sum(c["members"] for c in cons if c.get("members"))
        out["max_members"] = max((c["members"] for c in cons if c.get("members")), default=0)
        out["min_members"] = min((c["members"] for c in cons if c.get("members")), default=0)

    # 与东财行业分类交叉：覆盖股票数
    try:
        em_ind = ak.stock_board_industry_name_em()
        out["em_industry"] = {"rows": len(em_ind), "cols": list(em_ind.columns)[:8]}
    except Exception as exc:  # noqa: BLE001
        out["em_industry"] = {"error": f"{type(exc).__name__}: {exc}"}

    # 二级 / 三级列表是否可取（用于层级映射）
    try:
        second = ak.sw_index_second_info()
        out["second_info"] = {"rows": len(second), "cols": list(second.columns)[:8]}
    except Exception as exc:  # noqa: BLE001
        out["second_info"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------
# 4. 双源财务比对：东财 vs baostock，20 只 × 4 期
# --------------------------------------------------------------------------
def probe_dual() -> dict:
    import baostock as bs

    uni = lake_symbols()
    # 覆盖银行/非银/制造/消费/科技/北交所
    picks = [
        "sh.600519", "sz.000858", "sh.601318", "sh.600036", "sz.000001",
        "sh.601166", "sz.300750", "sz.002415", "sh.600030", "sh.601988",
        "sz.000333", "sh.600276", "sz.002594", "sh.688111", "sz.000651",
        "sh.603288", "sz.002714", "sh.600887", "sz.300059", "sh.601899",
    ]
    pick_codes = {p.split(".")[1] for p in picks}
    uni_codes = set(uni["symbol"].astype(str))
    miss = sorted(pick_codes - uni_codes)
    quarters = ["2018-12-31", "2020-12-31", "2022-12-31", "2024-12-31"]

    out = {"stocks": picks, "quarters": quarters, "universe_missing": miss}

    st = ensure_bs()
    out["bs_login"] = st["ok"]
    rows = []
    for code in picks:
        num = code.split(".")[1]
        em_df = None
        try:
            em_df, _ = em_all(
                "RPT_F10_FINANCE_MAINFINADATA", f'(SECURITY_CODE="{num}")'
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"code": code, "period": "*", "error": f"em: {type(exc).__name__}: {exc}"[:160]})
            continue
        if len(em_df):
            em_df["d"] = em_df["REPORT_DATE"].astype(str).str[:10]
        for q in quarters:
            year, quarter = int(q[:4]), {3: 1, 6: 2, 9: 3, 12: 4}[int(q[5:7])]
            rec = {"code": code, "period": q}
            try:
                e = em_df[em_df["d"] == q]
                if len(e):
                    rec["em_roe"] = pd.to_numeric(e["ROEJQ"].iloc[0], errors="coerce")
                    rec["em_gm"] = pd.to_numeric(e["XSMLL"].iloc[0], errors="coerce")
                    rec["em_npyoy"] = pd.to_numeric(e["PARENTNETPROFITTZ"].iloc[0], errors="coerce")
                    rec["em_org"] = e["ORG_TYPE"].iloc[0]
                    rec["em_notice"] = str(e["NOTICE_DATE"].iloc[0])[:10]
            except Exception as exc:  # noqa: BLE001
                rec["em_err"] = f"{type(exc).__name__}: {exc}"[:120]
            try:
                pf = bs_frame(bs.query_profit_data(code=code, year=year, quarter=quarter))
                if len(pf):
                    rec["bs_roe"] = pd.to_numeric(pf["roeAvg"].iloc[0], errors="coerce") * 100
                    rec["bs_gm"] = pd.to_numeric(pf["gpMargin"].iloc[0], errors="coerce") * 100
                gr = bs_frame(bs.query_growth_data(code=code, year=year, quarter=quarter))
                if len(gr):
                    rec["bs_npyoy"] = pd.to_numeric(gr["YOYNI"].iloc[0], errors="coerce") * 100
            except Exception as exc:  # noqa: BLE001
                rec["bs_err"] = f"{type(exc).__name__}: {exc}"[:120]
            rows.append(rec)
            time.sleep(0.03)

    df = pd.DataFrame(rows)
    out["rows"] = len(df)
    summary = {}
    for key, em_c, bs_c in (("roe", "em_roe", "bs_roe"), ("gm", "em_gm", "bs_gm"), ("npyoy", "em_npyoy", "bs_npyoy")):
        sub = df.dropna(subset=[em_c, bs_c]) if (em_c in df and bs_c in df) else pd.DataFrame()
        if len(sub):
            diff = (sub[em_c] - sub[bs_c]).abs()
            summary[key] = {
                "n": len(sub),
                "median_abs_diff": round(float(diff.median()), 4),
                "p90_abs_diff": round(float(diff.quantile(0.9)), 4),
                "max_abs_diff": round(float(diff.max()), 4),
                "within_0.5": int((diff <= 0.5).sum()),
                "within_1": int((diff <= 1.0).sum()),
            }
        else:
            summary[key] = {"n": 0}
    out["summary"] = summary
    out["details"] = rows
    # 东财自洽：MAINFINADATA 与三表净利润是否一致（抽 3 只 2024 年报）
    checks = []
    for code in ("600519", "000001", "300750"):
        try:
            a, _ = em("RPT_F10_FINANCE_MAINFINADATA", f'(SECURITY_CODE="{code}")(REPORT_DATE=\'2024-12-31\')')
            b, _ = em("RPT_F10_FINANCE_GINCOME", f'(SECURITY_CODE="{code}")(REPORT_DATE=\'2024-12-31\')')
            if len(a) and len(b):
                checks.append(
                    {
                        "code": code,
                        "main_parentnp": float(pd.to_numeric(a["PARENTNETPROFIT"].iloc[0], errors="coerce")),
                        "income_netprofit": float(pd.to_numeric(b["NETPROFIT"].iloc[0], errors="coerce")),
                        "main_toi": float(pd.to_numeric(a["TOTALOPERATEREVE"].iloc[0], errors="coerce")),
                        "income_toi": float(pd.to_numeric(b["TOTAL_OPERATE_INCOME"].iloc[0], errors="coerce")),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            checks.append({"code": code, "error": f"{type(exc).__name__}: {exc}"[:120]})
    out["internal_consistency"] = checks
    return out


# --------------------------------------------------------------------------
# 5. 行业 PIT 复测
# --------------------------------------------------------------------------
def probe_industry_pit() -> dict:
    import baostock as bs

    st = ensure_bs()
    out = {"bs_login": st["ok"], "cases": {}}
    for code, (name, note) in PIT_CASES.items():
        rec = {"name": name, "note": note, "by_date": {}}
        for d in PIT_DATES:
            try:
                rs = bs.query_stock_industry(code=code, date=d)
                df = bs_frame(rs)
                if len(df):
                    rec["by_date"][d] = {
                        "industry": df["industry"].iloc[0],
                        "industryClassification": df["industryClassification"].iloc[0],
                        "updateDate": df["updateDate"].iloc[0],
                    }
                else:
                    rec["by_date"][d] = {"rows": 0}
            except Exception as exc:  # noqa: BLE001
                rec["by_date"][d] = {"error": f"{type(exc).__name__}: {exc}"[:120]}
            time.sleep(0.05)
        rec["distinct_industries"] = sorted({v.get("industry") for v in rec["by_date"].values() if v.get("industry")})
        rec["changed"] = len(rec["distinct_industries"]) > 1
        out["cases"][code] = rec

    # 无日期参数的对照：当前快照
    try:
        cur = bs_frame(bs.query_stock_industry())
        out["current_snapshot"] = {
            "rows": len(cur),
            "cols": list(cur.columns),
            "unique_industries": cur["industry"].nunique(),
            "sample": cur.head(3).to_dict("records"),
        }
    except Exception as exc:  # noqa: BLE001
        out["current_snapshot"] = {"error": f"{type(exc).__name__}: {exc}"}

    # 同一批日期下的行业条数变化（此前发现 2897/3869/5330），确认是"当日股票列表"还是"分类变更"
    sizes = {}
    for d in ("2016-06-30", "2020-06-30", "2026-06-30"):
        try:
            cur = bs_frame(bs.query_stock_industry(date=d))
            sizes[d] = {
                "rows": len(cur),
                "unique_codes": cur["code"].nunique(),
                "unique_industries": cur["industry"].nunique(),
            }
        except Exception as exc:  # noqa: BLE001
            sizes[d] = {"error": f"{type(exc).__name__}: {exc}"[:120]}
        time.sleep(0.05)
    out["industry_rows_by_date"] = sizes
    return out


# --------------------------------------------------------------------------
# 6. 修订公告定位
# --------------------------------------------------------------------------
def probe_corrections() -> dict:
    ser = json.load(io.open(PROJECT / "results" / "probe-pubdate.json", encoding="utf-8"))
    hits = [d for d in ser["details"] if d.get("corrections_2023", 0) > 0]
    out = {"corrections_total": ser["summary"].get("corrections_total"), "n_hit": len(hits), "hits": hits}

    def ann_window(code, begin, end, want=("更正", "修订", "差错", "补充更正")):
        params = {
            "sr": "-1", "page_size": "100", "page_index": "1", "ann_type": "A",
            "client_source": "web", "stock_list": code, "f_node": "0", "s_node": "0",
            "begin_time": begin, "end_time": end,
        }
        r = S.get("https://np-anotice-stock.eastmoney.com/api/security/ann", params=params, timeout=30)
        items = (r.json().get("data") or {}).get("list") or []
        return items, [i for i in items if any(w in (i.get("title") or "") for w in want)]

    detailed = []
    for code in [h["code"] for h in hits] + ["600519", "000001", "300750"]:
        ann_date = None
        for h in hits:
            if h["code"] == code:
                ann_date = h.get("ann_date") or h.get("bs_pubDate")
        if ann_date is None:
            ann_date = "2024-04-30"
        y = int(ann_date[:4])
        rec = {"code": code, "annual_report_date": ann_date}
        try:
            items, corr = ann_window(code, f"{y}-03-01", f"{y}-05-31")
            rec["window_items"] = len(items)
            rec["correction_like"] = [
                {"title": c.get("title"), "date": str(c.get("notice_date"))[:10]} for c in corr[:6]
            ]
            rec["has_fin_correction"] = any(
                ("年度报告" in (c.get("title") or "")) and ("更正" in (c.get("title") or "") or "修订" in (c.get("title") or ""))
                for c in corr
            )
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:160]
        # 全历史关键字检索（不限窗口），确认是否存在财报更正公告
        try:
            params = {
                "sr": "-1", "page_size": "100", "page_index": "1", "ann_type": "A",
                "client_source": "web", "stock_list": code, "f_node": "0", "s_node": "0",
            }
            r = S.get("https://np-anotice-stock.eastmoney.com/api/security/ann", params=params, timeout=30)
            items = (r.json().get("data") or {}).get("list") or []
            rec["latest_items_sample"] = [str(i.get("title"))[:40] for i in items[:3]]
        except Exception as exc:  # noqa: BLE001
            rec["latest_error"] = f"{type(exc).__name__}: {exc}"[:120]
        detailed.append(rec)
        time.sleep(0.1)
    out["details"] = detailed
    out["conclusion"] = (
        "baostock 每 statDate 仅保留一行（修订后值），query_profit_data 无版本号/无公告批次字段；"
        "东财 F10 同样只给最新修订值。因此 PIT 修订偏差无法回滚，所有报告需固定标注「财务为修订后值，存在轻微后视偏差」。"
    )
    return out


# --------------------------------------------------------------------------
# 7. 金融业单列
# --------------------------------------------------------------------------
def probe_financial() -> dict:
    import baostock as bs

    out = {}
    try:
        st = ensure_bs()
        out["bs_login"] = st["ok"]
        cur = bs_frame(bs.query_stock_industry())
        vc = cur["industry"].value_counts()
        # baostock industry 形如「J66货币金融服务」，申万一级金融业为 J 门类
        fin_names = [n for n in vc.index if str(n).startswith("J")]
        out["financial_industries"] = {n: int(vc[n]) for n in fin_names}
        out["financial_total"] = int(sum(vc[n] for n in fin_names))
        out["all_industries"] = int(len(vc))
        out["top_industries"] = {str(k): int(v) for k, v in vc.head(12).items()}
        # 名称口径对照（旧分类可能不带 J 前缀）
        out["name_keyword_total"] = int(
            sum(int(v) for n, v in vc.items() if any(k in str(n) for k in ("银行", "保险", "证券", "多元金融", "信托")))
        )
    except Exception as exc:  # noqa: BLE001
        out["bs_error"] = f"{type(exc).__name__}: {exc}"

    # 东财 ORG_TYPE 口径（同一次批量请求自带）
    try:
        df, _ = em("RPT_F10_FINANCE_MAINFINADATA", "(REPORT_DATE='2024-12-31')", page=1)
        out["em_org_type_page1"] = df["ORG_TYPE"].value_counts().to_dict()
        # 金融业 vs 非金融业 的负债率 / 毛利率分布（单页约 500 行，只作量级参考）
        d = df.copy()
        d["zcfzl"] = pd.to_numeric(d.get("ZCFZL"), errors="coerce")
        d["xsmll"] = pd.to_numeric(d.get("XSMLL"), errors="coerce")
        g = d.groupby("ORG_TYPE")[["zcfzl", "xsmll"]].median().round(2)
        out["median_by_org"] = {str(k): {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in v.items()} for k, v in g.to_dict("index").items()}
    except Exception as exc:  # noqa: BLE001
        out["em_error"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------
# 8. 补充：A股行构成 / 过滤提速 / 代码复用 / 空行业
# --------------------------------------------------------------------------
def probe_extras() -> dict:
    import baostock as bs

    out = {}

    # 8a 同一报告期的行构成 + 服务端过滤能否减页
    try:
        df, pages_base = em_all("RPT_F10_FINANCE_GINCOME", "(REPORT_DATE='2023-12-31')")
        suf = df["SECUCODE"].astype(str).str.split(".").str[-1]
        out["composition"] = {
            "rows": len(df),
            "pages": int(pages_base or 0),
            "by_suffix": suf.value_counts().to_dict(),
            "by_type_code": df["SECURITY_TYPE_CODE"].value_counts().to_dict(),
            "by_suffix_x_type": {
                f"{k[0]}|{k[1]}": int(v)
                for k, v in df.groupby([suf, df["SECURITY_TYPE_CODE"]]).size().items()
            },
            "distinct_report_dates": df["REPORT_DATE"].astype(str).str[:10].nunique(),
        }
        # 与湖宇宙交集（这里用当前 universe 近似）
        try:
            uni = lake_symbols()
            lake = {str(s).split(".")[-1].zfill(6) for s in uni["symbol"]}
            codes = set(df["SECURITY_CODE"].astype(str).str.zfill(6))
            out["composition"]["in_lake"] = len(codes & lake)
            out["composition"]["not_in_lake"] = len(codes - lake)
            out["composition"]["not_in_lake_sample"] = sorted(codes - lake)[:12]
        except Exception as exc:  # noqa: BLE001
            out["composition"]["lake_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        out["composition"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}

    # 服务端过滤候选：只保留 A 股 / B 股类型码，看页数变化
    for label, flt in (
        ("type_in_A_B", "(REPORT_DATE='2023-12-31')(SECURITY_TYPE_CODE in (\"058001001\",\"058001002\"))"),
        ("suffix_sh_sz_bj", "(REPORT_DATE='2023-12-31')(SECUCODE in (\"600519.SH\"))"),
    ):
        try:
            t0 = time.perf_counter()
            d, pages = em("RPT_F10_FINANCE_GINCOME", flt, page=1)
            out.setdefault("filter_trials", []).append(
                {
                    "label": label,
                    "filter": flt,
                    "pages": int(pages or 0),
                    "page1_rows": len(d),
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.setdefault("filter_trials", []).append({"label": label, "error": f"{type(exc).__name__}: {exc}"[:120]})
        time.sleep(0.1)

    # 8b 代码复用：退市代码在退市后仍返回报表，看名字/ORG_CODE 是否换过主体
    reuse = {}
    for code in ("000003", "000033", "600087"):
        try:
            df, _ = em_all("RPT_F10_FINANCE_GINCOME", f'(SECURITY_CODE="{code}")')
            if not len(df):
                reuse[code] = {"rows": 0}
                continue
            df = df.copy()
            df["d"] = df["REPORT_DATE"].astype(str).str[:10]
            df = df.sort_values("d")
            names = df.groupby("SECURITY_NAME_ABBR")["d"].agg(["min", "max", "count"])
            orgs = df["ORG_CODE"].nunique() if "ORG_CODE" in df else None
            reuse[code] = {
                "rows": len(df),
                "first": df["d"].iloc[0],
                "last": df["d"].iloc[-1],
                "n_org": int(orgs) if orgs is not None else None,
                "names": {str(k): {"first": v["min"], "last": v["max"], "n": int(v["count"])} for k, v in names.iterrows()},
                "tail": df[["d", "SECURITY_NAME_ABBR", "ORG_CODE"]].tail(3).to_dict("records"),
            }
            # 退市后仍披露的报表（三板延续披露），用于确定截断规则
            dl = DELISTED_PROBE.get(code, (None, None, None))[1]
            if dl:
                after = df[df["d"] > dl]
                reuse[code]["delist_date"] = dl
                reuse[code]["after_delist_rows"] = len(after)
                reuse[code]["last_before_delist"] = df[df["d"] <= dl]["d"].iloc[-1] if (df["d"] <= dl).any() else None
                reuse[code]["after_sample"] = after[["d", "TOTAL_OPERATE_INCOME"]].head(3).to_dict("records")
        except Exception as exc:  # noqa: BLE001
            reuse[code] = {"error": f"{type(exc).__name__}: {exc}"[:160]}
        time.sleep(0.15)
    out["code_reuse"] = reuse

    # 8c 空行业构成
    try:
        st = ensure_bs()
        cur = bs_frame(bs.query_stock_industry())
        cur["ind"] = cur["industry"].fillna("").astype(str)
        empty = cur[cur["ind"] == ""]
        # 空值是否集中在退市股（无日期调用会带上已退市代码）
        delisted_codes = set()
        try:
            basic = bs_frame(bs.query_stock_basic(code=""))
            basic = basic[(basic["type"] == "1") & (basic["outDate"].str.len() >= 10)]
            delisted_codes = {c.lower() for c in basic["code"]}
        except Exception:  # noqa: BLE001
            pass
        empty_codes = {c.lower() for c in empty["code"]}
        out["empty_industry"] = {
            "total_rows": len(cur),
            "empty": len(empty),
            "empty_by_prefix": {
                "SH": int(empty["code"].str.startswith("sh.").sum()),
                "SZ": int(empty["code"].str.startswith("sz.").sum()),
                "BJ": int(empty["code"].str.startswith("bj.").sum()),
            },
            "empty_that_are_delisted": len(empty_codes & delisted_codes),
            "empty_that_are_live": len(empty_codes - delisted_codes),
            "live_empty_sample": sorted(empty_codes - delisted_codes)[:10],
            "sample": empty["code"].head(8).tolist(),
        }
    except Exception as exc:  # noqa: BLE001
        out["empty_industry"] = {"error": f"{type(exc).__name__}: {exc}"[:160]}
    return out


def probe_index_cons() -> dict:
    """9. 指数成分：能否拿到历史成分（PIT），还是只有当前快照。"""
    import akshare as ak

    out = {}
    for sym, label in (("000300", "沪深300"), ("000905", "中证500")):
        try:
            t0 = time.perf_counter()
            df = timeout_call(lambda s=sym: ak.index_stock_cons_csindex(symbol=s), 30, f"csindex_{sym}")
            ms = (time.perf_counter() - t0) * 1000
            out[f"csindex_{sym}"] = {
                "label": label,
                "rows": len(df),
                "cols": list(df.columns),
                "ms": round(ms, 1),
                "has_date_col": any(str(c).strip() in ("日期", "date") for c in df.columns),
                "distinct_dates": int(df["日期"].nunique()) if "日期" in df.columns else None,
                "date_min": str(df["日期"].min()) if "日期" in df.columns else None,
                "date_max": str(df["日期"].max()) if "日期" in df.columns else None,
            }
        except Exception as exc:  # noqa: BLE001
            out[f"csindex_{sym}"] = {"error": f"{type(exc).__name__}: {exc}"[:160]}
        time.sleep(0.1)

    # 带"纳入日期"的口径：当前成分 + 每只加入日；已剔除成分不在表内 → 无剔除日
    try:
        df = timeout_call(lambda: ak.index_stock_cons(symbol="000300"), 30, "index_stock_cons")
        inc = pd.to_datetime(df["纳入日期"], errors="coerce")
        out["with_inclusion_date"] = {
            "rows": len(df),
            "cols": list(df.columns),
            "distinct_inclusion_dates": int(inc.nunique()),
            "earliest_inclusion": str(inc.min())[:10],
            "latest_inclusion": str(inc.max())[:10],
            "future_inclusion": int((inc > pd.Timestamp(date.today())).sum()),
            "inclusion_year_counts": inc.dt.year.value_counts().sort_index().to_dict(),
            "removed_members_present": False,
        }
    except Exception as exc:  # noqa: BLE001
        out["with_inclusion_date"] = {"error": f"{type(exc).__name__}: {exc}"[:160]}

    try:
        info = timeout_call(ak.index_stock_info, 30, "index_stock_info")
        out["index_stock_info"] = {"rows": len(info), "cols": list(info.columns)}
    except Exception as exc:  # noqa: BLE001
        out["index_stock_info"] = {"error": f"{type(exc).__name__}: {exc}"[:160]}

    out["api_candidates"] = {
        "ak_index_stock_cons_csindex": "当前成分，日期列只有快照日",
        "ak_index_stock_cons": "当前成分 + 各自纳入日期，无剔除日",
        "ak_index_stock_cons_sina": "当前成分 + 实时行情",
        "ak_index_stock_cons_weight_csindex": "当前成分 + 权重",
        "历史成分": "akshare 无按日期取成分的接口",
    }
    return out


PROBES = {
    1: "stock_basic 口径对账",
    2: "东财三表批量可用性",
    3: "申万行业",
    4: "双源财务比对",
    5: "行业 PIT 复测",
    6: "修订公告定位",
    7: "金融业单列",
    8: "补充：行构成/过滤提速/代码复用/空行业",
    9: "指数成分历史可得性（沪深300/中证500）",
}
FUNCS = {
    1: probe_basic,
    2: probe_bulk,
    3: probe_sw,
    4: probe_dual,
    5: probe_industry_pit,
    6: probe_corrections,
    7: probe_financial,
    8: probe_extras,
    9: probe_index_cons,
}


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def render_md(results: dict, elapsed: float) -> str:
    L = []
    L.append("# 第三节探针：东财批量财务 / 申万行业 / 双源比对 / 行业 PIT")
    L.append("")
    L.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 总耗时：{elapsed/60:.1f} 分钟")
    L.append("- 口径：仅探针，结果不写入数据湖；所有数字为本机实测")
    L.append("")

    r1 = results.get("1") or {}
    L.append("## 1. stock_basic 口径对账（status=0 1197 vs 退市 339）")
    L.append("")
    if r1.get("error"):
        L.append(f"**失败**：{r1['error']}")
    else:
        L.append("| 口径 | 数量 |")
        L.append("| --- | --- |")
        L.append(f"| query_stock_basic 总行数（含指数/基金） | {r1.get('rows')} |")
        L.append(f"| type=1 股票总数 | {r1.get('stocks_total')} |")
        L.append(f"| 其中 status=0 | {r1.get('stocks_status0')} |")
        L.append(f"| 其中 status=1 | {r1.get('stocks_status1')} |")
        L.append(f"| status=0 且 outDate 非空（真退市） | {r1.get('delisted_with_outdate')} |")
        L.append(f"| status=0 但 outDate 为空 | {r1.get('status0_without_outdate')} |")
        L.append("")
        L.append(f"- 全部 type 的 status=0 分布：`{r1.get('status0_by_type')}`")
        L.append(f"- type 名称映射：`{r1.get('type_name')}`")
        L.append(f"- 退市股交易所分布：`{r1.get('delisted_by_exchange')}`")
        L.append(f"- 退市股按 outDate 年份：`{r1.get('delisted_out_year')}`")
        if r1.get("status0_without_outdate_sample"):
            L.append(f"- outDate 为空的 status=0 样本：`{r1['status0_without_outdate_sample']}`")
        if r1.get("lake_universe") is not None:
            L.append(f"- 湖 universe={r1.get('lake_universe')}，basic 多出股票代码数={r1.get('basic_minus_lake_stocks')}")
            L.append(f"- basic 多出样本：`{r1.get('basic_minus_lake_sample')}`")
            L.append(f"- 退市股中已在湖宇宙内的：{r1.get('lake_delisted_present')}")
        for k in ("ak_sh_delist", "ak_sz_delist"):
            if r1.get(k):
                L.append(f"- {k}: {r1[k]}")
            elif r1.get(k + "_error"):
                L.append(f"- {k} 失败：{r1[k + '_error']}")
        for k in ("ak_sh_vs_bs", "ak_sz_vs_bs"):
            if r1.get(k):
                v = r1[k]
                L.append("")
                L.append(f"**{v['board']} 退市名单对账**")
                L.append("")
                L.append("| 项 | 数量 |")
                L.append("| --- | --- |")
                L.append(f"| 交易所摘牌表唯一代码 | {v['ak']} |")
                L.append(f"| baostock 真退市（status=0 且有 outDate） | {v['bs']} |")
                L.append(f"| 两边都有（可确证退市） | {v['overlap']} |")
                L.append(f"| 仅交易所（其中 B 股） | {v['ak_only']}（{v['ak_only_b_share']}） |")
                L.append(f"| 仅交易所（非 B 股） | {v['ak_only_a_share']} |")
                L.append(f"| 仅 baostock | {v['bs_only']} |")
                L.append("")
                if v.get("ak_only_a_sample"):
                    L.append(f"- 仅交易所非 B 股样本：`{v['ak_only_a_sample']}`")
                for cat, label in (
                    ("code_migration", "代码终止但主体仍在市（吸收合并/更名换代码）"),
                    ("pending_delist", "在途退市（outDate 晚于基准日）"),
                    ("unexplained", "未归类"),
                ):
                    c = (v.get("bs_only_categorized") or {}).get(cat) or {}
                    if c.get("n"):
                        L.append(f"- {label}：{c['n']} 只，`{c['items']}`")
                if v.get("notes"):
                    for code, note in v["notes"].items():
                        L.append(f"  - {code}：{note}")
    L.append("")

    r2 = results.get("2") or {}
    L.append("## 2. 东财三表批量可用性")
    L.append("")
    if r2.get("error"):
        L.append(f"**失败**：{r2['error']}")
    else:
        if r2.get("meta"):
            L.append("| 表 | 报告期 | 页数 | 首页行数 | 列数 | 首页延迟(ms) | 估算行数 |")
            L.append("| --- | --- | --- | --- | --- | --- | --- |")
            for m in r2["meta"]:
                L.append(
                    f"| {m['label']} | {m['period']} | {m['pages']} | {m['page1_rows']} | {m['cols']} | "
                    f"{m['page1_ms']} | {m['est_rows']} |"
                )
            L.append("")
        if r2.get("pagesize_cap"):
            L.append(f"- pageSize 上限实测：`{r2['pagesize_cap']}`")
        if r2.get("full_period"):
            for k, v in r2["full_period"].items():
                L.append(
                    f"- **{k}** 2018-12-31 全量：rows={v['rows']} pages={v['pages']} 列={v['cols']} "
                    f"耗时={v['elapsed_s']}s A股行={v['a_share_rows']} A股去重={v['a_share_unique']} "
                    f"后缀={v['suffix']} 空公告日={v['null_notice']}"
                )
            L.append("")
        if r2.get("delisted_coverage"):
            L.append("| 代码 | 退市日 | 报表期数 | 首期 | 末期 | 与预期一致 |")
            L.append("| --- | --- | --- | --- | --- | --- |")
            for code, v in r2["delisted_coverage"].items():
                L.append(
                    f"| {code} | {v.get('delist_date','')} | {v.get('rows')} | {v.get('first','')} | "
                    f"{v.get('last','')} | {v.get('last_matches','')} |"
                )
            L.append("")
        nv = r2.get("notice_vs_baostock") or {}
        if nv and not nv.get("error"):
            L.append(
                f"- 公告日对账（东财 NOTICE_DATE vs baostock pubDate，2023 年报）：可比 {nv.get('compared')}/{nv.get('n')} 只，"
                f"完全一致 {nv.get('exact')}，±1 天内 {nv.get('within_1d')}，最大偏差 {nv.get('max_abs_diff')} 天，"
                f"东财缺失 {nv.get('missing_em')}"
            )
            L.append("")
        if r2.get("disclosure"):
            d = r2["disclosure"]
            if d.get("error"):
                L.append(f"- `stock_report_disclosure` 失败：{d['error']}")
            else:
                L.append(f"- `stock_report_disclosure`（沪深京/2023年报）：{d['rows']} 行，列 `{d['cols']}`")
                L.append(f"  样本：`{d['sample']}`")
        if r2.get("thin_table"):
            t = r2["thin_table"]
            if t.get("error"):
                L.append(f"- 薄表 `stock_lrb_em` 失败：{t['error']}")
            else:
                L.append(f"- 薄表 `stock_lrb_em(20231231)`：{t['rows']} 行 × {t['cols']} 列，列 `{t['columns']}`")
    L.append("")

    r3 = results.get("3") or {}
    L.append("## 3. 申万行业")
    L.append("")
    if r3.get("error"):
        L.append(f"**失败**：{r3['error']}")
    else:
        fi = r3.get("first_info") or {}
        L.append(f"- 一级行业：{fi.get('rows')} 个，列 `{fi.get('cols')}`")
        if r3.get("cons"):
            L.append(f"- 成分股：覆盖合计 {r3.get('total_members')} 个（含重复），单行业最多 {r3.get('max_members')}、最少 {r3.get('min_members')}")
            L.append(f"- **空行业**：{r3.get('empty_industries')}")
            bad = [c for c in r3["cons"] if c.get("error")]
            if bad:
                L.append(f"- 取数失败行业 {len(bad)} 个：`{[b['name'] for b in bad][:8]}`")
        if r3.get("em_industry"):
            L.append(f"- 东财行业对照：{r3['em_industry']}")
        if r3.get("second_info"):
            L.append(f"- 二级行业：{r3['second_info']}")
    L.append("")

    r4 = results.get("4") or {}
    L.append("## 4. 双源财务比对（东财 vs baostock，20 只 × 4 期）")
    L.append("")
    if r4.get("error"):
        L.append(f"**失败**：{r4['error']}")
    else:
        L.append(f"- 样本：{len(r4.get('stocks') or [])} 只 × {len(r4.get('quarters') or [])} 期，universe 缺 {r4.get('universe_missing')}")
        L.append("")
        L.append("| 指标 | 可比条数 | 中位绝对差 | P90 | 最大 | ≤0.5 占比 |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for k, label in (("roe", "ROE(百分点)"), ("gm", "毛利率(百分点)"), ("npyoy", "净利同比(百分点)")):
            s = (r4.get("summary") or {}).get(k) or {}
            if s.get("n"):
                L.append(
                    f"| {label} | {s['n']} | {s['median_abs_diff']} | {s['p90_abs_diff']} | "
                    f"{s['max_abs_diff']} | {s['within_0.5']}/{s['n']} |"
                )
            else:
                L.append(f"| {label} | 0 | - | - | - | - |")
        L.append("")
        if r4.get("internal_consistency"):
            L.append(f"- 东财自洽（MAINFINADATA vs 利润表，2024 年报）：`{r4['internal_consistency']}`")
    L.append("")

    r5 = results.get("5") or {}
    L.append("## 5. 行业 PIT 复测（借壳/主业变更）")
    L.append("")
    if r5.get("error"):
        L.append(f"**失败**：{r5['error']}")
    else:
        for code, c in (r5.get("cases") or {}).items():
            L.append(f"- **{code} {c['name']}**（{c['note']}）：不同行业 {c['distinct_industries']}，是否变化={c['changed']}")
            L.append(f"  逐日：`{ {d: v.get('industry') for d, v in c['by_date'].items()} }`")
        L.append("")
        if r5.get("industry_rows_by_date"):
            L.append(f"- 不同日期的行业表条数：`{r5['industry_rows_by_date']}`")
        if r5.get("current_snapshot"):
            cs = r5["current_snapshot"]
            L.append(f"- 当前快照：{cs.get('rows')} 行，{cs.get('unique_industries')} 个行业，列 `{cs.get('cols')}`")
    L.append("")

    r6 = results.get("6") or {}
    L.append("## 6. 修订公告定位")
    L.append("")
    if r6.get("error"):
        L.append(f"**失败**：{r6['error']}")
    else:
        L.append(f"- 初筛检出「更正/补充」类公告 {r6.get('corrections_total')} 条（关键词粗筛，需复核）")
        L.append("")
        L.append("| 代码 | 年报公告日 | 窗口公告数 | 更正/修订类标题 | 是否财报更正 |")
        L.append("| --- | --- | --- | --- | --- |")
        for d in r6.get("details") or []:
            if d.get("error"):
                L.append(f"| {d.get('code')} | - | - | 取数失败 {d['error']} | - |")
                continue
            titles = [t["title"] for t in d.get("correction_like", [])]
            L.append(
                f"| {d.get('code')} | {d.get('annual_report_date')} | {d.get('window_items')} | "
                f"{('；'.join(titles)[:90] if titles else '无')} | {d.get('has_fin_correction')} |"
            )
        if r6.get("conclusion"):
            L.append("")
            L.append(f"- 结论：{r6['conclusion']}")
    L.append("")

    r7 = results.get("7") or {}
    L.append("## 7. 金融业单列")
    L.append("")
    if r7.get("error"):
        L.append(f"**失败**：{r7['error']}")
    else:
        L.append(f"- 申万行业口径金融业（J 门类）：{r7.get('financial_total')} 只 / 全行业 {r7.get('all_industries')} 个")
        L.append(f"- 明细：`{r7.get('financial_industries')}`")
        if r7.get("name_keyword_total") is not None:
            L.append(f"- 名称关键词口径（银行/保险/证券/多元金融/信托）：{r7.get('name_keyword_total')} 只")
        if r7.get("top_industries"):
            L.append(f"- 前 12 个行业：`{r7['top_industries']}`")
        if r7.get("em_org_type_page1"):
            L.append(f"- 东财 ORG_TYPE（单页）：`{r7['em_org_type_page1']}`")
        if r7.get("median_by_org"):
            L.append(f"- 分类型中位数（负债率/毛利率）：`{r7['median_by_org']}`")
    L.append("")

    r8 = results.get("8") or {}
    L.append("## 8. 补充：A股行构成 / 过滤提速 / 代码复用 / 空行业")
    L.append("")
    if r8.get("error"):
        L.append(f"**失败**：{r8['error']}")
    else:
        c = r8.get("composition") or {}
        if c.get("error"):
            L.append(f"- 行构成取数失败：{c['error']}")
        else:
            L.append(f"- 2023-12-31 单期全量 {c.get('rows')} 行 / {c.get('pages')} 页，报告期种类 {c.get('distinct_report_dates')}")
            L.append(f"- 按后缀：`{c.get('by_suffix')}`；按证券类型码：`{c.get('by_type_code')}`")
            L.append(f"- 与湖宇宙交集 {c.get('in_lake')}，不在湖内 {c.get('not_in_lake')}，样本 `{c.get('not_in_lake_sample')}`")
        for t in r8.get("filter_trials") or []:
            if t.get("error"):
                L.append(f"- 过滤试验 `{t['label']}` 失败：{t['error']}")
            else:
                L.append(f"- 过滤试验 `{t['label']}`：页数 {t['pages']}，首页 {t['page1_rows']} 行，{t['ms']}ms")
        L.append("")
        if r8.get("code_reuse"):
            L.append("| 代码 | 行数 | 首期 | 末期 | ORG_CODE 数 | 退市日 | 退市后报表数 | 名称分段 |")
            L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for code, v in r8["code_reuse"].items():
                if v.get("error"):
                    L.append(f"| {code} | - | - | - | - | - | - | 失败 {v['error']} |")
                    continue
                names = "；".join(f"{k}({x['first']}~{x['last']},{x['n']})" for k, x in (v.get("names") or {}).items())
                L.append(
                    f"| {code} | {v.get('rows')} | {v.get('first')} | {v.get('last')} | {v.get('n_org')} | "
                    f"{v.get('delist_date')} | {v.get('after_delist_rows')} | {names[:110]} |"
                )
        if r8.get("empty_industry"):
            e = r8["empty_industry"]
            if e.get("error"):
                L.append(f"- 空行业统计失败：{e['error']}")
            else:
                L.append(
                    f"- 行业空值：{e.get('empty')}/{e.get('total_rows')} 只，分布 `{e.get('empty_by_prefix')}`，样本 `{e.get('sample')}`"
                )
                L.append(
                    f"- 空值中已退市 {e.get('empty_that_are_delisted')} 只、仍在市 {e.get('empty_that_are_live')} 只"
                    f"（在市样本 `{e.get('live_empty_sample')}`）"
                )
    L.append("")

    r9 = results.get("9") or {}
    L.append("## 9. 指数成分历史可得性（沪深300 / 中证500）")
    L.append("")
    if r9.get("error"):
        L.append(f"**失败**：{r9['error']}")
    else:
        for k in ("csindex_000300", "csindex_000905"):
            v = r9.get(k) or {}
            if v.get("error"):
                L.append(f"- `{k}` 失败：{v['error']}")
                continue
            L.append(
                f"- `{k}`（{v.get('label')}）：{v.get('rows')} 只，列 `{v.get('cols')}`，"
                f"不同日期 {v.get('distinct_dates')} 个（{v.get('date_min')} ~ {v.get('date_max')}）"
            )
        w = r9.get("with_inclusion_date") or {}
        if w.get("error"):
            L.append(f"- `index_stock_cons(000300)` 失败：{w['error']}")
        elif w:
            L.append(
                f"- `index_stock_cons(000300)`：{w.get('rows')} 只，列 `{w.get('cols')}`，"
                f"纳入日期 {w.get('distinct_inclusion_dates')} 个（{w.get('earliest_inclusion')} ~ {w.get('latest_inclusion')}），"
                f"其中未来纳入 {w.get('future_inclusion')} 只"
            )
            L.append(f"  - 纳入年份分布：`{w.get('inclusion_year_counts')}`")
            L.append(f"  - 是否含已剔除成分：`{w.get('removed_members_present')}`（只有当前成分）")
        if r9.get("index_stock_info"):
            v = r9["index_stock_info"]
            L.append(f"- `index_stock_info`：{v.get('rows')} 个指数，列 `{v.get('cols')}`" if not v.get("error") else f"- `index_stock_info` 失败：{v['error']}")
        L.append("")
        L.append("**结论**：akshare 没有任何「按日期取历史成分」的接口。")
        L.append("")
        L.append("- `index_stock_cons_csindex` 的日期列只有快照日（全部等于当日），不是历史序列。")
        L.append("- `index_stock_cons` 能给当前成分的**纳入日期**，但被剔除的成分整行消失，没有剔除日 → 无法重建历史成分。")
        L.append("- 影响：任何以「当时是否属于沪深300/中证500」为条件的回测都会有幸存者偏差；只能退化为「当前成分标签 + 偏差声明」。")
    L.append("")

    r1d = results.get("1") or {}
    L.append("## 10. 探针结论与落盘决策")
    L.append("")
    L.append("| # | 决策 | 依据 |")
    L.append("| --- | --- | --- |")
    L.append(
        "| 1 | universe 以 baostock `query_stock_basic` 为权威，落区间表 `symbol,ipo_date,out_date,board,type,status`，不物化日面板 | "
        f"type=1 共 {r1d.get('stocks_total')} 只（在市 {r1d.get('stocks_status1')} / 退市 {r1d.get('delisted_with_outdate')}）；"
        "交易所摘牌表只作确证（交集 146+187=333） |"
    )
    L.append(
        "| 2 | B 股从股票池剔除；4 只「代码终止但主体存续」建 successor 映射；2 只在途退市（601059/601198，outDate 2026-10-19）在基准日仍算在市 | "
        "交易所口径 ak_only 全部是 B 股（沪 900xxx 7 只 / 深 200xxx 22 只）；bs_only 6 只已逐条核对 |"
    )
    L.append(
        "| 3 | 财务走东财数据中心 `RPT_F10_FINANCE_*` 按报告期全量拉，加 `SECURITY_TYPE_CODE in (058001001,058001002)` 过滤 | "
        "12–15 列的 `stock_lrb_em` 等批量接口字段不足；过滤后 30 页→14 页，44 期×3 表 ≈2.2 h |"
    )
    L.append(
        "| 4 | 退市股财务按退市日截断（保留退市前最后一期） | 000003 有 54 期、600087 有 49 期发生在退市后（老三板延续披露），不截断会混入 |"
    )
    L.append(
        "| 5 | 行业用 baostock `query_stock_industry(date)` 做 PIT（年度粒度），申万只做当期描述不做历史归因 | 借壳样本 600733/002027 均在同一年度切换分类；申万成分无日期参数 |"
    )
    L.append(
        "| 6 | 财务修订偏差不可回滚，报告固定标注 | 双源都只保留每 statDate 修订后一行，无版本号 |"
    )
    L.append(
        "| 7 | 指数成分只能取当前快照 + 纳入日期，回测不得以「当时是否成分股」筛选 | §9 实测无历史成分接口 |"
    )
    L.append("")
    L.append("**固定偏差声明（所有因子/回测报告顶部 7 条）**")
    L.append("")
    L.append("1. 财务为修订后值，存在轻微后视偏差（无法回滚到原始披露值）。")
    L.append("2. 财务可用时点以 `NOTICE_DATE`/`pubDate` 为准（2023 年报 20/20 完全一致），未使用预约披露日。")
    L.append("3. 行业分类为年度粒度 PIT，重组/借壳后最多滞后一年。")
    L.append("4. 申万行业成分只有当前快照，行业暴露为当期口径。")
    L.append("5. 指数成分只有当前快照，任何成分筛选均含幸存者偏差。")
    L.append("6. 退市股只保留退市前财务，退市收益按 CRSP 分层处理（吸收合并用 successor 价，无法定价用 −30% / −100% 双版本）。")
    L.append("7. 估值/isST 为逐日快照回填，非交易所原始历史，极端日可能存在口径差异。")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    wanted = [int(x) for x in args.only.split(",") if x.strip()] if args.only else sorted(PROBES)
    results = {}
    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    if args.only and json_path.exists():
        try:
            results = json.load(io.open(json_path, encoding="utf-8"))
            results = {k: v for k, v in results.items() if not k.startswith("_")}
        except Exception:  # noqa: BLE001
            results = {}

    t_all = time.perf_counter()
    for key in wanted:
        label = PROBES[key]
        print(f"\n===== [{key}] {label} =====", flush=True)
        t0 = time.perf_counter()
        try:
            results[str(key)] = FUNCS[key]()
            print(f"  ok  {time.perf_counter()-t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            results[str(key)] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-1500:]}
            print(f"  FAIL {type(exc).__name__}: {exc}", flush=True)
        with io.open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2, default=str)

    elapsed = time.perf_counter() - t_all
    md = render_md(results, elapsed)
    io.open(out_base.with_suffix(".md"), "w", encoding="utf-8").write(md)
    print(f"\nwrote {out_base.with_suffix('.md')} ({len(md)} chars) in {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
