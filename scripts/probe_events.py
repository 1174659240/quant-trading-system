"""任务二第二步：事件因子可行性探测（只探测，不建库、不落数据）。

对 7 类事件逐个调用 akshare / 东财免费接口，记录：接口名、返回行数/列名、历史回溯起点
（日期列最小/最大值）、是否有「公告日」可用于 PIT、以及停牌/复权/覆盖的坑。

产出 `results/probe-events.{json,md}`，供用户确认后再决定是否抓取入库。

用法::

    python -m scripts.probe_events
    python -m scripts.probe_events --out results/probe-events
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import akshare as ak  # noqa: E402

from quant.fundamentals import page as em_page  # noqa: E402

DATE_HINTS = (
    "日期", "date", "DATE", "时间", "公告日", "变动日", "除权", "解禁", "REPORT", "NOTICE",
    "上榜", "预案", "登记", "派息", "变动", "截止",
)


def timeout_call(fn: Callable[[], Any], seconds: float, label: str) -> Any:
    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as error:  # noqa: BLE001
            box["error"] = error

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"{label} 超过 {seconds:.0f}s 未返回")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def summarise(frame: pd.DataFrame) -> dict:
    """行数、列名、日期列范围、是否含公告日。"""
    if isinstance(frame, tuple):  # fundamentals.page 返回 (DataFrame, pages)
        frame = frame[0]
    if frame is None or not isinstance(frame, pd.DataFrame):
        return {"rows": 0, "columns": [], "note": "返回不是 DataFrame"}
    out: dict[str, Any] = {"rows": int(len(frame)), "columns": [str(column) for column in frame.columns]}
    ranges = {}
    for column in frame.columns:
        name = str(column)
        if len(frame) == 0:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):  # 涨跌幅/数量这类数字列不当日期解析
            continue
        if any(hint.lower() in name.lower() for hint in DATE_HINTS):
            parsed = pd.to_datetime(frame[column], errors="coerce")
            valid = parsed.dropna()
            if len(valid):
                ranges[name] = [str(valid.min().date()), str(valid.max().date())]
    out["date_ranges"] = ranges
    out["has_notice_date"] = any(("公告" in key) or ("NOTICE" in key.upper()) for key in ranges)
    out["head"] = json.loads(frame.head(2).to_json(orient="records", force_ascii=False, date_format="iso"))
    return out


def _probe_incentive_reports() -> pd.DataFrame:
    """东财数据中心报表名存在性探测（激励类候选）。"""
    import requests

    from quant.fundamentals import EM_URL

    names = ["RPT_SHARE_INCENTIVE", "RPT_F10_EH_INCENTIVE", "RPT_INCENTIVE_PLAN", "RPT_ESOP_PLAN"]
    rows = []
    for name in names:
        try:
            resp = requests.get(
                EM_URL,
                params={
                    "reportName": name,
                    "columns": "ALL",
                    "pageNumber": 1,
                    "pageSize": 5,
                    "source": "WEB",
                    "client": "WEB",
                },
                timeout=20,
            )
            body = resp.json()
            result = body.get("result") or {}
            rows.append(
                {
                    "报表名": name,
                    "success": bool(body.get("success")),
                    "message": str(body.get("message"))[:60],
                    "count": result.get("count"),
                }
            )
        except Exception as error:  # noqa: BLE001
            rows.append({"报表名": name, "success": False, "message": f"{type(error).__name__}", "count": None})
    return pd.DataFrame(rows)


def _probe_st_flips() -> pd.DataFrame:
    """用本地日频 isST 矩阵统计戴帽/摘帽事件数（离线）。"""
    from quant.valuation import st_matrix

    flags = st_matrix()
    flags = flags.loc[:"2024-12-31"]
    rows = []
    for year, block in flags.groupby(flags.index.year):
        previous = block.shift(1).fillna(False).astype(bool)
        current = block.astype(bool)
        rows.append(
            {
                "年份": int(year),
                "交易天数": int(len(block)),
                "ST股日数": int(current.to_numpy().sum()),
                "戴帽次数": int((current & ~previous).to_numpy().sum()),
                "摘帽次数": int((~current & previous).to_numpy().sum()),
            }
        )
    return pd.DataFrame(rows)


PROBES: list[dict[str, Any]] = [
    {
        "group": "龙虎榜",
        "name": "stock_lhb_detail_em（区间明细）",
        "why": "上榜日 + 买卖席位，事件当日即公开",
        "call": lambda: ak.stock_lhb_detail_em(start_date="20240102", end_date="20240131"),
        "seconds": 60,
        "pit": "「上榜日」收盘后公布，最早可用 t+1 开盘；公告日字段=上榜日，无独立公告日",
        "risks": "只覆盖异动个股；区间过长会截断；ST/退市股也在内",
    },
    {
        "group": "龙虎榜",
        "name": "stock_lhb_ggtj_sina（个股上榜统计）",
        "why": "按股票聚合上榜次数",
        "call": lambda: ak.stock_lhb_ggtj_sina(symbol="5"),
        "seconds": 60,
        "pit": "聚合表无事件日，只能做「近 5 日」滚动特征",
        "risks": "新浪分档参数含义不清，回归测试需核对",
    },
    {
        "group": "回购",
        "name": "stock_repurchase_em（回购方案/进度）",
        "why": "回购公告日 + 金额 + 进度",
        "call": lambda: ak.stock_repurchase_em(),
        "seconds": 60,
        "pit": "含「最新公告日期」与「实施进度」，需用公告日而非方案日",
        "risks": "一条记录对应一次进度更新，去重口径要固定（同一方案多次公告）",
    },
    {
        "group": "大股东增持",
        "name": "stock_hold_management_detail_em（高管/股东增减持）",
        "why": "变动日 + 变动股数 + 变动后持股（全市场一次拉取）",
        "call": lambda: ak.stock_hold_management_detail_em(),
        "seconds": 240,
        "pit": "含「变动日期」与「公告日期」两列，PIT 用公告日期",
        "risks": "只覆盖董监高及关联方；大宗交易与竞价混在一起；接口全量较慢",
    },
    {
        "group": "大股东增持",
        "name": "stock_hold_management_detail_cninfo（巨潮，按增持/减持）",
        "why": "交叉校验东财口径（symbol 枚举『增持』『减持』，不是股票代码）",
        "call": lambda: ak.stock_hold_management_detail_cninfo(symbol="增持"),
        "seconds": 120,
        "pit": "巨潮明细含公告日",
        "risks": "参数是方向而非个股；返回条数大，需再按股票过滤",
    },
    {
        "group": "股权激励",
        "name": "stock_restricted_release_queue_em（限售解禁队列）",
        "why": "解禁日是硬事件，规模可算",
        "call": lambda: ak.stock_restricted_release_queue_em(symbol="688981"),
        "seconds": 90,
        "pit": "解禁日提前已知，无 PIT 问题",
        "risks": "这是「解禁」不是「激励授予」，只能替代使用；个股可能为空表",
    },
    {
        "group": "股权激励",
        "name": "东财报表名存在性（激励类 4 个候选）",
        "why": "确认免费接口有没有「激励计划」明细表",
        "call": _probe_incentive_reports,
        "seconds": 60,
        "pit": "若存在，看是否有 NOTICE_DATE",
        "risks": "实测 4 个候选报表名全部『报表配置不存在』→ 免费层无激励计划明细表",
    },
    {
        "group": "ST 摘帽",
        "name": "本地 isST 矩阵翻转统计（valuation 层）",
        "why": "用已落盘日频 isST 统计每年「戴帽/摘帽」次数，判断事件样本量",
        "call": _probe_st_flips,
        "seconds": 180,
        "pit": "isST 取自 baostock 日频字段，当日可见，可做 PIT",
        "risks": "只覆盖沪深（无北交所）；2014 年前退市股无估值文件",
    },
    {
        "group": "ST 摘帽",
        "name": "stock_zh_a_st_em（ST 板块当前名单）",
        "why": "确认接口可用；但只有当期快照",
        "call": lambda: ak.stock_zh_a_st_em(),
        "seconds": 60,
        "pit": "快照无历史，无法直接做 PIT 摘帽事件",
        "risks": "历史摘帽需靠「名称变更」或日线 isST 翻转推断",
    },
    {
        "group": "ST 摘帽",
        "name": "stock_info_change_name（名称变更历史）",
        "why": "ST 摘帽/戴帽在名称变更里留痕",
        "call": lambda: ak.stock_info_change_name(symbol="000001"),
        "seconds": 60,
        "pit": "含「变更日期」，可做 PIT（但名称变更≠摘帽，需结合当时是否 ST）",
        "risks": "新浪只保留部分历史；需逐股请求",
    },
    {
        "group": "高送转",
        "name": "stock_fhps_em（分红送配，按报告期）",
        "why": "送转比例 + 除权除息日 + 预案公告日",
        "call": lambda: ak.stock_fhps_em(date="20231231"),
        "seconds": 60,
        "pit": "含「预案公告日」「股权登记日」「除权除息日」，可做 PIT",
        "risks": "按报告期拉全市场，55 期；与已入库 dividends 表部分重叠",
    },
    {
        "group": "高送转",
        "name": "stock_fhps_detail_em（单股分红送配明细）",
        "why": "单股全历史",
        "call": lambda: ak.stock_fhps_detail_em(symbol="600519"),
        "seconds": 60,
        "pit": "含预案公告日",
        "risks": "需逐股请求",
    },
    {
        "group": "分红预案",
        "name": "stock_dividend_cninfo（巨潮分红）",
        "why": "预案 → 实施全流程",
        "call": lambda: ak.stock_dividend_cninfo(symbol="600519"),
        "seconds": 60,
        "pit": "含「预案公告日」「实施公告日」",
        "risks": "逐股请求；字段中文名需映射",
    },
    {
        "group": "分红预案",
        "name": "RPT_SHAREBONUS_DET（已入库的东财分红明细）",
        "why": "现有 dividends 层的数据源，复用即可",
        "call": lambda: em_page(
            "RPT_SHAREBONUS_DET",
            "(REPORT_DATE='2023-12-31')",
            "SECURITY_CODE,REPORT_DATE,PLAN_NOTICE_DATE,NOTICE_DATE,EQUITY_RECORD_DATE,EX_DIVIDEND_DATE,"
            "PRETAX_BONUS_RMB,BONUS_IT_RATIO,DIVIDENT_RATIO,ASSIGN_PROGRESS",
            1,
            size=500,
        ),
        "seconds": 60,
        "pit": "PLAN_NOTICE_DATE（预案公告）/NOTICE_DATE（实施公告）都有，已按 ex_date 做 PIT",
        "risks": "columns=ALL 会被服务端按 5 行/页截断，必须显式列名",
    },
]


VERDICTS = {
    "stock_lhb_detail_em（区间明细）": ("可用", "上榜日收盘后公布，t+1 开盘可交易；按区间逐次拉取，2014 年起可全量覆盖"),
    "stock_lhb_ggtj_sina（个股上榜统计）": ("仅辅助", "无事件日期，只能当『近 N 日上榜次数』特征，不能当事件"),
    "stock_repurchase_em（回购方案/进度）": ("可用", "最新公告日期 + 实施进度可做 PIT；同一方案多次公告需按方案去重"),
    "stock_hold_management_detail_em（高管/股东增减持）": ("可用", "全量 17.3 万条、起点 2004-11-29；日期=变动日（公告通常 T+2 内），偏乐观 1–2 日"),
    "stock_hold_management_detail_cninfo（巨潮，按增持/减持）": ("不可用于历史", "只返回近 1 年（2025-10 起），无历史回溯"),
    "stock_restricted_release_queue_em（限售解禁队列）": ("需逐股", "解禁日提前已知、无 PIT 问题；但只能逐股拉（5563 只）且个股可能空表"),
    "东财报表名存在性（激励类 4 个候选）": ("不可用", "4 个候选报表名全部『报表配置不存在』→ 免费层无股权激励明细表；只能以限售解禁替代"),
    "本地 isST 矩阵翻转统计（valuation 层）": ("可用", "已落盘日频 isST；2014–2024 累计戴帽 1,895 次、摘帽 630 次，样本充足"),
    "stock_zh_a_st_em（ST 板块当前名单）": ("不可用", "东财 push2 接口 RemoteDisconnected（与本机 push2his 现象一致），且只有当期快照"),
    "stock_info_change_name（名称变更历史）": ("仅辅助", "只给名称序列、无日期；摘帽判定须靠 isST 翻转"),
    "stock_fhps_em（分红送配，按报告期）": ("可用", "按报告期拉全市场，含预案公告日/股权登记日/除权除息日；55 期可覆盖 2013 起"),
    "stock_fhps_detail_em（单股分红送配明细）": ("需逐股", "单股全历史，含预案公告日，但空预案日较多"),
    "stock_dividend_cninfo（巨潮分红）": ("可用", "含『实施方案公告日期』与除权日，可做 PIT；逐股请求"),
    "RPT_SHAREBONUS_DET（已入库的东财分红明细）": ("已入库", "ref/dividends.parquet 已按 ex_date 做 PIT，直接复用"),
}


def render(results: list[dict], out: Path) -> str:
    lines = ["# 事件因子可行性探测（任务二·第二步）", ""]
    lines.append(f"- 生成时间：{pd.Timestamp.now(tz='Asia/Shanghai').isoformat(timespec='seconds')}")
    lines.append("- 只探测，未抓取、未入库；holdout（2025-01-01 起）未触碰。")
    lines.append("")
    lines.append("| 类别 | 接口 | 状态 | 行数 | 列数 | 历史起点 | 历史终点 | 有公告日 | 耗时 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in results:
        status = "OK" if row.get("ok") else "失败"
        ranges = row.get("date_ranges") or {}
        first = min((value[0] for value in ranges.values()), default="")
        last = max((value[1] for value in ranges.values()), default="")
        lines.append(
            f"| {row['group']} | `{row['name']}` | {status} | {row.get('rows', 0)} | "
            f"{len(row.get('columns', []))} | {first or '—'} | {last or '—'} | "
            f"{'是' if row.get('has_notice_date') else '否'} | {row.get('seconds', 0):.1f}s |"
        )
    lines.append("")
    lines.append("## 先给结论：哪些能做成事件因子")
    lines.append("")
    lines.append("| 类别 | 可行性 | 理由 |")
    lines.append("|---|---|---|")
    seen: set[str] = set()
    for row in results:
        if row["group"] in seen:
            continue
        seen.add(row["group"])
        verdicts = [VERDICTS.get(item["name"], ("未评估", "")) for item in results if item["group"] == row["group"]]
        rank = {"可用": 0, "已入库": 1, "需逐股": 2, "仅辅助": 3, "不可用于历史": 4, "不可用": 5, "未评估": 6}
        best = sorted(verdicts, key=lambda pair: rank.get(pair[0], 9))[0]
        lines.append(f"| {row['group']} | {best[0]} | {best[1]} |")
    lines.append("")
    lines.append("## 逐项说明")
    for row in results:
        lines.append(f"### {row['group']} · {row['name']}")
        lines.append("")
        lines.append(f"- 用途：{row['why']}")
        verdict = VERDICTS.get(row["name"])
        if verdict:
            lines.append(f"- 判定：**{verdict[0]}** — {verdict[1]}")
        lines.append(f"- PIT：{row['pit']}")
        lines.append(f"- 坑：{row['risks']}")
        if row.get("ok"):
            columns = "、".join(row.get("columns", [])[:24])
            lines.append(f"- 列名（前 24）：{columns}")
            if row.get("date_ranges"):
                lines.append(f"- 日期列范围：{json.dumps(row['date_ranges'], ensure_ascii=False)}")
        else:
            lines.append(f"- 错误：`{row.get('error')}`")
        lines.append("")
    text = "\n".join(lines)
    (out / "probe-events.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="事件因子可行性探测")
    parser.add_argument("--out", default="results/probe-events")
    parser.add_argument("--seconds", type=float, default=0, help="覆盖单项超时秒数")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for probe in PROBES:
        started = time.time()
        row = {
            "group": probe["group"],
            "name": probe["name"],
            "why": probe["why"],
            "pit": probe["pit"],
            "risks": probe["risks"],
        }
        try:
            frame = timeout_call(probe["call"], args.seconds or probe["seconds"], probe["name"])
            row.update(summarise(frame))
            row["ok"] = True
            print(f"[ok] {probe['name']} {row['rows']} 行 {time.time() - started:.1f}s", flush=True)
        except BaseException as error:  # noqa: BLE001
            row["ok"] = False
            row["error"] = f"{type(error).__name__}: {error}"
            print(f"[err] {probe['name']} {row['error']}", flush=True)
        row["seconds"] = round(time.time() - started, 1)
        results.append(row)

    (out / "probe-events.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render(results, out)
    ok = sum(1 for row in results if row.get("ok"))
    print(f"完成 {ok}/{len(results)} 可用 → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
