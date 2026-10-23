"""从公开源复现整个数据湖（一条命令，可中断可续跑）。

这个脚本是给「别人」用的：拿到仓库后，不需要我们那 3.8 GB 的数据，
自己从东财 / 腾讯 / 新浪 / baostock 重新采一遍，并和我们的数字对账。

**默认只重建到 2024-12-31，不含任何 holdout（2025-01-01 起）数据。**
holdout 是所有者的一次性验收集，协作者不应持有；相关阶段（``split``、holdout 的
parquet / valuation）默认跳过，只有所有者显式 ``--allow-holdout`` 且环境变量
``QUANT_OWNER=1`` 时才执行。

用法::

    python -m scripts.reproduce_lake --list                 # 看有哪些阶段、各要多久
    python -m scripts.reproduce_lake --dry-run              # 只打印将要执行的命令
    python -m scripts.reproduce_lake                        # 从头跑到尾（已存在的阶段自动跳过）
    python -m scripts.reproduce_lake --from raw             # 从某个阶段开始
    python -m scripts.reproduce_lake --stage ref,fund       # 只跑指定阶段
    python -m scripts.reproduce_lake --force --stage process  # 强制重跑
    python -m scripts.reproduce_lake --verify-only          # 只对账，不采集
    python -m scripts.reproduce_lake --audit-cutoff         # 检查本机数据有没有越过 2024-12-31
    python -m scripts.reproduce_lake --snapshot             # 生成参考计数（维护者用）

设计原则：
- 每个阶段 = 一组 ``python -m ...`` 命令 + 产物清单；产物在就跳过（``--force`` 除外）。
- 全部命令写日志（``logs/reproduce-YYYYmmdd-HHMM.log``），失败即停（``--keep-going`` 可继续）。
- 采集顺序固定：参考数据 → 原始层 → 精简层 → 列式层 → 估值/财务/事件 → 另类 → 历史段。
- 所有采集命令显式带 ``--end 2024-12-31``（原始层、估值、财务、事件），不依赖服务端默认。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path("D:/quant-data")
LOG_DIR = Path("logs")
REFERENCE = Path("docs/data-reference-counts.json")

#: 协作者可重建的最晚日期。holdout（2025-01-01 起）不在公开流程里。
SAMPLE_END = "2024-12-31"
HOLDOUT_START = "2025-01-01"


def is_owner() -> bool:
    """所有者模式：环境变量 ``QUANT_OWNER=1``。只是流程隔离，不是密码学隔离。"""
    return os.environ.get("QUANT_OWNER", "").strip() in {"1", "true", "yes"}


@dataclass
class Stage:
    name: str
    title: str
    cmds: list[list[str]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    minutes: int = 0
    note: str = ""
    source: str = ""
    #: 该阶段会产出 holdout 数据，默认不跑（仅所有者 + --allow-holdout 时跑）
    holdout: bool = False
    #: 仅所有者模式追加的命令与产物（如 holdout 的 parquet / valuation）
    owner_cmds: list[list[str]] = field(default_factory=list)
    owner_artifacts: list[str] = field(default_factory=list)


STAGES: list[Stage] = [
    Stage("env", "环境与依赖检查", artifacts=[], minutes=1,
          note="缺 pyarrow/requests/akshare/baostock 会直接报出来", source="—"),
    Stage(
        "meta",
        "初始化目录 + 交易日历",
        [["-m", "quant", "lake", "init"], ["-m", "quant", "lake", "calendar"]],
        artifacts=["meta/calendar.csv"],
        minutes=2,
        source="上证指数日线（腾讯/新浪）",
    ),
    Stage(
        "universe",
        "全市场代码表（含上市/退市日）",
        [["-m", "quant", "lake", "universe"], ["-m", "quant", "lake", "ref", "--what", "universe"]],
        artifacts=["ref/universe.csv", "meta/universe.csv"],
        minutes=15,
        source="新浪 hs_a 列表 + 东财代码表",
    ),
    Stage(
        "ref",
        "行业 PIT / 股本 PIT / 申万快照",
        [["-m", "quant", "lake", "ref", "--what", "industry,shares,sw", "--start-year", "2014"]],
        artifacts=["ref/industry.csv", "ref/shares.csv"],
        minutes=40,
        source="baostock 行业分类（年度版本）+ 东财股本变动表",
    ),
    Stage(
        "dividends",
        "PIT 分红明细",
        [["-m", "quant", "lake", "dividends", "--start", "2013-12-31"]],
        artifacts=["ref/dividends.parquet"],
        minutes=10,
        source="东财 RPT_SHAREBONUS_DET（ex_date 生效）",
    ),
    Stage(
        "actions",
        "除权除息事件（复权因子用）",
        [["-m", "quant", "lake", "actions"]],
        artifacts=["raw/actions.csv"],
        minutes=5,
        source="东财 datacenter 分红送转",
    ),
    Stage(
        "raw",
        "原始层全市场日线（不复权）",
        [["-m", "quant", "lake", "crawl", "--source", "auto", "--pause", "0.3",
          "--end", SAMPLE_END]],
        artifacts=["raw/daily"],
        minutes=300,
        note="5,575 只，按腾讯/新浪自动降级；断点续跑，可 Ctrl+C 后重跑；截止 2024-12-31",
        source="腾讯 fqkline（不复权）/ 新浪（北交所兜底）",
    ),
    Stage(
        "split",
        "切出 holdout 层并加锁（默认跳过）",
        [["-m", "quant", "lake", "split"]],
        artifacts=["holdout/daily"],
        minutes=5,
        holdout=True,
        note="holdout 是所有者的一次性验收集；协作者不应持有，默认不跑",
        source="本地 raw 切分（2025-01-01 起）",
    ),
    Stage(
        "process",
        "生成精简层（补停牌 + 前复权因子）",
        [["-m", "quant", "lake", "process"]],
        artifacts=["processed/daily"],
        minutes=15,
        source="本地 raw + actions",
    ),
    Stage(
        "parquet",
        "列式层（读取快 4.7 倍）",
        [["-m", "quant", "lake", "parquet", "--kinds", "processed"]],
        artifacts=["processed/parquet"],
        minutes=5,
        owner_cmds=[["-m", "quant", "lake", "parquet", "--kinds", "holdout", "--allow-holdout"]],
        owner_artifacts=["holdout/parquet"],
        source="本地 CSV 转换",
    ),
    Stage(
        "valuation",
        "估值 / 换手 / isST 日频快照",
        [["-m", "quant", "lake", "valuation", "--start", "2014-01-01",
          "--end", SAMPLE_END, "--pause", "0.1"]],
        artifacts=["valuation"],
        minutes=420,
        owner_cmds=[["-m", "quant", "lake", "valuation", "--holdout", "--allow-holdout"]],
        owner_artifacts=["holdout/valuation"],
        note="最慢的一步：5,500 只逐只拉；可用 --shard i/n 开多进程；样本内截止 2024-12-31",
        source="baostock query_history_k_data_plus（peTTM/pbMRQ/psTTM/isST/turn）",
    ),
    Stage(
        "turnpanel",
        "换手率宽表",
        [["-m", "quant", "lake", "turnpanel", "--cutoff", "2024-12-31"]],
        artifacts=["ref/turn_panel.parquet"],
        minutes=10,
        source="本地 valuation 透视",
    ),
    Stage(
        "fund",
        "财务三表（按报告期，PIT NOTICE_DATE）",
        [["-m", "quant", "lake", "fund", "--tables", "profit,balance,cashflow",
          "--start", "2013-03-31", "--end", SAMPLE_END]],
        artifacts=["fundamentals/profit", "fundamentals/balance", "fundamentals/cashflow"],
        minutes=90,
        source="东财 RPT_F10_FINANCE_GINCOME/GBALANCE/GCASHFLOW/MAINFINADATA",
    ),
    Stage(
        "mainfin",
        "主要财务指标（含金融股）",
        [["-m", "quant", "lake", "mainfin", "--start", "2013-03-31", "--end", SAMPLE_END]],
        artifacts=["fundamentals/indicator", "fundamentals/mainfin"],
        minutes=60,
        note="金融股只有这张表有 ROE/EPS/BPS",
        source="东财 RPT_F10_FINANCE_MAINFINADATA",
    ),
    Stage(
        "index",
        "指数成分快照",
        [["-m", "quant", "lake", "index",
          "--codes", "000300,000905,000852,000016,000688,399006"]],
        artifacts=["ref/index_members.csv"],
        minutes=5,
        note="只有当期快照 → 历史成分有幸存者偏差（已写入偏差声明）",
        source="东财指数成分",
    ),
    Stage(
        "delisted",
        "退市股日线 + 退市事件",
        [["-m", "quant", "lake", "delisted"]],
        artifacts=["delisted/daily", "delisted/events.csv"],
        minutes=60,
        source="baostock（后复权，截止退市日）",
    ),
    Stage(
        "events",
        "事件层（龙虎榜/回购/增减持/ST/送转）",
        [["-m", "quant", "lake", "events", "--what", "all", "--start", "2014-01-01",
          "--end", SAMPLE_END]],
        artifacts=["events/lhb.parquet", "events/hold.parquet", "events/st.parquet"],
        minutes=40,
        source="东财 datacenter 各接口（双日期 event_date + available_date）",
    ),
    Stage(
        "alt",
        "另类数据（预告/快报/解禁/质押）",
        [["-m", "scripts.growth_alt_collect", "--only", "all", "--resume"]],
        artifacts=["alt/forecast.parquet", "alt/unlock.parquet", "alt/pledge.parquet"],
        minutes=90,
        note="解禁/质押是成长策略 aux 块的输入；缺了 n8/n9 就复现不出主规格",
        source="东财 stock_yjyg_em / stock_yjkb_em / stock_restricted_release_detail_em / stock_gpzy_pledge_ratio_em",
    ),
    Stage(
        "history",
        "2007–2016 历史日线与历史财务",
        [["-m", "scripts.growth_history_fetch_em", "--task", "all"],
         ["-m", "scripts.growth_history_gapfill", "--consolidate"],
         ["-m", "scripts.growth_history_fetch_em", "--task", "audit"]],
        artifacts=["growth-history/parquet", "growth-history/fundamentals"],
        minutes=240,
        note="退市股走腾讯兜底；缺口 298 只（9.5%）是源侧限制，已在文档声明",
        source="新浪 stock_zh_a_daily + 后复权因子；腾讯兜底；东财历史财务",
    ),
    Stage(
        "early",
        "2001–2005 慢熊段日线（baostock）",
        [["-m", "scripts.crisis_early_fetch", "--sample", "2001-2005"],
         ["-m", "scripts.crisis_early_fetch", "--sample", "2001-2005", "--consolidate"],
         ["-m", "scripts.crisis_early_fetch", "--sample", "1997-1999", "--index-only"]],
        artifacts=["early/parquet", "early/indices"],
        minutes=300,
        note="baostock 单进程限速；并发登录会被拉黑（10001011）",
        source="baostock 日线 + 复权因子",
    ),
    Stage(
        "verify",
        "对账：lake status/verify + 与本仓库参考计数比对",
        [["-m", "quant", "lake", "status"], ["-m", "quant", "lake", "verify", "--limit", "300"]],
        artifacts=[],
        minutes=5,
        source="本地",
    ),
]

STAGE_MAP = {s.name: s for s in STAGES}


def log(msg: str) -> None:
    print(msg, flush=True)


def check_env() -> tuple[bool, str]:
    missing = []
    for mod in ("pandas", "numpy", "pyarrow", "requests", "akshare", "baostock"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(mod)
    free_gb = shutil.disk_usage(ROOT.drive + "\\" if ROOT.drive else ".").free / 2**30 if ROOT.drive else 0
    ok = not missing
    detail = f"缺依赖：{missing}" if missing else "依赖齐全"
    detail += f"；{ROOT.drive or '.'} 可用 {free_gb:.0f} GB（建议 ≥ 15 GB）"
    return ok, detail


def stage_done(stage: Stage, root: Path, *, owner: bool = False) -> bool:
    artifacts = list(stage.artifacts) + (stage.owner_artifacts if owner else [])
    if not artifacts:
        return False
    return all((root / a).exists() for a in artifacts)


def audit_cutoff(root: Path) -> bool:
    """扫每一层的日期列，确认没有任何数据越过 ``SAMPLE_END``。

    协作者跑完复现后应全部 ≤ 2024-12-31；本机如果有 holdout 层，会单独标出
    （那是所有者的一次性验收集，不在公开流程里）。
    """
    checks: list[tuple[str, str, str]] = [
        ("raw/daily", "raw/daily/*.csv", "date"),
        ("processed/parquet", "processed/parquet/**/*.parquet", "date"),
        ("valuation", "valuation/*.parquet", "date"),
        ("delisted/daily", "delisted/daily/*.parquet", "date"),
        ("growth-history", "growth-history/parquet/**/*.parquet", "date"),
        ("early", "early/bars/*/parquet/year=*/data.parquet", "date"),
        ("fundamentals", "fundamentals/*/*.parquet", "REPORT_DATE"),
        ("events", "events/*.parquet", "event_date"),
        ("alt", "alt/*.parquet", None),
        ("holdout/parquet", "holdout/parquet/**/*.parquet", "date"),
        ("holdout/valuation", "holdout/valuation/*.parquet", "date"),
    ]
    cutoff = str(SAMPLE_END)
    ok = True
    print(f"{'层':<22}{'文件':>7}{'最大日期':>14}  判定")
    for name, pattern, column in checks:
        files = sorted(root.glob(pattern))
        if not files:
            print(f"{name:<22}{0:>7}{'—':>14}  不存在（跳过）")
            continue
        newest = ""
        try:
            if files[0].suffix == ".parquet":
                import pyarrow.dataset as ds

                table = ds.dataset([str(f) for f in files], format="parquet")
                names = table.schema.names
                col = column if column in names else ("date" if "date" in names else None)
                if col is None:
                    print(f"{name:<22}{len(files):>7}{'无日期列':>14}  跳过")
                    continue
                values = table.to_table(columns=[col]).column(0).to_pylist()
                dates = [str(v)[:10] for v in values if v is not None]
                newest = max(dates) if dates else ""
            else:
                import pandas as pd

                dates = []
                for f in files:
                    try:
                        frame = pd.read_csv(f, usecols=[column], encoding="utf-8-sig")
                        dates.extend(str(v)[:10] for v in frame[column].dropna().unique())
                    except Exception:  # noqa: BLE001 - 单文件坏了不算致命
                        continue
                newest = max(dates) if dates else ""
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<22}{len(files):>7}{'读取失败':>14}  {exc}")
            continue
        is_holdout = name.startswith("holdout")
        if not newest:
            verdict = "无可核对日期"
        elif newest > cutoff:
            verdict = "holdout 层（所有者本机）" if is_holdout else f"✗ 越界 {newest} > {cutoff}"
            if not is_holdout:
                ok = False
        else:
            verdict = "✓"
        print(f"{name:<22}{len(files):>7}{newest or '—':>14}  {verdict}")
    print(f"\n[audit] {'通过：所有公开层都 ≤ ' + cutoff if ok else '失败：有层越过 ' + cutoff}")
    return ok


def run_cmd(cmd: list[str], logf, dry: bool) -> int:
    full = [sys.executable, *cmd]
    line = " ".join(full)
    log(f"    $ {line}")
    logf.write(line + "\n")
    logf.flush()
    if dry:
        return 0
    started = time.time()
    proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    tail: list[str] = []
    assert proc.stdout is not None
    for out in proc.stdout:
        logf.write(out)
        tail.append(out.rstrip())
        if len(tail) > 400:
            tail.pop(0)
        print(out, end="")
    code = proc.wait()
    log(f"    -> exit {code}  ({time.time() - started:.0f}s)")
    logf.write(f"-> exit {code}\n")
    return code


def units(root: Path) -> dict:
    """各层文件数/体积；行数只对 parquet 层读元数据（快）。"""
    rows: dict = {}
    for name, pattern in [
        ("raw/daily", "raw/daily/*.csv"),
        ("processed/parquet", "processed/parquet/**/*.parquet"),
        ("holdout/parquet", "holdout/parquet/**/*.parquet"),
        ("valuation", "valuation/*.parquet"),
        ("fundamentals", "fundamentals/*/*.parquet"),
        ("ref", "ref/*.parquet"),
        ("events", "events/*.parquet"),
        ("alt", "alt/*.parquet"),
        ("delisted", "delisted/daily/*.parquet"),
        ("growth-history/parquet", "growth-history/parquet/**/*.parquet"),
        ("early/parquet", "early/bars/*/parquet/year=*/data.parquet"),
        ("early/indices", "early/indices/*/*.csv"),
    ]:
        files = sorted(root.glob(pattern)) if root.exists() else []
        item = {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
        if files and files[0].suffix == ".parquet":
            item["rows"] = _parquet_rows(files)
        rows[name] = item
    return rows


def _parquet_rows(files: list[Path]) -> int:
    try:
        import pyarrow.parquet as pq
    except Exception:  # noqa: BLE001
        return -1
    total = 0
    for f in files:
        try:
            total += pq.ParquetFile(f).metadata.num_rows
        except Exception:  # noqa: BLE001
            return -1
    return total


def snapshot(root: Path, out: Path) -> None:
    data = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "root": str(root), "layers": units(root)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[snapshot] {out}")


def verify_against_reference(root: Path) -> bool:
    if not REFERENCE.exists():
        log(f"[verify] 没有参考计数 {REFERENCE}，跳过比对")
        return True
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    cur = units(root)
    ok = True
    log(f"{'层':<22}{'文件':>8}{'参考':>8}{'行数':>14}{'参考行数':>14}")
    for name, r in ref["layers"].items():
        c = cur.get(name, {})
        if name.startswith("holdout") and c.get("files", 0) == 0:
            log(f"{name:<22}{0:>8}{r['files']:>8}{'—':>14}{r.get('rows', -1):>14}  （未建 holdout，跳过）")
            continue
        files_ok = c.get("files", 0) == r["files"]
        rows_ok = r.get("rows", 0) in (-1, 0) or c.get("rows", -1) == r["rows"]
        if not (files_ok and rows_ok):
            ok = False
        log(f"{name:<22}{c.get('files', 0):>8}{r['files']:>8}{c.get('rows', -1):>14}{r.get('rows', -1):>14}"
            f"  {'✓' if files_ok and rows_ok else '✗'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从公开源复现数据湖")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--stage", default="", help="逗号分隔的阶段名")
    ap.add_argument("--from", dest="from_stage", default="")
    ap.add_argument("--to", dest="to_stage", default="")
    ap.add_argument("--force", action="store_true", help="产物已存在也重跑")
    ap.add_argument("--keep-going", action="store_true", help="某阶段失败后继续")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--audit-cutoff", action="store_true", help="检查数据湖是否越过 2024-12-31")
    ap.add_argument("--allow-holdout", action="store_true",
                    help="所有者：包含 holdout 阶段（需环境变量 QUANT_OWNER=1）")
    ap.add_argument("--snapshot", action="store_true", help="生成参考计数文件（维护者用）")
    args = ap.parse_args(argv)

    owner = is_owner()
    if args.allow_holdout and not owner:
        log("[拒绝] --allow-holdout 只对所有者开放：请设 QUANT_OWNER=1 再运行。")
        log("       协作者不需要 holdout（2025-01-01 起的数据不进入公开复现流程）。")
        return 2

    if args.list:
        log(f"{'阶段':<10}{'内容':<30}{'预计':>6}  产物")
        total = 0
        for s in STAGES:
            total += s.minutes
            mark = "  [holdout]" if s.holdout else ""
            log(f"{s.name:<10}{s.title:<30}{s.minutes:>4} 分  {', '.join(s.artifacts) or '—'}{mark}")
        log(f"{'合计':<10}{'':<30}{total:>4} 分（不含重试与限速等待；[holdout] 默认跳过）")
        return 0

    if args.snapshot:
        snapshot(args.root, REFERENCE)
        return 0

    if args.audit_cutoff:
        return 0 if audit_cutoff(args.root) else 1

    if args.verify_only:
        return 0 if verify_against_reference(args.root) else 1

    names = [s.strip() for s in args.stage.split(",") if s.strip()]
    if args.from_stage:
        names = [s.name for s in STAGES[STAGES.index(STAGE_MAP[args.from_stage]):]]
    if args.to_stage:
        idx = STAGES.index(STAGE_MAP[args.to_stage]) + 1
        names = [s.name for s in STAGES[:idx]]
    if not names:
        names = [s.name for s in STAGES]
    unknown = [n for n in names if n not in STAGE_MAP]
    if unknown:
        raise SystemExit(f"未知阶段：{unknown}")

    allow_holdout = bool(args.allow_holdout and owner)
    skipped_holdout = [n for n in names if STAGE_MAP[n].holdout and not allow_holdout]
    if skipped_holdout:
        names = [n for n in names if n not in skipped_holdout]

    LOG_DIR.mkdir(exist_ok=True)
    logpath = LOG_DIR / f"reproduce-{time.strftime('%Y%m%d-%H%M')}.log"
    started = time.time()
    failures: list[str] = []
    with logpath.open("w", encoding="utf-8") as logf:
        log(f"[root] {args.root}   日志：{logpath}")
        if skipped_holdout:
            log(f"[holdout] 跳过 {skipped_holdout}：公开流程只重建到 {SAMPLE_END}，"
                "holdout 是所有者的一次性验收集（需 QUANT_OWNER=1 且 --allow-holdout）")
        for name in names:
            stage = STAGE_MAP[name]
            if name == "env":
                ok, detail = check_env()
                log(f"[env] {'OK' if ok else 'FAIL'} — {detail}")
                if not ok and not args.keep_going:
                    return 1
                continue
            if stage_done(stage, args.root, owner=allow_holdout) and not args.force:
                log(f"[skip] {name}（产物已存在，--force 可重跑）")
                continue
            log(f"[stage] {name} — {stage.title}（预计 {stage.minutes} 分钟；源：{stage.source}）")
            cmds = list(stage.cmds) + (list(stage.owner_cmds) if allow_holdout else [])
            for cmd in cmds:
                code = run_cmd(cmd, logf, args.dry_run)
                if code != 0:
                    failures.append(f"{name}: exit {code}")
                    log(f"[fail] {name} 退出码 {code}")
                    if not args.keep_going:
                        log(f"[abort] 已停止；修复后可从 --from {name} 继续")
                        return 1
                    break
            log(f"[done] {name}  累计 {time.time() - started:.0f}s")

    if failures:
        log("[summary] 失败阶段：" + "; ".join(failures))
        return 1
    log(f"[summary] 全部完成，用时 {(time.time() - started)/60:.1f} 分钟；日志 {logpath}")
    if REFERENCE.exists():
        log("[verify] 与参考计数比对：")
        verify_against_reference(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
