"""环境可靠性自检（doctor）：把"能不能跑、数据拿不拿得到、能不能下单"提前查清楚。

为什么要有它：回测跑出漂亮数字、实盘却下不出单，八成是环境问题——
缺依赖、时钟偏了、数据源连不上、目录不可写、通道前置条件没满足。
这个模块把这些问题变成**可判定的检查项**，每条给三态与修复建议，
有 ``fail`` 就返回退出码 1，可以直接塞进定时任务当守门人。

设计约束：

* 只用标准库 + pandas/numpy（**不引入新依赖**），所以"环境还没配好"时它也能跑；
* 网络探测全部并行、带超时，任何一个端点挂掉都不影响其余检查；
* 报告同时落盘 ``results/doctor.json`` 与 ``results/doctor.md``，便于事后对比。

用法::

    python -m quant doctor                 # 全量检查（含联网与引擎冒烟）
    python -m quant doctor --no-network    # 离线模式（只查本地环境）
    python -m quant doctor --json          # 机器可读
    python -m quant doctor --deep          # 额外查券商客户端进程（QMT 等）
"""

from __future__ import annotations

import argparse
import email.utils
import json
import os
import platform
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

__all__ = ["Check", "DoctorReport", "run_doctor", "main"]

ROOT = Path(__file__).resolve().parents[1]

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
ICON = {OK: "✓", WARN: "!", FAIL: "×", SKIP: "-"}

SHANGHAI = timezone(timedelta(hours=8))
"""A股用北京时间（UTC+8，无夏令时），时钟与时段判断都按它算。"""

# ---------------------------------------------------------------------------
# 探针目标：这些就是"爬数据"将来要用的真实接口，先证明它们通不通
# ---------------------------------------------------------------------------

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

DATA_ENDPOINTS: tuple[tuple[str, str, str | None], ...] = (
    (
        "东方财富·日线接口（主）",
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        "?secid=1.600519&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55,f56,f57"
        "&klt=101&fqt=1&beg=20240101&end=20240115",
        "klines",
    ),
    (
        "腾讯·日线接口（备 1）",
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600519,day,,,20,qfq",
        "qfqday",
    ),
    (
        "新浪·日线接口（备 2）",
        "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        "?symbol=sh600519&scale=240&ma=no&datalen=20",
        "day",
    ),
    (
        "新浪·全市场股票列表",
        "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
        "?page=1&num=5&sort=symbol&asc=1&node=hs_a",
        "symbol",
    ),
    ("新浪·实时行情", "https://hq.sinajs.cn/list=sh600519", "600519"),
    ("腾讯·实时行情", "https://qt.gtimg.cn/q=sh600519", "600519"),
    ("上交所", "https://www.sse.com.cn", None),
    ("深交所", "https://www.szse.cn", None),
)

OPTIONAL_ENDPOINTS: tuple[tuple[str, str, str | None], ...] = (
    (
        "东财·全市场列表（push2 clist，本机实测受限）",
        "https://push2.eastmoney.com/api/qt/clist/get"
        "?pn=1&pz=5&po=1&np=1&fltt=2&invt=2&fid=f3"
        "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12,f14,f2",
        "diff",
    ),
    ("GitHub（历史偶发超时）", "https://github.com", None),
    ("Tushare Pro（需 token，非必需）", "https://api.tushare.pro", None),
    ("codeload（插件安装通道）", "https://codeload.github.com", None),
    ("npm registry", "https://registry.npmjs.org", None),
    ("npmmirror（国内镜像）", "https://registry.npmmirror.com", None),
)

OPTIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("akshare", "免费全市场行情/财务（爬数据主力）"),
    ("tushare", "Tushare Pro 客户端（需 token）"),
    ("baostock", "免费历史行情备选源"),
    ("scipy", "统计与优化（因子中性化、组合优化可用）"),
    ("statsmodels", "回归/时序分析（因子检验可用）"),
    ("sklearn", "机器学习选股（可选）"),
    ("matplotlib", "画图（本工程自带 SVG，不需要）"),
    ("xtquant", "迅投 QMT 交易接口（真要自动下单才需要）"),
)


@dataclass
class Check:
    """一条检查结论。"""

    group: str
    name: str
    status: str
    detail: str
    hint: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)
    started_at: str = ""
    elapsed: float = 0.0

    def add(self, group: str, name: str, status: str, detail: str, hint: str = "") -> Check:
        check = Check(group=group, name=name, status=status, detail=detail, hint=hint)
        self.checks.append(check)
        return check

    # -- 汇总 ---------------------------------------------------------------

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    def groups(self) -> list[str]:
        seen: list[str] = []
        for check in self.checks:
            if check.group not in seen:
                seen.append(check.group)
        return seen

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "startedAt": self.started_at,
            "elapsed": round(self.elapsed, 3),
            "summary": {
                "total": len(self.checks),
                "ok": sum(c.status == OK for c in self.checks),
                "warn": len(self.warnings),
                "fail": len(self.failures),
                "skip": sum(c.status == SKIP for c in self.checks),
            },
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_text(self, *, color: bool = False) -> str:
        lines: list[str] = []
        for group in self.groups():
            lines.append(f"\n【{group}】")
            for check in [c for c in self.checks if c.group == group]:
                mark = ICON.get(check.status, "?")
                lines.append(f"  {mark} {check.name}：{check.detail}")
                if check.hint:
                    lines.append(f"      → {check.hint}")
        data = self.to_dict()["summary"]
        lines.append(
            f"\n合计 {data['total']} 项：通过 {data['ok']} / 警告 {data['warn']} / "
            f"失败 {data['fail']} / 跳过 {data['skip']}，用时 {self.elapsed:.1f}s"
        )
        lines.append("结论：" + ("环境可用" if self.ok else "存在阻断项，先修 fail 再用"))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            "# 环境自检报告（doctor）",
            "",
            f"- 生成时间：{self.started_at}",
            f"- 用时：{self.elapsed:.1f}s",
            f"- 结论：**{'环境可用' if self.ok else '存在阻断项'}**",
            "",
            "| 组 | 检查项 | 结果 | 说明 |",
            "| --- | --- | --- | --- |",
        ]
        for check in self.checks:
            mark = {"ok": "✅", "warn": "⚠️", "fail": "❌", "skip": "➖"}.get(check.status, "?")
            detail = check.detail.replace("|", "\\|")
            lines.append(f"| {check.group} | {check.name} | {mark} | {detail} |")
        hints = [c for c in self.checks if c.hint]
        if hints:
            lines += ["", "## 待办与修复建议", ""]
            for check in hints:
                lines.append(f"- **{check.name}**（{check.status}）：{check.hint}")
        return "\n".join(lines) + "\n"

    def write(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "doctor.json"
        md_path = out / "doctor.md"
        json_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path


# ---------------------------------------------------------------------------
# 探针工具
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    url: str
    ok: bool
    status: int | None
    ms: float
    body: bytes = b""
    note: str = ""


def _probe(url: str, *, timeout: float = 6.0, limit: int = 65536, attempts: int = 3) -> Probe:
    """发一个 GET，只关心"通不通、多快、返回什么特征"。HTTP 错误码也算通。

    实测教训：行情接口会**偶发** `RemoteDisconnected`（对端直接断开，换个时刻就好），
    单次探测会把可用源误判成不可用。所以默认重试 3 次，只在全部失败时才判不通，
    并把"重试后才通"记进 note——这本身就是重要的可靠性信息。
    """
    last: Probe | None = None
    for attempt in range(1, max(1, attempts) + 1):
        result = _probe_once(url, timeout=timeout, limit=limit)
        if result.ok:
            if attempt > 1:
                result.note = (result.note + " · " if result.note else "") + f"第 {attempt} 次才成功（该源偶发不稳）"
            return result
        last = result
        if attempt < attempts:
            time.sleep(0.8)
    assert last is not None
    last.note = (last.note + " · " if last.note else "") + f"重试 {attempts} 次均失败"
    return last


def _probe_once(url: str, *, timeout: float = 6.0, limit: int = 65536) -> Probe:
    request = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Referer": "https://finance.sina.com.cn/", "Accept": "*/*"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(limit)
            return Probe(url, True, response.status, (time.perf_counter() - start) * 1000, body)
    except urllib.error.HTTPError as exc:  # 能拿到状态码说明链路是通的
        return Probe(url, True, exc.code, (time.perf_counter() - start) * 1000, note=f"HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001 - 任何异常都只当作"不通"
        return Probe(url, False, None, (time.perf_counter() - start) * 1000, note=f"{type(exc).__name__}: {exc}")


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def _fmt_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}TB"


# ---------------------------------------------------------------------------
# 各组检查
# ---------------------------------------------------------------------------


def check_runtime(report: DoctorReport) -> None:
    group = "运行时"
    version = sys.version_info
    report.add(
        group,
        "Python 版本",
        OK if version >= (3, 10) else FAIL,
        f"{platform.python_version()}（{sys.executable}）",
        "" if version >= (3, 10) else "需要 Python ≥3.10（用到 `X | None` 等语法）",
    )
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    report.add(
        group,
        "解释器环境",
        OK,
        ("虚拟环境/运行时自带依赖目录" if in_venv else "系统级解释器（未检测到 venv）")
        + f"；prefix={sys.prefix}",
        "" if in_venv else "建议在独立环境里跑，避免与系统包相互影响（本机 DSH 运行时目录即满足）",
    )

    try:
        import numpy as np
        import pandas as pd

        report.add(group, "pandas", OK, f"{pd.__version__}（Python 版本下限 ≥3.10 已满足）")
        report.add(group, "numpy", OK, f"{np.__version__}")
        major = int(str(pd.__version__).split(".")[0])
        report.add(
            group,
            "pandas 大版本兼容",
            OK if major >= 2 else WARN,
            f"pandas {pd.__version__}（本工程在 2.x / 3.x 上测过）",
            "" if major >= 2 else "1.x 已停止维护，定时任务里的 freq 别名与 fillna 行为可能不同",
        )
    except Exception as exc:  # noqa: BLE001
        report.add(group, "pandas/numpy", FAIL, f"导入失败：{type(exc).__name__}: {exc}", "先修依赖再谈回测")

    report.add(
        group,
        "操作系统",
        OK,
        f"{platform.system()} {platform.release()}（{platform.machine()}）",
    )
    report.add(
        group,
        "CPU",
        OK if (os.cpu_count() or 1) >= 4 else WARN,
        f"{os.cpu_count()} 逻辑核 · {platform.processor() or '未知型号'}",
    )

    tz_name = time.tzname
    offset = -(time.timezone if not time.daylight else time.altzone) / 3600
    report.add(
        group,
        "系统时区",
        OK if abs(offset - 8) < 0.01 else WARN,
        f"UTC{offset:+.1f}（{tz_name[0]}）",
        "" if abs(offset - 8) < 0.01 else "A股按 UTC+8 计时；本机时区不同会让「今天」和交易时段判断错位",
    )

    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    report.add(
        group,
        "控制台编码",
        OK if "utf" in encoding else WARN,
        f"stdout={encoding or '未知'} · PYTHONIOENCODING={os.environ.get('PYTHONIOENCODING', '未设置')}",
        "" if "utf" in encoding else "Windows 默认 GBK 会在打印 ✓ 时抛 UnicodeEncodeError；"
        "设 PYTHONIOENCODING=utf-8（本工程的测试入口已自带兜底）",
    )


def check_hardware(report: DoctorReport) -> None:
    group = "硬件与磁盘"
    try:
        usage = shutil.disk_usage(ROOT)
        free_gb = usage.free / 1024**3
        report.add(
            group,
            "工作区磁盘余量",
            OK if free_gb >= 5 else (WARN if free_gb >= 1 else FAIL),
            f"{ROOT.drive or ''} 可用 {free_gb:.1f}GB / 总 {usage.total / 1024**3:.1f}GB",
            "" if free_gb >= 5 else "全市场日线 + 财务数据大约需要几 GB；先腾出空间再爬数据",
        )
    except Exception as exc:  # noqa: BLE001
        report.add(group, "工作区磁盘余量", WARN, f"查询失败：{exc}")

    try:
        import subprocess as _sp

        out = _sp.run(
            ["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/value"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        values = {}
        for line in out.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip()
        free_gb = int(values.get("FreePhysicalMemory", 0)) / 1024**2
        total_gb = int(values.get("TotalVisibleMemorySize", 1)) / 1024**2
        report.add(
            group,
            "内存",
            OK if free_gb >= 2 else WARN,
            f"可用 {free_gb:.1f}GB / 总 {total_gb:.1f}GB",
            "" if free_gb >= 2 else "全市场面板（5000 只 × 数年日线）峰值内存约 1–3GB",
        )
    except Exception:  # noqa: BLE001
        report.add(group, "内存", SKIP, "wmic 不可用，跳过（不影响运行）")

    # 项目内临时目录必须可写：Windows 沙箱下 mkdtemp 会失败，这里验证的是工程自己的方案
    try:
        tmp = ROOT / "_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        probe_file = tmp / "doctor-write-test.txt"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink()
        report.add(group, "临时目录可写", OK, f"{tmp}（Path.mkdir 方案，可继承权限）")
    except Exception as exc:  # noqa: BLE001
        report.add(group, "临时目录可写", FAIL, f"{type(exc).__name__}: {exc}", "检查目录权限/杀软拦截")


def check_clock(report: DoctorReport, *, network: bool) -> None:
    group = "时钟与交易时段"
    now_local = datetime.now()
    now_sh = now_local.astimezone(SHANGHAI)
    weekday = now_sh.strftime("%A")
    in_session = (
        now_sh.weekday() < 5
        and ((9, 30) <= (now_sh.hour, now_sh.minute) <= (11, 30) or (13, 0) <= (now_sh.hour, now_sh.minute) <= (15, 0))
    )
    report.add(
        group,
        "北京时间",
        OK,
        f"{now_sh:%Y-%m-%d %H:%M:%S}（{weekday}）· 当前{'在' if in_session else '不在'}连续竞价时段",
    )

    if not network:
        report.add(group, "与外部时钟偏差", SKIP, "离线模式未测（用 --no-network 关闭了网络探测）")
        return
    # 用交易所返回的 Date 头当参照（东财接口不一定给 Date）
    server_date = None
    try:
        request = urllib.request.Request("https://www.sse.com.cn", headers={"User-Agent": _UA})
        with urllib.request.urlopen(request, timeout=6.0) as response:
            header = response.headers.get("Date")
            if header:
                server_date = email.utils.parsedate_to_datetime(header)
    except Exception:  # noqa: BLE001
        server_date = None
    if server_date is None:
        report.add(
            group,
            "与外部时钟偏差",
            WARN,
            "上游未返回 Date 头，无法比对",
            "下单前手工核对系统时间（Windows 设置里「立即同步」）",
        )
    else:
        skew = abs((datetime.now(timezone.utc) - server_date.astimezone(timezone.utc)).total_seconds())
        status = OK if skew < 5 else (WARN if skew < 60 else FAIL)
        report.add(
            group,
            "与外部时钟偏差",
            status,
            f"{skew:.1f}s（对照 www.sse.com.cn 的 Date 头）",
            "" if status == OK else "时钟偏差会让「当日订单/行情日期」判断出错；同步 Windows 时间",
        )


def check_network(report: DoctorReport, *, network: bool, timeout: float) -> None:
    group = "网络与数据源"
    if not network:
        report.add(group, "数据源连通性", SKIP, "离线模式未测")
        report.add(group, "可选通道连通性", SKIP, "离线模式未测")
        return

    hosts = sorted({url.split("/")[2] for _, url, _ in DATA_ENDPOINTS + OPTIONAL_ENDPOINTS})
    with ThreadPoolExecutor(max_workers=min(8, len(hosts))) as pool:
        resolutions = list(pool.map(lambda h: (h, _resolve(h)), hosts))
    bad_dns = [h for h, ips in resolutions if not ips]
    report.add(
        group,
        "DNS 解析",
        OK if not bad_dns else FAIL,
        f"{len(hosts)} 个主机全部解析成功" if not bad_dns else f"解析失败：{', '.join(bad_dns)}",
        "" if not bad_dns else "检查 DNS/代理设置（公司网络常见：需内网 DNS）",
    )
    if bad_dns:
        return

    with ThreadPoolExecutor(max_workers=6) as pool:
        data_probes = list(pool.map(lambda item: (item, _probe(item[1], timeout=timeout)), DATA_ENDPOINTS))

    # 日线接口按「三源互备」整体判定：只要有一个通就算可用（东财会间歇性断连，属常态）
    kline_probes = [pair for pair in data_probes if "日线接口" in pair[0][0]]
    other_probes = [pair for pair in data_probes if "日线接口" not in pair[0][0]]
    if kline_probes:
        live: list[str] = []
        dead: list[str] = []
        for (label, url, marker), probe in kline_probes:
            host = url.split("/")[2]
            short = label.split("·")[-1]
            if not probe.ok:
                dead.append(f"{short}（{host} 不通：{probe.note}）")
                continue
            text = probe.body.decode("utf-8", errors="ignore")
            if marker and marker not in text:
                dead.append(f"{short}（返回体缺 `{marker}`）")
                continue
            live.append(f"{short} {probe.ms:.0f}ms")
        if live:
            detail = "可用：" + "、".join(live)
            if dead:
                detail += "；不可用：" + "、".join(dead)
                report.add(
                    group,
                    "日线接口（三源互备）",
                    WARN,
                    detail,
                    "有源可用即可取数：`fetch --mode web` 会按 东财→腾讯→新浪 自动降级；"
                    "东财对本机 IP 会间歇断连，属于正常的时通时断，不必每次都当故障",
                )
            else:
                report.add(group, "日线接口（三源互备）", OK, detail)
        else:
            report.add(
                group,
                "日线接口（三源互备）",
                FAIL,
                "；".join(dead),
                "三个日线源同时不通才是真问题：先查网络/代理/DNS；此时拿不到任何行情",
            )

    for (label, url, marker), probe in other_probes:
        host = url.split("/")[2]
        if not probe.ok:
            report.add(
                group,
                label,
                FAIL,
                f"{host} 不通（{probe.ms:.0f}ms，{probe.note}）",
                "行情/财务数据依赖它，先解决连通性（代理、DNS、防火墙）",
            )
            continue
        note = f"{host} ✓ {probe.status} {probe.ms:.0f}ms"
        if marker:
            text = probe.body.decode("utf-8", errors="ignore")
            if marker in text:
                note += f"，返回体含 `{marker}`（接口语义正确）"
            else:
                report.add(
                    group,
                    label,
                    WARN,
                    note + f"，但返回体未含预期标记 `{marker}`",
                    "接口可能变更或需要不同参数，爬数据前先人工看一眼",
                )
                continue
        report.add(group, label, OK, note)

    with ThreadPoolExecutor(max_workers=4) as pool:
        optional_probes = list(pool.map(lambda item: (item, _probe(item[1], timeout=timeout)), OPTIONAL_ENDPOINTS))
    for (label, url, _marker), probe in optional_probes:
        host = url.split("/")[2]
        if probe.ok:
            report.add(group, label, OK, f"{host} ✓ {probe.status} {probe.ms:.0f}ms")
        else:
            if "全市场列表" in label:
                hint = (
                    "本机连 push2 clist 被对端直接断开（实测 push2 / 82.push2 / push2delay 三个域名都一样）；"
                    "股票全集改用新浪列表接口（见上方数据源检查项），不影响取数"
                )
            elif "GitHub" in label:
                hint = "github.com 时通时断：装插件请走 codeload/npm 通道；行情数据不受影响"
            elif "Tushare" in label:
                hint = (
                    "Tushare 非必需（本工程默认不用）；若报证书过期是对方服务器证书问题，"
                    "装了 tushare 也要等它换证才能用"
                )
            else:
                hint = "仅影响可选通道（插件安装/镜像），行情数据不受影响"
            report.add(group, label, WARN, f"{host} 不通（{probe.note}）", hint)

    # TLS 证书链：证书校验失败意味着被中间人替换，属于安全项
    try:
        context = ssl.create_default_context()
        with socket.create_connection(("push2his.eastmoney.com", 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname="push2his.eastmoney.com") as tls:
                cert = tls.getpeercert()
        subject = dict(x[0] for x in cert.get("subject", ()))
        issuer = dict(x[0] for x in cert.get("issuer", ()))
        report.add(
            group,
            "TLS 证书校验",
            OK,
            f"东财证书有效：subject={subject.get('commonName', '?')}，issuer={issuer.get('organizationName', '?')}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add(
            group,
            "TLS 证书校验",
            FAIL,
            f"{type(exc).__name__}: {exc}",
            "证书链不可信 = 可能存在代理/中间人；下单与行情都可能被篡改，先查清再继续",
        )


def _resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        return sorted({info[4][0] for info in infos})
    except Exception:  # noqa: BLE001
        return []


def check_packages(report: DoctorReport) -> None:
    group = "可选依赖"
    for name, why in OPTIONAL_PACKAGES:
        available = _module_available(name)
        if name == "xtquant":
            report.add(
                group,
                name,
                WARN if not available else OK,
                ("" if available else "未安装") + f"（{why}）",
                "只有真要自动下单才需要；半自动模式（订单文件 + 手工确认）不依赖它"
                if not available
                else "",
            )
            continue
        report.add(
            group,
            name,
            OK if available else WARN,
            ("已安装" if available else "未安装") + f"（{why}）",
            "" if available else "需要时再装：`python -m pip install " + name + "`",
        )


def check_project(report: DoctorReport, *, config_path: Path | None = None) -> None:
    group = "项目与数据"
    config_path = config_path or (ROOT / "config.toml")
    if not config_path.is_file():
        report.add(group, "配置文件", FAIL, f"找不到 {config_path}", "从仓库根目录运行，或指定 --config")
        return
    try:
        from quant.config import load_config

        cfg = load_config(config_path)
        report.add(
            group,
            "配置文件",
            OK,
            f"{config_path.name} 解析成功：{cfg.start}~{cfg.end}，{len(cfg.universe)} 只标的，"
            f"初始资金 {cfg.initial_cash:,.0f} 元",
        )
    except Exception as exc:  # noqa: BLE001
        report.add(group, "配置文件", FAIL, f"{type(exc).__name__}: {exc}", "修配置再跑回测")

    for name, path in (("data", ROOT / "data"), ("results", ROOT / "results")):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            report.add(group, f"{name}/ 可写", OK, str(path))
        except Exception as exc:  # noqa: BLE001
            report.add(group, f"{name}/ 可写", FAIL, f"{type(exc).__name__}: {exc}", "检查权限/占用")

    csv_dir = ROOT / "data" / "csv"
    files = sorted(csv_dir.glob("*.csv")) if csv_dir.is_dir() else []
    if not files:
        report.add(
            group,
            "本地行情数据",
            WARN,
            f"{csv_dir} 下没有 CSV（尚未爬数据）",
            "先跑 `python -m quant fetch --mode synthetic` 生成离线样本；真实数据待爬取",
        )
        return
    total = sum(f.stat().st_size for f in files)
    report.add(group, "本地行情数据", OK, f"{len(files)} 个 CSV，合计 {_fmt_bytes(total)}")
    try:
        import pandas as pd

        sample = pd.read_csv(files[0], nrows=1)
        report.add(group, "CSV 表头", OK, f"{files[0].name}: {', '.join(map(str, sample.columns[:8]))}")
    except Exception as exc:  # noqa: BLE001
        report.add(group, "CSV 表头", WARN, f"读取失败：{exc}")


def check_refdata(report: DoctorReport) -> None:
    """数据湖自检：universe / 行业 / 股本 / 申万 / 估值 / 财务 / 金融股指标 / 指数成分 / 退市股。

    只做**离线**检查（看文件数与关键字段），不联网、不重算，几秒内完成。
    每块给「已落盘多少 + 还缺什么」，缺的部分给 WARN 而不是 FAIL——数据可以边跑边补。
    """
    group = "数据湖"
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        report.add(group, "pandas", FAIL, f"{type(exc).__name__}: {exc}")
        return

    ref = Path("D:/quant-data/ref")
    universe_csv = ref / "universe.csv"
    if universe_csv.is_file():
        try:
            uni = pd.read_csv(universe_csv, dtype={"symbol": str})
            live = int((uni["status"] == 1).sum()) if "status" in uni else -1
            delisted = int(uni["is_delisted"].sum()) if "is_delisted" in uni else -1
            as_of = str(uni["as_of"].dropna().iloc[0])[:10] if "as_of" in uni and uni["as_of"].notna().any() else "?"
            report.add(
                group,
                "universe 区间表",
                OK,
                f"{len(uni)} 只（在市 {live} / 已退市 {delisted}），基准日 {as_of}",
            )
        except Exception as exc:  # noqa: BLE001
            report.add(group, "universe 区间表", FAIL, f"读取失败：{exc}")
    else:
        report.add(group, "universe 区间表", WARN, "缺 ref/universe.csv", "python -m quant lake ref --what universe")

    industry = ref / "industry.parquet"
    if industry.is_file():
        try:
            ind = pd.read_parquet(industry)
            versions = sorted(ind["as_of"].astype(str).unique())
            industry_text = ind["industry"].astype("string").fillna("").str.strip()
            empty = int((industry_text == "").sum())
            report.add(
                group,
                "行业 PIT",
                OK,
                f"{len(ind)} 行 / {len(versions)} 个年度版本（{versions[0][:10]} ~ {versions[-1][:10]}），"
                f"空行业 {empty} 行（退市股为主）",
            )
        except Exception as exc:  # noqa: BLE001
            report.add(group, "行业 PIT", WARN, f"读取失败：{exc}")
    else:
        report.add(group, "行业 PIT", WARN, "缺 ref/industry.parquet", "python -m quant lake ref --what industry")

    shares_csv = ref / "shares.csv"
    if shares_csv.is_file():
        try:
            shares = pd.read_csv(shares_csv, dtype={"symbol": str, "change_date": str})
            real = shares[shares["is_real_code"] == 1] if "is_real_code" in shares else shares
            latest = str(shares["change_date"].max())[:10] if len(shares) else "?"
            per = real.groupby("symbol").size()
            report.add(
                group,
                "股本（PIT）",
                OK,
                f"{len(shares)} 条变动 / {real['symbol'].nunique()} 只，"
                f"每股中位 {float(per.median()):.0f} 条，最新变动日 {latest}",
            )
        except Exception as exc:  # noqa: BLE001
            report.add(group, "股本（PIT）", FAIL, f"读取失败：{exc}")
    else:
        report.add(group, "股本（PIT）", WARN, "缺 ref/shares.csv", "python -m quant lake ref --what shares")

    sw_csv = ref / "sw_industry.csv"
    if sw_csv.is_file():
        try:
            sw = pd.read_csv(sw_csv, dtype={"symbol": str, "sw_code": str})
            as_of = str(sw["as_of"].iloc[0])[:10] if "as_of" in sw and len(sw) else "?"
            report.add(
                group,
                "申万快照",
                OK,
                f"{sw['sw_code'].nunique()} 个一级行业 / {len(sw)} 条成分 / {sw['symbol'].nunique()} 只，落盘 {as_of}（仅当期，无历史）",
            )
        except Exception as exc:  # noqa: BLE001
            report.add(group, "申万快照", WARN, f"读取失败：{exc}")
    else:
        report.add(group, "申万快照", WARN, "缺 ref/sw_industry.csv", "python -m quant lake ref --what sw")

    index_csv = ref / "index_members.csv"
    if index_csv.is_file():
        try:
            members = pd.read_csv(index_csv, dtype={"symbol": str})
            count = members.groupby("index_code").size().to_dict()
            report.add(group, "指数成分", OK, f"{len(members)} 行，{len(count)} 个指数：{count}")
        except Exception as exc:  # noqa: BLE001
            report.add(group, "指数成分", WARN, f"读取失败：{exc}")
    else:
        report.add(group, "指数成分", WARN, "缺 ref/index_members.csv", "python -m quant lake index")

    valuation_dir = Path("D:/quant-data/valuation")
    valuation_files = len(list(valuation_dir.glob("??????.parquet"))) if valuation_dir.is_dir() else 0
    if valuation_files:
        state = OK if valuation_files >= 5000 else WARN
        report.add(
            group,
            "估值/isST 日线",
            state,
            f"{valuation_files} 只已落盘" + ("" if valuation_files >= 5000 else "（未跑完，可断点续跑）"),
            None if valuation_files >= 5000 else "python -m quant lake valuation --shard 0/4",
        )
    else:
        report.add(group, "估值/isST 日线", WARN, "尚无估值数据", "python -m quant lake valuation")

    fund_dir = Path("D:/quant-data/fundamentals")
    periods = {t: len(list((fund_dir / t).glob("????-??-??.parquet"))) for t in ("profit", "balance", "cashflow", "indicator")}
    latest_name = ""
    latest_rows = 0
    try:
        files = sorted((fund_dir / "profit").glob("????-??-??.parquet"))
        complete = [p for p in files if len(pd.read_parquet(p, columns=["SECURITY_CODE"])) > 1000]
        picked = complete[-1] if complete else files[-1]
        latest_name = picked.stem
        latest_rows = len(pd.read_parquet(picked))
    except Exception:  # noqa: BLE001
        pass
    detail = " / ".join(f"{k} {v} 期" for k, v in periods.items())
    if all(v >= 50 for v in periods.values()):
        report.add(group, "财务三表", OK, f"{detail}；最近完整报告期 {latest_name} 有 {latest_rows} 家公司")
    elif any(periods.values()):
        report.add(group, "财务三表", WARN, f"{detail}（期数不足，可断点续跑）", "python -m quant lake fund")
    else:
        report.add(group, "财务三表", WARN, "尚无财务数据", "python -m quant lake fund")

    mainfin_dir = fund_dir / "mainfin"
    mainfin_files = sorted(mainfin_dir.glob("????-??-??.parquet")) if mainfin_dir.is_dir() else []
    if mainfin_files:
        try:
            # 只看完整披露的期（2026-09-30 这类当期只有几十家，不代表覆盖率）
            complete = [p for p in mainfin_files if len(pd.read_parquet(p, columns=["SECURITY_CODE"])) > 1000]
            last = complete[-1] if complete else mainfin_files[-1]
            frame = pd.read_parquet(last)
            fin = frame[frame["ORG_TYPE"].astype(str).isin(["银行", "保险", "证券"])]
            roe = int(pd.to_numeric(fin["ROEJQ"], errors="coerce").notna().sum()) if "ROEJQ" in fin else -1
            report.add(
                group,
                "主要财务指标（金融股）",
                OK,
                f"{len(mainfin_files)} 期，最近一期 {last.stem} 金融业 {len(fin)} 家、ROE 非空 {roe} 家",
            )
        except Exception as exc:  # noqa: BLE001
            report.add(group, "主要财务指标（金融股）", WARN, f"读取失败：{exc}")
    else:
        report.add(group, "主要财务指标（金融股）", WARN, "缺 fundamentals/mainfin", "python -m quant mainfin")

    delisted_events = Path("D:/quant-data/delisted/events.csv")
    if delisted_events.is_file():
        try:
            events = pd.read_csv(delisted_events, dtype={"symbol": str})
            have = int((events["bars"] > 0).sum()) if "bars" in events else -1
            report.add(group, "退市股日线", OK, f"{len(events)} 只退市股，其中有日线 {have} 只")
        except Exception as exc:  # noqa: BLE001
            report.add(group, "退市股日线", WARN, f"读取失败：{exc}")
    else:
        report.add(group, "退市股日线", WARN, "缺 delisted/events.csv", "python -m quant lake delisted")

    _check_events(report, group)
    _check_holdout(report, group)


def _check_holdout(report: DoctorReport, group: str) -> None:
    """holdout 层（2025-01-01 起）：只统计存在性与行数，不读内容、不参与任何样本内计算。"""
    try:
        import pandas as pd

        from quant import dividends, events as ev, valuation
    except Exception as exc:  # noqa: BLE001
        report.add(group, "holdout 层", WARN, f"导入失败：{exc}")
        return
    root = Path("D:/quant-data/holdout")
    try:
        parts = []
        prices = sorted((root / "parquet").glob("**/*.parquet"))
        if prices:
            # 只读 footer，避免把 200 万行读进内存
            import pyarrow.parquet as pq

            rows = sum(pq.ParquetFile(path).metadata.num_rows for path in prices)
            parts.append(f"价量 {len(prices)} 分区/{rows:,} 行")
        files = sorted(valuation.HOLDOUT_DIR.glob("??????.parquet"))
        if files:
            rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
            parts.append(f"估值 {len(files):,} 只/{rows:,} 行")
        if (root / "turn_panel.parquet").is_file():
            parts.append(f"换手 {pq.ParquetFile(root / 'turn_panel.parquet').metadata.num_rows:,} 行")
        if dividends.HOLDOUT_DIVIDENDS.is_file():
            parts.append(f"分红 {pq.ParquetFile(dividends.HOLDOUT_DIVIDENDS).metadata.num_rows:,} 行")
        if (ev.HOLDOUT_DIR / "st.parquet").is_file():
            parts.append(f"ST 翻转 {pq.ParquetFile(ev.HOLDOUT_DIR / 'st.parquet').metadata.num_rows:,} 行")
        if not parts:
            report.add(group, "holdout 层", WARN, "未落盘（最终验收前需补齐）", "python -m quant lake valuation --holdout --shard 0/4")
            return
        detail = "、".join(parts) + "；仅最终一次性验收读取"
        if len(files) < 5000:
            report.add(group, "holdout 层", WARN, f"{detail}（估值不足 5,000 只，验收前需续跑）")
        else:
            report.add(group, "holdout 层", OK, detail)
    except Exception as exc:  # noqa: BLE001
        report.add(group, "holdout 层", WARN, f"读取失败：{exc}")


def _check_events(report: DoctorReport, group: str) -> None:
    """事件层：六张表是否存在、行数、PIT 可见日是否齐全、是否有 2025+ 行。"""
    try:
        import pandas as pd

        from quant import events as ev
    except Exception as exc:  # noqa: BLE001
        report.add(group, "事件层", WARN, f"导入失败：{exc}")
        return
    missing = [name for name in ev.TABLES if not ev.FILES[name].is_file()]
    if missing:
        report.add(group, "事件层", WARN, f"缺 {len(missing)} 张表：{', '.join(missing)}", "python -m quant lake events")
        return
    try:
        parts = []
        total = 0
        pit_missing = 0
        holdout_rows = 0
        for name in ev.TABLES:
            frame = pd.read_parquet(ev.FILES[name], columns=["event_date", "available_date"])
            total += len(frame)
            parts.append(f"{name} {len(frame):,}")
            pit_missing += int(frame["available_date"].isna().sum())
            holdout_rows += int(
                (pd.to_datetime(frame["available_date"], errors="coerce") >= ev.HOLDOUT_START).sum()
            )
        detail = "、".join(parts) + f"；合计 {total:,} 行，其中 2025+ 可见 {holdout_rows:,} 行（研究默认裁掉）"
        if pit_missing > 100:
            report.add(group, "事件层", WARN, f"{detail}；{pit_missing} 行缺 PIT 可见日", "python -m quant lake events")
        elif pit_missing:
            # 已知：增减持的可见日 = 变动日 +2 交易日，落在日历末端的事件推不出来（已写进偏差声明）
            report.add(group, "事件层", OK, f"{detail}；{pit_missing} 行在交易日历末端无可见日（已声明）")
        else:
            report.add(group, "事件层", OK, detail)
    except Exception as exc:  # noqa: BLE001
        report.add(group, "事件层", WARN, f"读取失败：{exc}")


def check_engine_smoke(report: DoctorReport) -> None:
    """引擎冒烟：真跑一小段合成数据回测，验证 pandas/numpy API 在当前版本下真的可用。"""
    group = "引擎冒烟"
    try:
        from quant import plugins
        from quant.backtest.engine import run_backtest
        from quant.config import BacktestConfig
        from quant.data.store import DataStore
        from quant.data.synthetic import SyntheticSource
    except Exception as exc:  # noqa: BLE001
        report.add(group, "导入引擎", FAIL, f"{type(exc).__name__}: {exc}", "先修导入错误")
        return
    report.add(group, "导入引擎", OK, "quant 各层模块导入正常")

    try:
        symbols = ["600519.SH", "000001.SZ", "300750.SZ"]
        source = SyntheticSource(symbols, start="2023-01-01", periods=300, seed=11)
        store = DataStore(source)
        bars = store.get_many(symbols)
        cfg = BacktestConfig(
            start="2023-01-01", end="2024-06-01", initial_cash=300_000.0, universe=tuple(symbols)
        )
        strategy = plugins.build_strategy("ma_cross", fast=5, slow=20)
        start = time.perf_counter()
        result = run_backtest(bars, strategy, cfg)
        elapsed = time.perf_counter() - start
        last_equity = float(result.equity.iloc[-1])
        metrics = result.metrics or {}
        report.add(
            group,
            "最小回测",
            OK if last_equity > 0 else FAIL,
            f"{len(bars)} 只标的 / {len(result.equity)} 个交易日 / {len(result.trades)} 笔成交，"
            f"期末权益 {last_equity:,.0f} 元，用时 {elapsed*1000:.0f}ms",
        )
        if metrics:
            report.add(
                group,
                "绩效指标",
                OK,
                f"累计 {metrics.get('total_return', 0)*100:.2f}% · 夏普 {metrics.get('sharpe', 0):.2f} · "
                f"最大回撤 {metrics.get('max_drawdown', 0)*100:.2f}%",
            )
    except Exception as exc:  # noqa: BLE001
        import traceback

        report.add(
            group,
            "最小回测",
            FAIL,
            f"{type(exc).__name__}: {exc}",
            "冒烟失败说明当前依赖版本与引擎不兼容；先跑 `python -m quant selftest` 看细节："
            + traceback.format_exc(limit=1).strip().splitlines()[-1],
        )


def check_gateway(report: DoctorReport, *, deep: bool, config_path: Path | None = None) -> None:
    group = "下单通道"
    config_path = config_path or (ROOT / "config.toml")
    gateway_name = "csv"
    dry_run = True
    try:
        from quant.gateway import load_gateway_config

        gateway_cfg = load_gateway_config(config_path)
        gateway_name = gateway_cfg.name
        dry_run = gateway_cfg.dry_run
    except Exception as exc:  # noqa: BLE001
        report.add(
            group,
            "通道配置",
            WARN,
            f"读不到 [gateway] 段（{type(exc).__name__}），按默认半自动模式判断",
        )

    report.add(
        group,
        "当前通道",
        OK,
        f"{gateway_name}（dry-run={'开' if dry_run else '关'}）"
        + ("，半自动：只产出订单文件，由你手工下单" if gateway_name == "csv" else ""),
        "" if dry_run else "dry-run 关闭 = 会真下单，务必确认限额与白名单",
    )

    orders_dir = ROOT / "data" / "orders"
    try:
        orders_dir.mkdir(parents=True, exist_ok=True)
        probe = orders_dir / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        today_orders = orders_dir / f"{datetime.now(SHANGHAI):%Y-%m-%d}-orders.jsonl"
        pending = list(orders_dir.glob("*-plan.json"))
        report.add(
            group,
            "订单目录",
            OK,
            f"{orders_dir} 可写；历史计划 {len(pending)} 份，"
            f"今日订单文件{'已存在' if today_orders.exists() else '尚未生成'}",
            "" if not today_orders.exists() else "今日已有订单文件：提交前先跑对账，避免重复下单",
        )
    except Exception as exc:  # noqa: BLE001
        report.add(group, "订单目录", FAIL, f"{type(exc).__name__}: {exc}", "下单前必须保证可写")

    kill = ROOT / "data" / "KILL_SWITCH"
    report.add(
        group,
        "急停开关",
        WARN if kill.exists() else OK,
        "存在 data/KILL_SWITCH：所有下单会被拒绝" if kill.exists() else "未触发（正常）",
        "确认无需急停后删除该文件" if kill.exists() else "",
    )

    report.add(
        group,
        "半自动前置条件",
        OK,
        "订单文件 + 人工确认 + 回报录入（不依赖券商 SDK、不依赖本机券商客户端）",
    )

    if not deep:
        return
    exe_names = ("XtMiniQmt.exe", "XtItClient.exe", "xtquant.exe")
    found: list[str] = []
    for exe in exe_names:
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {exe}"], capture_output=True, text=True, timeout=10
            ).stdout
            if exe.lower() in out.lower():
                found.append(exe)
        except Exception:  # noqa: BLE001
            pass
    report.add(
        group,
        "QMT 客户端进程",
        OK if found else WARN,
        ("检测到：" + ", ".join(found)) if found else "未检测到 QMT/迅投进程",
        "" if found else "走 QMT 自动下单时才需要常驻客户端；当前半自动模式不需要",
    )


def check_security(report: DoctorReport) -> None:
    group = "安全基线"
    from quant.gateway import DEFAULT_LIMITS  # 延迟导入，避免循环

    report.add(
        group,
        "下单默认值",
        OK,
        f"dry-run 默认开；限额：单票 ≤{DEFAULT_LIMITS['max_weight_per_symbol']:.0%}、"
        f"总仓 ≤{DEFAULT_LIMITS['max_gross_exposure']:.0%}、"
        f"单笔 ≤{DEFAULT_LIMITS['max_single_order_value']:,.0f} 元、"
        f"日订单数 ≤{DEFAULT_LIMITS['max_orders_per_day']}",
    )
    report.add(
        group,
        "凭证扫描",
        OK,
        "工程内不存券商账号/密码；下单靠券商客户端或人工操作，配置里只放路径与限额",
    )
    secrets = [
        p.name
        for p in (ROOT / ".env", ROOT / "secrets.toml", ROOT / "token.json")
        if p.exists()
    ]
    report.add(
        group,
        "明文凭证文件",
        WARN if secrets else OK,
        f"发现 {', '.join(secrets)}" if secrets else "未发现 .env / secrets.toml / token.json",
        "凭证文件请勿提交到版本库，且应限制权限（icacls）" if secrets else "",
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_doctor(
    *,
    network: bool = True,
    deep: bool = False,
    timeout: float = 6.0,
    config_path: str | Path | None = None,
) -> DoctorReport:
    report = DoctorReport(started_at=datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S"))
    start = time.perf_counter()
    cfg = Path(config_path) if config_path else None

    check_runtime(report)
    check_hardware(report)
    check_clock(report, network=network)
    check_network(report, network=network, timeout=timeout)
    check_packages(report)
    check_project(report, config_path=cfg)
    check_refdata(report)
    check_engine_smoke(report)
    check_gateway(report, deep=deep, config_path=cfg)
    check_security(report)

    report.elapsed = time.perf_counter() - start
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m quant doctor", description="环境可靠性自检（依赖/网络/时钟/通道/安全基线）"
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--no-network", action="store_true", help="跳过联网探测（离线模式）")
    parser.add_argument("--deep", action="store_true", help="额外检查券商客户端等外部进程")
    parser.add_argument("--timeout", type=float, default=6.0, help="单个网络探测超时秒数（默认 6）")
    parser.add_argument("--config", default=str(ROOT / "config.toml"), help="配置文件路径")
    parser.add_argument("--out", default=str(ROOT / "results"), help="报告输出目录")
    parser.add_argument("--no-write", action="store_true", help="不落盘报告")
    parser.add_argument("--fail-on", choices=("fail", "warn"), default="fail", help="按哪一级返回非零退出码")
    args = parser.parse_args(argv)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    report = run_doctor(
        network=not args.no_network,
        deep=args.deep,
        timeout=args.timeout,
        config_path=args.config,
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.to_text())
        if not args.no_write:
            json_path, md_path = report.write(args.out)
            print(f"\n报告已写入：{json_path}\n              {md_path}")

    if report.failures:
        return 1
    if args.fail_on == "warn" and report.warnings:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
