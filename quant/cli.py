"""统一命令行入口：``python -m quant <子命令>``

子命令（参数与各脚本完全一致，这里只做分发）::

    python -m quant backtest [--strategy ...] [--source ...] ...
    python -m quant fetch    --mode synthetic|akshare|check
    python -m quant optimize --strategy ma_cross --grid "fast=5,10 slow=30,60"
    python -m quant paper    plan|apply|status
    python -m quant trade    plan|confirm|record|status|reconcile|limits   # 半自动实盘（默认只出订单文件）
    python -m quant doctor   [--deep] [--no-network] [--json]             # 环境可靠性自检
    python -m quant lake     init|universe|crawl|split|process|status|verify  # 本地数据湖
    python -m quant plugins  [--json]        # 看已注册的策略/数据源/风控与插件加载情况
    python -m quant selftest                 # 跑工程自带全部离线测试

``python -m quant`` 与 ``python -m scripts.<脚本>`` 等价；用 CLI 的好处是统一入口，
并会在启动时自动加载 ``plugins/`` 目录里的自定义扩展。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["SUBCOMMANDS", "entry", "main"]

ROOT = Path(__file__).resolve().parents[1]

SUBCOMMANDS = (
    "backtest",
    "fetch",
    "factor",
    "factor-market",
    "optimize",
    "paper",
    "trade",
    "doctor",
    "lake",
    "plugins",
    "selftest",
    "probe-delisted",
)

_HELP = {
    "backtest": "回测（单策略或 --compare 多策略对比）",
    "fetch": "取数 / 生成合成数据 / 数据体检",
    "factor": "因子研究报告：IC / 分层收益 / 因子相关性",
    "factor-market": "全市场因子研究 + 多因子回测（只用 processed 层）",
    "optimize": "网格搜索 + 样本内/样本外验证",
    "paper": "纸上交易：plan / apply / status",
    "trade": "半自动实盘：plan / confirm / record / status / reconcile / limits",
    "doctor": "环境可靠性自检（运行时/数据源/时钟/通道/护栏）",
    "lake": "本地数据湖：全市场原始落地 / 精简层 / 最终校核集隔离",
    "plugins": "列出已注册的策略、数据源、风控 overlay 与插件文件",
    "selftest": "跑 tests/run_all.py 的全部离线测试",
    "probe-delisted": "退市股/ST 改名 免费源探针（一次性）",
}


def _ensure_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _cmd_plugins(argv: list[str]) -> int:
    from quant import plugins

    parser = argparse.ArgumentParser(prog="python -m quant plugins", description=_HELP["plugins"])
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)

    plugins.load_plugins_from_config_file(ROOT / "config.toml", root=ROOT)
    info = plugins.describe()
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    for kind, label in (("strategies", "策略"), ("sources", "数据源"), ("overlays", "风控 overlay")):
        section = info[kind]
        print(f"\n【{label}】内置 {len(section['builtin'])} 个 / 自定义 {len(section['custom'])} 个")
        print("  内置：" + ", ".join(section["builtin"]))
        if section["custom"]:
            print("  自定义：" + ", ".join(section["custom"]))
    print(f"\n【插件文件】已加载 {len(info['loadedFiles'])} 个")
    for path in info["loadedFiles"]:
        print(f"  {path}")
    if info["errors"]:
        print(f"\n【加载失败】{len(info['errors'])} 个")
        for err in info["errors"]:
            last = err["error"].strip().splitlines()[-1]
            print(f"  {err['path']} -> {last}")
    print("\n新增扩展：把模块放进 plugins/ 目录（或用 config.toml 的 [plugins] 段指定路径），"
          "用 @strategy / @source / @overlay 注册即可被命令行按名字引用。")
    return 0


def _cmd_selftest(argv: list[str]) -> int:
    import runpy

    script = ROOT / "tests" / "run_all.py"
    if not script.is_file():
        print(f"找不到测试入口：{script}")
        return 1
    namespace = runpy.run_path(str(script))
    return int(namespace["main"]())


def _dispatch(name: str, argv: list[str]) -> int:
    if name == "plugins":
        return _cmd_plugins(argv)
    if name == "selftest":
        return _cmd_selftest(argv)
    if name == "doctor":
        from quant.doctor import main as doctor_main

        return int(doctor_main(argv))
    if name == "lake":
        from quant.lake import main as lake_main

        return int(lake_main(argv))
    if name == "backtest":
        from scripts import run_backtest as module
    elif name == "fetch":
        from scripts import fetch_data as module
    elif name == "factor":
        from scripts import factors as module
    elif name == "factor-market":
        from scripts import factor_market as module
    elif name == "optimize":
        from scripts import optimize as module
    elif name == "paper":
        from scripts import paper_trade as module
    elif name == "trade":
        from scripts import trade as module
    elif name == "probe-delisted":
        from scripts import probe_delisted as module
    else:  # pragma: no cover - argparse 已挡住
        raise SystemExit(f"未知子命令：{name}")
    return int(module.main(argv))


def main(argv: list[str] | None = None) -> int:
    _ensure_path()
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("子命令：")
        for name in SUBCOMMANDS:
            print(f"  {name:<9} {_HELP[name]}")
        return 0 if args else 1
    if args[0] in ("-v", "--version"):
        from quant import __version__

        print(f"ashare-quant {__version__}")
        return 0

    name, rest = args[0], args[1:]
    if name not in SUBCOMMANDS:
        print(f"未知子命令：{name}\n可选：{', '.join(SUBCOMMANDS)}")
        return 1
    return _dispatch(name, rest)


def entry() -> None:
    """``pyproject.toml`` 里 console_scripts 的入口。"""
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
