"""一键跑全部测试（不需要 pytest）::

    python tests/run_all.py

退出码 0 = 全绿；非 0 = 有失败（会打印失败的测试名与异常）。
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import setup_console  # noqa: E402

setup_console()

import test_actions  # noqa: E402
import test_backtest  # noqa: E402
import test_data  # noqa: E402
import test_events  # noqa: E402
import test_factors  # noqa: E402
import test_gateway  # noqa: E402
import test_growth  # noqa: E402
import test_growth_backtest  # noqa: E402
import test_guard  # noqa: E402
import test_holdout  # noqa: E402
import test_indicators  # noqa: E402
import test_lake  # noqa: E402
import test_methodology  # noqa: E402
import test_parquet  # noqa: E402
import test_refdata  # noqa: E402
import test_regime  # noqa: E402
import test_plugins  # noqa: E402
import test_web_source  # noqa: E402

SUITES = [
    ("数据层", test_data),
    ("指标", test_indicators),
    ("回测/风控/报表", test_backtest),
    ("插件/CLI", test_plugins),
    ("多因子", test_factors),
    ("成长因子", test_growth),
    ("成长组合构造", test_growth_backtest),
    ("方法论", test_methodology),
    ("订单护栏/账本", test_guard),
    ("下单通道", test_gateway),
    ("联网行情源", test_web_source),
    ("数据湖", test_lake),
    ("Parquet 存储", test_parquet),
    ("参考数据", test_refdata),
    ("事件层", test_events),
    ("holdout 隔离", test_holdout),
    ("危机识别", test_regime),
    ("复权/停牌", test_actions),
]


def suite_key(module) -> str:
    """稳定、可在 shell 里用的套件键（test_refdata -> refdata）。"""
    return module.__name__.removeprefix("test_")


#: 需要真实数据湖（D:/quant-data）或外网的套件；干净机器上应跳过。
#: factors 里有「每个已注册因子都能算出来」的断言，mcap/fmcap 要读 ref/shares.csv。
LAKE_SUITES = frozenset({"factors", "refdata", "events", "holdout", "web_source"})


def select(argv: list[str]) -> tuple[list[tuple[str, object]], int]:
    """按命令行参数挑套件；返回 (套件列表, 被跳过的数量)。"""
    only = None
    for i, arg in enumerate(argv):
        if arg == "--suite" and i + 1 < len(argv):
            only = argv[i + 1]
    skip_lake = "--skip-lake" in argv
    if "--list" in argv:
        for name, module in SUITES:
            key = suite_key(module)
            flag = "lake" if key in LAKE_SUITES else "offline"
            print(f"{key}\t{flag}\t{name}")
        raise SystemExit(0)
    picked: list[tuple[str, object]] = []
    skipped = 0
    for name, module in SUITES:
        key = suite_key(module)
        if only is not None and only not in (key, name):
            continue
        if skip_lake and key in LAKE_SUITES:
            skipped += 1
            continue
        picked.append((name, module))
    if only is not None and not picked:
        raise SystemExit(f"没有匹配的套件：{only}（--list 看可用键）")
    return picked, skipped


def main(argv: list[str] | None = None) -> int:
    suites, skipped = select(list(sys.argv[1:] if argv is None else argv))
    if skipped:
        print(f"(跳过 {skipped} 个需要数据湖/外网的套件)")
    failed: list[tuple[str, str, str]] = []
    passed = 0
    started = time.time()
    for suite_name, module in suites:
        tests = [
            (name, obj)
            for name, obj in vars(module).items()
            if name.startswith("test_") and callable(obj)
        ]
        tests.sort(key=lambda kv: getattr(kv[1], "__code__").co_firstlineno)
        print(f"\n=== {suite_name}（{len(tests)} 项）===")
        for name, fn in tests:
            try:
                fn()
                passed += 1
            except Exception:
                failed.append((suite_name, name, traceback.format_exc()))
                print(f"  ✗ {name} 失败")
    elapsed = time.time() - started
    print("\n" + "=" * 62)
    if failed:
        print(f"结果：{passed} 通过 / {len(failed)} 失败，用时 {elapsed:.1f}s")
        for suite_name, name, tb in failed:
            print(f"\n----- {suite_name} :: {name} -----\n{tb}")
        return 1
    print(f"结果：全部 {passed} 项通过，用时 {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
