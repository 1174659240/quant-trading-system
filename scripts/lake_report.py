import sys

from pathlib import Path

import pandas as pd

sys.path.insert(0, r"D:\workspace\量化交易系统")
from quant.lake import Lake  # noqa: E402

root = Path(r"D:\quant-data")
lake = Lake(root=root)


def scan(folder: Path):
    files = sorted(folder.glob("*.csv"))
    rows, first, last = 0, None, None
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=["date"])
        if len(frame):
            lo, hi = frame["date"].min(), frame["date"].max()
            first = lo if first is None or lo < first else first
            last = hi if last is None or hi > last else last
            rows += len(frame)
    return len(files), rows, first, last


def size(folder: Path) -> str:
    total = sum(p.stat().st_size for p in folder.glob("*.csv"))
    return f"{total / 1024 / 1024:.0f} MB"


u = lake.read_universe()
lines = []
for name, folder in (("raw", root / "raw" / "daily"), ("processed", root / "processed" / "daily"), ("holdout", root / "holdout" / "daily")):
    n, rows, first, last = scan(folder)
    lines.append(f"| {name} | {n} | {rows:,} | {first} → {last} | {size(folder)} |")

missing = [c for c in u if not lake.raw_path(c).exists()]
lock = (root / "holdout" / ".LOCKED").exists()
processed_missing = len(u) - len(list((root / "processed" / "daily").glob("*.csv")))

sample = []
for code in ("600519", "000001", "300750", "688239"):
    for layer in ("raw", "processed", "holdout"):
        path = root / layer / "daily" / f"{code}.csv"
        if not path.exists():
            sample.append((code, layer, "-", "-", "-"))
            continue
        f = pd.read_csv(path, encoding="utf-8-sig")
        sample.append((code, layer, len(f), f["date"].min(), f["date"].max()))

report = [
    "# 数据湖落地汇总（全市场日线）",
    "",
    f"- 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}",
    f"- 股票池：{len(u)} 只（沪深主板/创业板/科创板/北交所，`meta/universe.csv`）",
    "- 原始层 = 源站不复权原样留档；processed/holdout = 同一套口径（前复权 + 停牌补齐 + 未复权原值列），"
    "holdout 恒从原始层重算，只存 2025-01-01 起的冻结样本。",
    "",
    "| 层 | 文件数 | 行数 | 日期覆盖 | 体积 |",
    "|---|---|---|---|---|",
    *lines,
    "",
    f"- 原始层缺数据：{len(missing)} 只{('（' + '、'.join(missing) + '）') if missing else ''}；"
    f"processed 比股票池少 {processed_missing} 只（训练窗口 <120 根的新股，属预期）。",
    f"- 校核集锁：{'已生成 `.LOCKED`' if lock else '缺失'}；默认读取会抛 PermissionError。",
    "",
    "## 抽样核对",
    "",
    "| 代码 | 层 | 行数 | 起 | 止 |",
    "|---|---|---|---|---|",
    *[f"| {c} | {l} | {n} | {a} | {b} |" for c, l, n, a, b in sample],
    "",
]
out = Path(r"D:\workspace\量化交易系统\results\lake-summary.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))
