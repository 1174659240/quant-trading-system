"""行情落盘 / 生成示例数据。

用法::

    # 1) 生成离线示例 CSV（不联网，任何机器都能跑通全流程）
    python -m scripts.fetch_data --mode synthetic --symbols 600519.SH,000001.SZ --out data/csv

    # 2) 联网抓真实 A 股日线（东财 → 腾讯 → 新浪自动降级，推荐）
    python -m scripts.fetch_data --mode web --symbols 600519.SH,000001.SZ,000300.SH --start 2018-01-01

    # 3) 用 akshare 拉真实日线（需 pip install akshare）
    python -m scripts.fetch_data --mode akshare --symbols 600519,000001 --start 2018-01-01

    # 4) 只看全市场代码表（新浪列表接口），写到 data/universe.csv
    python -m scripts.fetch_data --mode list --limit 500

    # 5) 只体检已有 CSV（列名、缺口、异常涨跌幅）
    python -m scripts.fetch_data --mode check --out data/csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.data import CsvSource, DataError, DataStore, make_gbm_frame


def fetch_synthetic(symbols: list[str], out: Path, start: str, periods: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i, symbol in enumerate(symbols):
        frame = make_gbm_frame(symbol, start=start, periods=periods, seed=seed + i)
        path = out / f"{symbol}.csv"
        frame.to_csv(path, index=True, index_label="date", encoding="utf-8-sig")
        print(f"已生成 {path}  {len(frame)} 行  {frame.index.min().date()}~{frame.index.max().date()}")


def fetch_akshare(symbols: list[str], out: Path, start: str, end: str | None, adjust: str) -> None:
    from quant.data.akshare_source import AkshareSource

    try:
        source = AkshareSource(adjust=adjust)
    except DataError as exc:
        raise SystemExit(str(exc)) from exc
    out.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        frame = source.load(symbol, start, end)
        path = out / f"{symbol}.csv"
        frame.to_csv(path, index=True, index_label="date", encoding="utf-8-sig")
        print(f"已保存 {path}  {len(frame)} 行  {frame.index.min().date()}~{frame.index.max().date()}")


def fetch_web(
    symbols: list[str],
    out: Path,
    start: str,
    end: str | None,
    adjust: str,
    source: str,
    *,
    retries: int = 2,
) -> int:
    """联网抓日线并落盘；单只失败不中断整批，返回失败数量。"""
    from quant.data.web_source import fetch_many

    out.mkdir(parents=True, exist_ok=True)
    print(f"抓取 {len(symbols)} 只 · 区间 {start}~{end or '最新'} · 复权 {adjust} · 源 {source}")
    frames, used, failures = fetch_many(
        symbols, start=start, end=end, adjust=adjust, source=source, retries=retries
    )
    for symbol, frame in frames.items():
        path = out / f"{symbol}.csv"
        frame.to_csv(path, index=True, index_label="date", encoding="utf-8-sig")
    if failures:
        print("\n以下标的未取到数据（其余已保存）：")
        for item in failures:
            print(f"  ! {item}")
    sources: dict[str, int] = {}
    for name in used.values():
        sources[name] = sources.get(name, 0) + 1
    if sources:
        print("\n各源命中：" + " · ".join(f"{k}={v}" for k, v in sorted(sources.items())))
    return len(failures)


def fetch_list(limit: int | None, out_file: Path) -> int:
    """把全市场 A 股代码写到 CSV（新浪列表接口）。"""
    from quant.data.web_source import list_symbols

    symbols = list_symbols(limit)
    if not symbols:
        raise SystemExit("列表接口没有返回任何代码（网络问题？）")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols}).to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"已写入 {out_file}，共 {len(symbols)} 只 A 股")
    return 0


def check(out: Path) -> int:
    if not out.exists():
        raise SystemExit(f"目录不存在: {out}")
    source = CsvSource(out)
    symbols = source.symbols()
    if not symbols:
        raise SystemExit(f"{out} 下没有 CSV")
    store = DataStore(source)
    bars = store.get_many(symbols)
    desc = store.describe(list(bars))
    print(desc.to_string(index=False))
    problems = 0
    for symbol, frame in bars.items():
        ret = frame["close"].pct_change()
        jumps = ret[ret.abs() > 0.11]
        if len(jumps):
            problems += 1
            print(f"  ! {symbol} 有 {len(jumps)} 天涨跌幅超过 11%（未复权/停牌/数据错误？）：{list(jumps.index[:3])}")
        gap = frame.index.to_series().diff().dt.days.dropna()
        if len(gap) and gap.max() > 30:
            print(f"  ~ {symbol} 最长数据空档 {int(gap.max())} 天（可能是停牌或缺失）")
    print("体检完成" + (f"，发现 {problems} 个需要人工确认的标的" if problems else "，未发现硬伤"))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="行情数据准备")
    p.add_argument("--mode", default="synthetic", choices=["synthetic", "web", "akshare", "list", "check"])
    p.add_argument("--symbols", default="600519.SH,000001.SZ,300750.SZ,601318.SH")
    p.add_argument("--out", default="data/csv")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--periods", type=int, default=1500, help="合成数据的交易日数量")
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", "none"], help="复权方式（web 用）")
    p.add_argument("--source", default="auto", choices=["auto", "eastmoney", "tencent", "sina"], help="web 优先源")
    p.add_argument("--limit", type=int, default=None, help="list 模式：只取前 N 只")
    p.add_argument("--universe-file", default="data/universe.csv", help="list 模式的输出文件")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    failures = 0
    if args.mode == "synthetic":
        fetch_synthetic(symbols, out, args.start, args.periods, args.seed)
    elif args.mode == "web":
        failures = fetch_web(symbols, out, args.start, args.end, args.adjust, args.source)
    elif args.mode == "akshare":
        fetch_akshare(symbols, out, args.start, args.end, args.adjust)
    elif args.mode == "list":
        return fetch_list(args.limit, Path(args.universe_file))
    else:
        return check(out)
    print(f"\n下一步：python -m scripts.run_backtest --source csv --csv-dir {out} --strategy ma_cross")
    return 1 if failures and failures == len(symbols) else 0


if __name__ == "__main__":
    raise SystemExit(main())
