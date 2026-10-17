"""并行跑估值快照：把 universe 切成 N 片，各起一个子进程（各自 baostock 会话）。

用法：
    python scripts/run_valuation_shards.py --workers 4 --start 2014-01-01

每片日志写 ``results/valuation_shard{i}.log``，落盘在 ``D:/quant-data/valuation/<代码>.parquet``。
文件级 resume：片子进程崩了直接重跑同一条命令即可，已落盘的会跳过。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--pause", type=float, default=0.05)
    args = ap.parse_args()

    logs = ROOT / "results"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    t0 = time.perf_counter()
    for i in range(args.workers):
        log = logs / f"valuation_shard{i}.log"
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            "-u",
            "-m",
            "quant",
            "lake",
            "valuation",
            "--shard",
            f"{i}/{args.workers}",
            "--start",
            args.start,
            "--pause",
            str(args.pause),
        ]
        if args.end:
            cmd += ["--end", args.end]
        handle = log.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        procs.append((i, proc, handle, log))
        print(f"shard {i}/{args.workers} pid={proc.pid} → {log}", flush=True)
        time.sleep(1.0)

    failed = 0
    for i, proc, handle, log in procs:
        code = proc.wait()
        handle.close()
        tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-4:])
        print(f"shard {i} exit={code} 用时 {time.perf_counter() - t0:.0f}s\n{tail}\n", flush=True)
        if code != 0:
            failed += 1
    print(f"全部结束：{len(procs) - failed}/{len(procs)} 片成功，总用时 {time.perf_counter() - t0:.0f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
