"""阶段 1：抓 holdout（2025-01-01 ~ 2026-10-16）估值序列，落独立目录 ``D:/quant-data/holdout/valuation``。

与 ``scripts/run_valuation_shards.py`` 同构（每片一个子进程、各自 baostock 会话），
但写 holdout 目录 → 样本内 valuation 文件的 sha 不变。

用法：``python -m scripts.run_valuation_holdout --workers 4 --start 2025-01-01 --end 2026-10-16``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results"
HOLDOUT = Path("D:/quant-data/holdout/valuation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-10-16")
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT.mkdir(parents=True, exist_ok=True)
    procs = []
    started = time.time()
    for i in range(args.workers):
        log = LOG_DIR / f"holdout-valuation_shard{i}.log"
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "quant",
            "lake",
            "valuation",
            "--start",
            args.start,
            "--end",
            args.end,
            "--pause",
            str(args.pause),
            "--shard",
            f"{i}/{args.workers}",
            "--holdout",
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        handle = log.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
        procs.append((proc, handle, log))
        print(f"启动分片 {i}/{args.workers} → {log}", flush=True)

    code = 0
    for i, (proc, handle, log) in enumerate(procs):
        proc.wait()
        handle.close()
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-1:] if log.is_file() else []
        print(f"分片 {i} 退出码 {proc.returncode} {tail}", flush=True)
        code = code or proc.returncode
    print(f"全部完成 {time.time() - started:.0f}s → {HOLDOUT}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
