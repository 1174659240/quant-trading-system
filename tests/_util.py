"""测试公共设施：工作区内的临时目录 + 控制台 UTF-8 输出。

两个 Windows 坑，都在这里一次性处理掉：

1. 控制台编码可能是 GBK，打印 ``✓`` 之类字符会直接抛 UnicodeEncodeError；
2. ``tempfile.mkdtemp`` 会把目录权限收紧成「仅创建者可用」并**切断继承**，
   在 DSH 的受限进程里后续写入会被拒（PermissionError / WinError 5）。
   所以这里自己用 ``mkdir()`` 建目录（正常继承父目录权限），并放在项目内的
   ``_tmp/``，跑完即删，不碰系统临时目录。
"""

from __future__ import annotations

import itertools
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / "_tmp"
_counter = itertools.count(1)


def setup_console() -> None:
    """把 stdout/stderr 切到 UTF-8；失败就算了，不影响测试结果。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


@contextmanager
def tmpdir(name: str = "test") -> Iterator[Path]:
    """给测试一个可写的临时目录（项目内 ``_tmp/<name>-<pid>-<n>``）。"""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path: Path | None = None
    for _ in range(200):
        candidate = TMP_ROOT / f"{name}-{os.getpid()}-{next(_counter)}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        path = candidate
        break
    if path is None:
        raise RuntimeError(f"无法在 {TMP_ROOT} 下创建临时目录")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
