"""仓库纪律自检：在 CI 与本地都能跑，只依赖标准库 + git。

检查项（任一失败即返回 1）：
1. 体积：单个跟踪文件 > 5 MB 视为误提交（数据/结果应留在本地）。
2. 数据文件：禁止 .parquet/.feather/.h5/.pkl/.npy/.db/.sqlite；
   .csv/.gz 只允许出现在 results/releases/ 下（冻结证据）。
3. 凭据：扫描 token/私钥/密码样式，命中即失败。
4. 冻结证据：results/releases/*/manifest.csv 里每个文件的 sha256 必须对得上。
5. 策略台账：docs/策略报告/registry.json 合法，且每条策略的报告文件存在。
6. 忽略规则：_tmp/、data/、results/ 确实被 git 忽略，results/releases/ 不被忽略。

用法：
    python scripts/repo_guard.py            # 全量检查
    python scripts/repo_guard.py --quiet    # 只打印失败项
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024

BANNED_SUFFIX = {".parquet", ".feather", ".h5", ".hdf5", ".pkl", ".pickle",
                 ".npy", ".npz", ".db", ".sqlite", ".sqlite3", ".zip", ".7z", ".rar"}
RELEASE_ONLY_SUFFIX = {".csv", ".gz", ".tsv"}
ALLOWED_RELEASE_DIR = "results/releases/"

SECRET_PATTERNS = [
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("GitHub fine-grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("OpenAI/DeepSeek key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("AWS key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("URL 里带账号", re.compile(r"https://[A-Za-z0-9._-]+:[^@\s/]+@")),
    ("私钥", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("硬编码密码", re.compile(r"(?i)\b(password|passwd|api[_-]?key|secret)\s*[:=]\s*[\"'][^\"'\s]{6,}[\"']")),
]
TEXT_SUFFIX = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".csv", ".txt",
               ".cfg", ".ini", ".ps1", ".sh", ".mjs", ".js", ".ts", ".html"}
ALLOWED_STATUS = {"active", "frozen", "superseded", "archived", "draft", "oos-contaminated"}


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败：{out.stderr.strip()}")
    return out.stdout


def tracked_files() -> list[str]:
    return [line for line in git("ls-files", "-z").split("\0") if line]


def check_sizes(files: list[str], fail: list[str], warn: list[str]) -> None:
    for name in files:
        path = ROOT / name
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            fail.append(f"体积超限 {size / 1048576:.1f}MB：{name}（阈值 5MB）")
        elif size > 1024 * 1024:
            warn.append(f"较大文件 {size / 1024:.0f}KB：{name}")


def check_data_files(files: list[str], fail: list[str]) -> None:
    for name in files:
        suffix = Path(name).suffix.lower()
        if suffix in BANNED_SUFFIX:
            fail.append(f"禁止提交的数据文件：{name}")
        elif suffix in RELEASE_ONLY_SUFFIX and not name.startswith(ALLOWED_RELEASE_DIR):
            fail.append(f"{suffix} 只允许放在 {ALLOWED_RELEASE_DIR} 下：{name}")


def check_secrets(files: list[str], fail: list[str]) -> None:
    for name in files:
        if Path(name).suffix.lower() not in TEXT_SUFFIX:
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if len(line) > 4000:
                continue
            for label, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    fail.append(f"疑似{label}：{name}:{line_no}")


def check_releases(fail: list[str]) -> int:
    checked = 0
    for manifest in sorted((ROOT / "results" / "releases").glob("*/manifest.csv")):
        base = manifest.parent
        with manifest.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rel = (row.get("relpath") or row.get("path") or "").strip()
                want = (row.get("sha256") or "").strip().lower()
                if not rel or not want:
                    continue
                target = base / rel
                if not target.is_file():
                    target = base / "artifacts" / rel
                if not target.is_file():
                    fail.append(f"冻结证据缺文件：{target.relative_to(ROOT).as_posix()}")
                    continue
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                if digest != want:
                    fail.append(f"冻结证据被改动：{target.relative_to(ROOT).as_posix()}")
                checked += 1
    return checked


def check_registry(fail: list[str]) -> int:
    path = ROOT / "docs" / "策略报告" / "registry.json"
    if not path.is_file():
        fail.append("缺少 docs/策略报告/registry.json")
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail.append(f"registry.json 解析失败：{exc}")
        return 0
    strategies = data.get("strategies") or data
    if isinstance(strategies, dict):
        rows = [{"id": k, **(v or {})} for k, v in strategies.items()]
    elif isinstance(strategies, list):
        rows = [row for row in strategies if isinstance(row, dict)]
    else:
        fail.append("registry.json 结构不符：strategies 应为对象或数组")
        return 0
    count = 0
    for meta in rows:
        count += 1
        name = meta.get("id") or meta.get("name") or f"#{count}"
        doc = meta.get("report")
        if not doc:
            fail.append(f"registry.json 里 {name} 缺 report 字段")
            continue
        if not (ROOT / doc).is_file():
            fail.append(f"registry.json 指向的报告不存在：{doc}")
        status = meta.get("status")
        if status is not None:
            tokens = [part.strip() for part in str(status).split("/") if part.strip()]
            bad = [t for t in tokens if t not in ALLOWED_STATUS]
            if bad:
                fail.append(f"registry.json 里 {name} 的 status 不合法：{status}")
    return count


def check_ignore(fail: list[str]) -> None:
    cases = [("_tmp/probe.py", True), ("data/cache/x.csv", True), ("results/x.csv", True),
             ("results/releases/v1.0/manifest.csv", False)]
    for probe, should_ignore in cases:
        out = subprocess.run(["git", "check-ignore", "-q", probe], cwd=ROOT)
        ignored = out.returncode == 0
        if ignored != should_ignore:
            verb = "应被忽略却未忽略" if should_ignore else "应被跟踪却被忽略"
            fail.append(f"忽略规则异常（{verb}）：{probe}")


def main() -> int:
    ap = argparse.ArgumentParser(description="仓库纪律自检")
    ap.add_argument("--quiet", action="store_true", help="只打印失败项")
    args = ap.parse_args()

    files = tracked_files()
    fail: list[str] = []
    warn: list[str] = []

    check_sizes(files, fail, warn)
    check_data_files(files, fail)
    check_secrets(files, fail)
    release_files = check_releases(fail)
    strategies = check_registry(fail)
    check_ignore(fail)

    if not args.quiet:
        print(f"跟踪文件 {len(files)} 个；冻结证据 {release_files} 个文件已校验；策略 {strategies} 条")
        for item in warn:
            print(f"  ! {item}")
    if fail:
        print(f"\n仓库自检失败（{len(fail)} 项）：")
        for item in fail:
            print(f"  x {item}")
        return 1
    if not args.quiet:
        print("仓库自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
