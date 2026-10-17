"""CSV 数据源：把本地导出的日线文件读成契约表。

目录约定（默认）::

    <root>/data/csv/600519.csv
    <root>/data/csv/000001.SZ.csv

支持 .csv/.txt/.tsv，自动嗅探分隔符与编码（utf-8 / utf-8-sig / gbk）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DataSource, DataError, normalize_frame

_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1")


class CsvSource(DataSource):
    """从文件夹里按标的代码取 CSV 日线。

    参数
    ----
    root
        CSV 所在目录，或单个 CSV 文件路径。
    symbol_column
        若一个文件里混放多只标的，用这一列区分；为 None 时按“一个文件一只标的”处理。
    """

    name = "csv"

    def __init__(self, root: str | Path, *, symbol_column: str | None = None) -> None:
        self.root = Path(root)
        self.symbol_column = symbol_column
        if not self.root.exists():
            raise DataError(f"CSV 数据目录不存在: {self.root}")

    # ---- 定位与读取 ----
    @staticmethod
    def _bare(code: str) -> str:
        """把 ``sh600519`` / ``600519.SH`` / ``600519`` 统一成 ``600519``。"""
        text = str(code).strip().lower()
        for prefix in ("sh", "sz", "bj"):
            if text.startswith(prefix) and text[2:3].isdigit():
                text = text[2:]
                break
        return text.split(".")[0]

    @staticmethod
    def _guess_exchange(bare: str) -> str:
        if bare.startswith("6"):
            return "sh"
        if bare.startswith(("0", "3")):
            return "sz"
        if bare.startswith(("4", "8")):
            return "bj"
        return ""

    def _files(self) -> list[Path]:
        if self.root.is_file():
            return [self.root]
        exts = (".csv", ".txt", ".tsv")
        return sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in exts)

    def _candidates(self, symbol: str) -> list[Path]:
        """按「精确文件名 > 带交易所前缀/后缀 > 同代码」三级匹配。"""
        if self.root.is_file():
            return [self.root]
        target = self._bare(symbol)
        wanted = {symbol.strip().lower(), target, f"sh{target}", f"sz{target}", f"bj{target}"}
        scored: list[tuple[int, Path]] = []
        for path in self._files():
            stem = path.stem.lower()
            if stem in {symbol.strip().lower(), target}:
                score = 3  # 文件名与给定代码完全一致
            elif stem in wanted:
                score = 2  # sh600519 / 000001.sz 这类等价写法
            elif self._bare(stem) == target:
                score = 1  # 同代码不同后缀
            else:
                continue
            scored.append((score, path))
        if not scored:
            return []
        best = max(score for score, _ in scored)
        picked = [path for score, path in scored if score == best]
        if len(picked) > 1 and best == 1:
            # 裸代码同时命中多个交易所时，按代码段猜一个
            guess = self._guess_exchange(target)
            narrowed = [p for p in picked if p.stem.lower().endswith(f".{guess}")] if guess else []
            if len(narrowed) == 1:
                return narrowed
        return picked

    def _read_raw(self, path: Path) -> pd.DataFrame:
        last: Exception | None = None
        for enc in _ENCODINGS:
            try:
                return pd.read_csv(path, encoding=enc, sep=None, engine="python")
            except UnicodeDecodeError as exc:  # 换编码再试
                last = exc
                continue
            except Exception as exc:  # 分隔符/结构问题，不再换编码
                raise DataError(f"读取 {path.name} 失败: {exc}") from exc
        raise DataError(f"读取 {path.name} 失败（编码都试过了）: {last}")

    def load(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        paths = self._candidates(symbol)
        if not paths:
            raise DataError(f"{self.root} 下找不到 {symbol} 的 CSV 文件")
        frames: list[pd.DataFrame] = []
        for path in paths:
            raw = self._read_raw(path)
            if self.symbol_column and self.symbol_column in raw.columns:
                raw = raw[raw[self.symbol_column].astype(str) == symbol]
            frames.append(normalize_frame(raw, symbol=symbol))
        frame = pd.concat(frames) if len(frames) > 1 else frames[0]
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        return _slice(frame, start, end)

    def symbols(self) -> list[str]:
        if self.root.is_file():
            return [self.root.stem]
        return sorted({p.stem for p in self._files()})


def _slice(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end)]
    return frame
