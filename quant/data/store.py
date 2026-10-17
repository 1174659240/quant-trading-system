"""数据仓库：统一入口 + 内存/磁盘缓存 + 多标的对齐。

上层（策略、回测）只跟 DataStore 打交道，不直接碰具体数据源::

    store = DataStore(SyntheticSource())
    bars = store.get("600519.SH", "2023-01-01", "2023-12-31")
    closes = store.close_matrix(["600519.SH", "000001.SZ"], "2023-01-01", "2023-12-31")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DataSource, DataError, normalize_frame


class DataStore:
    def __init__(
        self,
        source: DataSource,
        *,
        cache_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        self.source = source
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.data_dir = Path(data_dir) if data_dir else None
        self._memory: dict[str, pd.DataFrame] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- 缓存键 ----
    @staticmethod
    def _key(symbol: str, start: str | None, end: str | None) -> str:
        return f"{symbol}|{start or ''}|{end or ''}"

    def _cache_path(self, symbol: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{symbol.replace('/', '_')}.csv"

    # ---- 取数 ----
    def get(self, symbol: str, start: str | None = None, end: str | None = None, *, refresh: bool = False) -> pd.DataFrame:
        key = self._key(symbol, start, end)
        if not refresh and key in self._memory:
            return self._memory[key].copy()

        full = None
        if not refresh:
            full = self._load_disk(symbol)
        if full is None:
            full = self.source.load(symbol)  # 全量，便于跨区间复用
            full = normalize_frame(full, symbol=symbol)
            self._save_disk(symbol, full)

        frame = full
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index <= pd.Timestamp(end)]
        if len(frame) == 0:
            raise DataError(f"{symbol}: 区间 {start}~{end} 内没有行情数据")
        self._memory[key] = frame
        return frame.copy()

    def get_many(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        *,
        refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """批量取数；`symbols=None` 表示「数据源里能列出来的全部」。"""
        targets = list(symbols) if symbols is not None else list(self.source.symbols())
        out: dict[str, pd.DataFrame] = {}
        for s in targets:
            try:
                out[s] = self.get(s, start, end, refresh=refresh)
            except Exception as exc:  # 单只缺失不该拖垮全池
                print(f"[DataStore] 跳过 {s}: {exc}")
        if not out:
            raise DataError(f"区间 {start}~{end} 内没有任何可用标的")
        return out

    # ---- 对齐与派生 ----
    @staticmethod
    def calendar(bars: dict[str, pd.DataFrame], *, union: bool = True) -> pd.DatetimeIndex:
        idx: pd.DatetimeIndex | None = None
        for frame in bars.values():
            idx = frame.index if idx is None else (idx.union(frame.index) if union else idx.intersection(frame.index))
        if idx is None:
            raise DataError("空的 bars 集合")
        return pd.DatetimeIndex(idx.sort_values(), name="date")

    def close_matrix(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        field: str = "close",
    ) -> pd.DataFrame:
        bars = self.get_many(symbols, start, end)
        mat = pd.DataFrame({s: f[field] for s, f in bars.items()})
        return mat.sort_index()

    def amount_matrix(self, symbols: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
        return self.close_matrix(symbols, start, end, field="amount")

    def describe(self, symbols: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """一行一只标的的体检表：起止日期、交易日数、价格区间、年化波动。"""
        bars = self.get_many(symbols, start, end)
        rows = []
        for s, f in bars.items():
            ret = f["close"].pct_change().dropna()
            rows.append(
                {
                    "symbol": s,
                    "rows": len(f),
                    "start": f.index[0].date(),
                    "end": f.index[-1].date(),
                    "first_close": float(f["close"].iloc[0]),
                    "last_close": float(f["close"].iloc[-1]),
                    "total_return": float(f["close"].iloc[-1] / f["close"].iloc[0] - 1),
                    "ann_vol": float(ret.std(ddof=1) * (244**0.5)) if len(ret) > 1 else 0.0,
                    "avg_amount": float(f["amount"].mean()) if "amount" in f else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    # ---- 磁盘缓存 ----
    def _load_disk(self, symbol: str) -> pd.DataFrame | None:
        path = self._cache_path(symbol)
        if path is None or not path.is_file():
            return None
        try:
            frame = pd.read_csv(path)
            return normalize_frame(frame, symbol=symbol)
        except Exception:
            return None

    def _save_disk(self, symbol: str, frame: pd.DataFrame) -> None:
        path = self._cache_path(symbol)
        if path is None or path.is_file():
            return
        try:
            frame.to_csv(path, index=True, index_label="date", encoding="utf-8-sig")
        except Exception:
            pass
