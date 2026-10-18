"""本地行情数据湖：全市场日线落地、原始/精简分层、最终校核集物理隔离。

设计约定（工程纪律，不是建议）：

* ``raw/``        原始落地层。一次抓取一个文件，**只增不改**，回测与因子一律可从这里重建。
* ``processed/``  精简层。由 ``raw/`` 里 **2025-01-01 之前**的数据清洗生成，日常研究只读这一层。
* ``holdout/``    最终校核集。2025-01-01 及以后的数据单独落盘并加 ``.LOCKED`` 标记，
                  代码默认拒绝读取；必须显式 ``allow_holdout=True`` 才放行，且只该在
                  用户明确要求做「最终校核」时使用。

原始层保存的是**抓取时刻的前复权快照**：前复权价格会随之后的除权除息变化，
所以这个快照本身是口径固定、可复现的历史记录；源、抓取时间、行数都记在
``meta/fetch_log.jsonl``，任何一次回测都能追到它用了哪一版数据。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .data.base import DataError, normalize_frame

DEFAULT_ROOT = Path("D:/quant-data")
HOLDOUT_START = "2025-01-01"
RAW_START = "2016-01-01"
LOCK_NAME = ".LOCKED"
MIN_ROWS = 120


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    """先写 ``.tmp`` 再原子替换，避免抓一半留下半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _longest_run(flags) -> int:
    """最长连续 1 的长度（用来衡量最长停牌天数）。"""
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return int(best)


def _frame_to_csv_rows(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """把 normalized frame（DatetimeIndex）摊成落盘表。"""
    out = frame.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.insert(0, "symbol", symbol)
    return out


def _ensure_status_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """processed/holdout 的列恒定为 {prev_close, suspended, tradable}。

    没有交易日历时 ``materialize_suspensions`` 不会加 ``suspended`` 列；
    这里补齐为 0，保证 CSV 层与 Parquet 层 schema 一致、下游不用分支判断。
    """
    out = frame.copy()
    if "suspended" not in out.columns:
        out["suspended"] = 0
    out["suspended"] = out["suspended"].fillna(0).clip(0, 1).round().astype("int8")
    # 统一口径：prev_close 一律是「前一交易日收盘」，首行为 NaN（不用 raw 层首行 open 的占位值）
    out["prev_close"] = out["close"].shift(1)
    out["tradable"] = ((out["volume"] > 0) & (out["suspended"] == 0)).astype("int8")
    return out


@dataclass
class Lake:
    """数据湖门面。所有路径都从这里派生，不散落在脚本里。"""

    root: Path = DEFAULT_ROOT
    holdout_start: str = HOLDOUT_START
    raw_start: str = RAW_START
    adjust: str = "none"
    allow_holdout: bool = False
    min_rows: int = MIN_ROWS          # 精简层门槛（够不够做因子）
    min_raw_rows: int = 2             # 原始层门槛：新股/次新只有 1~4 根也留档，精简层再用 min_rows 过滤

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._log_path = self.meta / "fetch_log.jsonl"
        self._progress_path = self.meta / "crawl_progress.json"
        self._manifest_path = self.meta / "manifest.json"

    # ------------------------------------------------------------------ 路径
    @property
    def raw(self) -> Path:
        return self.root / "raw" / "daily"

    @property
    def processed(self) -> Path:
        return self.root / "processed" / "daily"

    @property
    def holdout(self) -> Path:
        return self.root / "holdout" / "daily"

    @property
    def processed_parquet(self) -> Path:
        """精简层的列式副本（按 year=YYYY 分区）；只是副本，CSV 仍是权威数据。"""
        return self.root / "processed" / "parquet"

    @property
    def holdout_parquet(self) -> Path:
        return self.root / "holdout" / "parquet"

    @property
    def meta(self) -> Path:
        return self.root / "meta"

    def raw_path(self, symbol: str) -> Path:
        return self.raw / f"{symbol}.csv"

    def processed_path(self, symbol: str) -> Path:
        return self.processed / f"{symbol}.csv"

    def holdout_path(self, symbol: str) -> Path:
        return self.holdout / f"{symbol}.csv"

    @property
    def universe_path(self) -> Path:
        return self.meta / "universe.csv"

    @property
    def actions_path(self) -> Path:
        """除权除息事件表（原始事件，全市场一份）。"""
        return self.root / "raw" / "actions.csv"

    @property
    def calendar_path(self) -> Path:
        """交易日历（指数日线的日期序列，全市场一份）。"""
        return self.meta / "calendar.csv"

    @property
    def lock_path(self) -> Path:
        return self.root / "holdout" / LOCK_NAME

    # ------------------------------------------------------------------ 初始化
    def init(self) -> dict:
        for path in (self.raw, self.processed, self.holdout, self.meta):
            path.mkdir(parents=True, exist_ok=True)
        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text(
                "# A股全市场日线数据湖\n\n"
                f"- 根目录：`{self.root}`\n"
                f"- `raw/daily/<代码>.csv`：原始落地层（抓取时刻前复权快照，只增不改）\n"
                f"- `processed/daily/<代码>.csv`：精简层（由 raw 中 {self.holdout_start} 之前的数据生成）\n"
                f"- `holdout/daily/<代码>.csv`：最终校核集（{self.holdout_start} 及以后，默认禁止读取）\n"
                f"- `meta/`：universe.csv、fetch_log.jsonl、manifest.json、quality.jsonl\n\n"
                "纪律：任何回测/因子只能用 `processed/`；`holdout/` 只在用户明确要求最终校核时解锁。\n",
                encoding="utf-8",
            )
        base = {
            "root": str(self.root),
            "created": _now_iso(),
            "holdout_start": self.holdout_start,
            "adjust": self.adjust,
            "min_rows": self.min_rows,
        }
        self.save_manifest({**base, **self.load_manifest()})
        return self.manifest(include_files=False)

    # ------------------------------------------------------------------ 清单
    def load_manifest(self) -> dict:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_manifest(self, data: dict) -> None:
        """写 manifest。多个抓取进程并行时可能撞上文件占用，重试几次后放弃（manifest 只是元数据）。"""
        self.meta.mkdir(parents=True, exist_ok=True)
        tmp = self._manifest_path.with_name(f"manifest.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(6):
            try:
                os.replace(tmp, self._manifest_path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        try:
            tmp.unlink()
        except OSError:
            pass

    def manifest(self, *, include_files: bool = True) -> dict:
        data = self.load_manifest()
        data.setdefault("root", str(self.root))
        data.setdefault("created", _now_iso())
        data["holdout_start"] = self.holdout_start
        data["adjust"] = self.adjust
        if include_files:
            data["raw"] = self._dir_stats(self.raw)
            data["processed"] = self._dir_stats(self.processed)
            data["holdout"] = self._dir_stats(self.holdout)
            data["lock"] = self.lock_path.exists()
        return data

    @staticmethod
    def _dir_stats(folder: Path) -> dict:
        files = sorted(folder.glob("*.csv")) if folder.exists() else []
        total = sum(p.stat().st_size for p in files)
        dates: list[str] = []
        rows = 0
        for path in files[:50]:
            try:
                frame = pd.read_csv(path, usecols=["date"], encoding="utf-8-sig")
                dates.extend([str(frame["date"].iloc[0]), str(frame["date"].iloc[-1])])
                rows += len(frame)
            except Exception:
                continue
        return {
            "files": len(files),
            "bytes": total,
            "sample_rows": rows,
            "first": min(dates) if dates else None,
            "last": max(dates) if dates else None,
            "mb": round(total / 1024 / 1024, 2),
        }

    # ------------------------------------------------------------------ 代码表
    def read_universe(self) -> list[str]:
        if not self.universe_path.exists():
            return []
        frame = pd.read_csv(self.universe_path, dtype=str, encoding="utf-8-sig")
        if "symbol" in frame.columns:
            return [s for s in frame["symbol"].tolist() if s and s == s]
        return [s for s in frame.iloc[:, 0].tolist() if s and s == s]

    def write_universe(self, frame: "pd.DataFrame | list") -> Path:
        """写入代码表；``symbol`` 为 6 位数字代码。接受 DataFrame 或代码列表。"""
        self.meta.mkdir(parents=True, exist_ok=True)
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame({"symbol": [str(x) for x in frame]})
        out = frame.copy()
        if "symbol" not in out.columns:
            for candidate in ("代码", "code", "证券代码"):
                if candidate in out.columns:
                    out = out.rename(columns={candidate: "symbol"})
                    break
        raw = out["symbol"].astype(str).str.strip().str.upper()
        # 统一成 6 位裸代码（文件名也统一）：600519.SH / sh600519 / 920000.BJ → 600519 / 600519 / 920000
        bare = raw.str.replace(r"\.(SH|SZ|BJ|SS)$", "", regex=True).str.replace(r"^(SH|SZ|BJ)", "", regex=True)
        exchange = []
        for code, original in zip(bare, raw):
            if original.endswith((".SH", ".SS")) or original.startswith("SH"):
                exchange.append("SH")
            elif original.endswith(".SZ") or original.startswith("SZ"):
                exchange.append("SZ")
            elif original.endswith(".BJ") or original.startswith("BJ"):
                exchange.append("BJ")
            elif str(code)[:2] == "92" or str(code)[0] in "48":
                exchange.append("BJ")
            elif str(code)[0] in "69":
                exchange.append("SH")
            else:
                exchange.append("SZ")
        out["symbol"] = bare.str.zfill(6)
        out["exchange"] = exchange
        out = out.drop_duplicates("symbol").reset_index(drop=True)
        out.to_csv(self.universe_path, index=False, encoding="utf-8-sig")
        return self.universe_path

    def ensure_actions(self, *, force: bool = False, max_age_days: float = 30.0) -> "pd.DataFrame":
        """确保除权除息事件表就绪（缺了或太旧就重拉一次）。"""
        from .data.corporate_actions import fetch_events, load_events, save_events

        path = self.actions_path
        if path.exists() and not force:
            age_days = (time.time() - path.stat().st_mtime) / 86400.0
            if age_days <= max_age_days:
                return load_events(path)
        events = fetch_events(self.raw_start)
        save_events(events, path)
        return events

    def load_actions(self) -> "pd.DataFrame":
        from .data.corporate_actions import load_events

        return load_events(self.actions_path)

    # ------------------------------------------------------------------ 原始落地
    def persist_raw(self, symbol: str, frame: pd.DataFrame, *, source: str = "unknown", rows: int = 0) -> Path:
        """把一段日线写进原始层（清洗掉非法行，日期严格递增）。"""
        # 先在「脏数据」状态下剔除非法/未来行，再交给 normalize_frame 做契约校验；
        # 否则 normalize_frame 会直接抛 DataError，脏行就没机会被丢掉。
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index()
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.loc[frame["date"] <= pd.Timestamp.now().normalize()]
        for column in ("close", "open", "high", "low"):
            frame = frame.loc[pd.to_numeric(frame[column], errors="coerce") > 0]
        frame = frame.sort_values("date").drop_duplicates("date", keep="last")
        if len(frame) == 0:
            raise DataError(f"{symbol}: 没有可落地的有效行")
        frame = normalize_frame(frame, symbol=symbol)
        frame = frame.loc[frame.index <= pd.Timestamp.now().normalize()]
        if len(frame) == 0:
            raise DataError(f"{symbol}: 没有可落地的有效行")
        path = self.raw_path(symbol)
        _atomic_write(_frame_to_csv_rows(frame, symbol=symbol), path)
        self._append_log(
            {
                "ts": _now_iso(),
                "symbol": symbol,
                "source": source,
                "adjust": self.adjust,
                "rows": int(len(frame)),
                "first": frame.index[0].strftime("%Y-%m-%d"),
                "last": frame.index[-1].strftime("%Y-%m-%d"),
                "bytes": path.stat().st_size,
            }
        )
        return path

    def _append_log(self, record: dict) -> None:
        self.meta.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_log(self) -> list[dict]:
        if not self._log_path.exists():
            return []
        out: list[dict] = []
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------ 抓取
    def plan(self, symbols: Sequence[str], *, resume: bool = True, max_age_days: float = 7.0) -> dict:
        """算出要抓 / 跳过哪些代码（续跑靠它，避免重复请求）。"""
        symbols = [str(s) for s in symbols]
        if not resume:
            return {"todo": symbols, "skipped": []}
        last: dict[str, str] = {}
        for rec in self.read_log():
            if rec.get("rows"):
                last[str(rec.get("symbol"))] = str(rec.get("ts", ""))
        now = time.time()
        todo, skipped = [], []
        for symbol in symbols:
            path = self.raw_path(symbol)
            if not path.exists():
                todo.append(symbol)
                continue
            stamp = last.get(symbol)
            fresh = False
            if stamp:
                try:
                    age = (now - datetime.fromisoformat(stamp).timestamp()) / 86400.0
                    fresh = age <= max_age_days
                except ValueError:
                    fresh = False
            if fresh:
                skipped.append(symbol)
            else:
                todo.append(symbol)
        return {"todo": todo, "skipped": skipped}

    @staticmethod
    def source_for(symbol: str) -> str:
        """按交易所选源：北交所腾讯没有历史（只回 1 根），走新浪；沪深走腾讯。"""
        code = str(symbol).split(".")[0].strip()
        if code[:2] == "92" or code[:1] in ("4", "8"):
            return "sina"
        return "tencent"

    def crawl(
        self,
        symbols: Sequence[str] | None = None,
        *,
        source: str = "auto",
        start: str | None = None,
        pause: float = 0.30,
        limit: int | None = None,
        resume: bool = True,
        force: bool = False,
        max_age_days: float = 7.0,
        progress_every: int = 20,
        verbose: bool = True,
        write_manifest: bool = True,
    ) -> dict:
        """批量抓日线写入原始层。返回统计；可随时中断，下次自动续跑。

        ``source="auto"`` 表示按交易所自动选源（北交所→新浪，沪深→腾讯）。
        """
        from .data.web_source import WebKlineSource  # 延迟导入，离线测试不受网络依赖影响

        symbols = [str(s) for s in (symbols if symbols is not None else self.read_universe())]
        if not symbols:
            raise DataError("代码表为空：先跑 `lake universe` 抓全市场列表")
        plan = self.plan(symbols, resume=resume and not force, max_age_days=max_age_days)
        todo = plan["todo"][: limit or None]
        started = time.time()
        ok, failed = [], []
        clients: dict[str, WebKlineSource] = {}

        def client_for(symbol: str) -> tuple[WebKlineSource, str]:
            name = self.source_for(symbol) if source == "auto" else source
            if name not in clients:
                clients[name] = WebKlineSource(source=name, adjust=self.adjust, retries=2, pause=1.0)
            return clients[name], name

        for index, symbol in enumerate(todo, start=1):
            try:
                client, name = client_for(symbol)
                frame = client.load(symbol, start=start or self.raw_start)
                if frame is None or len(frame) < self.min_raw_rows:
                    raise DataError(f"行数过少（{0 if frame is None else len(frame)} < {self.min_raw_rows}）")
                self.persist_raw(symbol, frame, source=name)
                ok.append(symbol)
            except Exception as exc:  # 单只失败不拖垮整批
                failed.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
                if verbose:
                    print(f"  ✗ {symbol} {type(exc).__name__}: {exc}")
            if verbose and index % progress_every == 0:
                elapsed = time.time() - started
                rate = index / elapsed if elapsed else 0.0
                self._save_progress(
                    {
                        "done": index,
                        "total": len(todo),
                        "ok": len(ok),
                        "failed": len(failed),
                        "elapsed_s": round(elapsed, 1),
                        "per_symbol_s": round(1 / rate, 2) if rate else None,
                        "eta_s": round((len(todo) - index) / rate, 1) if rate else None,
                        "last_symbol": symbol,
                    }
                )
            time.sleep(max(0.0, pause))
        summary = {
            "requested": len(symbols),
            "todo": len(todo),
            "skipped": len(plan["skipped"]),
            "ok": len(ok),
            "failed": len(failed),
            "failures": failed[:50],
            "elapsed_s": round(time.time() - started, 1),
            "finished": _now_iso(),
        }
        if write_manifest:
            self.save_manifest({**self.load_manifest(), "last_crawl": summary, "symbols": len(symbols)})
            # 全部抓完就清掉断点文件，免得 status 里还挂着旧 ETA
            if limit is None and not self.plan(symbols, resume=True, max_age_days=max_age_days)["todo"]:
                self._progress_path.unlink(missing_ok=True)
        return summary

    def _save_progress(self, data: dict) -> None:
        self.meta.mkdir(parents=True, exist_ok=True)
        self._progress_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def crawl_progress(self) -> dict:
        if not self._progress_path.exists():
            return {}
        try:
            return json.loads(self._progress_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ------------------------------------------------------------------ 日历
    def load_calendar(self) -> pd.DatetimeIndex:
        """读本地交易日历；没有就返回空（派生层会退化为「不补停牌」）。"""
        from .data.calendar import load_calendar

        return load_calendar(self.calendar_path)

    def ensure_calendar(self, *, force: bool = False, start: str | None = None) -> pd.DatetimeIndex:
        """联网取指数日线生成交易日历并落盘（``meta/calendar.csv``）。"""
        from .data.calendar import fetch_calendar, save_calendar

        existing = self.load_calendar()
        if len(existing) and not force:
            return existing
        dates = fetch_calendar(start or self.raw_start)
        save_calendar(dates, self.calendar_path)
        self.save_manifest({**self.load_manifest(), "calendar": {"rows": int(len(dates)),
                                                                 "first": str(dates[0].date()),
                                                                 "last": str(dates[-1].date()),
                                                                 "fetched": _now_iso()}})
        return dates

    # ------------------------------------------------------------------ 分层
    def split_holdout(self) -> dict:
        """把原始层里 ``holdout_start`` 之后的行切到 holdout 层，并加锁。"""
        from .data.calendar import materialize_suspensions
        from .data.corporate_actions import apply_qfq

        self.holdout.mkdir(parents=True, exist_ok=True)
        cut = pd.Timestamp(self.holdout_start)
        events = self.load_actions()
        events_by = {str(sym): part for sym, part in events.groupby("symbol")} if len(events) else {}
        calendar = self.load_calendar()
        done, skipped, rows = [], [], 0
        for path in sorted(self.raw.glob("*.csv")):
            symbol = path.stem
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"symbol": str}).set_index("date")
            except Exception:
                skipped.append(symbol)
                continue
            frame = apply_qfq(frame, events_by.get(symbol))
            if len(calendar):
                frame = materialize_suspensions(frame, calendar)
            # 与 processed 层同口径：prev_close / suspended / tradable（停牌行 tradable=0）
            frame = _ensure_status_columns(frame)
            part = frame.loc[frame.index >= cut]
            if len(part) == 0:
                skipped.append(symbol)
                continue
            part = part.reset_index()
            part["date"] = pd.to_datetime(part["date"]).dt.strftime("%Y-%m-%d")
            _atomic_write(part, self.holdout_path(symbol))
            done.append(symbol)
            rows += len(part)
        self.lock_path.write_text(
            "FINAL VALIDATION SET — 最终校核集。\n"
            f"范围：{self.holdout_start} 及以后（截止 {_now_iso()}）。\n"
            "未经用户明确许可，任何脚本/分析不得读取本目录。\n"
            "解锁方式：Lake(allow_holdout=True) 或 CLI 的 --allow-holdout。\n",
            encoding="utf-8",
        )
        summary = {
            "files": len(done),
            "rows": int(rows),
            "skipped": len(skipped),
            "holdout_start": self.holdout_start,
            "finished": _now_iso(),
        }
        self.save_manifest({**self.load_manifest(), "last_split": summary, "holdout_locked": True})
        return summary

    def build_processed(self) -> dict:
        """由原始层的「训练区间」生成精简层；holdout 区间一律不进 processed。"""
        from .data.calendar import materialize_suspensions
        from .data.corporate_actions import apply_qfq, sanity_report

        self.processed.mkdir(parents=True, exist_ok=True)
        cut = pd.Timestamp(self.holdout_start)
        events = self.load_actions()
        calendar = self.load_calendar()
        train_calendar = calendar[calendar < cut] if len(calendar) else calendar
        by_symbol = {sym: part for sym, part in events.groupby("symbol")} if len(events) else {}
        quality_path = self.meta / "quality.jsonl"
        if quality_path.exists():
            quality_path.unlink()
        done, skipped, rows = [], [], 0
        for path in sorted(self.raw.glob("*.csv")):
            symbol = path.stem
            try:
                raw = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"symbol": str}).set_index("date").sort_index()
            except Exception:
                skipped.append(symbol)
                continue
            raw = apply_qfq(raw, by_symbol.get(symbol))
            train = raw.loc[raw.index < cut].copy()
            if len(train) == 0:
                skipped.append(symbol)
                continue
            # 停牌日：上游不返回行，这里按交易日历补成 suspended 行（价取前收、量为 0）
            if len(train_calendar):
                train = materialize_suspensions(train, train_calendar)
            if len(train) < self.min_rows:
                skipped.append(symbol)
                continue
            train = train[~train.index.duplicated(keep="last")]
            train = train[(train["close"] > 0) & (train["volume"] >= 0)]
            train = _ensure_status_columns(train)
            train = train.drop(columns=[c for c in ("symbol",) if c in train.columns])
            train = train.reset_index()
            train["date"] = pd.to_datetime(train["date"]).dt.strftime("%Y-%m-%d")
            train.insert(0, "symbol", symbol)
            _atomic_write(train, self.processed_path(symbol))
            done.append(symbol)
            rows += len(train)
            dates = pd.to_datetime(train["date"])
            window = (
                train_calendar[(train_calendar >= dates.min()) & (train_calendar <= dates.max())]
                if len(train_calendar)
                else pd.DatetimeIndex([])
            )
            missing_days = int(len(window) - len(dates)) if len(window) else 0
            suspended = train.get("suspended", pd.Series(0, index=train.index))
            report = sanity_report(train)
            with quality_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "symbol": symbol,
                            "rows": int(len(train)),
                            "first": str(train["date"].iloc[0]),
                            "last": str(train["date"].iloc[-1]),
                            "missing_days": missing_days,
                            "zero_volume_days": int((train["volume"] == 0).sum()),
                            "suspended_days": int(suspended.sum()),
                            "longest_suspension": _longest_run(suspended),
                            "adj_outliers": report["outliers"],
                            "max_abs_pct": report["max_abs_pct"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary = {
            "files": len(done),
            "rows": int(rows),
            "skipped": len(skipped),
            "holdout_start": self.holdout_start,
            "finished": _now_iso(),
        }
        self.save_manifest({**self.load_manifest(), "last_process": summary})
        return summary

    # ------------------------------------------------------------------ 读取
    def read(self, symbol: str, *, kind: str = "processed", format: str = "auto") -> pd.DataFrame:
        """读某一层的数据。holdout 需要显式解锁；``format`` 可选 auto/csv/parquet。"""
        if kind == "holdout":
            if not self.allow_holdout:
                raise PermissionError(
                    "holdout 是最终校核集，默认禁止读取；确实要做最终校核时用 allow_holdout=True / --allow-holdout。"
                )
            path = self.holdout_path(symbol)
            parquet_dir = self.holdout_parquet
        elif kind == "raw":
            path = self.raw_path(symbol)
            parquet_dir = None
        else:
            path = self.processed_path(symbol)
            parquet_dir = self.processed_parquet
        if format not in ("auto", "csv", "parquet"):
            raise ValueError(f"format 只能是 auto/csv/parquet，收到 {format!r}")
        use_parquet = format == "parquet" or (
            format == "auto" and parquet_dir is not None and parquet_dir.exists()
        )
        if use_parquet and parquet_dir is not None and parquet_dir.exists():
            from . import parquet_store

            frame = parquet_store.read_symbol(symbol, parquet_dir)
            if len(frame):
                if kind != "raw" and "symbol" not in frame.columns:
                    frame.insert(0, "symbol", symbol)
                return frame
            if format == "parquet":
                raise FileNotFoundError(f"{parquet_dir} 里没有 {symbol}")
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"symbol": str}).set_index("date")
        return frame.sort_index()

    def convert_parquet(
        self, *, kinds: Sequence[str] = ("processed",), overwrite: bool = False, sample: int = 10
    ) -> dict:
        """把 processed/holdout 的 CSV 层转成 Parquet 副本；raw 层不动。"""
        from . import parquet_store

        out: dict[str, dict] = {}
        for kind in kinds:
            if kind == "processed":
                src, dst = self.processed, self.processed_parquet
            elif kind == "holdout":
                if not self.allow_holdout:
                    raise PermissionError("转换 holdout 会读取校核集，需要 allow_holdout=True / --allow-holdout")
                src, dst = self.holdout, self.holdout_parquet
            else:
                raise ValueError(f"只支持 processed/holdout，收到 {kind!r}")
            out[kind] = parquet_store.convert(src, dst, overwrite=overwrite)
            out[kind]["compare"] = parquet_store.compare_csv(src, dst, sample=sample)
        summary = {
            "finished": _now_iso(),
            "layers": out,
            "ok": all(layer["compare"]["ok"] for layer in out.values()),
        }
        self.save_manifest({**self.load_manifest(), "last_parquet": {k: v for k, v in summary.items() if k != "layers"}})
        return summary

    def load_frames(
        self,
        symbols: Sequence[str] | None = None,
        *,
        kind: str = "processed",
        start: str | None = None,
        end: str | None = None,
        format: str = "auto",
    ) -> dict[str, pd.DataFrame]:
        """批量读成「策略/回测能直接用的 dict[代码] -> DataFrame（date 索引）」。

        列与 :class:`quant.data.base.FrameSpec` 一致：raw 层是原始价；processed/holdout 已是后复权价。
        读不到的单只直接跳过（返回的字典里没有它），并在 stderr 打印一行，方便知道池子被削了多少。
        """
        codes = [str(s) for s in (symbols if symbols is not None else self.read_universe())]
        lo = pd.Timestamp(start) if start else None
        hi = pd.Timestamp(end) if end else None
        out: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        for code in codes:
            try:
                frame = self.read(code, kind=kind, format=format)
            except FileNotFoundError:
                missing.append(code)
                continue
            frame = frame.drop(columns=[c for c in ("symbol",) if c in frame.columns])
            if lo is not None:
                frame = frame.loc[frame.index >= lo]
            if hi is not None:
                frame = frame.loc[frame.index <= hi]
            if len(frame):
                out[code] = frame
        if missing:
            print(f"[lake] {len(missing)} 只没有 {kind} 数据，已跳过（如 {missing[:3]}）", file=sys.stderr)
        return out

    # ------------------------------------------------------------------ 体检
    def verify(self, *, limit: int | None = None) -> dict:
        """抽查原始层与精简层：日期单调、价格为正、无未来日期、训练/校核不重叠。"""
        cut = pd.Timestamp(self.holdout_start)
        issues: list[dict] = []
        checked = 0
        raw_files = sorted(self.raw.glob("*.csv"))[: limit or None]
        for path in raw_files:
            checked += 1
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], dtype={"symbol": str}).set_index("date").sort_index()
            except Exception as exc:
                issues.append({"symbol": path.stem, "kind": "raw", "issue": f"读取失败: {exc}"})
                continue
            if not frame.index.is_monotonic_increasing:
                issues.append({"symbol": path.stem, "kind": "raw", "issue": "日期不单调"})
            if not ((frame["close"] > 0).all() and (frame["open"] > 0).all()):
                issues.append({"symbol": path.stem, "kind": "raw", "issue": "存在非正价格"})
            proc = self.processed_path(path.stem)
            if proc.exists():
                train = pd.read_csv(proc, encoding="utf-8-sig", parse_dates=["date"])
                if len(train) and pd.to_datetime(train["date"]).max() >= cut:
                    issues.append({"symbol": path.stem, "kind": "processed", "issue": "精简层混入校核区间"})
            hold = self.holdout_path(path.stem)
            if hold.exists():
                part = pd.read_csv(hold, encoding="utf-8-sig", parse_dates=["date"], dtype={"symbol": str})
                if len(part) and pd.to_datetime(part["date"]).min() < cut:
                    issues.append({"symbol": path.stem, "kind": "holdout", "issue": "校核集混入训练区间"})
        if self.processed_parquet.exists():
            from . import parquet_store

            try:
                cmp = parquet_store.compare_csv(self.processed, self.processed_parquet, sample=5)
                for msg in cmp["mismatch"][:20]:
                    issues.append({"symbol": "-", "kind": "parquet", "issue": msg})
            except Exception as exc:
                issues.append({"symbol": "-", "kind": "parquet", "issue": f"Parquet 层对拍失败: {exc}"})
        return {"checked": checked, "issues": issues, "issue_count": len(issues)}


# ---------------------------------------------------------------- CLI

_HELP = """lake —— 本地数据湖

用法：
  python -m quant lake init        [--root D:/quant-data]
  python -m quant lake universe    [--root ...]            # 抓全市场代码表
  python -m quant lake ref         [--what universe,industry,shares,sw] [--start-year 2014] [--as-of YYYY-MM-DD]
                                                           # 参考数据：universe 区间表 / 行业 PIT 快照 / 股本 / 申万快照
  python -m quant lake mainfin     [--start 2013-03-31] [--force] [--verify]
                                                           # 主要财务指标（金融股 ROE/EPS/BPS/股本，源自 MAINFINADATA）
  python -m quant lake dividends   [--start 2013-12-31] [--end 2024-12-31] [--force]
                                                           # PIT 分红明细（东财 RPT_SHAREBONUS_DET，ex_date 生效）
  python -m quant lake turnpanel   [--cutoff 2024-12-31]   # 换手率宽表（valuation 的 turn 列透视，按 holdout 裁剪）
  python -m quant lake valuation   [--start 2014-01-01] [--end YYYY-MM-DD] [--limit N] [--pause 0.1]
                                                           # 估值/ST 日频快照（peTTM/pbMRQ/isST），断点续跑
  python -m quant lake fund        [--tables profit,balance,cashflow] [--start 2013-03-31] [--limit N]
                                                           # 财务三表（东财 RPT_F10_FINANCE_*），按报告期落盘
  python -m quant lake index       [--codes 000300,000905]  # 指数成分快照（只有当前版本，带幸存者偏差）
  python -m quant lake delisted    [--limit N] [--events-only]  # 退市股日线（后复权）+ 退市事件表
  python -m quant lake events      [--what lhb,repurchase,hold,st,fhps,all] [--start 2014-01-01]
                                   [--summary]  # 事件层：龙虎榜/回购/增减持/ST 翻转/送转分红（双日期字段）
  python -m quant lake calendar    # 拉交易日历（停牌补齐用）
  python -m quant lake actions     # 拉除权除息事件表（复权用）
  python -m quant lake crawl       [--limit N] [--pause 0.3] [--force] [--source tencent]
  python -m quant lake split       # 2025 起切到 holdout 并加锁
  python -m quant lake process     # 由 raw 的训练区间生成 processed
  python -m quant lake parquet     [--kinds processed,holdout] [--overwrite]
  python -m quant lake status      [--json]
  python -m quant lake verify      [--limit N]
说明：holdout 默认禁止读取，确需最终校核时加 --allow-holdout。
"""


def _build_lake(args) -> Lake:
    return Lake(root=Path(args.root), holdout_start=args.holdout_start, adjust=args.adjust, allow_holdout=args.allow_holdout)


def main(argv: list[str] | None = None) -> int:
    import argparse

    argv = list(argv if argv is not None else [])
    if not argv or argv[0] in ("-h", "--help"):
        print(_HELP)
        return 0 if argv else 1
    action = argv[0]
    parser = argparse.ArgumentParser(prog=f"quant lake {action}")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--holdout-start", default=HOLDOUT_START)
    parser.add_argument("--adjust", default="none", choices=["none", "qfq", "hfq"], help="原始层复权口径；默认不复权，复权在精简层自己算")
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--json", action="store_true")
    if action == "crawl":
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--pause", type=float, default=0.30)
        parser.add_argument("--source", default="tencent", choices=["tencent", "sina", "eastmoney", "auto"])
        parser.add_argument("--start", default=None)
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--symbols", default=None, help="逗号分隔，缺省用代码表")
    if action == "verify":
        parser.add_argument("--limit", type=int, default=None)
    if action in ("actions", "calendar"):
        parser.add_argument("--force", action="store_true")
    if action == "ref":
        parser.add_argument(
            "--what",
            default="universe,industry",
            help="逗号分隔：universe,industry,shares,sw",
        )
        parser.add_argument("--start-year", type=int, default=2014, help="行业快照起始年份")
        parser.add_argument("--as-of", default=None, help="基准日 YYYY-MM-DD，缺省今天")
        parser.add_argument("--check-bs", type=int, default=30, help="（已弃用，保留兼容）")
        parser.add_argument("--share-pages", type=int, default=None, help="股本变动表最多翻多少页（调试用）")
        parser.add_argument("--force-sw", action="store_true", help="申万快照重抓")
    if action == "mainfin":
        parser.add_argument("--start", default="2013-03-31")
        parser.add_argument("--end", default=None)
        parser.add_argument("--force", action="store_true", help="已落盘的也重建")
        parser.add_argument("--verify", action="store_true", help="只打印金融股抽查")
    if action == "valuation":
        parser.add_argument("--start", default="2014-01-01")
        parser.add_argument("--end", default=None)
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--pause", type=float, default=0.10)
        parser.add_argument("--symbols", default=None, help="逗号分隔，缺省用 universe 全表")
        parser.add_argument("--force", action="store_true", help="已落盘的也重抓")
        parser.add_argument("--shard", default=None, help="i/n：只处理第 i 片（0 起），用于多进程并行")
    if action == "fund":
        parser.add_argument("--tables", default="profit,balance,cashflow")
        parser.add_argument("--start", default="2013-03-31")
        parser.add_argument("--end", default=None)
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--pause", type=float, default=0.05)
        parser.add_argument("--force", action="store_true", help="已落盘的也重抓")
    if action == "dividends":
        parser.add_argument("--start", default="2013-12-31")
        parser.add_argument("--end", default="2024-12-31")
        parser.add_argument("--force", action="store_true")
    if action == "turnpanel":
        parser.add_argument("--cutoff", default="2024-12-31")
        parser.add_argument("--start", default="2014-01-01")
    if action == "index":
        parser.add_argument("--codes", default=None, help="逗号分隔指数代码，缺省用内置清单")
        parser.add_argument("--no-merge", action="store_true", help="不与既有快照合并，整体覆盖")
    if action == "delisted":
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--pause", type=float, default=0.05)
        parser.add_argument("--events-only", action="store_true", help="只重建事件表，不抓日线")
        parser.add_argument("--force", action="store_true")
    if action == "events":
        parser.add_argument("--what", default="all",
                            help="逗号分隔：lhb,repurchase,hold,st,fhps,all")
        parser.add_argument("--start", default="2014-01-01")
        parser.add_argument("--end", default=None)
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--summary", action="store_true", help="只打印已落盘事件表概况")
    if action == "parquet":
        parser.add_argument("--kinds", default="processed", help="逗号分隔：processed,holdout")
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--sample", type=int, default=10, help="对拍抽样的代码数")
    args = parser.parse_args(argv[1:])
    lake = _build_lake(args)

    if action == "init":
        data = lake.init()
        print(f"数据湖就绪：{lake.root}")
        print(f"  raw={lake.raw}  processed={lake.processed}  holdout={lake.holdout}")
    elif action == "universe":
        from .data.web_source import list_symbols

        frame = list_symbols()
        path = lake.write_universe(frame)
        print(f"代码表：{len(frame)} 只 → {path}")
        data = {"symbols": int(len(frame)), "path": str(path)}
    elif action == "ref":
        from . import refdata

        wanted = [w.strip() for w in args.what.split(",") if w.strip()]
        data = {"as_of": args.as_of}
        for what in wanted:
            if what == "universe":
                data["universe"] = refdata.build_universe(as_of=args.as_of)
                u = data["universe"]
                print(
                    f"universe：{u['rows']} 行（在市 {u['live']} / 退市 {u['delisted']} / 在途退市 {u['pending_delist']}），"
                    f"B 股 {u['b_share']}，继任 {u['with_successor']} → {u['written'].get('parquet') or u['written']['csv']}"
                )
                print(f"  板块分布：{u['board_counts']}")
            elif what == "industry":
                data["industry"] = refdata.build_industry(start_year=args.start_year, as_of=args.as_of)
                i = data["industry"]
                print(f"industry：{i['rows']} 行 / {i['dates']} 版，最新版非空行业 {i['live_covered']} 只 → {i['written'].get('parquet') or i['written']['csv']}")
                print(f"  各版条数：{i['per_date']}")
            elif what == "shares":
                data["shares"] = refdata.build_shares(limit=args.share_pages)
                s = data["shares"]
                print(
                    f"shares：{s['rows']} 行 / {s['real_symbols']} 个真实代码（东财内部码 {s['bogus_symbols']}）"
                    f" / 变动日 {s['first']} ~ {s['last']}，在 universe 内 {s['in_universe']}，"
                    f"universe 缺 {s['universe_missing']} → {s['written'].get('parquet') or s['written']['csv']}"
                )
                print(f"  PIT 取法：只认 change_date <= t 的最近一条（shares_at(frame, day)）")
            elif what == "sw":
                if refdata.SW_CSV.exists() and not args.force_sw:
                    print(f"申万快照已存在（{refdata.SW_CSV}），加 --force-sw 重抓")
                    data["sw"] = {"skipped": True, "path": str(refdata.SW_CSV)}
                else:
                    data["sw"] = refdata.build_sw_industry()
                    w = data["sw"]
                    print(
                        f"sw：{w['industries']} 个一级行业 / {w['members']} 条成分 / "
                        f"覆盖 {w['covered_symbols']} 只，失败 {len(w['failed'])} → {w['written']}"
                    )
            else:
                print(f"未知参考数据：{what}（可选 universe,industry,shares,sw）")
                return 1
        data["manifest"] = str(refdata.write_manifest(data))
    elif action == "crawl":
        symbols = None
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        summary = lake.crawl(
            symbols,
            source=args.source,
            start=args.start,
            limit=args.limit,
            pause=args.pause,
            force=args.force,
        )
        print(
            f"\n抓取结束：成功 {summary['ok']} / 失败 {summary['failed']} / 跳过 {summary['skipped']}"
            f"（todo {summary['todo']}，用时 {summary['elapsed_s']}s）"
        )
        for item in summary["failures"][:10]:
            print(f"  ✗ {item['symbol']} {item['error']}")
        data = summary
    elif action == "calendar":
        dates = lake.ensure_calendar(force=args.force)
        print(f"交易日历：{len(dates)} 天 → {lake.calendar_path}")
        print(f"  {dates[0].date()} ~ {dates[-1].date()}")
        data = {"days": int(len(dates)), "path": str(lake.calendar_path)}
    elif action == "actions":
        events = lake.ensure_actions(force=args.force)
        print(f"除权除息事件：{len(events)} 条 → {lake.actions_path}")
        data = {"events": int(len(events)), "path": str(lake.actions_path)}
    elif action == "split":
        data = lake.split_holdout()
        print(f"最终校核集：{data['files']} 只 / {data['rows']} 行 → {lake.holdout}（已加锁）")
    elif action == "process":
        data = lake.build_processed()
        print(f"精简层：{data['files']} 只 / {data['rows']} 行 → {lake.processed}（跳过 {data['skipped']}）")
    elif action == "valuation":
        from . import valuation

        symbols = None
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        summary = valuation.build(
            symbols,
            start=args.start,
            end=args.end,
            limit=args.limit,
            pause=args.pause,
            resume=not args.force,
            shard=None if not args.shard else tuple(int(x) for x in args.shard.split("/")),
        )
        print(
            f"\n估值快照：写入 {summary['written']} 只 / {summary['rows']} 行，"
            f"空表 {summary['empty']}，失败 {summary['failed']}（用时 {summary['seconds']}s）"
        )
        if summary["empty_sample"]:
            print(f"  空表样本：{summary['empty_sample'][:10]}")
        for symbol, msg in list(summary["failures"].items())[:10]:
            print(f"  ✗ {symbol} {msg}")
        print(f"  目录：{summary['dir']}")
        data = summary
    elif action == "fund":
        from . import fundamentals

        tables = tuple(t.strip() for t in args.tables.split(",") if t.strip())
        summary = fundamentals.build(
            tables=tables,
            start=args.start,
            end=args.end,
            limit=args.limit,
            pause=args.pause,
            resume=not args.force,
        )
        print(
            f"\n财务三表：写入 {summary['written']} 期/{len(summary['tables'])} 表 / {summary['rows']} 行，"
            f"失败 {summary['failed']}（用时 {summary['seconds']}s）"
        )
        for key, msg in list(summary["failures"].items())[:10]:
            print(f"  ✗ {key} {msg}")
        print(f"  目录：{summary['dir']}")
        data = summary
    elif action == "mainfin":
        from . import mainfin

        if args.verify:
            frame = mainfin.verify()
            print(frame.to_string(index=False) if not frame.empty else "（无数据，先跑 mainfin）")
            data = {"verify_rows": int(len(frame))}
        else:
            summary = mainfin.build(start=args.start, end=args.end, force=args.force)
            print(
                f"\n主要财务指标：写入 {summary['written']} 期（跳过 {summary['skipped']}）/ "
                f"{summary['rows']} 行，缺源 {len(summary['missing_source'])}（用时 {summary['seconds']}s）"
            )
            print(f"  目录：{summary['dir']}")
            frame = mainfin.verify()
            if not frame.empty:
                print("  金融股抽查（最近一期）：")
                last = frame["report_date"].max()
                print(frame[frame["report_date"] == last].to_string(index=False))
            data = summary
    elif action == "dividends":
        from . import dividends

        summary = dividends.fetch(args.start, args.end, resume=not args.force)
        data = {k: v for k, v in summary.items() if k != "log"}
        print(
            f"\n分红明细：{data['periods']} 期 / {data['rows']} 行 / {data['symbols']} 只 → {dividends.DIVIDENDS}"
        )
    elif action == "turnpanel":
        from . import valuation

        data = valuation.build_turn_panel(cutoff=args.cutoff, start=args.start)
        print(
            f"\n换手宽表：{data['rows']} 行 × {data['symbols']} 列"
            f"（{data['first']} ~ {data['last']}，{data['mb']} MB）→ {data['path']}"
        )
    elif action == "index":
        from . import index_members

        codes = None if not args.codes else [c.strip() for c in args.codes.split(",") if c.strip()]
        summary = index_members.build(codes, merge=not args.no_merge)
        print(f"\n指数成分：{summary['rows']} 行 → {summary['written']}")
        for code, info in summary["indices"].items():
            print(f"  {code} {info['name']}: {info['members']} 只")
        if summary["failures"]:
            for code, msg in summary["failures"].items():
                print(f"  ✗ {code} {msg}")
        data = summary
        data["manifest"] = str(index_members.write_manifest(summary))
    elif action == "delisted":
        from . import delisted

        data = {}
        if not args.events_only:
            data["crawl"] = delisted.build(limit=args.limit, pause=args.pause, resume=not args.force)
            c = data["crawl"]
            print(
                f"\n退市股日线：写入 {c['written']}/{c['targets']} 只 / {c['rows']} 行，"
                f"空表 {c['empty']}，失败 {c['failed']}（用时 {c['seconds']}s）"
            )
            for symbol, msg in list(c["failures"].items())[:10]:
                print(f"  ✗ {symbol} {msg}")
        data["events"] = delisted.build_events()
        e = data["events"]
        print(
            f"退市事件表：{e['targets']} 只（有日线 {e['with_bars']} / 无 {e['without_bars']}），"
            f"日线合计 {e['rows']} 行，退市日-最后交易日 中位 {e['gap_median']} 天 → {e['written']}"
        )
    elif action == "events":
        from . import events

        wanted = [w.strip() for w in args.what.split(",") if w.strip()]
        if "all" in wanted:
            wanted = ["lhb", "repurchase", "hold", "st", "fhps"]
        data = {"what": wanted}
        if not args.summary:
            if "lhb" in wanted:
                data["lhb"] = events.fetch_lhb(start=args.start, end=args.end, resume=not args.force)
            if "repurchase" in wanted:
                data["repurchase"] = events.fetch_repurchase()
            if "hold" in wanted:
                data["hold"] = events.fetch_hold()
            if "st" in wanted:
                data["st"] = events.build_st(start=args.start)
            if "fhps" in wanted:
                data["fhps"] = events.fetch_fhps(start="2013-12-31", resume=not args.force)
        table = events.summary()
        print("\n事件表：")
        print(table.to_string(index=False))
        data["summary"] = table.to_dict("records")
    elif action == "parquet":
        from . import parquet_store

        if not parquet_store.available():
            print("缺少 pyarrow：先执行 `python -m pip install pyarrow`（CSV 层不受影响）")
            return 2
        kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
        data = lake.convert_parquet(kinds=kinds, overwrite=args.overwrite, sample=args.sample)
        for kind, info in data["layers"].items():
            cmp = info["compare"]
            print(
                f"{kind:10s} {info['rows']:>9d} 行 / {info['symbols']:>5d} 只 / {info['mb']:>7.1f} MB "
                f"/ {info['files']} 个分区 → {info['out']}"
            )
            if cmp["ok"]:
                print(f"           对拍通过：行数/代码数/日期范围/抽样 {cmp['checked_symbols']} 只逐值一致")
            else:
                for msg in cmp["mismatch"][:10]:
                    print(f"           ! {msg}")
    elif action == "status":
        data = lake.manifest()
        progress = lake.crawl_progress()
        if args.json:
            print(json.dumps({**data, "progress": progress}, ensure_ascii=False, indent=2))
        else:
            for name in ("raw", "processed", "holdout"):
                info = data.get(name, {})
                print(
                    f"{name:10s} {info.get('files', 0):5d} 只  {info.get('mb', 0):8.1f} MB  "
                    f"{info.get('first') or '-':10s} → {info.get('last') or '-':10s}"
                )
            for name, folder in (("pq-process", lake.processed_parquet), ("pq-holdout", lake.holdout_parquet)):
                if folder.exists():
                    parts = list(folder.rglob("*.parquet"))
                    mb = sum(p.stat().st_size for p in parts) / 1024 / 1024
                    print(f"{name:10s} {len(parts):5d} 个分区  {mb:8.1f} MB  （列式副本，CSV 仍是权威）")
            last = data.get("last_crawl") or {}
            if progress and progress.get("done"):
                print(
                    f"进度      {progress['done']}/{progress['total']}  ok={progress['ok']} "
                    f"fail={progress['failed']}  ETA={progress.get('eta_s')}s"
                )
            elif last:
                raw_files = data.get("raw", {}).get("files", 0)
                total = data.get("symbols") or 0
                print(
                    f"上次抓取  增量 成功 {last.get('ok')} / 失败 {last.get('failed')} @ {last.get('finished')}"
                    f"（原始层覆盖 {raw_files}/{total or '?'}）"
                )
            print(f"holdout   锁定={'是' if data.get('lock') else '否'}  起点={data.get('holdout_start')}")
    elif action == "verify":
        data = lake.verify(limit=args.limit)
        print(f"抽查 {data['checked']} 只，问题 {data['issue_count']} 条")
        for issue in data["issues"][:20]:
            print(f"  ! {issue['symbol']} [{issue['kind']}] {issue['issue']}")
    else:
        print(_HELP)
        return 1
    if args.json and action != "status":
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
