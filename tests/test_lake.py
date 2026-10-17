"""数据湖测试：分层、隔离、续跑、清洗，全部离线。"""

from __future__ import annotations

import json

import pandas as pd

from quant.data.web_source import WebKlineSource  # noqa: F401  # 仅确保模块可导入，不联网
from quant.lake import HOLDOUT_START, Lake

from _util import setup_console, tmpdir

setup_console()


def _bars(dates, base: float = 10.0, symbol: str = "600000") -> pd.DataFrame:
    rows = []
    for i, day in enumerate(dates):
        close = base + i
        rows.append(
            {
                "date": day,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1000 + i,
                "amount": (1000 + i) * close,
            }
        )
    from quant.data.base import normalize_frame

    return normalize_frame(pd.DataFrame(rows), symbol=symbol)


_DATES = ["2024-12-27", "2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]


def _lake(root, **kwargs):
    """测试用数据湖：把最小行数降到 3，免得为了凑 120 行造一堆假数据。"""
    return Lake(root=root, min_rows=3, **kwargs)


def test_init_creates_layers():
    with tmpdir("lake-init") as root:
        lake = _lake(root)
        data = lake.init()
        for folder in (lake.raw, lake.processed, lake.holdout, lake.meta):
            assert folder.is_dir(), folder
        assert (root / "README.md").is_file()
        assert data["holdout_start"] == HOLDOUT_START
        assert lake.load_manifest()["root"] == str(root)


def test_persist_raw_writes_rows_and_log():
    with tmpdir("lake-raw") as root:
        lake = _lake(root)
        lake.init()
        frame = _bars(_DATES)
        path = lake.persist_raw("600000", frame, source="tencent")
        assert path.is_file()
        table = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
        assert list(table.columns[:3]) == ["symbol", "date", "open"]
        assert (table["symbol"] == "600000").all()
        assert len(table) == len(_DATES)
        log = lake.read_log()
        assert log and log[-1]["symbol"] == "600000" and log[-1]["source"] == "tencent"
        assert log[-1]["rows"] == len(_DATES)
        # 原子写不留 .tmp
        assert not list(lake.raw.glob("*.tmp"))


def test_persist_raw_drops_invalid_and_future_rows():
    with tmpdir("lake-clean") as root:
        lake = _lake(root)
        lake.init()
        dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2099-01-01"]
        frame = _bars(dates)
        frame.loc[pd.Timestamp("2025-01-03"), "close"] = 0.0  # 非正价格
        lake.persist_raw("600001", frame)
        table = pd.read_csv(lake.raw_path("600001"), encoding="utf-8-sig", dtype={"symbol": str})
        assert "2025-01-03" not in set(table["date"])  # 坏行被剔除
        assert "2099-01-01" not in set(table["date"])  # 未来行被剔除
        assert set(table["date"]) == {"2025-01-02", "2025-01-06"}


def test_split_holdout_boundary_and_lock():
    with tmpdir("lake-split") as root:
        lake = _lake(root)
        lake.init()
        lake.persist_raw("600002", _bars(_DATES))
        summary = lake.split_holdout()
        assert summary["files"] == 1
        hold = pd.read_csv(lake.holdout_path("600002"), encoding="utf-8-sig", dtype={"symbol": str})
        assert set(hold["date"]) == {"2025-01-02", "2025-01-03"}
        assert lake.lock_path.is_file()
        assert "最终校核集" in lake.lock_path.read_text(encoding="utf-8")


def test_holdout_read_requires_explicit_unlock():
    with tmpdir("lake-lock") as root:
        lake = _lake(root)
        lake.init()
        lake.persist_raw("600003", _bars(_DATES))
        lake.split_holdout()
        try:
            lake.read("600003", kind="holdout")
        except PermissionError as exc:
            assert "最终校核" in str(exc)
        else:
            raise AssertionError("holdout 未加锁也能读")
        unlocked = _lake(root, allow_holdout=True)
        frame = unlocked.read("600003", kind="holdout")
        assert len(frame) == 2


def test_processed_excludes_holdout_and_computes_fields():
    with tmpdir("lake-process") as root:
        lake = _lake(root)
        lake.init()
        lake.persist_raw("600004", _bars(_DATES))
        lake.split_holdout()
        summary = lake.build_processed()
        assert summary["files"] == 1
        train = pd.read_csv(lake.processed_path("600004"), encoding="utf-8-sig", parse_dates=["date"], dtype={"symbol": str})
        assert train["date"].max() < pd.Timestamp(HOLDOUT_START)
        assert len(train) == 3
        assert train["prev_close"].iloc[0] != train["prev_close"].iloc[0]  # 首行 NaN
        assert (train["tradable"] == 1).all()
        quality = lake.meta / "quality.jsonl"
        assert quality.is_file()
        record = json.loads(quality.read_text(encoding="utf-8").splitlines()[0])
        assert record["symbol"] == "600004" and record["rows"] == 3


def test_load_frames_bulk_and_skips_missing():
    with tmpdir("lake-frames") as root:
        lake = _lake(root)
        lake.init()
        lake.persist_raw("600005", _bars(_DATES))
        lake.persist_raw("600006", _bars(_DATES))
        lake.split_holdout()
        lake.build_processed()
        frames = lake.load_frames(["600005", "600006", "600999"], kind="processed")
        assert set(frames) == {"600005", "600006"}
        one = frames["600005"]
        assert one.index.name == "date" and "symbol" not in one.columns
        assert {"open", "high", "low", "close", "volume"}.issubset(one.columns)
        # 时间窗裁剪
        windowed = lake.load_frames(["600005"], kind="processed", start=_DATES[1])
        assert len(windowed["600005"]) == 2
        # holdout 不开锁时抛权限错误，而不是被当成"文件缺失"静默吞掉
        try:
            lake.load_frames(["600005"], kind="holdout")
        except PermissionError:
            pass
        else:
            raise AssertionError("load_frames 吞掉了 holdout 的权限错误")


def test_plan_resumes_and_honours_staleness():
    with tmpdir("lake-plan") as root:
        lake = _lake(root)
        lake.init()
        lake.persist_raw("600005", _bars(_DATES))
        plan = lake.plan(["600005", "600006"])
        assert plan["skipped"] == ["600005"]
        assert plan["todo"] == ["600006"]
        assert lake.plan(["600005"], resume=False)["todo"] == ["600005"]
        # 追加一条很久以前的日志 → 视为过期，需要重抓
        lake._append_log({"ts": "2020-01-01T00:00:00+08:00", "symbol": "600005", "rows": 5})
        lake.persist_raw("600007", _bars(_DATES))
        plan2 = lake.plan(["600005"], max_age_days=7.0)
        assert plan2["todo"] == ["600005"]


def test_verify_catches_processed_leaking_holdout():
    with tmpdir("lake-verify") as root:
        lake = _lake(root)
        lake.init()
        lake.persist_raw("600008", _bars(_DATES))
        lake.build_processed()
        assert lake.verify()["issue_count"] == 0
        leak = pd.read_csv(lake.processed_path("600008"), encoding="utf-8-sig", dtype={"symbol": str})
        leak = pd.concat(
            [
                leak,
                pd.DataFrame(
                    [
                        {
                            "symbol": "600008",
                            "date": "2025-01-02",
                            "open": 1.0,
                            "high": 1.1,
                            "low": 0.9,
                            "close": 1.0,
                            "volume": 100,
                            "amount": 100.0,
                            "prev_close": 1.0,
                            "tradable": 1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        leak.to_csv(lake.processed_path("600008"), index=False, encoding="utf-8-sig")
        report = lake.verify()
        assert report["issue_count"] == 1
        assert report["issues"][0]["kind"] == "processed"


def test_write_universe_normalizes_codes():
    with tmpdir("lake-universe") as root:
        lake = _lake(root)
        lake.init()
        frame = pd.DataFrame({"代码": ["1", "600000", "600000", "300750"], "名称": ["a", "b", "c", "d"]})
        lake.write_universe(frame)
        symbols = lake.read_universe()
        assert "000001" in symbols and "600000" in symbols and "300750" in symbols
        assert len(symbols) == 3  # 去重


def test_status_counts_files():
    with tmpdir("lake-status") as root:
        lake = _lake(root)
        lake.init()
        lake.persist_raw("600009", _bars(_DATES))
        lake.split_holdout()
        lake.build_processed()
        data = lake.manifest()
        assert data["raw"]["files"] == 1
        assert data["holdout"]["files"] == 1
        assert data["processed"]["files"] == 1
        assert data["lock"] is True


def test_cli_init_and_unknown_action():
    from quant.lake import main

    with tmpdir("lake-cli") as root:
        assert main(["init", "--root", str(root)]) == 0
        assert (root / "README.md").is_file()
        assert main(["nope", "--root", str(root)]) == 1
        assert main([]) == 1
