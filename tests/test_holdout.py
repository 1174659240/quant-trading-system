"""holdout 层（2025-01-01 起）隔离性测试。

核心不变量：样本内读取路径不得看到 2025+ 数据；holdout 文件必须显式指定才读；
数据层冻结的 totals 不得因 holdout 落盘而变化。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant import dividends, events, valuation
from scripts import freeze_datalayer

HOLDOUT_ROOT = Path("D:/quant-data/holdout")


def test_valuation_root_paths_are_separate() -> bool:
    in_sample = valuation.path_for("600519")
    holdout = valuation.path_for("600519", root=valuation.HOLDOUT_DIR)
    assert in_sample != holdout
    assert "holdout" in str(holdout)
    assert valuation.HOLDOUT_DIR != valuation.VALUATION_DIR
    return True


def test_dividends_default_excludes_holdout() -> bool:
    frame = dividends.load()
    if len(frame) == 0:
        return True
    ex = frame["ex_date"].dropna()
    assert ex.max() <= events.CUTOFF, f"样本内分红出现 {ex.max()}"
    if dividends.HOLDOUT_DIVIDENDS.is_file():
        with_holdout = dividends.load(include_holdout=True)
        assert len(with_holdout) >= len(frame)
    return True


def test_events_holdout_files_registered() -> bool:
    assert set(events.HOLDOUT_FILES) == set(events.TABLES)
    assert all(str(path).startswith(str(events.HOLDOUT_DIR)) for path in events.HOLDOUT_FILES.values())
    assert events.HOLDOUT_DIR != events.EVENTS_DIR
    return True


def test_events_default_load_hides_holdout_rows() -> bool:
    frame = events.load("lhb")
    if len(frame) == 0:
        return True
    assert frame["available_date"].max() <= events.CUTOFF
    return True


def test_freeze_totals_ignore_holdout() -> bool:
    plain = freeze_datalayer.collect()
    with_holdout = freeze_datalayer.collect(include_holdout=True)
    assert plain["totals"] == with_holdout["totals"], "holdout 不得改变样本内冻结 totals"
    assert "holdout" in with_holdout
    return True


def test_holdout_price_layer_is_locked() -> bool:
    lock = HOLDOUT_ROOT / ".LOCKED"
    if not HOLDOUT_ROOT.is_dir():
        return True
    assert lock.is_file(), "holdout 目录必须带 .LOCKED 标记"
    text = lock.read_text(encoding="utf-8", errors="replace")
    assert "2025-01-01" in text
    return True
