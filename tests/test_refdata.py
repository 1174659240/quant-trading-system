"""参考数据层测试：universe 区间/行业 asof/指数成分/退市处置假设，全部离线。"""

from __future__ import annotations

import pandas as pd

from quant import delisted, index_members, refdata

from _util import setup_console

setup_console()


def test_board_of() -> None:
    cases = {
        "600519": "主板",
        "601318": "主板",
        "000001": "主板",
        "002594": "主板",
        "300750": "创业板",
        "301269": "创业板",
        "688111": "科创板",
        "689009": "科创板",
        "830799": "北交所",
        "920819": "北交所",
        "900901": "B股",
        "200002": "B股",
    }
    for code, expected in cases.items():
        actual = refdata.board_of(code)
        assert actual == expected, f"{code} 板块应为 {expected}，实际 {actual}"


def test_universe_semantics() -> None:
    """区间表语义：在途退市不算已退市，两者互斥。"""
    frame = pd.DataFrame(
        {
            "symbol": ["600001", "601059", "600519"],
            "status": [0, 0, 1],
            "out_date": ["2009-12-29", "2026-12-31", ""],
        }
    )
    as_of = "2026-10-17"
    has_out = frame["out_date"].fillna("").astype(str)
    pending = (frame["status"] == 0) & (has_out > as_of)
    delisted_flag = (frame["status"] == 0) & (has_out != "") & (has_out <= as_of)
    assert not (pending & delisted_flag).any(), "在途退市与已退市必须互斥"
    assert delisted_flag.tolist() == [True, False, False]
    assert pending.tolist() == [False, True, False]


def test_industry_asof_no_lookahead() -> None:
    """行业 asof 只能取 <= 当日的最近一版，不能用到未来版本。"""
    frame = pd.DataFrame(
        {
            "symbol": ["600733", "600733", "600733"],
            "as_of": ["2017-12-31", "2018-12-31", "2020-12-31"],
            "industry": ["K70房地产业", "C36汽车制造业", "C36汽车制造业"],
        }
    )
    picked = refdata.industry_asof(frame, "2018-06-30")
    assert len(picked) == 1
    assert picked["industry"].iloc[0] == "K70房地产业"
    assert picked["as_of"].iloc[0] == "2017-12-31"

    later = refdata.industry_asof(frame, "2019-01-01")
    assert later["industry"].iloc[0] == "C36汽车制造业"


def test_delisting_return_modes() -> None:
    assert delisted.delisting_return("600087", "last") == 0.0
    assert delisted.delisting_return("600087", "haircut30") == -0.30
    assert delisted.delisting_return("600087", "total_loss") == -1.0
    try:
        delisted.delisting_return("600087", "unknown")
    except ValueError:
        pass
    else:  # pragma: no cover - 防御性
        raise AssertionError("未知退市处置假设应报 ValueError")


def test_index_members_frame_schema() -> None:
    """指数成分表至少要有这些列（数据未落盘时跳过，不让测试依赖数据湖）。"""
    frame = index_members.load()
    if len(frame) == 0:
        return
    for column in ("index_code", "index_name", "symbol", "as_of", "source"):
        assert column in frame.columns, f"指数成分缺列 {column}"
    assert frame["symbol"].astype(str).str.len().max() <= 6


def test_delisted_path_and_events_schema() -> None:
    path = delisted.path_for("000033")
    assert path.name == "000033.parquet"
    events = delisted.EVENTS_CSV
    assert events.name == "events.csv"
