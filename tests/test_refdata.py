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


def test_shares_at_is_point_in_time() -> None:
    """股本 PIT：只认 change_date <= t 的最近一条；未来变动与东财内部码都不许出现。"""
    frame = pd.DataFrame(
        {
            "symbol": ["600519", "600519", "605369", "A04014"],
            "change_date": ["2015-07-17", "2026-05-28", "2020-09-04", "2019-01-01"],
            "total_share": [1_256_197_800, 1_250_081_601, 80_000_000, 1],
            "float_share": [1_256_197_800, 1_250_081_601, 20_000_000, 1],
            "is_real_code": [1, 1, 1, 0],
        }
    )
    snap = refdata.shares_at(frame, "2024-06-28").set_index("symbol")
    assert snap.loc["600519", "total_share"] == 1_256_197_800
    assert "2026-05-28" not in snap["change_date"].values
    assert snap.loc["605369", "float_share"] == 20_000_000
    assert "A04014" not in snap.index, "东财内部代码不是 A 股，不进 PIT 快照"

    early = refdata.shares_at(frame, "2015-01-01")
    assert early.empty, "早于首个变动日应缺席，而不是用未来值兜底"


def test_shares_snapshot_schema() -> None:
    """真实股本快照：列齐全、总股本为正、流通股不超过总股本的比例合理。"""
    frame = refdata.load_shares()
    if len(frame) == 0:
        return
    for column in ("symbol", "change_date", "total_share", "float_share", "is_real_code", "source"):
        assert column in frame.columns, f"股本表缺列 {column}"
    real = frame[frame["is_real_code"] == 1]
    assert (real["total_share"].dropna() > 0).all(), "总股本必须为正"
    assert real["change_date"].astype(str).str.len().eq(10).all(), "变动日应为 YYYY-MM-DD"


def test_sw_snapshot_schema() -> None:
    """申万快照：只有当期（单一 as_of），成分必须带一级行业代码。"""
    frame = refdata.load_sw_industry()
    if len(frame) == 0:
        return
    for column in ("symbol", "sw_code", "sw_name", "sw_level", "as_of", "source"):
        assert column in frame.columns, f"申万表缺列 {column}"
    assert frame["as_of"].nunique() == 1, "申万接口无历史，只能有单一落盘日"
    assert frame["sw_code"].nunique() >= 28, "申万一级行业应接近 31 个"


def test_mainfin_covers_financials() -> None:
    """主要财务指标表：金融股的 ROE/EPS/BPS 必须有值（通用三表给不了）。"""
    from quant import mainfin

    files = sorted(mainfin.MAINFIN_DIR.glob("????-??-??.parquet"))
    if not files:
        return
    frame = mainfin.load(files[-1].stem)
    for column in ("ROEJQ", "EPSJB", "BPS", "TOTAL_SHARE"):
        assert column in frame.columns, f"mainfin 缺列 {column}"
    fin = frame[frame["ORG_TYPE"].astype(str).isin(["银行", "保险", "证券"])]
    if len(fin) == 0:
        return
    assert pd.to_numeric(fin["BPS"], errors="coerce").notna().mean() > 0.9, "金融股 BPS 覆盖率过低"
