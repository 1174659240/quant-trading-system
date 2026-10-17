"""联网行情源的离线测试：代码解析、三个源的解析器、降级链路。

**不联网**——用录制下来的真实返回片段（截断）做输入，所以可复现、CI 友好。
真正的连通性由 `python -m quant doctor` 负责验证。
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _util import setup_console  # noqa: E402

import quant.data.web_source as web  # noqa: E402
from quant.data.base import DataError  # noqa: E402
from quant.data.web_source import (  # noqa: E402
    WebKlineSource,
    parse_eastmoney,
    parse_sina,
    parse_symbol,
    parse_tencent,
    secid,
    tx_symbol,
)

setup_console()


# 录制的真实返回（已截断到两根，字段结构一致）
EASTMONEY_SAMPLE = json.dumps(
    {
        "rc": 0,
        "data": {
            "code": "600519",
            "klines": [
                "2024-01-02,1685.00,1672.00,1690.00,1660.00,32135,5392876800.00,1.43,-1.94,-33.00,0.26",
                "2024-01-03,1670.01,1675.00,1680.00,1661.00,28567,4773120000.00,1.14,0.18,3.00,0.23",
            ],
        },
    }
)

TENCENT_SAMPLE = json.dumps(
    {
        "code": 0,
        "msg": "",
        "data": {
            "sh600519": {
                "qfqday": [
                    ["2024-01-02", "1685.000", "1672.000", "1690.000", "1660.000", "32135.00"],
                    ["2024-01-03", "1670.010", "1675.000", "1680.000", "1661.000", "28567.00"],
                ],
                "qt": {"sh600519": ["1", "贵州茅台"]},
            }
        },
    }
)

SINA_SAMPLE = json.dumps(
    [
        {"day": "2024-01-02", "open": "1685.000", "high": "1690.000", "low": "1660.000",
         "close": "1672.000", "volume": "3213500"},
        {"day": "2024-01-03", "open": "1670.010", "high": "1680.000", "low": "1661.000",
         "close": "1675.000", "volume": "2856700"},
    ]
)


@contextmanager
def patched(table: dict[str, Callable[..., object]]) -> Iterator[None]:
    """把模块里的源函数临时换掉，退出时**一定**还原（避免污染其他测试）。"""
    original = dict(web.SOURCES)
    web.SOURCES.update(table)
    try:
        yield
    finally:
        web.SOURCES.clear()
        web.SOURCES.update(original)


def _raises(value: object) -> Callable[..., object]:
    def call(*args, **kwargs):
        raise RuntimeError(str(value))

    return call


def _returns(frame: object) -> Callable[..., object]:
    def call(*args, **kwargs):
        return frame

    return call


# ---------------------------------------------------------------------------
# 代码解析
# ---------------------------------------------------------------------------


def test_parse_symbol_forms() -> None:
    assert parse_symbol("600519.SH") == ("600519", "SH")
    assert parse_symbol("sh600519") == ("600519", "SH")
    assert parse_symbol("600519") == ("600519", "SH")
    assert parse_symbol("000001.SZ") == ("000001", "SZ")
    assert parse_symbol("sz000001") == ("000001", "SZ")
    assert parse_symbol("300750") == ("300750", "SZ")
    assert parse_symbol("430047.BJ") == ("430047", "BJ")
    assert parse_symbol("830799") == ("830799", "BJ")
    assert secid("000001.SZ") == "0.000001"
    assert secid("600519.SH") == "1.600519"
    assert tx_symbol("600519.SH") == "sh600519"
    assert tx_symbol("000001.SZ") == "sz000001"


def test_parse_symbol_rejects_garbage() -> None:
    for bad in ("", "60051", "abcdef", "600519.XX", "SH12345"):
        try:
            parse_symbol(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} 应该报错")


# ---------------------------------------------------------------------------
# 三个解析器
# ---------------------------------------------------------------------------


def test_parse_eastmoney_rows() -> None:
    frame = parse_eastmoney(EASTMONEY_SAMPLE)
    assert len(frame) == 2
    assert list(frame.columns) == ["date", "open", "close", "high", "low", "volume", "amount"]
    first = frame.iloc[0]
    assert first["open"] == 1685.0
    assert first["close"] == 1672.0
    assert first["volume"] == 32135 * 100  # 手 → 股
    assert first["amount"] == 5392876800.0
    assert frame.iloc[1]["high"] == 1680.0


def test_parse_tencent_rows() -> None:
    frame = parse_tencent(TENCENT_SAMPLE)
    assert len(frame) == 2
    assert "amount" not in frame.columns  # 上游不给成交额
    first = frame.iloc[0]
    assert (first["open"], first["close"], first["high"], first["low"]) == (1685.0, 1672.0, 1690.0, 1660.0)
    assert first["volume"] == 32135 * 100


def test_parse_sina_rows() -> None:
    frame = parse_sina(SINA_SAMPLE)
    assert len(frame) == 2
    assert frame.iloc[0]["volume"] == 3213500  # 新浪已是股
    assert frame.iloc[1]["close"] == 1675.0


def test_parsers_reject_empty() -> None:
    cases = (
        (parse_eastmoney, json.dumps({"data": {}})),
        (parse_tencent, json.dumps({"data": {}})),
        (parse_sina, "[]"),
    )
    for fn, payload in cases:
        try:
            fn(payload)
        except DataError:
            continue
        raise AssertionError(f"{fn.__name__} 空返回应该抛 DataError")


# ---------------------------------------------------------------------------
# 数据源行为
# ---------------------------------------------------------------------------


def test_load_normalizes_and_estimates_amount() -> None:
    source = WebKlineSource(source="tencent", amount_from_typical=True)
    with patched({"tencent": _returns(parse_tencent(TENCENT_SAMPLE))}):
        frame = source.load("600519.SH", "2024-01-01", "2024-12-31")
    assert list(frame.columns)[:6] == ["open", "high", "low", "close", "volume", "amount"]
    assert frame.index.name == "date"
    assert len(frame) == 2
    assert source.last_source == "tencent"
    assert source.amount_estimated is True
    typical = (1690.0 + 1660.0 + 1672.0) / 3
    assert abs(frame.iloc[0]["amount"] - 3213500 * typical) < 1e-6
    assert frame.iloc[0]["prev_close"] == frame.iloc[0]["open"]  # 首行用开盘价兜底
    assert frame.iloc[1]["prev_close"] == 1672.0


def test_eastmoney_amount_kept_as_is() -> None:
    source = WebKlineSource(source="eastmoney")
    with patched({"eastmoney": _returns(parse_eastmoney(EASTMONEY_SAMPLE))}):
        frame = source.load("600519.SH")
    assert frame.iloc[0]["amount"] == 5392876800.0
    assert source.amount_estimated is False  # 官方给的成交额，不是估算


def test_load_falls_back_to_next_source() -> None:
    source = WebKlineSource(source="auto", retries=1, pause=0.0)
    calls: list[str] = []

    def failing(*args, **kwargs):
        calls.append("eastmoney")
        raise RuntimeError("模拟对端断开")

    def good(*args, **kwargs):
        calls.append("tencent")
        return parse_tencent(TENCENT_SAMPLE)

    with patched({"eastmoney": failing, "tencent": good}):
        frame = source.load("600519.SH")
    assert len(frame) == 2
    assert source.last_source == "tencent"
    assert calls == ["eastmoney", "tencent"]


def test_load_retries_then_succeeds() -> None:
    source = WebKlineSource(source="eastmoney", retries=3, pause=0.0)
    attempts: list[int] = []

    def flaky(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("RemoteDisconnected")
        return parse_eastmoney(EASTMONEY_SAMPLE)

    with patched({"eastmoney": flaky}):
        frame = source.load("600519.SH")
    assert len(frame) == 2
    assert len(attempts) == 3


def test_load_raises_when_all_sources_fail() -> None:
    source = WebKlineSource(source="auto", retries=1, pause=0.0)
    table = {name: _raises("模拟网络不通") for name in ("eastmoney", "tencent", "sina")}
    with patched(table):
        try:
            source.load("600519.SH")
        except DataError as exc:
            message = str(exc)
            assert "都失败" in message
            assert "模拟网络不通" in message
            assert len(source.errors) == 3
            return
    raise AssertionError("三个源都失败时应该抛 DataError")


def test_date_range_is_clipped() -> None:
    source = WebKlineSource(source="sina")
    with patched({"sina": _returns(parse_sina(SINA_SAMPLE))}):
        frame = source.load("600519.SH", "2024-01-03", None)
        assert len(frame) == 1
        assert frame.index[0].date().isoformat() == "2024-01-03"
        try:
            source.load("600519.SH", "2025-01-01", None)
        except DataError as exc:
            assert "没有数据" in str(exc)
        else:
            raise AssertionError("区间为空时应该抛 DataError")


def test_zero_volume_series_rejected() -> None:
    source = WebKlineSource(source="sina", retries=1, pause=0.0)
    zeros = json.dumps(
        [
            {"day": "2024-01-02", "open": "1", "high": "1", "low": "1", "close": "1", "volume": "0"},
            {"day": "2024-01-03", "open": "1", "high": "1", "low": "1", "close": "1", "volume": "0"},
        ]
    )
    with patched({"sina": _returns(parse_sina(zeros))}):
        try:
            source.load("600519.SH")
        except DataError as exc:
            assert "成交量" in str(exc)
            return
    raise AssertionError("全零成交量应该被拒")


def test_unknown_source_name_rejected() -> None:
    try:
        WebKlineSource(source="bloomberg")
    except ValueError as exc:
        assert "未知数据源" in str(exc)
        return
    raise AssertionError("未知源名应该报错")


def test_plugin_registry_exposes_web_source() -> None:
    from quant import plugins

    assert "web" in plugins.source_names()
    source = plugins.open_source("web", source="tencent", retries=1)
    assert isinstance(source, WebKlineSource)
    assert source.order == ["tencent"]
