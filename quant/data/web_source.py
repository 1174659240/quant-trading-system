"""联网行情源：东方财富 / 腾讯 / 新浪，三源互备 + 重试 + 自动切换。

为什么不用 akshare？akshare 很全，但它把上游接口封成一层，接口一变就整体失效，
而且它在**每次调用**里都可能换源。这里只保留三条最稳的日线通道，自己解析、自己降级，
顺带把"哪个源在什么时候可用"变成可见的事实（`last_source` / doctor 的探测结果）。

三个源的实测特征（2026-10-17，本机）：

====================  ==========================  ==============  =========
源                    接口                         复权            成交额
====================  ==========================  ==============  =========
东财（主）             push2his / klt=101          fqt=0/1/2 可选  官方给（元）
腾讯（备 1）           ifzq fqkline / kline        qfq / 不复权     按典型价估算
新浪（备 2）           CN_MarketData.getKLineData  不复权          按典型价估算
====================  ==========================  ==============  =========

约定：**volume 一律换算成股**（上游以手给的 ×100），`prev_close` 由 close 递推，
估算出来的 amount 会在 `WebKlineSource.amount_estimated` 上标记。
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .base import DataError, DataSource, normalize_frame

__all__ = [
    "WebKlineSource",
    "SOURCES",
    "list_symbols",
    "parse_symbol",
    "secid",
    "tx_symbol",
    "parse_eastmoney",
    "parse_tencent",
    "parse_sina",
    "fetch_eastmoney",
    "fetch_tencent",
    "fetch_sina",
]

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

#: 交易所 → 东财 secid 前缀
_EMID = {"SH": "1", "SZ": "0", "BJ": "0"}
#: 复权方式 → 东财 fqt
_EMFQT = {"none": "0", "qfq": "1", "hfq": "2"}


# ---------------------------------------------------------------------------
# 代码规范化
# ---------------------------------------------------------------------------


def parse_symbol(symbol: str) -> tuple[str, str]:
    """把各种写法的 A 股代码拆成 ``(6 位代码, 交易所)``。

    支持 ``600519.SH`` / ``sh600519`` / ``600519`` / ``000001.SZ`` / ``430047.BJ``；
    没有后缀时按代码前缀猜：6/9 开头＝沪，0/2/3 开头＝深，4/8 开头＝北。
    """
    text = str(symbol).strip().upper().replace(" ", "")
    if not text:
        raise ValueError("代码不能为空")
    for suffix in (".SH", ".SZ", ".BJ", ".SS"):
        if text.endswith(suffix):
            code = text[: -len(suffix)]
            exchange = "SH" if suffix in (".SH", ".SS") else suffix[1:]
            return _check_code(code, exchange, symbol)
    for prefix in ("SH", "SZ", "BJ"):
        if text.startswith(prefix):
            return _check_code(text[len(prefix) :], prefix, symbol)
    code = text
    if len(code) == 6 and code.isdigit():
        if code[:2] == "92":  # 北交所新代码段（920xxx）
            return code, "BJ"
        if code[0] in "69":
            return code, "SH"
        if code[0] in "023":
            return code, "SZ"
        if code[0] in "48":
            return code, "BJ"
    raise ValueError(f"无法解析的标的代码：{symbol}（试试 600519.SH / 000001.SZ 这种写法）")


def _check_code(code: str, exchange: str, original: str) -> tuple[str, str]:
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"无法解析的标的代码：{original}（代码部分应是 6 位数字）")
    if exchange not in _EMID:
        raise ValueError(f"不支持的交易所：{exchange}（支持 SH / SZ / BJ）")
    return code, exchange


def secid(symbol: str) -> str:
    """东财 secid，例如 ``1.600519``。"""
    code, exchange = parse_symbol(symbol)
    return f"{_EMID[exchange]}.{code}"


def tx_symbol(symbol: str) -> str:
    """腾讯/新浪用的小写前缀代码，例如 ``sh600519``。"""
    code, exchange = parse_symbol(symbol)
    return f"{exchange.lower()}{code}"


def _yyyymmdd(value: str | None) -> str:
    if not value:
        return ""
    return pd.Timestamp(value).strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    frame: pd.DataFrame
    source: str
    attempts: int = 1


def _drop_bad_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """丢掉四价有非正值的行（停牌/退市/占位行，上游偶尔给 0）。"""
    prices = frame[["open", "high", "low", "close"]]
    keep = (prices > 0).all(axis=1)
    return frame[keep].reset_index(drop=True)


def _get(url: str, *, timeout: float = 10.0, referer: str = "https://quote.eastmoney.com/") -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Referer": referer, "Accept": "*/*"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 三个源的解析器（纯函数，方便离线测试）
# ---------------------------------------------------------------------------


def parse_eastmoney(text: str) -> pd.DataFrame:
    """东财 push2his 的 `klines` 字符串数组 → 原始列。

    每条形如 ``日期,开盘,收盘,最高,最低,成交量(手),成交额(元)...``。
    """
    payload = json.loads(text)
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise DataError("东财返回里没有 klines（可能是代码不存在或区间无数据）")
    rows = []
    for item in klines:
        parts = str(item).split(",")
        if len(parts) < 7:
            continue
        rows.append(
            {
                "date": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]) * 100.0,  # 手 → 股
                "amount": float(parts[6]),
            }
        )
    if not rows:
        raise DataError("东财 klines 解析后为空")
    frame = _drop_bad_rows(pd.DataFrame(rows))
    if frame.empty:
        raise DataError("东财 klines 全是无效价格（停牌/占位行）")
    return frame


def parse_tencent(text: str) -> pd.DataFrame:
    """腾讯 ifzq 的 ``qfqday`` / ``day`` 数组 → 原始列。

    每条形如 ``[日期, 开, 收, 高, 低, 成交量(手), ...]``；成交额缺失，由调用方估算。
    """
    payload = json.loads(text)
    data = payload.get("data") or {}
    if not data:
        raise DataError("腾讯返回里没有 data")
    node = next(iter(data.values()))
    series = node.get("qfqday") or node.get("day") or node.get("hfqday") or []
    if not series:
        raise DataError("腾讯返回里没有 qfqday/day 序列")
    rows = []
    for item in series:
        if len(item) < 6:
            continue
        rows.append(
            {
                "date": item[0],
                "open": float(item[1]),
                "close": float(item[2]),
                "high": float(item[3]),
                "low": float(item[4]),
                "volume": float(item[5]) * 100.0,  # 手 → 股
            }
        )
    if not rows:
        raise DataError("腾讯序列解析后为空")
    frame = _drop_bad_rows(pd.DataFrame(rows))
    if frame.empty:
        raise DataError("腾讯序列全是无效价格（停牌/占位行）")
    return frame


def parse_sina(text: str) -> pd.DataFrame:
    """新浪 getKLineData 的 JSON 数组 → 原始列（volume 上游已是股）。"""
    payload = json.loads(text)
    if not isinstance(payload, list) or not payload:
        raise DataError("新浪返回不是非空数组（代码可能不对）")
    rows = []
    for item in payload:
        rows.append(
            {
                "date": item.get("day"),
                "open": float(item["open"]),
                "close": float(item["close"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "volume": float(item["volume"]),
            }
        )
    frame = _drop_bad_rows(pd.DataFrame(rows))
    if frame.empty:
        raise DataError("新浪序列全是无效价格（停牌/占位行）")
    return frame


# ---------------------------------------------------------------------------
# 三个源的抓取器
# ---------------------------------------------------------------------------


def fetch_eastmoney(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    *,
    adjust: str = "qfq",
    timeout: float = 10.0,
) -> pd.DataFrame:
    if adjust not in _EMFQT:
        raise ValueError(f"adjust 只能是 {sorted(_EMFQT)}，收到 {adjust}")
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={secid(symbol)}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&klt=101&fqt={_EMFQT[adjust]}&beg={_yyyymmdd(start) or '0'}&end={_yyyymmdd(end) or '20500101'}"
    )
    return parse_eastmoney(_get(url, timeout=timeout))


def fetch_tencent(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    *,
    adjust: str = "qfq",
    timeout: float = 10.0,
    count: int = 640,
    max_pages: int = 8,
) -> pd.DataFrame:
    """腾讯日线；单次最多约 640 根，长区间会**自动向前翻页**。"""
    code = tx_symbol(symbol)
    start_ts = pd.Timestamp(start) if start else None
    end_text = pd.Timestamp(end).strftime("%Y-%m-%d") if end else ""
    frames: list[pd.DataFrame] = []
    for _page in range(max(1, max_pages)):
        url = _tencent_url(code, adjust, end_text, count)
        frame = parse_tencent(_get(url, timeout=timeout, referer="https://gu.qq.com/"))
        frames.append(frame)
        oldest = pd.Timestamp(frame["date"].min())
        if start_ts is not None and oldest <= start_ts:
            break
        if len(frame) < count:  # 已经取到该标的最早一根
            break
        end_text = (oldest - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(0.05)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="first").sort_values("date")
    return merged.reset_index(drop=True)


def _tencent_url(code: str, adjust: str, end_text: str, count: int) -> str:
    if adjust in ("qfq", "hfq"):
        return (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={code},day,,{end_text},{count},{adjust}"
        )
    return f"https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={code},day,,{end_text},{count}"


def fetch_sina(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    *,
    adjust: str = "none",
    timeout: float = 10.0,
    datalen: int = 1023,
) -> pd.DataFrame:
    """新浪只有不复权日线（前 1023 根），区间由本地裁剪。"""
    url = (
        "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        f"?symbol={tx_symbol(symbol)}&scale=240&ma=no&datalen={max(1, min(int(datalen), 1023))}"
    )
    return parse_sina(_get(url, timeout=timeout, referer="https://finance.sina.com.cn/"))


SOURCES: dict[str, Callable[..., pd.DataFrame]] = {
    "eastmoney": fetch_eastmoney,
    "tencent": fetch_tencent,
    "sina": fetch_sina,
}

#: 除了主源，谁提供成交额（其余按典型价估算）
_HAS_AMOUNT = {"eastmoney"}


# ---------------------------------------------------------------------------
# 数据源实现
# ---------------------------------------------------------------------------


@dataclass
class WebKlineSource(DataSource):
    """联网日线源：默认按 东财 → 腾讯 → 新浪 的顺序尝试，谁通用谁。

    参数
    ----
    source
        ``"auto"``（默认，按顺序降级）或指定某一个：``eastmoney`` / ``tencent`` / ``sina``。
    adjust
        复权方式：``qfq`` / ``hfq`` / ``none``。新浪只支持 ``none``（会照抓，但复权口径不同）。
    retries
        每个源的重试次数（行情接口偶发 `RemoteDisconnected`，重试一次通常就好）。
    pause
        重试之间的等待秒数。
    amount_from_typical
        上游没给成交额时，用 ``volume × (high+low+close)/3`` 估算（会在 `amount_estimated` 标记）。
    """

    source: str = "auto"
    adjust: str = "qfq"
    retries: int = 3
    timeout: float = 10.0
    pause: float = 1.0
    amount_from_typical: bool = True
    verbose: bool = False
    name: str = "web"
    last_source: str = ""
    amount_estimated: bool = False
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source != "auto" and self.source not in SOURCES:
            raise ValueError(f"未知数据源 {self.source}，可选：auto / {', '.join(SOURCES)}")
        self.order = list(SOURCES) if self.source == "auto" else [self.source]

    def fetch_raw(self, symbol: str, start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, str]:
        """按顺序尝试各源，返回 ``(原始列 DataFrame, 源名)``。"""
        problems: list[str] = []
        for name in self.order:
            fetcher = SOURCES[name]
            for attempt in range(1, max(1, self.retries) + 1):
                try:
                    frame = fetcher(symbol, start, end, adjust=self.adjust, timeout=self.timeout)
                    if self.verbose and (attempt > 1 or name != self.order[0]):
                        print(f"  [{symbol}] 源 {name} 成功（第 {attempt} 次尝试）")
                    return frame, name
                except Exception as exc:  # noqa: BLE001 - 任何失败都降级
                    note = f"{name}#{attempt}: {type(exc).__name__}: {exc}"
                    problems.append(note)
                    if self.verbose:
                        print(f"  [{symbol}] {note}")
                    if attempt < self.retries:
                        time.sleep(self.pause)
        self.errors = problems
        raise DataError(f"{symbol}: 三个行情源都失败了：" + " | ".join(problems[-6:]))

    def load(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """抓单只标的的日线，返回契约格式（volume 单位＝股）。"""
        raw, name = self.fetch_raw(symbol, start, end)
        self.last_source = name
        frame = raw.copy()
        if "amount" not in frame.columns and self.amount_from_typical:
            typical = frame[["high", "low", "close"]].mean(axis=1)
            frame["amount"] = frame["volume"] * typical
            self.amount_estimated = True
        else:
            self.amount_estimated = name not in _HAS_AMOUNT
        frame = normalize_frame(frame, symbol=symbol)
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index <= pd.Timestamp(end)]
        if len(frame) == 0:
            raise DataError(f"{symbol}: 区间 {start}~{end} 内没有数据（源 {name}）")
        if (frame["volume"] <= 0).mean() > 0.2:
            raise DataError(f"{symbol}: 超过两成交易日成交量为 0（源 {name} 数据可疑，可能停牌或代码错误）")
        return frame

    def symbols(self) -> Iterable[str]:
        """全市场代码（新浪列表接口；想限量请直接用 :func:`list_symbols`）。"""
        return list_symbols()


def list_symbols(limit: int | None = None, *, timeout: float = 10.0, page_size: int = 100) -> list[str]:
    """从新浪的沪深 A 股列表接口取全市场代码（``600519.SH`` 形式），按代码升序。

    ``limit`` 只取前 N 只（列表按代码升序，想要流动性好的样本建议自己按成交额筛）。
    """
    out: list[str] = []
    page = 1
    while True:
        url = (
            "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
            f"?page={page}&num={page_size}&sort=symbol&asc=1&node=hs_a"
        )
        text = _get(url, timeout=timeout, referer="https://vip.stock.finance.sina.com.cn/")
        text = text.strip()
        if not text or text in ("null", "[]"):
            break
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            break
        if not items:
            break
        for item in items:
            try:
                code, exchange = parse_symbol(str(item.get("symbol", "")))
            except ValueError:
                continue
            out.append(f"{code}.{exchange}")
            if limit is not None and len(out) >= limit:
                return out
        if len(items) < page_size:
            break
        page += 1
        if page > 120:  # 保险丝，别把接口问崩
            break
    return out


def fetch_many(
    symbols: Iterable[str],
    *,
    start: str | None = None,
    end: str | None = None,
    adjust: str = "qfq",
    retries: int = 3,
    timeout: float = 10.0,
    pause: float = 0.15,
    source: str = "auto",
    verbose: bool = True,
    second_pass: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], list[str]]:
    """批量抓取，返回 ``(数据, 每只用的源, 失败原因)``；单只失败不中断整批。

    失败会自动补抓一轮（`second_pass`）——实测本机会出现**整批 DNS 瞬时失败**
    （`getaddrinfo failed`），隔几秒再来一次就好，没必要重跑整个命令。
    """
    web = WebKlineSource(source=source, adjust=adjust, retries=retries, timeout=timeout, verbose=verbose)
    frames: dict[str, pd.DataFrame] = {}
    used: dict[str, str] = {}
    failures: list[str] = []
    items = [str(s) for s in symbols]
    pending = list(items)

    def _pass(targets: list[str], round_no: int) -> list[str]:
        still: list[str] = []
        for i, symbol in enumerate(targets, 1):
            try:
                frame = web.load(symbol, start, end)
            except Exception as exc:  # noqa: BLE001
                still.append(f"{symbol}: {exc}")
                if verbose:
                    print(f"[{round_no}/{i}/{len(targets)}] {symbol} 失败：{str(exc)[:160]}")
                continue
            frames[symbol] = frame
            used[symbol] = web.last_source
            if verbose:
                print(
                    f"[{round_no}/{i}/{len(targets)}] {symbol} {len(frame)} 根 · 源 {web.last_source} · "
                    f"{frame.index[0].date()}~{frame.index[-1].date()}"
                    + ("（成交额估算）" if web.amount_estimated else "")
                )
            if pause:
                time.sleep(pause)
        return still

    failures = _pass(pending, 1)
    if failures and second_pass:
        retry_symbols = [item.split(":", 1)[0] for item in failures]
        if verbose:
            print(f"\n{len(retry_symbols)} 只失败，等 3 秒后补抓一轮…")
        time.sleep(3.0)
        failures = _pass(retry_symbols, 2)
    return frames, used, failures
