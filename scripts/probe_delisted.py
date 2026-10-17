"""退市股 / ST 改名的免费源探针（一次性，30–60 分钟内跑完，不阻塞因子主线）。

回答三个问题，全部写进 ``results/probe-delisted.md``：

1. 已退市股票在免费源里还能不能取到历史日线？取到的最后日期是哪天（≈退市日）？
2. 东财的「名称变更」接口能否拿到 ST/*ST/退市警示的历史区间？字段长什么样？
3. 如果两者都拿不到，因子研究只能声明「当期存活样本」，这个声明要怎么写。

    python -m quant probe-delisted
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DELISTED = {
    "600001": ("邯郸钢铁", "2009-12 吸并退市"),
    "000003": ("PT金田A", "2002-06 退市"),
    "600087": ("退市长油", "2014-06 退市"),
    "000033": ("新都退", "2019-07 退市"),
    "000405": ("ST鑫光", "2005-12 退市"),
}
LIVE_FOR_NAME_TEST = ["000004", "600078", "600145"]

UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, timeout: float = 12.0) -> tuple[int, str]:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def probe_eastmoney(symbol: str) -> dict:
    market = "1" if symbol.startswith(("6", "9")) else "0"
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get?"
        f"secid={market}.{symbol}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56&klt=101&fqt=0"
        "&beg=19900101&end=20500101&lmt=10000"
    )
    try:
        status, text = _get(url)
        data = json.loads(text).get("data") or {}
        klines = data.get("klines") or []
        return {
            "ok": bool(klines),
            "status": status,
            "name": data.get("name"),
            "rows": len(klines),
            "first": klines[0].split(",")[0] if klines else None,
            "last": klines[-1].split(",")[0] if klines else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def probe_tencent(symbol: str) -> dict:
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    out: dict = {}
    for label, path, key_hint in (
        ("kline/kline", "kline/kline", "day"),
        ("fqkline/get", "fqkline/get", "qfqday"),
    ):
        url = f"https://web.ifzq.gtimg.cn/appstock/app/{path}?param={prefix}{symbol},day,,,640,qfq"
        try:
            status, text = _get(url)
            payload = json.loads(text)
            node = (payload.get("data") or {}).get(prefix + symbol) or {}
            rows = node.get(key_hint) or node.get("day") or []
            out[label] = {
                "status": status,
                "code": payload.get("code"),
                "rows": len(rows),
                "first": rows[0][0] if rows else None,
                "last": rows[-1][0] if rows else None,
            }
        except Exception as exc:  # noqa: BLE001
            out[label] = {"error": f"{type(exc).__name__}: {exc}"}
        time.sleep(0.4)
    return out


def probe_sina(symbol: str) -> dict:
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    url = (
        "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?"
        f"symbol={prefix}{symbol}&scale=240&ma=no&datalen=1023"
    )
    try:
        status, text = _get(url)
        rows = json.loads(text)
        return {
            "ok": bool(rows),
            "status": status,
            "rows": len(rows) if isinstance(rows, list) else 0,
            "first": rows[0].get("day") if rows else None,
            "last": rows[-1].get("day") if rows else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def probe_name_change(symbol: str) -> dict:
    """东财 F10 名称变更（ST 区间的主要免费来源）。"""
    params = {
        "reportName": "RPT_F10_NAME_CHANGE",
        "columns": "SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,CHANGE_DATE,NAME_BEFORE,NAME_AFTER",
        "pageSize": "50",
        "pageNumber": "1",
        "sortColumns": "CHANGE_DATE",
        "sortTypes": "-1",
        "filter": f'(SECURITY_CODE="{symbol}")',
        "source": "HSF10",
        "client": "PC",
    }
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get?" + urllib.parse.urlencode(params)
    try:
        status, text = _get(url)
        payload = json.loads(text)
        data = (payload.get("result") or {}).get("data") or []
        return {
            "ok": bool(data),
            "status": status,
            "rows": len(data),
            "sample": data[:5],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    out: dict = {"delisted": {}, "name_change": {}, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    for symbol, (name, note) in DELISTED.items():
        print(f"-- {symbol} {name}（{note}）", flush=True)
        out["delisted"][symbol] = {
            "name": name,
            "note": note,
            "eastmoney": probe_eastmoney(symbol),
            "tencent": probe_tencent(symbol),
            "sina": probe_sina(symbol),
        }
        time.sleep(0.3)

    for symbol in LIVE_FOR_NAME_TEST:
        print(f"-- 名称变更接口 {symbol}", flush=True)
        out["name_change"][symbol] = probe_name_change(symbol)
        time.sleep(0.3)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe-delisted.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 退市股 / ST 改名 免费源探针", "", f"- 时间：{out['generated_at']}", ""]
    lines.append("## 1) 已退市股票的历史日线")
    lines.append("")
    lines.append("| 代码 | 名称 | 东财 push2his | 腾讯 kline | 腾讯 fqkline | 新浪 getKLineData |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for symbol, item in out["delisted"].items():
        em, tx, sina = item["eastmoney"], item["tencent"], item["sina"]
        def cell_em(node):
            return f"{node.get('rows', 0)} 行 {node.get('first')}~{node.get('last')}" if node.get("ok") else f"失败 {node.get('error', node.get('status', ''))}"
        def cell_tx(node):
            return f"{node.get('rows', 0)} 行 {node.get('first')}~{node.get('last')}" if node.get("rows") else f"空（{node.get('error', node.get('status'))}）"
        def cell_sina(node):
            return f"{node.get('rows', 0)} 行 {node.get('first')}~{node.get('last')}" if node.get("ok") else f"空（{node.get('error', node.get('status'))}）"
        lines.append(
            f"| {symbol} | {item['name']} | {cell_em(em)} | {cell_tx(tx.get('kline/kline', {}))} | "
            f"{cell_tx(tx.get('fqkline/get', {}))} | {cell_sina(sina)} |"
        )
    lines.append("")
    lines.append("## 2) 东财名称变更接口（ST/*ST/退市警示区间）")
    lines.append("")
    for symbol, node in out["name_change"].items():
        lines.append(f"- `{symbol}`：{'成功' if node.get('ok') else '失败'}，{node.get('rows', 0)} 条")
        for row in node.get("sample", [])[:3]:
            lines.append(f"  - {row}")
    (out_dir / "probe-delisted.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写入 {out_dir / 'probe-delisted.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
