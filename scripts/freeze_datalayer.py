"""数据层冻结指纹：逐层实测文件数/行数/字节数，产出可复核的 JSON。

用途：第二、三步任何结论都引用这里的数字，避免“凭记忆写行数”。
用法： python -m scripts.freeze_datalayer [--out results/datalayer-freeze.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant import delisted, fundamentals, mainfin, refdata, valuation

PROCESSED = Path("D:/quant-data/processed")
REF = refdata.REF_DIR
OUT_DEFAULT = Path("results/datalayer-freeze.json")


def _dir_stats(root: Path, pattern: str = "*") -> dict[str, Any]:
    files = [p for p in sorted(root.glob(pattern)) if p.is_file()]
    return {
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "mb": round(sum(p.stat().st_size for p in files) / 1e6, 1),
    }


def _parquet_rows(root: Path, pattern: str = "*.parquet") -> int:
    """只读 parquet footer 统计行数，不解码数据（5,482 个文件也只要几秒）。"""
    from pyarrow import parquet as pq

    total = 0
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            try:
                total += pq.ParquetFile(path).metadata.num_rows
            except Exception:  # noqa: BLE001 - 损坏文件不影响整体统计
                continue
    return total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _layer(name: str, payload: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"layer": name, **payload, **extra}


def collect() -> dict[str, Any]:
    layers: list[dict[str, Any]] = []

    # 0. 爬取层（原始代码表 + 逐股 CSV + 年度 parquet）
    meta = PROCESSED / "parquet-meta.json"
    crawl_meta = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    layers.append(
        _layer(
            "crawl_daily",
            {"files": int(crawl_meta.get("csv_files", 0)), "rows": int(crawl_meta.get("rows", 0))},
            path=str(PROCESSED / "daily"),
            symbols=int(crawl_meta.get("symbols", 0)) if isinstance(crawl_meta.get("symbols"), int) else None,
            first=crawl_meta.get("first"),
            last=crawl_meta.get("last"),
            parity={
                "files": len([p for p in (PROCESSED / "parquet").glob("year=*/*.parquet") if p.is_file()]),
                "rows": _parquet_rows(PROCESSED / "parquet", "year=*/*.parquet"),
                "mb": round(_dir_stats(PROCESSED / "parquet", "year=*/*.parquet")["bytes"] / 1e6, 1),
            },
            note="新浪/腾讯/东财三源降级爬取，2024-12-31 截止",
        )
    )

    # 1. universe
    uni = refdata.load_universe()
    layers.append(
        _layer(
            "universe",
            {"files": len(list(REF.glob("universe.*"))), "rows": int(len(uni))},
            path=str(REF / "universe.csv"),
            live=int((uni["status"] == 1).sum()),
            delisted=int(uni["is_delisted"].sum()),
            pending=int(uni["is_pending_delist"].sum()),
            boards=uni["board"].value_counts().to_dict(),
            as_of=str(uni["as_of"].iloc[0]),
            source="baostock.query_stock_basic(type=1)",
            pit="区间表：ipo_date / out_date，不物化日面板",
            gap="不含北交所（baostock 无覆盖）；B 股已剔除",
        )
    )

    # 2. 行业 PIT
    industry = refdata.load_industry()
    layers.append(
        _layer(
            "industry",
            {"files": 2, "rows": int(len(industry))},
            path=str(REF / "industry.parquet"),
            versions=int(industry["as_of"].nunique()),
            last_as_of=str(industry["as_of"].max()),
            empty_industry=int(
                (industry["industry"].astype("string").fillna("").str.strip() == "").sum()
            ),
            source="baostock.query_stock_industry(date=年末交易日)",
            pit="年度粒度：classify_date <= as_of，重组可能滞后一年",
            gap="退市股 2025 年后被 baostock 清空行业",
        )
    )

    # 3. 股本 PIT
    shares = refdata.load_shares()
    real = shares[shares["is_real_code"] == 1] if len(shares) else shares
    in_uni = set(real["symbol"]) & set(uni["symbol"]) if len(real) else set()
    layers.append(
        _layer(
            "shares",
            {"files": 2, "rows": int(len(shares))},
            path=str(REF / "shares.csv"),
            symbols=int(real["symbol"].nunique()) if len(real) else 0,
            in_universe=len(in_uni),
            universe_missing=int(len(set(uni["symbol"]) - set(real["symbol"]))) if len(real) else 0,
            internal_codes=int(shares.loc[shares["is_real_code"] == 0, "symbol"].nunique())
            if len(shares)
            else 0,
            source="eastmoney RPT_F10_EH_EQUITY（逐次变动）",
            pit="shares_at(day) 只取 change_date <= day 的最近一条",
            gap="个别记录 float_share > total_share，下游需过滤",
        )
    )

    # 4. 申万快照
    sw = refdata.load_sw_industry()
    layers.append(
        _layer(
            "sw_industry",
            {"files": len(list(REF.glob("sw_*.csv"))) + len(list(REF.glob("sw_*.parquet"))), "rows": int(len(sw))},
            path=str(REF / "sw_industry.csv"),
            industries=int(sw["sw_code"].nunique()) if len(sw) else 0,
            as_of=str(sw["as_of"].iloc[0]) if len(sw) else None,
            source="akshare index_component_sw（裸代码，去掉 .SI）",
            pit="仅当期快照，无历史 → 只作描述",
            gap="不能做历史行业归属",
        )
    )

    # 4b. PIT 分红（任务一股息率用）
    div_path = REF / "dividends.parquet"
    if div_path.exists():
        div = pd.read_parquet(div_path, columns=["symbol", "report_date", "ex_date"])
        layers.append(
            _layer(
                "dividends",
                {"files": 1, "rows": int(len(div))},
                path=str(div_path),
                symbols=int(div["symbol"].nunique()),
                first=str(div["report_date"].min())[:10],
                last=str(div["report_date"].max())[:10],
                source="eastmoney datacenter RPT_SHAREBONUS_DET",
                pit="ex_date 生效，滚动 365 个自然日；report_date/ex_date > 2024-12-31 已裁掉（holdout）",
                gap="无北交所；只有「已实施」行，预案期不可见",
            )
        )

    # 4c. 换手率宽表（valuation 的 turn 列透视，任务一/二共用）
    turn_path = REF / "turn_panel.parquet"
    if turn_path.exists():
        turn = pd.read_parquet(turn_path)
        layers.append(
            _layer(
                "turn_panel",
                {"files": 1, "rows": int(len(turn))},
                path=str(turn_path),
                symbols=int(turn.shape[1] - 1),
                first=str(turn["date"].iloc[0])[:10],
                last=str(turn["date"].iloc[-1])[:10],
                source="valuation/*.parquet 的 turn 列 pivot（2014-01-02~2024-12-31）",
                pit="逐日快照",
                gap="与 valuation 层同源（无北交所）；153 只在 2024-12-31 后上市，窗口内无数据被 pivot 丢弃",
            )
        )

    # 4d. 事件层（龙虎榜/回购/增减持/ST 翻转/送转分红）
    events_dir = Path("D:/quant-data/events")
    if events_dir.exists():
        from quant import events as event_layer

        table = event_layer.summary()
        layers.append(
            _layer(
                "events",
                {"files": int(table["exists"].sum()), "rows": int(table.loc[table["exists"], "rows"].sum())},
                path=str(events_dir),
                symbols=int(table.loc[table["exists"], "symbols"].max()) if len(table) else 0,
                first=str(table["start"].min())[:10],
                last=str(table["end"].max())[:10],
                tables="、".join(f"{row.event}({row.rows})" for row in table.itertuples() if row.exists),
                source="akshare stock_lhb_detail_em / stock_lhb_jgmmtj_em / stock_repurchase_em / "
                       "stock_hold_management_detail_em / stock_fhps_em；ST 由 valuation.isST 翻转",
                pit="双日期：event_date 发生日 + available_date 可见日"
                    "（龙虎榜=上榜日、回购=最新公告日、增减持=变动日+2 交易日、ST=翻转日、分红=预案公告日）；"
                    "研究默认裁 <= 2024-12-31（holdout 锁定）",
                gap="回测/因子只用 available_date；lhb 的 post_* 为未来信息仅供事件研究；"
                    f"hold 有 {int(table.loc[table.event == 'hold', 'pit_missing'].sum())} 行因超出交易日历无可见日",
            )
        )

    # 5. 估值 / isST
    val_root = valuation.VALUATION_DIR
    val_files = sorted(val_root.glob("??????.parquet"))
    layers.append(
        _layer(
            "valuation",
            {**_dir_stats(val_root, "*.parquet"), "rows": _parquet_rows(val_root)},
            path=str(val_root),
            symbols=len(val_files),
            empty_tables=int(
                sum(1 for p in val_files if pd.read_parquet(p, columns=["date"]).empty)
            )
            if val_files
            else 0,
            fields="date,close,preclose,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
            source="baostock.query_history_k_data_plus(adjustflag=3)",
            pit="逐日快照；停牌日沿用最近有效值",
            gap="无北交所；81 只 2014 年前退市股无文件（不是空文件）",
        )
    )

    # 6. 财务四表 + mainfin
    for table in ("profit", "balance", "cashflow", "indicator"):
        root = fundamentals.FUND_DIR / table
        files = sorted(root.glob("????-??-??.parquet"))
        rows = [len(pd.read_parquet(p, columns=["SECURITY_CODE"])) for p in files] if files else []
        complete = [n for n in rows if n > 1000]
        layers.append(
            _layer(
                f"fundamentals_{table}",
                {**_dir_stats(root, "*.parquet"), "rows": _parquet_rows(root)},
                path=str(root),
                periods=len(files),
                complete_periods=len(complete),
                first=files[0].stem if files else None,
                last=files[-1].stem if files else None,
                source="eastmoney datacenter RPT_F10_FINANCE_*",
                pit="NOTICE_DATE（缺失 +110 天兜底）",
                gap="修订后值；退市后仍披露需按退市日截断",
            )
        )

    mainfin_files = sorted(mainfin.MAINFIN_DIR.glob("????-??-??.parquet"))
    layers.append(
        _layer(
            "mainfin",
            {**_dir_stats(mainfin.MAINFIN_DIR, "*.parquet"), "rows": _parquet_rows(mainfin.MAINFIN_DIR)},
            path=str(mainfin.MAINFIN_DIR),
            periods=len(mainfin_files),
            first=mainfin_files[0].stem if mainfin_files else None,
            last=mainfin_files[-1].stem if mainfin_files else None,
            source="indicator 表 31 列投影（不重抓）",
            pit="同 NOTICE_DATE",
            gap="2026-09-30 仅 51 家披露",
        )
    )

    # 7. 指数成分
    members = pd.read_csv(REF / "index_members.csv", dtype={"symbol": str})
    layers.append(
        _layer(
            "index_members",
            {"files": 2, "rows": int(len(members))},
            path=str(REF / "index_members.csv"),
            indices=int(members["index_code"].nunique()) if len(members) else 0,
            as_of=str(members["as_of"].max()) if len(members) else None,
            source="akshare index_stock_cons_csindex / index_stock_cons",
            pit="仅当期快照",
            gap="无历史成分 → 幸存者偏差",
        )
    )

    # 8. 退市股
    events_path = delisted.EVENTS_CSV
    events = pd.read_csv(events_path, dtype={"symbol": str}) if events_path.exists() else pd.DataFrame()
    layers.append(
        _layer(
            "delisted",
            {**_dir_stats(delisted.DELISTED_DIR / "daily", "*.parquet"), "rows": _parquet_rows(delisted.DELISTED_DIR / "daily")},
            path=str(delisted.DELISTED_DIR / "daily"),
            targets=int(len(events)),
            with_bars=int((events["bars"] > 0).sum()) if len(events) else 0,
            source="baostock 后复权日线（adjustflag=1），2014 起",
            pit="截止退市日；gap_days 中位 0",
            gap="81 只 2014 前退市无日线",
        )
    )

    manifest = REF / "manifest.json"
    counted = [item for item in layers if item["layer"] != "mainfin"]
    totals = {
        "rows": int(sum(item.get("rows", 0) for item in counted)),
        "files": int(sum(item.get("files", 0) for item in counted)),
        "excluded": ["mainfin（indicator 的投影，避免重复计数）"],
    }
    return {
        "frozen_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(timespec="seconds"),
        "manifest": {
            "path": str(manifest),
            "sha256_16": _sha256(manifest) if manifest.exists() else None,
        },
        "totals": totals,
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="数据层冻结指纹")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    payload = collect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"冻结 {len(payload['layers'])} 层 -> {out}")
    for item in payload["layers"]:
        print(
            f"  {item['layer']:<24} files={item.get('files', 0):>6} rows={item.get('rows', 0):>9}"
        )
    print(f"合计 rows={payload['totals']['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
