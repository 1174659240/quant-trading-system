# 存储格式核对：Parquet 迁移口径逐条对照一手来源

- 日期：2026-10-17
- 目的：把「Parquet 主存储改造」的每条口径对照一手来源核对；与来源不符的写明，不凭想象落地。
- 来源快照：`D:\workspace\_diag\sources-20261017\`（当日抓取的 HTML/MD 原文，离线可复核）。
- 阅读方式：本文件是核对记录，不是新方案；最终落地以 `docs/数据层改造方案.md` 的修订版为准。

## 1. 一手原文（逐字摘录）

### 1.1 Parquet 官方（Row Group Size）
> We recommend large row groups (512MB - 1GB). Since an entire row group might need to be read, we want it to completely fit on one HDFS block... An optimized read setup would be: 1GB row groups, 1GB HDFS block size, 1 HDFS block per HDFS file.

来源：<https://parquet.apache.org/docs/file-format/configurations/>（快照 `parquet-configs.html`）。注意这段建议的锚点就是 **HDFS block**。

### 1.2 DuckDB（row group / 文件大小）
> DuckDB works best on Parquet files with row groups of 100K-1M rows each. The reason for this is that DuckDB can only parallelize over row groups...

基准表（同一聚合查询）：960 行 8.77s；3,840 行 4.33s；15,360 行 1.58s；61,440 行 0.94s；**122,880 行 0.87s（最好）**；245,760 行 0.93s；491,520 行 0.95s；983,040 行 0.97s；1,966,080 行 0.88s。原文结论：<5,000 行差 5–10 倍，5,000–20,000 行差 1.5–2.5 倍，**>100,000 行差异约 10%**。

> The ideal range is between 100 MB and 10 GB per individual Parquet file.

来源：<https://duckdb.org/docs/lts/guides/performance/file_formats>（快照 `duckdb-fileformats.html`）。

### 1.3 DuckDB Parquet Tips
- `ROW_GROUP_SIZE` 默认 **122,880**；每文件 row group 数 ≥ 查询线程数；组越大压缩机会越多、写内存越大、窄时间范围查询读放大越多。
- 来源：<https://www.duckdb.org/docs/lts/data/parquet/tips>。

### 1.4 MotherDuck（对象存储导入）
- row group 100K–1M 行；每文件 100K–10M 行；单文件 **10–256 MB 最优**；>256 MB 仍可用，但“DuckDB 在一个文件内只能按 row group 并行”。
- 压缩：Snappy（默认，解压快）或 ZSTD（压缩率更高，适合冷存储）。
- 来源：<https://motherduck.com/docs/key-tasks/cloud-storage/s3-import-best-practices/>。

### 1.5 Dask Hive 分区
> Avoid High Cardinality ... avoid partitioning on ... any column containing many unique values ... if you try to partition on a column with millions of unique values, then to_parquet will need to generate millions of directories.

来源：<https://docs.dask.org/en/latest/dataframe-hive.html>（快照 `dask-hive.html`）。

### 1.6 Qlib PIT
> For each feature, it contains 4 columns, i.e. date, period, value, _next.
> date: 数据入库日（决定可见性）；period: 报表期；value: 数值；_next: 下一版本字节索引。
> 4 列 dtype：`date u4 / period u4 / value f8 / _next u4`；只支持季/年频。

来源：<https://qlib.readthedocs.io/en/stable/_sources/advanced/PIT.rst.txt>（快照在 funnel 日志同目录）。

### 1.7 Qlib 数据层（存储形态）
Qlib 主存储是**逐标的 .bin**（列式二进制），CSV/Parquet 只是导入格式（`scripts/dump_bin.py`）。→ 它的分区经验不能直接类比 Parquet 文件布局。

来源：<https://qlib.readthedocs.io/en/latest/component/data.html>（快照 `qlib-data.html`）。

### 1.8 专业交易系统 Nautilus Trader
`ParquetDataCatalog` 的 `max_rows_per_group` **默认 5,000 行**；原文称“对典型交易数据（tick、bar）是良好平衡”，权衡是写内存 vs 读放大。

来源：<https://leeroopedia.com/index.php?title=Heuristic:Nautechsystems_Nautilus_trader_Parquet_Row_Group_Tuning>（二手条目，注明“auto-imported”）。

### 1.9 CRSP（调整价口径）——**未取到一手**
- WRDS 支持页被 Cloudflare 403；CRSP 官方 PDF 跳转 Morningstar；Michigan 课程 PDF 返回 406；NBER 附录是 PDF，环境不支持抓 PDF。
- 因此「不复权价 + 累积因子、调整价 = PRC / CFACPR、退市收益单列」标注为**通行方法学（二手）**，未直接核对 CRSP 文档原文。
- 与我们口径的关系：我们的 `adj_factor` 是**前复权系数（最新日=1）**，`qfq = raw × factor`；CRSP 是 `adjusted = raw / CFACPR`，两者互为倒数关系，方法学同构，且因子可由我们自己的事件表复算（`k=(P0−c)/(1+b)/P0`）。

## 2. 逐条核对结论

| # | 你的口径 | 核对结果 | 结论 |
| --- | --- | --- | --- |
| 1 | processed/holdout 存不复权 OHLC + adj_factor，read 现算 qfq（CRSP 范式） | 方法学方向成立；CRSP 一手未取到（见 1.9） | **采纳**，但标注“CRSP 类比为二手” |
| 2 | processed 从 raw 重建；旧 CSV 与新 parquet 双后端 allclose 5e-4 | 可行 | **采纳**；⚠️ 默认 `read()` 语义必须仍是 qfq（旧 CSV 物理 qfq），仅 `adjust="none"` 读原始 |
| 3 | 按 year 分区（低基数）；symbol 高基数不可分区 | Dask 原文明确 Avoid High Cardinality ✅；DuckDB/Spark 无“禁止”原文，属通用工程共识 | **采纳**：year 分区；raw 按 symbol 目录仅作归档，全市场扫描禁走 raw |
| 4 | 250K 行/组、约 3.8 组/年、9 年约 34 组 ≥ 20 线程 | 算术成立（processed 951 万行/9 年 ≈ 95 万行/年）；DuckDB：245,760 组 0.93s vs 最优 0.87s | **采纳 250K**（可配置；122,880 也几乎等效）。注意：**单年文件只有 ≈4 组**，单年查询只能 4 路并行；34 组是**跨年合计** |
| 5 | zstd lvl3 | MotherDuck 明确 ZSTD 压缩率更好 ✅；zstd 默认等级即 3，无官方“等级”建议 | **采纳**；等级标注为经验值 |
| 6 | 单文件 100MB–10GB（DuckDB 区间） | DuckDB 原文确有 “100 MB and 10 GB” ✅；MotherDuck 对 S3 建议 10–256MB（远程场景） | **采纳**；两个区间场景不同，别混用 |
| 7 | 512MB–1GB 是 HDFS 分布式场景，直接套会让单文件只有 1 个巨型 row group | 官方原文确实围绕 HDFS block ✅；我们全历史约 760MB → 512MB 组只有 1–2 个 | **你的判断成立**：按行数（10 万–25 万/组）而非字节套用 |
| 8 | Qlib PIT 四列 date/period/value/_next，公告日≠报告期 | 原文逐字确认 ✅（1.6） | **采纳**；本阶段无财报数据，schema 预留 `announce_date` |
| 9 | 退市末段收益显式赋值（CRSP delisting return） | 方法学成立（CRSP 一手未取到）；我们的探针：新浪 1023 根 / 腾讯 fqkline 640 根可取，但现役 kline/kline 对退市股返回空 | **本阶段不纳入**（universe 为当期存活），报告标注“幸存者偏差，收益为乐观上界” |
| 10 | 停牌维持“填前收 + suspended=1、不产生假收益” | 与 Qlib 全 NaN 是两种成熟流派，已测 | **维持现状**（你已拍板） |
| 11 | DuckDB 仅验收用、不进引擎 dependencies | 合理；pip 包为 MIT，体积约十几 MB | **建议**：装为可选 dev 工具，`verify` 内 `try/except` 提示，引擎不 import；是否安装听你 |

## 3. 对我原方案的两处修正

1. **row group 数表述**：原方案“9 年约 34 组 ≥ 本机 20 线程”只在**跨年全历史扫描**成立；单年查询只有 ≈4 组。若将来常做单年研究，把组降到 122,880（每年 ≈8 组）或 65,536（≈15 组）更均衡。DuckDB 基准里 122,880 是最优，245,760 仅差 7%，所以**默认 122,880**、可配 250K 更稳妥。
2. **单文件 vs 按年**：DuckDB 本地建议单文件 100MB–10GB，9 年单文件约 760MB 落在区间内；但增量更新要重写整文件、且单文件 row group 只有 ≈34 个。结论不变（按年），但**用实测数据定**，不拍脑袋。

## 4. 未解决 / 待你拍板

- DuckDB 是否安装为验收工具（第 11 条）。
- 退市股是否在下一阶段补抓并纳入（现阶段不纳入，报告已标注）。
- PIT 的 `announce_date` 预留字段是否本阶段就写进 schema（无数据，仅占位）。
