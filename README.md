# 量化交易系统（A股 · Python 轻量引擎）

一个**只依赖 pandas / numpy** 的 A 股日线研究引擎：数据层 → 指标 → 策略 → 回测 → 风控 → 报表，外加
全市场数据湖、多因子框架、半自动实盘通道（带事前护栏）和环境自检。全部离线可复现，142 项自测。

> 仅供研究与工程验证，不构成投资建议。回测能过不代表实盘能赚；接真钱前请读 `docs/实盘接入与合规.md`。

## 目录结构

```
量化交易系统/
├─ quant/                  引擎（pip install -e . 或直接 python -m quant）
│  ├─ data/                行情契约、缓存、CSV/合成/AKShare/联网三源、日历、分红送转
│  ├─ indicators.py        均线/RSI/MACD/布林/ATR/KDJ/OBV/重采样
│  ├─ strategy/            策略契约 + 10 个内置策略
│  ├─ backtest/            撮合、组合（T+1、整手、费用）、绩效指标
│  ├─ factors/             多因子：面板、预处理、IC/分层、合成
│  ├─ gateway/             下单通道（CSV 半自动 / 模拟盘 / QMT 适配）
│  ├─ risk.py  execution/  风控、模拟盘执行
│  ├─ lake.py              全市场数据湖（原始/精简/校核集）
│  ├─ doctor.py            环境自检
│  └─ plugins.py  cli.py   策略插件注册表、命令行
├─ scripts/                fetch_data / run_backtest / optimize / paper_trade / factors / trade
├─ tests/                  174 项离线自测（纯 assert；Parquet 套件在无 pyarrow 时自动跳过）
├─ docs/                    **先从 docs/README.md（文档总索引）进**；策略报告在 docs/策略报告/
│                           另有数据层核对表、实盘接入与合规、插件与安全审计、研究档案
├─ config.toml             策略、成本、风控、通道、护栏配置
└─ data/、results/         运行产物（已 gitignore）
```

## 快速开始

```bash
# 0) 需要一个有 pandas / numpy 的 Python 3.10+
# 1) 自测：142 项，约 8 秒
python tests/run_all.py

# 2) 环境自检（运行时/时钟/数据源/下单通道/护栏），出报告到 results/doctor.md
python -m quant doctor

# 3) 生成离线示例行情（合成数据，不联网、可复现）
python -m scripts.fetch_data --mode synthetic --start 2019-01-01 --end 2024-12-31

# 3b) 从网上抓真实日线（东财 → 腾讯 → 新浪 三源互备，自动翻页取长历史）
python -m quant fetch --mode web --symbols 600519.SH,000300.SH --start 2018-01-01

# 4) 回测（读 config.toml，出报表到 results/<策略>/）
python -m scripts.run_backtest

# 4b) 因子研究报告：IC / 分层收益 / 因子相关性 → results/factor-report/
python -m scripts.factors

# 4c) 全市场因子研究 + 回测 + Q1–Q5 分层 + 去冗余 + 频率网格（约 16 分钟）
python -m quant factor-market --out results/market-2026-10-17 --format parquet \
  --layer-horizon 5 --dedup-threshold 0.7 --grid M,2W,W --grid-topk 30

# 5) 多策略对比
python -m scripts.run_backtest --compare ma_cross,momentum_rank,multifactor

# 6) 参数网格 + 样本外检验
python -m scripts.optimize --strategy momentum_rank --grid "lookback=[60,120,250]"

# 7) 模拟盘：出建议单 → 记录成交 → 看账户
python -m scripts.paper_trade plan --out results/paper/orders.csv
python -m scripts.paper_trade apply --orders results/paper/orders.csv
python -m scripts.paper_trade status

# 8) 半自动实盘（默认只出订单文件，不真实下单）
python -m scripts.trade suggest --out results/orders/latest.csv
python -m scripts.trade check --orders results/orders/latest.csv
```

## 三条设计约定（决定了结果怎么解读）

1. **信号在 T 日收盘算，T+1 开盘成交**：`weights.shift(1)`，回测里不存在"看见当天收盘再按当天开盘买"。
2. **成本与规则按 A 股真实口径**：佣金万 2.5（最低 5 元）、印花税千 0.5（卖出）、过户费十万分之一、
   滑点按 bps、整手 100 股、T+1、涨跌停不可成交（`[rules].price_limit_overrides` 可覆盖 ST/新股）。
3. **收益率用净值算，不用价格**：现金、持仓市值、费用都进净值；绩效报告口径见下。

## 已内建的 A 股规则

- 整手买入、卖出可零股；T+1 可用数量；停牌不成交；
- 涨停不能买、跌停不能卖（按前收 ±10%，可配置）；
- 费用：佣金（默认 0.00025，最低 5）、印花税 0.0005、过户费 0.00001、滑点 bps、单票成交量参与上限 10%；
- 数据层强制：日期索引名为 `date`、列 `open/high/low/close/volume`（成交量单位 **股**），
  `amount`/`prev_close` 可选；加载时统一清洗、排序、去重、丢弃非正价行。

## 内置策略

`buy_and_hold`、`ma_cross`、`macd_trend`、`rsi_reversion`、`boll_reversion`、`donchian`、
`momentum_rank`、`factor_score`、`trend_filtered_momentum`、`multifactor`。

策略契约：`generate(bars) -> weights (DataFrame 日期×代码, 0~1)`，可另给 `rebalance_mask`、
`hold_to_rebalance`、`clip_weights`。换手控制、排名取前 K、单票上限都走策略参数。

### 已冻结的研究策略

三条经过预注册、样本内/历史/样本外检验的具体策略，口径、业绩与使用边界见 `docs/策略报告/`
（入口 `docs/策略报告/README.md`，机器可读台账 `docs/策略报告/registry.json`）：

| 策略 | 定位 | 样本内 2017–2024（14.2bp） | 样本外 |
|---|---|---|---|
| 小盘成长 v1.0（cap25） | 卫星仓、进攻 | 年化 9.48% / 夏普 0.460 / MDD −31.85% | holdout 2025-01~2026-10：21.50% / 1.118 / −16.24%（已消费） |
| 红利低波 | 底仓、防守 | 7.63% / 0.483 / −22.84% | 未重新封存 |
| 红利低波增强 | 同上，仅改执行 | 9.53% / 0.621 / −21.66% | 未重新封存 |

## 风控（`[risk]` 段，按顺序生效）

单票上限 → 行业上限（可选）→ 总仓位上限 → 单日换手上限 → 回撤熔断（触发后清仓并冷却）。
回测和实盘走同一个 `apply_risk`，所以"回测里的风控"就是"实盘的风控"。

## 多因子框架（日线低频）

```python
from quant.factors import FactorLibrary, build_panel, composite_signals, evaluate_factor

panel = build_panel(frames)            # 长表：date/symbol/open/high/low/close/volume
lib = FactorLibrary()
for name, spec in lib.registry.items():  # momentum/value/quality/volatility/size/liquidity
    panel[name] = lib.compute(name, panel)
report = evaluate_factor(panel, "mom_60", forward=5)   # IC / ICIR / 分层收益 / 换手
signal = composite_signals(panel, specs=[("mom_60", 1.0), ("low_vol_20", -0.5)])
```

`scripts/factors.py` 会把 IC 表、分层收益、因子相关性写成 Markdown（`results/factor-report/`）。
默认做**去极值 → 标准化 → 中性化（可选）**，分位分组、按日截面计算，避免未来函数。

## 数据湖：全市场日线落盘（`python -m quant lake`）

研究用的全市场日线放在 **D 盘**（C 盘只放代码，避免被行情占满）：

| 层 | 路径 | 内容 |
|---|---|---|
| 原始层 | `D:\quant-data\raw\daily\<代码>.csv` | 不复权日线（`--adjust none`），源站原始值；**永久保留、不覆盖** |
| 复权因子 | `D:\quant-data\raw\adjust_factors.csv` | 由分红送转（东财 datacenter）算出的前复权缩放因子 `k`（`build_factor_index`） |
| 分红事件 | `D:\quant-data\raw\actions.csv` | 2015 年至今约 3.8 万条除权除息事件（`lake actions`） |
| 交易日历 | `D:\quant-data\meta\calendar.csv` | 上证指数日线日期，用来补停牌 |
| 元数据 | `D:\quant-data\meta\universe.csv`、`manifest.json`、`crawl_progress.json`、`fetch_log.jsonl` | 代码表 / 状态 / 断点 / 逐只抓取日志 |
| 精简层 | `D:\quant-data\processed\daily\<代码>.csv` | ≤2024-12-31、按日历补齐停牌、价格已按前复权因子缩放（另有 `adj_factor`/`prev_close`/`tradable` 列） |
| 校核集 | `D:\quant-data\holdout\daily\<代码>.csv` + `.LOCKED` | 2025-01-01 起；**默认禁止读取**，只有 `allow_holdout=True` 才放行 |
| 精简层（列式） | `D:\quant-data\processed\parquet\year=YYYY\data.parquet` | 与精简层同内容，zstd 压缩、按年分区；**CSV 仍是权威**，列式层可随时重建 |
| 校核集（列式） | `D:\quant-data\holdout\parquet\year=YYYY\data.parquet` | 同上；读取仍要求 `--allow-holdout` |

**落地结果（2026-10-17 全量首跑）**：raw 5,575 只 / 1,076.7 万行 / 2016-01-04 → 2026-10-16 / 748 MB；
processed 5,278 只 / 855.8 万行 / ≤2024-12-31 / 745 MB（298 只是训练窗口不足 120 根的新股，属预期）；
holdout 5,575 只 / 234.4 万行 / 2025-01-02 → 2026-10-16 / 200 MB；缺 920196 一只（新股只有 1 根）。
详情见 `results/lake-summary.md`，重新生成：`python scripts/lake_report.py`；过程记录与坑见 `docs/数据湖落地记录.md`。

**列式副本（2026-10-17 追加）**：`processed` 8,557,867 行 → 178.9 MB（9 个年份分区，转换 95 s，与 CSV 逐值对拍通过）；
`holdout` 2,343,739 行 → 52.2 MB（2 个分区，41 s，同样对拍通过）。全市场读取实测：逐只 `read_csv` 29.3 s → Parquet 整层扫描 6.2 s。
`raw` 层保持 CSV 原样；列式层是可重建副本，转换与对拍可用 `lake parquet` + `lake verify` 复现。

```bash
python -m quant lake init                     # 建目录 + manifest
python -m quant lake universe                 # 全市场代码表（新浪 hs_a，约 5570 只，含北交所 92/43/83/87 段）
python -m quant lake calendar                 # 交易日历（停牌补齐用）
python -m quant lake actions --force          # 分红送转事件（约 4 万条，30 秒）
python -m quant lake crawl --source auto      # 抓原始层（沪深走腾讯、北交所走新浪，自动续跑，可随时 Ctrl+C）
python -m quant lake split --force            # 原始 → 校核集（2025-01-01 起）+ 写 .LOCKED
python -m quant lake process --force          # 原始 → 精简层（≤2024-12-31、补停牌、套后复权因子）
python -m quant lake parquet                    # 精简层 → Parquet 列式副本（需要 pyarrow，可重建）
python -m quant lake parquet --kinds holdout --allow-holdout   # 校核集也转一份（读取仍受锁）
python -m quant lake status                   # 各层行数 / 覆盖区间 / 完整性
python -m quant lake verify                   # 非正价、缺口、重复行、因子 sanity 抽查
```

三条口径（写进代码，也写进报告）：

1. **不复权入库，复权在生成精简层时算**：`processed`/`holdout` 的 OHLC 已乘前复权因子 `k`（另存 `adj_factor` 列），`k` 来自 `build_factor_index`
   （`k = (P0 − 现金红利) / (1 + 送股) / P0`，按除权日逐事件连乘，向量化 `searchsorted` 实现，单只 ~2ms）；成交量/成交额不下调，原始层 `close` 始终是真实现金价。
2. **停牌必须显式入库**：按交易日历 reindex，缺行填 `suspended=1`、`volume=0`、`amount=0`，
   OHLC 全部等于前收（不产生假收益）；只在「上市首日 ~ 最新日」之间补，不凭空造上市前行。
3. **校核集加锁**：`holdout/daily/` 默认不可读，`lake read --kind holdout` 会直接报错；
   任何调参、选因子只允许碰 `processed/`，防止样本内挑参挑到过拟合。

源可用性（2026-10 实测，会变，`doctor` 会实时告诉你）：东财 `push2his` 间歇断连；腾讯 `fqkline` 已下线（501），
但 `kline`（不复权）稳定、单页 640 根可翻页；**北交所腾讯只回 1 根，必须走新浪**（`source_for` 自动切）；
新浪单只上限 1023 根（约 4 年），适合兜底和北交所。

## 下单通道与事前护栏（半自动实盘）

`[gateway]` 选通道：`csv`（默认，只写订单文件，人工在券商 App 下）、`paper`（模拟盘）、`qmt`（需自备 xtquant 环境）。
无论哪个通道，**下单前都过一遍护栏**（`quant/guard.py`）：

- 单票金额/仓位上限、单日总下单金额上限、白名单/黑名单；
- 涨跌停、停牌、T+1 可卖数量校验；
- 与账本对账：持仓数量、可用资金不足直接拒单并写原因；
- 所有决策写 `results/ledger/*.jsonl`，可审计、可复盘。

`python -m scripts.trade check` 是**干跑**：只打印"会下什么单、为什么被拒"，不发任何请求。

## 环境自检

`python -m quant doctor` 检查：Python/依赖版本、系统时钟与最近交易日、行情源（东财/腾讯/新浪/日线接口）、
交易通道、护栏配置、数据目录可写性、插件加载，结果写 `results/doctor.md`，异常时退出码非 0（可进 CI/开机自检）。

## 加自己的策略 / 数据源 / 风控（不用改引擎）

- 策略：任何带 `generate(bars)` 的类，`plugins.py` 里注册名字即可 `--strategy your_name`；
- 数据源：实现 `load(symbol, start, end) -> DataFrame`（列契约见上），在 `WebKlineSource.SOURCES` 或 `plugins.py` 挂上；
- 风控：`risk.py` 加函数，在 `apply_risk` 链里插入一步。

## 和 DSH 里的金融插件怎么分工

| 场景 | 用什么 |
|---|---|
| 基金研究、净值/持仓穿透、经理画像 | `fund-research` 插件（确定性计算 + 版本化报告） |
| 个股贝叶斯后验、风险指标 | `bayes-stock-predict` 插件 |
| 个股 K 线图、短线盯盘 | `dsh-kline` / `stock-info` 插件 |
| 策略回测、多因子、全市场数据、下单通道 | **本项目** |

审计结论（启动项、权限、提示词注入面）见 `docs/插件与安全审计.md`；想复现就跑 `node tools/audit_plugins.mjs`。

## 接真实数据

**方式一（默认，不用装东西）**：内置三源联网抓取，东财 → 腾讯 → 新浪自动降级。

```bash
python -m quant fetch --mode web --symbols 600519.SH,000300.SH --start 2018-01-01
python -m quant fetch --mode web --symbols 600519.SH --source tencent --adjust none
python -m quant fetch --mode list --limit 300          # 全市场股票列表 → data/universe.csv
```

腾讯单次上限约 640 根，代码自动向前翻页，所以 `--start 2018-01-01` 能取到 2018 年至今（约 2100 根）。
腾讯不返回成交额，用 `成交量 ×(最高+最低+收盘)/3` 估算并在报告里标注。

**方式二（可选）**：akshare

```bash
pip install akshare
python -m scripts.fetch_data --mode akshare --symbols 600519,000001 --start 2018-01-01
python -m scripts.run_backtest --source csv --csv-dir data/csv
```

**方式三**：自己导出 CSV 放进 `data/csv/`。列名中英文都认，『手』会自动乘 100；
文件名 `600519.SH.csv` / `sh600519.csv` / `600519.csv` 都能匹配。

体检已下好的数据：`python -m scripts.fetch_data --mode check --out data/csv`

## 产物

- `results/<策略>/report.md`：绩效表、回撤明细、月度/年度收益、交易明细摘要；
- `results/<策略>/equity.csv`、`trades.csv`、`positions.csv`；
- `results/factor-report/*.md`：IC、分层、相关性；
- `results/doctor.md`、`results/ledger/*.jsonl`、`results/orders/*.csv`。

## 已知边界（别把它当"稳赚系统"）

- 只做日线低频，无日内/高频、无盘口、无融券做空；
- 复权因子来自东财分红送转，**送转数据缺失会低估除权跳空的影响**（`lake verify` 会抽查）；
- 新浪兜底只有 1023 根，北交所历史天然短；
- 未接实盘券商 API 的稳定版本，QMT 通道需要用户自备环境并自负风险；
- 回测假设"涨停不买、跌停不卖"，但无法模拟排队与滑点分布，真实成交率会低于回测。
- 全市场多因子在样本内**跑输同池等权**（−24.26% vs +24.91%，Alpha −5.27%/年）：IC 有信息但覆盖不了成本，
  且正向因子收益峰值在 Q4 而非最高分位（见 `docs/全市场因子研究记录.md` 第八、十一节）；参数是样本内选的，不具外推意义。

## 自测覆盖

142 项，全部离线、可复现（`python tests/run_all.py`，约 8 秒，11 个套件）：
数据契约/缓存、指标、撮合与费用/T+1/涨跌停、风控链、报表指标、插件注册、多因子 IC/分层、
订单护栏与账本、CSV 通道、联网三源解析（录制报文，不联网）、数据湖（分层/加锁/续跑）、复权与停牌补齐、Parquet 转换与对拍。

改代码请先跑 `python tests/run_all.py`；动数据层或复权逻辑，务必同步更新 `tests/test_actions.py`、`tests/test_lake.py` 与 `tests/test_parquet.py`。
