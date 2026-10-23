# 策略报告总目录

这个目录放**每条策略一份的「总体情况介绍」**——不是研究底稿，也不是某次实验的结论，而是「这条策略是什么、怎么跑、能期待多少、哪里会坏、怎么复现、要改的时候动哪里」。

整个仓库的文档地图见 [../README.md](../README.md)。

## 现有策略

| 策略 | 类型 | 样本内 2017–2024（14.2bp） | 样本外 | 状态 | 报告 |
|---|---|---|---|---|---|
| **小盘成长**（v1.0 / cap25） | 进攻、小容量、卫星仓 | 年化 9.48% · 夏普 0.460 · MDD −31.85% | holdout 2025-01~2026-10：年化 21.50% · 夏普 1.118 · MDD −16.24%（已消费一次） | **已冻结（v1.0）** | [小盘成长.md](小盘成长.md) |
| **红利低波** | 防守、底仓 | 年化 7.63% · 夏普 0.483 · MDD −22.84% | 未封存（不使用 2025+ 数据） | 冻结口径、可回滚 | [红利低波.md](红利低波.md) |
| **红利低波增强** | 同上，仅改执行（只在调仓日成交） | 年化 9.53% · 夏普 0.621 · MDD −21.66% | 不适用（半成品） | **半成品**，可回滚 | [红利低波增强.md](红利低波增强.md) |

> 三条策略的口径、数据层与成本模型同源（`processed` 日线 + PIT 财务/事件层 + 14.2bp 往返）。
> 「样本内」互相可比；「样本外」状态各不相同，**不要**把某条的样本外结果当另一条的预期。

## 阅读顺序

1. 先看每份报告的 **§0 一句话** 和 **使用边界**，判断这条策略是否适合当前用途；
2. 再看业绩表（样本内 + 样本外 + 基准）；
3. 最后看复现命令——所有数字都能用仓库内脚本重算。

---

## 修改指南（要动之前先看这张表）

### 小盘成长（`growth_cap25`）

| 想改什么 | 改哪里（文件:行） | 改完必须重跑 | 失效范围 |
|---|---|---|---|
| 因子集合 / 分块权重 | `quant/factors/growth.py:102` `BLOCK_WEIGHTS`、`:104` `BLOCKS` | `growth_factor_ic` → `risk_system --runs A,E --sample main` → `growth_robust --fast` | 全部样本内数字；样本外结论**不能复用** |
| 选股口径（跳过哪 5 个因子、并入解禁） | `scripts/growth_backtest.py:59` `DROPPED`、`:68` `UNLOCK`、`:78` `SPECS` | `growth_backtest` → `risk_system` | 同上 |
| 持仓数 / 总仓位 / 单票上限 / 行业上限 | `scripts/growth_backtest.py:85-88` | `risk_system --runs A,E --sample main` | 业绩、行业集中度、换手 |
| 缓冲带 / 分档滑点 | `scripts/growth_risk.py:71` `BUFFER_RANK`、`:76` `SLIP_TIERS`、`:94` `BASE_BPS` | `risk_system` 或 `growth_holdout` | 换手、费用、加权滑点 |
| 极端清仓层（E）阈值 / 开关 | `scripts/risk_system.py:35-44`（`EXT_*`/`FRACTIONS`/`STEP_DAYS`）、`:307` `DEFAULT_LAYER` | 先 `risk_system_diag` 看分量，再 `risk_system --sample history` | 回撤、历史样本、泛化结论 |
| 左侧分批回场（可选层） | `scripts/growth_leftside.py:42` `RISK_MODES`、`:70-74` 阈值 | `growth_leftside` → `growth_leftside_report` | 两样本对照结论 |
| 换手诊断 / 变体 | `scripts/growth_turnover.py` | `--diag` / `--runs` / `--report` | 换手专项结论 |
| 数据目录 / 财报告期上限 | `quant/factors/growth.py:49-50` | 全部 | 一切 |

**冻结口径端到端**：`scripts/growth_holdout.py:49-62`（窗口、`TOP_K`、`BUFFER_RANK`、`BASE_BPS`）。改这里等于改「验收标准」，必须先写预注册。

### 红利低波（`redlow`）

| 想改什么 | 改哪里 | 改完必须重跑 | 失效范围 |
|---|---|---|---|
| 因子组合 | `scripts/final_spec.py:40` `FACTORS` | `strategy_registry` | 全部业绩 |
| 持仓数 / 仓位 / 单票上限 | `scripts/final_spec.py:72` 附近 | `strategy_registry` | 业绩、容量 |
| 成本档位 | `scripts/redlow_study.py:138` `cost_presets` | `strategy_registry` | 费用与三档对比 |
| 基准 | `scripts/redlow_study.py:174` `pool_benchmark` | `strategy_registry` | 超额、信息比率 |
| 剔除规则（ST/次新/一字板/龙虎榜） | `scripts/redlow_study.py:81` `build_block` | `strategy_registry` | 池规模、业绩 |

### 红利低波增强（`redlow_enhanced`）

| 想改什么 | 改哪里 | 改完必须重跑 | 失效范围 |
|---|---|---|---|
| 选股（应与红利低波保持一致） | `scripts/improvement_shadow.py:79-80` `FACTORS_*` | `strategy_registry` | 两条策略的可比性 |
| 执行策略（逐日 / 月频 / 带宽） | `scripts/improvement_shadow.py`（`ShadowEngine(policy=...)`） | `strategy_registry` | 两策略之差 |
| 换手预算（buffer 2x / 单月上限 30%） | `scripts/strategy_optimize.py:798-800` | **尚未进入默认规格**，要用须单独预注册 | — |

> 表里没有的改动（新增因子、改阈值、加风控）一律先写预注册文档，再动代码。

---

## 改一条策略的标准流程

1. **先写预注册**：`docs/<主题>-预注册.md`，写死阈值、判据、试验次数；没有这一步不许跑。
2. **改代码**（或复制脚本到新文件），新结果落**新目录**，绝不覆盖旧目录。
3. **重跑并登记**：把试验次数 +N 记进台账（`results/growth/robustness/trial-ledger.json` 之类），供 DSR 使用。
4. **更新台账** `registry.json`：`version` 升级、`performance` 换新、`artifacts` 加新目录、`git_tag` 填新标签。
5. **写报告**：新版本单独一节或单独文件；旧报告标 `superseded by vX.Y`，不删（可供回溯）。
6. **提交 + 打标签 + 归档**：`git tag -a vX.Y-<主题>-<日期>`，把关键产物复制到 `results/releases/<tag>/` 并更新 `manifest.csv`。

**禁止**：在已消费的 holdout 上试参数；只改报告不改代码/台账；把「样本外好看」当成「可以加仓」的依据。

---

## 「保存一条策略」的含义（本仓库约定）

一条策略被「保存/冻结」需要同时具备四件东西：

1. **报告**（本目录下的 md，口径、业绩、边界、复现齐备）；
2. **台账**（`registry.json` 里的一条记录，机器可读的冻结参数与产物路径）；
3. **git 提交 + 标签 + 产物归档**（例如 `v1.0-growth-cap25-20261022`；`results/` 默认不入库，冻结版本的证据复制到 `results/releases/<tag>/` 并记 sha256，随标签一起提交）；
4. **冻结后的纪律**：改因子/参数/规则即视为新版本，旧报告的样本外结论作废，需重新封存再验。

## 报告固定结构（新增策略照此写）

`元信息表 → 0 一句话 → 1 定位与预期 → 2 策略定义（池/因子/合成/组合/执行/风控/成本） → 3 数据与 PIT → 4 业绩（样本内/历史/holdout/拼接） → 5 稳健性 → 6 已知缺陷与使用边界 → 7 复现与产物 → 8 变更历史 → 9 维护指南（改什么影响什么）`

## 关联文档

- 全局地图：[../README.md](../README.md)
- 小盘成长：`docs/小盘成长策略设计与数据核对.md`（设计 + 全部分阶段结论）、`docs/小盘成长策略-进度日志.md`、`docs/极端风险清仓预注册.md`、`docs/小盘成长策略-左侧回场预注册.md`、`docs/QVIX回补与多危机泛化预注册.md`、`docs/稳健改进三项-预注册.md`
- 红利低波 / 增强：`docs/红利低波与容量因子研究报告.md`（因子研究底稿）、`docs/事件因子研究报告.md`、`docs/样本外验证报告.md`、`results/strategy-registry/STRATEGIES.md`（两策略逐项对照）
- 通用：`docs/风险清仓与前沿因子.md`、`docs/全市场因子研究记录.md`、`docs/数据层齐全度核对表.md`
