# 上传到 GitHub 的说明（哪些进去了、哪些没有）

> 目的：让拿到仓库的人知道**代码与结论是完整的，数据不在仓库里**，以及怎么自己把数据补回来。
> 最近核对：2026-10-22，仓库 `main` 分支。

## 1. 仓库里有什么

| 目录 / 文件 | 内容 | 规模 |
|---|---|---|
| `quant/` | 回测引擎、数据层读取、因子库、策略、风控、gateway、doctor | 63 个文件 |
| `scripts/` | 全部研究/回测/诊断脚本（每个结论都能追溯到脚本） | 73 个文件 |
| `docs/` | 设计文档、研究报告、三条策略报告、上传说明 | 24 个文件 |
| `tests/` | 离线纯 assert 测试（`python tests/run_all.py`） | 20 个文件 |
| `results/releases/` | 冻结版本的证据摘要（metrics / report / 指纹 sha256） | 37 个文件 |
| `config.toml` `pyproject.toml` `README.md` `.gitignore` | 配置与入口 | — |

合计约 **3.4 MB / 224 个文件**（`.git` 另约 2 MB）。

## 2. 没有上传的东西（按体积）

| 路径 | 体积 | 为什么没上传 |
|---|---|---|
| `D:\quant-data\`（整个数据湖） | **3.79 GB / 33,642 个文件** | 行情/估值/财务原始与处理层，单文件与仓库体积都远超 GitHub 限制；且可由公开源重建 |
| `results/`（冻结证据以外） | 194.6 MB | 回测中间产物、逐日净值、权重、成交明细；体积大且可复算 |
| `_tmp/` | 330.7 MB | 一次性探针脚本、日志、临时缓存 |
| `data/` | 6.3 MB | 本地缓存 / 手工导入的 csv |
| `__pycache__/`、`*.pyc` | — | 编译缓存 |

数据湖细分（供参考，均为**未上传**）：

| 子目录 | 体积 | 内容 |
|---|---|---|
| `processed/` | 923.9 MB | 2016–2024 日线（已补停牌、复权因子） |
| `raw/` | 748.6 MB | 不复权原始日线 + 除权除息 + 复权因子 |
| `valuation/` | 790.0 MB | 逐日估值/ST 快照 |
| `holdout/` | 442.2 MB | 2025-01-02 起的封存样本 |
| `fundamentals/` | 299.0 MB | 财务三表 + indicator/mainfin（按报告期，含 NOTICE_DATE） |
| `growth-history/` | 263.2 MB | 2007–2016 历史日线（含退市股）+ 2006–2012 财务 |
| `alt/` | 143.4 MB | 业绩预告/快报、解禁、质押、股东户数、两融 |
| `early/` | 108.0 MB | 2001–2005 日线（慢熊样本） |
| `ref/` | 82.5 MB | 股本、行业 PIT、指数成分、分红、换手宽表 |
| `events/` | 39.6 MB | 龙虎榜、回购、增减持、ST、送转 |
| `delisted/` | 34.2 MB | 退市股日线与事件 |
| `meta/` | 2.1 MB | 日历、manifest、抓取日志 |

## 3. 怎么把数据补回来

数据层不是一次性下载，而是分步冻结的（详见 `docs/数据层齐全度核对表.md`）：

```bash
python -m quant fetch                 # 日线/估值等基础层（公开源，慢）
python -m scripts.freeze_datalayer    # 冻结并生成指纹/manifest
python -m scripts.lake_report         # 完整性报告
```

历史段（2007–2016 / 2001–2005）与另类数据由下面这些脚本重采（都有限速与断点续跑）：

```bash
python -m scripts.growth_history_fetch_em --task all      # 2007–2016 日线 + 2006–2012 财务
python -m scripts.growth_history_gapfill                  # 补齐退市股缺口
python -m scripts.crisis_early_fetch --sample 2001-2005   # 2001–2005 慢熊样本
python -m scripts.growth_alt_collect                      # 预告/快报/解禁/质押
```

注意：这些接口有**限流与拉黑**风险（baostock 多进程并发登录会被拉黑；东财会间歇性断连），脚本里已经写了退避重试与单进程限速。

## 4. 怎么再推一次（本机）

推荐用一次性 PAT（Fine-grained，权限只勾 *Repository creation* + 目标仓库 *Contents: Read and write*，有效期 7 天），token 不落盘：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File _tmp\push_to_github.ps1
```

脚本会：校验 token → 创建（或复用）仓库 → 用 `GIT_ASKPASS` 推送 `main` 与全部标签 → 打印仓库地址；结束后删除临时 askpass 文件。`_tmp/` 在 `.gitignore` 里，这个脚本本身不会进入仓库。

## 5. 敏感信息检查

- 仓库内**没有** API key / token / 密码 / 券商账号；`config.toml` 的 `qmt_account` 仍是占位符，`dry_run = true`。
- `config.toml` 里的标的池是示例 20 只，不含个人信息。
- 提交历史里的作者是 `abc11 <abc11@localhost>`（本地默认身份），不是邮箱，不会泄露联系方式。
