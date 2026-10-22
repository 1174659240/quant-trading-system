# 参与与贡献

这个仓库主要是一个人的研究记录，但也欢迎提问、报错和复现验证。

## 提 issue 之前

1. 先看 [`docs/README.md`](docs/README.md)，特别是
   [`docs/数据层齐全度核对表.md`](docs/数据层齐全度核对表.md)（数据口径与 15 条偏差声明）。
2. 数据类问题先跑到 `python -m scripts.reproduce_lake --verify-only`，把输出贴上来。
   允许有差异的层只有：`raw/daily`（退市/新上市会变）、`valuation`（个别停牌股缺）、
   `growth-history`（298 只源侧缺口）。
3. 策略结论类问题，请带上复现命令与产物路径（`results/` 不进 git，但路径能说明跑的是哪一版）。

## 报 bug 时请给

- 复现命令（完整一行，含参数）；
- Python 版本、`pandas` / `numpy` 版本；
- 报错栈或不符合预期的数字（截图也行），以及你预期是什么；
- 如果和数据有关，附 `--verify-only` 的差异层。

## 改代码的规矩

- **不新增第三方依赖**。核心只依赖 pandas / numpy；采集脚本用 akshare / baostock 属于可选，
  但要在 `pyproject.toml` 的 extras 里声明。
- 改动要过 `python tests/run_all.py`（210 项离线自测，约 2 分钟，不需要数据湖）。
- **不改 `quant/` 的语义**：因子、撮合、费用、风控的任何改动都会改变历史结论，
  必须写成新版本并重跑受影响的报告。
- **数据与结果不进 git**：`D:/quant-data` 与 `results/` 都在 `.gitignore` 里，
  只有 `results/releases/` 的冻结归档例外。
- **策略改动先预注册**：任何阈值、因子、权重的调整，先把规则写进 `docs/*预注册*.md` 并提交，
  再跑结果；不允许看到结果后回头改规则。样本外（holdout）只能消费一次。
- 提交信息用一句话说明"改了什么、为什么"，不要只写 "update"。

## 想复现或扩展

- 复现数据：`python -m scripts.reproduce_lake`（见 [`docs/数据复现手册.md`](docs/数据复现手册.md)）。
- 改一条策略：先读 [`docs/策略报告/README.md`](docs/策略报告/README.md) 的「修改指南」，
  那里有每个参数在哪个文件哪一行、改完要重跑什么。
- 加一个新策略：按 `docs/策略报告/README.md` 的六步流程走（预注册 → 实现 → 登记试验次数 →
  更新台账 → 写报告 → 提交并打标签归档）。

## 不接受的改动

- 用未来的数据或修订后的财报去做"当时"的信号；
- 在已消费的 holdout 上调参数；
- 把回测结果当收益承诺，或在文档里写"稳赚""必涨"这类表述。
