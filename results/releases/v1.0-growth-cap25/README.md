# 冻结发布：小盘成长 v1.0（cap25）

| 项目 | 内容 |
|---|---|
| 标签 | `v1.0-growth-cap25-20261022` |
| 冻结日 | 2026-10-22 |
| 报告 | `docs/策略报告/小盘成长.md` |
| 台账 | `docs/策略报告/registry.json` → `growth_cap25` |
| 数据层指纹 | `f2f8ec729eef0355`（`D:/quant-data/ref/manifest.json`，样本内 2017-2024 冻结层） |
| 样本外 | **已消费一次**（2025-01-02 ~ 2026-10-16），不可回调 |

---

## 这份归档是什么

`results/` 不进 git（`.gitignore` 默认忽略），但「冻结」的意义是**多年后还能证明当时跑出来的数字**。所以这里把**产出结论的最小集合**复制进仓库并记 sha256：报告、指标、逐年/逐月、净值、成交、权重、事件、基准。

- `artifacts/holdout/`：样本外一次性验收的全部证据（含 `consumed.json`、四次运行的 `runs-log.json`）
- `artifacts/insample/`：冻结主规格（cap25 + E 极端清仓）的样本内指标、净值、权重、行业集中度
- `artifacts/docs/`：七份结论文档（稳健性 v1/v2、消融、阶段 C、换手诊断、阶段 3 回测与风控、左侧回场对照）
- `manifest.csv`：每个文件的来源路径、字节数、sha256

重新校验：

```bash
$py -X utf8 -c "import hashlib,pathlib;[print(r.split(',')[0], hashlib.sha256((pathlib.Path('results/releases/v1.0-growth-cap25/artifacts')/r.split(',')[0]).read_bytes()).hexdigest()==r.split(',')[3]) for r in open('results/releases/v1.0-growth-cap25/manifest.csv',encoding='utf-8').read().splitlines()[1:]]"
```

## 复现全部数字

```bash
$py -u scripts/growth_holdout.py --runs 1        # 样本外（已消费，重跑只为校验）
$py -u scripts/risk_system.py --runs A,E --sample main   # 样本内主规格（cap25 已在 growth_backtest.SECTOR_CAP）
$py -u scripts/growth_robust.py --fast           # 七项稳健性检验
$py -u scripts/growth_ablate.py                  # 三项消融
$py -u scripts/growth_turnover.py --report       # 换手专项
```

## 两条纪律

1. **不得用这份归档里的任何数字回头调参**。样本外已被看过，任何基于它的修改都会让结论失效。
2. **下一次真检验只能向前**：用 2026-10 之后的新数据（`scripts/growth_paper.py --mode live` 逐日纸面跟踪，或小资金实盘），并在开跑前重新登记阈值。

## 行尾归一化（2026-10-22）

归档第一次提交时工作区在 Windows 上，文本文件是 CRLF；Git 按 .gitattributes 入库时归一化成 LF，
于是同一份文件在 Linux（CI）和 Windows（本机）算出的 sha256 不同。本次把归档内 27 个文本文件统一成 LF，
**除行尾外内容没有任何变化**，manifest.csv 的 `bytes` 与 sha256 已按归一化后的字节重算。
此后 results/releases/** 在 .gitattributes 里标为 -text，任何平台检出都是同一份字节。
