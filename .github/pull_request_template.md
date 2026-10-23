## 这个 PR 做了什么

<!-- 一两句话说明改动与动机。涉及策略/因子/参数的，写明预注册文档路径。 -->

## 影响范围

- [ ] 只改文档 / 注释
- [ ] 改代码但不改数值结果
- [ ] 改数值结果（必须附新旧对比，且说明是否属于已登记的试验次数）
- [ ] 改冻结口径 / 因子集 / 权重 / 风控规则（**必须**先有预注册文档，且由所有者 review）

## 必须自查的项

- [ ] `python scripts/repo_guard.py` 通过
- [ ] `python tests/run_all.py` 通过（本机有数据湖时跑全量；无数据湖用 `--skip-lake`）
- [ ] 没有提交数据文件（`.parquet/.csv/.gz` 只允许出现在 `results/releases/`）、没有提交 `results/` 运行产物
- [ ] 没有提交任何 token、密钥、账号密码
- [ ] 改动 `results/releases/` 时，`manifest.csv` 的 sha256 已同步更新
- [ ] 新增/修改策略时，`docs/策略报告/registry.json` 与对应报告已更新
- [ ] 若新增试验（换参数、加因子、改规则），已在 PR 描述里登记次数，供 DSR 重新计算

## 与冻结结论的关系

<!-- 若改动了已冻结的策略（如 growth_cap25 / redlow / redlow_enhanced），
     说明旧结论是否因此失效、哪份报告需要标 superseded。 -->

## 复现方式

```bash
# 改了数值结果的，给出可复算命令
```
