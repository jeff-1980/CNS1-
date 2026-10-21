# 对象 25 登记：kavianpour/CARMA-DACA-fault-diagnosis

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-21。

## 论文与代码

- 论文：M. Kavianpour, A. Ramezani & M. T. H. Beheshti，*A class alignment method based on graph convolution neural network for bearing fault diagnosis in presence of missing data and changing working conditions*，Measurement 199 (2022) 111536，DOI 10.1016/j.measurement.2022.111536（Elsevier 订阅，审计方无全文）。
- 仓库：`kavianpour/CARMA-DACA-fault-diagnosis`，commit `602f43e9bd0cf711211251902176b4b33ccd04b8`（2026-06-05），作者本人仓库（与对象 01、19 同一作者），README 写明仓库现提供完整 PyTorch 实现。**代码晚于论文约 4 年**，按 461bc21 规则 (1) 纳入，日期差作为协变量。本地 `~/audit_ext/main_objects/25_kavianpour__CARMA-DACA-fault-diagnosis`。入口 `train.py`。

## 头条表格

- 论文全文不可得，**论文表格数值无法登记**。README “Headline results” 以任务 **D1→D2、10% 目标域标签** 的混淆矩阵为头条（1170 个样本中 1164 个正确），并称闭合工况任务（C1→C2、D1→D2）10% 标签时 CARMA-DACA 达 99.78%。S6 以 README 数值作参考对照。

## 方法组（9 个）

- 论文 Table 3 / Fig. 7 的 9 个方法在仓库中全部实现为同一骨干上的开关（`methods.py`，顺序同 `ALL_METHODS`）：WDCNN、CARMA、DTLCNN、DSACNN、HDAN、DAGCN、CARMA-M、CARMA-C、CARMA-DACA（提出的方法）。9 个方法组，组名 `<方法>|CWRU D1->D2 10%`。

## 配置（作者代码默认，`config.py`）

- 任务：表中有 2 个闭合工况任务 × 4 个标签比例 + 6 个跨工况任务。按 461bc21 规则 (2)，取 README 突出的头条：**D1（12 kHz、3 hp）→ D2（48 kHz、3 hp），目标域标签 10%**（`--label 0.10`，即 `config.py` 默认值）。
- 数据：CWRU 驱动端，12 kHz 源域与 48 kHz 目标域，10 类；长 1024 不重叠窗，每类每工况 117 个窗；目标域每 4 点只保留 1 点（75% 缺失，零填充）；目标域 10% 带标签，其余为无标签样本。
- 训练：Adam lr 1e−3，batch 128，**500 轮**，第 1 轮只用分类损失（源域 + 带标签目标域，warm-up），之后加入对抗与对齐损失（alpha 1、beta 0.05）；其余全部为 `config.py` 默认值。

## 种子方案

- 作者代码：第 r 次重复用 `seed0 + r`（`seed0 = 0`），设 numpy 与 torch，数据窗口、目标域标签划分与缺失模式也随种子变。直接用作者的 **种子 0–9**。

## 作者的选模规则

- `fit_model` 每轮在 `target_test` 上算准确率并记最好值（只打印），训练结束后再评估一次作为 `final_acc`；`run_method` 报告的是 `final_acc` 的均值。**作者口径 = 最后一轮（第 500 轮）**；没有验证集。
- **选模集**：无。**报告集**：`target_test` = 目标域无标签样本（同时参与训练中的对抗与对齐，属直推式设置，只记录）。
- 补丁：不改作者源码。驱动脚本把 `train.evaluate` 换成包装：先调用作者原函数得到准确率，再在 `isolate` 内补算同一集合的交叉熵；第 1–500 次调用记为第 1–500 轮（`select_split=None`、`report_split="test"`），第 501 次调用即作者的 `final_acc`，`finish(final_acc / 100)`；`author_pick(500)`。

## 数据来源

- 12 kHz 文件：本机只读 `~/论文CNS/data/cwru_12k_de`（编号与作者 `CWRU_12K` 一致）。48 kHz 文件：用作者 `download_cwru.py` 从 CWRU 官网下载到 `~/audit_ext/data/cwru25`。

## 泄漏编码（只记录，不修复）

- 报告集是训练中用于域对齐的同一批目标域无标签样本（直推式）。
- 每轮在报告集上评估，但作者报告最后一轮。
