# 对象 34 登记：JiahongChen/FRAN

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-22。

## 论文与代码

- 论文：J. Chen, J. Wang, J. Zhu, T. H. Lee & C. De Silva，*Unsupervised Cross-domain Fault Diagnosis Using Feature Representation Alignment Networks for Rotating Machinery*，IEEE/ASME Transactions on Mechatronics (2020)，DOI 10.1109/TMECH.2020.3046277（OpenAlex：closed）。
- 仓库：`JiahongChen/FRAN`，commit `58dd5c4a6162d0d76cd3a2b562319ca761397aac`（2020-12-25），README 称 “the Python+PyTorch code to reproduce the results of Fault Severity Diagnosis in paper”。**待核实项“框架”**：README 的环境一节写 TensorFlow 1.6.0，但代码全部是 PyTorch（`main.py` 导入 torch，模型为 `nn.Module`），核实为 PyTorch，成立。入口 `batchrun.sh` → `main.py`。

## 头条表格

- 论文全文不可得，README 无结果表，**头条表格的数值无法登记**；S6 对本对象记为“无法对照”。

## 方法组（1 个）

- 仓库只有 FRAN。只有 **1 个方法组**：`FRAN|CWRU DE->FE`。H-B2 对本对象无组对可算。

## 配置（作者代码默认）

- 数据：仓库自带的作者预处理 CWRU 数据（`CWRU_dataset/*.7z`，解压为 `CWRU_<域>.npy`，每个域含 Normal / OR / IR 三类）。
- 任务：作者 `batchrun.sh` 遍历 8 × 8 个源→目标任务；论文表格不可见，按 461bc21 规则 (2) 取代码默认 **DE → FE**（`main.py` 参数默认值）。
- 训练：100 轮，batch 64，Adam lr 1e−4（weight decay 5e−4），lambda 1.0，entropy 权重 0.1，`thres_rec` 1e−4，其余全部为 `main.py` 默认。

## 种子方案

- 作者代码不设任何种子（`batchrun.sh` 本身就做 10 次不设种子的重复）。按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。

## 作者的选模规则

- 每轮训练后在目标域全部样本上评估准确率（目标域同时用于无监督适配，直推式），`correct >= max_correct` 时更新；结束时报告并写入文件的是 **`max_correct`，即各轮目标域准确率的最大值**。**作者口径 = 在报告集上取最好（author ≈ best_rep，平局取较晚轮）**。
- **选模集**：无。**报告集**：目标域全部样本。
- 补丁：`main.py` 中每轮评估后加钩子，记录作者算出的目标域准确率（`select_split=None`、`report_split="test"`），并在 `isolate` 内补算同一目标域的交叉熵；按作者 `>=` 规则标 `author_pick`；结束时 `finish(max_correct)`。另：`source_loader_iter.next()` 在当前 PyTorch 已不存在，改为 `next(source_loader_iter)`（依赖适配，偏离记录）。

## 数据来源

- 仓库自带 `CWRU_dataset/CWRU_DE.7z`、`CWRU_FE.7z`（用 py7zr 解压到工作目录）。

## 泄漏编码（只记录，不修复）

- 报告集是训练中用于无监督适配的同一批目标域样本（直推式）。
- 每轮在报告集上评估，作者报告报告集上的最大值。
