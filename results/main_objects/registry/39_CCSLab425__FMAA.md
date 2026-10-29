# 对象 39 登记：CCSLab425/FMAA

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-22。

## 论文与代码

- 论文：*Few-shot Metric Adversarial Adaptation for Cross-machine Fault Diagnosis*，2024 IEEE 63rd Conference on Decision and Control (CDC)，DOI 10.1109/CDC56724.2024.10886709（Crossref 核实，**待核实项“同行评审发表 DOI”成立**；全文不可得）。
- 仓库：`CCSLab425/FMAA`，commit `79f3939f87cdf6a44e6fa859afb1ab71cd213cac`（2025-04-07），README 称 “pytorch implement for the CDC 2024 paper”。本地 `~/audit_ext/main_objects/39_CCSLab425__FMAA`。入口 `main.py`。

## 头条表格

- 论文全文不可得，README 无结果表，**头条表格的数值无法登记**；S6 对本对象记为“无法对照”。

## 方法组（1 个）

- `main.py` 中 `tasks = [0]`、`experiments = ['FMAA+LSMMD+Attention']`，只运行提出的方法。只有 **1 个方法组**：`FMAA|CWRU 1hp->SBDS 1K`。

## 配置（作者代码默认）

- 数据：仓库自带作者预处理的 `datasets/CWRU_1hp_4.mat`（源域）与 `datasets/SBDS_1K_4_06.mat`（目标域，苏州大学滚珠丝杠数据），4 类；`main.py` 的迁移任务列表只有这一对（跨机器）。
- 训练：三步——源域预训练特征提取器 30 轮，预训练组判别器 50 轮，对抗训练 100 轮（warm steps 40），每类 1 个目标域少样本，batch 64，lr 1e−3，其余超参数全部为 `main.py` 默认。

## 种子方案

- 作者在每次运行开头 `torch.manual_seed(1)`（及 CUDA 种子）。按预注册改为 **种子 0–9**。（组采样 `sample_groups` 以轮次为种子，作者设计，不改。）

## 作者的选模规则

- 对抗训练每轮在目标域测试集上评估（批平均准确率，四舍五入到 4 位），`test_accuracy >= best_acc` 时更新并保存模型；结束时报告 `best_acc`。**作者口径 = 在报告集上取最好（author ≈ best_rep，平局取较晚轮）**。
- **选模集**：无。**报告集**：目标域测试集。
- 审计的“轮次”= 对抗训练轮 1–100（前两步预训练为作者固定的上游步骤）。
- 补丁：`model_evaluation` 中在作者算出 `test_accuracy`、`test_loss` 后调用钩子记录（不另做评估）；按作者 `>=` 规则标 `author_pick`；`main.py` 的 `torch.manual_seed(1)` 改为审计种子，结束时把 `best_acc` 交给 `finish`。

## 数据来源

- 仓库自带 `datasets/*.mat`。

## 泄漏编码（只记录，不修复）

- 每轮在报告集上评估并取最好；目标域少样本取自同一目标域数据。
