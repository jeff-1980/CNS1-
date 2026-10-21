# 对象 11 登记：Vitoyoung/HETPM_intelligent_fault_diagnosis

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119。本登记在任何训练之前提交。登记日期 2026-10-21。

## 论文与代码

- 论文：X. Yang & Y. Li，*Using Heterogeneous Extractor to Transfer Local-Global Knowledge for Cross-Domain Rolling Bearing Fault Diagnosis*，IEEE Transactions on Instrumentation and Measurement 74 (2025) 3546612，DOI 10.1109/TIM.2025.3582302。
- 仓库：`Vitoyoung/HETPM_intelligent_fault_diagnosis`，commit `51abc83dd3820297d66ad6db821d367b2aa65e0e`（2025-08-17）；README 自称 “official implementation”。本地 `~/audit_ext/main_objects/11_Vitoyoung__HETPM_intelligent_fault_diagnosis/HETPM_open`。入口 `HETPM_auto.py` → `utils/train_util_HETPM.py`。

## 头条表格

- 论文非开放获取，审计方经 IEEE Xplore 只能看到摘要，**头条表格的数值无法登记**；S6（复现）对本对象记为“无法对照”。仓库里没有结果表或结果文件。

## 方法组（1 个）

- 仓库只实现了提出的方法 HETPM（主分支 CNN `CNN_4_1d` + 辅助分支 CNN-ViT `VitLiConv_features_4096V1_1d`，作者默认），对比方法不在仓库中。只有 **1 个方法组**：`HETPM|JNU 0->1`。H-B2 对本对象无组对可算。
- 报告头：作者默认 `--predict_by F2_predict`，`train()` 返回的 `last_acc_F2` 即主分支 CNN（F2）的目标域准确率；目标域损失在作者代码里也按 F2 计算。本对象以 F2 为报告值。

## 配置（作者默认）

- 数据集：JNU（README 列出的两个公开数据集之一；默认 `PT1TL_B` 是作者自己的实验台数据，不公开；仓库的 HUST 加载器读取的是 `B_70Hz.csv` 这类变转速 CSV，与本机已有的 Hanoi HUST `.mat` 数据不是同一数据集）。`--data_name JNUB`。
- 任务：按 461bc21 规则 (2)，论文表格无法查看，取作者代码遍历顺序中的第一个任务 **0 → 1**（600 rpm → 800 rpm）。
- 预处理（`JNU_B.py`）：每个文件按长度 1024、步长 298 切片，取前 400 个片段（代码先随机抽 `index_sample`，但随后按 `data1[j]` 取，实际是前 400 个，只记录）；每个域内按 `train_test_split(test_size=0.2, random_state=40, stratify)` 分为 train / val；4 类（内圈、正常、外圈、滚动体）。
- 训练：100 轮（`max_epoch`），batch 32，Adam，lr 1e-3（两分支），step 衰减（第 50 轮 ×0.1），`weight_decay` 1e-5，`num_k` 4，`func` CDD，互教阈值 0.99，其余全部为 `HETPM_auto.py` 默认值。
- 评估：每轮依次跑 `source_train`、`source_val`、`target_val`，在 `target_val`（目标域 20%）上算 F1 / F2 / 平均三个头的准确率。

## 种子方案

- 作者代码里没有任何设种子的语句（`repeat_times` 次重复不设种子）。按预注册做 **10 次不设种子的重复**，记录里的 `seed` 字段为重复序号 0–9。

## 作者的选模规则

- `train()` 只在最后一轮（`epoch == max_epoch - 1`）记下 `last_acc_F1 / F2 / Avg` 并返回，`HETPM_auto.py` 对重复取均值与标准差报告。**作者口径 = 最后一轮（第 100 轮）**。没有保存模型或按验证挑轮次的逻辑。
- **选模集**：无。**报告集**：目标域 `target_val`（目标域数据的 20%）。`source_val` 每轮计算但不参与选择。
- 补丁：`select_split=None`、`report_split="test"`；每轮记作者算出的目标域 F2 准确率与损失；结束时 `author_pick(100)`、`finish(last_acc_F2)`。

## 数据来源

- 本机 `~/JNU-Bearing-Dataset`（ib/n/ob/tb × 600/800/1000，12 个 CSV，只读）。作者代码中 `data_direction` 写死为 Windows 路径，README 第 2 条要求使用者自行修改；改为本机路径（偏离记录）。

## 泄漏编码（只记录，不修复）

- 相邻切片重叠约 71%（长度 1024、步长 298），随机划分 train / val：目标域训练部分（无标签，参与对齐）与报告用的 `target_val` 存在片段重叠。
- 每轮都在报告集上评估，但作者报告最后一轮，不在报告集上挑轮次。
