# 对象 32 登记：Xiaohan-Chen/CoWS

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-21。

## 论文与代码

- 论文：X. Chen, R. Yang, Y. Xue & Z. Wang，*CoWS: Self-Supervised Representation Pre-Training for Cross-Machine Fault Diagnosis*，IEEE Transactions on Emerging Topics in Computational Intelligence (2026)，DOI 10.1109/TETCI.2026.3654383。
- 仓库：`Xiaohan-Chen/CoWS`，commit `63faaf52de990f481c7844cc13302e42b0beb167`（2026-01-29），README 称 “The officieal implementation of the paper”。本地 `~/audit_ext/main_objects/32_Xiaohan-Chen__CoWS`。

## 头条表格

- README 结果表（仓库 `results.png`，论文主表）：在 CWRU 上自监督预训练后，跨机器迁移到 XJTU-SY（WC1–3）、HIT（WC1–4）、HUST（WC1–4）的准确率（均值 ± 标准差），对比 Random、BYOL、SimSiam、SSPCL、SSF、AFFE、TFC、FSSSL、TFAI、SMCLN 与 CoWS。
- 本对象登记的单元：**XJTU-SY WC1，CoWS 99.52 ± 1.18**。

## 方法组（1 个）

- 仓库只有 CoWS（预训练 `CoWS.py`、迁移 `Transfer.py`），表中其余预训练方法的代码不在仓库。只有 **1 个方法组**：`CoWS|CWRU->XJTU-SY WC1`。H-B2 对本对象无组对可算。

## 配置（作者代码默认）

- 任务：表中 3 个目标数据集共 11 个工况，按 461bc21 规则 (2) 取表中第一个：**XJTU-SY WC1**，即 `Transfer.py --tgt_data XJTUBearing --tgt_load 0 --labels 0,1,2,3`（XJTU 该工况 4 类，`labels` 只决定类别数；其余参数全部为 `Configs/transfer.py` 默认）。
- 预训练：使用作者随仓库发布的 CoWS 预训练权重 `Checkpoint/CoWS_Bearing_load0.pth`（CWRU load 0 上自监督预训练的 1D ResNet50），`Transfer.py` 就是从这个文件加载；**不重跑预训练**（预训练是迁移阶段之前、作者已固定的上游步骤，与头条表的口径一致：表中比较的是不同预训练方法在同一迁移流程下的结果）。
- 迁移：仓库自带 `Data/XJTUBearing/load0.pkl`（每类 240 条，长 1024）；每类前 10 条训练、接下来 100 条验证、最后 100 条测试（固定切片）；1D ResNet50 加载预训练权重，重置全连接层；SGD momentum 0.9、weight decay 5e−4，分类层 lr 1e−2、骨干 lr 1e−3，余弦退火，batch 256，**40 轮**。

## 种子方案

- `Transfer.py` 不设任何种子（全连接层重置、数据打乱都不设种子）。按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。

## 作者的选模规则

- 每轮在验证集上算准确率，`val_acc >= best_acc` 时立即在测试集上评估并记下 `test_acc`；结束时报告最后一次记下的 `test_acc`。**作者口径 = 验证集准确率最后一个达到最大值的轮**（平局取较晚轮）。
- **选模集**：验证集（每类 100 条）。**报告集**：测试集（每类 100 条）。
- 补丁：`Transfer.py` 训练循环中加一个钩子，每轮验证后记录作者算出的验证准确率与损失（`select_split = "val"`），并在 `isolate` 内用作者同一 `evaluate` 函数算测试准确率与损失（`report_split = "test"`）；作者更新 `test_acc` 时 `author_pick`；结束时 `finish(作者最终 test_acc / 100)`。

## 数据来源

- 仓库自带 `Data/XJTUBearing/load0.pkl`。

## 泄漏编码（只记录，不修复）

- 训练、验证、测试为同一工况数据的固定前后切片；切片在原始信号上是否重叠无法从 pkl 核查。
- 不在报告集上挑轮次（按验证集挑）。
