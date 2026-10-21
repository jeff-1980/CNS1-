# 对象 20 登记：Littleor/MLDSO

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-21。

## 论文与代码

- 论文：D. Zhang, K. Zheng, Y. Bai, D. Yao, D. Yang & S. Wang，*Few-shot bearing fault diagnosis based on meta-learning with discriminant space optimization*，Measurement Science and Technology 33 (2022) 115024，DOI 10.1088/1361-6501/ac8303（OpenAlex：closed，无开放全文）。
- 仓库：`Littleor/MLDSO`，commit `6819ac3371fdbc42d98217a3fb555895bb996704`（2024-12-12），README 称 “condensed source codes for the paper”。本地 `~/audit_ext/main_objects/20_Littleor__MLDSO`。入口 `train.py`（README：`python train.py`，参数在 `config.py`）。

## 头条表格

- 论文非开放获取，README 与仓库中没有结果表或结果文件，**头条表格的数值无法登记**；S6（复现）对本对象记为“无法对照”。

## 方法组（1 个）

- 仓库只实现 MLDSO（4 层一维 CNN 特征提取器 + 原型网络 + 三元组损失 + 中心损失），没有对比方法。只有 **1 个方法组**：`MLDSO|CWRU 1730->1750 5-shot`。H-B2 对本对象无组对可算。

## 配置（作者代码默认，`config.py`）

- 数据：仓库自带 `data/CWRU/{1730,1750,1772,1797}/*.mat`，10 类；每个文件从头按长度 2048、步长 150 滑窗取 400 个片段（`over_sampling_size`），`add_snr = 0`（不加噪）。
- 任务（作者默认，论文表格无法查看，按 461bc21 规则 (2) 取代码默认）：**源域 1730 rpm → 目标域 1750 rpm，5-shot**。
- 训练：(1) 预训练：在源域每类前 5 个片段（`train_set_size = 5`）上做 240 个 episode（每类支持 2、查询 3），Adam lr 1e-3，三元组损失 + 中心损失；(2) 读回预训练的 `best_model.pt`，在目标域每类前 5 个片段上同样微调 240 个 episode（支持 2、查询 3）；(3) 测试：目标域每类第 6–10 个片段为支持集（5-shot），其余 390 个为查询集，按原型最近距离分类，报告准确率。
- 设备：`config_init` 在有 GPU 时自动选 GPU（`easydl.select_GPUs`）并用 `DataParallel`。

## 种子方案

- 作者代码在开始时 `seed_init(43)`（torch、numpy、random、cuda）。按预注册改为 **种子 43–52**（作者种子起连续 10 个）。

## 作者的选模规则

- 预训练阶段：每个 episode 后若训练 episode 的准确率 ≥ 历史最好且损失 ≤ 历史最低，则保存 `best_model.pt`；微调阶段读回这个文件。选择依据是训练 episode 本身（不是验证集，也不是测试集）。
- 微调阶段同样按训练 episode 保存 `eval_model.pt`，但之后**从不读回**；测试用的是微调第 240 个 episode 结束时的模型。**作者口径 = 微调最后一个 episode（第 240 个）**。
- 审计的“轮次”= 微调 episode 1–240（预训练选出的起点是作者固定的上游步骤，照作者规则）。
- **选模集**：无。**报告集**：目标域测试查询集（每类 390 个，共 3900 个）。
- 补丁：每个微调 episode 的参数更新之后，在 `audit.torch_utils.isolate` 内用作者自己的 `calculate_accuracy(network, eval_support_set, eval_query_set)`（不画 t-SNE）算测试准确率；作者代码在评估模式下不算损失（返回 −1），补丁另用同一前向的原型 softmax 交叉熵 `−log p(y|x)` 作为测试损失（偏离记录）。`select_split=None`、`report_split="test"`；`author_pick(240)`；`finish(作者打印的 Test Acc)`。

## 数据来源

- 仓库自带 `data/CWRU`（CWRU 驱动端 12 kHz，按转速分目录）。

## 泄漏编码（只记录，不修复）

- 滑窗步长 150 < 长度 2048，相邻片段重叠约 93%；目标域微调用的前 5 个片段、测试支持集（第 6–10 个）与测试查询集（第 11–400 个）在原始信号上相互重叠。
- 不在报告集上挑轮次。
