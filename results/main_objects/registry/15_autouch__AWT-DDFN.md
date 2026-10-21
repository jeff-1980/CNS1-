# 对象 15 登记：autouch/AWT-DDFN

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119。本登记在任何训练之前提交。登记日期 2026-10-21。

## 论文与代码

- 论文：*AWT-DDFN: an end-to-end adaptive wavelet transform-dual domain fusion network for bearing fault diagnosis*，Mechanical Systems and Signal Processing（2026-08 上线），DOI 10.1016/j.ymssp.2026.114827。
- 仓库：`autouch/AWT-DDFN`，commit `a77c7baffbe341dd446c33a5a572f64ba061b545`（2026-09-19）。README 写明是该论文的开源代码，并给出论文主要使用的数据集下载链接（GitHub release `N15_M07_F10.zip`，153 MB）。本地 `~/audit_ext/main_objects/15_autouch__AWT-DDFN`，入口 `train.py`。
- 资料包里此对象为 pending（“框架待核实”）：核实结果为 PyTorch，有完整训练脚本，数据公开可下载，满足纳入条件。

## 头条表格

- 论文非开放获取（OpenAlex：closed），仓库没有结果表，**头条表格的数值无法登记**；S6 对本对象记为“无法对照”。

## 方法组（1 个）

仓库只有提出的方法 `AWT_DDFN`（`models/mymodel.py`）。只有 **1 个方法组**：`AWT-DDFN|PU N15_M07_F10 white -10dB`。H-B2 对本对象无组对可算。

## 配置（作者默认，`train.py`）

- 数据：作者 release 的 `N15_M07_F10`（Paderborn 数据集 N15_M07_F10 工况，作者整理成每类一个 CSV、每行一个样本）。`load_classification_data`：每类随机取 1000 行（`seed=1` 固定），整体加噪后按 `train_test_split(stratify, random_state=42)` 分为 60% / 20% / 20%。
- 噪声：`train.py` 主程序遍历 `snr_list = [-10, -7, -4, -1, 2]`、`noise_type = 'white'`，每个 SNR 跑 1 次。按 461bc21 规则 (2)，论文表格无法查看，取作者遍历顺序中的第一个：**白噪声 −10 dB**。
- 训练：batch 128，Adam lr 3e-4，CosineAnnealingLR，**300 轮**。
- 运行方式：按作者目录结构，在工作目录下放 `AWTL-DDFN1/N15_M07_F10`（数据软链接），结果写入工作目录下的 `AWTL-DDFN1/result/...`。

## 种子方案

- 作者代码对训练不设种子（数据抽样与加噪由 `load_classification_data` 内的 `seed=1` 固定，划分 `random_state=42` 固定）。按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。各次重复只在模型初始化与 batch 顺序上不同。

## 作者的选模规则

- 每轮在验证集上算准确率，`valid_accuracy >= max(valid_accuracies)` 时保存 `model_best_weights.pth`（真正的 checkpoint，并列取后者）；训练结束后加载最好 checkpoint，在测试集上报告 Acc / Pre / Rec / F1（宏平均），并另外报告最后一轮模型的同一组指标。
- `author` 协议 = 验证集准确率**最后一个**达到最大值的轮（按作者“best_model test result”）。**选模集**：验证集（20%）；**报告集**：测试集（20%）。
- 作者代码每轮本来就在测试集上评估并打印测试准确率与损失：补丁直接记录作者算出的验证集与测试集准确率、损失，不需要额外评估；`author_pick` 按作者同一规则标出；结束时 `finish(best_model 测试准确率)`。

## 数据来源

- `https://github.com/autouch/AWT-DDFN/releases/download/Dataset/N15_M07_F10.zip`（作者 README 给出），下载到 `~/audit_ext/data/awt/`。C 盘真实剩余 ≥ 10 GB 时下载。

## 泄漏编码（只记录，不修复）

- 样本是 CSV 中的行，是否来自同一信号的重叠切片取决于作者的整理过程（release 中不可见）；随机划分训练 / 验证 / 测试。
- 每轮在测试集上评估（作者打印），但作者报告的是验证集挑出的模型，另报最后一轮。
