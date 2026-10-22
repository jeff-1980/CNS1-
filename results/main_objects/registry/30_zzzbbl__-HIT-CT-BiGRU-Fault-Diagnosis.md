# 对象 30 登记：zzzbbl/-HIT-CT-BiGRU-Fault-Diagnosis

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-22。

## 论文与代码

- 论文：Y. Wang & B. Zhang，*A fault diagnosis method for aero-engine inter-shaft bearings based on 1DCNN-Transformer-BiGRU*，Scientific Reports (2026)，DOI 10.1038/s41598-026-61710-4（开放获取）。
- 仓库：`zzzbbl/-HIT-CT-BiGRU-Fault-Diagnosis`，commit `767edb0d9298d5417cfed44b36793d8d4ec9c1f2`（2026-09-11），README 称 “Official implementation”，`docs/paper_code_mapping.md` 给出论文实验与入口的对应。本地 `~/audit_ext/main_objects/30_zzzbbl__-HIT-CT-BiGRU-Fault-Diagnosis`。

## 头条表格与方法组（4 个）

- 论文 Table 3（与已有方法比较）：CT-BiGRU 准确率 99.72%；对比方法 TST、CNN、LSTM、RWI-LSTM、MD-CNN-BGRU、ISONet，论文写明部分结果取自已发表文献而非本文复现；仓库没有为这些对比方法提供运行入口（`paper_code_mapping.md` 只映射主实验与模块消融）。
- 论文 Table 6（模块消融，作者以 `src/ablation.py` + `configs/hit_ablation.yaml` 复现）：CT-BiGRU 99.72%，对比 1DCNN-Transformer、1DCNN-BiGRU（最好的消融模型，CT-BiGRU 高 1.27 个百分点）、Transformer-BiGRU。
- 由于 Table 3 的对比组在仓库中没有作者给出的运行方式，本对象取作者脚本可直接复现、且包含提出方法的 **Table 6 的 4 个组**（预注册上限 4 组）：`ct_bigru`、`cnn_transformer`、`cnn_bigru`、`transformer_bigru`，组名 `<模型>|HIT`。S6 以 Table 6 / Table 3 中 CT-BiGRU 的 99.72% 及 Table 6 各消融行对照（数值待训练后从论文页补录）。

## 配置（作者代码默认）

- 数据：HIT 原始数组 `data1.npy`–`data5.npy`（论文数据可用性声明指向 HouLeiHIT/HIT-dataset，完整数组在其 Google Drive；下载到 `~/audit_ext/data/hit23`，与对象 23 共用）；`src/prepare_hit_dataset.py` 默认参数（通道 2–5，共 4 个；按样本 `train_test_split(test_size=0.3)` 后再分 20% 验证，`random_state=42`；RobustScaler 用训练集拟合），只运行一次。
- 训练：`src/ablation.py` 对每个模型以 `configs/hit_ablation.yaml` 调 `src/train.py`：Adam lr 1e−3、weight decay 0，batch 16，**50 轮**（`min_epochs = 50`、早停耐心 20，实际不会早停），训练集数据增强，类别加权交叉熵。
- **代码与论文描述不一致处（只记录，按 461bc21 规则 (1) 不视为 E8，因 README 未声明差异）**：论文写训练 / 测试按 28 个工况组 7:3 划分，代码按样本随机划分；论文输入 20480 × 6，代码默认取 4 个通道；论文 weight decay 1e−5，配置为 0。

## 种子方案

- 作者 `set_seed(cfg['seed'])`（默认 42，设 random / numpy / torch）。按预注册用 **种子 0–9**（改配置中的 `seed`；数据划分在预处理时固定为 42，不随种子变）。

## 作者的选模规则

- `train_model` 每轮在验证集上评估，`val_accuracy > best_acc`（严格大于，平局取较早轮）时保存 `best_model.pth`；训练结束后加载该文件，在测试集上报告 accuracy / precision / recall / F1（`test_metrics.json`）。**作者口径 = 验证集准确率第一个达到最大值的轮**。
- **选模集**：验证集。**报告集**：测试集。
- 补丁：`training.py` 的 `train_model` 每轮验证后调用钩子：记录作者算出的验证准确率与损失（`select_split="val"`），并在 `isolate` 内用作者同一 `run_epoch`（不传优化器）在测试集上算准确率与损失（`report_split="test"`）；按作者规则标 `author_pick`；`train.py` 结束时把作者 `evaluate_model` 的测试准确率交给 `finish`。驱动按作者 `ablation.py` 的方式为每个模型生成配置，在进程内调用 `train.py` 的 `main`。

## 数据来源

- `~/audit_ext/data/hit23/data{1..5}.npy`。

## 泄漏编码（只记录，不修复）

- 代码按样本随机划分（论文声明按工况划分），同一工况的样本同时出现在训练与测试中。
- 不在报告集上挑轮次（按验证集挑）。
