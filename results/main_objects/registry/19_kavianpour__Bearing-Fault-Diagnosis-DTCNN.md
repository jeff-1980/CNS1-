# 对象 19 登记：kavianpour/Bearing-Fault-Diagnosis-DTCNN

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-21。

## 论文与代码

- 论文：M. Kavianpour, M. Ghorvei, A. Ramezani & M. T. H. Beheshti，*Intelligent Fault Diagnosis of Rolling Bearing Based on Deep Transfer Learning Using Time-Frequency Representation*，2021 7th International Conference on Signal Processing and Intelligent Systems (ICSPIS)，DOI 10.1109/ICSPIS54653.2021.9729385（OpenAlex：closed，无开放全文）。
- 仓库：`kavianpour/Bearing-Fault-Diagnosis-DTCNN`，commit `502effedc7fc50b2dea76c03cee9da1bc41ca7db`（2026-08-03），README 自称 “Official implementation”。**代码晚于论文约 5 年**（与对象 01 同一作者，仓库于 2026 年重写），按 461bc21 规则 (1) 纳入，日期差作为协变量。本地 `~/audit_ext/main_objects/19_kavianpour__Bearing-Fault-Diagnosis-DTCNN`。入口 `train.py`。

## 头条表格

- 论文 Table III（仓库 `assets/results_table3.png`，10 次平均）：3 个冻结的 ImageNet 预训练骨干（DenseNet201、ResNet152V2、MobileNetV2）× 6 个分类器（SoftMax 全连接头、SVM、DT、LR、KNN、RF）× 4 个负载（A–D = 0–3 hp），另报训练时间与平均值。
- 本对象登记的单元（数据集 A）：DenseNet201+SoftMax **100**、ResNet152V2+SoftMax **100**、MobileNetV2+SoftMax **99.6**。

## 方法组（3 个）

- 审计对象是逐 epoch 训练、存在“取哪一轮”问题的部分。表中只有 SoftMax 头（3 层全连接，Adam 训练 30 轮）逐轮训练；SVM、DT、LR、KNN、RF 是一次性拟合的 scikit-learn 分类器，没有 epoch，**不纳入本审计**（只记录）。
- 方法组：`DenseNet201+SoftMax|CWRU-A`、`ResNet152V2+SoftMax|CWRU-A`、`MobileNetV2+SoftMax|CWRU-A`。
- 任务：Table III 有 4 个负载，论文未突出某一负载（只突出分类器组合），按 461bc21 规则 (2) 取第一个：**数据集 A（0 hp）**。

## 配置（作者代码默认，`config.py` + `train.py`）

- 数据：CWRU 12 kHz 驱动端，10 类；每类每负载滑窗（长 1024、步长 350）取前 300 个片段，按种子打乱后 250 训练 / 50 测试；Morlet CWT（64 个尺度）→ 双线性缩放到 224×224 → 归一化到 [0,1] → 复制成 RGB。
- 特征：冻结的 ImageNet 预训练骨干（`keras.applications`，`include_top=False`）+ GAP，特征一次性算出。
- SoftMax 头：Dense 1024–512–10，Adam lr 1e-3，batch 32，**30 轮**，`SparseCategoricalCrossentropy`，`fit(..., verbose=0)`（默认打乱）；训练后在测试特征上 `predict` 取 argmax 计准确率。
- 运行方式：`python train.py --backbones <骨干> --classifiers SoftMax --loads 0`（README 示例的单组合写法；只跑 SoftMax 不影响 SoftMax 的数据与训练，因为每个组合都重新 `set_seed` 并重建数据）。
- ImageNet 预训练权重由 Keras 首次使用时从官方地址下载（DenseNet201 约 80 MB、ResNet152V2 约 230 MB、MobileNetV2 约 9 MB）。

## 种子方案

- 作者代码：第 `run` 次重复用 `seed = cfg.seed + run`（`cfg.seed = 0`，`runs = 10`），同时设 `np.random` 与 `tf.random`，数据打乱与划分也随种子变。直接用作者的 **种子 0–9**。

## 作者的选模规则

- SoftMax 头训练满 30 轮后直接在测试集上预测，没有验证集、没有早停或回存。**作者口径 = 最后一轮（第 30 轮）**。
- **选模集**：无。**报告集**：测试集（每类 50 个，共 500 个）。
- 补丁：不改作者源码。审计驱动脚本调用作者的 `evaluate_backbone`，只把 `train.train_deep_head` 换成一份逐字照抄、仅在 `head.fit` 里多加一个 Keras 回调的版本：每轮结束时用作者同一 `predict` + argmax 在测试特征上算准确率，并算测试交叉熵（只做前向，不训练、不涉及随机操作）。`select_split=None`、`report_split="test"`；`author_pick(30)`；`finish(作者 evaluate_backbone 返回的 SoftMax 准确率 / 100)`。

## 数据来源

- 本机只读 `~/论文CNS/data/cwru_12k_de`（文件名 `97.mat` … `237.mat`，与作者 `CWRU_FILES` 编号一致，作者 `download_cwru.py` 下载的就是这 40 个文件）。

## 泄漏编码（只记录，不修复）

- 滑窗步长 350 < 长度 1024，相邻片段重叠约 66%；同一负载内随机打乱后再分训练 / 测试，测试片段与训练片段存在重叠。
- 不在报告集上挑轮次。
