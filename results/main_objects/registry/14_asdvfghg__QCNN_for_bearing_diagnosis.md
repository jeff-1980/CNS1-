# 对象 14 登记：asdvfghg/QCNN_for_bearing_diagnosis

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119。本登记在任何训练之前提交。登记日期 2026-10-21。

## 论文与代码

- 论文：Liao, Dong, Sun, Sun, Zhang, Fan，*Attention-Embedded Quadratic Network (Qttention) for Effective and Interpretable Bearing Fault Diagnosis*，IEEE TIM 72 (2023)，DOI 10.1109/TIM.2023.3259031。
- 仓库：`asdvfghg/QCNN_for_bearing_diagnosis`，commit `e325210b1ca19bdbfc731959685f6a2a2c9d7ddc`（2024-08-27），README 自称 “official repository”。本地 `~/audit_ext/main_objects/14_asdvfghg__QCNN_for_bearing_diagnosis`。入口 `train.py`。

## 头条表格

README “Main Results” 图（论文主表，“All results are run 5 times to calculate the average”），F1 (%)。按 461bc21 规则 (2) 取表中第一个工况与第一个 SNR，即 **CWRU-0HP、−6 dB**（也是仓库自带数据与 `train.py` 默认 SNR）：

| 组 | 论文 F1 (%)，CWRU-0HP，−6 dB |
|---|---|
| QCNN | 98.95 |
| WDCNN | 88.83 |

表中其余基线（RNN-WDCNN、MA1DCNN、DCA-BiGRU、AResNet）用的是各自官方代码，不在本仓库中。

## 方法组（2 个）

`train.py` 通过 `chosen_model` 训练 QCNN 或 WDCNN：`QCNN|CWRU-0HP -6dB`、`WDCNN|CWRU-0HP -6dB`。

## 配置

- 超参数：采用 **README 超参数表中 CWRU-0HP、−6 dB 这一格的取值**：QCNN lr 0.05、α 0.005；WDCNN lr 0.3。`train.py` 末尾的默认值（lr 0.5、α 0.03）与这一格不符，是作者调参用的通用初值；头条结果对应的是表中的取值。写入偏离记录。
- 其余取 `train.py` 默认与 README 设置表：batch 64，50 轮（与设置表中 CosineAnnealingLR `T_max=50` 一致），SGD momentum 0.9（QCNN 另有 weight decay 1e-4 与 ReLinear 分组学习率）。
- 噪声：`add_noise = True`、`snr = -6`。`train.py` 默认 `add_noise = False`，此时要读 `data/0HP_Train`、`data/0HP_Test`，这两个目录仓库里没有，且作者的 `data_split.py` 只生成加噪的划分（`noise=False` 时会因变量未定义而报错），按原样无法运行；仓库自带的正是 −6 dB 的划分，论文表也是加噪设置。写入偏离记录。
- 数据：`prepro` 每类从训练文件随机取 750 个长 2048 的窗（`enc_step = 28` 重叠增强，`normal = False`），`StratifiedShuffleSplit(test_size=1/3)` 分出验证集；测试：每类从独立测试文件取 250 个窗，10 类。
- 运行方式：`WANDB_MODE=offline`。

## 种子方案

作者有种子机制（`config.seed`，默认 42）。用种子 **0–9**。测试窗的随机抽取也由同一种子决定（作者设计）。

## 作者的选模规则

- 每轮在验证集上算准确率，`acc >= max_acc` 时把 `state_dict` 存为 `Models/best_checkpoint.pth`（真正的拷贝，与对象 13 不同）；训练结束后加载该 checkpoint，在测试集上算宏平均 F1 并报告。
- `author` 协议 = 验证集准确率**最后一个**达到最大值的轮（`>=`，并列取后者）。**选模集**：验证集；**报告集**：测试集（独立文件）。
- 作者报告值是 F1，不是准确率。补丁在每轮记作者算出的验证集准确率与损失（`select_split = "val"`），并在 `isolate` 内补算测试集准确率、损失与宏平均 F1（准确率与损失进 `"test"` 记录，F1 另存）；按作者同一规则标出 `author_pick`；作者 `inference` 返回的 F1 与记录中作者轮次的 F1 另行核对（`author_check_f1`）。
- 作者的验证损失是各批损失之和（未除以批数）；每轮批数固定，按原值记录，不影响 `minloss` 的取轮。

## 数据来源

- 仓库自带 `data/0HP_TrainNoised_-6`、`data/0HP_TestNoised_-6`（作者 `data_split.py` 按每个信号前 2/3 训练、后 1/3 测试切分后加噪生成）。0HP 正常数据为 97.mat，不涉及 99.mat 问题。

## 泄漏编码（只记录，不修复）

- 训练窗重叠增强后随机划分训练 / 验证，验证集与训练集样本高度重叠；测试来自信号后 1/3，与训练不重叠。
- 每个 batch 都重新创建优化器与调度器（动量每步清零，余弦调度不生效），只记录。
