# 对象 13 登记：asdvfghg/BearingPGA-Net

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119。本登记在任何训练之前提交。登记日期 2026-10-21。

## 论文与代码

- 论文：Liao, Wei, Xie, Zeng, Sun, Zhang, Zhang, Fan，*BearingPGA-Net: A Lightweight and Deployable Bearing Fault Diagnosis Network via Decoupled Knowledge Distillation and FPGA Acceleration*，IEEE TIM 73 (2024)，DOI 10.1109/TIM.2023.3346517。IEEE 版非开放获取；审计方读的是作者 arXiv 版 2307.16363（文本经 pypdf 提取）。
- 仓库：`asdvfghg/BearingPGA-Net`，commit `96392f3527498a38c8d2e00911deb810e168fb79`（2024-03-27），README 自称 “official repository”。本地 `~/audit_ext/main_objects/13_asdvfghg__BearingPGA-Net`。

## 头条表格

arXiv 版消融表（“average F1 score (%) of teacher and student models and student models without decoupled knowledge distillation”），**所有结果为 10 次运行的平均**。仓库默认配置 CWRU-0HP、SNR −6 dB 对应的行：

| 组 | 论文 F1 (%)，CWRU-0HP，−6 dB |
|---|---|
| Teacher | 98.57 |
| Student（BearingPGA-Net，DKD） | 97.20 |
| Student w/o KD | 93.20 |

按 461bc21 规则 (2)：取仓库默认、也是论文表中第一个工况（0HP）与第一个 SNR（−6 dB）。论文正文另有与 WDCNN、LEFE-Net、CLFormer、KDSCNN 的对比，这些基线的代码不在仓库中。

## 方法组（2 个）

| 组 | 入口 | 说明 |
|---|---|---|
| `BearingPGA-Net (DKD)|CWRU-0HP -6dB` | `train_dkd.py` | 先训练 Teacher（`TCNN`，75 轮），再以 DKD 训练 Student（`SCNN`，75 轮），在测试集上报告 Student |
| `Student w/o KD|CWRU-0HP -6dB` | `train_single.py` | 只训练 Student（75 轮），无蒸馏 |

Teacher 不单列为组：作者代码不在测试集上评估 Teacher，没有可核对的作者报告值。

## 配置（作者默认）

- `train_dkd.py`：batch 64，75 轮，Teacher lr 0.1、Student lr 0.1，SGD momentum 0.9，α = 0.2，β = 5，γ = 1，warmup 5，`snr = -6`，`chosen_data = '0HP'`。
- `train_single.py`：batch 64，75 轮，lr 0.5（脚本默认），**`snr` 设为 −6**（脚本默认是 8，但仓库只附了 −6 dB 的数据，论文表中 Student w/o KD 也是 −6 dB 这一行），`chosen_data = '0HP'`。
- 数据：仓库自带 `data/0HP_TrainNoised_-6`（训练 + 验证）与 `data/0HP_TestNoised_-6`（测试），10 类，由作者 `data_split.py` 生成（已加 −6 dB 噪声）。`prepro` 每类随机取 750 个长 2048 的窗（训练文件，`enc_step = 28` 的重叠增强），再按 `StratifiedShuffleSplit(test_size=1/3)` 分出验证集；测试：每类 250 个窗，由 `inference10` 用 10 个不同的随机种子各抽一次。
- 运行方式：`WANDB_MODE=offline`（作者代码必须调用 wandb，且写死了作者自己的 entity；离线模式不改代码）。

## 种子方案

- 作者有种子机制（`config.seed`，默认 42，`random_seed()` 设 torch / numpy / random 并开 cudnn 确定性）。
- **不能用 0–9**：`inference10` 用 `seed - 5 … seed + 4` 作为测试抽样种子，`seed = 0` 时会调用 `np.random.seed(-5)` 报错。故用 **42–51**（作者默认 42 起连续 10 个），写入偏离记录。测试抽样种子随之是作者设计里的 `seed ± 5`。

## 作者的选模规则

- 代码意图：每轮在验证集上算准确率，`acc >= acc_max` 时 `net_best = student_model`。但这是**引用赋值、不是拷贝**，训练结束时 `net_best` 指向的就是最后一轮的模型。**实际作者口径 = 最后一轮（第 75 轮）**。Teacher 同理（`tnet_best = teacher_model`）。
- 作者报告值：`inference10` 在 10 次测试抽样上的**宏平均 F1** 的均值（`test_f1`），论文表中的数也是 F1。
- **选模集**：验证集（训练文件中 1/3）。**报告集**：测试集（独立的测试文件）。
- 补丁：每轮记作者算出的验证集准确率与损失（`select_split = "val"`）；在 `isolate` 内用作者自己的 `inference` 逻辑，在作者 `inference10` 会用到的同 10 个测试抽样上算测试准确率、损失与宏平均 F1（准确率与损失进 `"test"` 记录，F1 另存）；结束时 `author_pick(75)`；作者报告的 `test_f1` 与记录的最后一轮 10 次抽样平均 F1 另行核对（`author_check_f1`），`author_check` 按准确率无作者报告值，记为不可核对。

## 数据来源

- 仓库自带的 `data/0HP_TrainNoised_-6`、`data/0HP_TestNoised_-6`（各 10 个 `.mat`）。作者 `prepro` 读取每个文件中所有含 “DE” 的键并保留最后一个；这些文件是作者生成的派生数据，不含 99.mat 问题（0HP 正常数据为 97.mat）。

## 泄漏编码（只记录，不修复）

- 训练窗以 28 点步长重叠增强后随机划分训练 / 验证，验证集与训练集样本高度重叠；测试来自独立文件（作者 `data_split.py` 先切分再加噪）。
- 测试抽样每次运行不同（种子随运行变），报告的是 10 次测试抽样的均值。
