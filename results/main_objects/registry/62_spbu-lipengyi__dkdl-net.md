# 对象 62 登记：spbu-lipengyi/dkdl-net

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-24。方法组与修复按 jeff 2026-10-23 裁定。

## 论文与代码

- 论文：*DKDL-Net: A Lightweight Bearing Diagnosis Detection Model via Decoupled Knowledge Distillation and LoRA Fine-tuning*，Scientific Reports 2025，DOI 10.1038/s41598-025-06734-y（开放获取）。**待核实项“框架”**：PyTorch，成立。
- 仓库：`spbu-lipengyi/dkdl-net`，commit `56b3c881512fee1ee49000a4f8b4206a613b62d6`（2024-06-09）。本地 `~/audit_ext/main_objects/62_spbu-lipengyi__dkdl-net`。入口 `train_dkd.py`（教师 + DKD 学生）、`train_single.py`（无 KD 学生）、`fine_s.py`（LoRA 微调）。

## 头条表格

- 论文 Table 7（宏平均 F1，%）：Teacher 99.59、Student 97.52、DKDL-Net（DKD 学生 + LoRA）99.50。论文未写噪声、SNR 与负载；代码默认 0HP、SNR −6 dB。S6：Teacher、Student 两组对照；DKD 学生（无 LoRA）在论文中没有单独数值。

## 方法组（3 个，jeff 裁定）

- `Teacher|CWRU 0HP -6dB`（`train_dkd.train_teacher`，TCNN）、`DKD-Student|CWRU 0HP -6dB`（`train_dkd.train_student`，SCNN + DKD）、`Student|CWRU 0HP -6dB`（`train_single.train`，SCNN 无 KD）。
- **LoRA 组（DKDL-Net）不跑**：`fine_s.py` 第 91 行把 `TCNN.state_dict(ckpt)` 误用为加载，学生权重实际没有载入，LoRA 模型从零训练、不冻结任何参数，LoRA B 也非零初始化，与论文描述的“在 DKD 学生上做 LoRA 微调”不符。

## 配置（作者代码默认）

- 数据：CWRU 12 kHz 驱动端 0 hp 的 10 个文件（97、105、118、130、169、185、197、209、222、234；论文 Table 4 的 10 类），从本机 `~/论文CNS/data/cwru_12k_de` 只读复制到 `~/audit_ext/work62/data/0HP`；用作者 `utils/data_split.preprocess(noise=True, snr=-6)` 一次生成加噪的训练段（每个信号前 2/3）与测试段（后 1/3）（作者噪声不设种子，只生成一次）。`prepro`：训练每类 750 个窗（随机起点后步长 28 连取，长 2048），`StratifiedShuffleSplit` 分训练 5000 / 验证 2500；测试每类 250 个窗。标签按 `os.listdir` 顺序。
- `train_dkd.py` 默认：batch 128、75 轮、SGD 0.1（教师与学生）、alpha 0.2、beta 5、gamma 1、温度 2.5、warmup 5、SNR −6、0HP。
- `train_single.py` 默认：batch 64、75 轮、SGD lr 0.5、0HP，**但其默认 SNR 为 8 dB；为与另两组同一任务，取 −6 dB（偏离记录）**。

## 种子方案

- 作者训练只用种子 42（`random_seed`），`inference10` 用 `config.seed − 5 … config.seed + 4`（即 37–46）的 10 个种子重抽测试窗。按预注册**训练用种子 0–9**；测试重抽按作者协议固定为 37–46（即以 `config.seed = 42` 调用作者的 `inference10`），各组同一组测试集。

## 作者的选模规则

- 训练循环中 `tnet_best = teacher_model`、`net_best = student_model` 只保存对活动模型的引用，因此返回并评估的是**最后一轮**的模型。**作者口径 = 最后一轮**。报告值 = 作者 `inference10` 在 10 次测试重抽上的宏平均 F1 均值。**选模集**：无（验证集只用于打印）。**报告集**：测试集（10 次重抽）。
- 补丁：`train_dkd.py`、`train_single.py` 加模块级 `AUDIT_HOOK`，在每轮验证阶段结束后调用（各组一行，共 3 行）：记录作者的验证准确率与验证损失，并在 `isolate` 中补算 10 次测试重抽上的平均准确率、交叉熵与宏平均 F1；作者 `inference10` 的 F1 与记录中作者轮次的 F1 另行核对（`author_check_f1`）。
- **一行修复（jeff 裁定）**：`train_dkd.py` 的 `inference10` 少了 `test_X = test_X[:, np.newaxis, :]`（`train_single.py` 中有），测试时会因输入维度错误崩溃，补上这一行。

## 泄漏编码（只记录，不修复）

- 验证集与训练集来自同一批步长 28 的重叠窗口，几乎重复；测试集的标准化在测试集自身上拟合（直推式）；加噪不设种子。
- 每个批次都重新创建优化器与调度器（动量每步清零，余弦调度不生效），只记录。
