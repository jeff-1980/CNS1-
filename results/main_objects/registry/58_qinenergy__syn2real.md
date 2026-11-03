# 对象 58 登记：qinenergy/syn2real

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-23。

## 论文与代码

- 论文：Q. Wang, C. Taal, O. Fink，*Integrating Expert Knowledge with Domain Adaptation for Unsupervised Fault Diagnosis*，IEEE TIM 2021，DOI 10.1109/TIM.2021.3127654；预印本 arXiv 2107.01849。
- 仓库：`qinenergy/syn2real`，commit `603fbfd0f0c414ed034f20d213f16b7251ad6553`（2022-11-07），README 称复现论文 CWRU 合成→真实结果。本地 `~/audit_ext/main_objects/58_qinenergy__syn2real`。唯一入口 `code/Syn2real_CWRU.ipynb`（Colab，TensorFlow 1.x）。**待核实项“框架”**：笔记本为 TensorFlow 1.15（`%tensorflow_version 1.x`、`tf.contrib`、`tf.Session`），成立。

## 头条表格

- 论文 Table III（合成 CWRU 有标签源域 → 真实 CWRU 目标域，4 类，**平衡准确率**，10 个种子均值）：Source-only 0.596、DANN 0.726、Conditional（CDAN）0.754、**Proposed（CDAN + mixup）0.823**（IAST 0.760 不在仓库中）。S6 以此对照。

## 方法组（4 个）

- 笔记本中的四个设置：`Source-only`（单元 4）、`DANN`（单元 7）、`CDAN`（单元 9）、`CDAN-mixup`（单元 11，提出方法），组名 `<方法>|Syn->Real CWRU 4-class`。单元 6 的 “balanced DANN”（以全部目标测试数据作无标签训练数据，对应 Table II）不跑。

## 配置（作者笔记本默认）

- 数据：仓库自带 `data/preprocessed/` 的四个 `.npy`（合成与真实 CWRU 驱动端包络谱，4000 × 1000），与 GitHub release 中的同名文件一致；标签合并为 4 类（健康、内圈、滚动体、外圈）。目标无标签训练数据：健康 1200 + 内圈 5% + 滚动体 1% + 外圈 10%（笔记本注释中的百分比与代码不符，以代码为准）。
- 训练：100 轮，Adam lr 1e−3，batch 128，隐藏维 256，GRL 系数 p = 0.1，mixup β = 1。
- **笔记本转脚本**：驱动逐字执行笔记本的代码单元——单元 0 去掉 Colab 魔法命令（`%tensorflow_version`、`!rm`、`!wget`、`!md5sum`；数据直接读仓库内的 `data/preprocessed/`），单元 1–2 原样并只加一行钩子，各组执行其单元中 `res = []` 之前的数据准备与 `args`，再按作者的循环 `args.seed = i; run()`。执行的代码另存为 `audit_syn2real_cells.py` 随补丁提交。

## 种子方案

- 作者在 `run()` 中 `tf.set_random_seed(args.seed)`、`seed(args.seed)`，循环 i = 0..9。**种子 0–9**，与作者一致。

## 作者的选模规则

- 每轮在目标测试集上评估一次，只保留最后一轮的值；报告最后一轮的平衡准确率（10 个种子的均值 ± 标准差）。**作者口径 = 最后一轮（author = last）**。**选模集**：无。**报告集**：真实 CWRU 目标测试集。
- 记录：钩子在作者每轮评估之后记录平衡准确率（论文指标，作为 `acc`）、普通准确率（另存）与测试损失；`finish` 接收作者 `run()` 返回的平衡准确率。

## 环境

- 由对象 04 的 conda 环境（Python 3.8、nvidia-tensorflow 1.15.5+nv21.12）复制，补装 tqdm。

## 泄漏编码（只记录，不修复）

- 直推式：目标域无标签训练数据是目标测试集的子集。
- 源域与目标测试集的健康样本取自同一批真实记录的两半，且各复制 6 次；同一文件内的窗口相互重叠。
- 每轮在测试集上评估（作者取最后一轮）。
