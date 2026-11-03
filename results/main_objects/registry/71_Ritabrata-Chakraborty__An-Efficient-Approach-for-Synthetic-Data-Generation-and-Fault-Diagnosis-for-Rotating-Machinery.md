# 对象 71 登记：Ritabrata-Chakraborty/An-Efficient-Approach-for-Synthetic-Data-Generation-and-Fault-Diagnosis-for-Rotating-Machinery

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练（含冒烟）之前提交。登记日期 2026-10-24。

## 论文与代码

- 论文：*An efficient approach for synthetic data generation and fault diagnosis for rotating machinery*，PHM 2025（IET Conference Proceedings 2025(10):241–247），DOI 10.1049/icp.2025.2364（全文未取得）。
- 仓库：commit `4adf0ac07cefc0db39db67ca6ec7a81c55eb6b3c`，README 称 “official implementation”；入口 `Final.ipynb`（README “Run Complete Pipeline”）。

## 头条表格

- 论文全文不可得，README 无结果表，头条论文数值无法登记（S6 记为“无法对照”）。
- 仓库中保存的 notebook 输出（作者一次运行，仅作参照，不是论文值）：测试准确率 Real100+Gen0 **95.60 %**、Real100+Gen400 **97.90 %**、Real100+Gen900 **96.15 %**；`Results/480_training_time.npy` 记录 GAN 训练 7850 s。

## 方法组（3 个，作者 notebook 实际训练的配置 `['100','500','1000']`）

- `Real100+Gen0|CWRU DE 10-class (N 1000, faults 100 real)`（不增强）
- `Real100+Gen400|…`
- `Real100+Gen900|…`
- 分类器都是 ACWGAN-GP 的 TCN 判别器结构，从头训练 20 轮；生成样本来自本重复中从头训练的生成器。

## 配置（notebook 默认，全部不改）

- 数据：仓库自带 `Datasets/CWRU/DE/*.csv`（10 类）；每个文件整体 min-max 到 [−1, 1]，长度 1824、步长 36 的重叠切段，`random.seed(42)` 打乱后按顺序切分。
- GAN：每个故障类 480 段、N 类 4800 段；50 轮，batch 32，Adam 1e−4 (0.5, 0.9)，GP 系数 10，n_critic 5。
- 分类器：故障类各 100 段真实样本 + 0 / 400 / 900 段生成样本，N 类 1000 段；20 轮，Adam 1e−4；验证集每类 200 段、测试集每类 200 段（两者不重叠）。

## 种子方案

- notebook 只给切段打乱设 `random.seed(42)`，torch / numpy 不设种子 → **10 次不设种子的重复**，`seed` = 重复序号 0–9；每次重复都**从头训练 GAN**（工作副本中不含仓库附带的 GAN / 分类器权重，每次重复开始前删除上一次的权重，否则作者代码会直接载入已有权重）。

## 作者的选模规则

- 意图：按验证准确率（严格 >）保存最好的模型，再在测试集上报告。
- **实际**：`best_model_state = discriminator.state_dict()` 保存的是引用，之后的训练继续原地修改这些张量，最后写盘的是**最后一轮**的权重。审计按实际行为记 `author_pick` = 最后一轮，另外记录意图中的最佳验证轮（`intended_best_val_epoch`）。核对：最后一轮记录的测试准确率应与作者打印的 `Classification Accuracy` 一致。
- **选模集**：验证集（与测试集不重叠）。**报告集**：测试集。

## 运行方式与改动

- driver `audit_run71.py` 按顺序执行 notebook 的代码单元格；跳过单元格 4（原始信号作图）、15（相似度指标并写 Generated_Data）、17（作图），这些单元格都不影响分类器。`plt.show` 改为关图（无显示环境）。
- 钩子：单元格 26 的每轮验证之后调用一次（隔离状态下另算测试集准确率与损失）；单元格 27 打印的测试准确率交给 `AUDIT_FINAL`。

## 泄漏编码 / 作者代码问题（只记录，不修复）

- **GAN 训练集包含分类器的验证集和测试集**：GAN 阶段用每类打乱后的前 480 段，分类器阶段的测试集是第 100–299 段、验证集是第 300–499 段，都在这前 480 段之内；生成器见过测试样本。
- 归一化用整个文件的 min/max（含测试段）；相邻切段重叠 98 %，训练段与测试段的时间片重叠。
- 生成器 `forward` 把 `self.main` 调用两次（BN 统计更新两次）；保存“最佳”模型实际保存最后一轮（见上）。

## 预算

- 作者记录 GAN 一次训练 7850 s；10 次重复预计远超 3 h 总预算及每次 2 GPU-h 上限。冒烟后实测每轮耗时再请 jeff 决定。
