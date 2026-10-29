# 对象 08 登记：junior209lsj/FaultDiagnosisOptimizerBenchmark

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119。组与种子的套用方式由 jeff 2026-10-20 裁定（见下）。本登记在任何训练之前提交。登记日期 2026-10-20。

## 论文与代码

- 论文：Lee & Kim，*Impact of Deep Learning Optimizers and Hyperparameter Tuning on the Performance of Bearing Fault Diagnosis*，IEEE Access 11 (2023) 55046–55070，DOI 10.1109/ACCESS.2023.3281910（开放获取，CC BY；审计方经 IEEE Xplore 网页读到正文与表 6、表 8）。
- 仓库：`junior209lsj/FaultDiagnosisOptimizerBenchmark`，commit `536b97040cfd66cee194c0376c475d3554d77f2e`（2023-09-05）；仓库创建于 2023-05-04，与论文同期（不进入“代码晚于论文 ≥ 1 年”的剔除集）。本地 `~/audit_ext/main_objects/08_junior209lsj__FaultDiagnosisOptimizerBenchmark`。
- 入口：`benchmark.train(...)` 与 `benchmark.test(...)`（PyTorch Lightning 1.7.7）；用法以作者的 `tutorial.ipynb` 为准。

## 研究设计与头条表格

- 这是一篇基准研究：2 个数据集（CWRU、MFPT）× 7 个模型 × 4 个优化器（SGD、Momentum、RMSProp、Adam）× 3 个 batch size（16、64、128）。第一步在初始搜索空间用对数尺度准随机采样训练 64 次，第二步在收窄空间（表 6）再训练 16 次；每次训练 100 轮，**按验证集损失最低的一轮选模**，在测试集上报告。
- 头条表格：表 8 “Comparison of Candidate Models in Terms of Accuracy, Model Size, and Computation Load”。WDCNN 在 CWRU（无噪声）上 **99.81 ± 0.08**（95% 置信区间），CWRU 加噪 69.81 ± 1.08。论文没有按优化器分列的无噪声数值表（第二步按优化器与 batch size 的结果只以图 16 给出加噪数据）。

## 方法组（4 个）与套用方式（jeff 2026-10-20 裁定）

4 组 = 4 个优化器，模型固定为 WDCNN（作者教程的示例模型），数据 CWRU 无噪声。每组的超参数取**论文表 6 中 WDCNN 的收窄搜索空间里的第一个准随机样本**。作者的 `fdob.log_qsample` 用不加扰的 Halton 序列（`scramble=False`），第一个点在每一维都是 0，对应每个超参数取搜索区间的下界：

| 组 | 表 6 收窄空间（WDCNN，CWRU） | 第一个准随机样本 → 实际取值 |
|---|---|---|
| SGD | η ∈ [10⁻², 10⁰] | lr = 0.01 |
| Momentum | η ∈ [10⁻³, 10⁻¹]，1−γ ∈ [10⁻¹, 10⁰] | lr = 0.001，momentum = 0.9 |
| RMSProp（α = 0.99，ε = 10⁻⁸） | η ∈ [10⁻⁴, 10⁻²]，1−γ ∈ [10⁻¹, 10⁰] | lr = 0.0001，momentum = 0.9，alpha = 0.99，eps = 1e-8 |
| Adam（β₁ = 0.9，β₂ = 0.999，ε = 10⁻⁸） | η ∈ [10⁻⁴, 10⁻²] | lr = 0.0001，betas = (0.9, 0.999)，eps = 1e-8 |

组名：`WDCNN|CWRU bs64|<优化器>`。

## 配置（作者默认 / 论文设置）

- 数据：按作者 `download_cwru("…", "12k")` 的文件表读取 12 kHz 驱动端数据；依教程排除标签 999（0.028 英寸与 @03、@12 外圈）与 0 hp 负载，10 类。每条信号按时间顺序 60% / 20% / 20% 切成训练 / 验证 / 测试（`split_dataframe`），再按长度 4096、步长 2048 切片（`build_from_dataframe`），送入模型时截取前 2048 点（WDCNN 输入长度）。
- 训练：`benchmark.train`，100 轮（论文），**batch size 64**（论文研究了 16 / 64 / 128，没有单独的“主设置”；取居中的 64，并在此登记），`CrossEntropyLoss`，`n_classes = 10`。
- 选模：`ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)`，验证间隔 `val_check_interval = len(train) // batch_size` 个批次（约每轮一次）。
- 测试：`benchmark.test(test_loader, …)`，在测试集上报告 `test_acc`。

## 种子方案

作者有种子机制（`benchmark.train / test` 的 `random_seed` 参数，设 torch / numpy / random 种子并开 `torch.use_deterministic_algorithms(True)`）。用种子 **0–9**。

## 作者的选模规则与待核实的 bug

- 意图上的规则：验证集损失最低的那一轮（`ModelCheckpoint`），在测试集上报告。**选模集 = 验证集**，**报告集 = 测试集**，二者独立。
- 疑似 bug（冒烟中核实，只记录不修复）：`train()` 末尾与 `test()` 里都写的是 `training_module.load_from_checkpoint(...)`，但没有接收返回值。PyTorch Lightning 1.7 中 `load_from_checkpoint` 是类方法，返回新对象、不改原对象。若属实：`train()` 末尾 `trainer.test(..., dataloaders=val_loader)` 评估的是最后一轮的权重；`test()` 评估的是用同一种子新建、**未训练**的模型。
- 认定口径（试跑后按实测写入偏离记录）：`author` 协议记为 `ModelCheckpoint` 实际保存的那一轮（验证损失最低）；作者代码自己打印的最终测试数作为 `finish()` 的核对值，若与记录不一致，`author_check` 即为不一致，并在偏离记录中说明原因。

## 逐 epoch 记录方式

- 不改作者源码：审计驱动脚本在调用 `benchmark.train` 前，给 `pytorch_lightning.Trainer` 注入一个回调。每次验证结束时记录：作者算出的验证集准确率与损失（`select_split = "val"`）；在 `audit.torch_utils.isolate` 内用同一前向计算测试集准确率与损失（`report_split = "test"`）；`ModelCheckpoint` 的最好值更新时 `author_pick`。

## 数据来源

- 本机 `~/论文CNS/data/cwru_12k_de` 已有作者文件表中的 10 类 40 个文件，按作者命名（如 `N_000_1.mat`）建软链接；作者文件表中其余文件（0.028 英寸、@03、@12 外圈，标签 999，随后被排除）由作者代码按原链接从 engineering.case.edu 下载。
- 读取的键：作者代码遍历所有含 “DE” 的键并取最后一个。99.mat 含 `X098_DE_time` 与 `X099_DE_time`，按文件中的顺序取到的是 `X099_DE_time`（正确的 1750 rpm 正常数据）。

## 泄漏编码（只记录，不修复）

- 训练 / 验证 / 测试按信号时间顺序切分，切片不跨段，三者无样本重叠。
- 选模在独立验证集上做，不碰测试集。
