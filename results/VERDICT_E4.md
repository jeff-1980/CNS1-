# 判定 E4：**EARLY_CKPT**。更强的发现是：harness 逐位复现了 paperB `cwru_clean_v2`

> 2026-09-24。预注册 `PREREG_E4_paperB_checkpoint.md`（SHA256 `a3ab8c93…`，写于训练之前；`analyze_e4.py` 运行时校验通过）。执行偏离见 `DEVIATIONS_E4.md`（仅一项：`num_workers` 4→0，数值等价）。
> 数据 `e4_paperB_harness.json`（15 个实例，每个实例的 best 与 final 权重都保存了 SHA256）；判定 `e4_verdict.json`；事后分析 `e4_posthoc_collapsed_cells.json`；图 `e4_checkpoint.png`。

## 1. 预注册判定

| 条件 | 结果 |
|---|---|
| best_epoch ≤ 5 | **15/15**（14 个为 epoch 1，drsn seed3 为 epoch 2） |
| d(best) < d(final)（对照 paperB 同架构均值曲线） | **13/15**（例外为 wdcnn s0、drsn s2，均为 5 种子均值指纹的噪声） |
| 标签 | **`EARLY_CKPT`** |

## 2. 预注册之外、更强的结果：150/150 个混淆矩阵逐格相同

这一项是**描述性核对，未预注册**，但它不依赖任何阈值。
harness 从头到尾不读取 paperB 的结果文件。它自己训练出的 best checkpoint，在 10 个 SNR 档上的混淆矩阵与 `paperB_supplement/results/collapse.zip` 中 `cwru_clean_v2/{cnn1d,wdcnn,drsn}/seed{0–4}_summary.json` 的矩阵**逐格计数完全相同**：15 个实例 × 10 档 = 150 个矩阵，最大计数差 = 0。

由此成立以下三点：
1. **harness 精确重建了缺失的 v2 驱动脚本**：预注册 §1 中根据元数据推断的切分、NA- 训练方式和测试集都正确
2. **paperB 公布的 `cwru_clean_v2` 结果（cnn1d / wdcnn / drsn）就是训练 epoch 1（drsn s3 为 epoch 2）的模型**：每个 epoch 只有 44 步，也就是训练了 44–88 步。这一结论不再是"行为一致"，而是逐位相同
3. 同一批运行的第 100 轮模型可以直接比较（下一节）

## 3. 第 100 轮上的 paperB 判据（事后分析，描述性）

崩溃 cell 的定义沿用 paperB 判据：5 个种子浓度的均值 ≥ 90%。

| 架构 | best（= paperB 公布值） | 第 100 轮 | −8 dB 靶标（第 100 轮） |
|---|---|---|---|
| cnn1d | 6 个 cell（−8…+2 dB），IR 5/5 | **0 个 cell**（浓度 80–87%） | OR 2 / IR 3 |
| wdcnn | 3 个（−8…−4 dB） | **5 个**（−8…0 dB） | IR 5/5 |
| drsn | 2 个（−8、−6 dB） | **5 个**（−8…0 dB） | IR 5/5 |
| 合计 | 11 个（与 paperB 清单一致） | 10 个 | |

- **wdcnn、drsn**：崩溃现象与 IR 靶标在完整训练后**依然存在，而且扩展到更高的 SNR**。paperB 的核心现象在这两个架构上与训练时点无关，按第 100 轮计算反而更强
- **cnn1d**：6 个崩溃 cell 全部是**训练早期状态**。训练到最后不再满足 90% 判据，靶标也在 OR 与 IR 之间分裂（与 E1 在 Test_40 上的结果一致）
- 附带观察：第 100 轮模型在中等 SNR 下更容易崩溃。例如 wdcnn 0 dB 的浓度 69.6% → 96.0%，cnn1d +4 dB 的准确率 0.645 → 0.269。也就是说，只训练 1 个 epoch 的模型在噪声下反而更稳健

## 4. 范围与未评估项

| 对象 | 状态 |
|---|---|
| `cwru_clean_v2` cnn1d / wdcnn / drsn（25 个崩溃 cell 中占 11 个） | **已验证**（逐位复现） |
| `cwru_clean_v2` lstm / transformer / vibrmamba（25 个中占 3 个） | best_val 全为 1.0，**风险相同，本轮未评估**（vibrmamba 需要 `mamba_ssm`，本环境没有） |
| `ablation_v3` mamba2_no_noise_train（25 个中占 3 个） | best_val 5/5 为 1.0，**未评估**（需要 `mamba_ssm`） |
| `main_awgn_v2`（NA+，35 个运行） | best_val 35/35 为 1.0，**未评估**。训练时加了噪声，但验证集是干净的，同样可能在早期就达到 1.0 |
| JNU `jnu_modeA_v2` / PU `paperB_pu_collapse`（25 个中占 8 个） | best_val 为 1.0 的分别只有 10/35 和 0/35，选中的**不一定是早期 checkpoint**，未评估 |

## 5. 允许与不允许的表述

**允许**："用 paperB 代码包中的训练函数与 v2 切分重新训练，得到的严格规则 checkpoint 与 paperB 公布的 `cwru_clean_v2`（cnn1d/wdcnn/drsn）混淆矩阵逐格相同，均位于 epoch 1–2。按同一判据，在第 100 轮：wdcnn 与 drsn 的 IR 崩溃依然存在并扩展到更高 SNR，cnn1d 的崩溃 cell 由 6 个减为 0 个。"

**不允许**：
- ~~"paperB 的崩溃现象是早停造成的假象"~~：在 wdcnn、drsn 上，第 100 轮的现象更强
- ~~"paperB 的全部 CWRU / JNU / PU 结果都是 epoch 1 模型"~~：只验证了第 4 节中第一行
- ~~"早期 checkpoint 不是合法模型"~~：严格规则选模本身是常见做法。问题在于这一选模方式没有在论文中作为结论的适用条件报告出来

## 6. 这件事的性质与建议（需要你决定）

paperB（MST 发表版，`10.1088/1361-6501/ae9438`）是你们自己的已发表论文。本结果**不否定**它的核心现象：在 wdcnn、drsn 上，完整训练后现象更强。但它影响三件事：
1. **已发表的崩溃 cell 计数与架构分布**：其中 cnn1d 的 6 个 cell 属于 epoch-1 状态
2. **"模型训练充分"这一隐含前提**：44–88 步的模型在论文里被当作训练完成的模型报告
3. **方法描述**："训练 100 个 epoch"在文字上属实，但实际报告的权重来自 epoch 1

是否需要勘误或致编辑部，取决于论文正文如何描述训练与选模，以及 cnn1d 的 cell 在论证中的分量。**这由你和共同作者判断，我不替你下结论。** 在此之前建议补齐第 4 节中的未评估项，至少包括 `main_awgn_v2`（不需要 `mamba_ssm` 的 5 个架构）和其余 3 个 clean 架构，这样影响范围就能完整确定。
