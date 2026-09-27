# 判定 WP1：协议基线补全 —— H1 **不成立**；CWRU 的 17 个 NA- 崩溃 cell 全部复现

> 2026-09-24 23:17 运行结束。预注册 `PREREG_WP1_protocol_baseline.md`（SHA256 `257f20a3…`，`analyze_wp1.py` 运行时校验通过）。偏离：`DEVIATIONS_WP1.md`（15:07 会话重置中断，15:15 续跑）。
> 数据：`wp1_{cwru_clean,cwru_clean_M,cwru_m2,cwru_awgn,cwru_awgn_M,jnu,jnu_M}.json`（90 个实例 + 3 个转速识别实例，每个实例都保存了 best / final 权重及 SHA256）；判定 `wp1_verdict.json`。
> 按规划 §0：best checkpoint 就是 paperB 的选模协议，第 100 轮只是另一种协议；下文不含任何"paperB 有误"的含义。

## 1. 预注册假设 H1：**NOT_SUPPORTED**

H1：在 paperB `best_val = 1.0` 的 NA- 实例中，本次 best_epoch ≤ 5 的比例 ≥ 80%。
**结果：20/30 = 67%，未达到 80%。**

| 单元 | best_epoch ≤ 5 |
|---|---|
| C-clean / lstm | 5/5 |
| C-clean / transformer | 5/5 |
| C-clean-M / vibrmamba | 5/5 |
| C-m2 / mamba2 | 5/5 |
| **J / cnn1d** | **0/5**（best_epoch 23–73） |
| **J / wdcnn** | **0/5**（best_epoch 10–54） |

原因是 JNU 的验证集准确率很晚才达到 1.0：cnn1d 在第 23–73 轮，wdcnn 在第 10–54 轮。CWRU 的 16 文件验证集则在第 1–2 轮就达到 1.0。
**按预注册的后果条款：A3（训练时点）的证据范围不扩展，仍限于 E4 的 cnn1d / wdcnn / drsn。** 下面第 3 节中 CWRU 其他架构的结果只作为描述性证据。

## 2. 复现等级（逐单元）

| 臂 | 架构 | 等级 | 曲线 MAE | 与事前预期 |
|---|---|---|---|---|
| C-clean | lstm | BEHAVIORAL | 0.030 | ✗（预期 BITWISE） |
| C-clean | transformer | **BITWISE** | 0 | ✓ |
| C-clean-M | vibrmamba | **BITWISE** | 0 | ✗（预期非 BITWISE） |
| C-m2 | mamba2（非融合路径） | **BITWISE** | 0 | ✗（预期非 BITWISE） |
| C-awgn | 5 个非 Mamba 架构 | BEHAVIORAL | 0.011–0.022 | ✓（不可能 BITWISE） |
| C-awgn-M | vibrmamba / mamba2 | BEHAVIORAL | 0.029 / 0.017 | ✓ |
| J | 5 个非 Mamba 架构 | BEHAVIORAL | 0.003–0.013 | —（转速未确证） |
| J-M | vibrmamba / mamba2 | BEHAVIORAL | 0.005 / 0.005 | — |

- Mamba 类出乎预期地逐位一致，说明 paperB 的运行环境很可能与 `mamba-cu12`（torch 2.5 系列）相近；也说明非融合路径在数值上与 paperB 的结果完全相同
- lstm 没有逐位一致，原因未查明（候选：cuDNN RNN 内核的非确定性）；不做事后排查

**合并 E4：paperB 最终 25 个 cell 中 CWRU 的 NA- 部分共 17 个**（cwru_clean_v2 14 个 + ablation_v3 mamba2 3 个），**16 个逐位复现**（cnn1d 6、wdcnn 3、drsn 2、vibrmamba 2、mamba2 3），**1 个行为复现**（lstm −8 dB：paperB 91.52%，本次 91.8%，种子靶标同为 Normal×1 + IR×4）。

**JNU（仅行为比较）**：准确率曲线接近，但 **崩溃靶标只部分复现**。paperB 的 4 个 JNU cell 中：
- cnn1d −8 dB（Ball，paperB 96.91%，本次 97.0%）、transformer −8 dB（IR，97.08%，本次 96.2%）两个复现
- **lstm −8 dB 未复现**：paperB 5 个种子全为 Ball（98.58%），本次为 IR×3 + Ball×2（81.9%）
- **transformer −4 dB 未过阈值**：paperB 92.55%，本次 89.5%

→ **准确率曲线一致不代表靶标一致**。JNU 结论只能写"准确率行为一致，靶标部分一致"。

## 3. 描述性结果：best（paperB 协议） vs 第 100 轮

崩溃 cell 按 paperB 判据计（5 个种子浓度均值 ≥ 90%，在 paperB 的 SNR 网格上）。

**CWRU NA-（合并 E4，7 个架构）**

| 架构 | best | 第 100 轮 | 靶标 |
|---|---|---|---|
| cnn1d | 6 | **0** | IR → OR/IR 分裂（E4） |
| wdcnn | 3 | 5 | IR |
| drsn | 2 | 5 | IR |
| lstm | 1 | 2 | Normal×1 + IR×4 |
| transformer | 0 | 0 | — |
| vibrmamba | 2 | 4 | IR |
| mamba2 | 3 | 4 | IR |
| **合计** | **17** | **20** | |

cnn1d 是 7 个架构中唯一一个在完整训练后崩溃 cell 减少的；另外 5 个会崩溃的架构，在第 100 轮都崩溃得更多，并扩展到更高的 SNR。

**CWRU NA+（main_awgn_v2，7 个架构）**：best 与第 100 轮都是 **0 个崩溃 cell**，best_epoch 为 1–17。与 paperB 一致：该实验在 paperB 的最终 25 个 cell 中没有任何一个。

**JNU（7 个架构，仅行为比较）**：best → 第 100 轮变化很小（cnn1d 1→1，drsn 0→1，lstm 1→0，transformer 1→1，其余为 0）。JNU 上本来就没有"早期 checkpoint"现象，这和第 1 节一致。

## 4. 对规划的含义

1. **A3（训练时点）受数据集约束**。早期 checkpoint 只在验证集第 1–2 轮就饱和的 CWRU 上出现；JNU 的验证集饱和得晚，严格选模规则自然选中较晚的 checkpoint。
   事后假设（**未预注册，仅作为 WP3 的候选**）：训练时点效应的出现条件是"验证集早饱和"。WP3 应把验证集饱和轮次作为预注册的协变量
2. **CWRU 的 NA- 基线已完整**：17 个 cell 全部可以复现，第 100 轮结果齐全，WP3 可以直接复用这些权重
3. **NA+ 对照已确认**：加噪训练下 7 个架构都没有崩溃 cell，可作为 WP5 的"设计选择"基线之一
4. **JNU 的靶标只部分复现**：WP4 若要用 JNU 作确认集，必须以本次重训的结果为准，不能引用 paperB 的 JNU 靶标
5. **PU（WP1b）尚未做**：训练脚本需要重建

## 5. 允许与不允许的表述

**允许**：
- "按 paperB 的协议重训，CWRU 上 17 个 NA- 崩溃 cell 中 16 个逐位复现、1 个行为复现"
- "在 CWRU 上，6 个会崩溃的架构中有 5 个在完整训练后崩溃更广；cnn1d 是例外"
- "JNU 上 best checkpoint 位于训练中后期"

**不允许**：
- ~~"paperB 的结果普遍来自早期 checkpoint"~~（H1 不成立，JNU 不是）
- ~~"JNU 复现了 paperB"~~（只有准确率行为一致，靶标部分一致）
- ~~"训练时点效应是普遍规律"~~
