# 预注册 WP1：paperB 协议基线补全（best vs 第 100 轮）

> 2026-09-24，写于任何 WP1 训练之前。规划：`strategy/subjournal-plan.md` v11.1 §4 WP1。
> 训练代码：`code/wp1_harness.py`（由 `e4_paperB_harness.py` 泛化而来，训练循环逐行保持一致）。继承 E4 的偏离：`num_workers=0`（见 `DEVIATIONS_E4.md`）。

## 1. 范围（共 92 个训练实例）

| 臂 | paperB 目录 | 训练方式 | 架构 | 种子 | 环境 |
|---|---|---|---|---|---|
| C-clean | `cwru/cwru_clean_v2` | NA-（`noise_type=None`） | lstm, transformer | 0–4 | `python` |
| C-clean-M | 同上 | NA- | vibrmamba | 0–4 | `mamba-cu12` |
| C-m2 | `cwru/ablation_v3/mamba2_no_noise_train` | NA- | mamba2 | 0–4 | `mamba-cu12` |
| C-awgn | `cwru/main_awgn_v2` | NA+（`noise_type="awgn"`，`train_snr_db` 取 TrainConfig 默认值） | cnn1d, wdcnn, drsn, lstm, transformer | 0–4 | `python` |
| C-awgn-M | 同上 | NA+ | vibrmamba, mamba2 | 0–4 | `mamba-cu12` |
| J | `jnu/jnu_modeA_v2` | NA-（依据：README 把 NA+ 单列为 `jnu_noise_aware/`）；见 §3 | cnn1d, wdcnn, drsn, lstm, transformer | 0–4 | `python` |
| J-M | 同上 | 同 J | vibrmamba, mamba2 | 0–4 | `mamba-cu12` |

CWRU 使用 16 文件 v2 切分（与 E4 相同，测试集 [944, 237, 236, 236]）；JNU 使用 `build_jnu_modeA_splits`（测试集 [581, 190, 190, 190]）。
**不在范围内**：PU `paperB_pu_collapse`。该实验的训练脚本不在代码包内（60 轮、batch 256），需要另行重建，列为 WP1b。

## 2. 记录内容（每个实例）

与 E4 相同：best 与 final 两个 checkpoint 在 10 个 SNR 档上的混淆矩阵；逐 epoch 的 −8 dB 探针；第 1/2/3/5/10/25/50/100 轮的完整扫描；val 曲线；权重及其 SHA256。训练中途不打印靶标。

## 3. JNU 转速识别（先于 J 臂运行）

三种转速切出的类计数完全相同，无法从元数据识别。做法：wdcnn seed0 在 NA- 下分别用 rpm ∈ {600, 800, 1000} 训练。
- 若恰好一种转速的 best checkpoint 与 paperB `jnu_modeA_v2/wdcnn/seed0` 在 3 个 SNR 档上的混淆矩阵逐格相同，就用这一种；J 臂标记为 **可逐位核对**
- 若都不相同：选 per-SNR 准确率曲线 MAE 最小的一种，J 臂标记为 **仅行为比较**，结论中注明"转速与训练方式未能确证"
- 这 3 个识别实例保留，不计入 §5 的判定

## 4. 复现等级（每个 实验 × 架构 单元）

- **BITWISE**：5 个种子的 best checkpoint，在 paperB 公布的全部 SNR 档上混淆矩阵逐格相同
- **BEHAVIORAL**：不满足 BITWISE，但 5 种子均值的 per-SNR 准确率曲线与 paperB 的 MAE ≤ 0.05
- **NOT_REPRODUCED**：以上两者都不满足

**事前预期**（写下来是为了事后核对，不是判据）：
- C-clean 的 lstm、transformer：BITWISE（与 E4 同一环境；`cudnn.deterministic=True`）
- vibrmamba、mamba2：非 BITWISE（torch 与 mamba-ssm 版本不同于 paperB；mamba2 走非融合路径）
- C-awgn、C-awgn-M：**不可能 BITWISE**（训练噪声由加载器进程中的 numpy 抽取，`num_workers=0` 后抽样序列不同）

## 5. 预注册假设

**H1（由 E4 外推）**：在 paperB `best_val_acc = 1.0` 的 NA- 单元中（C-clean、C-clean-M、C-m2，以及 J 臂中 paperB best_val = 1.0 的实例），本次 best_epoch ≤ 5 的实例占 **≥ 80%**。
- 成立 → 规划 A3（训练时点）的证据范围扩展到这些单元
- 不成立 → 按架构报告，A3 保持"仅 cnn1d/wdcnn/drsn"

**不设其他假设。** 以下各项只做描述性报告：各单元 best 与第 100 轮的崩溃 cell 数（paperB 判据：5 种子浓度均值 ≥ 90%）及 −8 dB 靶标；NA+ 单元的 best_epoch 分布；各单元 25 个 cell 的归属变化。

## 6. 报告规则

- 逐单元报告，**不跨架构、不跨实验合并**
- mamba2 的所有结果注明"非融合路径"
- 任何非 BITWISE 单元不得写成"复现了 paperB"；BEHAVIORAL 只能写"行为与 paperB 一致"
- 按规划 §0：不写任何暗示 paperB 有误的表述；best checkpoint 即 paperB 的协议，第 100 轮只是另一种协议
- 运行失败的实例如实列出，不补跑到"好看"为止；如需补跑，写入 `DEVIATIONS_WP1.md`

## 7. 运行顺序

GPU 串行：(1) JNU 转速识别 3 个 → (2) `python` 环境：C-clean、C-awgn、J → (3) `mamba-cu12` 环境：C-clean-M、C-m2、C-awgn-M、J-M。
分析脚本 `analyze_wp1.py` 在运行时校验本文件的 SHA256。
