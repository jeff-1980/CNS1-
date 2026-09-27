# 预注册 E4：paperB `cwru_clean_v2` 保存的是否为早期 checkpoint

> 2026-09-24 写定，写于任何 E4 训练之前。本文件的 SHA256 存于 `results/registry/prereg_e4_sha256.txt`；`analyze_e4.py` 运行时校验，哈希不符即拒绝判定。
> 上游：`VERDICT_ext.md` 综合第 3、4 点。

## 0. 假设

paperB 的训练循环按"验证准确率严格提升"选择 checkpoint，并在最后重新载入该 checkpoint。`cwru_clean_v2` 的 30/30 个结果中 best_val_acc = 1.0。本项目在相同数据下看到首次达到 1.0 出现在 epoch 1–2（E1）。
**H_early**：paperB 报告的 `cwru_clean_v2` 模型行为接近训练早期（严格规则选中的）checkpoint，而不是第 100 轮的模型。

## 1. Harness（以 paperB 代码为准）

- **原样使用** `paperB_supplement/code.zip` 中的函数：`train/trainer.py` 的 `set_seed`、`build_model`、`train_one_epoch`、`evaluate`、`evaluate_snr_sweep`；`data/dataset.py` 的 `make_dataloaders`（`drop_last=True`，`num_workers=4`）；`train/config.py` 的 `TrainConfig` 默认值（100 epoch、batch 64、AdamW 1e-3 / wd 1e-4、warmup 5 + CosineAnnealingLR(T_max=95, eta_min=1e-6)、grad clip 1.0、d_model 64）；类权重按 `run_experiment` 的写法（逆频率，归一到和为 K）
- **外层循环按 `run_experiment` 逐行重建**，只做三处替换：
  (a) 切分改为 `splits_v2.build_cwru_splits(16 文件, loads 0–3, 1024, 0.5, split_seed=42)` 的 train / val / test；
  (b) `noise_type=None`（NA-，即 `clean_NA-`）；
  (c) SNR 扫描在 test 集上进行，10 档 {−8, −6, …, 10}，与 `cwru_clean_v2` 的键和测试行和 [944, 237, 236, 236] 一致
- **已知偏离**：补充材料没有收录生成 `cwru_clean_v2` 的 v2 驱动脚本，上述 (a)–(c) 由结果文件的元数据（`training_regime=clean_NA-`、`split_protocol=leak_free_v2_group_by_file`、`split_seed=42`、测试行和）推断得出
- `mamba_ssm` 以空桩模块代替（只被 Mamba 类模型使用，这些模型不在本实验中）；参数量已核对：cnn1d 323012 / wdcnn 98020 / drsn 109188，与 paperB 一致
- 附加记录（在 `torch.random.fork_rng` 内进行，不扰动训练随机数流）：每个 epoch 的 val_acc；每个 epoch 在 test 集上 −8 dB 的靶标与浓度（1 次噪声抽样）；在 epoch {1, 2, 3, 5, 10, 25, 50, 100} 上做完整 10 档 SNR 扫描

## 2. 实例

cnn1d / wdcnn / drsn × seed 0–4 × 100 epoch = **15 个实例**。对每个实例，保存严格规则选中的 checkpoint（best）与第 100 轮的 checkpoint（final），并分别做 `evaluate_snr_sweep`。

## 3. 指纹距离

对每个实例和每个 checkpoint c，d(c) = 10 档 SNR 上 |acc_本项目(c) − acc_paperB| 的均值。其中 acc_paperB 取**同架构 5 个种子的均值曲线**（主分析；种子编号本身在不同硬件上不可复现）。同种子配对的距离只作描述。

## 4. 判定（按顺序）

1. **NOT_EARLY**：best_epoch > 5 的实例 ≥ 12/15 → 在 paperB 的循环下，严格规则**不会**选中早期 checkpoint，H_early 按原表述被否定
2. **EARLY_CKPT**：best_epoch ≤ 5 的实例 ≥ 12/15，**且** d(best) < d(final) 的实例 ≥ 12/15（符号检验单侧 p = 0.018）
3. **FINAL_LIKE**：d(final) < d(best) 的实例 ≥ 12/15
4. 其余情况 → **INDETERMINATE**

逐架构结果也要报告。若某个架构（尤其 cnn1d）与总体方向相反，照实写出。

## 5. 次要结果（描述性）

- best 与 final checkpoint 在 −8 dB 下的靶标 vs paperB（三个架构都是 IR 5/5）
- d(epoch) 曲线（在第 1 节列出的 8 个 epoch 上）：距离最小的 epoch
- 各实例的 val 首次达到 1.0 的 epoch

## 6. 解释约束

- `EARLY_CKPT` 允许的表述："paperB 的 `cwru_clean_v2` 结果与训练早期 checkpoint 的行为一致，而与第 100 轮不一致"。**不允许**写成"paperB 保存了 epoch N 的模型"（缺少驱动脚本和原始权重）
- 任何标签都不构成对 paperB 结论的否定：早期 checkpoint 同样是一个合法的模型。标签只影响"靶标"这一结论的**适用条件**（训练时点）
