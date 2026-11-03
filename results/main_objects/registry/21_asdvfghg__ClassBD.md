# 对象 21 登记：asdvfghg/ClassBD

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-22。数据处理按 jeff 2026-10-21、2026-10-22 裁定。

## 论文与代码

- 论文：Liao, He, Li, Sun, Zhang, Zhang，*Classifier-guided neural blind deconvolution: A physics-informed denoising module for bearing fault diagnosis under noisy conditions*，Mechanical Systems and Signal Processing 222 (2025) 111750，DOI 10.1016/j.ymssp.2024.111750。期刊版不开放（ScienceDirect 有人机验证，未绕过）；表格取自作者 arXiv 预印本 2404.15341v1（2024-04-11，同一作者、同一方法）。
- 仓库：`asdvfghg/ClassBD`，commit `08530497e388ed691936f15d6c55267f277ed270`（2025-03-28），README 自称 “official repository”。本地 `~/audit_ext/main_objects/21_asdvfghg__ClassBD`。入口 `train_local.py`。

## 头条表格

- 预印本 Table 4（PU 数据集，宏平均 F1 与 FPR，“All reported results represent the average of ten independent runs”）。按 461bc21 规则 (2) 取第一个工况与第一个 SNR：**N09M07F10、−4 dB**（也是 `train_local.py` 默认），ClassBD（分类器为 WDCNN）**F1 92.08%**。
- 同一格在预印本 Table 12（ClassBD 接不同骨干网络）中为：MobileNetV3 84.49、ResNet18 84.99、WDCNN 79.06、Transformer 82.34。**Table 4 与 Table 12 对 ClassBD-WDCNN 给出的数值不一致（92.08 对 79.06）**，记录在案；S6 对 ClassBD-WDCNN 以 Table 4 的 92.08 为主，Table 12 的数值作为各骨干的对照。

## 方法组（4 个）

`train_local.py` 的 `chosen_model` 提供四个带 ClassBD 模块的骨干，正好 4 组：

| 组 | `chosen_model` | 预印本 Table 12（−4 dB）|
|---|---|---|
| `ClassBD-WDCNN\|PU N09M07F10 -4dB` | `bdcnn`（默认） | 79.06（Table 4：92.08） |
| `ClassBD-ResNet18\|PU N09M07F10 -4dB` | `bdresnet` | 84.99 |
| `ClassBD-MobileNetV3\|PU N09M07F10 -4dB` | `bdmobile` | 84.49 |
| `ClassBD-Transformer\|PU N09M07F10 -4dB` | `bdtransformer` | 82.34 |

不带 ClassBD 的原始骨干不在仓库中，不跑。

## 配置（作者代码默认）

- `train_local.py` 的 `__main__` 配置：200 轮，batch 128，SGD（momentum 0.9）lr 0.5，CosineAnnealingLR，Gaussian 噪声，SNR −4 dB，PU `N09_M07_F10`，14 类。论文称学习率从 {0.1, 0.3, 0.5, 0.8} 中选，未给出各骨干的取值，四组都用代码默认 0.5。
- 数据：`Paderborn_Processing`：每个轴承 20 个 .mat 文件取 `vibration_1`；第 1–19 个文件各取 20 个长 2048、步长 2048 的窗，随机 0.8:0.2 分为训练（5776）/验证（1444）；第 20 个文件全部窗作测试（380）；按子序列加噪后用训练集均值方差标准化。
- 运行方式：`WANDB_MODE=disabled`（`train_local.py` 只导入 wandb，不调用）。

## 数据来源（偏离，jeff 裁定）

- README 要求 `data/Paderborn/N09_M07_F10/…` 结构，或用作者 Google Drive 上重新整理的 PU 数据。不用作者的整理版，改由 Paderborn 官方原始 .mat 文件按 README 结构用软链接摆放（`~/audit_ext/work21/ClassBD/data/Paderborn/N09_M07_F10`，共 19 个轴承 × 20 = 380 个文件，与 `PUProcessing.py` 的 `HBdata`/`RDBdata` 一致；`KI04` 在类别映射中出现但不在列表中，作者也未使用）。
- 其中 16 个轴承来自本机 `~/data_pu`（只读引用，不改动）；K006、KA30、KI16 在 `~/data_pu/_rars` 中的压缩包是截断的（unrar t 校验失败），由 jeff 从 Paderborn 官网手动下载（`C:\Users\ThinkPad\Downloads`），`unrar t` 校验通过后解压到 `~/audit_ext/data/pu21raw`。

## 种子方案

作者有种子机制（`config['seed']`，默认 42，`random_seed` 设 torch/numpy/random 与 cudnn deterministic）。用种子 **0–9**。训练/验证划分与噪声也由该种子决定（作者设计）。

## 作者的选模规则

- 每轮在验证集上算准确率，`acc > max_acc` 时保存 checkpoint；训练结束后加载该 checkpoint，在测试集上算宏平均 F1 并打印（`inference`）。**`author` 协议 = 验证集准确率首次达到最大值的轮（严格 `>`）**。**选模集**：验证集；**报告集**：测试集（第 20 个文件）。
- 作者报告值是 F1，不是准确率。补丁在 `train()` 每轮验证后调用钩子，记录作者算出的验证准确率与验证损失（作者的损失为各批 UW 总损失之和），并在 `isolate` 内补算测试集准确率、交叉熵与宏平均 F1（准确率与交叉熵进 `"test"` 记录，F1 另存）；按作者同一规则标 `author_pick`；作者 `inference` 返回的 F1 与记录中作者轮次的 F1 另行核对（`author_check_f1`）。测试集取自 `main()` 构建的同一个 DataLoader（驱动包装模块内的 `DataLoader` 名称捕获，不改作者代码）。

## 泄漏编码（只记录，不修复）

- 训练/验证窗来自同一批文件随机划分；测试来自独立的第 20 个文件。
- 每个 batch 都重新创建优化器与调度器（动量每步清零，余弦调度不生效）；`UW` 每次调用新建不参与优化的权重参数；验证阶段的 `torch.no_grad()` 不起作用；验证 DataLoader `shuffle=True, drop_last=True`（每轮只评估 1408/1444 个验证样本）。只记录。
