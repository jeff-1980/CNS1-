# 对象 42 登记：caie8201/Multi-Scale-CNN-Transformer-Hybrid-Network

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-22。任务与方法组按 jeff 2026-10-22 裁定。

## 论文与代码

- 论文：Journal of Vibroengineering 2026，DOI 10.21595/jve.2026.26312（Extrica 开放获取，https://www.extrica.com/article/26312）。
- 仓库：`caie8201/Multi-Scale-CNN-Transformer-Hybrid-Network`，commit `d4fc44057bc0c62511ef2af5f25fa632f94634df`。本地 `~/audit_ext/main_objects/42_caie8201__Multi-Scale-CNN-Transformer-Hybrid-Network`。单文件 `main.py`（消融脚本）。

## 头条表格

- 论文 Table 6（PU 数据集诊断结果，%，每个任务独立运行 5 次，均值±标准差），任务 **T3**（源域 C0、C1、C2 → 目标域 C3）：Ours-A 98.46±0.19，Ours-B 97.66±1.21，Ours-C 98.45±0.16，**Ours-D（完整提出方法）98.66±0.42**。S6 对照 Ours-D 的 98.66。

## 任务选择（偏离，jeff 裁定）

- 作者代码默认 `--config_id 3`（HUST1data9.mat），该数据不在作者公开的 Drive 中，论文也不使用 HUST。改用论文 Table 6 的 PU T3，对应代码配置 1（`PU3200data14.mat`，`speeds [[0,1,2,3]]` → 源域 load0–2、目标域 load3，14 类，每域 1120 个样本，长度 3200）。数据取自作者 Drive 上的 `PU3200data14.mat`。

## 方法组（4 个，按论文第 4.2 节逐一对应代码开关）

| 组 | 论文定义 | `run_ablation_variant` 开关 |
|---|---|---|
| `Ours-D\|PU T3` | 完整模型：Transformer 编码器 + PAM + LLU | transformer=True, pam=True, llu=True |
| `Ours-C\|PU T3` | Ours-B + PAM | transformer=True, pam=True, llu=False |
| `Ours-B\|PU T3` | Ours-A + 标准 Transformer 编码器 | transformer=True, pam=False, llu=False（`nn.TransformerEncoderLayer`） |
| `Ours-A\|PU T3` | 只保留多尺度 CNN + 全局平均池化 | transformer=False, pam=False, llu=False（`compressed.mean(dim=2)`） |

- 开关组合与论文定义一一对上，故按裁定跑四组。
- 记录在案：仓库 `main()` 中 `ablation_configs` 唯一有效的一行名为 “+ Transformer + PAM + LLU”，开关却全为 False（即 Ours-A）；完整模型那一行被注释掉。驱动直接按上表调用作者的 `run_ablation_variant`，不经 `main()`，因此不需要再改这一行；`main()` 未改动。此前“恢复被注释的提出方法行”的安排由本裁定的四组方案取代，Ours-D 即该行的开关组合。

## 配置（作者代码默认，配置 1）

- 90 轮，batch 64，Adam lr 3e−3、weight decay 1e−4，CosineAnnealingWarmRestarts（T_0=30，T_mult=2，eta_min 1e−4），patience 90（不会提前停止），embed 72，depth 3，heads 4，token 64，dropout 0.1，逐样本标准化。

## 种子方案

- 作者每次重复用种子 42+rep（配置 1 的 `rt=1`，论文为 5 次）。按预注册改为 **种子 0–9**，经 `run_ablation_variant(seed=…)` 传入（作者在函数内设 torch、numpy、CUDA 种子和 cudnn deterministic）。

## 作者的选模规则

- 每轮在目标域（load3）测试集上评估，`test_acc > best_acc` 时更新；返回并报告 `best_acc`。**作者口径 = 在报告集上取最好（author = best_rep，平局取较早轮）**。
- **选模集**：无。**报告集**：目标域测试集。
- 补丁：`main.py` 加模块级 `AUDIT_HOOK = None`，在作者 `val_acc_history.append(test_acc)` 之后调用钩子；钩子记录作者算出的测试准确率，并在 `isolate` 中另算测试交叉熵；按作者 `>` 规则标 `author_pick`；`finish` 接收作者返回的 `best_acc`。

## 数据来源

- 作者 Google Drive 公开文件 `PU3200data14.mat`（39,246,529 字节），存放于 `~/audit_ext/data/pu42/`，运行目录 `~/audit_ext/work42` 内软链接。

## 泄漏编码（只记录，不修复）

- 每轮在报告集（目标域）上评估并取最好。
