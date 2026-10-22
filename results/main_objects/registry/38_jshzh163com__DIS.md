# 对象 38 登记：jshzh163com/DIS

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-22。

## 论文与代码

- 论文：*Domain interference suppression for reliable fault diagnosis under unseen operating conditions*，Mechanical Systems and Signal Processing (2026)，DOI 10.1016/j.ymssp.2026.114457（OpenAlex：closed）。
- 仓库：`jshzh163com/DIS`，commit `3d817309465a610ce5e3fac8719bb411fcde5526`（2026-10-16），README 称 “This repository provides PyTorch code for DIS training on the SQV bearing dataset”。本地 `~/audit_ext/main_objects/38_jshzh163com__DIS`。入口 `train.py`。

## 头条表格

- 论文全文不可得，README 无结果表，**头条表格的数值无法登记**；S6 对本对象记为“无法对照”。

## 方法组（1 个）

- 仓库只实现 DIS（`alg/algs/DIS.py`）。只有 **1 个方法组**：`DIS|SQV test_env 0`。H-B2 对本对象无组对可算。

## 配置（作者代码默认）

- 数据：README 默认 `python train.py --dataset SQV`。SQV 公开数据（作者 README 给出的 Google Drive，`SQV-public.rar`，原始为 txt）。作者先用 MATLAB 脚本 `matlab/SQV_text_to_mat.m` + `get_speed.m` 把 txt 转为 mat，再用 `scripts/prepare_sqv.py`（默认参数：域 1–6，段长 3200、重叠 512，每个文件首尾各取 50 段）生成 `SQV_x.npy` / `SQV_y.npy`；7 类（IF_1–3、OF_1–3、N），6 个域。
- **MATLAB 步骤按 d154119 改写为 Python**（`audit_sqv_text_to_mat.py`）：只做读取 txt 第 17 行起的第 2 列、转速通道取负、找转速信号自上而下穿过阈值 5 的位置、截取首尾两个穿越点之间的振动信号并依序存为 `<条件>_<k>.mat`。确定性步骤，**不涉及重采样**，因此不做两种滤波器的敏感性对照。公开压缩包中正常状态的文件夹名为 `NC`，作者脚本与 `prepare_sqv.py` 用 `N`，按作者命名输出（偏离记录）。
- 训练：`train.py` 默认：留出域 `test_envs=[0]`（按 461bc21 规则 (2) 取代码默认），其余域按 `split_style` 默认随机 8:2 分训练 / 验证；先训练教师网络，再训练 120 轮 × 每轮 2 步（batch 32，SGD lr 1e−2、momentum 0.9、weight decay 5e−4，第 84、108 轮学习率 × 0.1）；第 0、5、10、…、115 轮及第 119 轮评估（共 25 个评估点）。

## 种子方案

- 作者有种子机制（`--seed`，默认 0，`set_random_seed` 设 random / numpy / torch 并开 cudnn 确定性）。按预注册用 **种子 0–9**。

## 作者的选模规则

- 每个评估点在训练域、验证集（源域 20%）、目标域上算准确率；`valid > best_valid` 时更新 `best_valid` 并记下当时的目标域准确率；**另外**，`valid >= best_valid 且 target >= target_acc` 时也更新 `target_acc`（验证集平局时取目标域更高者）。结束时报告 `DG result: target_acc`。**作者口径 = 验证集最好的评估点，平局时按目标域取高**（含对报告集的平局择优，只记录）。
- **选模集**：验证集（源域 20%）。**报告集**：目标域（留出域 0）。
- 补丁：审计的“轮次”= 评估点 1–25。驱动在作者每个评估点之后，记录作者算出的验证与目标域准确率，并在 `isolate` 内补算两者的交叉熵；按作者上述规则（含平局规则）标 `author_pick`；`finish(DG result)`。

## 数据来源

- `~/audit_ext/data/sqv38/SQV-public.rar`（作者 README 链接），解压后经上述 Python 转换与作者 `prepare_sqv.py` 生成 `~/audit_ext/work38/data/SQV/`。

## 泄漏编码（只记录，不修复）

- 作者选模规则在验证集平局时按报告集（目标域）择优。
- 每个记录内切段重叠 512 点，验证集与训练集来自同一记录。
