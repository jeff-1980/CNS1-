# 对象 53 登记：ChenQian0618/TFN

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-23。

## 论文与代码

- 论文：Q. Chen, X. Dong, G. Tu, D. Wang, C. Cheng, B. Zhao, Z. Peng，*TFN: An interpretable neural network with time-frequency transform embedded for intelligent fault diagnosis*，Mechanical Systems and Signal Processing 207 (2024) 110952，DOI 10.1016/j.ymssp.2023.110952；预印本 arXiv 2209.01992（期刊版不开放，以下取自预印本）。
- 仓库：`ChenQian0618/TFN`，commit `bb8af32c380fa047631b960e6a08ed0b261f1874`（2025-03-28），作者即第一作者。本地 `~/audit_ext/main_objects/53_ChenQian0618__TFN`。入口 `main.py`（`PostProcess/TrainSequentially.py` 逐模型调用 `main.py`）。

## 头条表格

- CWRU 诊断实验的结果只以 **Fig. 8**（各模型、各预处理层通道数的测试准确率柱状图，10 次重复）给出，没有数值表格；**S6 对本对象记为“无法对照（只有图）”**。

## 方法组（4 个）

- 论文 CWRU 实验含三类模型：骨干（Backbone-CNN、Random-CNN）、对比方法（SincNet、WKN、W-CNN，不在仓库中）、TFN（STTF、Chirplet、Morlet 三种核）。仓库提供 `Backbone_CNN`、`Random_CNN`、`TFN_STTF`、`TFN_Chirplet`、`TFN_Morlet`。按预注册上限 4 组，取论文正文强调的三个 TFN 与作为比较基准的骨干：`TFN-STTF`、`TFN-Chirplet`、`TFN-Morlet`、`Backbone-CNN`（组名 `<模型>|CWRU 48k DE load3`）。Random-CNN 不跑。

## 配置（作者代码默认）

- 数据：`Datasets/CWRU.py` + `get_files/CWRU_get_files.py` 默认 `load_condition=3`：正常 `Normal Baseline Data/100.mat`，故障 `48k Drive End Bearing Fault Data/` 的 112、125、138、177、192、204、217、229、241（1730 rpm），读每个文件中**第一个**含 `_DE_time` 的变量键；长 1024 不重叠切段，每类取前 450 段（`balance_label`），共 4500；`train_test_split(test_size=0.3, random_state=40, stratify)` 分训练 3150 / “val” 1350（mean-std 标准化）。
- **`217.mat` 中同时有 `X215_DE_time` 与 `X217_DE_time`，作者代码取到的是第一个键 `X215_DE_time`**（与 99.mat 问题同类），只记录，不修复。
- 训练：`main.py` 默认：Adam lr 1e−3，StepLR(1, 0.99)，batch 64，50 轮，预处理层通道数 `mid_channel = 32`，kernel 11，`clamp_flag=True`，无噪声（SNR 1000 → None）。
- 代码与论文描述不一致处（只记录，按 461bc21 规则 (1) 以代码为准）：论文写 60% 训练 / 40% 测试，代码默认 `test_size = 0.3`；论文对 TFN 报告 16/32/64/128 四种通道数，代码默认 32。

## 数据来源

- CWRU 官方网站（engineering.case.edu/sites/default/files/<编号>.mat）下载上述 10 个文件，逐个与服务器 Content-Length 核对完整，存放于 `~/audit_ext/data/cwru53/`（按 README 的目录名组织）。作者代码会在该目录写入缓存 `data_buffer.pkl`。

## 种子方案

- 作者在 `main.py`、`utils/train_utils.py`、`Models/TFconvlayer.py`、`Datasets/*` 的模块顶层固定 `seed = 999`（torch / numpy / random）。按预注册改为 **种子 0–9**：驱动先导入作者全部模块（作者的顶层设种照常执行），再用审计种子重设 torch、numpy、random，然后以 `runpy` 运行 `main.py`。数据划分 `random_state=40` 为作者固定值，不改。

## 作者的选模规则

- 每轮在 “val” 集（即 30% 测试集）上评估；作者的结果统计流程（`PostProcess/Acc_statistic.py` → `PlotAccuracy.py`）绘制的是 **各次运行 val 准确率的最大值（`mean max acc`）**，Fig. 8 即此图。**作者口径 = 在报告集上取最好（author = best_rep，平局取较早轮，`epoch_acc > best_acc`）**。另有 “final acc”（最后 5 轮均值）只写入表格，不作作者口径。
- **选模集**：无。**报告集**：“val”（30% 测试集）。
- 补丁：`utils/train_utils.py` 加模块级 `AUDIT_HOOK = None`，在每个阶段 `Records[...acc]` 追加之后调用，记录作者算出的准确率与损失；`finish` 接收 val 准确率的最大值。

## 泄漏编码（只记录，不修复）

- 没有独立测试集：每轮在报告集上评估并取最好。
- 同一信号切段后随机划分训练 / 测试（段不重叠，但来自同一条记录）。
