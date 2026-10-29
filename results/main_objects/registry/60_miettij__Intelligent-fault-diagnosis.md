# 对象 60 登记：miettij/Intelligent-fault-diagnosis

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-24。按 jeff 2026-10-23 裁定：最大轮数取论文 Table A1。

## 论文与代码

- 论文：*Whitening CNN-Based Rotor System Fault Diagnosis Model Features*，Applied Sciences 2022, 12, 4411，DOI 10.3390/app12094411（开放获取，全文取自 Aalto 机构库）。
- 仓库：`miettij/Intelligent-fault-diagnosis`，commit `4de8525fa62e2474c0aef938440d1bfe33617d7c`（2022-04-28），README 称为论文源码。本地 `~/audit_ext/main_objects/60_miettij__Intelligent-fault-diagnosis`。入口 `code/main.py`（`run_*.sh`）。

## 头条表格

- CWRU 结果只以图给出（Fig. 7 驱动端、Fig. 8 驱动端 + 风扇端，各 10 次试验均值），无数值表；图中的数值标签（PDF 文本层）在 DE + FE（与 `run_*.sh` 的 `--DE-FE True` 一致）、**1 → 2 hp** 任务上为：WDCNN 100.0、SRDCNN 99.01、WDCNN (ND) 99.83、SRDCNN (ND) 100.0（Ince 99.95、Ince (ND) 100.0 不跑）。S6 以此对照。

## 方法组（4 个）

- `WDCNN`、`WDCNN-ND`（network deconvolution）、`SRDCNN`、`SRDCNN-ND`，组名 `<模型>|CWRU DE+FE 1->2hp`。Ince 两组不跑（预注册上限 4 组；论文强调 ND 对 WDCNN、SRDCNN 的提升）。

## 配置

- 数据：仓库自带 `original.tmp/12k/drive_end2.zip`（README 步骤解压），10 类（正常、滚动体 / 内圈 / 外圈 × 0.007 / 0.014 / 0.021），每类 1、2、3 hp 各一个 `.mat`；DE 与 FE 两通道；训练：每类 1980 个重叠的 2048 点窗口；测试：另一负载每类 59 个不重叠窗口。读变量键：正常 2 hp 文件写死为 `X099_DE_time` / `X099_FE_time`，其余取最后一个含 `DE` / `FE` 的键（99.mat 问题只记录不修复）。
- 训练参数取 `run_*.sh`（与论文 Table A1 的 batch / lr 一致，SRDCNN-ND 用 DE + FE 的 batch 128、lr 0.1）；**最大轮数按 Table A1**：WDCNN 70、SRDCNN 60、WDCNN (ND) 70、SRDCNN (ND) 40（脚本默认 `--epochs 2`，偏离，jeff 裁定）。Adam；学习率每 5 轮减半（代码，论文写每 10 轮）；ND 的 `--firstl-iter 25`（脚本，Table A1 为 15）——以代码为准，只记录。
- 只跑 `main.py` 三源负载循环中**源负载 1 hp** 这一段（驱动按 `main.py` 的 cwru 分支逐步调用作者的数据集、模型、`cwru_train` 与 `cwru_test`），在 2 hp（报告）与 3 hp（另记）上测试。

## 种子方案

- 代码不设种子，按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。

## 作者的选模规则

- `cwru_train`：每轮在轮换的验证折（训练窗口的 1/5，每轮换一折）上算验证损失；**验证损失严格下降时保存权重，否则把权重回滚到最好的权重并继续**；连续 8 次未改善或达到最大轮数时停止。最终模型 = 验证损失最小的权重。**作者口径 = 验证损失最小轮（严格 <，平局取较早轮）**。**选模集**：源域验证折；**报告集**：目标负载 2 hp 测试集。
- 补丁：`optalgos/cwru_train.py` 加模块级 `AUDIT_HOOK`，在每轮算出验证损失后、作者回滚权重之前调用（2 行）：记录作者的验证损失，并在 `isolate` 中补算验证准确率、2 hp 测试准确率与交叉熵、3 hp 测试准确率；`finish` 接收作者 `cwru_test` 写入 `teststats.txt` 的 `Accuracy 1 --> 2`。

## 依赖适配（一行修复，偏离记录）

- `models/wdcnn_deconv.py`、`models/srdcnn_deconv.py` 的 `FastDeconv` 在 GPU 分支调用 `_ConvNd.__init__` 时缺 `padding_mode`（当前 PyTorch 必需），按 CPU 分支补 `padding_mode='zeros'`（各 1 行）。
- `main.py` 在有 GPU 时读不存在的 `../original/12k/drive_end2/`；驱动直接使用 README 解压出的 `../original.tmp/12k/drive_end2/`（`main.py` 无 GPU 时的路径），不改作者文件。

## 泄漏编码（只记录，不修复）

- 验证折每轮轮换，其余轮次的验证窗口都参与训练；折边界与相邻训练窗口重叠。
- 训练与测试为不同负载的记录（跨域），测试不参与选模。
