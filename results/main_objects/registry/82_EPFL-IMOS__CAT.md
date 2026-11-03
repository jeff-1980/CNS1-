# 对象 82 登记：EPFL-IMOS/CAT

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d 及 jeff 2026-10-24 的 6 小时规则。本登记在任何训练之前提交。登记日期 2026-10-25。

## 论文与代码

- 论文：Forest & Fink, *Calibrated Adaptive Teacher for Domain-Adaptive Intelligent Fault Diagnosis*，Sensors 2024, 24(23), 7539，DOI 10.3390/s24237539（MDPI 页面返回 403；数值取自 arXiv 2312.02826v2，与期刊版同日发布，假定一致）。
- 仓库：`EPFL-IMOS/CAT`，commit `354cf94aef4f0afd1f81d7ef645bd40ffc68d8d5`，README 署名两位作者并引用该论文。入口 `train_advanced.py`（单模型 DA 基线）与 `train_advanced_ts.py`（教师-学生，README 命令）。

## 头条表格（arXiv v2 Table 5，PU 时域，任务 0 → 1，teacher 目标准确率，5 个种子均值）

| 组 | 论文 |
|---|---|
| DANN* | 36.53 |
| AT* | 49.36 |
| CAT*-TempScaling | 52.67 |
| CAT*-CPCS | 52.42 |

- 12 个任务的平均（时域）：DANN* 47.02、AT* 53.13、CAT*-TempScaling 54.50、CAT*-CPCS 54.42。Source-only 与 DANN 行抄自 UDTL，不是作者运行，不作为方法组。

## 方法组（4 个，论文表中作者自己运行的方法）

- `DANN*|PU TD 13-class N15M07F10->N09M07F10`：`train_advanced.py --domain_adversarial True --adversarial_loss DA --mcc_loss --sdat`（DANN + MCC 损失 + SDAT）。
- `AT*|…`：`train_advanced_ts.py` + `--self_training True --self_training_criterion confidence --adaptive_confidence_threshold`。
- `CAT*-TempScaling|…`：AT* + `--calibration TempScaling`（README 命令）。
- `CAT*-CPCS|…`：AT* + `--calibration CPCS`。
- 共同：`--data_name PU --transfer_task [[0],[1]]`（README 任务），其余参数取代码默认：300 轮，batch 64，Adam 1e−3、wd 1e−5，MultiStep(150, 250)，`middle_epoch` 50、`self_training_epoch` 50、`calibration_epoch` 150，EMA α 0.999。

## 数据

- PU（帕德博恩）13 个真实损伤轴承（KA04、KA15、KA16、KA22、KA30、KB23、KB24、KB27、KI14、KI16、KI17、KI18、KI21），每个轴承每个工况只读文件 `_1`，通道 `fl[0][0][2][0][6][2]`（UDTL 惯例下为 vibration_1；**论文正文写用电机电流信号，与代码不一致，只记录**），不重叠 1024 点切段、mean-std 归一化。源 N15_M07_F10（1500 rpm），目标 N09_M07_F10（900 rpm）。
- 来源：本机 `~/data_pu` 与对象 21 时下载的 PU 文件（只读软链接到 `~/audit_ext/work82/PU/`）；KA30、KI16 在 N15 工况的文件从 Zenodo 镜像（10.5281/zenodo.15845309）下载到 `D:\audit_data\pu_zenodo`，md5 与 Zenodo 一致、`unrar t` 通过，N09 文件与 jeff 手动下载的副本逐字节相同；只解压用到的 8 个文件到 WSL。

## 种子方案

- 作者 `--seed`（默认 42，设 torch / numpy / random 与 cudnn deterministic；源域、目标域 80/20 划分也随之变化）→ **种子 0–9**。

## 作者的选模规则

- 每轮在 `source_val` 与 `target_val`（目标域 20%，即测试集）上评估；论文明确**报告最后一轮的 teacher 准确率**（教师-学生组）或最后一轮模型准确率（DANN*），“不用测试标签早停”。仅在 `--save_weights` 时另存目标测试集上最好的模型（默认不开启）。**作者口径 = 最后一轮**。
- **选模集**：无。**报告集**：`target_val`（教师-学生组记 teacher，学生另记为 `test_student`）。作者报告值由 driver 独立读取作者 TensorBoard 日志中最后一个 `accuracy/target_val[_teacher]` 标量。
- 温度标定只用源域验证集标签（作者代码）。

## 补丁

- 两个训练循环各加 2 行：模块级 `AUDIT_HOOK`，在每个验证阶段的日志之后调用一次（记作者算出的准确率与损失）。无依赖适配（冒烟中发现再记）。
- 环境：复制对象 76 的环境（torch 2.1.2），补装 tensorboard、torchvision 0.16.2。

## 泄漏编码（只记录，不修复）

- 目标域无独立验证集；作者口径为最后一轮，不用测试集选模（但 `--save_weights` 的 best 模型按测试集选，默认关闭）。
