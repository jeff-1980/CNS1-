# 对象 77 登记：iridescentcloud/machine_fault_diagnosis

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d 及 jeff 2026-10-24 的 6 小时规则；按 jeff 2026-10-25 裁定。登记日期 2026-10-25。冒烟（2 轮）在登记前做过一次以确认可运行（jeff 裁定前提），不计入结果。

## 论文与代码

- 论文：Zhang, Hao, Lv, Fan, *The combination model of CNN and GCN for machine fault diagnosis*，PLOS ONE 2023，DOI 10.1371/journal.pone.0292381（开放获取；数据与代码声明指向本仓库）。
- 仓库：`iridescentcloud/machine_fault_diagnosis`，commit `b7130fd5271f301a7e3639aa506900b8aa04decb`（2026-10-25 重新拉取，首次拉取时 TLS 中断）。README 为空；入口 `GACNN/train.py`。

## 头条表格（论文，5 次 60/40 随机划分均值）

- GACNN：CWRU 99.44 %、MFPT 97.11 %、**SEU 92.80 %**（本审计跑 SEU）。

## 方法组（1 个，jeff 裁定用默认 SEU）

- `GACNN_2|SEU bearingset ch0 10-class`（默认模型 `GACNN_2`）。论文对比的 CNN、ChebyNet、MRF-GCN 在 `model/` 中，但不在本次范围。

## 配置（`train.py` 默认）

- 数据：仓库自带 SEU（东南大学）`gearbox/bearingset` 10 个 csv（ball / comb / health / inner / outer × 20_0 / 30_2），通道 0；每类前 50 × 1024 点不重叠切段，Max-Min 归一化，FFT（FD 输入）；每 10 个样本组成一个完全图（Gaussian kernel 边权），共 50 张图；`train_test_split(train_size=0.6, shuffle=True)` 无 `random_state`。
- 训练：GACNN_2，100 轮，batch 32，Adam 1e−3，NLL；无验证集。

## 种子方案

- 无种子 → **10 次不设种子的重复**，`seed` = 重复序号 0–9。

## 作者的选模规则

- 只在最后一轮之后测一次测试集（节点准确率），**作者口径 = 最后一轮**。审计在每轮之后用作者的测试计算在 `isolate` 中补算测试准确率与 NLL。**选模集**：无；**报告集**：测试图（20 张，200 个节点）。jeff 裁定：纳入敏感性分析，算不出的口径记 n/a。

## 改动（逐条写入偏离记录）

- jeff 裁定：删除 `utils/train_utils.py` 中导入缺失源码模块（`model/GACNN.py`、`model/GACNN2.py`、`dataset/xjtu_data.py`，仓库只有 .pyc）的语句（第 6 行去掉 `xjtu_data`，删第 11、12 行）；默认流程不使用这些模块。
- 平台适配：`dataset/MFPT_data.py` 第 84 行有模块级调试调用（导入时即执行，读作者本机 `D:\Users\hcy\...` 路径），改为仓库自带的 `MFPT Fault Data Sets`（该调用 `random_state=1`，不影响全局随机数）。
- 钩子 3 行（模块级 `AUDIT_HOOK` / `AUDIT_FINAL`）。环境复制对象 76 的环境，补装 matplotlib。

## 泄漏编码（只记录，不修复）

- 无验证集；按图（10 个连续切段）随机划分，同一记录的相邻切段分属训练与测试图。
