# 对象 37 登记：JC952/P2PCHF

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-22。

## 论文与代码

- 论文：H. Ma, J. Wei, G. Zhang, Q. Wang, X. Kong & J. Du，*Heterogeneous Federated Learning: Client-Side Collaborative Update Interdomain Generalization Method for Intelligent Fault Diagnosis*，IEEE Internet of Things Journal 12(5) (2025) 5704–5718，DOI 10.1109/JIOT.2024.3489617（全文不可得）。
- 仓库：`JC952/P2PCHF`，commit `c5c31f09e7fb247c36fcdd2ec9cb21699b454dd6`（2025-06-23），README 称 “the source code for” 该论文。**待核实项“框架”**：核实为 PyTorch，成立。入口 `main.py`。

## 头条表格

- 论文全文不可得，README 无结果表，**头条表格的数值无法登记**；S6 对本对象记为“无法对照”。

## 方法组（1 个）

- 仓库只有 P2PCHF（`--model p2pchf`，客户端为 5 个异构网络 `Client_net1–5`）。只有 **1 个方法组**：`P2PCHF|HUST bearing + CWRU9`。

## 配置（jeff 2026-10-22 裁定：用作者设置 3）

- 代码有 3 套设置：设置 1（`parse_args` 默认）为 SDUST 齿轮箱私有数据 + CWRU7 公共数据，仓库中没有 SDUST 数据，且不是轴承；设置 2 为 BJUT 齿轮箱 + CWRU5（作者 `__main__` 实际循环的设置）；设置 3 为 **HUST 轴承（私有，5 个转速域 60/65/70/75/80，9 类，每域 50% 测试）+ CWRU9（公共无标签数据）**。按 jeff 裁定取**设置 3**，写入偏离记录。
- 超参数取作者 `utils/best_args.py` 中 `bearing-hust` 的值：30 个通信轮，本地 lr 1e−4、公共 lr 1e−4，本地 15 轮、公共 2 轮，batch 64，temp 0.2（`parse_args` 中又被设为 0.5，照作者代码），fisl 1.5，fccm 按作者 `__main__` 传入 0.005；`--model_setting 4`、`--structure heterogeneity`、`--setting domain_skew` 等为默认。`add_management_args` 把 `--lr` 设为必填，但代码不使用 `args.lr`；按作者参数传 `--lr 0.0001`（无实际作用）。

## 种子方案

- `--seed` 参数存在但从未调用 `set_random_seed`，训练不设种子（作者 `__main__` 自身做 5 次重复）。按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。

## 作者的选模规则

- 每个通信轮先做公共数据协同更新，然后评估（各客户端在本域测试集上的 intra 准确率、在其他域测试集上的 inter 准确率），再做本地更新。结果写入 `M3.csv`：取**平均 inter 准确率最高的通信轮**（第一个最大值），记录该轮各客户端的 inter 准确率。**作者口径 = 在报告集上取最好（author ≈ best_rep）**。
- **选模集**：无。**报告集**：各客户端在其他域测试集上的准确率（跨域泛化，各客户端平均）。
- 审计的“轮次”= 通信轮 1–30。补丁：`utils/training.py` 中评估后调用钩子，记录作者算出的平均 inter 准确率（`report_split="test"`），并在 `isolate` 内补算对应的交叉熵（各客户端在其他域测试集上的平均）；按作者规则标 `author_pick`；`finish(max(mean_inter)/100)`。

## 数据来源

- 仓库自带 `datasets/HUST-Bearing/*/BearingData_*_9_Norm.mat` 与 `datasets/CWRU9/*/BearingData_*_9_Norm.mat`（作者处理后的数据）。

## 泄漏编码（只记录，不修复）

- 每个通信轮在报告集上评估并取最好轮。
