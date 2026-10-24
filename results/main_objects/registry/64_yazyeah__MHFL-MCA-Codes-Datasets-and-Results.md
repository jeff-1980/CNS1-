# 对象 64 登记：yazyeah/MHFL-MCA-Codes-Datasets-and-Results

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-24。任务与组按 jeff 2026-10-23、2026-10-24 裁定。

## 论文与代码

- 论文：Advanced Engineering Informatics 2026，DOI 10.1016/j.aei.2026.105277（MHFL-MCA；期刊页不开放，数值取自仓库附带的稿件 `.tex` 与结果 CSV）。
- 仓库：`yazyeah/MHFL-MCA-Codes-Datasets-and-Results`，commit `9848887fd845aafa97d22ccc47304f67bf38972b`。本地 `~/audit_ext/main_objects/64_yazyeah__MHFL-MCA-Codes-Datasets-and-Results`。入口 `Codes/Codes for KAIST/<方法>_KAIST.py`（TensorFlow 2.7 / Keras）。

## 头条表格（KAIST，N = 5，0 Nm → 2 Nm / 4 Nm，10 次去最高最低后的均值 ± 标准差）

| 组 | 论文（稿件 Table 9 / 10） | 仓库结果 CSV（Final_Summary_Stats_trimmed） |
|---|---|---|
| MHFL-MCA | 97.39 ± 1.95 / 91.56 ± 8.32 | 97.39 ± 1.84 / 91.56 ± 3.93 |
| MSF-DFormer | 74.55 / 78.62 | 77.88 / 70.90 |
| CFFN | — | 46.44 / 44.04 |

- 论文与仓库 CSV 在 MSF-DFormer 及 MHFL-MCA 的标准差上不一致，只记录。S6 以论文值为主，CSV 为辅。

## 任务与方法组（jeff 裁定）

- 只跑 **KAIST、N = 5**（每类 5 个训练样本）任务；UO 任务因在测试集上调参不跑。报告集 = 2 Nm 测试集（另记 4 Nm）。
- 3 组：`MHFL-MCA|KAIST 0->2Nm N=5`（`MHCNN_KAIST.py`，保留作者 Optuna 流程）、`MSF-DFormer|…`（`MSF-DFormer_KAIST.py`，固定超参数）、`CFFN|…`（`CFFN_KAIST.py`；先训练双分支 EDCN、再离线 CCA、最后训练 softmax 分类器，**逐轮记录只针对最后的分类器阶段**，每轮用作者的特征提取器 + CCA + 当前分类器在测试集上评估）。
- KDCNN-DF 不跑：其报告准确率来自深度森林（RandomForest + ExtraTrees 级联），最终分类器不是逐轮训练的深度模型（比照对象 50）。

## 配置（作者脚本默认）

- 数据：仓库 Git LFS 中的 KAIST 振动 `.mat` 与电流 `.tdms`（30 个文件），本机无 git-lfs，由 GitHub LFS 媒体地址逐个下载并按指针中的 sha256 与大小校验（30 / 30 通过），存放于 `~/audit_ext/data/kaist64`。5 类（正常、内圈 ×2、外圈 ×2），段长 2048，每类每负载最多 400 段；训练 / 验证取自 0 Nm（每类 5 个训练、50 个验证，哈希检查不重叠），测试为 2 Nm 与 4 Nm 全部段；训练与测试均加 0 dB 噪声。
- MHFL-MCA：Stage 1 Optuna（30 次试验、每次 30 轮，0 Nm 调参划分，目标为验证准确率），Stage 2 Adamax（最优 lr / batch）、80 轮、一致性正则。MSF-DFormer：Adamax 固定 lr、70 轮。CFFN：EDCN 与分类器各 80 轮，Adamax lr 8.47e−4。
- 对作者脚本的源码级改动（执行的代码另存为 `audit_<脚本名>` 随补丁提交）：路径占位符（README 要求用户替换）→ 本机路径；`SAMPLE_RANGE` → 只 N = 5；种子；不跑 Stage 3（噪声鲁棒性研究）；各 1 行挂审计回调与交出作者报告值。

## 种子方案

- 作者每次运行 `set_global_seed(n_train*100 + run_idx)`。按预注册改为 **种子 0–9**（`seed = run_idx − 1`），该种子同时决定训练 / 验证抽样。Optuna 采样器未设种子（作者设计），每组运行一次。

## 作者的选模规则

- 不保存 checkpoint，**报告最后一轮**（author = last）；2 Nm / 4 Nm 测试准确率由作者 `eval_on_test`（CFFN 为 `eval_on_test_cffn`）在固定噪声（`RandomState(seed + 999)`；CFFN 为 `seed + 10` / `seed + 20`）下计算。**选模集**：无（0 Nm 验证集只用于 Optuna 与曲线）。**报告集**：2 Nm 测试集。
- 审计回调每轮用与作者评估完全相同的加噪测试输入计算 2 Nm 准确率与交叉熵、4 Nm 准确率，并记录作者的验证准确率 / 损失。

## 环境

- conda Python 3.9 + cudatoolkit 11.2 + cuDNN 8.1，按仓库 `requirements-lock.txt` 安装（tensorflow 2.7.0、keras 2.7.0、optuna 4.5.0 等）。

## 泄漏编码（只记录，不修复）

- KAIST 协议本身干净（独立验证集、跨负载测试）；测试集每轮被评估（作者报告最后一轮）。
