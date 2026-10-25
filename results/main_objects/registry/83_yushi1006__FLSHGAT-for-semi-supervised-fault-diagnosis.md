# 对象 83 登记：yushi1006/FLSHGAT-for-semi-supervised-fault-diagnosis

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d 及 jeff 2026-10-24 的 6 小时规则；按 jeff 2026-10-25 裁定。登记日期 2026-10-25。冒烟（3 轮）在登记前做过一次，不计入结果。

## 论文与代码

- 论文：Lei Wang, Zaigang Chen, Zhiwen Liu, Jianhai Yan, *Feature–label similarity-aware hypergraph attention network: A semi-supervised fault diagnosis method for machinery under extremely low label rate scenarios*，MSSP 260 (2026) 115007，DOI 10.1016/j.ymssp.2026.115007（作者名单取自 Crossref 与 OpenAlex；全文在付费墙后未取得）。
- 仓库：`yushi1006/FLSHGAT-for-semi-supervised-fault-diagnosis`，commit `b95fd8612de98dcd782e17b8ad7901937ff1f30a`；README 称 official implementation，文件头署名 Lei Wang（Southwest Jiaotong University），即第一作者 → 按 jeff 裁定视为作者代码。入口 `train hypergraph for UoO.py`。

## 头条表格

- 全文不可得，**头条数值无法登记**（S6 记为“无法对照”）。仓库只覆盖论文的 uOttawa 时变转速数据集。

## 方法组（1 个）

- `FLSHGAT|uOttawa time-varying 5-class 1 label/class`（仓库只有提出的方法）。

## 配置（代码默认）

- 数据：Huang & Baddour, *Bearing Vibration Data under Time-varying Rotational Speed Conditions*（Mendeley Data v43hmbwxpm v2），只用 5 个文件 `H-A-1、I-A-1、O-A-1、B-A-1、C-A-1.mat` 的 `Channel_1`；从 Mendeley 公开 API 下载到 `D:\audit_data\uo83`，sha256 与 Mendeley 公布值一致，训练时经软链接直接读取（目录名与 Mendeley 一致）。
- 每类前 150 个 4096 点不重叠窗口（前 100 训练、后 50 测试），最后一类 300 / 200；min-max 后 |FFT| 前 2048 点；训练图与测试图各用 KNN 超图（k = 5）；每类只给 1 个标签（`random.seed(100)` 选取）。
- 训练：200 轮全批量，Adam 1e−3、wd 5e−4，MultiStep(100, 150, 0.1)。

## 种子方案

- 只有选标签样本的 `random.seed(100)`，torch / numpy 不设种子 → **10 次不设种子的重复**。

## 作者的选模规则

- 每轮在测试图上评估；**测试准确率严格创新高时保存“best model”**（最后一轮也强制保存）。作者口径 = 测试准确率第一个严格最大的轮；`AUDIT_FINAL` 取作者最后一次严格创新高时的准确率。论文的报告规则（best 或 last）无法核实。**选模集**：无；**报告集**：测试图节点。测试损失记作者计算的三项之和。

## 改动

- 平台适配 1 行：`datasets/UoO_hypergraph.py` 中类别目录顺序依赖 Windows 的字母序 `os.listdir`，改为 `sorted(os.listdir(root))`（与作者在 Windows 上的顺序相同，之后作者再交换第 0、4 个）。
- 钩子 3 行。环境复制对象 76 的环境（torch 2.1.2、PyG 2.8.1）。

## 泄漏编码（只记录，不修复）

- 测试集选模；测试阶段把上一轮模型在测试图上的预测（`labels_hat`）作为下一轮测试输入的标签特征（逐轮自反馈）。
