# 对象 24 登记：Erikdai/Fault-Diagnosis-of-Aero-engine-Inter-shaft-Bearings

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-21。

## 论文与代码

- 论文：C. Dai, G. Su & X. Li，*Fault Diagnosis of Aero-Engine Inter-Shaft Bearings Using a Parallel SE-Depthwise Separable Convolutional Neural Network and Transformer Model*，2024 7th International Conference on Mechatronics and Computer Technology Engineering (MCTE)，pp. 304–308，DOI 10.1109/MCTE62505.2024.11117663（IEEE 订阅；OpenAlex 查不到该 DOI）。
- 仓库：`Erikdai/Fault-Diagnosis-of-Aero-engine-Inter-shaft-Bearings`，commit `658c64286fdd547d577437f2ad5280585aa22e68`（2026-10-15），README 称 “the PyTorch implementation of the model proposed in” 该论文。**代码最后提交晚于论文约 2 年**，按 461bc21 规则 (1) 纳入，日期差作为协变量。本地 `~/audit_ext/main_objects/24_Erikdai__Fault-Diagnosis-of-Aero-engine-Inter-shaft-Bearings`。入口为三个笔记本，按 README 顺序：`create_dataset_1.ipynb` → `create_dataset_2.ipynb` → `SEDSCN-TRANS.ipynb`。

## 头条表格

- 论文全文不可得，**论文表格数值无法登记**。README 给出仓库参考运行结果（CWRU 0 hp、10 类，测试集 224 个窗口）：Accuracy **99.55%**、Macro F1 0.995。S6 以 README 值作参考对照，并注明不是论文表。

## 方法组（1 个）

- 仓库只有提出的 SEDSCN-Trans（SE 深度可分离 1D CNN 支路 + Transformer 支路 + 交叉注意力融合）。只有 **1 个方法组**：`SEDSCN-Trans|CWRU 0hp 10c`。H-B2 对本对象无组对可算。

## 配置（作者笔记本默认）

- 数据：仓库自带 `crwu_dataset/` 10 个 `.mat`（CWRU 12 kHz 驱动端，0 hp：97、105、118、130、169、185、197、209、222、234），每类取前 119808 点；窗长 1024、重叠 50%，每类 233 个窗口；全部窗口用 `sklearn.utils.shuffle`（未设种子）打乱后按 70 / 20 / 10 分为训练 / 验证 / 测试。
- 特征：每个窗口 VMD（K = 4，alpha 2000，tau 0，tol 1e−7，`vmdpy`）4 个模态 + FFT 幅值谱，得 5 × 1024。
- 训练：Adam 3e−4，batch 32，**30 轮**，`CrossEntropyLoss(reduction='sum')`；三个 DataLoader 都 `shuffle=True, drop_last=True`。
- 数据准备（前两个笔记本）只运行一次，10 次重复共用同一划分，与作者流程一致（作者也是一次性生成并用 `joblib` 存盘）；划分本身未设种子，审计这一次的划分如实保存。

## 种子方案

- 作者训练笔记本 `torch.manual_seed(100)`。按预注册改为 **种子 100–109**（作者种子起连续 10 个）。

## 作者的选模规则

- 训练循环每轮在验证集上算准确率与损失（注意：验证时**没有**切到 `model.eval()`，dropout 0.5 仍生效，只记录），`val_accuracy > best_accuracy` 时执行 `best_model = model`。这是对同一对象的引用，不是拷贝，所以训练结束保存的 `best_model_cnn_transformer_catt3ORI.pt` 实为**最后一轮**模型。随后测试单元加载该文件，`model.eval()` 后在测试集上报告准确率与分类报告。**作者口径 = 最后一轮（第 30 轮）**（作者意图是验证集最好，但代码实际效果为最后一轮，只记录）。
- **选模集**：验证集（20%，466 个窗口）。**报告集**：测试集（10%，233 个窗口；作者的测试 DataLoader `shuffle=True, drop_last=True`，每次评估随机丢掉 9 个，实际报告的是随机 224 个）。
- 补丁：把训练笔记本的代码单元原样抽成脚本，只在训练循环里加记录：每轮记作者算出的验证准确率与损失（`select_split = "val"`，作者口径，train 模式）；在 `isolate` 内以 `eval` 模式在**全部 233 个**测试窗口上算测试准确率与损失（`report_split = "test"`）；`author_pick(30)`；作者测试单元逐字照跑，取其报告的准确率 `finish(...)`；同时记下作者那次实际用到的 224 个测试窗口，用最后一轮记录的逐样本预测在这 224 个上重算准确率，与作者值核对（`author_check_subset`）。

## 数据来源

- 仓库自带 `crwu_dataset/*.mat`。

## 泄漏编码（只记录，不修复）

- 窗口重叠 50%，所有窗口先整体打乱再划分，测试窗口与训练窗口在原始信号上相邻重叠。
- 不在报告集上挑轮次；验证评估未切换 eval 模式。
