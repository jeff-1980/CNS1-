# 对象 23 登记：DABINB/CSST-Net

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-22。

## 论文与代码

- 论文：B. Wang, Y. Xiong & L. Tan，*A High-Precision Aeroengine Bearing Fault Diagnosis Based on Spatial Enhancement Convolution and Vision Transformer*，IEEE Transactions on Instrumentation and Measurement 74 (2025)，DOI 10.1109/TIM.2024.3502884（OpenAlex：closed）。
- 仓库：`DABINB/CSST-Net`，commit `b445512383d3f511c02dae2a5a2756a82eb5c73b`（2024-12-18），README 称 “official PyTorch codes for the paper”。本地 `~/audit_ext/main_objects/23_DABINB__CSST-Net`。入口 `CSST_Net/dataset.py`（数据）→ `CSST_Net/train.py`。

## 头条表格

- 论文全文不可得，README 无结果表，**头条表格的数值无法登记**；S6 对本对象记为“无法对照”。

## 方法组（1 个）

- 仓库只有 CSST-Net。只有 **1 个方法组**：`CSST-Net|HIT SNR 6dB`。H-B2 对本对象无组对可算。

## 配置（作者代码默认）

- 数据：作者 README 给出的 Google Drive（即 HIT 数据集作者 HouLeiHIT 的 Drive 文件夹）中的 `data1.npy`、`data3.npy`、`data4.npy`、`data5.npy`（正常、内圈 1、内圈 2、外圈；`data2.npy` 不用），下载到 `~/audit_ext/data/hit23`。
- 预处理（`dataset.py`，原样运行一次）：每个文件按 20 个转速工况各取 18 条样本的 3 个加速度通道拼接，加白噪声（SNR −2、0、2、6、10 dB 各一份），按通道标准化，长 1024 不重叠切窗，每类每工况取样后共 200 个窗；每个窗 3 个通道各做 `cmorl3-3` CWT（256 个尺度）并缩放到 224 × 224；4 类共 800 个样本。
- 任务：`train.py` 默认读取 **SNR 6 dB**（`data_6.npy`），按 461bc21 规则 (2)（表格不可见时取代码默认）。
- 训练：`train_test_split(test_size=0.3, random_state=42)`（训练 560、测试 240），双精度模型，Adam lr 4e−4，余弦退火，batch 32，**30 轮**，交叉熵。

## 种子方案

- `train.py` 不设任何种子（模型初始化、batch 打乱、评估抽样都不设种子；只有训练 / 测试划分固定 `random_state=42`；`dataset.py` 的加噪也不设种子，数据只生成一次）。按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。

## 作者的选模规则

- 每轮最后一个 batch 后，作者在测试集上评估（`torch.randperm(240)[:240]`，即全部 240 个测试样本的一个排列）并记 `Te_acc`、`Te_loss`；`Te_acc >= best 且 Te_loss <= best_loss` 时记 `best_model_weights = Model.state_dict()`——这是对模型参数的引用而非拷贝，保存的实际是最终权重。训练结束后作者以最终模型在测试集上打印 `Accuracy`。**作者口径 = 最后一轮（第 30 轮）**（作者意图是测试集上取最好，代码实际效果为最后一轮，只记录）。
- **选模集**：无（作者以测试集作选择依据，但实际未生效）。**报告集**：测试集（240 个）。
- 补丁：`train.py` 中加一个钩子，在作者每轮测试评估后记录作者算出的 `Te_acc`、`Te_loss`（`select_split=None`、`report_split="test"`，不另做评估）；训练结束时把作者打印的 `Accuracy` 交给记录器；`author_pick(30)`、`finish(Accuracy)`。驱动以 `runpy` 按 `__main__` 运行 `train.py`。

## 数据来源

- `~/audit_ext/data/hit23/data{1,3,4,5}.npy`（作者 README 链接）。`dataset.py` 以 Windows 路径写出 `predata\data_6.npy`；在 Linux 上该文件名含反斜杠，工作目录中以软链接 `predata/data_6.npy` 指向它（不改代码，偏离记录）。

## 泄漏编码（只记录，不修复）

- 加噪与切窗后整体随机划分训练 / 测试，同一工况、同一原始记录的相邻窗口可同时出现在训练与测试中。
- 每轮在报告集上评估，作者意图在报告集上取最好（实际为最后一轮）。
