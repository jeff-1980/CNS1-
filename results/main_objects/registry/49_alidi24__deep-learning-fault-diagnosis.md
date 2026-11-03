# 对象 49 登记：alidi24/deep-learning-fault-diagnosis

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-23。按 jeff 2026-10-23 裁定纳入（数据为仿真信号，另报去掉本对象的敏感性分析）。

## 论文与代码

- 论文：A. Dibaj 等，*Fault detection of offshore wind turbine drivetrains in different environmental conditions through optimal selection of vibration measurements*，Renewable Energy 203 (2023) 161–176，DOI 10.1016/j.renene.2022.12.049（CC-BY，全文取自挪威开放仓库 NVA，hdl 11250/3051501）。
- 仓库：`alidi24/deep-learning-fault-diagnosis`，commit `d7f979c3363edc6dcfaf3d2f2e9782b68ec47669`（2025-02-27），作者即第一作者（pyproject 署名 Ali Dibaj）。本地 `~/audit_ext/main_objects/49_alidi24__deep-learning-fault-diagnosis`。入口 `train.py`。

## 头条表格

- 论文 Table 12（以优化后的 A1A + A3A 测点为输入，三种模型的分类结果）：**CNN 在 EC1、EC2、EC3 的总体准确率均为 100%**（各类均 100%）；对比的 SVM、MLP 用人工特征，不在仓库中。S6 以 CNN 的 100% 对照。

## 方法组（1 个）

- 仓库只实现论文推荐的 PCA-CNN（A1A、A3A 两个测点做 PCA 取第 1 主成分后输入 CNN）：`PCA-CNN|NREL 5MW A1A+A3A`。

## 配置（作者代码默认，`src/config.py`）

- 数据：Hugging Face 公开数据集 `alidi/wind-turbine-5mw-bearing-dataset`（CC-BY-4.0，无需登录，原始数据见 Zenodo 10.5281/zenodo.7674842）：12 条长信号 = 3 个环境工况（EC1–EC3）× 4 类（健康、主轴承、高速轴轴承、低速级行星轴承故障），**为 NREL 5MW 传动链高保真模型的仿真信号**。每条信号对 A1A、A3A 逐样本做 PCA 取第 1 主成分，按帧长 8192、跳长 6144 分帧；三种工况合并训练。按类分层 `train_test_split(random_state=42)`：测试 20%，其余中 10% 作验证：训练 1000 / 验证 112 / 测试 280。
- 训练：Lightning `Trainer`，Adam lr 1e−3、weight decay 0.01，batch 32，最多 500 轮，EarlyStopping（val_loss，耐心 40），ReduceLROnPlateau（val_loss，因子 0.9，耐心 20），ModelCheckpoint（val_loss 最小）。
- 代码与论文描述不一致处（只记录，按 461bc21 规则 (1) 不视为 E8，README 未声明差异）：论文模型用 Keras/TensorFlow 实现、各环境工况分别训练、训练与测试来自不同仿真、信号段长 2000；仓库为 2025 年的 PyTorch Lightning 实现，三个工况合并，同一条信号分帧后随机划分，帧长 8192。

## 种子方案

- 作者只给数据划分设种子（42），训练不设 torch 种子（无 `seed_everything`）。按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。

## 作者的选模规则

- ModelCheckpoint 按 **验证集 val_loss 最小**保存（严格更好才替换，平局取较早轮）；训练结束后加载该 checkpoint 运行 `trainer.test`，报告测试准确率。**选模集**：验证集（112）；**报告集**：测试集（280）。
- 不改作者文件：驱动在作者 `setup_callbacks()` 返回的回调列表末尾追加一个审计回调，每轮验证结束时记录作者记录的 `val_acc`、`val_loss`（Lightning 按轮汇总的均值），并在 `isolate` 中用作者的模型与损失函数补算测试集准确率与交叉熵；按 val_loss 最小规则标 `author_pick`；`finish` 接收作者 `trainer.test` 的 `test_acc`。

## 泄漏编码（只记录，不修复）

- 训练、验证、测试帧来自同一条仿真信号（相邻帧重叠 25%），随机划分。
- `config["is_aug"] = "False"` 是字符串，在 `if is_train and self.config["is_aug"]` 中为真，训练时实际执行了数据增强（作者代码缺陷，只记录）。
- 模型输出层后接 Softmax 再用 CrossEntropyLoss（双重 softmax），只记录。

## 数据说明（jeff 裁定）

- 数据是仿真信号而非实测；主分析纳入本对象，另报去掉本对象的敏感性分析。
