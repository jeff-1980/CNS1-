# 对象 02 登记：giabao804/few-shot-mamba

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则见 DEVIATIONS 2026-10-20（commit 461bc21）与缺失数据规则（f2dcd3c）。本登记在任何训练之前提交。登记日期 2026-10-20。

## 论文与代码

- 论文：Truong, Tran, Than, Nguyen, Pham 等，*SC-MambaFew: Few-shot learning based on Mamba and selective spatial-channel attention for bearing fault diagnosis*，Computers and Electrical Engineering 123 (2025) 110004，DOI 10.1016/j.compeleceng.2024.110004（OpenAlex 记录上线日期 2024-12-26）。
- 仓库：`giabao804/few-shot-mamba`，commit `41bd1757102d7a03f7750dc3d0fb0b95e8f9410d`（检索时 HEAD，2026-05-28 “update README”）；本地 `~/audit_ext/main_objects/02_giabao804__few-shot-mamba`。README 自称 “official implementation”。
- 代码与论文时间：仓库创建于 2024-09-30，早于论文上线（约 −87 天）；最后推送 2026-05-28。不进入“代码晚于论文 ≥ 1 年”的敏感性剔除集。

## 头条表格

- 论文全文审计方无访问权限（Elsevier 403；OpenAlex 无摘要）。仓库里没有结果表或结果图。**头条表格的数值无法登记**；S6（复现）对本对象记为“无法对照”。
- 只能从代码与脚本确定配置（见下）。

## 方法组（1 个）

- 仓库只含提出的方法 `MainNet`（`net/new_proposed.py`，训练脚本里命名 `SCMamba`）。训练脚本里对比方法（QS_Former、RelationNet、MatchingNet、ProtoNet、CosineClassifier、SA_CovaMNet、CovarianceNet_64、MFNet、EnsembleNet）已被注释掉，这些类在仓库里不存在。
- 故本对象只有 **1 个方法组**：`SCMamba|CWRU 1-shot 60`。H-B2（组对显著性）对本对象无组对可算。

## 配置（作者默认）

按 461bc21 规则 (2)：头条表格无法查看，取仓库给出的第一个配置——README 与 `train.sh` 先列 CWRU、先列 1-shot，`train.sh` 对 CWRU 传 `--training_samples_CWRU 60`：

- 调用：`python train_1shot.py --dataset CWRU --training_samples_CWRU 60 --data_path <CWRU 目录>`（与 `bash train.sh 1 CWRU` 相同，`--model_name` 被脚本内覆盖为 `SCMamba`）。另传 `--cfs_matrix`（作者自己的开关，关掉训练后的混淆矩阵 / t-SNE 绘图）与 `--path_weights`（作者的权重目录参数，指向审计目录）。
- 数据：CWRU 12 kHz 驱动端，`12DriveEndFault` 三个转速 1772 / 1750 / 1730 rpm（负载 1/2/3 hp），10 类（正常 + 球 / 内圈 / 外圈@6 × 0.007/0.014/0.021），同类的三个转速并成一类；DE 与 FE 两通道拼接，窗长 2048。训练：每个文件取起点 0 与 80 的 2 个窗，共 60 个；测试：每个文件从 60000 起每 2048 取 25 个窗，共 750 个。输入转成 64×64 对数 STFT 谱图（`function.to_spectrum`）。
- 训练：10-way 1-shot，每轮 100 个训练 episode、每 episode 每类 1 个 query；Adam lr 1e-3，StepLR(step 10, γ 0.1)，100 轮，损失 `ContrastiveLoss`（即 softmax 交叉熵）。
- 评估：每轮在测试集上抽 75 个 episode（每类 1 个 query，共 750 个 query）算准确率。
- 未选的配置及原因：CWRU 5-shot（`train_5shot.py` 调用 `CWRU(...)` 少传 `data_path`，按作者代码无法运行，只记录）；HUST（`train.sh` 传 `--dataset HUST`，与脚本里的选项名 `HUST_bearing` 不符，按作者脚本无法运行；直接调 `--dataset HUST_bearing` 可以）。HUST 数据按授权下载到 `~/data_hust`（Mendeley Data cbv7jyx4p9 v3，99 个 .mat，640 MB，逐个对 Mendeley 给出的 SHA-256 核对；登记提交时下载仍在进行，完成情况写进偏离记录），本对象正式跑不用。

## 种子方案

- 作者有种子机制但写死：`function.seed_func()` 固定 `random / numpy / torch` 种子为 42，并设 `cudnn.deterministic=True`；脚本里另有 `--seed`（默认 42）参数但没有被用到。
- 本对象用种子 **0–9**：补丁把作者自己的 `--seed` 参数接进 `seed_func(seed)`（不改其他行为），逐条写入偏离记录。
- 记录：CWRU 数据在 `seed_func` **之前**加载，加载时 `sklearn.utils.shuffle(train_cuts)` 用未设种子的全局 numpy 随机数，只打乱同一文件内 2 个训练窗的先后；随后 `_shuffle` 用固定 `random.Random(0)`。

## 作者的选模规则

- 每轮结束在**测试集**上算准确率；准确率严格高于此前最好值时保存整个模型（`torch.save(net, ...)`，删除旧文件），最后返回并报告这次最好的准确率 `pred_acc`。
- `author` 协议 = 测试集准确率第一个严格最大值所在的轮（与 `best_rep` 同定义）。**选模集 = 报告集 = 测试集**；没有验证集。
- 补丁：`select_split=None`、`report_split="test"`；每轮记作者算出的准确率；准确率严格上升时 `author_pick(epoch)`；训练结束 `finish(pred_acc)`。
- 测试损失：作者不算。补丁在作者评估之前、同一随机数状态下（`isolate` 包住）用作者的训练损失 `ContrastiveLoss` 在同一批测试 episode 上补算，随后作者的评估抽到的是同样的 episode。

## 数据来源

- CWRU：`/home/jeffwork/论文CNS/data/cwru_12k_de`（只读）。作者代码按 `metadata.txt` 找 `<data_path>/12DriveEndFault/<rpm>/<故障>.mat` 与 `NormalBaseline/<rpm>/Normal.mat`，找不到就去已失效的 csegroups 链接下载；本对象在 `~/audit_ext/data/cwru_fewshot/` 建同名软链接指向本地编号文件（30 个），不改作者代码。
- 读取的键：作者 `fliter_key` 取第一个含 `FE_time` 的键，拼出 DE 与 FE 两个键。98.mat → `X098_DE_time + X098_FE_time`；100.mat → `X100_*`；**99.mat（1750 rpm 正常）含 `X098_*` 与 `X099_*` 两组键，作者代码读到的是 `X098_*`**，即 1750 rpm 的“正常”数据实为 1772 rpm 正常数据的重复。只记录，不修复。

## 泄漏编码（只记录，不修复）

- 选模在测试集上做，报告的数就是测试集上挑出的最好轮。
- 测试 episode 每轮重新随机抽（`FewshotDataset.__getitem__` 里 `torch.randperm`），每轮的“测试准确率”是在不同的 750 个 query 抽样上算的。
- 正常类：1750 rpm 与 1772 rpm 的正常样本来自同一信号（99.mat 读错键），两者的训练窗起点相同（0 与 80）、测试窗起点相同，正常类里有逐样本重复，且一份重复在训练、一份在测试的情况不存在（训练窗在 0–2128，测试窗在 60000 以后）。
- 训练窗 2 个（起点 0 与 80）高度重叠。
