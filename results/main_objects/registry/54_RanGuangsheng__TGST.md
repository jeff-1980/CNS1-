# 对象 54 登记：RanGuangsheng/TGST

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-23。

## 论文与代码

- 论文：《基于语义指导和动态扩展的轴承故障诊断框架》，机械工程学报 2026, 62(11)，DOI 10.3901/JME.260770（README 注明）。全文未取得。
- 仓库：`RanGuangsheng/TGST`，commit `6a5ac7028adcaacfe30c8394dbe99566b8ad5d9a`（2026-07-26）。本地 `~/audit_ext/main_objects/54_RanGuangsheng__TGST`。入口 `tgst_project/scripts/experiment_v2.py`（主实验）与 `experiment_ablation.py`（消融）。

## 头条表格

- README “主实验 + 消融（λ_cl=1.0, temp=0.01, 50 epochs）”表（论文全文不可得，以 README 表为头条）：

| 组 | README 准确率 |
|---|---|
| TGST（正确语义） | 99.58% |
| Baseline（无文本） | 77.50% |
| Random（随机文本） | 77.92% |

- 动态扩展（4→5 类）实验不属本审计的选模问题，不跑。

## 方法组（3 个）

`TGST|CWRU 4-class`、`Baseline|CWRU 4-class`、`Random-text|CWRU 4-class`，对应 README 的三条命令：
- TGST：`experiment_v2.py --mode tgst --n_classes 4 --epochs 50 --lambda_cl 1.0 --temperature 0.01`
- Baseline：`experiment_v2.py --mode baseline --n_classes 4 --epochs 50`（README 命令未给 λ/温度，取代码默认）
- Random：`experiment_ablation.py --ablation random --epochs 50 --lambda_cl 1.0 --temperature 0.01`
- shuffle 消融不在 README 表中，不跑。

## 配置（README 命令 + 代码默认）

- 数据：`data/cwru_v2.py`：类 0 Normal 文件 97、98、99；类 1 “Inner_0.007” 107、108；类 2 “Outer_0.007” 135、136、137；类 3 “Ball_0.007” 159、160、161；读变量键 `X{编号:03d}_DE_time`。每类拼接后整体标准化，长 512、步长 256 的段中以 `RandomState(42)` 抽 200 段，前 70% 训练、后 30% 测试：训练 560 / 测试 240。
- **文件与类名的对应只记录、不修复**：按 CWRU 数据中心的文件编号表，97–99 是 0–2 HP 的正常数据（README 写 0HP）；135–137 是 48 kHz 驱动端外圈 0.007″（6 点）；159、160 是 12 kHz 驱动端外圈 0.007″（12 点），161 是 48 kHz 驱动端外圈 0.007″（12 点）——即代码中的 “Ball_0.007” 类实为外圈故障，且混用 12 kHz 与 48 kHz 文件。
- 训练：AdamW lr 1e−4、weight decay 1e−4，CosineAnnealingLR，batch 8，50 轮，d_model 64，2 层编码器；TGST 用冻结的 `bert-base-chinese`（从 Hugging Face 下载）。

## 数据来源

- CWRU 官方网站下载上述 11 个文件，逐个与服务器 Content-Length 核对完整，存放于 `~/audit_ext/data/cwru54/`。作者脚本把数据目录写死为 `~/tgst_project/datasets/CWRU`；运行脚本把进程的 `HOME` 设为 `~/audit_ext/work54home`（其中 `tgst_project/datasets/CWRU` 软链接到上述目录），不改作者代码；`HF_HOME` 指向本机的 Hugging Face 缓存。

## 种子方案

- 作者只给数据抽样与文本生成设种子（`seed=42`），训练不设 torch 种子。按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。

## 作者的选模规则

- 每轮在测试集上评估；`test_acc > best_acc` 时另存 `*_best.pth`，但**报告的“测试准确率”是训练结束后对最终模型的评估**（写入 report.txt 并打印）。**作者口径 = 最后一轮（author = last）**。
- **选模集**：无。**报告集**：测试集（30%）。
- 补丁：两个脚本各加 3 行——取 runpy 注入的 `AUDIT_HOOK`、`AUDIT_FINAL`；每轮 `evaluate_model` 之后调用钩子（记录作者算出的测试准确率，并在 `isolate` 中另算测试交叉熵）；最终评估之后把准确率交给 `AUDIT_FINAL`。

## 泄漏编码（只记录，不修复）

- **TGST 的文本输入由样本的类别标签生成**（`generate_unique_texts(self.labels)`），测试集同样如此：测试时模型的输入中含有由真实标签生成的文本。这是论文方法本身的设计，只记录。
- 训练 / 测试段取自同一条拼接信号（随机起点的段，前 70% / 后 30% 按抽样顺序划分，段之间可重叠 50%）。
- 每轮在测试集上评估（作者报告最终模型，但另存测试集上最好的权重）。
