# 对象 36 登记：Luojunqing666/Eviformer-An-uncertainty-fault-diagnosis-framework-guided-by-evidential-deep-learning

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-22。

## 论文与代码

- 论文：J. Luo, F. Li, X. Xu, W. Zhao & D. Zhang，*Eviformer: An uncertainty fault diagnosis framework guided by evidential deep learning*，Engineering Applications of Artificial Intelligence 161 (2025) 112328，DOI 10.1016/j.engappai.2025.112328（OpenAlex：green，无可取全文链接）。
- 仓库：`Luojunqing666/Eviformer-An-uncertainty-fault-diagnosis-framework-guided-by-evidential-deep-learning`，commit `ab6b66632379a1960f211387d77406d7979290eb`（2026-05-26），作者本人仓库。**代码晚于论文约 8 个月**，按 461bc21 规则 (1) 纳入，日期差作为协变量。入口 `main.py --train`。

## 头条表格

- 论文全文不可得，README 无结果表，**头条表格的数值无法登记**；S6 对本对象记为“无法对照”。

## 方法组（4 个）

- 仓库实现 3 个骨干（LeNet、1D-ViT、MC-SwinT）× 4 种损失（交叉熵、EDL-MSE、EDL-Digamma、EDL-Log）。论文表格不可见，取 **README “Usage” 中作者列出的 4 条训练命令**作为 4 组（预注册上限 4 组）：
  - `MC-SwinT+EDL-Digamma`（README 标注 recommended，即提出的方法）：`--model mcswint --uncertainty --digamma`
  - `1D-ViT+EDL-Log`：`--model vit --uncertainty --log`
  - `LeNet+EDL-MSE`：`--model lenet --uncertainty --mse`
  - `MC-SwinT+CE`（不带不确定性）：`--model mcswint`
- 组名 `<组>|CWRU 1797`。

## 配置（作者代码默认）

- 数据：CWRU 12 kHz 驱动端、1797 rpm（`load_cwru_dataset` 默认 `rpm=1797`），正常 97 号 + 9 个故障文件，长 1024 不重叠切窗，10 类；`train_test_split(test_size=0.2, random_state=40, stratify)` 分训练 / 验证；归一化 0–1。
- 训练：Adam lr 1e−3、weight decay 0.005，`StepLR(7, 0.5)`，batch 16，**100 轮**（README 命令均为 `--epochs 100`），`--num_classes 10`。

## 种子方案

- 代码不设任何种子（只有数据划分固定 `random_state=40`）。按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。

## 作者的选模规则

- `train_model` 每轮在 `val` 上评估，`epoch_acc > best_acc`（严格大于，平局取较早轮）时深拷贝权重；结束时打印 `Best val Acc` 并加载该权重。仓库没有独立测试集，`val` 即报告集。**作者口径 = 在报告集上取最好（author ≈ best_rep）**。
- **选模集**：无独立选模集（以报告集选）。**报告集**：`val`（20%）。
- 补丁：`train.py` 每轮 `val` 阶段结束后调用钩子，记录作者算出的 `val` 准确率与损失作为报告集（`select_split=None`、`report_split="test"`，不另做评估）；按作者严格大于规则标 `author_pick`；`finish(best_acc)`。

## 数据来源

- 本机只读 `~/论文CNS/data/cwru_12k_de`，按作者目录结构以软链接放在工作目录 `<root>/12k Drive End Bearing Fault Data/`（作者代码在该目录中同时读取正常文件 97.mat）。

## 泄漏编码（只记录，不修复）

- 不重叠切窗后随机划分，同一记录的相邻窗口分属训练与验证。
- 每轮在报告集上评估并取最好。
