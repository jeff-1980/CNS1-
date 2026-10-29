# 对象 73 登记：a1018680161/Few-shot-Transfer-Learning

> 依据 `code/audit/main/ATTEMPT_PROTOCOL.md` 第 1 步；预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）；通用规则 461bc21、f2dcd3c、5ec85e7、d154119、5d67d8d。本登记在任何训练之前提交。登记日期 2026-10-24。按 jeff 2026-10-24 裁定试跑 4 组、平台适配逐条记录。

## 论文与代码

- 论文：*Few-shot transfer learning for intelligent fault diagnosis of machine*，Measurement 2020，DOI 10.1016/j.measurement.2020.108202（不开放，全文未取得）。
- 仓库：`a1018680161/Few-shot-Transfer-Learning`，commit `060fab6164b75db0302a47bbde886792d26d1b61`（2020-07-13），README 称论文代码。本地 `~/audit_ext/main_objects/73_a1018680161__Few-shot-Transfer-Learning`。

## 头条表格

- 论文全文不可得，README 无结果数值，**头条数值无法登记**；S6 对本对象记为“无法对照”。仓库只覆盖论文三个数据集中的 PU。

## 方法组（4 个，README 命令，1-shot、5-way、FFT 输入、1-D 模型）

- `Meta-transfer|…`：`main_meta_transfer.py -s 1`（关系网络元迁移）。
- `Feature-transfer|…`：`main_finetune.py -s 1 -u 0`（在源域预训练编码器，冻结后只训练新分类层）。
- `Full-finetune|…`：`main_finetune.py -s 1 -u 4`（预训练后全部微调）。
- `Direct|…`：`mian_direct.py -s 1`（目标域直接训练）。
- 任务名：`PU N09M07F10 artificial->natural 5-way 1-shot`。

## 配置

- 数据：PU 原始 `.mat`，工况 N09_M07_F10，每个轴承 20 个文件；源域（人工损伤）KA01、KA03、KA05、KA07、KA08、KI01、KI03、KI07，目标域（真实损伤）K001、KA04、KB23、KB27、KI04；不重叠 2048 点切段后取 |FFT| 前 1024 点。来源：本机 `~/data_pu`（只读，软链接到 `~/audit_ext/work73/pu/<轴承>/`）；KI07 在本机的压缩包已截断，从 Zenodo 镜像（10.5281/zenodo.15845309，同对象 21）下载，MD5 与 Zenodo 一致、`unrar t` 通过，与本机截断包共有的 62 个文件 CRC32 全部相同。
- 元迁移：代码默认 **5 个 episode**（每 5 个 episode 测试一次，即只评估一次）；jeff 裁定“元学习轮数按论文，论文未写则按代码默认”，论文不可得，按代码默认，因此本组只有一个评估点。Adam 1e−3，每类 15 个查询样本。
- 微调 / 特征迁移：源域预训练 200 轮（Adam 1e−3、batch 1000），目标域微调 50 次迭代（SGD 0.01、动量 0.9），每次迭代后在其余目标样本上测试。直接训练：Adam 1e−3、50 次迭代。
- 作者把预训练编码器缓存到 `.\result_finetune/models_save/`；同一重复中 `-u 0` 训练并保存、`-u 4` 载入（作者流程），每个重复开始前删除缓存以重新预训练。

## 种子方案

- 只有 `random.seed(1)` 用于文件顺序，torch / numpy 不设种子。按预注册做 **10 次不设种子的重复**，记录中 `seed` 为重复序号 0–9。

## 作者的选模规则

- 元迁移：报告最后一次测试准确率（另存测试集上最好的 checkpoint，不用于报告）。**作者口径 = 最后一个评估点**。
- 微调 / 特征迁移 / 直接训练：报告**最后 10 次迭代测试准确率的均值**（`final accuracy`）。审计以最后一次迭代为 `author_pick`，并另行核对记录中最后 10 次的均值与作者报告值（`author_check_mean10`）。
- **选模集**：无。**报告集**：目标域测试样本。

## 平台适配（jeff 2026-10-24 允许，逐条）

1. 两个数据生成文件的数据根目录 `D:\Study\Data\公开\轴承-PU` → 本机目录。
2. 两个数据生成文件中按 `'\\'` 拆分路径（mat 变量名、轴承名、类别名）→ `os.sep`。
3. 三个入口的 `matplotlib.use('TkAgg')` → `'Agg'`（无显示环境）。
4. `main_meta_transfer.py` 中 `.__iter__().next()` → `next(...)`（当前 PyTorch 已移除）。
5. `main_finetune.py` 中 `from collections import Iterable` → `collections.abc`（Python ≥ 3.10）。
- 其余 Windows 风格路径字符串（`.\\tempdata`、`%s\\train.log`）在 Linux 上作为普通文件名照常工作，未改。
- 钩子：三个入口各加若干行（取注入的钩子、每次测试后调用、最终值交出）。

## 泄漏编码 / 作者代码问题（只记录，不修复）

- 支持集与查询集取自同一批记录；元迁移另存测试集上最好的 checkpoint。
- 直接训练与微调只对 `classifier_test` 调用 `zero_grad()`，编码器梯度在迭代间累积（作者缺陷，影响结果）。
