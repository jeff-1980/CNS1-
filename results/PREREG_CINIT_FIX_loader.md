# 预注册 C-INIT-FIX：修复 CWRU 99.mat 读取键后的固定初始化确认实验

> 2026-10-13，写于任何 C-INIT-FIX 训练之前。触发：Codex 会话《MSSP26-7400 桌拒后处理顺序：先核查 CWRU 同源信号污染》；项目卡 §27。
> 本实验**不覆盖**任何历史结果、预注册标签或 release v1.0；历史结果在审计文件中标注为"训练集含测试同源 Normal 记录"。
> 代码：`code/cinit_fix.py`（由 `code/cinit.py` 派生，只改输出路径、E5 只跑 draw 1、启动时断言加载器已修复）；分析：`audit/analyze_cinit_fix.py`（运行时校验本文件 SHA256）。

## 0. 已核定的缺陷与审计（写于本文件之前，均为只读）

- `cwru_loader._extract_de_signal` 取 .mat 中首个含 `DE_time` 的键。CWRU 的 99.mat 含 `X098_DE_time`（98.mat 的副本）与 `X099_DE_time` 两个键，旧加载器取到前者，因此"99.mat"的信号与 98.mat 逐字节相同。40 个文件中只有 99.mat 有此情况（`audit/exposure_audit.py`）
- `split_seed=42` 下 99 在 train、98 在 test（40 文件与 16 文件两套都一样），所以 944/944 个 Normal 测试窗口在 train 划分中都有逐字节副本
- 逐实例重放取样（`audit/exposure_by_instance.csv`；2×2 / E3 / E2 / E1 的取样哈希与存档逐一吻合）：2×2、E3、C-INIT 2×2 全部 180 个实例的 Normal 训练窗口中 65–70% 为测试副本；E5 与 C-INIT E5 仅 draw 1（Normal_load2）受影响，各 63/189
- 事后推理检查（`audit/rescore_leakfree.py`，未预注册）：把 Normal 测试窗口换成未见过的 100.mat 或真实 X099 后，已完成的实例中 −8 dB 主导类别与崩溃事件与原测试集逐实例相同。这只检验了**测试侧**；**训练侧**（模型只见过 2 个而非 3 个不同的 Normal 记录、且其中一个是测试记录）需重训
- 修复：`_load_mat_signal` 优先取与文件编号一致的键（`X{nnn}_DE_time`）。修复后两套文件集都无重复信号、0 个测试窗口在 train/val 中有副本、文件到划分的分配不变、测试集与原来逐窗口相同（`audit/verify_fix.py`）。`CWRU_LEGACY_FIRST_KEY=1` 只用于复现存档

## 1. 范围（143 个训练实例）

| 部分 | 设计 | 架构 | 实例 |
|---|---|---|---|
| F-2×2 | 与 C-INIT 2×2 完全相同（4 臂 × 5 种子，`torch.manual_seed(20000+seed)` 先于构造），只换修复后的加载器 | wdcnn, drsn, lstm, cnn1d | 80 |
| F-E5d1 | 与 C-INIT E5 相同，只跑受影响的 draw 1（7 个尺寸集合 × 3 种子） | wdcnn, drsn, lstm | 63 |

训练函数、评估函数、噪声种子、测试集、步数与预热均不变。修复后 Normal 训练池由 1419 变为 1421 个窗口，所以 2×2 的 Normal 抽样会选到不同窗口（故障窗口不变）；这是数据修复的必然结果，不另作因素。

## 2. 主判定（只看 wdcnn、drsn）

沿用 C-INIT 的复现判据：0.007″ 两臂 −8 dB 主导类别为 IR 的比例 ≥ 0.8，且三尺寸两臂为 OR 的比例 ≥ 0.8（每臂 n = 5）。

- **LEAK_ROBUST**：wdcnn 与 drsn 都满足
- **LEAK_SENSITIVE**：wdcnn 或 drsn 有一个不满足（逐架构报告是哪一个）

## 3. 次要（描述性，不改主判定）

1. lstm、cnn1d：按 C-INIT 同一规则给出 REPLICATED / NOT_REPLICATED，并与 C-INIT 原结果对照
2. 比例：同一覆盖下两种比例的 IR 占比差 < 0.4（与 C-INIT 相同）
3. 逐实例（臂、架构、种子相同）主导类别与 C-INIT 原结果一致的个数
4. 崩溃事件（concentration ≥ 0.90）个数，与 C-INIT 原结果对照
5. E5：用 C-INIT E5 的 draw 0、2 加上 F-E5d1 的 draw 1，按 `analyze_e5.label()` 重新给出三个架构的标签，与 ANY_LARGER_SIZE 对照；并报告 draw 1 的抽取层面多数类别与 draw 0、2 是否一致
6. +10 dB 准确率与 Normal 召回，与 C-INIT 原结果对照（预期修复后可能略低，因为测试 Normal 不再被记忆）

## 4. 稿件规则（事先确定）

- LEAK_ROBUST → 稿件保留核心主张，在方法与局限中如实写明缺陷、审计与修复后重跑结果，并以修复后结果作为主证据；历史结果保留为"含同源记录的原运行"
- LEAK_SENSITIVE → 不满足的架构从核心主张中删除；如 wdcnn、drsn 都不满足，核心主张撤回，稿件不转投，转为负结果记录
- 无论结果如何：参考协议（E4 / WP1，即前作 paperB 的 CWRU 协议）同样含 944 个测试副本，本实验**不**处理前作，只在项目卡中记录并报告给作者

## 5. 局限

- GPU 训练非逐位确定；修复同时改变了 Normal 抽样，所以与 C-INIT 原结果的逐实例差异不能全部归于泄漏
- 只覆盖 C-INIT（稿件主证据）；原 2×2、E1–E3、E5 原运行与参考协议不重训
