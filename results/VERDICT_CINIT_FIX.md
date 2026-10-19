# 判定 C-INIT-FIX：修复 CWRU 加载器后的固定初始化确认 → LEAK_ROBUST

> 2026-10-13。预注册 `PREREG_CINIT_FIX_loader.md`（SHA-256 `7fca7c52…`，训练前登记；`audit/analyze_cinit_fix.py` 运行时校验通过）。数据 `cinit_fix_2x2.json`（80）、`cinit_fix_e5.json`（63），判定 `audit/cinit_fix_verdict.json`。缺陷与审计见 `AUDIT_CWRU_99mat.md`。
> 有效性：143/143 个实例的初始权重哈希与原 C-INIT 同（架构，种子）一致。GPU 在全部 125 次监控采样中处于温控降频状态（只影响耗时）。原预注册判定（COVERAGE、ANY_LARGER_SIZE、C-INIT 的 NARROW）保持不变。

## 1. 主判定：LEAK_ROBUST

| 架构 | 0.007″ 两臂 | 三尺寸两臂 | 判据 | 与原 C-INIT 逐实例一致 | 崩溃事件（修复后 / C-INIT） | +10 dB 准确率（修复后 / C-INIT） |
|---|---|---|---|---|---|---|
| wdcnn | IR 10/10 | OR 10/10 | REPLICATED | 20/20 | 18/20 / 20/20 | 0.874 / 0.872 |
| drsn | IR 10/10 | OR 10/10 | REPLICATED | 20/20 | 20/20 / 19/20 | 0.841 / 0.840 |

两个架构的比例效应都满足预设无效应判据（IR 占比差均为 0）。

## 2. 次要（描述性）

| 架构 | 修复后 | 原 C-INIT | 逐实例一致 | 读法 |
|---|---|---|---|---|
| lstm | 0.007″：IR 8/10（Normal 2）；三尺寸：OR 9/10（IR 1）→ REPLICATED | NOT_REPLICATED（三尺寸 OR 7/10） | 16/20 | lstm 在原运行、C-INIT、修复后三次中有两次满足判据；仍属于“运行间不稳定”，**不**据此把 lstm 升入核心主张 |
| cnn1d | 0.007″：IR 3/10；三尺寸：OR 10/10 → NOT_REPLICATED | 同 | 20/20 | 不变 |

E5（C-INIT 的 draw 0、2 + 修复后的 draw 1）：

| 架构 | 标签 | 修复后 draw 1 与受影响 draw 1 多数类别一致 | 单尺寸集合 |
|---|---|---|---|
| wdcnn | ANY_LARGER_SIZE（不变） | 7/7 | 不变 |
| drsn | ANY_LARGER_SIZE（不变） | 7/7（0.007″ + 0.021″ 在 draw 1 仍为 IR，draw 0、2 为 OR：差异来自 draw 1 的文件，不来自泄漏） | 不变 |
| lstm | **FILE_DEPENDENT**（原 ANY_LARGER_SIZE） | 5/7：0.007″ + 0.014″ 与三尺寸集合在修复后 draw 1 变为 IR | 不变（0.007″ IR、0.014″ OR、0.021″ Ball） |

E5 draw 1 崩溃事件 52/63（受影响 53/63）；+10 dB 准确率 0.779（受影响 0.782，draw 0、2 为 0.757）。

## 3. 对稿件的含义（按预注册 §4）

- 核心主张（wdcnn、drsn 的尺寸覆盖切换；单尺寸结果）在修复后成立，作为主证据
- 方法与局限必须写明：原加载器把 98.mat 的副本当作 99.mat 读入，所有 CWRU 原运行的 Normal 训练窗口与测试同源；审计、测试侧检查与修复后重训的结果
- lstm 的 E5 标签改为 FILE_DEPENDENT，混合尺寸结果在 lstm 上更不稳定；摘要中“单尺寸 … 在 wdcnn、drsn 和 lstm 中两次运行都成立”仍成立（单尺寸格不变），但不应再写 lstm 的 E5 标签为 ANY_LARGER_SIZE
- 原 2×2、E1–E3、E5 原运行与参考协议未重训，稿中标为“含同源记录的原运行”；参考协议对应前作 paperB，本项目不处理

## 4. 局限

- 修复同时改变了 Normal 抽样（池由 1419 变为 1421），与 C-INIT 的逐实例差异不能全部归于泄漏
- GPU 训练非逐位确定
