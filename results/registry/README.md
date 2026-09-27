# 训练实例注册表（门 A 第 2 项）

> 2026-09-23。脚本 `code/build_registry.py`（主表）+ 本目录下的补字段步骤。统计单位 = **训练实例**（一次训练得到一个权重）。

## 文件

| 文件 | 内容 |
|---|---|
| `training_instances.csv` | 本项目训练实例，175 行 = **168 分析用** + 7 冒烟（`role=smoke`，不进任何统计） |
| `paperB_result_files.csv` | paperB 结果包逐文件清单：**185 个结果 JSON**（另有 2 个 `config.json`，合计 187 个文件）。**与本项目实例分开计数，不相加** |
| `raw_hashes.csv` / `raw_hashes_manifest.md` | 原始数据与代码哈希（门 A 第 1 项） |

## 168 个分析实例的构成

| 来源 JSON | 数据集 / 协议 | 实例 | 轮数（逐 run 读日志） | 权重 |
|---|---|---|---|---|
| cwru_prior_screen | CWRU 40 文件 | 25 | **60** | 无 |
| cwru_ep100_probe | CWRU 40 文件 | 6 | 100 | 无 |
| cwru_16file_probe | CWRU 16 文件 | 6 | 100 | 无 |
| cwru40_ckpt_probe | CWRU 40 文件（`ep100_probe` 的配置重复） | 6 | 100 | 有 |
| cwru16_ckpt_probe | CWRU 16 文件（`16file_probe` 的配置重复） | 6 | 100 | 有 |
| jnu_prior_screen | JNU Mode A 600 rpm | 25 | 100 | 无 |
| jnu_fine_snr | JNU 细网格（= 三臂 orig） | 25 | 100 | 有 |
| jnu_arm_balanced / subsample | JNU 三臂 | 25 + 25 | 100 | 无 |
| pu_fine_snr | PU 同工况时间切分 | 19（计划 25） | 100 | 无 |

- 此前各文档写的 **156 漏计了共同测试集那轮的 12 个重训实例**；配置重复的实例各自独立训练，因此分别计数，并在 `config_replicate_of` 列标注
- 带权重哈希 **37**，缺失 131（早期扫掠没有存权重）

## 更正：CWRU 原记录是 60 轮，不是 40 轮

`CONTRIBUTION_DIFF` §〇 与 `subjournal-plan` 1c 写的"40 轮（`cwru_run.log` 25 run 全在 ep 40 结束）"有误。
当时用 `grep -c "ep 40"` 统计，而 60 轮的 run 同样会打印 `ep 40` 那一行。逐 run 读取最后一个 epoch 行后：**25/25 均为 60 轮**。
结论方向不变：100 轮时仍全部 → OR（6/6 + 6/6）。

## §4 三个"未定稿"统计量的定稿

| 统计量 | 原报告 | 注册表范围内重算 | 定稿 |
|---|---|---|---|
| JNU 三臂 balanced（秩基，含删失） | η²=0.544, p=0.0028 | η²=0.544, p=0.0028（`analyze_arms.py`，n=25） | ✅ 原样定稿 |
| JNU 三臂 subsample（秩基，含删失） | η²=0.538, p=0.0058 | η²=0.538, p=0.0058（n=25） | ✅ 原样定稿 |
| PU 架构效应 | η²=0.539, **p=0.0072**（原始 onset；第三轮删失约定之前的方法） | 原始 onset：η²=0.539, p=0.0060（5000 次置换，与原值相差不到 1 个 MC 标准误）；**秩基：η²=0.438, p=0.029** | ⚠️ **按现行约定改报秩基 η²=0.44, p=0.029**；n=19，4 个架构（缺 drsn）。19 个 run 均无删失，所以两种方法的差异来自秩变换本身，而非删失处理 |

## 已知边界

- `code_hash_at_run` = not captured：训练时没有记录代码哈希；`prior_screen.py` 此后改过（CWRU_ROOT 环境变量、warmup 补丁）。`code_hash_prior_screen_now` 只记录当前版本
- 评估噪声的随机数没有单独设种（沿用训练后的 torch 全局随机状态）；`cross_eval.py` 起才对评估噪声单独设种
- 冒烟与探针实例（7 个）不进入任何统计

## 2026-09-25 补登：2×2 起的 403 个实例

- 脚本 `code/extend_registry.py`：只追加，按 `instance_id` 跳过已有行（重复运行新增 0 行）；补登前的表备份为 `training_instances.pre_extend_2026-09-25.csv`
- 来源：arm2x2 60、arm2x2_lt 40、ext_e1 40、ext_e2 45、E4 15、WP1 93（含 3 个转速识别）、WP2 110
- **每个实例的权重文件都重新计算了 SHA256，并与结果 JSON 中记录的值逐一比对（511 个权重文件，全部一致）**。E4 与 WP1 每个实例有 best / final 两份权重，`weights_path` / `weights_sha256` 以 `best=… | final=…` 记录
- `code_hash_at_run`：运行时未记录代码哈希，填 "not captured"，并附当前运行脚本的 SHA256 前 16 位（运行后脚本未再修改的，可据此核对）
- `config_replicate_of`：ext_e1 的 ID0N0S0 = arm2x2 C007_R31 的同配置重训；wp1_jnu 的 wdcnn s0 = wp1_jnu_rpmid_600 的同配置重训
- **合计 578 行 = 571 个分析实例 + 7 个冒烟**

## 2026-09-25 补登：B 线确认集 45 个实例

- `b_confirm.json`：PU 3 个新工况 + JNU 800 / 1000 rpm × wdcnn / drsn / lstm × 3 种子；权重 SHA256 复核一致
- **合计 623 行 = 616 分析 + 7 冒烟**。WP5 与 B 线开发阶段为纯推理，不新增训练实例

## 2026-09-26 补登：E5 189 个实例 + C-INIT 269 个实例

- `e5_size_id.json`（189）：E5 尺寸识别；权重 SHA256 复核一致 → 812 行 = 805 分析 + 7 冒烟
- `cinit_2x2.json`（80）、`cinit_e5.json`（189）：固定初始化确认实验（`torch.manual_seed(20000+seed)` 先于模型构造；`subset_protocol` 列含 `init_sha256` 前 16 位）；权重 SHA256 复核一致
- **合计 1081 行 = 1074 分析 + 7 冒烟**。WP5b v1 / v2、PU 已见轴承重分析、崩溃定义敏感性、初始化重建都是对已有权重的推理，不新增训练实例
- **初始化说明**：C-INIT 之前的干预实验（arm2x2、arm2x2_lt、ext_e1、ext_e2、wp2、e5）都是先构造模型、后设训练种子，同种子各臂的初始化只部分配对；逐实例重建见 `init_reconstruction.json` / `init_reconstruction_summary.json`
