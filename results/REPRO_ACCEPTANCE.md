# 复现包验收记录（2026-09-27）

> 包目录：`release/CNS1/`（由 `release/build_release.py` 从工作目录组装；源文件不改动）。验收在独立副本 `release/_check/` 中进行，环境 `cns1-repro`（python 3.11.16，numpy 2.4.6，matplotlib 3.11.2，scipy 1.17.1，无 torch），以 `env -i` 清空环境变量运行。

## 1. 生成入口能否跑通

| 检查 | 结果 |
|---|---|
| `code/mssp_figs.py` 重新生成 Fig. 1–5、S1 | 6/6 张，与投稿版逐像素比较，差异 > 0.02 的像素数均为 **0** |
| `code/build_supplement.py` 重新生成 Table S1–S7 | 与投稿版 `MSSP_SUPPLEMENT.md` **逐字节相同** |
| 预注册分析 `analyze_2x2 / e5 / cinit / wp2 / wp5 / b / wp5b2`、`wp5_f1_paired` | 全部退出码 0，判定与存档一致：COVERAGE、ANY_LARGER_SIZE ×3、NARROW、NO_SHIFT、NO_CONSEQUENCE / SILENT_UNDER_CONFIDENCE、KILLED |
| `analyze_ext.py e1 e2 e3` | `ext_verdict.json` 与存档 SHA-256 相同 |
| `analyze_e4`、`analyze_wp1`（需参考基准的公开结果包，`PAPERB_RESULTS`） | 退出码 0 |
| `collapse_defs.py` | 重写后与存档相同，只少 `non_unanimous_instances` 字段（该字段由 `s2_two_instances.py` 重推理后补入，README 已注明） |
| 训练脚本 `--smoke`（arm2x2、cinit、pu_wp2、e4_paperB_harness、wp1_harness）| 全部通过（主环境，GPU） |
| 重推理 `wp5b2_selective.py --smoke`（用存档权重） | 8 个实例的结果与存档 WP5b2 记录**完全一致** |

## 2. 路径与依赖

- 包内代码的本机绝对路径全部改为环境变量（`CNS_ROOT`、`CWRU40_DIR`、`CWRU16_DIR`、`PU_DIR`、`JNU_DIR`、`PAPERB_CODE`、`PAPERB_RESULTS`），默认值相对仓库根目录；包内 `.py` / `.sh` 已无 `/home/jeffwork` 或 `/mnt/c`
- 作图原依赖会话内加载的样式函数，已抽成 `code/figstyle.py`（22 项 rcParams 与原样一致，逐像素比较为证）
- 依赖：`requirements-figures.txt`（作图、补充表、分析只需 numpy / scipy / matplotlib）；`environment.yml`（训练：torch 2.13.0+cu130；Mamba 另需 torch 2.5.1+cu124、mamba-ssm 2.2.4）
- 权重：`code/verify_weights.py` 按注册表校验 **1051 个权重文件，0 个不符、0 个缺失**；138 行早期筛选实例未保留权重，注册表标为 `missing`
- 结果 JSON 中保留生成时的本机绝对路径（11 个文件），脚本不使用这些字段；README 已说明

## 3. 数据获取说明

- README 列出 CWRU、PU、JNU 的来源、文件命名与目录要求；原始数据不再分发
- `code/prepare_data.py` 按 `raw_hashes.csv` 校验：CWRU 40/40、JNU 12/12、PU 2240/2240 个文件，**0 缺失、0 哈希不符**；并生成 CWRU 16 文件子集（符号链接）

## 4. 未覆盖 / 待办

- 完整重训未做（GPU 时间以天计，且训练非逐位确定）；只验证了 smoke 与重推理
- 参考基准（Wang and Tang 2026）的代码包与结果包不随本仓库分发，参考协议相关脚本需用户自行取得
- 权重约 1.1 GB，需另存（GitHub Release 附件单个 < 2 GB，或 Zenodo），README 中 `[WEIGHTS LINK]` 待填
- §3.7 的仓库地址、版本号与 DOI 待上传后填写
