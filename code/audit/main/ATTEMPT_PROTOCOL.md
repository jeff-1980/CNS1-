# 正式阶段：逐个对象的试跑规程

依据：预注册 `results/PREREG_AUDIT_main.md` §2、§3（OSF 2tqv6）。对象与顺序：`results/main_frame/2026-10-19/attempt_order.csv`；代码已按固定 commit 取到 `~/audit_ext/main_objects/NN_owner__repo/`。一次只做一个对象，做完一个汇报一次再做下一个。

## 每个对象的步骤

1. **登记（跑之前提交）**：写 `results/main_objects/registry/NN_owner__repo.md`，内容：
   - 论文 DOI、仓库与 commit、对应的头条表格（表号与表里的数）；
   - 方法组（最多 4 个，超过时取论文正文强调的 4 个）；
   - 每组的配置（数据集、划分、迁移任务、轮数、超参，均用作者默认）；
   - 种子方案：作者有种子机制用 0–9，没有就做 10 次不设种子的重复；
   - 作者的选模规则（读代码得出：保存 / 报告哪一轮，用哪个划分）；选模集与报告集各是什么；
   - 数据来源路径；CWRU 对象写明读的是哪个 `.mat` 变量键（99.mat 问题只记录不修复）。
   先提交并推送这份登记，再开始任何训练。
2. **环境**：在 `~/audit_ext/envs/NN`（venv 或 conda）里装依赖，不动审计主环境和其他项目的环境。从开始修环境起计时，**最多 2 小时**；超时记为失败（原因 `env_timeout`）。
3. **补丁**：除了让代码在当前依赖上能跑的最小改动（如 `.view`→`.reshape`），只加逐 epoch 记录：
   - 用 `audit.main.epoch_record.Recorder`，每轮结束时记报告集（必记）和选模集（有就记）的准确率与损失；
   - 作者本来就每轮评估报告集的，直接记作者算出的数；需要额外评估的，包在 `audit.torch_utils.isolate([model], device)` 里，用作者自己的评估函数；
   - 调 `author_pick(epoch)` 标出作者代码实际选中的那一轮，结束时 `finish(作者打印或返回的最终数)`。
   - 所有改动存成 `results/main_objects/patches/NN_owner__repo.diff`（`git diff`），逐条写进 `results/DEVIATIONS_AUDIT.md`。
4. **冒烟**：每组 1 个种子跑 2 轮，确认记录文件生成、`author_check.matches` 为真、显存与内存正常（WSL 内存远低于 35 GB）。
5. **时间估计**：由冒烟得出单次完整训练的时长。单次超过 2 GPU 小时记为排除（原因 `E6_runtime`）。总时长（组数 × 10 × 单次）超过 3 小时的，先汇报估计再等确认。
6. **正式跑**（缺失数据规则见 OSF 表单补充 `results/registry/prereg_audit_main_osf_form_additions.md` 第 1 条：崩溃的运行用同一种子**重跑一次**，再失败才记为缺失；首次崩溃仍记录并报告；组内完整种子 ≥ 5 才进 H-B 分析）：`AUDIT_RECORD_DIR=~/audit_ext/main_records` 下逐组逐种子跑。一次只占用一个 GPU 任务。跑完运行 `python -m audit.main.protocols ~/audit_ext/main_records/NN --out ../results/main_objects/protocols/NN.json`。
7. **记结果**：在 `results/main_objects/attempts.csv` 追加一行：`attempt_order,full_name,status,reason,env_minutes,smoke_ok,runs_done,author_check_all,notes`。
   - `status` = `runnable` / `failed` / `excluded`；`reason` 用 `env_timeout`、`data_unavailable`、`code_broken`、`no_training_script`、`E1_not_author`、`E5_private_data`、`E6_runtime`、`author_check_failed` 之一，或简短说明。
   - pending 对象核实不通过时记 `excluded` 和对应原因码。
8. **提交**：登记、补丁 diff、`protocols/NN.json`、`attempts.csv`、偏离记录一并提交推送到 `claude/project-thread-ogrtsz`。原始记录目录与模型权重不提交。

## 不做的事

- 不修复作者代码里的泄漏或错误，只记录。
- 不改数据、切分、轮数和超参。
- 不联系作者。
- 下载数据前先核对 C 盘真实剩余（`df /mnt/c`），保证剩余 ≥ 10 GB。
