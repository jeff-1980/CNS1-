# 审计 harness（适配器版）

对应预注册 `results/PREREG_AUDIT_pilot.md`（冻结哈希见 `results/registry/prereg_audit_pilot_sha256.txt`）。E4/WP1 的 harness（`code/e4_paperB_harness.py`、`code/wp1_harness.py`）保持原样，不改。

| 文件 | 作用 |
|---|---|
| `harness_core.py` | 与框架无关的循环：原样驱动作者训练，在线跟踪 4 个选模协议（author / last / minvalloss / budget50），探针 epoch 完整评估，checkpoint 落盘与 SHA-256 |
| `torch_utils.py` | RNG（torch CPU+CUDA、numpy、random）与模型模式隔离；与落盘格式无关的权重哈希 |
| `adapters/paperb.py` | P1 paperB 适配器，逐行对应 E4 的 `run_one` |
| `provenance.py` | §3.4 溯源检验 d(epoch) |
| `run_audit.py` | 命令行入口 |
| `partI_reanalysis.py` | 第 I 部分（论文A）再分析：配对 t 检验判决改变、BH-FDR、符号翻转零假设下的反转期望、架构对层级、5 种子子集分布（§0b） |
| `test_harness_core.py` | 纯 numpy 的逻辑测试（含“隔离失效能被检出”） |

在 `code/` 目录下运行。数据与 paperB 代码路径和 E4 相同，由环境变量指定；jeff-work 上是：

```bash
export CNS_ROOT=~/论文CNS PAPERB_CODE=~/论文CNS/tmp/pbsrc_root CWRU16_DIR=~/论文CNS/tmp/cwru_16
```


```bash
python -m audit.test_harness_core                                   # 逻辑测试，无需 GPU
python -m audit.run_audit --adapter paperB:cnn1d --smoke
python -m audit.run_audit --adapter paperB:cnn1d --check-isolation 5 # 有/无探针最终权重哈希须相同
python -m audit.run_audit --adapter paperB:cnn1d --seeds 0,1,2,3,4 --out ../results/audit_paperB_cnn1d.json --ckpt-dir ../results/ckpt_audit
```

正式重跑只能在 OSF 时间戳之后。新对象（P2–P5）各写一个 `adapters/<name>.py`，实现 `harness_core.Adapter` 的方法，尽量直接调用作者代码；作者代码不算验证损失时在 `val_loss()` 里补算（在隔离中进行）。

## 第 II 部分（`ext/`，第三方代码原样重跑）

| 文件 | 作用 |
|---|---|
| `ext/run_p2_zhao.sh` / `run_p3_udtl.sh` / `run_p4_tfn.sh` | P2–P4 原样运行作者代码 |
| `ext/setup_p5_data.sh`、`ext/run_p5_pdm.sh`、`ext/pdm_hook.py` | P5：按 Kaggle 目录建 CWRU 软链接；运行 pdm-bench，钩子补记逐 epoch 的 test |
| `ext/epoch_protocols.py` | 从日志或记录算 4 个协议（P5 另加 best） |
| `ext/partII_summary.py` | H1、各协议报告值、结论变化与 §4 试点判定 |

P5 在 jeff-work 上：

```bash
export AUDIT_PY=~/audit_ext/venv/bin/python PDM_DIR=~/audit_ext/pdm-bench PDM_WORKSPACE=~/audit_ext/data/pdm OUT_DIR=~/audit_ext/runs
CWRU12_SRC=~/audit_ext/data/cwru_zhao bash audit/ext/setup_p5_data.sh
OUT_DIR=~/audit_ext/smoke MODELS=cnn1d SEEDS=41 EPOCHS=2 bash audit/ext/run_p5_pdm.sh   # 冒烟
bash audit/ext/run_p5_pdm.sh
python -m audit.ext.epoch_protocols pdm ~/audit_ext/runs/p5/*/seed* --out ~/audit_ext/results/p5_protocols.json
```
