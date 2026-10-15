# 审计 harness（适配器版）

对应预注册 `results/PREREG_AUDIT_pilot.md`（冻结哈希见 `results/registry/prereg_audit_pilot_sha256.txt`）。E4/WP1 的 harness（`code/e4_paperB_harness.py`、`code/wp1_harness.py`）保持原样，不改。

| 文件 | 作用 |
|---|---|
| `harness_core.py` | 与框架无关的循环：原样驱动作者训练，在线跟踪 4 个选模协议（author / last / minvalloss / budget50），探针 epoch 完整评估，checkpoint 落盘与 SHA-256 |
| `torch_utils.py` | RNG（torch CPU+CUDA、numpy、random）与模型模式隔离；与落盘格式无关的权重哈希 |
| `adapters/paperb.py` | P1 paperB 适配器，逐行对应 E4 的 `run_one` |
| `provenance.py` | §3.4 溯源检验 d(epoch) |
| `run_audit.py` | 命令行入口 |
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
