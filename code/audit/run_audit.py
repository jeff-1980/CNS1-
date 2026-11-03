"""审计重跑入口。在 code/ 目录下运行：python -m audit.run_audit --adapter paperB:cnn1d --seeds 0,1,2,3,4 --out ... --ckpt-dir ...
先 --smoke，再 --check-isolation，最后正式跑。正式跑必须在 OSF 时间戳之后（results/PREREG_AUDIT_pilot.md）。"""
import argparse, json
from . import harness_core as H


def make_adapter(spec):
    kind, _, arg = spec.partition(":")
    if kind == "paperB":
        from .adapters.paperb import PaperBAdapter
        return PaperBAdapter(arg)
    raise SystemExit(f"unknown adapter {kind}; available: paperB:<model>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out"); ap.add_argument("--ckpt-dir")
    ap.add_argument("--eval-every-epoch", action="store_true", help="每个 epoch 都做完整测试评估（§3.4 溯源用，较慢）")
    ap.add_argument("--smoke", action="store_true", help="只构建适配器并打印数据 / 轮数，不训练")
    ap.add_argument("--check-isolation", type=int, metavar="EPOCHS", help="把轮数截到 EPOCHS，同一种子跑有/无探针两遍并比较最终权重哈希")
    a = ap.parse_args()
    ad = make_adapter(a.adapter)
    if a.smoke:
        print(ad.name, "n_epochs", ad.n_epochs, "budget_epoch", H.budget_epoch(ad.n_epochs), json.dumps(ad.meta())); return
    if a.check_isolation:
        ad.n_epochs = a.check_isolation
        r = H.check_isolation(ad, int(a.seeds.split(",")[0])); print(json.dumps(r))
        raise SystemExit(0 if r["ok"] else 1)
    if not a.out: raise SystemExit("--out required")
    H.run_many(ad, [int(s) for s in a.seeds.split(",")], a.out, ckpt_dir=a.ckpt_dir, eval_every_epoch=a.eval_every_epoch)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
