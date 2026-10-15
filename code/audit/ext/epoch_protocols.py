"""第 II 部分：从作者代码的逐 epoch 记录算 4 个选模协议的报告值（预注册 §3.3；执行方案 audit/partII_rerun_plan）。

适用于选模划分 = 报告划分的代码库（P2 Zhao benchmark、P3 UDTL、P4 TFN）：该划分的逐 epoch 准确率/损失
就足以给出每个协议下“论文会报告的数”。P5 pdm-bench 只有最后一轮的 test，单独处理。
  author     第一个严格最大值（P2/P3 的 `epoch_acc > best_acc`；P4 同）
  last       最后一轮
  minvalloss 报告划分上损失最小的 epoch（严格下降取最早）
  budget50   第 ceil(0.5 × 总轮数) 轮（1 起计）
在 code/ 下运行：python -m audit.ext.epoch_protocols {zhao|udtl|tfn} RUN_DIR [RUN_DIR ...] --out OUT.json"""
import argparse, glob, json, math, os, re

LOG_RE = re.compile(r"Epoch: (\d+) (\S+)-Loss: ([\d.naninf]+) \S+-Acc: ([\d.naninf]+)")


def parse_log(path, phase):
    """Zhao / UDTL 的 training.log：返回按 epoch 排序的 (acc, loss)。phase = 'val'（Zhao）或 'target_val'（UDTL）。"""
    rows = {}
    for line in open(path, errors="replace"):
        m = LOG_RE.search(line)
        if m and m.group(2) == phase:
            rows[int(m.group(1))] = (float(m.group(4)), float(m.group(3)))
    eps = sorted(rows)
    if eps != list(range(len(eps))): raise ValueError(f"{path}: epochs not contiguous from 0: {eps[:5]}…")
    return [rows[e][0] for e in eps], [rows[e][1] for e in eps]


def parse_tfn(path):
    """TFN 的 Records.mat：val_acc / val_loss。"""
    from scipy.io import loadmat
    r = loadmat(path, squeeze_me=True)
    return [float(x) for x in r["val_acc"]], [float(x) for x in r["val_loss"]]


def protocols(acc, loss):
    n = len(acc)
    if n == 0: raise ValueError("empty record")
    best, a_ep = -math.inf, None
    for i, x in enumerate(acc):
        if x > best: best, a_ep = x, i
    l_ep = min(range(n), key=lambda i: (loss[i], i))
    b_ep = max(1, math.ceil(0.5 * n)) - 1
    pick = dict(author=a_ep, last=n - 1, minvalloss=l_ep, budget50=b_ep)
    return {k: dict(epoch=e + 1, acc=acc[e], loss=loss[e]) for k, e in pick.items()}


def find_record(run_dir, kind):
    pat = {"zhao": "**/training.log", "udtl": "**/training.log", "tfn": "**/Records.mat"}[kind]
    hits = sorted(glob.glob(os.path.join(run_dir, pat), recursive=True))
    if len(hits) != 1: raise ValueError(f"{run_dir}: expected 1 record, found {len(hits)}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["zhao", "udtl", "tfn"]); ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for d in a.run_dirs:
        rec = find_record(d, a.kind)
        acc, loss = parse_tfn(rec) if a.kind == "tfn" else parse_log(rec, "val" if a.kind == "zhao" else "target_val")
        out.append(dict(run_dir=os.path.abspath(d), record=rec, n_epochs=len(acc), acc_curve=acc, loss_curve=loss,
                        protocols=protocols(acc, loss)))
        p = out[-1]["protocols"]
        print(os.path.basename(d.rstrip("/")), " ".join(f"{k}=ep{v['epoch']}:{v['acc']:.4f}" for k, v in p.items()))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
