"""§3.4 checkpoint 溯源检验：公布值与各 epoch 评估之间的距离 d(epoch)。

公布值格式（JSON）：{"<snr>": {"confusion_matrix": [[...]]} 或 {"accuracy": x}, ...}
距离：有混淆矩阵时 = 逐格平均绝对差（计数）；只有准确率时 = 各档平均绝对差。
d = 0（混淆矩阵逐格相同）记为 BITWISE 溯源。"""
import json, sys
import numpy as np


def distance(published, evaluated):
    keys = sorted(set(published) & set(evaluated), key=float)
    if not keys: return None, "no_common_keys"
    if all("confusion_matrix" in published[k] for k in keys):
        d = [np.abs(np.array(published[k]["confusion_matrix"]) - np.array(evaluated[k]["confusion_matrix"])).mean() for k in keys]
        return float(np.mean(d)), "confusion_matrix"
    d = [abs(float(published[k]["accuracy"]) - float(evaluated[k]["accuracy"])) for k in keys]
    return float(np.mean(d)), "accuracy"


def trace(published, run):
    """run = harness_core.run_one 的输出。返回逐 epoch 距离、最小距离与对应 epoch 区间、溯源等级。"""
    rows = []
    for ep, ev in sorted(run["eval_by_epoch"].items(), key=lambda kv: int(kv[0])):
        d, kind = distance(published, ev); rows.append(dict(epoch=int(ep), d=d, kind=kind))
    ds = [r["d"] for r in rows if r["d"] is not None]
    if not ds: return dict(rows=rows, grade="NO_DATA")
    dmin = min(ds); eps = [r["epoch"] for r in rows if r["d"] == dmin]
    grade = "BITWISE" if dmin == 0 and rows[0]["kind"] == "confusion_matrix" else "NEAREST"
    return dict(rows=rows, d_min=dmin, epochs_at_min=eps, grade=grade,
                evaluated_epochs=[r["epoch"] for r in rows])


if __name__ == "__main__":
    pub, runs = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    for r in runs:
        t = trace(pub, r)
        print(r["adapter"], "seed", r["seed"], t["grade"], "d_min", t.get("d_min"), "epochs", t.get("epochs_at_min"))
