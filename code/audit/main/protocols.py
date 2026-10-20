"""正式阶段 §3.3：从 epoch_record 的记录算 5 个协议下的报告值。
  author    作者代码实际选中的 epoch（记录器 author_pick；未调用时 = 最后一轮）
  last      最后一轮
  best_rep  报告集准确率第一个严格最大值（反事实上界）
  minloss   选模集损失最小的一轮（无选模集或无损失时用报告集损失；并列取最早）
  budget50  第 ceil(0.5·E) 轮
在 code/ 下运行：python -m audit.main.protocols RECORD_DIR --out OUT.json（RECORD_DIR 下所有 seed*.json）"""
import argparse, glob, json, math, os


def protocols(r):
    eps = sorted(map(int, r["epochs"]))
    if eps != list(range(1, len(eps) + 1)): raise ValueError(f"{r['obj']}/{r['group']}/seed{r['seed']}: epochs not 1..E: {eps[:5]}")
    rep = [r["epochs"][str(e)][r["report_split"]] for e in eps]
    sel = r["select_split"]
    lossrow = [r["epochs"][str(e)].get(sel) if sel else None for e in eps]
    if not all(x and x.get("loss") is not None for x in lossrow): lossrow, loss_src = rep, r["report_split"]
    else: loss_src = sel
    n = len(eps)
    best, b_ep = -math.inf, None
    for i, x in enumerate(rep):
        if x["acc"] > best: best, b_ep = x["acc"], i
    l_ep = min(range(n), key=lambda i: (math.inf if lossrow[i]["loss"] is None else lossrow[i]["loss"], i))
    a_ep = (r["author_epoch"] or n) - 1
    pick = dict(author=a_ep, last=n - 1, best_rep=b_ep, minloss=l_ep, budget50=max(1, math.ceil(0.5 * n)) - 1)
    return dict(obj=r["obj"], group=r["group"], seed=r["seed"], n_epochs=n, minloss_split=loss_src,
                author_check=r.get("author_check"), acc_curve=[x["acc"] for x in rep],
                protocols={k: dict(epoch=e + 1, acc=rep[e]["acc"]) for k, e in pick.items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("record_dir"); ap.add_argument("--out", required=True); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.record_dir, "**", "seed*.json"), recursive=True))
    out = [protocols(json.load(open(f))) for f in files]
    json.dump(out, open(a.out, "w"), indent=1)
    bad = [f"{o['group']}/seed{o['seed']}" for o in out if o["author_check"] and not o["author_check"]["matches"]]
    print(f"{len(out)} runs; author check failed: {bad or 'none'}")


if __name__ == "__main__":
    main()
