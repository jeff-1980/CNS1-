"""正式阶段 §3.3：从 epoch_record 的记录算 5 个协议下的报告值。
  author    作者代码实际选中的 epoch（记录器 author_pick；未调用时 = 最后一轮）
  last      最后一轮
  best_rep  报告集准确率第一个严格最大值（反事实上界）
  minloss   选模集损失最小的一轮（无选模集或无损失时用报告集损失；并列取最早）
  budget50  第 ceil(0.5·E) 轮
稀疏测试（jeff 2026-10-24）：作者每隔若干轮才测一次时，第 k 个测试点当作第 k 轮（E = 测试点数，budget50 取中间测试点，
口径定义不变，输出另记原始轮次）；只有 1 个测试点的组不参加口径比较，只列入 report_only。逐轮测试的对象结果不变。
在 code/ 下运行：python -m audit.main.protocols RECORD_DIR --out OUT.json（RECORD_DIR 下所有 seed*.json）"""
import argparse, glob, json, math, os


def protocols(r):
    eps = sorted(map(int, r["epochs"]))
    sparse = eps != list(range(1, len(eps) + 1))
    if sparse and len(eps) == 1: return None   # single test point: report only, no protocol comparison
    rep = [r["epochs"][str(e)][r["report_split"]] for e in eps]
    sel = r["select_split"]
    lossrow = [r["epochs"][str(e)].get(sel) if sel else None for e in eps]
    if not all(x and x.get("loss") is not None for x in lossrow): lossrow, loss_src = rep, r["report_split"]
    else: loss_src = sel
    n = len(eps)
    best, b_ep = -math.inf, None
    for i, x in enumerate(rep):
        if x["acc"] > best: best, b_ep = x["acc"], i
    ok = lambda v: v is not None and not math.isnan(v)
    l_ep = min(range(n), key=lambda i: (math.inf if lossrow[i]["loss"] is None else lossrow[i]["loss"], i))
    if not any(ok(x["loss"]) for x in lossrow): l_ep = None   # no loss recorded at all: minloss undefined
    a_ep = (eps.index(r["author_epoch"]) if sparse else r["author_epoch"] - 1) if r["author_epoch"] else n - 1
    pick = dict(author=a_ep, last=n - 1, best_rep=b_ep, minloss=l_ep, budget50=max(1, math.ceil(0.5 * n)) - 1)
    out = dict(obj=r["obj"], group=r["group"], seed=r["seed"], n_epochs=n, minloss_split=loss_src,
               author_check=r.get("author_check"), acc_curve=[x["acc"] for x in rep],
               protocols={k: dict(epoch=None if e is None else e + 1, acc=None if e is None else rep[e]["acc"])
                          for k, e in pick.items()})
    if sparse:
        out["sparse_eval_epochs"] = eps
        for k, e in pick.items(): out["protocols"][k]["orig_epoch"] = None if e is None else eps[e]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("record_dir"); ap.add_argument("--out", required=True); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.record_dir, "**", "seed*.json"), recursive=True))
    recs = [json.load(open(f)) for f in files]
    out = [o for o in map(protocols, recs) if o is not None]
    only = [dict(obj=r["obj"], group=r["group"], seed=r["seed"], author_check=r.get("author_check"),
                 epochs=sorted(map(int, r["epochs"]))) for r in recs if protocols(r) is None]
    if only:
        json.dump(only, open(a.out.replace(".json", "_report_only.json"), "w"), indent=1)
        print(f"{len(only)} single-test-point runs listed as report only")
    json.dump(out, open(a.out, "w"), indent=1)
    bad = [f"{o['group']}/seed{o['seed']}" for o in out if o["author_check"] and not o["author_check"]["matches"]]
    print(f"{len(out)} runs; author check failed: {bad or 'none'}")


if __name__ == "__main__":
    main()
