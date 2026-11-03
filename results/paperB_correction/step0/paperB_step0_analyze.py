"""paperB step 0: per-cell summary (published / orig re-evaluated / x099) in the format of paperB's cell_stats.csv,
and the comparisons the correction needs (collapsed cells, attractor class, accuracy, Normal recall, rescue)."""
import csv, json, sys
import numpy as np

out = sys.argv[1]; published_cells = sys.argv[2]
rows = json.load(open(f"{out}/runs.json"))
NAMES = ["Normal", "Inner Race", "Outer Race", "Ball"]
cells = {}
for r in rows:
    cells.setdefault((r["group"], r["model"], r["snr"]), []).append(r)


def stats(rs, key):
    acc = np.array([r[f"{key}_acc"] for r in rs])
    cms = [np.array(r[f"{key}_cm"]) for r in rs]
    conc = np.array([cm.sum(0).max() / cm.sum() for cm in cms])
    top = [int(cm.sum(0).argmax()) for cm in cms]
    mode = max(set(top), key=top.count)
    nrec = np.array([cm[0, 0] / cm[0].sum() for cm in cms])
    return dict(mean_acc=acc.mean(), std_acc=acc.std(ddof=1), mean_conc=conc.mean(), top=NAMES[mode],
                top_consistent=len(set(top)) == 1, collapsed=bool(conc.mean() >= 0.90), normal_recall=nrec.mean())


table = []
for (g, m, snr), rs in sorted(cells.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
    row = dict(group=g, model=m, snr=snr, n_seeds=len(rs))
    for k in ("published", "orig", "x099"):
        for f, v in stats(rs, k).items():
            row[f"{k}_{f}"] = v
    table.append(row)
with open(f"{out}/cells.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

# check our published-side stats against paperB's own cell_stats.csv
pub = {(r["group"], r["model"], float(r["snr"])): r for r in csv.DictReader(open(published_cells)) if r["dataset"] == "CWRU"}
match = [abs(float(pub[(t["group"], t["model"], t["snr"])]["mean_acc"]) - t["published_mean_acc"]) < 1e-6 and
         abs(float(pub[(t["group"], t["model"], t["snr"])]["mean_conc"]) - t["published_mean_conc"]) < 1e-6 and
         (pub[(t["group"], t["model"], t["snr"])]["collapsed_conc0.90_main"] == "True") == t["published_collapsed"]
         for t in table if (t["group"], t["model"], t["snr"]) in pub]
S = dict(n_cells=len(table), cells_matched_to_paperB_cell_stats=f"{sum(match)}/{len(match)}")

for k in ("published", "orig", "x099"):
    S[f"collapsed_{k}"] = sorted(f"{t['group']}/{t['model']}@{t['snr']:g}:{t[k + '_top']}" for t in table if t[f"{k}_collapsed"])
S["n_collapsed"] = {k: len(S[f"collapsed_{k}"]) for k in ("published", "orig", "x099")}
S["collapsed_changed_orig_vs_x099"] = sorted(set(S["collapsed_orig"]) ^ set(S["collapsed_x099"]))
S["attractor_changed_orig_vs_x099"] = [f"{t['group']}/{t['model']}@{t['snr']:g}: {t['orig_top']} -> {t['x099_top']}"
                                       for t in table if (t["orig_collapsed"] or t["x099_collapsed"]) and t["orig_top"] != t["x099_top"]]
d = np.array([t["x099_mean_acc"] - t["orig_mean_acc"] for t in table])
S["acc_change_x099_minus_orig_pp"] = dict(median=float(np.median(d) * 100), min=float(d.min() * 100), max=float(d.max() * 100),
                                          n_abs_gt_1pp=int((np.abs(d) > 0.01).sum()), n_abs_gt_5pp=int((np.abs(d) > 0.05).sum()))
dn = np.array([t["x099_normal_recall"] - t["orig_normal_recall"] for t in table])
S["normal_recall_change_pp"] = dict(median=float(np.median(dn) * 100), min=float(dn.min() * 100), max=float(dn.max() * 100))
e = np.array([t["orig_mean_acc"] - t["published_mean_acc"] for t in table])
S["reeval_orig_minus_published_pp"] = dict(median_abs=float(np.median(np.abs(e)) * 100), max_abs=float(np.abs(e).max() * 100))
big = sorted(table, key=lambda t: -abs(t["x099_mean_acc"] - t["orig_mean_acc"]))[:10]
S["largest_acc_changes"] = [f"{t['group']}/{t['model']}@{t['snr']:g}: {t['orig_mean_acc']*100:.1f} -> {t['x099_mean_acc']*100:.1f} "
                            f"(Normal recall {t['orig_normal_recall']*100:.1f} -> {t['x099_normal_recall']*100:.1f})" for t in big]

# rescue (NA+ main_awgn_v2 minus NA-), on the cells collapsed under NA-
T = {(t["group"], t["model"], t["snr"]): t for t in table}
resc = []
for t in table:
    if t["group"] in ("cwru_clean_v2",) or (t["group"] == "ablation_v3" and t["model"] == "mamba2_no_noise_train"):
        arch = "mamba2" if t["group"] == "ablation_v3" else t["model"]
        p = T.get(("main_awgn_v2", arch, t["snr"]))
        if p and (t["published_collapsed"] or t["x099_collapsed"]):
            resc.append(dict(cell=f"{arch}@{t['snr']:g}", published=(p["published_mean_acc"] - t["published_mean_acc"]) * 100,
                             orig=(p["orig_mean_acc"] - t["orig_mean_acc"]) * 100, x099=(p["x099_mean_acc"] - t["x099_mean_acc"]) * 100))
S["rescue_pp_on_collapsed_cells"] = dict(n=len(resc), **{k: float(np.mean([r[k] for r in resc])) for k in ("published", "orig", "x099")})
S["rescue_cells"] = resc
json.dump(S, open(f"{out}/summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in S.items() if k not in ("rescue_cells",)}, indent=1, default=float))
