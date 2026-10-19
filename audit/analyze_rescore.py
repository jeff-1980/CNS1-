import json, collections, sys
CLS = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}
d = json.load(open("rescore_leakfree.json"))
print("instances", len(d), "| orig re-inference reproduces archive:", sum(r["orig_reproduces_archive"] for r in d), "/", len(d))
out = {"n": len(d), "reproduce": sum(r["orig_reproduces_archive"] for r in d), "cells": {}}
for src in ("arm2x2.json", "cinit_2x2.json", "arm2x2_lt.json"):
    for m in ("wdcnn", "drsn", "lstm", "cnn1d", "transformer"):
        for cov in ("C007", "CALL"):
            rs = [r for r in d if r["source"] == src and r["model"] == m and r["arm"].startswith(cov)]
            if not rs: continue
            cell = {}
            for t in ("orig", "N100", "N099"):
                c = collections.Counter(CLS[r[t]["-8.0"]["top"]] for r in rs)
                cell[t] = dict(dominant=dict(c), events=sum(r[t]["-8.0"]["conc"] >= 0.9 for r in rs),
                               normal_recall_10dB=round(sum(r[t]["10.0"]["dist"][0] for r in rs) / 1, 0))
            same = sum(r["orig"]["-8.0"]["top"] == r["N100"]["-8.0"]["top"] == r["N099"]["-8.0"]["top"] for r in rs)
            out["cells"][f"{src}|{m}|{cov}"] = dict(n=len(rs), same_dominant_all_three=same, **cell)
            print(f"{src:15s} {m:11s} {cov}: n={len(rs):2d} same dominant across tests {same}/{len(rs)} | orig {cell['orig']['dominant']} ev{cell['orig']['events']} | N100 {cell['N100']['dominant']} ev{cell['N100']['events']} | N099 {cell['N099']['dominant']} ev{cell['N099']['events']}")
# clean (+10 dB) Normal recall: share of Normal test windows predicted Normal, orig vs unseen
def nrec(r, t):
    dist = r[t]["10.0"]["dist"]; return None
acc = collections.defaultdict(list)
for r in d:
    for t in ("orig", "N100", "N099"): acc[(r["model"], t)].append(r[t]["10.0"]["acc"])
print("+10 dB overall accuracy (mean):", {f"{m}|{t}": round(sum(v) / len(v), 3) for (m, t), v in sorted(acc.items())})
out["acc10"] = {f"{m}|{t}": sum(v) / len(v) for (m, t), v in acc.items()}
json.dump(out, open("rescore_leakfree_summary.json", "w"), indent=1)
