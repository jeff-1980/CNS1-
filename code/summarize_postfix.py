import json, os, collections, numpy as np
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results") + "/"; A = R
P = json.load(open(A + "postfix_inference.json")); CW = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}
two = json.load(open(R + "cinit_fix_2x2.json")); armcov = {r["arm"]: r["coverage"] for r in two}
S = {}
print("=== A: no added noise (mean over instances) ===")
def grp(r):
    return (r["exp"], r["model"], "0.007 only" if r["exp"] == "2x2" and armcov[r["arm"]] == "007" else ("all sizes" if r["exp"] == "2x2" else r["set"]))
g = collections.defaultdict(list)
for r in P["A"]: g[grp(r)].append(r)
SA = {}
for k in sorted(g):
    rs = g[k]; acc = np.mean([r["acc"] for r in rs]); f1 = np.mean([r["macro_f1"] for r in rs]); rec = np.mean([r["recall"] for r in rs], 0)
    bys = {s: np.mean([r["recall_by_test_size"][s] for r in rs]) for s in rs[0]["recall_by_test_size"]}
    SA["|".join(k)] = dict(n=len(rs), acc=acc, macro_f1=f1, recall=rec.tolist(), by_size=bys)
    print(f"{k[0]:4s} {k[1]:6s} {k[2]:16s} n={len(rs):2d} acc {acc:.3f} F1 {f1:.3f} recall N/IR/OR/B {np.round(rec,3)} by size " + " ".join(f"{s}:{v:.2f}" for s, v in bys.items()))
# compare with +10 dB records for 2x2
for m in ("wdcnn", "drsn"):
    p10 = np.mean([r["observed"]["10.0"]["acc_mean"] for r in two if r["model"] == m]); cl = np.mean([r["acc"] for r in P["A"] if r["exp"] == "2x2" and r["model"] == m])
    print(f"2x2 {m}: no-noise acc {cl:.3f} vs +10 dB acc {p10:.3f}")
S["A"] = SA
print("\n=== B: B2 recalibrated on corrected 2x2 (clean-trained) ===")
SB = {}
for m in ("wdcnn", "drsn", "lstm", "cnn1d"):
    rs = [r for r in P["B"] if r["model"] == m]; d = {}
    for s in ("10.0", "4.0", "0.0", "-8.0"):
        b0 = [r["by_snr"][s]["B0"] for r in rs]; b2 = [r["by_snr"][s]["B2"] for r in rs]
        na = sum(x["N_ACCEPTED"] for x in b2); ne = sum(x["N_ERR_ACCEPTED"] for x in b2)
        d[s] = dict(FAR=np.mean([x["FAR"] for x in b0]), MFR=np.mean([x["MFR"] for x in b0]), cov=np.mean([x["COVERAGE"] for x in b2]),
                    cov_min=min(x["COVERAGE"] for x in b2), cov_max=max(x["COVERAGE"] for x in b2), risk_pooled=ne / na if na else None,
                    silent_median=float(np.nanmedian([x["SILENT"] for x in b2 if x["SILENT"] is not None])),
                    share_silent_ge_half=float(np.mean([x["SILENT"] >= 0.5 for x in b2 if x["SILENT"] is not None])))
    rc = np.nanmean([r["by_snr"]["-8.0"]["rc"] for r in rs], 0); covs = rs[0]["by_snr"]["-8.0"]["rc_cov"]; d["rc50"] = float(rc[covs.index(0.5)])
    SB[m] = d
    x = d["-8.0"]; print(f"{m:6s} −8dB: FAR {x['FAR']:.2f} MFR {x['MFR']:.2f} | B2 coverage {x['cov']:.2f} [{x['cov_min']:.2f},{x['cov_max']:.2f}] risk {x['risk_pooled']:.2f} silent med {x['silent_median']:.2f} share≥0.5 {x['share_silent_ge_half']:.2f} | risk@50% {d['rc50']:.2f}"
          f" || +4dB FAR {d['4.0']['FAR']:.2f} MFR {d['4.0']['MFR']:.2f}")
allp = [r for r in P["B"] if r["model"] in ("wdcnn", "drsn", "cnn1d")]
print("primary (cnn1d,wdcnn,drsn) −8 dB FAR", np.mean([r["by_snr"]["-8.0"]["B0"]["FAR"] for r in allp]), "MFR", np.mean([r["by_snr"]["-8.0"]["B0"]["MFR"] for r in allp]))
three = [r for r in P["B"] if r["model"] in ("wdcnn", "drsn", "lstm")]
print("wdcnn/drsn/lstm −8 dB FAR", np.mean([r["by_snr"]["-8.0"]["B0"]["FAR"] for r in three]), "MFR", np.mean([r["by_snr"]["-8.0"]["B0"]["MFR"] for r in three]))
sil = [r["by_snr"]["-8.0"]["B2"]["SILENT"] for r in P["B"] if r["model"] in ("cnn1d", "wdcnn", "drsn") and r["by_snr"]["-8.0"]["B2"]["SILENT"] is not None]
print("primary instances with ≥ half of errors accepted:", np.mean(np.array(sil) >= 0.5), len(sil))
S["B"] = SB
print("\n=== C: noise-preprocessing controls, −8 dB (2x2 corrected) ===")
SC = {}
mode = lambda L: collections.Counter(x["top"] for x in L).most_common(1)[0][0]
for m in ("wdcnn", "drsn", "lstm", "cnn1d"):
    for cov in ("007", "ALL"):
        rs = [r for r in P["C"] if r["model"] == m and armcov[r["arm"]] == cov]
        w = [mode(r["white"]) for r in rs]; z = [mode(r["renorm"]) for r in rs]; gn = [r["gain"]["top"] for r in rs]
        same_z = np.mean([a == b for a, b in zip(w, z)]); same_g = np.mean([a == b for a, b in zip(w, gn)])
        cz = np.mean([np.mean([x["conc"] for x in r["renorm"]]) for r in rs]); cg = np.mean([r["gain"]["conc"] for r in rs]); ag = np.mean([r["gain"]["acc"] for r in rs])
        SC[f"{m}|{cov}"] = dict(white=dict(collections.Counter(CW[x] for x in w)), renorm=dict(collections.Counter(CW[x] for x in z)), gain=dict(collections.Counter(CW[x] for x in gn)),
                               same_renorm=same_z, same_gain=same_g, conc_renorm=cz, conc_gain=cg, acc_gain=ag)
        print(f"{m:6s} {cov:3s} white {dict(collections.Counter(CW[x] for x in w))} | renorm {dict(collections.Counter(CW[x] for x in z))} same {same_z:.2f} conc {cz:.2f} | gain-only {dict(collections.Counter(CW[x] for x in gn))} same {same_g:.2f} conc {cg:.2f} acc {ag:.2f}")
S["C"] = SC
print("\n=== ΔC on corrected 2x2 ===")
SD = {}
for m in ("cnn1d", "wdcnn", "drsn", "lstm"):
    ir = lambda cov: np.mean([r["observed"]["-8.0"]["top_mode"] == 1 for r in two if r["model"] == m and r["coverage"] == cov])
    dr = lambda ratio: np.mean([r["observed"]["-8.0"]["top_mode"] == 1 for r in two if r["model"] == m and r["ratio"] == ratio])
    SD[m] = dict(dC=ir("007") - ir("ALL"), IR007=ir("007"), IRALL=ir("ALL"))
    print(f"{m:6s} P(IR|0.007) {ir('007'):.2f} P(IR|all) {ir('ALL'):.2f} ΔC {ir('007')-ir('ALL'):+.2f}")
print("ΔC mean over primary (cnn1d, wdcnn, drsn):", np.mean([SD[m]["dC"] for m in ("cnn1d", "wdcnn", "drsn")]))
# paired by seed (within seed and ratio)
for m in ("wdcnn", "drsn"):
    diffs = []
    for s in range(5):
        for ratio in ("R11", "R31"):
            a = [r for r in two if r["model"] == m and r["seed"] == s and r["ratio"] == ratio]
            v = {r["coverage"]: int(r["observed"]["-8.0"]["top_mode"] == 1) for r in a}; diffs.append(v["007"] - v["ALL"])
    print(f"{m} paired (seed×ratio) differences: {collections.Counter(diffs)}")
S["D"] = SD
json.dump(S, open(A + "postfix_summary.json", "w"), indent=1, default=float)
