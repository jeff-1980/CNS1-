"""按 PREREG_E5_size_identification.md 分析。运行时校验预注册哈希。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, hashlib, sys, collections, pathlib
import numpy as np
R = pathlib.Path((_ROOT + "/results"))
if hashlib.sha256((R / "PREREG_E5_size_identification.md").read_bytes()).hexdigest() != (R / "registry/prereg_e5_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
CLS = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}; SETS = ["S007", "S014", "S021", "S007+014", "S007+021", "S014+021", "S_ALL"]; MULTI = ["S007+014", "S007+021", "S014+021"]
M = ["wdcnn", "drsn", "lstm"]; DRAWS = [0, 1, 2]
d = json.load(open(R / "e5_size_id.json"))
assert len({(r["set"], r["draw"], r["model"], r["seed"]) for r in d}) == len(d)

def label(m, snr="-8.0", need=False):
    def isIR(r):
        o = r["observed"][snr]
        if need and o["conc_mean"] < 0.90: return None
        return int(o["top_mode"] == 1)
    pdraw = {}
    for s in SETS:
        for dr in DRAWS:
            v = [isIR(r) for r in d if r["model"] == m and r["set"] == s and r["draw"] == dr]; v = [x for x in v if x is not None]
            pdraw[(s, dr)] = (np.mean(v) if v else np.nan, len(v))
    P = {s: float(np.nanmean([pdraw[(s, dr)][0] for dr in DRAWS])) for s in SETS}
    A, B = P["S007"], P["S_ALL"]
    rng_ = {s: np.nanmax([pdraw[(s, dr)][0] for dr in DRAWS]) - np.nanmin([pdraw[(s, dr)][0] for dr in DRAWS]) for s in SETS}
    if A - B < 0.5: lab = "NO_REPLICATION"
    elif rng_["S007"] >= 2 / 3 - 1e-9 or rng_["S_ALL"] >= 2 / 3 - 1e-9: lab = "FILE_DEPENDENT"
    elif P["S014"] <= A - 0.5 and P["S021"] <= A - 0.5: lab = "ANY_LARGER_SIZE"
    elif P["S014"] > A - 0.5 and P["S021"] > A - 0.5 and all(P[s] <= A - 0.5 for s in MULTI): lab = "DIVERSITY"
    elif (P["S014"] <= A - 0.5) != (P["S021"] <= A - 0.5): lab = "SIZE_SPECIFIC"
    else: lab = "MIXED"
    return lab, P, {f"{s}|d{dr}": v for (s, dr), v in pdraw.items()}, rng_

out = dict(n=len(d), expected=189, per_arch={})
for m in M:
    n_m = sum(r["model"] == m for r in d)
    if n_m < 63: out["per_arch"][m] = dict(label="NOT_COMPLETE", n=n_m); print(m, "NOT_COMPLETE", n_m); continue
    lab, P, pdraw, rng_ = label(m); lab6, P6, _, _ = label(m, "-6.0"); lab9, P9, _, _ = label(m, need=True)
    dist = {s: dict(collections.Counter(CLS[r["observed"]["-8.0"]["top_mode"]] for r in d if r["model"] == m and r["set"] == s)) for s in SETS}
    conc = {s: [round(min(r["observed"]["-8.0"]["conc_mean"] for r in d if r["model"] == m and r["set"] == s), 2), round(max(r["observed"]["-8.0"]["conc_mean"] for r in d if r["model"] == m and r["set"] == s), 2)] for s in SETS}
    acc10 = {s: round(float(np.mean([r["observed"]["10.0"]["acc_mean"] for r in d if r["model"] == m and r["set"] == s])), 3) for s in SETS}
    out["per_arch"][m] = dict(label=lab, P_IR=P, p_IR_by_draw=pdraw, draw_range=rng_, dominant_dist=dist, conc_range=conc, acc10=acc10, label_m6dB=lab6, label_conc090=lab9, P_IR_conc090=P9)
    print(f"\n{m}: {lab}  (−6 dB: {lab6}; conc≥0.90: {lab9})")
    for s in SETS: print(f"   {s:9s} P_IR={P[s]:.2f} draws={[round(pdraw[f'{s}|d{dr}'][0], 2) for dr in DRAWS]} dist={dist[s]} conc={conc[s]} acc+10={acc10[s]}")
labs = [out["per_arch"][m]["label"] for m in M]; c = collections.Counter(labs); top, k = c.most_common(1)[0]
ok = {"ANY_LARGER_SIZE", "DIVERSITY", "SIZE_SPECIFIC"}
out["manuscript_rule"] = (f"CLAIM:{top}" if (k >= 2 and top in ok) else "NARROW_TO_FILE_COMPOSITION")
print("\n>>> 逐架构:", labs, "→ 稿件:", out["manuscript_rule"])
json.dump(out, open(R / "e5_verdict.json", "w"), indent=1, default=float, ensure_ascii=False)
