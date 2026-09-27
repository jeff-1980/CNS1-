"""按 PREREG_E4_paperB_checkpoint.md 分析。运行时校验预注册哈希。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, hashlib, sys, glob, collections, pathlib
import numpy as np
RES = pathlib.Path((_ROOT + "/results"))
PRE = RES / "PREREG_E4_paperB_checkpoint.md"
if hashlib.sha256(PRE.read_bytes()).hexdigest() != (RES / "registry/prereg_e4_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
PB = (_PAPERB_RESULTS + "/cwru/cwru_clean_v2")
SNRS = [f"{s:.1f}" for s in (-8, -6, -4, -2, 0, 2, 4, 6, 8, 10)]
CLS = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}
M = ("cnn1d", "wdcnn", "drsn")
pb = {m: {json.load(open(f))["seed"]: json.load(open(f)) for f in glob.glob(f"{PB}/{m}/seed*_summary.json")} for m in M}
pb_curve = {m: np.array([np.mean([pb[m][s]["snr_sweep"][k]["accuracy"] for s in pb[m]]) for k in SNRS]) for m in M}
pb_top8 = {m: dict(collections.Counter(CLS[int(np.array(pb[m][s]["snr_sweep"]["-8.0"]["confusion_matrix"]).sum(0).argmax())] for s in pb[m])) for m in M}
d = json.load(open(RES / "e4_paperB_harness.json")); assert len(d) == 15, len(d)
curve = lambda sw: np.array([sw[k]["accuracy"] for k in SNRS])
dist = lambda sw, ref: float(np.mean(np.abs(curve(sw) - ref)))
rows = []
for r in d:
    m, s = r["model"], r["seed"]
    first1 = next((i + 1 for i, v in enumerate(r["val_curve"]) if v >= 1.0), None)
    rows.append(dict(model=m, seed=s, best_epoch=r["best_epoch"], first_val_1=first1, best_val=r["best_val_acc"],
                     d_best=dist(r["best"], pb_curve[m]), d_final=dist(r["final"], pb_curve[m]),
                     d_best_same_seed=dist(r["best"], curve(pb[m][s]["snr_sweep"])), d_final_same_seed=dist(r["final"], curve(pb[m][s]["snr_sweep"])),
                     top8_best=CLS[r["best"]["-8.0"]["top"]], top8_final=CLS[r["final"]["-8.0"]["top"]],
                     conc8_best=round(r["best"]["-8.0"]["conc"], 3), conc8_final=round(r["final"]["-8.0"]["conc"], 3),
                     d_by_epoch={e: dist(sw, pb_curve[m]) for e, sw in r["sweeps_by_epoch"].items()}))
n_early = sum(r["best_epoch"] <= 5 for r in rows); n_late = sum(r["best_epoch"] > 5 for r in rows)
n_bcloser = sum(r["d_best"] < r["d_final"] for r in rows); n_fcloser = sum(r["d_final"] < r["d_best"] for r in rows)
if n_late >= 12: label = "NOT_EARLY"
elif n_early >= 12 and n_bcloser >= 12: label = "EARLY_CKPT"
elif n_fcloser >= 12: label = "FINAL_LIKE"
else: label = "INDETERMINATE"
print(f"best_epoch≤5: {n_early}/15 | d_best<d_final: {n_bcloser}/15 | d_final<d_best: {n_fcloser}/15 → {label}")
per = {}
for m in M:
    rs = [r for r in rows if r["model"] == m]
    per[m] = dict(best_epochs=[r["best_epoch"] for r in rs], first_val_1=[r["first_val_1"] for r in rs],
                  d_best=[round(r["d_best"], 3) for r in rs], d_final=[round(r["d_final"], 3) for r in rs],
                  top8_best=dict(collections.Counter(r["top8_best"] for r in rs)), top8_final=dict(collections.Counter(r["top8_final"] for r in rs)),
                  top8_paperB=pb_top8[m],
                  d_by_epoch_mean={e: round(float(np.mean([r["d_by_epoch"][e] for r in rs])), 3) for e in rs[0]["d_by_epoch"]})
    print(f"\n{m}: best_epoch={per[m]['best_epochs']} first_val=1 at {per[m]['first_val_1']}")
    print(f"   d_best={per[m]['d_best']}  d_final={per[m]['d_final']}")
    print(f"   −8 dB target best={per[m]['top8_best']} final={per[m]['top8_final']} paperB={per[m]['top8_paperB']}")
    print(f"   d(epoch) mean: {per[m]['d_by_epoch_mean']}")
print("\npaperB mean curves:", {m: np.round(pb_curve[m], 3).tolist() for m in M})
print("ours best mean:", {m: np.round(np.mean([curve(r['best']) for r in d if r['model'] == m], 0), 3).tolist() for m in M})
print("ours final mean:", {m: np.round(np.mean([curve(r['final']) for r in d if r['model'] == m], 0), 3).tolist() for m in M})
json.dump(dict(label=label, n_best_epoch_le5=n_early, n_dbest_lt_dfinal=n_bcloser, n_dfinal_lt_dbest=n_fcloser, per_arch=per, rows=rows,
               paperB_mean_curve={m: pb_curve[m].tolist() for m in M}), open(RES / "e4_verdict.json", "w"), indent=1, default=float)
