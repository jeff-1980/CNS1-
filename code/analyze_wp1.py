"""按 PREREG_WP1_protocol_baseline.md 分析。运行时校验预注册哈希。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, hashlib, sys, glob, os, collections, pathlib
import numpy as np
R = pathlib.Path((_ROOT + "/results")); PB = _PAPERB_RESULTS
if hashlib.sha256((R / "PREREG_WP1_protocol_baseline.md").read_bytes()).hexdigest() != (R / "registry/prereg_wp1_sha256.txt").read_text().split()[0]:
    sys.exit("预注册哈希不符，拒绝判定")
CLS = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}
ARMS = [  # (臂, 结果文件, paperB 路径模板, 架构)
    ("C-clean",   "wp1_cwru_clean.json",   "cwru/cwru_clean_v2/{m}/seed{s}_summary.json", ["lstm", "transformer"]),
    ("C-clean-M", "wp1_cwru_clean_M.json", "cwru/cwru_clean_v2/{m}/seed{s}_summary.json", ["vibrmamba"]),
    ("C-m2",      "wp1_cwru_m2.json",      "cwru/ablation_v3/mamba2_no_noise_train/seed{s}/summary.json", ["mamba2"]),
    ("C-awgn",    "wp1_cwru_awgn.json",    "cwru/main_awgn_v2/{m}/seed{s}_summary.json", ["cnn1d", "wdcnn", "drsn", "lstm", "transformer"]),
    ("C-awgn-M",  "wp1_cwru_awgn_M.json",  "cwru/main_awgn_v2/{m}/seed{s}_summary.json", ["vibrmamba", "mamba2"]),
    ("J",         "wp1_jnu.json",          "jnu/jnu_modeA_v2/{m}/seed{s}_summary.json", ["cnn1d", "wdcnn", "drsn", "lstm", "transformer"]),
    ("J-M",       "wp1_jnu_M.json",        "jnu/jnu_modeA_v2/{m}/seed{s}_summary.json", ["vibrmamba", "mamba2"]),
]
NA_MINUS = {"C-clean", "C-clean-M", "C-m2", "J", "J-M"}
jnu_choice = json.load(open(R / "wp1_jnu_rpm_choice.json"))
cells, h1_pool, missing = [], [], []
for arm, fn, tpl, models in ARMS:
    if not (R / fn).exists(): missing.append(fn); continue
    d = {(r["model"], r["seed"]): r for r in json.load(open(R / fn))}
    for m in models:
        rs = [d[(m, s)] for s in range(5) if (m, s) in d]
        if len(rs) < 5: missing.append(f"{arm}/{m}: {len(rs)}/5")
        if not rs: continue
        pbs = {r["seed"]: json.load(open(f"{PB}/{tpl.format(m=m, s=r['seed'])}")) for r in rs}
        snrs = list(pbs[rs[0]["seed"]]["snr_sweep"].keys())
        bitwise = all(np.array_equal(np.array(r["best"][k]["confusion_matrix"]), np.array(pbs[r["seed"]]["snr_sweep"][k]["confusion_matrix"])) for r in rs for k in snrs)
        ours = np.mean([[r["best"][k]["accuracy"] for k in snrs] for r in rs], 0)
        theirs = np.mean([[pbs[r["seed"]]["snr_sweep"][k]["accuracy"] for k in snrs] for r in rs], 0)
        mae = float(np.mean(np.abs(ours - theirs)))
        level = "BITWISE" if bitwise else ("BEHAVIORAL" if mae <= 0.05 else "NOT_REPRODUCED")
        if arm.startswith("J") and level == "BITWISE" and jnu_choice["mode"] != "BITWISE_CHECKABLE": level += "(?)"
        def coll(ck):
            out = {}
            for k in snrs:
                conc = np.mean([r[ck][k]["conc"] for r in rs])
                if conc >= 0.90: out[k] = (round(conc * 100, 1), dict(collections.Counter(CLS[r[ck][k]["top"]] for r in rs)))
            return out
        cb, cf = coll("best"), coll("final")
        be = [r["best_epoch"] for r in rs]
        if arm in NA_MINUS:
            for r in rs:
                if pbs[r["seed"]]["best_val_acc"] >= 0.99999: h1_pool.append((arm, m, r["seed"], r["best_epoch"]))
        cells.append(dict(arm=arm, model=m, n=len(rs), level=level, mae=round(mae, 4), best_epochs=be,
                          pb_best_val=[round(pbs[r["seed"]]["best_val_acc"], 4) for r in rs],
                          collapsed_best=cb, collapsed_final=cf, n_coll_best=len(cb), n_coll_final=len(cf),
                          unfused=(m == "mamba2"), snr_grid=snrs))
n_h1 = len(h1_pool); k_h1 = sum(e <= 5 for *_, e in h1_pool)
h1 = "NA" if n_h1 == 0 else ("SUPPORTED" if k_h1 / n_h1 >= 0.80 else "NOT_SUPPORTED")
print(f"JNU rpm: {jnu_choice['chosen']} ({jnu_choice['mode']})  missing: {missing or 'none'}\n")
print(f"{'arm':10s} {'model':12s} {'level':15s} {'MAE':>6s} best_epochs        coll best→final")
for c in cells:
    print(f"{c['arm']:10s} {c['model']:12s} {c['level']:15s} {c['mae']:6.3f} {str(c['best_epochs']):18s} {c['n_coll_best']} → {c['n_coll_final']}")
print(f"\nH1: {k_h1}/{n_h1} NA- instances (paperB best_val=1.0) with best_epoch ≤ 5 → {h1}")
by = collections.defaultdict(lambda: [0, 0])
for a, m, s, e in h1_pool: by[(a, m)][0] += e <= 5; by[(a, m)][1] += 1
print("   per cell:", {f"{a}/{m}": f"{k}/{n}" for (a, m), (k, n) in by.items()})
json.dump(dict(jnu_rpm=jnu_choice, missing=missing, cells=cells, H1=dict(k=k_h1, n=n_h1, label=h1, pool=h1_pool)),
          open(R / "wp1_verdict.json", "w"), indent=1, default=str)
