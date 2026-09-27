"""预注册 WP1 §3：JNU 转速识别。只输出匹配情况与 MAE，不输出靶标。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, sys, numpy as np, hashlib
R = (_ROOT + "/results")
pre = open(f"{R}/PREREG_WP1_protocol_baseline.md", "rb").read()
assert hashlib.sha256(pre).hexdigest() == open(f"{R}/registry/prereg_wp1_sha256.txt").read().split()[0], "prereg hash mismatch"
pb = json.load(open((_PAPERB_RESULTS + "/jnu/jnu_modeA_v2/wdcnn/seed0_summary.json")))["snr_sweep"]
out = {}
for rpm in (600, 800, 1000):
    r = json.load(open(f"{R}/wp1_jnu_rpmid_{rpm}.json"))[0]
    eq = all(np.array_equal(np.array(r["best"][k]["confusion_matrix"]), np.array(pb[k]["confusion_matrix"])) for k in pb)
    mae = float(np.mean([abs(r["best"][k]["accuracy"] - pb[k]["accuracy"]) for k in pb]))
    out[rpm] = dict(bitwise=eq, mae=round(mae, 4))
bw = [k for k, v in out.items() if v["bitwise"]]
chosen, mode = (bw[0], "BITWISE_CHECKABLE") if len(bw) == 1 else (min(out, key=lambda k: out[k]["mae"]), "BEHAVIORAL_ONLY")
json.dump(dict(per_rpm=out, chosen=chosen, mode=mode), open(f"{R}/wp1_jnu_rpm_choice.json", "w"), indent=1)
print(out, "->", chosen, mode); open(f"{R}/wp1_jnu_rpm.txt", "w").write(str(chosen))
