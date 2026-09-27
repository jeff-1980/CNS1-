"""WP5b 事后分析汇总（审稿意见 7）。不改变 WP5 的预注册判定。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import json, collections, pathlib, numpy as np
R = pathlib.Path((_ROOT + "/results"))
d = []
for f in ("wp5b_main.json", "wp5b_mamba.json"):
    if (R / f).exists(): d += json.load(open(R / f))
DS = lambda r: "CWRU" if r["group"].startswith("cwru") else "PU"
out = dict(note="POST HOC selective-classification analysis (review comment 7). Denominators: coverage = accepted/all windows; acc_err_over_all = accepted errors/all windows; risk_accepted = errors/accepted windows; accept_by_class = accepted/all windows of that class.", n=len(d), cells={})
g = collections.defaultdict(list)
for r in d: g[(DS(r), r["model"])].append(r)
for (ds, m), rs in sorted(g.items()):
    c = {}
    for snr in ("10.0", "0.0", "-8.0"):
        for b in ("B1", "B2"):
            x = [r["by_snr"][snr][b] for r in rs]
            c[f"{snr}|{b}"] = dict(coverage=float(np.mean([v["coverage"] for v in x])), acc_err_over_all=float(np.mean([v["acc_err_over_all"] for v in x])),
                                   risk_accepted=float(np.nanmean([v["risk_accepted"] for v in x])) if any(not np.isnan(v["risk_accepted"]) for v in x) else None,
                                   accept_healthy=float(np.mean([v["accept_by_class"]["0"] for v in x])),
                                   accept_fault=float(np.mean([np.mean([v["accept_by_class"][k] for k in v["accept_by_class"] if k != "0"]) for v in x])),
                                   aurc=float(np.mean([v["aurc"] for v in x])), rc=np.mean([v["rc"] for v in x], 0).round(4).tolist())
    out["cells"][f"{ds}|{m}"] = dict(n=len(rs), **c)
    b2, b10 = c["-8.0|B2"], c["10.0|B2"]
    print(f"{ds:4s} {m:11s} n={len(rs):3d} | −8 dB B2: cov {b2['coverage']:.2f} accErr/all {b2['acc_err_over_all']:.2f} risk|acc {b2['risk_accepted'] if b2['risk_accepted'] is None else round(b2['risk_accepted'],2)} accH {b2['accept_healthy']:.2f} accF {b2['accept_fault']:.2f} AURC {b2['aurc']:.2f} | +10 dB cov {b10['coverage']:.2f} risk|acc {b10['risk_accepted'] if b10['risk_accepted'] is None else round(b10['risk_accepted'],2)}")
json.dump(out, open(R / "wp5b_summary.json", "w"), indent=1)
