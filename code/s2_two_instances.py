"""二轮复核 R7-2：Table S2 中 5 次噪声抽样主导类别不一致的 2 个实例，重推理保存全部 5 次类别分布，比较两种聚合。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
from pu_wp2 import load_pools, block_codes, build_test
from prior_screen import REGISTRY
from data.noise_augment import add_awgn_tensor
dev = torch.device("cuda"); R = (_ROOT + "/results"); CLS = ["Healthy", "OR", "IR"]
pools = load_pools(block_codes("A")); Xte, yte = build_test(pools, "A"); Xte = torch.as_tensor(Xte); yte = torch.as_tensor(yte)
old = {(r["arm"], r["model"], r["seed"]): r for r in json.load(open(f"{R}/wp2_A.json"))}
out = []
for key in (("A-L12b", "wdcnn", 3), ("A-L1", "lstm", 3)):
    r = old[key]; net = REGISTRY[key[1]](3).to(dev); net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), r["ckpt"]), map_location=dev)); net.eval()
    D = []
    with torch.no_grad():
        for rep in range(5):
            torch.manual_seed(1000 + 17 * key[2] + rep)
            p = torch.cat([net(add_awgn_tensor(Xte[i:i + 256].to(dev), -8.0)).argmax(1).cpu() for i in range(0, len(Xte), 256)]).numpy()
            D.append((np.bincount(p, minlength=3) / len(p)).tolist())
    D = np.array(D); tops = D.argmax(1).tolist(); a = D.max(1).mean(); b = D.mean(0).max()
    rec = dict(arm=key[0], model=key[1], seed=key[2], dists=D.round(4).tolist(), tops=tops, archived_tops=r["observed"]["-8.0"]["tops"], reproduced=tops == r["observed"]["-8.0"]["tops"],
               conc_mean_of_max=float(a), archived_conc_mean=r["observed"]["-8.0"]["conc_mean"], max_of_mean_dist=float(b), class_of_mean_dist=CLS[int(D.mean(0).argmax())], dominant_mode=CLS[int(np.bincount(tops, minlength=3).argmax())])
    out.append(rec); print({k: v for k, v in rec.items() if k != "dists"})
s = json.load(open(f"{R}/collapse_defs_sensitivity.json")); s["non_unanimous_instances"] = out; json.dump(s, open(f"{R}/collapse_defs_sensitivity.json", "w"), indent=1, ensure_ascii=False)
