"""B 线阶段 B2：确认。严格按 PREREG_B（a8f62cc0…）§5 与 B_RULE_FREEZE.md（ddc426eb…）。C-a 在每个实例训练前计算并落盘。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import sys, os, json, hashlib, types, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
stub = types.ModuleType("mamba_ssm"); stub.Mamba = stub.Mamba2 = None; sys.modules["mamba_ssm"] = stub
import numpy as np, torch
from b_common import rule_a, rule_bc, outcome, as3, dev
from prior_screen import REGISTRY
import pu_wp2 as W
import wp5_consequence as W5
sys.path.insert(0, _PAPERB_CODE)
from pbsrc.data.splits_v2 import build_jnu_modeA_splits
from data.splits_v2 import _segment_and_window, GAP_SAMPLES
R = (_ROOT + "/results"); C = f"{R}/ckpt_b"; os.makedirs(C, exist_ok=True)
for f, h in (("PREREG_B_hierarchical_attractor.md", "prereg_b_sha256.txt"), ("B_RULE_FREEZE.md", "b_rule_freeze_sha256.txt")):
    assert hashlib.sha256(open(f"{R}/{f}", "rb").read()).hexdigest() == open(f"{R}/registry/{h}").read().split()[0], f
CONDS = [("PU", "N09_M07_F10"), ("PU", "N15_M01_F10"), ("PU", "N15_M07_F04"), ("JNU", 800), ("JNU", 1000)]
MODELS, SEEDS, PER_CLASS_JNU = ["wdcnn", "drsn", "lstm"], [0, 1, 2], 1200
def pu_data(op):
    W.OP = op; pools = W.load_pools(W.block_codes("R")); Xt, yt = W.build_test(pools, "R")
    avail = W._codes_present(W.PU_ROOT); Xc, yc = [], []
    for c in W.block_codes("R"):
        for f in [f for f in avail[c] if f.name.startswith(op)]:
            sig = W._load_pu_signal(f); n = len(sig); a0 = int(n * 0.6) + GAP_SAMPLES; z = a0 + int(n * 0.2); tmp, y_, w_ = [], [], []
            _segment_and_window(sig[a0:z], f.stem, "val", 0, W.SEG, W.STEP, True, a0, tmp, y_, w_); Xc += tmp; yc += [W.label_of(c)] * len(tmp)
    perm = np.random.default_rng(0).permutation(len(Xc)); Xc = np.stack(Xc)[perm]; yc = np.array(yc)[perm]
    return dict(train=lambda s: W.build_train(pools, "R-L1", s)[:2], Xc=Xc, yc=yc, Xt=Xt, yt=yt, ncls=3)
def jnu_data(rpm):
    b = build_jnu_modeA_splits(_JNU, rpm=rpm)
    def train(s):
        rng = np.random.default_rng([s, rpm]); idx = np.concatenate([rng.choice(np.where(b.y["train"] == c)[0], min(PER_CLASS_JNU, int((b.y["train"] == c).sum())), replace=False) for c in range(4)])
        return b.X["train"][idx].astype(np.float32), b.y["train"][idx].astype(np.int64)
    return dict(train=train, Xc=b.X["val"], yc=b.y["val"], Xt=b.X["test"], yt=b.y["test"], ncls=4)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    out = f"{R}/b_confirm.json" if not a.smoke else "/tmp/b_confirm_smoke.json"
    res = json.load(open(out)) if os.path.exists(out) and not a.smoke else []; done = {(r["dataset"], str(r["cond"]), r["model"], r["seed"]) for r in res}
    for ds, cond in (CONDS[:1] + CONDS[3:4] if a.smoke else CONDS):
        D = pu_data(cond) if ds == "PU" else jnu_data(cond); Xc, Xt = as3(D["Xc"]), as3(D["Xt"])
        print(f"== {ds} {cond}: test {tuple(Xt.shape)} {np.bincount(D['yt']).tolist()} calib {len(Xc)}", flush=True)
        for s in SEEDS[:1] if a.smoke else SEEDS:
            Xtr, ytr = D["train"](s); pa, psf = rule_a(Xtr, ytr)          # 训练前：C-a
            for m in MODELS[:1] if a.smoke else MODELS:
                if (ds, str(cond), m, s) in done: continue
                rec = dict(dataset=ds, cond=cond, model=m, seed=s, n_train=int(len(ytr)), class_counts=np.bincount(ytr).tolist(), a=pa, a_sf=psf, a_computed_before_training=True)
                if a.smoke: print("   ", rec, flush=True); continue
                t0 = time.time(); net, steps = W.train_fixed(REGISTRY[m](D["ncls"]), torch.from_numpy(Xtr), torch.from_numpy(ytr), dev, s, log=lambda x: None); net.eval()
                ck = f"{C}/{ds}_{cond}_{m}_s{s}.pt"; torch.save(net.state_dict(), ck)
                rec.update(steps=steps, ckpt=ck, ckpt_sha256=hashlib.sha256(open(ck, "rb").read()).hexdigest(), **rule_bc(net, Xc, D["yc"], s), **outcome(net, Xt, s, D["ncls"]), wall_s=round(time.time() - t0, 1))
                res.append(rec); tmp = out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, out)
                print(f"    done {len(res)}/45 wall={rec['wall_s']:.0f}s", flush=True)   # 不打印结局
    print("=== ALL DONE ===", len(res), flush=True)
if __name__ == "__main__":
    main()
