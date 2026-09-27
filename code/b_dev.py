"""B 线阶段 B1：开发集上计算 C-a / C-b / C-c 与 −8 dB 宏类结局（已有权重，纯推理）。按 PREREG_B（a8f62cc0…）与 DEVIATIONS_B.md。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import sys, os, json, hashlib, types, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "pbx")
stub = types.ModuleType("mamba_ssm"); stub.Mamba = stub.Mamba2 = None; sys.modules["mamba_ssm"] = stub
import numpy as np, torch
from b_common import rule_a, rule_bc, outcome, as3, dev
from data.splits_v2 import build_cwru_splits as build40
from prior_screen import REGISTRY
import pu_wp2 as W, arm2x2 as A
from wp5_consequence import pu_calib
sys.path.insert(0, _PAPERB_CODE)
from pbsrc.train import trainer as T
from pbsrc.train.config import TrainConfig
from pbsrc.data.splits_v2 import build_cwru_splits as build16, build_jnu_modeA_splits
R = (_ROOT + "/results")
assert hashlib.sha256(open(f"{R}/PREREG_B_hierarchical_attractor.md", "rb").read()).hexdigest() == open(f"{R}/registry/prereg_b_sha256.txt").read().split()[0]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ck = lambda p: os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), p))
out = []; t0 = time.time()
def run(net, ckp, shv, Xtr, ytr, Xc, yc, Xt, seed, ncls, meta):
    assert sha(ckp) == shv; net.load_state_dict(torch.load(ckp, map_location=dev)); net.eval()
    a, am = rule_a(Xtr, ytr); bc = rule_bc(net, Xc, yc, seed); o = outcome(net, Xt, seed, ncls)
    out.append(dict(**meta, seed=seed, a=a, a_sf=am, **bc, **o))
# CWRU 40：2×2 与 lt
b = build40(_CWRU40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
fids = [w.file_id for w in b.windows if w.split == "train"]; pool = (fids, b.X["train"], b.y["train"])
Xc40, yc40, Xt40 = as3(b.X["val"]), b.y["val"], as3(b.X["test"])
for fn in ("arm2x2.json", "arm2x2_lt.json"):
    for r in json.load(open(f"{R}/{fn}")):
        Xtr, ytr, _, idx = A.build_arm(pool, r["arm"], r["seed"]); assert hashlib.md5(idx.tobytes()).hexdigest() == r["idx_md5"]
        run(REGISTRY[r["model"]](4).to(dev), ck(r["ckpt"]), r["ckpt_sha256"], Xtr, ytr, Xc40, yc40, Xt40, r["seed"], 4, dict(dataset="CWRU", cond=f"40file {r['arm']}", model=r["model"], src=fn))
print("cwru40 done", len(out), f"{time.time()-t0:.0f}s", flush=True)
# CWRU 16 与 JNU 600：paperB 协议 NA-，best + final
b16 = build16(_CWRU16, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
bj = build_jnu_modeA_splits(_JNU, rpm=600)
for fn, bb, ds, cond in (("e4_paperB_harness.json", b16, "CWRU", "16file NA-"), ("wp1_cwru_clean.json", b16, "CWRU", "16file NA-"), ("wp1_jnu.json", bj, "JNU", "600rpm NA-")):
    Xc, yc, Xt = as3(bb.X["val"]), bb.y["val"], as3(bb.X["test"])
    for r in json.load(open(f"{R}/{fn}")):
        for tag in ("best", "final"):
            run(T.build_model(TrainConfig(model_name=r["model"])).to(dev), ck(r[f"ckpt_{tag}"]), r[f"ckpt_{tag}_sha256"], bb.X["train"], bb.y["train"], Xc, yc, Xt, r["seed"], 4,
                dict(dataset=ds, cond=f"{cond} {tag}", model=r["model"], src=fn))
    print(fn, "done", len(out), f"{time.time()-t0:.0f}s", flush=True)
# PU WP2
for blk, fns in (("R", ("wp2_R.json", "wp2_R_single.json")), ("A", ("wp2_A.json",))):
    pools = W.load_pools(W.block_codes(blk)); Xt, _ = W.build_test(pools, blk); Xt = as3(Xt)
    Xc = as3(pu_calib(blk)); yc = np.concatenate([[W.label_of(c)] * 0 for c in []] or [np.zeros(0)])
    # 校准窗标签：按 pu_calib 的代码顺序重建
    avail = W._codes_present(W.PU_ROOT); yl = []
    from data.splits_v2 import _segment_and_window, GAP_SAMPLES
    for c in W.block_codes(blk):
        for f in [f for f in avail[c] if f.name.startswith(W.OP)]:
            sig = W._load_pu_signal(f); n = len(sig); a0 = int(n * 0.6) + GAP_SAMPLES; z = a0 + int(n * 0.2); tmp, y_, w_ = [], [], []
            _segment_and_window(sig[a0:z], f.stem, "val", 0, W.SEG, W.STEP, True, a0, tmp, y_, w_); yl += [W.label_of(c)] * len(tmp)
    yc = np.array(yl); assert len(yc) == len(Xc)
    perm = np.random.default_rng(0).permutation(len(Xc)); Xc, yc = Xc[torch.as_tensor(perm).to(dev)], yc[perm]   # 探针取前 2000 窗时覆盖各类
    for fn in fns:
        for r in json.load(open(f"{R}/{fn}")):
            Xtr, ytr, _ = W.build_train(pools, r["arm"], r["seed"])
            run(REGISTRY[r["model"]](3).to(dev), ck(r["ckpt"]), r["ckpt_sha256"], Xtr, ytr, Xc, yc, Xt, r["seed"], 3, dict(dataset="PU", cond=f"N15_M07_F10 {r['arm']}", model=r["model"], src=fn))
    print("PU", blk, "done", len(out), f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open(f"{R}/b_dev.json", "w"), indent=1)
print("=== ALL DONE ===", len(out), flush=True)
