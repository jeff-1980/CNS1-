"""paperB correction, step 0 (results/paperB_correction/ASSESSMENT.md, f96b1ae): inference only.

For every saved CWRU same-condition checkpoint of paperB (main_awgn_v2 35, cwru_clean_v2 30, ablation_v3 best.pt 15),
evaluate the AWGN SNR sweep on two test sets that differ ONLY in the Normal windows:
  orig : paperB's test split as built by its own code (Normal = 98.mat, whose windows also sit in training as "99.mat")
  x099 : the same non-Normal test windows; Normal = windows of the true X099_DE_time signal from 99.mat, never trained on
Non-Normal windows come first in both sets and noise is seeded per SNR, so their noisy inputs are identical across sets.
Runs from a working copy of paperB's src/ (nothing under ~/论文1 is written); checkpoints and data are read only.
usage: python3 paperB_step0.py --src <copy of 论文1/src parent> --ckpt ~/论文1/results --data ~/data_cwru --out <dir>
"""
import argparse, json, os, sys, time
import numpy as np, scipy.io as sio, torch
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
for k in ("src", "ckpt", "data", "out"): ap.add_argument(f"--{k}", required=True)
a = ap.parse_args()
sys.path.insert(0, a.src)
from src.data.splits_v2 import build_cwru_splits, SPLIT_SEED_DEFAULT
from src.data.dataset import BearingDataset
from src.train.config import TrainConfig
from src.train.trainer import build_model, evaluate

SNR = [-8., -6., -4., -2., 0., 2., 4., 6., 8., 10.]
dev = torch.device("cuda")
os.makedirs(a.out, exist_ok=True)

# test split exactly as paperB built it (legacy loader: "99.mat" = X098)
b = build_cwru_splits(a.data, split_seed=SPLIT_SEED_DEFAULT)
Xte, yte = b.X["test"], b.y["test"]
te_files = sorted({w.file_id for w in b.windows if w.split == "test"})
tr_files = sorted({w.file_id for w in b.windows if w.split == "train"})
nonN = yte != 0
# true X099 windows, same windowing as splits_v2 (1024, 50% overlap, per-window z-score)
sig = sio.loadmat(os.path.join(a.data, "99.mat"))["X099_DE_time"].squeeze().astype(np.float32)
W = np.stack([(s - s.mean()) / (s.std() + 1e-8) for s in (sig[i:i + 1024] for i in range(0, len(sig) - 1024 + 1, 512))])
W = W.reshape((len(W),) + Xte.shape[1:]).astype(Xte.dtype)   # same layout as paperB's arrays
sets = {"orig": (np.concatenate([Xte[nonN], Xte[~nonN]]), np.concatenate([yte[nonN], yte[~nonN]])),
        "x099": (np.concatenate([Xte[nonN], W]), np.concatenate([yte[nonN], np.zeros(len(W), int)]))}
meta = dict(test_files=te_files, train_files=tr_files, n_test=len(yte), n_normal_orig=int((~nonN).sum()),
            n_normal_x099=len(W), x099_len=len(sig), split_seed=SPLIT_SEED_DEFAULT,
            note="noise seed per SNR = 20261103 + index; non-Normal windows first in both sets")
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(meta, flush=True)


def cfg_for(model_name, seed):
    return TrainConfig(model_name=model_name, in_channels=1, d_model=64, num_layers=4, num_classes=4, seed=seed)


jobs = []
for g in ("main_awgn_v2", "cwru_clean_v2"):
    for m in sorted(os.listdir(os.path.join(a.ckpt, g))):
        for s in range(5):
            p = os.path.join(a.ckpt, g, m, f"best_seed{s}.pt")
            if os.path.exists(p):
                jobs.append((g, m, m, s, p, os.path.join(a.ckpt, g, m, f"seed{s}_summary.json")))
for v, mn in (("mamba2_full", "mamba2"), ("mamba2_no_noise_train", "mamba2"), ("mamba2_no_conv_embed", "mamba2_no_conv_embed")):
    for s in range(5):
        jobs.append(("ablation_v3", v, mn, s, os.path.join(a.ckpt, "ablation_v3", v, f"seed{s}", "best.pt"),
                     os.path.join(a.ckpt, "ablation_v3", v, f"seed{s}", "summary.json")))

rows = []
for g, label, mn, s, p, summ in jobs:
    t0 = time.time()
    model = build_model(cfg_for(mn, s)).to(dev)
    model.load_state_dict(torch.load(p, map_location=dev), strict=True)
    model.eval()
    pub = json.load(open(summ))["snr_sweep"]
    for k, snr in enumerate(SNR):
        r = dict(group=g, model=label, seed=s, snr=snr, published_acc=pub[str(snr)]["accuracy"],
                 published_cm=pub[str(snr)]["confusion_matrix"])
        for name, (X, y) in sets.items():
            torch.manual_seed(20261103 + k)
            dl = DataLoader(BearingDataset(X, y, noise_type="awgn", snr_db=snr), batch_size=256, shuffle=False, num_workers=0)
            with torch.no_grad():
                met = evaluate(model, dl, dev, 4)
            r[f"{name}_acc"] = float(met["accuracy"]); r[f"{name}_cm"] = np.asarray(met["confusion_matrix"]).tolist()
        rows.append(r)
    print(g, label, s, f"{time.time() - t0:.1f}s", flush=True)
json.dump(rows, open(os.path.join(a.out, "runs.json"), "w"))
print("done", len(rows), "rows")
