"""WP1（泛化自 E4，训练循环不变）。E4：用 paperB 代码包的训练函数原样训练 16 文件 v2 切分。严格按 results/PREREG_E4_paperB_checkpoint.md。
外层循环逐行对应 paperB train/trainer.py::run_experiment，只替换 (a) 切分 (b) noise_type=None (c) SNR 扫描用 test 集。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import sys, os, types, json, time, hashlib, argparse, copy
try:
    import mamba_ssm  # noqa
    HAVE_MAMBA = True
except Exception:
    stub = types.ModuleType("mamba_ssm"); stub.Mamba = stub.Mamba2 = None; sys.modules["mamba_ssm"] = stub; HAVE_MAMBA = False
sys.path.insert(0, _PAPERB_CODE)
import numpy as np, torch, torch.nn as nn
from pbsrc.train import trainer as T
from pbsrc.train.config import TrainConfig
from pbsrc.data.dataset import make_dataloaders
from pbsrc.data.splits_v2 import build_cwru_splits, build_jnu_modeA_splits

ROOT16 = _CWRU16
SNRS = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
SWEEP_EPOCHS = {1, 2, 3, 5, 10, 25, 50, 100}

def sweep(model, X, y, dev, snrs):
    """paperB evaluate_snr_sweep，包在 RNG 隔离里（torch + numpy + python random）。"""
    import random
    ps, rs = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev.type == "cuda" else []):
        out = T.evaluate_snr_sweep(model, X, y, snrs, "awgn", dev, 64, 4)
    np.random.set_state(ps); random.setstate(rs)
    res = {}
    for s, m in out.items():
        cm = np.array(m["confusion_matrix"]); col = cm.sum(0)
        res[f"{s:.1f}"] = dict(accuracy=float(m["accuracy"]), confusion_matrix=cm.tolist(), top=int(col.argmax()), conc=float(col.max() / col.sum()))
    return res

def unfuse_mamba2(model):
    """causal-conv1d 融合核要求步长为 8 的倍数（本配置投影宽度 514）→ 改走 Mamba2 自带的非融合路径（数学相同，见预注册 §4）。"""
    import mamba_ssm.modules.mamba2 as M2
    n = 0
    for mod in model.modules():
        if isinstance(mod, M2.Mamba2): mod.use_mem_eff_path = False; n += 1
    M2.causal_conv1d_fn = None
    assert n > 0, "no Mamba2 blocks found"

def run_one(b, m, seed, dev, noise_type=None):
    cfg = TrainConfig(model_name=m, seed=seed, noise_type=noise_type, eval_snr_list=SNRS, eval_noise_type="awgn", save_best=True)
    T.set_seed(cfg.seed)
    X_tr, y_tr = b.X["train"].astype(np.float32), b.y["train"].astype(np.int64)
    X_val, y_val = b.X["val"].astype(np.float32), b.y["val"].astype(np.int64)
    X_te, y_te = b.X["test"].astype(np.float32), b.y["test"].astype(np.int64)
    train_dl, val_dl = make_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=cfg.batch_size, noise_type=cfg.noise_type,
                                        snr_db=cfg.train_snr_db, num_workers=0)   # 偏离：沙盒禁用多进程 socket；数值等价（见 DEVIATIONS_E4.md）
    model = T.build_model(cfg).to(dev)
    if m == "mamba2": unfuse_mamba2(model)
    counts = np.bincount(y_tr, minlength=cfg.num_classes).astype(float)
    weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32, device=dev)
    weights = weights / weights.sum() * cfg.num_classes
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6)
    def lr_warmup(epoch):
        if epoch < cfg.warmup_epochs: return (epoch + 1) / cfg.warmup_epochs
        return 1.0
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_warmup)
    best_val_acc, best_state, best_epoch = 0.0, None, None
    val_curve, probe8, sweeps = [], [], {}
    for epoch in range(cfg.epochs):
        tr_loss = T.train_one_epoch(model, train_dl, optimizer, criterion, dev, cfg.grad_clip)
        if epoch < cfg.warmup_epochs: warmup_sched.step()
        elif scheduler is not None: scheduler.step()
        val_acc = T.evaluate(model, val_dl, dev, cfg.num_classes)["accuracy"]
        val_curve.append(round(float(val_acc), 5))
        if val_acc > best_val_acc and cfg.save_best:
            best_val_acc = val_acc; best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        p = sweep(model, X_te, y_te, dev, [-8.0])["-8.0"]; probe8.append(dict(top=p["top"], conc=round(p["conc"], 4)))
        if epoch + 1 in SWEEP_EPOCHS: sweeps[str(epoch + 1)] = sweep(model, X_te, y_te, dev, SNRS)
        if (epoch + 1) % 25 == 0: print(f"    ep{epoch+1:3d} loss={tr_loss:.4f}", flush=True)   # 不打印 val/靶标
    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    final = sweep(model, X_te, y_te, dev, SNRS)
    model.load_state_dict(best_state); best = sweep(model, X_te, y_te, dev, SNRS)
    return dict(model=m, seed=seed, best_val_acc=float(best_val_acc), best_epoch=best_epoch, val_curve=val_curve,
                probe8=probe8, sweeps_by_epoch=sweeps, final=final, best=best, class_weights=weights.cpu().tolist(),
                n_train=int(len(y_tr)), steps_per_epoch=len(train_dl)), best_state, final_state

EXPS = {"cwru_clean": ("cwru", None), "cwru_awgn": ("cwru", "awgn"), "cwru_m2": ("cwru", None), "jnu": ("jnu", None)}
JNU_ROOT = _JNU

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPS)); ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--rpm", type=int, default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ds, noise = EXPS[a.exp]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ds == "cwru": b = build_cwru_splits(ROOT16, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
    else: b = build_jnu_modeA_splits(JNU_ROOT, rpm=a.rpm)
    models = a.models.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    if a.smoke:
        print(a.exp, "noise", noise, "rpm", a.rpm, "HAVE_MAMBA", HAVE_MAMBA)
        for s in ("train", "val", "test"): print(" ", s, b.X[s].shape, np.bincount(b.y[s], minlength=4).tolist())
        for m in models:
            net = T.build_model(TrainConfig(model_name=m)).to(dev)
            if m == "mamba2": unfuse_mamba2(net)
            x = torch.from_numpy(b.X["test"][:8].astype(np.float32)).to(dev)
            with torch.no_grad(): print(" ", m, T.count_params(net), tuple(net(x).shape))
        return
    os.makedirs(a.ckpt_dir, exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["model"], r["seed"]) for r in res}
    for s in seeds:
        for m in models:
            if (m, s) in done: continue
            print(f"--- {a.exp} {m} seed{s} ---", flush=True); t0 = time.time()
            rec, bst, fin = run_one(b, m, s, dev, noise_type=noise)
            rec.update(exp=a.exp, noise_type=noise, rpm=a.rpm, unfused=(m == "mamba2"), torch=torch.__version__)
            for tag, st in (("best", bst), ("final", fin)):
                ck = os.path.join(a.ckpt_dir, f"{a.exp}_{m}_s{s}{'_rpm%d' % a.rpm if a.rpm else ''}_{tag}.pt"); torch.save(st, ck)
                rec[f"ckpt_{tag}"] = ck; rec[f"ckpt_{tag}_sha256"] = hashlib.sha256(open(ck, "rb").read()).hexdigest()
            rec["wall_s"] = round(time.time() - t0, 1); res.append(rec)
            tmp = a.out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, a.out)
            print(f"    done {len(res)} wall={rec['wall_s']:.0f}s", flush=True)
    print("=== ALL DONE ===", flush=True)

if __name__ == "__main__":
    main()
