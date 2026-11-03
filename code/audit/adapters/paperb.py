"""P1 适配器：paperB（Wang et al. 2026, MST）。逐行对应 code/e4_paperB_harness.py::run_one（E4 已冻结，不改）。
差别只在外层：选模协议与探针由 harness_core 统一处理；E4 的逐轮 -8 dB 探针在这里由 eval_by_epoch 取代。"""
import os, sys, types
import numpy as np, torch, torch.nn as nn
from .. import torch_utils as TU
from ..harness_core import Adapter

_ROOT = os.environ.get("CNS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_PAPERB_CODE = os.environ.get("PAPERB_CODE", os.path.join(_ROOT, "external/paperB_code"))
_CWRU16 = os.environ.get("CWRU16_DIR", os.path.join(_ROOT, "data/cwru_16"))
SNRS = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def _import_pbsrc():
    try:
        import mamba_ssm  # noqa
    except Exception:
        stub = types.ModuleType("mamba_ssm"); stub.Mamba = stub.Mamba2 = None; sys.modules["mamba_ssm"] = stub
    if _PAPERB_CODE not in sys.path: sys.path.insert(0, _PAPERB_CODE)
    from pbsrc.train import trainer as T
    from pbsrc.train.config import TrainConfig
    from pbsrc.data.dataset import make_dataloaders
    from pbsrc.data.splits_v2 import build_cwru_splits
    return T, TrainConfig, make_dataloaders, build_cwru_splits


class PaperBAdapter(Adapter):
    def __init__(self, model_name, device=None):
        self.T, self.TrainConfig, self.make_dataloaders, build = _import_pbsrc()
        self.model_name = model_name
        self.name = f"paperB_{model_name}"
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.b = build(_CWRU16, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
        self.n_epochs = self.TrainConfig(model_name=model_name).epochs

    def setup(self, seed):
        T, b = self.T, self.b
        cfg = self.cfg = self.TrainConfig(model_name=self.model_name, seed=seed, noise_type=None, eval_snr_list=SNRS, eval_noise_type="awgn", save_best=True)
        T.set_seed(cfg.seed)
        X_tr, y_tr = b.X["train"].astype(np.float32), b.y["train"].astype(np.int64)
        self.X_val, self.y_val = b.X["val"].astype(np.float32), b.y["val"].astype(np.int64)
        self.X_te, self.y_te = b.X["test"].astype(np.float32), b.y["test"].astype(np.int64)
        self.train_dl, self.val_dl = self.make_dataloaders(X_tr, y_tr, self.X_val, self.y_val, batch_size=cfg.batch_size, noise_type=cfg.noise_type,
                                                           snr_db=cfg.train_snr_db, num_workers=0)   # 与 E4 相同的偏离（DEVIATIONS_E4.md）
        self.model = T.build_model(cfg).to(self.dev)
        counts = np.bincount(y_tr, minlength=cfg.num_classes).astype(float)
        w = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32, device=self.dev)
        w = w / w.sum() * cfg.num_classes
        self.criterion = nn.CrossEntropyLoss(weight=w)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6)
        def lr_warmup(epoch):
            if epoch < cfg.warmup_epochs: return (epoch + 1) / cfg.warmup_epochs
            return 1.0
        self.warmup_sched = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_warmup)
        self.best_val_acc = 0.0

    def train_epoch(self, ep):
        cfg = self.cfg
        loss = self.T.train_one_epoch(self.model, self.train_dl, self.optimizer, self.criterion, self.dev, cfg.grad_clip)
        if ep - 1 < cfg.warmup_epochs: self.warmup_sched.step()
        elif self.scheduler is not None: self.scheduler.step()
        return {"loss": loss}

    def validate(self, ep):
        return {"acc": float(self.T.evaluate(self.model, self.val_dl, self.dev, self.cfg.num_classes)["accuracy"])}

    def author_selects(self, history):
        # paperB：val_acc > best_val_acc（严格提升）且 save_best
        acc = history[-1]["val_acc"]
        if acc > self.best_val_acc and self.cfg.save_best:
            self.best_val_acc = acc; return True
        return False

    def val_loss(self):
        # 作者代码不算验证损失：用作者的训练损失（类别加权 CE）在干净验证集上补算，不经 DataLoader
        # 分批累加：加权和 / 权重和，与整批一次算的 CrossEntropyLoss(weight=w) 定义一致
        w = self.criterion.weight; self.model.eval(); tot, wsum = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(self.y_val), 256):
                x = torch.from_numpy(self.X_val[i:i + 256]).to(self.dev); y = torch.from_numpy(self.y_val[i:i + 256]).to(self.dev)
                tot += float(nn.functional.cross_entropy(self.model(x), y, weight=w, reduction="sum")); wsum += float(w[y].sum())
        return tot / wsum

    def evaluate(self):
        out = self.T.evaluate_snr_sweep(self.model, self.X_te, self.y_te, SNRS, "awgn", self.dev, 64, 4)
        res = {}
        for s, m in out.items():
            cm = np.array(m["confusion_matrix"]); col = cm.sum(0)
            res[f"{s:.1f}"] = dict(accuracy=float(m["accuracy"]), confusion_matrix=cm.tolist(), top=int(col.argmax()), conc=float(col.max() / col.sum()))
        return res

    def isolate(self): return TU.isolate([self.model], self.dev)
    def snapshot(self): return TU.snapshot(self.model)
    def load(self, state): self.model.load_state_dict(state)
    def save(self, state, path): return TU.save(state, path)
    def meta(self): return dict(model=self.model_name, n_train=int(len(self.b.y["train"])), torch=torch.__version__, device=str(self.dev))
