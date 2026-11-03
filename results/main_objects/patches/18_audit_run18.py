"""[audit] Driver for object 18 (registry: results/main_objects/registry/18_*.md).
Copies the authors' isonet_run.py __main__ verbatim except save_dir, then writes the authors' own per-epoch
history (val_* = test set) into the audit record. 10 unseeded repeats. Run with cwd = the repo (data_root
'./hit_data'). Needs PYTHONPATH=~/audit_cns/code and AUDIT_RECORD_DIR."""
import argparse, json, math, os, sys, time
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import torch
from ljp.engine import Trainer
from ljp.optimizers import Adan, AdaBound, DiffGrad, AdaBelief, CAME, Sophia
from ljp.dataset.hit import HIT
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from ljp.models.isonet import ISONet
from audit.main.epoch_record import Recorder

p = argparse.ArgumentParser()
p.add_argument("--repeats", default="0-9")
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--save_dir", required=True)
a = p.parse_args()
lo, hi = map(int, a.repeats.split("-")) if "-" in a.repeats else (int(a.repeats),) * 2

torch.backends.cudnn.benchmark = True


def author_main(epochs, save_dir):   # isonet_run.py __main__, save_dir and epochs passed in
    device = f'cuda:0'
    layer_cfg = '4l064'
    kernel_size = 9
    train_bs = 32  # 8,16,32,64,128
    data = HIT(data_root='./hit_data',
               train_bs=train_bs,
               train_frac=0.1)

    model = ISONet(
        layer_cfg=layer_cfg,
        kernel_size=kernel_size,
        n_output=data.num_classes,
        inchannels=data.inchannels,
        activator='mish',
    )

    net = Trainer(model)

    net.compile(loss_dict={'loss_func': torch.nn.CrossEntropyLoss},
                optimizer_dict={'optimizer': Adan,
                                'lr': 5e-4,
                                # 'weight_decay': 1e-4
                                },
                lr_scheduler_dict={
                    'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                    'step_size': 60,
                    'gamma': 0.1,
                },
                metrics_dict={
                    "acc": Accuracy('multiclass', num_classes=data.num_classes, average='weighted').to(device),
                    "precision": Precision('multiclass', num_classes=data.num_classes, average='weighted').to(device),
                    "recall": Recall('multiclass', num_classes=data.num_classes, average='weighted').to(device),
                    "f1": F1Score('multiclass', num_classes=data.num_classes, average='weighted').to(device),
                    "auc": AUROC('multiclass', num_classes=data.num_classes, average='weighted').to(device),
                },
                numpy_metric=False,
                monitor='val_acc',
                monitor_mode='max')
    return net.fit(data=data, epochs=epochs, device=device, verbose=2, use_compile=False, use_amp=False,
                   save_dir=save_dir)


for r in range(lo, hi + 1):
    rec = Recorder(obj="18", group="ISONet|HIT", seed=r, select_split=None, report_split="test",
                   note=f"epochs={a.epochs}; unseeded repeat {r}; test = authors' val_* (test_x.npy, 2412); weighted F1 in test_f1")
    t0 = time.time()
    results, _, history = author_main(a.epochs, os.path.join(a.save_dir, f"rep{r}"))
    rec.d["test_f1"] = {}
    for i in range(len(history)):
        e = i + 1
        rec.log(e, "test", float(history["val_acc"][i]), float(history["val_loss"][i]))
        rec.d["test_f1"][str(e)] = float(history["val_f1"][i])
    rec.author_pick(int(results["best_epoch"]))
    rec.finish(float(results["best_val_acc"]))
    f1_rec = rec.d["test_f1"][str(results["best_epoch"])]
    f1_rep = float(results["best_history"]["val_f1"])
    rec.d["author_reported_f1"] = f1_rep
    rec.d["author_check_f1"] = dict(recorded=f1_rec, reported=f1_rep, abs_diff=abs(f1_rec - f1_rep),
                                    matches=math.isclose(f1_rec, f1_rep, abs_tol=1e-9))
    rec._dump()
    print(json.dumps(dict(repeat=r, seconds=round(time.time() - t0, 1), best_epoch=int(results["best_epoch"]),
                          best_val_acc=float(results["best_val_acc"]), last_acc=float(history["val_acc"].iloc[-1]),
                          check=rec.d.get("author_check"), check_f1=rec.d["author_check_f1"])), flush=True)
