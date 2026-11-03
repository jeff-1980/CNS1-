"""[audit] Driver for object 19 (registry: results/main_objects/registry/19_*.md).
Calls the authors' train.evaluate_backbone(backbone, cfg, load 0, seed, ["SoftMax"]) for seeds 0..9 with
train.train_deep_head replaced by a verbatim copy that only adds a Keras callback to head.fit: after every epoch
it predicts the test features (the authors' own predict + argmax) and records test acc / cross-entropy.
Needs PYTHONPATH=~/audit_cns/code and AUDIT_RECORD_DIR; cwd anywhere."""
import argparse, copy, json, math, os, sys, time
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import numpy as np
import tensorflow as tf
import train as T
from config import cfg as default_cfg
from model import build_classifier
from audit.main.epoch_record import Recorder

p = argparse.ArgumentParser()
p.add_argument("--backbones", default="DenseNet201,ResNet152V2,MobileNetV2")
p.add_argument("--seeds", default="0-9")
p.add_argument("--epochs", type=int, default=None)
p.add_argument("--data-root", required=True)
a = p.parse_args()
lo, hi = map(int, a.seeds.split("-")) if "-" in a.seeds else (int(a.seeds),) * 2

cfg = copy.copy(default_cfg)
cfg.data_root = a.data_root
if a.epochs is not None:
    cfg.epochs = a.epochs
REC = None
Y_TEST = None     # set per call from the authors' bundle (see wrapper below)


class TestEval(tf.keras.callbacks.Callback):
    def __init__(self, x_test):
        super().__init__(); self.x_test = x_test

    def on_epoch_end(self, epoch, logs=None):
        prob = self.model.predict(self.x_test, verbose=0)
        acc = float(np.mean(np.argmax(prob, axis=1) == Y_TEST))
        loss = float(np.mean(-np.log(np.clip(prob[np.arange(len(Y_TEST)), Y_TEST], 1e-12, None))))
        REC.log(epoch + 1, "test", acc, loss)


def train_deep_head(features_train, y_train, features_test, cfg):   # verbatim copy of train.train_deep_head + callback
    """Three fully connected layers trained on the frozen GAP features."""
    head = build_classifier(features_train.shape[1], cfg)
    head.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 metrics=["accuracy"])
    head.fit(features_train, y_train, epochs=cfg.epochs,
             batch_size=cfg.batch_size, verbose=0, callbacks=[TestEval(features_test)])   # [audit] callback added
    return np.argmax(head.predict(features_test, verbose=0), axis=1)


T.train_deep_head = train_deep_head
_extract = T.extract_features


def extract_features(extractor, dataset):   # pass-through; keeps the test labels for the callback
    global Y_TEST
    f, y = _extract(extractor, dataset)
    Y_TEST = y          # called for train, then test -> ends holding the test labels
    return f, y


T.extract_features = extract_features

for bb in a.backbones.split(","):
    for s in range(lo, hi + 1):
        REC = Recorder(obj="19", group=f"{bb}+SoftMax|CWRU-A", seed=s, select_split=None, report_split="test",
                       note=f"epochs={cfg.epochs}; load 0 (A); test = 50/class; acc from authors' predict+argmax")
        t0 = time.time()
        res = T.evaluate_backbone(bb, cfg, 0, cfg.seed + s, ["SoftMax"])
        acc, _ = res["SoftMax"]
        REC.author_pick(cfg.epochs)
        REC.finish(acc / 100)   # classic.accuracy returns percent
        REC.d["author_reported_raw"] = float(acc)
        REC._dump()
        print(json.dumps(dict(backbone=bb, seed=s, seconds=round(time.time() - t0, 1), reported=float(acc),
                              check=REC.d["author_check"])), flush=True)
        tf.keras.backend.clear_session()
