"""Known-truth test (results/known_truth/PLAN.md, 678548b): split a fixed-order test set into a selection half S and a
truth half T, stratified by class, seed 20261102, and log both every epoch through the usual Recorder (splits "S", "T";
"test" stays the full test set, recomputed from the same predictions, so it can be checked against the authors' value).
Only evaluation is touched: masks are computed from labels with their own Generator, no global RNG is used."""
import numpy as np

SEED = 20261102


def halves(y, seed=SEED):
    """Boolean masks (S, T): within each class (in sorted order) a random half, floor, goes to S."""
    y = np.asarray(y).ravel(); rng = np.random.default_rng(seed); S = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c); S[idx[rng.permutation(len(idx))[:len(idx) // 2]]] = True
    return S, ~S


class Splitter:
    """Caches the halves of each test set (by key) and checks that its label order never changes between epochs."""

    def __init__(self, seed=SEED):
        self.seed, self.ref = seed, {}

    def masks(self, key, y):
        y = np.asarray(y).ravel()
        if key not in self.ref:
            self.ref[key] = (y.copy(),) + halves(y, self.seed)
        ref, S, T = self.ref[key]
        if not np.array_equal(ref, y):
            raise RuntimeError(f"test set {key!r}: label order changed between epochs; halves would not be fixed")
        return S, T


def acc(pred, y, m=None):
    pred, y = np.asarray(pred).ravel(), np.asarray(y).ravel()
    m = np.ones(len(y), bool) if m is None else m
    return float(np.mean(pred[m] == y[m]))


def balanced(pred, y, m=None, classes=None):
    pred, y = np.asarray(pred).ravel(), np.asarray(y).ravel()
    m = np.ones(len(y), bool) if m is None else m
    classes = np.unique(y) if classes is None else classes
    return float(np.mean([np.mean(pred[m & (y == c)] == c) for c in classes]))


def log_halves(rec, epoch, pred, y, splitter, key="test", metric=acc, loss=None):
    """Log full / S / T for one test set; returns the three values."""
    S, T = splitter.masks(key, y)
    full, s, t = metric(pred, y), metric(pred, y, S), metric(pred, y, T)
    rec.log(epoch, "test", full, loss); rec.log(epoch, "S", s); rec.log(epoch, "T", t)
    return full, s, t
