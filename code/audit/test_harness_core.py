"""harness_core 的逻辑测试（只用 numpy，无需 torch / GPU）。在 code/ 下运行：python -m audit.test_harness_core"""
import contextlib, hashlib, random
import numpy as np
from . import harness_core as H


class Toy(H.Adapter):
    """线性模型 + 随机小批量 SGD。评估故意消耗 numpy/python RNG，用来检验隔离是否生效。"""
    name = "toy"

    def __init__(self, n_epochs=12, leak=False):
        self.n_epochs = n_epochs; self.leak = leak

    def setup(self, seed):
        np.random.seed(seed); random.seed(seed)
        self.X = np.random.randn(200, 5); self.y = self.X @ np.arange(1, 6) + 0.5 * np.random.randn(200)
        self.Xv, self.yv = self.X[150:], self.y[150:]; self.X, self.y = self.X[:150], self.y[:150]
        self.w = np.zeros(5); self.best = -np.inf

    def train_epoch(self, ep):
        idx = np.random.permutation(len(self.y))
        for i in range(0, len(idx), 16):
            j = idx[i:i + 16]; g = self.X[j].T @ (self.X[j] @ self.w - self.y[j]) / len(j)
            self.w -= 0.02 * g
        return {"loss": float(np.mean((self.X @ self.w - self.y) ** 2))}

    def validate(self, ep):
        return {"acc": float(-np.mean((self.Xv @ self.w - self.yv) ** 2))}

    def author_selects(self, history):
        a = history[-1]["val_acc"]
        if a > self.best: self.best = a; return True
        return False

    def val_loss(self):
        np.random.rand(); random.random()          # 故意消耗 RNG
        return float(np.mean((self.Xv @ self.w - self.yv) ** 2))

    def evaluate(self):
        np.random.rand(3); random.random()
        return {"0.0": {"accuracy": float(-np.mean((self.Xv @ self.w - self.yv) ** 2))}}

    @contextlib.contextmanager
    def isolate(self):
        ps, rs = np.random.get_state(), random.getstate()
        try: yield
        finally:
            if not self.leak: np.random.set_state(ps); random.setstate(rs)

    def snapshot(self): return self.w.copy()
    def load(self, s): self.w = s.copy()
    def save(self, s, path): return hashlib.sha256(s.tobytes()).hexdigest()


def test_isolation_holds():
    assert H.check_isolation(Toy(), 3)["ok"]


def test_isolation_detects_leak():
    assert not H.check_isolation(Toy(leak=True), 3)["ok"]


def test_protocols():
    r = H.run_one(Toy(n_epochs=12), 0)
    p = r["protocols"]
    assert set(p) == set(H.PROTOCOLS)
    assert p["last"]["epoch"] == 12 and p["budget50"]["epoch"] == 6 and r["budget_epoch"] == 6
    assert set(r["eval_by_epoch"]) == {"1", "2", "3", "5", "10", "12"}
    accs = [h["val_acc"] for h in r["history"]]
    assert p["author"]["epoch"] == int(np.argmax(accs)) + 1           # 严格提升 = 第一个最大值
    losses = r["val_loss"]
    assert p["minvalloss"]["epoch"] == int(np.argmin(losses)) + 1
    # 选中 checkpoint 的评估与该 epoch 的探针一致
    assert p["last"]["eval"] == r["eval_by_epoch"]["12"]


def test_eval_every_epoch_and_odd_budget():
    r = H.run_one(Toy(n_epochs=7), 1, eval_every_epoch=True)
    assert len(r["eval_by_epoch"]) == 7 and r["budget_epoch"] == 4


def test_provenance():
    from . import provenance as P
    r = H.run_one(Toy(n_epochs=12), 0, eval_every_epoch=True)
    pub = r["eval_by_epoch"]["2"]
    t = P.trace(pub, r)
    assert t["d_min"] == 0 and 2 in t["epochs_at_min"] and t["grade"] == "NEAREST"   # 只有准确率时不给 BITWISE
    cm_pub = {"0.0": {"confusion_matrix": [[5, 0], [1, 4]]}}
    run = {"eval_by_epoch": {"1": {"0.0": {"confusion_matrix": [[5, 0], [1, 4]]}}, "2": {"0.0": {"confusion_matrix": [[4, 1], [1, 4]]}}}}
    t = P.trace(cm_pub, run)
    assert t["grade"] == "BITWISE" and t["epochs_at_min"] == [1] and t["rows"][1]["d"] == 0.5


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
