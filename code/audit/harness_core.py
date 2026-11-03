"""审计 harness 核心（与框架无关）。按 results/PREREG_AUDIT_pilot.md §3.2–§3.4。

每篇被审论文写一个小适配器（见 Adapter），本文件负责：
- 原样驱动作者的训练循环（适配器的 train_epoch / validate 就是作者代码，属于原始轨迹）
- 在线跟踪 4 个选模协议（§3.3）：author（作者默认规则）/ last / minvalloss / budget50
- 在 PROBE_EPOCHS 上做完整测试评估（§3.4 溯源用）；所有额外评估都在 adapter.isolate() 里，
  不改变训练轨迹（check_isolation 可在目标机器上实测这一点）
- 每个协议选中的 checkpoint 落盘并记 SHA-256
本文件不 import torch，因此可在无 GPU 的环境里用 test_harness_core.py 验证逻辑。"""
import json, math, os, time

PROTOCOLS = ("author", "last", "minvalloss", "budget50")
PROBE_EPOCHS = {1, 2, 3, 5, 10, 25, 50}   # 另加最后一轮


class Adapter:
    """被审论文的适配器接口。所有方法都应尽量直接调用作者代码。"""
    name = "base"
    n_epochs = 0

    def setup(self, seed):
        """按作者代码设种子、构建数据 / 模型 / 优化器。"""
        raise NotImplementedError

    def train_epoch(self, epoch):
        """作者代码的一个 epoch（含学习率调度）。返回 dict（如 {"loss": ...}）。"""
        raise NotImplementedError

    def validate(self, epoch):
        """作者代码自己的验证步骤（属于原始轨迹，不隔离）。返回 {"acc": float, ...}。"""
        raise NotImplementedError

    def author_selects(self, history):
        """作者默认的保存规则：给定到本轮为止的 validate 记录，本轮是否保存。"""
        raise NotImplementedError

    def val_loss(self):
        """验证集损失（作者训练损失函数），在 isolate() 里调用；作者代码不算时由适配器补算。None = 不可得。"""
        return None

    def evaluate(self):
        """测试集完整评估（含混淆矩阵 / 逐档曲线），在 isolate() 里调用。返回可 JSON 序列化的 dict。"""
        raise NotImplementedError

    def isolate(self):
        """上下文管理器：进入时保存全部 RNG 与模型模式，退出时恢复。"""
        raise NotImplementedError

    def snapshot(self):
        """当前权重的独立拷贝（CPU）。"""
        raise NotImplementedError

    def load(self, state):
        raise NotImplementedError

    def save(self, state, path):
        """落盘并返回 SHA-256。"""
        raise NotImplementedError

    def meta(self):
        """写入结果的附加信息（代码版本、数据切分摘要等）。"""
        return {}


def budget_epoch(n_epochs):
    """固定预算 = 作者默认总 epoch 的 50%（向上取整，至少 1）。"""
    return max(1, math.ceil(0.5 * n_epochs))


def run_one(ad, seed, ckpt_dir=None, eval_every_epoch=False, probe=True):
    """跑一个 (适配器, 种子)。返回结果 dict。probe=False 只用于 check_isolation 的对照臂。"""
    ad.setup(seed)
    n = ad.n_epochs
    b_ep = budget_epoch(n)
    hist, val_losses, by_epoch = [], [], {}
    sel = {p: None for p in PROTOCOLS}          # 协议 -> (epoch, state)
    min_loss = math.inf
    for ep in range(1, n + 1):
        tr = ad.train_epoch(ep)
        v = ad.validate(ep)
        hist.append(dict(epoch=ep, **{k: _num(x) for k, x in (tr or {}).items() if k != "epoch"}, **{"val_" + k: _num(x) for k, x in v.items()}))
        if ad.author_selects(hist):
            sel["author"] = (ep, ad.snapshot())
        if probe:
            with ad.isolate():
                vl = ad.val_loss()
            val_losses.append(None if vl is None else float(vl))
            if vl is not None and vl < min_loss:   # 严格下降，取最早
                min_loss = vl; sel["minvalloss"] = (ep, ad.snapshot())
            if eval_every_epoch or ep in PROBE_EPOCHS or ep == n:
                with ad.isolate():
                    by_epoch[str(ep)] = ad.evaluate()
        if ep == b_ep:
            sel["budget50"] = (ep, ad.snapshot())
    sel["last"] = (n, ad.snapshot())
    final_state = sel["last"][1]
    out = dict(adapter=ad.name, seed=seed, n_epochs=n, budget_epoch=b_ep, history=hist,
               val_loss=val_losses, eval_by_epoch=by_epoch, protocols={}, meta=ad.meta())
    if not probe:
        out["final_sha256"] = ad.save(final_state, None)
        return out
    for p in PROTOCOLS:
        if sel[p] is None:
            out["protocols"][p] = dict(epoch=None, status="unavailable"); continue
        ep, st = sel[p]
        ad.load(st)
        with ad.isolate():
            ev = by_epoch.get(str(ep)) or ad.evaluate()
        rec = dict(epoch=ep, eval=ev)
        if ckpt_dir:
            path = os.path.join(ckpt_dir, f"{ad.name}_s{seed}_{p}.pt")
            rec["ckpt"] = path; rec["sha256"] = ad.save(st, path)
        out["protocols"][p] = rec
    ad.load(final_state)
    out["final_sha256"] = ad.save(final_state, None)
    return out


def check_isolation(ad, seed, eval_every_epoch=True):
    """同一种子跑两遍：有探针 / 无探针。最终权重哈希相同 = 探针没有改变训练轨迹。"""
    a = run_one(ad, seed, probe=True, eval_every_epoch=eval_every_epoch)
    b = run_one(ad, seed, probe=False)
    ok = a["final_sha256"] == b["final_sha256"] and [h for h in a["history"]] == [h for h in b["history"]]
    return dict(ok=ok, with_probe=a["final_sha256"], without_probe=b["final_sha256"])


def run_many(ad, seeds, out_path, ckpt_dir=None, eval_every_epoch=False):
    """断点续跑：已完成的种子跳过；每完成一个种子原子写回。"""
    if ckpt_dir: os.makedirs(ckpt_dir, exist_ok=True)
    res = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["adapter"], r["seed"]) for r in res}
    for s in seeds:
        if (ad.name, s) in done: continue
        t0 = time.time(); print(f"--- {ad.name} seed{s} ---", flush=True)
        rec = run_one(ad, s, ckpt_dir=ckpt_dir, eval_every_epoch=eval_every_epoch)
        rec["wall_s"] = round(time.time() - t0, 1); res.append(rec)
        tmp = out_path + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, out_path)
        print(f"    done {len(res)} wall={rec['wall_s']:.0f}s", flush=True)
    return res


def _num(x):
    try: return round(float(x), 6)
    except (TypeError, ValueError): return x
