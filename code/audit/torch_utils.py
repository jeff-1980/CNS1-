"""适配器共用的 torch 工具：RNG 与模型模式隔离、权重快照与哈希。"""
import contextlib, hashlib, io, random
import numpy as np, torch


@contextlib.contextmanager
def isolate(models, device):
    """保存并恢复 torch（CPU + 当前 CUDA）、numpy、python random 状态，以及各模型的 train/eval 模式。
    DataLoader 每次迭代都会从 torch 默认生成器取 base_seed，所以即使不打乱也要隔离。"""
    ps, rs = np.random.get_state(), random.getstate()
    modes = [m.training for m in models]
    devs = [torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devs):
            yield
    finally:
        np.random.set_state(ps); random.setstate(rs)
        for m, t in zip(models, modes): m.train(t)


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_sha256(state):
    """与落盘方式无关的权重哈希：按键名排序后逐张量哈希原始字节。"""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def save(state, path):
    """path=None 只算哈希；否则 torch.save 并返回张量哈希（文件字节哈希另记在 ckpt 旁）。"""
    if path is not None:
        torch.save(state, path)
        with open(path, "rb") as f: fh = hashlib.sha256(f.read()).hexdigest()
        with open(path + ".sha256", "w") as f: f.write(fh + "\n")
    return state_sha256(state)
