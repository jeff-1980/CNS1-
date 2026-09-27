"""验证初始化重建：对选定 E5 实例，用重建的初始权重、相同数据与种子重训，比较最终权重张量与存档权重是否逐位一致。"""
import sys, os, json, hashlib, torch, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "pbx"))
torch.backends.cudnn.benchmark = False
import e5_size_id as E
from prior_screen import REGISTRY
from arm2x2 import train_last, ROOT40
from data.splits_v2 import build_cwru_splits
FRESH = torch.get_rng_state()
th = lambda sd: hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in sd.values())).hexdigest()
rec = json.load(open(os.path.join(HERE, "../results/init_reconstruction.json")))["e5"]["rows"]
d = {(r["set"], r["draw"], r["model"], r["seed"]): r for r in json.load(open(os.path.join(HERE, "../results/e5_size_id.json")))}
b = build_cwru_splits(ROOT40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42); pool = E.pool_index(b); picks = {dr: E.draw_files(pool, dr) for dr in E.DRAWS}
out = []
KEYS = [("S014", 0, "wdcnn", 0), ("S_ALL", 2, "wdcnn", 1), ("S007", 1, "wdcnn", 2)]
for key in KEYS:
    rr = next(r for r in rec if (r["cell"], r["draw"], r["model"], r["seed"]) == key)
    # 重放：新进程默认状态（pos 0）或 manual_seed(1000+17·prev+4)
    if rr["prev_seed"] is None: torch.set_rng_state(FRESH)
    else: torch.manual_seed(1000 + 17 * rr["prev_seed"] + 4)
    net = REGISTRY[key[2]](4); assert th(net.state_dict()) == rr["init_sha256"], ("init mismatch", key)
    X, y, _ = E.build(pool, picks[key[1]], key[0], key[3], key[1])
    net, steps, _ = train_last(net, torch.from_numpy(X), torch.from_numpy(y), torch.device("cuda"), key[3], epochs=E.EPOCHS, log=lambda x: None)
    saved = torch.load(d[key]["ckpt"] if os.path.isabs(d[key]["ckpt"]) else os.path.join(HERE, d[key]["ckpt"]), map_location="cpu")
    same = th(net.state_dict()) == th(saved)
    maxdiff = max(float((a.detach().cpu().float() - saved[k].float()).abs().max()) for k, a in net.state_dict().items())
    out.append(dict(key=key, prev_seed=rr["prev_seed"], bitwise=same, max_abs_diff=maxdiff)); print(out[-1], flush=True)
json.dump(out, open(os.path.join(HERE, "../results/init_recon_validation.json"), "w"), indent=1)
