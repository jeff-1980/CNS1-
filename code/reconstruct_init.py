"""审稿二轮 R1：重建每个历史训练实例的实际初始权重（只读、不训练）。
事实依据：模型在 train_*() 设 torch.manual_seed(seed) 之前构造；评估把 CPU 生成器重置为 manual_seed(1000+17·seed+r)，噪声在 GPU 上生成，
因此评估结束后 CPU 生成器状态 = manual_seed(1000+17·seed+4)，下一个实例的初始权重只取决于 (架构, 上一个实例的种子) 或"新进程"。
按日志中的实际构造顺序，在新子进程中逐段重放，得到每个实例的 init_sha256。"""
import sys, os, re, json, subprocess, collections
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "../logs"); R = os.path.join(HERE, "../results")
PY = sys.executable
PAT = re.compile(r"^--- (\S+)(?: d(\d))? (\w+) (?:seed|s)(\d+)")
def segments(log, split_marks=("###",)):
    segs, cur = [], []
    for line in open(os.path.join(L, log)):
        if line.startswith("###") and cur: segs.append(cur); cur = []
        m = PAT.match(line)
        if m: cur.append(dict(cell=m.group(1), draw=None if m.group(2) is None else int(m.group(2)), model=m.group(3), seed=int(m.group(4))))
    if cur: segs.append(cur)
    return segs
EXPS = {"arm2x2": ("arm2x2.log", 4, "arm2x2"), "arm2x2_lt": ("ext_e3.log", 4, "arm2x2"), "ext_e1": ("ext_e1.log", 4, "arm_ext"), "ext_e2": ("ext_e2.log", 4, "arm_ext"),
        "wp2": ("wp2.log", 3, "pu_wp2"), "e5": ("e5.log", 4, "e5_size_id")}
CHILD = r'''
import sys, json, hashlib, io, torch
sys.path.insert(0, {here!r}); sys.path.insert(0, {pbx!r})
import {mod}
from prior_screen import REGISTRY
out = []
for j in json.loads(sys.argv[1]):
    net = REGISTRY[j["model"]]({ncls}); buf = io.BytesIO()
    h = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
    out.append(h); torch.manual_seed(1000 + 17 * j["seed"] + 4)
print(json.dumps(out))
'''
res = {}
for exp, (log, ncls, mod) in EXPS.items():
    rows = []
    for k, seg in enumerate(segments(log)):
        code = CHILD.format(here=HERE, pbx=os.path.join(HERE, "pbx"), mod=mod, ncls=ncls)
        p = subprocess.run([PY, "-c", code, json.dumps(seg)], capture_output=True, text=True, cwd=HERE)
        hs = json.loads(p.stdout.strip().splitlines()[-1])
        for i, (j, h) in enumerate(zip(seg, hs)):
            rows.append(dict(**j, process=k, pos=i, prev_seed=None if i == 0 else seg[i - 1]["seed"], init_sha256=h))
    # E5：进程 1 中断的实例在进程 2/3 里重训，保留每个键最后一次构造
    last = {}
    for r in rows: last[(r["cell"], r["draw"], r["model"], r["seed"])] = r
    rows = list(last.values())
    g = collections.defaultdict(set)
    for r in rows: g[(r["model"], r["seed"])].add(r["init_sha256"])
    shared = sum(len(v) == 1 for v in g.values())
    res[exp] = dict(n=len(rows), n_model_seed=len(g), model_seed_all_arms_same_init=shared, rows=rows,
                    distinct_inits_per_model_seed={f"{m}|s{s}": len(v) for (m, s), v in sorted(g.items())})
    print(f"{exp:10s} n={len(rows):3d} (model,seed) groups={len(g):2d} with one shared init across all arms: {shared}")
json.dump(res, open(os.path.join(R, "init_reconstruction.json"), "w"), indent=1)
