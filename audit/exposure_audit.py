"""CWRU 同源记录审计：重放每个历史训练实例的取样，统计其训练集中含多少个与测试集逐字节相同的 Normal 窗口。
事实：loader 取 99.mat 的首个 DE 键 X098_DE_time（=98.mat 信号）；split_seed=42 下 99 在 train、98 在 test。
只读：不训练、不推理、不改动任何结果文件。输出 audit/exposure_by_instance.csv 与 audit/exposure_summary.json。
"""
import sys, os, json, csv, hashlib, collections
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.join(HERE, "../code"); R = os.path.join(HERE, "../results")
sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(CODE, "pbx"))
import numpy as np
from data.splits_v2 import build_cwru_splits
from arm2x2 import build_arm, ROOT40
import e5_size_id as E
from arm_ext import pool_of, e1_indices, e2_indices
ROOT16 = os.environ.get("CWRU16_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/cwru_16"))

def h(a): return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
b40 = build_cwru_splits(ROOT40, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
b16 = build_cwru_splits(ROOT16, loads=(0, 1, 2, 3), segment_len=1024, overlap=0.5, split_seed=42)
TEST = {s: {h(x) for x, y in zip(b.X["test"], b.y["test"]) if y == 0} for s, b in (("40", b40), ("16", b16))}
assert len(TEST["40"]) == 944 and len(TEST["16"]) == 944

def expo(X, y, setname):
    hs = [h(x) for x, yy in zip(X, y) if yy == 0]
    hit = [q for q in hs if q in TEST[setname]]
    return dict(n_normal_train=len(hs), n_normal_from_test_copy=len(hit), distinct_test_windows_in_train=len(set(hit)))

rows = []
def add(exp, src, rec, X, y, setname, md5_ok=None):
    rows.append(dict(exp=exp, source_json=src, arm=rec.get("arm") or rec.get("set") or rec.get("cell") or "", draw=rec.get("draw", ""),
                     model=rec["model"], seed=rec["seed"], md5_replay_match=md5_ok, test_set=setname, **expo(X, y, setname)))

P40 = pool_of(b40); P16 = pool_of(b16)
pool2 = (P40["fids"], P40["X"], P40["y"])
for src in ("arm2x2.json", "arm2x2_lt.json", "cinit_2x2.json"):
    for r in json.load(open(os.path.join(R, src))):
        X, y, _, idx = build_arm(pool2, r["arm"], r["seed"])
        ok = (hashlib.md5(idx.tobytes()).hexdigest() == r["idx_md5"]) if "idx_md5" in r else None
        assert list(np.bincount(y, minlength=4)) == r["class_counts"], (src, r["arm"], r["seed"])
        add({"arm2x2.json": "2x2", "arm2x2_lt.json": "E3 (2x2 lstm/transformer)", "cinit_2x2.json": "C-INIT 2x2"}[src], src, r, X, y, "40", ok)

pool5 = E.pool_index(b40)
for src in ("e5_size_id.json", "cinit_e5.json"):
    for r in json.load(open(os.path.join(R, src))):
        pick = E.draw_files(pool5, r["draw"]); X, y, used = E.build(pool5, pick, r["set"], r["seed"], r["draw"])
        assert used["Normal"][0] == r["files"]["Normal"][0], (src, used["Normal"], r["files"]["Normal"])
        add("E5" if src.startswith("e5") else "C-INIT E5", src, r, X, y, "40", None)

for r in json.load(open(os.path.join(R, "ext_e1.json"))):
    pool, idx = e1_indices(P40, P16, r["ID"], r["N"], r["seed"]); P = P40 if pool == "P40" else P16
    add("E1", "ext_e1.json", r, P["X"][idx], P["y"][idx], "40" if pool == "P40" else "16", hashlib.md5(np.asarray(idx).tobytes()).hexdigest() == r["idx_md5"])
for r in json.load(open(os.path.join(R, "ext_e2.json"))):
    idx, _ = e2_indices(P40, r["cell"], r["seed"])
    add("E2", "ext_e2.json", r, P40["X"][idx], P40["y"][idx], "40", hashlib.md5(np.asarray(idx).tobytes()).hexdigest() == r["idx_md5"])

# reference protocol (paperB training functions): full 16-file train split
full16 = expo(b16.X["train"], b16.y["train"], "16")
for src, exp in (("e4_paperB_harness.json", "E4 reference protocol"), ("wp1_cwru_clean.json", "WP1 CWRU clean"), ("wp1_cwru_awgn.json", "WP1 CWRU awgn"),
                 ("wp1_cwru_clean_M.json", "WP1 CWRU clean (Mamba)"), ("wp1_cwru_m2.json", "WP1 CWRU mamba2"), ("wp1_cwru_awgn_M.json", "WP1 CWRU awgn (Mamba)")):
    p = os.path.join(R, src)
    if not os.path.exists(p): continue
    for r in json.load(open(p)):
        rows.append(dict(exp=exp, source_json=src, arm="", draw="", model=r["model"], seed=r["seed"], md5_replay_match=None, test_set="16", **full16))

os.makedirs(HERE, exist_ok=True)
with open(os.path.join(HERE, "exposure_by_instance.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
S = collections.defaultdict(list)
for r in rows: S[r["exp"]].append(r)
summ = {}
for k, v in S.items():
    d = [x["distinct_test_windows_in_train"] for x in v]; fr = [x["n_normal_from_test_copy"] / max(1, x["n_normal_train"]) for x in v]
    md = [x["md5_replay_match"] for x in v if x["md5_replay_match"] is not None]
    summ[k] = dict(n=len(v), n_exposed=sum(x > 0 for x in d), distinct_test_windows_min=min(d), distinct_test_windows_max=max(d),
                   share_of_normal_train_that_is_test_copy=[round(min(fr), 3), round(max(fr), 3)], idx_md5_replay=f"{sum(md)}/{len(md)}" if md else "n/a")
    if k in ("E5", "C-INIT E5"):
        summ[k]["by_draw"] = {dr: dict(n=sum(1 for x in v if x["draw"] == dr), exposed=sum(1 for x in v if x["draw"] == dr and x["distinct_test_windows_in_train"] > 0)) for dr in (0, 1, 2)}
json.dump(dict(fact="99.mat first DE key = X098_DE_time, byte-identical to 98.mat; split_seed=42 puts 99 in train, 98 in test (both 40- and 16-file sets); 944/944 Normal test windows have byte-identical copies in the train split",
               per_experiment=summ), open(os.path.join(HERE, "exposure_summary.json"), "w"), indent=1)
for k, v in summ.items(): print(f"{k:28s} {v}")
