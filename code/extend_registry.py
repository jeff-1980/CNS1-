"""注册表补登：2×2 起的训练实例（2026-09-25）。只追加、按 instance_id 跳过已有行，可重复运行。
每个实例的权重文件重新计算 SHA256 并与结果 JSON 中记录的值比对；不一致即中止。"""
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import csv, json, hashlib, pathlib, collections, sys
ROOT = pathlib.Path(_ROOT); R = ROOT / "results"; REG = R / "registry/training_instances.csv"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def wpath(p): return (ROOT / "code" / p).resolve() if p.startswith("..") else pathlib.Path(p)
SCRIPT = {"arm2x2": "arm2x2.py", "arm2x2_lt": "arm2x2.py", "ext_e1": "arm_ext.py", "ext_e2": "arm_ext.py", "e4": "e4_paperB_harness.py", "wp1": "wp1_harness.py", "wp2": "pu_wp2.py"}
SH = {k: sha(ROOT / "code" / v)[:16] for k, v in SCRIPT.items()}
NOISE = "add_awgn_tensor (per-window power)"
def snrs(obs): return "|".join(sorted(obs, key=lambda s: -float(s)))
rows = []
def add(**kw): rows.append(kw)
def check(rec, pk, sk):
    p = wpath(rec[pk]); assert p.exists(), p; got = sha(p); assert got == rec[sk], (p, got[:12], rec[sk][:12]); return str(p.relative_to(ROOT)), got
for fn, fam in (("arm2x2.json", "arm2x2"), ("arm2x2_lt.json", "arm2x2_lt")):
    for r in json.load(open(R / fn)):
        wp, ws = check(r, "ckpt", "ckpt_sha256")
        add(instance_id=f"{fn[:-5]}::{r['arm']}::{r['model']}::s{r['seed']}", source_json=fn, role="analysis", dataset="CWRU", subset_protocol=f"40file-pool 2x2 arm={r['arm']} (coverage={r['coverage']}, ratio={r['ratio']})",
            split_seed=42, model=r["model"], seed=r["seed"], seed_type="torch.manual_seed+np.random.seed(seed); eval noise torch.manual_seed(1000+17*seed+rep)", epochs=f"{r['steps']} steps, last weights",
            n_train=r["n_train"], best_val_acc="", noise_impl=NOISE, snr_grid=snrs(r["observed"]), collapse_rule="top_mode over 5 noise draws; conc_mean",
            weights_path=wp, weights_sha256=ws, code_hash_at_run=f"not captured; script sha now {SH[fam]}", code_hash_prior_screen_now="", n_train_source="result json", config_replicate_of="")
for fn, fam in (("ext_e1.json", "ext_e1"), ("ext_e2.json", "ext_e2")):
    for r in json.load(open(R / fn)):
        wp, ws = check(r, "ckpt", "ckpt_sha256")
        rep = f"arm2x2::C007_R31::{r['model']}::s{r['seed']}" if fam == "ext_e1" and r["cell"] == "ID0N0S0" else ""
        add(instance_id=f"{fn[:-5]}::{r['cell']}::{r['model']}::s{r['seed']}", source_json=fn, role="analysis", dataset="CWRU", subset_protocol=f"{r['exp']} cell={r['cell']}",
            split_seed=42, model=r["model"], seed=r["seed"], seed_type="torch.manual_seed+np.random.seed(seed)", epochs=f"{r['steps']} steps ({r['epochs']} epochs), last weights" + ("" if r.get("best_epoch") is None else f"; best epoch {r['best_epoch']}"),
            n_train=r["n_train"], best_val_acc="" if r.get("best_val") is None else r["best_val"], noise_impl=NOISE, snr_grid=snrs(r["final"]), collapse_rule="top_mode over 5 noise draws; conc_mean",
            weights_path=wp, weights_sha256=ws, code_hash_at_run=f"not captured; script sha now {SH[fam]}", code_hash_prior_screen_now="", n_train_source="result json", config_replicate_of=rep)
def harness(fn, fam, dataset, proto, iid_prefix, rep_fn=None):
    for r in json.load(open(R / fn)):
        wb, sb = check(r, "ckpt_best", "ckpt_best_sha256"); wf, sf = check(r, "ckpt_final", "ckpt_final_sha256")
        iid = f"{iid_prefix}::{r['model']}::s{r['seed']}"
        add(instance_id=iid, source_json=fn, role="analysis", dataset=dataset, subset_protocol=proto(r), split_seed=42 if dataset == "CWRU" else "time-split", model=r["model"], seed=r["seed"],
            seed_type="paperB set_seed(seed) via TrainConfig; eval in fork_rng", epochs=f"100 epochs; best epoch {r['best_epoch']} (strict val improvement) + final",
            n_train=r["n_train"], best_val_acc=r["best_val_acc"], noise_impl="paperB evaluate_snr_sweep (awgn)" + ("; NA+ training awgn" if r.get("noise_type") == "awgn" else ""),
            snr_grid="10|8|6|4|2|0|-2|-4|-6|-8", collapse_rule="paperB: mean per-seed conc>=90%", weights_path=f"best={wb} | final={wf}", weights_sha256=f"best={sb} | final={sf}",
            code_hash_at_run=f"not captured; script sha now {SH[fam]}" + (f"; torch {r['torch']}" if r.get("torch") else ""), code_hash_prior_screen_now="", n_train_source="result json",
            config_replicate_of=rep_fn(r) if rep_fn else "")
harness("e4_paperB_harness.json", "e4", "CWRU", lambda r: "16file v2 split, NA- (paperB cwru_clean_v2 protocol)", "e4_paperB_harness")
for rpm in (600, 800, 1000):
    harness(f"wp1_jnu_rpmid_{rpm}.json", "wp1", "JNU", lambda r: f"jnu_modeA_v2 protocol, rpm={r['rpm']} (rpm identification)", f"wp1_jnu_rpmid_{rpm}")
WP1 = {"wp1_cwru_clean.json": ("CWRU", "16file v2, NA- (cwru_clean_v2)"), "wp1_cwru_clean_M.json": ("CWRU", "16file v2, NA- (cwru_clean_v2)"),
       "wp1_cwru_m2.json": ("CWRU", "16file v2, NA- (ablation_v3 mamba2_no_noise_train; unfused path)"), "wp1_cwru_awgn.json": ("CWRU", "16file v2, NA+ awgn (main_awgn_v2)"),
       "wp1_cwru_awgn_M.json": ("CWRU", "16file v2, NA+ awgn (main_awgn_v2)"), "wp1_jnu.json": ("JNU", "jnu_modeA_v2, rpm=600 (behavioral only)"), "wp1_jnu_M.json": ("JNU", "jnu_modeA_v2, rpm=600 (behavioral only)")}
for fn, (ds, pr) in WP1.items():
    harness(fn, "wp1", ds, (lambda pr: (lambda r: pr + ("; unfused" if r.get("unfused") and "unfused" not in pr else "")))(pr), fn[:-5],
            rep_fn=(lambda r: "wp1_jnu_rpmid_600::wdcnn::s0" if r["model"] == "wdcnn" and r["seed"] == 0 else "") if fn == "wp1_jnu.json" else None)
for fn in ("wp2_R.json", "wp2_A.json", "wp2_R_single.json"):
    for r in json.load(open(R / fn)):
        wp, ws = check(r, "ckpt", "ckpt_sha256")
        add(instance_id=f"{fn[:-5]}::{r['arm']}::{r['model']}::s{r['seed']}", source_json=fn, role="analysis", dataset="PU", subset_protocol=f"WP2 block {r['block']} arm {r['arm']}: H=K001+K002, OR={'+'.join(r['OR'])}, IR={'+'.join(r['IR'])}; N15_M07_F10; per-file time split 60/20/20",
            split_seed="time-split; window draw rng(seed, crc32(code))", model=r["model"], seed=r["seed"], seed_type="torch.manual_seed+np.random.seed(seed); eval noise torch.manual_seed(1000+17*seed+rep)",
            epochs=f"{r['steps']} steps, last weights", n_train=r["n_train"], best_val_acc="", noise_impl=NOISE, snr_grid=snrs(r["observed"]), collapse_rule="top_mode over 5 noise draws; conc_mean",
            weights_path=wp, weights_sha256=ws, code_hash_at_run=f"not captured; script sha now {SH['wp2']}", code_hash_prior_screen_now="", n_train_source="result json", config_replicate_of="")
SH["b"] = sha(ROOT / "code/b_confirm.py")[:16]
if (R / "b_confirm.json").exists():
    for r in json.load(open(R / "b_confirm.json")):
        p = pathlib.Path(r["ckpt"]); assert sha(p) == r["ckpt_sha256"], p
        proto = (f"B confirm PU {r['cond']}: R-L1 composition (H=K001+K002, OR=KA04+KA22, IR=KI21+KI14), per-file time split" if r["dataset"] == "PU" else f"B confirm JNU {r['cond']} rpm: modeA split, train <=1200/class")
        add(instance_id=f"b_confirm::{r['dataset']}_{r['cond']}::{r['model']}::s{r['seed']}", source_json="b_confirm.json", role="analysis", dataset=r["dataset"], subset_protocol=proto,
            split_seed="time-split" if r["dataset"] == "PU" else "modeA", model=r["model"], seed=r["seed"], seed_type="torch.manual_seed+np.random.seed(seed); eval noise torch.manual_seed(5000+17*seed+rep)",
            epochs=f"{r['steps']} steps, last weights", n_train=r["n_train"], best_val_acc="", noise_impl=NOISE, snr_grid="-8", collapse_rule="top_mode over 5 draws; conc>=0.60 = attractor",
            weights_path=str(p.relative_to(ROOT)), weights_sha256=r["ckpt_sha256"], code_hash_at_run=f"not captured; script sha now {SH['b']}", code_hash_prior_screen_now="", n_train_source="result json", config_replicate_of="")
SH["e5"] = sha(ROOT / "code/e5_size_id.py")[:16]
if (R / "e5_size_id.json").exists():
    for r in json.load(open(R / "e5_size_id.json")):
        wp, ws = check(r, "ckpt", "ckpt_sha256")
        add(instance_id=f"e5_size_id::{r['set']}::d{r['draw']}::{r['model']}::s{r['seed']}", source_json="e5_size_id.json", role="analysis", dataset="CWRU",
            subset_protocol=f"E5 train+val pool (30 files) set={r['set']} draw={r['draw']} files=" + ";".join(v[0] for v in r["files"].values()),
            split_seed=42, model=r["model"], seed=r["seed"], seed_type="torch.manual_seed+np.random.seed(seed); eval noise torch.manual_seed(1000+17*seed+rep)",
            epochs=f"{r['steps']} steps, last weights", n_train=r["n_train"], best_val_acc="", noise_impl=NOISE, snr_grid=snrs(r["observed"]),
            collapse_rule="top_mode over 5 noise draws; conc_mean", weights_path=wp, weights_sha256=ws, code_hash_at_run=f"script sha {SH['e5']}",
            code_hash_prior_screen_now="", n_train_source="result json", config_replicate_of="")
SH["cinit"] = sha(ROOT / "code/cinit.py")[:16]
for fn in ("cinit_2x2.json", "cinit_e5.json"):
    if not (R / fn).exists(): continue
    for r in json.load(open(R / fn)):
        wp, ws = check(r, "ckpt", "ckpt_sha256")
        key = f"{r['arm']}" if r["exp"] == "C-2x2" else f"{r['set']}::d{r['draw']}"
        add(instance_id=f"{fn[:-5]}::{key}::{r['model']}::s{r['seed']}", source_json=fn, role="analysis", dataset="CWRU",
            subset_protocol=f"{r['exp']} (fixed init: torch.manual_seed(20000+seed) before model construction; init_sha256={r['init_sha256'][:16]}) " + key,
            split_seed=42, model=r["model"], seed=r["seed"], seed_type="init torch.manual_seed(20000+seed); train torch.manual_seed+np.random.seed(seed); eval noise torch.manual_seed(1000+17*seed+rep)",
            epochs=f"{r['steps']} steps, last weights", n_train=r["n_train"], best_val_acc="", noise_impl=NOISE, snr_grid=snrs(r["observed"]),
            collapse_rule="top_mode over 5 noise draws; conc_mean", weights_path=wp, weights_sha256=ws, code_hash_at_run=f"script sha {SH['cinit']}",
            code_hash_prior_screen_now="", n_train_source="result json", config_replicate_of="")
old = list(csv.DictReader(open(REG, newline=""))); cols = list(old[0].keys()); have = {r["instance_id"] for r in old}
ids = [r["instance_id"] for r in rows]; assert len(ids) == len(set(ids)), "duplicate new ids"
new = [r for r in rows if r["instance_id"] not in have]
with open(REG, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\r\n"); [w.writerow({c: r.get(c, "") for c in cols}) for r in new]
tot = list(csv.DictReader(open(REG, newline="")))
print("candidates", len(rows), "| appended", len(new), "| registry rows", len(tot), "| analysis", sum(r["role"] == "analysis" for r in tot), "| smoke", sum(r["role"] == "smoke" for r in tot))
print("by source (new):", dict(collections.Counter(r["source_json"] for r in rows)))
