"""
prior_screen.py — 先验预筛实验

命题：只用【干净训练集】的倒数第二层特征几何，能否在见到任何噪声之前
预测出该模型将来在低 SNR 下的崩溃靶标类？

与 paperB 已发表内容的界限：
  paperB 的 P1(collapse_ratio) / P2(argmax W·z_noise) 全部在【含噪特征】上测量 → 事后诊断。
  本实验只用【干净训练特征】 → 部署前预筛。这是唯一的新主张。

预测量（全部只用干净训练集）：
  G1  argmin_k ||m_k - mu||          几何中心度（含偏置分类头适用）
  G2  argmax_k <u_k, mu>             全局均值方向对齐（无偏置分类头适用）
  G3  argmax_k <W_k, mu> + b_k       干净特征全局均值处的 logit（含偏置）
  G4  argmin_k ||m_k||               类均值范数最小者（对照）
  G5  argmax_k b_k                   纯偏置（对照：分类头先验）

真值：该模型在低 SNR 测试集上、预测集中度 >= 90% 时的主预测类。
"""
from __future__ import annotations
import os as _os
_ROOT = _os.environ.get("CNS_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PAPERB_CODE = _os.environ.get("PAPERB_CODE", _os.path.join(_ROOT, "external/paperB_code"))
_PAPERB_RESULTS = _os.environ.get("PAPERB_RESULTS", _os.path.join(_ROOT, "external/paperB_results"))
_CWRU40 = _os.environ.get("CWRU40_DIR", _os.path.join(_ROOT, "data/cwru_12k_de"))
_CWRU16 = _os.environ.get("CWRU16_DIR", _os.path.join(_ROOT, "data/cwru_16"))
_PU = _os.environ.get("PU_DIR", _os.path.join(_ROOT, "data/paderborn"))
_JNU = _os.environ.get("JNU_DIR", _os.path.join(_ROOT, "data/jnu"))
_JNU_RAW = _os.environ.get("JNU_RAW_DIR", _JNU)
import os, sys, json, time, argparse, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# pbx/ = paperB 代码的子集：data/ 与 splits_v2 逐字节相同（MD5 已校验），
# 只把 models/__init__.py 换成不含 mamba_ssm 的导入清单（该扩展需 CUDA 编译，本机不可用）。
CODE = Path("pbx")
sys.path.insert(0, str(CODE))

from data.splits_v2 import (build_cwru_splits, build_jnu_modeA_splits,
                            verify_no_leakage)
from data.noise_augment import add_awgn_tensor
# 逐模块导入：models/__init__.py 会连带 import mamba_ssm（本机无此扩展）
from models.cnn_1d import CNN1D
from models.wdcnn import WDCNN
from models.lstm_model import LSTMClassifier
from models.transformer_1d import Transformer1D
from models.drsn import DRSN
_HAS_MAMBA = False   # vibrmamba / mamba2 需 mamba_ssm（CUDA 扩展），本机不可用

# WSL 侧的完好副本（40/40 可读）。Obsidian 库里那份 输入/论文8/data/cwru_12k_de
# 有 16 个 0 字节残留文件（Normal 全部 + 各类 0.007" 档），已核实不可用。
CWRU_ROOT = os.environ.get("CWRU_ROOT", _CWRU40)  # 环境变量可覆盖（16 文件子集判决实验）
# JNU：论文CNS/data/jnu 是 cp -L 拷入的完好副本（12 csv, 75M）。
# JNU 原始目录可用 JNU_DIR / JNU_RAW_DIR 指定。
JNU_ROOT = _JNU
CLASS_NAMES = {0: "Normal", 1: "InnerRace", 2: "OuterRace", 3: "Ball"}

# mamba2 需要 mamba-ssm 扩展，本机没有 → 用可复现的 6 架构
# num_classes 参数化：JNU/CWRU = 4 类，PU = 3 类（PU 无 Ball 类，KB* 复合已排除）。
REGISTRY = {
    "cnn1d":       lambda K=4: CNN1D(in_channels=1, d_model=64, num_classes=K),
    "wdcnn":       lambda K=4: WDCNN(in_channels=1, d_model=64, num_classes=K),
    "lstm":        lambda K=4: LSTMClassifier(in_channels=1, d_model=64, num_classes=K, hidden=128),
    "transformer": lambda K=4: Transformer1D(in_channels=1, d_model=64, num_classes=K,
                                             nhead=8, dim_feedforward=256),
    "drsn":        lambda K=4: DRSN(in_channels=1, d_model=64, num_classes=K, num_blocks=4),
}



def get_head(model: nn.Module) -> nn.Linear:
    """所有 6 个架构的分类头都叫 classifier 且是 nn.Linear。"""
    head = getattr(model, "classifier", None)
    assert isinstance(head, nn.Linear), f"unexpected head: {type(head)}"
    return head


@torch.no_grad()
def penultimate(model: nn.Module, X: torch.Tensor, device, bs=256) -> np.ndarray:
    """通过 forward hook 抓分类头的输入 = 倒数第二层特征。"""
    head = get_head(model)
    buf = []
    h = head.register_forward_hook(lambda m, inp, out: buf.append(inp[0].detach().cpu()))
    model.eval()
    for i in range(0, len(X), bs):
        model(X[i:i+bs].to(device))
    h.remove()
    return torch.cat(buf).numpy()


def train_one(model, Xtr, ytr, Xva, yva, device, epochs=40, bs=64, lr=1e-3,
              wd=1e-4, noise_aware=False, seed=0, log=None):
    """按 paperB 的 TrainConfig 口径训练（cosine 调度 + grad clip）。
    noise_aware=False 复现「clean training」条件 —— 那是崩溃发生的条件。"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # paperB TrainConfig 口径：cosine + warmup_epochs=5（此前缺 warmup，已补）
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # paperB 用 class-weighted loss。CWRU 不平衡比 1.17 时影响可忽略，
    # 但 JNU 是 3.005（多数类 Normal），这一条会实质影响结果 —— 必须加权。
    cnt = torch.bincount(ytr, minlength=int(ytr.max().item()) + 1).float()
    cw = (cnt.sum() / (len(cnt) * cnt.clamp(min=1))).to(device)
    lossf = nn.CrossEntropyLoss(weight=cw)
    n = len(Xtr); best_va, best_state = -1.0, None
    train_snrs = [-6., -4., -2., 0., 2., 4., 6., 10.]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            if noise_aware:
                snr = float(np.random.choice(train_snrs))
                xb = add_awgn_tensor(xb, snr)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(Xva), 256):
                pv.append(model(Xva[i:i+256].to(device)).argmax(1).cpu())
            va = (torch.cat(pv) == yva).float().mean().item()
        if va > best_va:
            best_va = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log and (ep + 1) % 10 == 0:
            log(f"      ep{ep+1:3d} val={va:.4f} best={best_va:.4f}")

    model.load_state_dict(best_state)
    return model, best_va


def prior_predictors(feats: np.ndarray, labels: np.ndarray, head: nn.Linear, K=4) -> dict:
    """只用干净训练特征 + 训练好的分类头权重，算 5 个候选预测量。"""
    W = head.weight.detach().cpu().numpy()                       # [K, d]
    b = (head.bias.detach().cpu().numpy() if head.bias is not None
         else np.zeros(K, dtype=np.float32))
    mu = feats.mean(axis=0)                                      # 全局均值
    M = np.stack([feats[labels == k].mean(axis=0) for k in range(K)])   # 类均值 [K, d]
    Mc = M - mu
    U = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-12)  # 单位化中心化类均值

    return dict(
        G1_argmin_dist   = int(np.argmin(np.linalg.norm(Mc, axis=1))),
        G2_argmax_align  = int(np.argmax(U @ mu)),
        G3_logit_at_mu   = int(np.argmax(W @ mu + b)),
        G4_argmin_norm   = int(np.argmin(np.linalg.norm(M, axis=1))),
        G5_argmax_bias   = int(np.argmax(b)),
        has_bias         = head.bias is not None,
        dist_to_mu       = np.linalg.norm(Mc, axis=1).tolist(),
        logits_at_mu     = (W @ mu + b).tolist(),
        align_u_mu       = (U @ mu).tolist(),
        bias             = b.tolist(),
        # NC 诊断：类均值几何离 simplex ETF 有多远
        nc_gram          = (U @ U.T).tolist(),
    )


@torch.no_grad()
def observed_target(model, Xte, yte, device, snrs, conc_thresh=0.90, K=4) -> dict:
    """真值：低 SNR 下崩溃事件的主预测类（paperB 的 concentration>=90% 判据）。"""
    model.eval()
    out = {}
    for snr in snrs:
        preds = []
        for i in range(0, len(Xte), 256):
            xb = add_awgn_tensor(Xte[i:i+256].to(device), snr)
            preds.append(model(xb).argmax(1).cpu())
        p = torch.cat(preds).numpy()
        cnt = np.bincount(p, minlength=K)
        conc = cnt.max() / len(p)
        out[f"{snr:.1f}"] = dict(
            accuracy=float((p == yte.numpy()).mean()),
            top_pred=int(cnt.argmax()),
            concentration=float(conc),
            collapsed=bool(conc >= conc_thresh),
            pred_distribution=cnt.tolist(),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)   # paperB TrainConfig 口径
    ap.add_argument("--dataset", choices=["cwru", "jnu", "pu"], default="cwru")
    # 类平衡干预（仅 jnu）：测「类不平衡 → 种子方差份额」的因果。
    # 三臂设计与理由见 jnu_balance.py 的模块文档。
    ap.add_argument("--arm", choices=["orig", "balanced", "subsample"],
                    default="orig")
    ap.add_argument("--rpm", type=int, default=600)       # JNU Mode A：已发表用 600
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=list(REGISTRY))
    ap.add_argument("--out", default="prior_screen_results.json")
    ap.add_argument("--snrs", type=float, nargs="+", default=None,
                    help="覆盖评测 SNR 网格（dB）。默认用数据集的已发表口径。")
    ap.add_argument("--ckpt-dir", default=None,
                    help="给定则把训练好的权重存到此目录，换 SNR 网格时可复用。")
    ap.add_argument("--pu-op", default="N15_M07_F10",
                    help="PU 工况（默认 1500rpm/0.7Nm/1000N，PU 文档标准工况）。")
    ap.add_argument("--pu-files", type=int, default=6,
                    help="PU 每个轴承个体取多少个 .mat（每个 = 4s @64kHz）。")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logf = open("prior_screen.log", "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"=== device={device} torch={torch.__version__} ===")

    if args.dataset == "pu":
        # 阶段 1′ 的第二个数据集。按轴承个体分组切分（train/val/test 用不同
        # 个体，比 JNU 的时间轴切分更严），且严格类平衡 —— 这正是
        # VERDICT_cns_value §5 风险 2 要求检验的条件。
        # sys.path[0] 已被 pbx/ 占据（见文件头），需显式加回脚本自身目录
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pu_splits import build_pu_time_splits, CLASS_NAMES_PU
        # 与 JNU Mode A 同口径（时间轴 60/20/20 + gap）。个体分组切分下
        # 模型训不起来（val 0.243 < 随机 0.333），见 pu_splits 文档字符串。
        bundle = build_pu_time_splits(op=args.pu_op, segment_len=1024,
                                      overlap=0.5, codes_per_class=3,
                                      files_per_code=args.pu_files,
                                      balanced=True)
        CLASS_NAMES = dict(CLASS_NAMES_PU)
        NUM_CLASSES = 3
        SNRS = [-8., -4., 0.]
        log(f"PU 个体分配: {bundle.pu_assign}")
    elif args.dataset == "cwru":
        bundle = build_cwru_splits(CWRU_ROOT, loads=(0, 1, 2, 3), segment_len=1024,
                                  overlap=0.5, split_seed=42)
        SNRS = [-8., -6., -4., -2., 0.]
    else:
        # JNU Mode A：单转速内时间轴连续划分。rpm=600 由用户确认，
        # 且测试集类分布 [581,190,190,190] 与已发表混淆矩阵行和逐格匹配。
        bundle = build_jnu_modeA_splits(JNU_ROOT, rpm=args.rpm,
                                       segment_len=1024, overlap=0.5)
        SNRS = [-8., -4., 0.]          # 对齐已发表 jnu_modeA_v2 的 EVAL_SNRS
    if args.dataset != "pu":
        CLASS_NAMES = {0: "Normal", 1: "InnerRace", 2: "OuterRace", 3: "Ball"}
        NUM_CLASSES = 4
    if args.snrs is not None:           # 细网格检验用；默认仍为已发表口径
        SNRS = sorted(args.snrs, reverse=True)
        log(f"SNR 网格被覆盖为 {SNRS}（{len(SNRS)} 档）")
    verify_no_leakage(bundle, f"{args.dataset}{args.rpm if args.dataset=='jnu' else ''}")
    # 类平衡干预：只筛训练集，val/test 保持原分布 —— 否则评测口径变了，
    # onset 与 orig 臂不可比。加权损失会在平衡后自动变为全 1 权重。
    # 下采样是 per-seed 的（见 jnu_balance 文档）：若全 run 共用一次抽样，
    # 该抽样就成了一个固定的额外设计因子，会把方差归属推向"架构"。
    # 因此筛选放在 run 循环内，此处只做前置检查。
    _rebal = None
    if args.arm != "orig":
        if args.dataset != "jnu":
            raise SystemExit("--arm 目前只为 jnu 实现（依赖其 3:1:1:1 结构）")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from jnu_balance import rebalance_indices, describe
        _rebal = rebalance_indices

    Xtr = torch.from_numpy(bundle.X["train"]); ytr = torch.from_numpy(bundle.y["train"])
    Xtr_full, ytr_full = Xtr, ytr
    Xva = torch.from_numpy(bundle.X["val"]);   yva = torch.from_numpy(bundle.y["val"])
    Xte = torch.from_numpy(bundle.X["test"]);  yte = torch.from_numpy(bundle.y["test"])
    log(f"train={tuple(Xtr.shape)} val={tuple(Xva.shape)} test={tuple(Xte.shape)}")
    log(f"train class counts={np.bincount(ytr.numpy(), minlength=NUM_CLASSES).tolist()}"
        f"{' (筛选前；每 run 的实际计数见下方 arm= 行)' if _rebal else ''}  "
        f"test={np.bincount(yte.numpy(), minlength=NUM_CLASSES).tolist()}")

    results = []
    for mname in args.models:
        for seed in args.seeds:
            t0 = time.time()
            log(f"\n--- {mname} seed{seed} ---")
            if _rebal is not None:
                k = _rebal(ytr_full.numpy(), args.arm, seed)
                Xtr, ytr = Xtr_full[k], ytr_full[k]
                log(f"      arm={args.arm} "
                    f"{describe(ytr_full.numpy(), k)}")
            model = REGISTRY[mname](NUM_CLASSES)
            model, best_va = train_one(model, Xtr, ytr, Xva, yva, device,
                                       epochs=args.epochs, seed=seed,
                                       noise_aware=False, log=log)
            # 先验量：只看干净训练集
            F = penultimate(model, Xtr, device)
            pri = prior_predictors(F, ytr.numpy(), get_head(model), K=NUM_CLASSES)
            # 真值：含噪测试
            obs = observed_target(model, Xte, yte, device, SNRS, K=NUM_CLASSES)
            # c = p/n 的两个因子都要落盘：p=倒数第二层维度，n=训练样本数。
            # BBP 阈值 (1+sqrt(c))^2 需要它们，事后无法从旧结果反推。
            rec = dict(model=mname, seed=seed, best_val_acc=best_va,
                       feat_dim=int(F.shape[1]), n_train=int(Xtr.shape[0]),
                       aspect_c=float(F.shape[1]) / float(Xtr.shape[0]),
                       snr_grid=[float(s) for s in SNRS],
                       prior=pri, observed=obs,
                       wall_s=round(time.time() - t0, 1))
            if args.ckpt_dir:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(args.ckpt_dir, f"{mname}_s{seed}.pt"))
            results.append(rec)
            coll = [(s, v["top_pred"]) for s, v in obs.items() if v["collapsed"]]
            log(f"    val={best_va:.4f} G1={CLASS_NAMES[pri['G1_argmin_dist']]} "
                f"G2={CLASS_NAMES[pri['G2_argmax_align']]} G3={CLASS_NAMES[pri['G3_logit_at_mu']]} "
                f"| collapsed cells={[(s, CLASS_NAMES[t]) for s, t in coll]} "
                f"| {rec['wall_s']}s")
            json.dump(results, open(args.out, "w"), indent=1)

    log(f"\n=== done: {len(results)} runs → {args.out} ===")


if __name__ == "__main__":
    main()
