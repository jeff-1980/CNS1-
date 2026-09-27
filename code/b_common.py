"""B 线共用函数：谱平坦度、纯噪声探针、C-a / C-b / C-c、−8 dB 宏类结局。按 PREREG_B_hierarchical_attractor.md。"""
import numpy as np, torch, torch.nn as nn
from data.noise_augment import add_awgn_tensor
dev = torch.device("cuda")
def sf(x):
    P = torch.fft.rfft(x.reshape(len(x), -1).double(), dim=-1).abs().pow(2)[:, 1:] + 1e-20
    return (P.log().mean(-1).exp() / P.mean(-1)).float()
def as3(X): X = torch.as_tensor(np.asarray(X, np.float32)); return (X[:, None] if X.dim() == 2 else X).to(dev)
def rule_a(Xtr, ytr):
    """C-a：干净训练窗各类平均 SF；Healthy(0) 最高 → 'H'。只依赖训练数据。"""
    Xtr = as3(Xtr); ytr = torch.as_tensor(np.asarray(ytr)).to(dev); s = torch.cat([sf(Xtr[i:i + 2048]) for i in range(0, len(Xtr), 2048)])
    m = {int(c): float(s[ytr == c].mean()) for c in torch.unique(ytr)}
    return ("H" if max(m, key=m.get) == 0 else "F"), m
def last_linear(net):
    L = [m for m in net.modules() if isinstance(m, nn.Linear)]; return L[-1]
def feats_preds(net, X, bs=512):
    buf = []; h = last_linear(net).register_forward_hook(lambda m, i, o: buf.append(i[0].detach().float().reshape(len(i[0]), -1)))
    ps = []
    with torch.no_grad():
        for i in range(0, len(X), bs): ps.append(net(X[i:i + bs]).argmax(1))
    h.remove(); return torch.cat(buf), torch.cat(ps)
def probes(Xc, seed, n=2000):
    Xc = Xc[:n]; P = Xc.pow(2).mean(-1, keepdim=True); g = torch.Generator(device=dev).manual_seed(9000 + seed)
    return torch.randn(Xc.shape, generator=g, device=dev) * P.sqrt()
def rule_bc(net, Xc, yc, seed):
    """C-b：纯噪声探针投票；C-c：探针质心 vs 干净校准窗类质心（最后线性层输入）。"""
    yc = torch.as_tensor(np.asarray(yc)).to(dev); Z = probes(Xc, seed)
    fz, pz = feats_preds(net, Z); fc, _ = feats_preds(net, Xc)
    frac_h = float((pz == 0).float().mean())
    cen = {int(c): fc[yc == c].mean(0) for c in torch.unique(yc)}; mu = fz.mean(0)
    dist = {c: float((mu - v).norm()) for c, v in cen.items()}
    dh = dist[0]; df = min(v for c, v in dist.items() if c != 0)
    return dict(b=("H" if frac_h > 0.5 else "F"), frac_h=frac_h, c=("H" if dh < df else "F"), d_h=dh, d_f=df)
def outcome(net, Xt, seed, ncls, snr=-8.0, nrep=5):
    tops, concs = [], []
    with torch.no_grad():
        for rep in range(nrep):
            torch.manual_seed(5000 + 17 * seed + rep); Xn = add_awgn_tensor(Xt, snr)
            p = torch.cat([net(Xn[i:i + 512]).argmax(1) for i in range(0, len(Xn), 512)]); c = torch.bincount(p, minlength=ncls)
            tops.append(int(c.argmax())); concs.append(float(c.max()) / len(p))
    top = int(np.bincount(tops, minlength=ncls).argmax()); conc = float(np.mean(concs))
    return dict(top=top, conc=conc, macro=("H" if top == 0 else "F"), attractor=conc >= 0.60)
