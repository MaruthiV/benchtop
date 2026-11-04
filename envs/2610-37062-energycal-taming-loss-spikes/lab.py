import math
import numpy as np
import torch
from sklearn.datasets import load_digits

_D = load_digits()
_X = (_D.data / 16.0).astype(np.float32)
_Y = _D.target.astype(np.int64)

STEPS = 250
BATCH = 64
D_MODEL = 32
D_HID = 32
TOPK = 2
BETA = 0.95
WD = 0.05
WARM = 10
RHO_REF = 0.8
P_EXP = 1.0
UNIFORM_Q = 0.6
WINDOW = 20
SPIKE_MULT = 1.5
LOSS_CAP = 20.0
ADAM_LR = 0.003
NS_COEF = (3.4445, -4.7750, 2.0315)


def _ns(A):
    # Batched 5-step quintic Newton-Schulz orthogonalization over the last two dims.
    X = A / (A.flatten(-2).norm(dim=-1)[..., None, None] + 1e-7)
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.transpose(-2, -1)
    a, b, c = NS_COEF
    for _ in range(5):
        S = X @ X.transpose(-2, -1)
        X = a * X + (b * S + c * (S @ S)) @ X
    if tr:
        X = X.transpose(-2, -1)
    return X


def _forward(P, x, E):
    z = torch.relu(x @ P["Win"])
    logits = z @ P["Wr"]
    tv, ti = torch.topk(logits, TOPK, dim=1)
    gate = torch.zeros_like(logits).scatter(1, ti, torch.softmax(tv, dim=1))
    h = torch.relu(torch.einsum("bd,edh->ebh", z, P["W1"]))
    o = torch.einsum("ebh,ehd->ebd", h, P["W2"])
    z = z + torch.einsum("be,ebd->bd", gate, o)
    return z @ P["Wout"]


def _spike_burden(L):
    T = len(L)
    tot, nflag = 0.0, 0
    mx = 1.0
    for t in range(WINDOW, T):
        m = float(np.median(L[t - WINDOW:t]))
        if L[t] > SPIKE_MULT * m:
            tot += L[t] - m
            nflag += 1
        mx = max(mx, L[t] / max(m, 1e-6))
    n = max(T - WINDOW, 1)
    return tot / n, nflag / n, mx


def run(config: dict, seed: int) -> dict:
    torch.set_num_threads(1)
    method = config.get("method", "muon")
    lr = float(config.get("lr", 0.1))
    E = int(config.get("n_experts", 8))

    # ---- data (independent rng) ----
    drng = np.random.default_rng(seed)
    perm = drng.permutation(len(_X))
    ntr = 1200
    tr_idx, te_idx = perm[:ntr], perm[ntr:]
    Xtr, Ytr = torch.from_numpy(_X[tr_idx]), torch.from_numpy(_Y[tr_idx])
    Xte, Yte = torch.from_numpy(_X[te_idx]), torch.from_numpy(_Y[te_idx])
    brng = np.random.default_rng(seed + 7919)
    batches = brng.integers(0, ntr, size=(STEPS, BATCH))

    # ---- model init (independent generator) ----
    g = torch.Generator().manual_seed(seed + 104729)
    def rn(*s, fan):
        return torch.randn(*s, generator=g) / math.sqrt(fan)
    P = {
        "Win": rn(64, D_MODEL, fan=64),
        "Wr": rn(D_MODEL, E, fan=D_MODEL),
        "W1": rn(E, D_MODEL, D_HID, fan=D_MODEL),
        "W2": rn(E, D_HID, D_MODEL, fan=D_HID),
        "Wout": rn(D_MODEL, 10, fan=D_MODEL),
    }
    for v in P.values():
        v.requires_grad_(True)
    names = list(P.keys())
    muon_names = ["Win", "W1", "W2", "Wout"]
    routed = {"W1", "W2"}
    mom = {k: torch.zeros_like(P[k]) for k in muon_names}
    am = torch.zeros_like(P["Wr"])
    av = torch.zeros_like(P["Wr"])

    losses = []
    qsum, qn = 0.0, 0
    diverged = False
    for t in range(STEPS):
        idx = torch.from_numpy(batches[t])
        out = _forward(P, Xtr[idx], E)
        loss = torch.nn.functional.cross_entropy(out, Ytr[idx])
        lv = float(loss.detach())
        if not math.isfinite(lv) or lv > LOSS_CAP:
            lv = LOSS_CAP
        losses.append(lv)
        if not math.isfinite(float(loss)):
            diverged = True
            break
        grads = torch.autograd.grad(loss, [P[k] for k in names])
        gd = dict(zip(names, grads))
        gn = math.sqrt(sum(float((x ** 2).sum()) for x in gd.values()))
        sc = min(1.0, 1.0 / (gn + 1e-12))
        eta = lr * min(1.0, (t + 1) / WARM)
        with torch.no_grad():
            for k in muon_names:
                G = gd[k] * sc
                mom[k] = BETA * mom[k] + G
                A = G + BETA * mom[k]
                D = _ns(A)
                if k in routed:
                    if method == "energycal":
                        d = min(D.shape[-2], D.shape[-1])
                        rho = (D ** 2).flatten(-2).sum(-1) / d
                        q = torch.clamp((rho / RHO_REF) ** P_EXP, 0.0, 1.0)
                        qsum += float(q.mean())
                        qn += 1
                        D = D * q[:, None, None]
                    elif method == "uniform_scale":
                        D = D * UNIFORM_Q
                        qsum += UNIFORM_Q
                        qn += 1
                    else:
                        qsum += 1.0
                        qn += 1
                P[k].mul_(1.0 - eta * WD).sub_(eta * D)
            # Adam fallback for router
            G = gd["Wr"] * sc
            am.mul_(0.9).add_(0.1 * G)
            av.mul_(0.999).add_(0.001 * G * G)
            mh = am / (1 - 0.9 ** (t + 1))
            vh = av / (1 - 0.999 ** (t + 1))
            P["Wr"].sub_(ADAM_LR * mh / (vh.sqrt() + 1e-8))
        if not all(bool(torch.isfinite(v).all()) for v in P.values()):
            diverged = True
            break

    while len(losses) < STEPS:
        losses.append(LOSS_CAP)

    burden, frac, mx = _spike_burden(losses)

    with torch.no_grad():
        if diverged:
            te_loss, te_acc = LOSS_CAP, 0.1
        else:
            out = _forward(P, Xte, E)
            te_loss = float(torch.nn.functional.cross_entropy(out, Yte))
            if not math.isfinite(te_loss):
                te_loss = LOSS_CAP
            te_loss = min(te_loss, LOSS_CAP)
            te_acc = float((out.argmax(1) == Yte).float().mean())

    return {
        "spike_burden": float(burden),
        "final_test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "spike_frac": float(frac),
        "mean_q": float(qsum / qn) if qn else 1.0,
    }
