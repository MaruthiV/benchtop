import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

torch.set_num_threads(1)

_D = load_digits()
_X_ALL = (_D.data / 16.0).astype(np.float32)
_Y_ALL = _D.target.astype(np.int64)

N_TRAIN = 1200
STEPS = 60
BATCH = 128
LR = 0.1
MERGE_K = 10        # last K iterates averaged into the anchor
ALPHA = 0.2         # weight on worst-case FIM-trace term
DAMP = 1e-3
HID = 32
DRIFT_REL = 0.3     # relative norm of random drift
N_DRIFT = 3


def _forward(p, X):
    W1, b1, W2, b2 = p
    z1 = X @ W1 + b1
    a1 = torch.relu(z1)
    return z1, a1, a1 @ W2 + b2


def _fim_terms(p, X, y):
    z1, a1, z2 = _forward(p, X)
    W2 = p[2]
    d2 = torch.softmax(z2, 1) - F.one_hot(y, 10).float()
    d1 = (d2 @ W2.T) * (z1 > 0).float()
    return a1, d2, d1


def _fim_trace(p, X, y):
    a1, d2, d1 = _fim_terms(p, X, y)
    tr = ((a1 ** 2).sum(1) + 1) * (d2 ** 2).sum(1) + ((X ** 2).sum(1) + 1) * (d1 ** 2).sum(1)
    return tr.mean()


def _fim_diag(p, X, y):
    with torch.no_grad():
        a1, d2, d1 = _fim_terms(p, X, y)
        n = X.shape[0]
        return [(X ** 2).T @ (d1 ** 2) / n, (d1 ** 2).mean(0),
                (a1 ** 2).T @ (d2 ** 2) / n, (d2 ** 2).mean(0)]


def _acc(p, X, y):
    with torch.no_grad():
        return (_forward(p, X)[2].argmax(1) == y).float().mean().item()


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "ewc")
    lam = float(config.get("ewc_lambda", 10))
    T = int(config.get("n_tasks", 5))
    rho = float(config.get("rho", 0.05))

    rng_data = np.random.default_rng(seed)
    idx = rng_data.permutation(len(_Y_ALL))
    tr_i, te_i = idx[:N_TRAIN], idx[N_TRAIN:]
    perms = [np.arange(64)] + [rng_data.permutation(64) for _ in range(T - 1)]
    mu = _X_ALL[tr_i].mean(0)
    sd = _X_ALL[tr_i].std(0) + 1e-3
    Xtr0 = (_X_ALL[tr_i] - mu) / sd
    Xte0 = (_X_ALL[te_i] - mu) / sd
    ytr = torch.tensor(_Y_ALL[tr_i])
    yte = torch.tensor(_Y_ALL[te_i])
    Xtr = [torch.tensor(Xtr0[:, pm]) for pm in perms]
    Xte = [torch.tensor(Xte0[:, pm]) for pm in perms]

    g = torch.Generator().manual_seed(seed + 1)
    theta = [torch.randn(64, HID, generator=g) * np.sqrt(2 / 64), torch.zeros(HID),
             torch.randn(HID, 10, generator=g) * np.sqrt(2 / HID), torch.zeros(10)]
    rng_batch = np.random.default_rng(seed + 2)
    rng_drift = np.random.default_rng(seed + 3)

    use_fs = method == "fs_ewc"
    use_merge = method in ("fs_ewc", "merge_only")
    Lam = [torch.zeros_like(t) for t in theta]
    anchor = [t.clone() for t in theta]
    clampc = 0.9 / LR
    acc_after = []

    for t in range(T):
        X = Xtr[t]
        th = [x.clone() for x in theta]
        acc_sum = [torch.zeros_like(x) for x in th]
        ncount = 0
        for s in range(STEPS):
            b = torch.tensor(rng_batch.choice(N_TRAIN, BATCH, replace=False))
            xb, yb = X[b], ytr[b]
            ps = [x.clone().requires_grad_() for x in th]
            loss = F.cross_entropy(_forward(ps, xb)[2], yb)
            grads = list(torch.autograd.grad(loss, ps))
            if use_fs:
                ps2 = [x.clone().requires_grad_() for x in th]
                gt = torch.autograd.grad(_fim_trace(ps2, xb, yb), ps2)
                nrm = torch.sqrt(sum((q ** 2).sum() for q in gt)) + 1e-12
                ps3 = [(x + rho * q / nrm).detach().requires_grad_() for x, q in zip(th, gt)]
                h = torch.autograd.grad(_fim_trace(ps3, xb, yb), ps3)
                grads = [a + ALPHA * c for a, c in zip(grads, h)]
            if t > 0:
                grads = [a + torch.clamp(lam * L, max=clampc) * (x - an)
                         for a, L, x, an in zip(grads, Lam, th, anchor)]
            th = [(x - LR * a).detach() for x, a in zip(th, grads)]
            if use_merge and s >= STEPS - MERGE_K:
                acc_sum = [a + x for a, x in zip(acc_sum, th)]
                ncount += 1
        theta = [a / ncount for a in acc_sum] if use_merge else th
        anchor = [x.clone() for x in theta]
        # consolidate: normalized + damped empirical FIM at retained anchor
        Fd = _fim_diag(theta, X, ytr)
        mx = max(f.max() for f in Fd) + 1e-12
        Lam = [L + f / mx + DAMP for L, f in zip(Lam, Fd)]
        acc_after.append(_acc(theta, Xte[t], yte))

    final = [_acc(theta, Xte[k], yte) for k in range(T)]
    avg_final = float(np.mean(final))
    forgetting = float(np.mean([acc_after[k] - final[k] for k in range(T - 1)]))

    tn = float(torch.sqrt(sum((x ** 2).sum() for x in theta)))
    drift = []
    for _ in range(N_DRIFT):
        noise = [torch.tensor(rng_drift.standard_normal(x.shape).astype(np.float32)) for x in theta]
        nn_ = float(torch.sqrt(sum((q ** 2).sum() for q in noise)))
        pert = [x + DRIFT_REL * tn * q / nn_ for x, q in zip(theta, noise)]
        drift.append(np.mean([_acc(pert, Xte[k], yte) for k in range(T)]))

    return {
        "avg_final_acc": avg_final,
        "forgetting": forgetting,
        "last_task_acc": float(final[-1]),
        "first_task_final_acc": float(final[0]),
        "drift_acc": float(np.mean(drift)),
    }
