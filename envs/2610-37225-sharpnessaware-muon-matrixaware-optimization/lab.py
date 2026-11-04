import math
import numpy as np
import torch
from sklearn.datasets import load_digits

torch.set_num_threads(1)

_D = load_digits()
_X = (_D.data / 16.0).astype(np.float32)
_Y = _D.target.astype(np.int64)

STEPS = 150
HIDDEN = 64
LR = 0.05
LR_BIAS = 0.05
MU = 0.9
N_TEST = 400
EVAL_RHO = 0.05


def _ns_orth(M, iters=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M / (M.norm() + 1e-7)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    for _ in range(iters):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if tr else X


def _forward(p, x):
    h = torch.relu(x @ p[0] + p[1])
    return h @ p[2] + p[3]


def _loss(p, x, y):
    return torch.nn.functional.cross_entropy(_forward(p, x), y)


def _grads(p, x, y):
    q = [t.detach().clone().requires_grad_(True) for t in p]
    l = _loss(q, x, y)
    return [g.detach() for g in torch.autograd.grad(l, q)]


def _perturbed(p, d, rho):
    # blockwise Frobenius-normalized perturbation (zero if direction is zero)
    out = []
    for t, v in zip(p, d):
        n = v.norm()
        out.append(t + rho * v / n if n > 0 else t.clone())
    return out


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "muon")
    B = int(config.get("batch_size", 64))
    rho = float(config.get("rho", 0.1))
    n_train = int(config.get("train_n", 600))

    # data rng (independent of method)
    drng = np.random.default_rng(seed)
    perm = drng.permutation(len(_X))
    te = perm[:N_TEST]
    tr = perm[N_TEST:N_TEST + n_train]
    Xtr, Ytr = torch.from_numpy(_X[tr]), torch.from_numpy(_Y[tr])
    Xte, Yte = torch.from_numpy(_X[te]), torch.from_numpy(_Y[te])
    brng = np.random.default_rng([seed, 1])
    batches = brng.integers(0, len(tr), size=(STEPS, B))

    # model init rng
    g = torch.Generator().manual_seed(seed)
    p = [
        torch.randn(64, HIDDEN, generator=g) / math.sqrt(64),
        torch.zeros(HIDDEN),
        torch.randn(HIDDEN, 10, generator=g) / math.sqrt(HIDDEN),
        torch.zeros(10),
    ]
    M = [torch.zeros_like(t) for t in p]

    for t in range(STEPS):
        idx = torch.from_numpy(batches[t])
        xb, yb = Xtr[idx], Ytr[idx]
        lr_scale = 0.5 * (1 + math.cos(math.pi * t / STEPS))
        if method == "muon":
            gr = _grads(p, xb, yb)
        elif method == "musam":
            g0 = _grads(p, xb, yb)
            gr = _grads(_perturbed(p, g0, rho), xb, yb)
        elif method == "momusam":
            gr = _grads(_perturbed(p, M, rho), xb, yb)
        else:
            raise ValueError(method)
        with torch.no_grad():
            for i in range(4):
                M[i] = MU * M[i] + (1 - MU) * gr[i]
                if p[i].dim() == 2:
                    O = _ns_orth(M[i])
                    scale = math.sqrt(max(1.0, p[i].shape[1] / p[i].shape[0]))
                    p[i] = p[i] - LR * lr_scale * scale * O
                else:
                    p[i] = p[i] - LR_BIAS * lr_scale * M[i] / (1 - MU ** (t + 1))

    with torch.no_grad():
        lt = _forward(p, Xte)
        test_acc = (lt.argmax(1) == Yte).float().mean().item()
        test_loss = torch.nn.functional.cross_entropy(lt, Yte).item()
        train_acc = (_forward(p, Xtr).argmax(1) == Ytr).float().mean().item()
        base = _loss(p, Xtr, Ytr).item()
    gf = _grads(p, Xtr, Ytr)
    with torch.no_grad():
        sharp = _loss(_perturbed(p, gf, EVAL_RHO), Xtr, Ytr).item() - base

    return {
        "test_acc": float(test_acc),
        "train_acc": float(train_acc),
        "gap": float(train_acc - test_acc),
        "test_loss": float(test_loss),
        "sharpness": float(sharp),
    }
