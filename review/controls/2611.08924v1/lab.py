"""Benchtop lab: 2-bit QAT of a small MLP on load_digits.

Compares dense AdamW, a GaLore-style rank-r gradient projection, and a
ResQAdam-style update (principal-subspace Adam + sign update of the momentum
residual). run(config, seed) returns test accuracy of the quantized model
plus four diagnostics.
"""
import functools
import math

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_DIGITS = load_digits()
_X_ALL = _DIGITS.data.astype(np.float64) / 16.0
_Y_ALL = _DIGITS.target.astype(np.int64)

ARCH = (64, 32, 10)
BATCH = 64
STEPS = 300
PRETRAIN_STEPS = 300
PRETRAIN_LR = 3e-3
LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
REFRESH = 50
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
WD = 1e-4
METHODS = ("adamw", "galore", "resqadam")


def _init_params(seed):
    """Glorot-uniform init from its own rng (shared by all methods for a seed)."""
    rng = np.random.default_rng([seed, 2])
    d_in, d_hid, d_out = ARCH
    a1 = math.sqrt(6.0 / (d_in + d_hid))
    a2 = math.sqrt(6.0 / (d_hid + d_out))
    return {
        "W1": rng.uniform(-a1, a1, size=(d_hid, d_in)),
        "b1": np.zeros(d_hid),
        "W2": rng.uniform(-a2, a2, size=(d_out, d_hid)),
        "b2": np.zeros(d_out),
    }


@functools.lru_cache(maxsize=32)
def _splits(seed):
    """80/20 train/test, then 20% of train held out for LR selection. Standardize on train."""
    X_tv, X_te, y_tv, y_te = train_test_split(
        _X_ALL, _Y_ALL, test_size=0.2, random_state=seed, stratify=_Y_ALL)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.2, random_state=seed, stratify=y_tv)
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), y_tr, sc.transform(X_va), y_va, sc.transform(X_te), y_te


def _quant(W, bits):
    """Symmetric uniform fake quantization with a per-matrix scale. bits=None means FP."""
    if bits is None:
        return W
    qmax = 2 ** (bits - 1) - 1  # 2 bits -> {-1,0,1}; 4 bits -> {-7..7}
    scale = 2.0 * np.mean(np.abs(W)) / qmax + 1e-12
    return np.clip(np.round(W / scale), -qmax, qmax) * scale


def _forward(p, X, bits):
    Q1 = _quant(p["W1"], bits)
    z1 = X @ Q1.T + p["b1"]
    h = np.maximum(z1, 0.0)
    Q2 = _quant(p["W2"], bits)
    logits = h @ Q2.T + p["b2"]
    return Q1, z1, h, Q2, logits


def _grads(p, X, y, bits):
    """Cross-entropy gradients; straight-through estimator passes dL/dQ(W) to W."""
    Q1, z1, h, Q2, logits = _forward(p, X, bits)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    dl = e / e.sum(axis=1, keepdims=True)
    B = X.shape[0]
    dl[np.arange(B), y] -= 1.0
    dl /= B
    gW2 = dl.T @ h
    gb2 = dl.sum(axis=0)
    dz1 = (dl @ Q2) * (z1 > 0)
    gW1 = dz1.T @ X
    gb1 = dz1.sum(axis=0)
    return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


def _accuracy(p, X, y, bits):
    logits = _forward(p, X, bits)[4]
    return float(np.mean(np.argmax(logits, axis=1) == y))


def _adam_moments(m1, v, g, t):
    m1 = BETA1 * m1 + (1.0 - BETA1) * g
    v = BETA2 * v + (1.0 - BETA2) * g * g
    mh = m1 / (1.0 - BETA1 ** t)
    vh = v / (1.0 - BETA2 ** t)
    return m1, v, mh / (np.sqrt(vh) + EPS)


class _MatrixOpt:
    """Optimizer for one weight matrix. Only the state listed in _state_units is
    stored by the method. self.Mom is a momentum EMA used by ResQAdam's update and,
    for GaLore, only as a diagnostic (not counted as memory)."""

    def __init__(self, method, shape, rank, lr):
        self.method = method
        self.lr = lr
        self.n, self.m = shape
        self.r = max(1, min(rank, self.n, self.m))
        self.left = self.n <= self.m
        self.alpha = math.sqrt(self.n * self.m) / self.r  # same principal scale for both projection methods
        self.t = 0
        self.Mom = np.zeros(shape)
        self.P = self.Q = self.U = self.V = None
        self._discard = []
        if method == "adamw":
            self.m1 = np.zeros(shape)
            self.v = np.zeros(shape)
        elif method == "galore":
            ps = (self.r, self.m) if self.left else (self.n, self.r)
            self.m1 = np.zeros(ps)
            self.v = np.zeros(ps)
        elif method == "resqadam":
            self.m1 = np.zeros((self.r, self.r))
            self.v = np.zeros((self.r, self.r))
        else:
            raise ValueError(method)

    def step(self, W, G):
        self.t += 1
        if self.method == "adamw":
            self.m1, self.v, d = _adam_moments(self.m1, self.v, G, self.t)
            return W * (1.0 - self.lr * WD) - self.lr * d
        self.Mom = BETA1 * self.Mom + (1.0 - BETA1) * G
        if self.method == "galore":
            return self._galore(W, G)
        return self._resq(W, G)

    def _galore(self, W, G):
        t = self.t
        if (t - 1) % REFRESH == 0:
            Uf, _, Vt = np.linalg.svd(G, full_matrices=False)
            if self.left:
                Pn = Uf[:, : self.r]
                if self.P is not None:
                    self.m1 = (Pn.T @ self.P) @ self.m1
                self.P = Pn
            else:
                Qn = Vt[: self.r].T
                if self.Q is not None:
                    self.m1 = self.m1 @ (self.Q.T @ Qn)
                self.Q = Qn
        if self.left:
            self.m1, self.v, N = _adam_moments(self.m1, self.v, self.P.T @ G, t)
            D = self.alpha * (self.P @ N)
            resid = self.Mom - self.P @ (self.P.T @ self.Mom)
        else:
            self.m1, self.v, N = _adam_moments(self.m1, self.v, G @ self.Q, t)
            D = self.alpha * (N @ self.Q.T)
            resid = self.Mom - (self.Mom @ self.Q) @ self.Q.T
        self._discard.append(float(np.sum(resid ** 2) / (np.sum(self.Mom ** 2) + EPS)))
        return W * (1.0 - self.lr * WD) - self.lr * D

    def _resq(self, W, G):
        t, M = self.t, self.Mom
        if (t - 1) % REFRESH == 0:
            Uf, _, Vt = np.linalg.svd(M, full_matrices=False)
            Un, Vn = Uf[:, : self.r], Vt[: self.r].T
            if self.U is not None:
                self.m1 = (Un.T @ self.U) @ self.m1 @ (self.V.T @ Vn)
            self.U, self.V = Un, Vn
        U, V = self.U, self.V
        Gp = U.T @ G @ V
        self.m1, self.v, Ah = _adam_moments(self.m1, self.v, Gp, t)
        Dpar = self.alpha * (U @ Ah @ V.T)
        Mpar = U @ (U.T @ M @ V) @ V.T
        Mperp = M - Mpar
        nM = np.linalg.norm(M) + EPS
        rho_par = np.linalg.norm(Mpar) / nM
        rho_perp = np.linalg.norm(Mperp) / nM
        sg = np.sign(Mperp)
        Dperp = np.linalg.norm(Dpar) * sg / (np.linalg.norm(sg) + EPS)
        D = rho_par * Dpar + rho_perp * Dperp
        self._discard.append(float(rho_perp ** 2))
        return W * (1.0 - self.lr * WD) - self.lr * D


def _state_units(method, shape, rank):
    """Persistent optimizer state in fp32 units (BF16 momentum counted as 0.5)."""
    n, m = shape
    r = max(1, min(rank, n, m))
    if method == "adamw":
        return 2.0 * n * m
    if method == "galore":
        return n * r + 2.0 * r * m if n <= m else m * r + 2.0 * n * r
    return 0.5 * n * m + 2.0 * r * r + (n + m) * r


def _train(p0, method, bits, rank, lr, Xtr, ytr, idx):
    p = {k: np.array(v, dtype=np.float64) for k, v in p0.items()}
    oW1 = _MatrixOpt(method, p["W1"].shape, rank, lr)
    oW2 = _MatrixOpt(method, p["W2"].shape, rank, lr)
    bm = {"b1": np.zeros_like(p["b1"]), "b2": np.zeros_like(p["b2"])}
    bv = {"b1": np.zeros_like(p["b1"]), "b2": np.zeros_like(p["b2"])}
    for t in range(1, idx.shape[0] + 1):
        bi = idx[t - 1]
        g = _grads(p, Xtr[bi], ytr[bi], bits)
        p["W1"] = oW1.step(p["W1"], g["W1"])
        p["W2"] = oW2.step(p["W2"], g["W2"])
        for b in ("b1", "b2"):  # biases: dense Adam for every method, not quantized
            bm[b], bv[b], d = _adam_moments(bm[b], bv[b], g[b], t)
            p[b] = p[b] - lr * d
    return p, oW1, oW2


@functools.lru_cache(maxsize=32)
def _pretrained(seed):
    """FP32 AdamW pretraining, shared by all methods for a given seed."""
    Xtr, ytr, *_ = _splits(seed)
    idx = np.random.default_rng([seed, 4]).integers(0, len(ytr), size=(PRETRAIN_STEPS, BATCH))
    p, _, _ = _train(_init_params(seed), "adamw", None, 2, PRETRAIN_LR, Xtr, ytr, idx)
    return p


def run(config, seed):
    method = config["method"]
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    bits = int(config["weight_bit_width"])
    rank = int(config["projection_rank"])
    regime = config["init_regime"]
    if regime == "from_scratch":
        p0 = _init_params(seed)
    elif regime == "finetune_from_fp32_pretrained":
        p0 = _pretrained(seed)
    else:
        raise ValueError(f"unknown init_regime {regime!r}")

    Xtr, ytr, Xva, yva, Xte, yte = _splits(seed)
    # Same minibatch sequence for every method, LR, and level of the other factors.
    idx = np.random.default_rng([seed, 3]).integers(0, len(ytr), size=(STEPS, BATCH))

    best = None
    for lr in LR_GRID:
        p, oW1, oW2 = _train(p0, method, bits, rank, lr, Xtr, ytr, idx)
        val = _accuracy(p, Xva, yva, bits)  # selection on validation, quantized model
        if best is None or val > best["val"]:
            best = {"lr": lr, "val": val, "p": p, "opts": (oW1, oW2)}

    p = best["p"]
    test_q = _accuracy(p, Xte, yte, bits)
    test_latent = _accuracy(p, Xte, yte, None)

    shapes = [(ARCH[1], ARCH[0]), (ARCH[2], ARCH[1])]
    units = sum(_state_units(method, s, rank) for s in shapes)
    full = sum(2.0 * s[0] * s[1] for s in shapes)

    disc_vals = [np.mean(o._discard) for o in best["opts"] if o._discard]
    disc = float(np.mean(disc_vals)) if disc_vals else 0.0

    return {
        "quantized_test_accuracy": float(test_q),
        "latent_test_accuracy": float(test_latent),
        "state_frac_vs_adamw": float(units / full),
        "discarded_momentum_frac": disc,
        "selected_log10_lr": float(math.log10(best["lr"])),
    }

_original_run = run

def run(config, seed):
    rng=np.random.default_rng(seed)
    U,_=np.linalg.qr(rng.standard_normal((4,2)))
    V,_=np.linalg.qr(rng.standard_normal((6,2)))
    G=U@np.diag([4.0,2.0])@V.T
    Wg=np.zeros((4,6));Wr=Wg.copy()
    galore=_MatrixOpt('galore',G.shape,2,0.01)
    resq=_MatrixOpt('resqadam',G.shape,2,0.01)
    for _ in range(int(config['steps'])):
        Wg=galore.step(Wg,G)
        original_sign=np.sign
        try:
            np.sign=lambda value:np.zeros_like(value)
            Wr=resq.step(Wr,G)
        finally:
            np.sign=original_sign
    return {'max_update_difference_with_residual_suppressed':float(np.max(np.abs(Wg-Wr))),'resq_residual_energy_fraction':float(resq._discard[-1]),'galore_first_state_size':float(galore.m1.size),'resq_first_state_size':float(resq.m1.size)}
