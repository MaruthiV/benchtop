import warnings
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

N_SENS = 8
N_T = 12
N_GAS = 3
N_CLS = 6          # 3 single gases + 3 binary mixtures
PAIRS = [(0, 1), (0, 2), (1, 2)]
D_EXP = 2000       # expansion size
K_WTA = 100        # 5% active
FAN_IN = 10        # inputs sampled per expansion unit
N_TEST_PER_CLASS = 100
P_GRID = (0.05, 0.2, 1.0)
TIMES = np.arange(1, N_T + 1) * 1.5


def _make_world(rng):
    G = rng.lognormal(mean=-0.2, sigma=0.5, size=(N_GAS, N_SENS))
    tau = rng.uniform(8.0, 25.0, size=N_SENS)
    signs = rng.choice([-1.0, 1.0], size=(len(PAIRS), N_SENS))
    return G, tau, signs


def _steady(world, cls, nl, rng):
    G, tau, signs = world
    n = len(cls)
    r = np.zeros((n, N_SENS))
    for i, c in enumerate(cls):
        conc = rng.uniform(0.6, 1.4, size=2)
        if c < N_GAS:
            lin = conc[0] * G[c]
            inter = 0.0
        else:
            a, b = PAIRS[c - N_GAS]
            lin = conc[0] * G[a] + conc[1] * G[b]
            inter = conc[0] * conc[1] * signs[c - N_GAS] * np.sqrt(G[a] * G[b])
        A = 1.5
        r[i] = (1 - nl) * lin + nl * (A * np.tanh(lin / A) + 0.8 * inter)
    return r


def _sample(world, cls, nl, noise, rng):
    G, tau, signs = world
    n = len(cls)
    r = _steady(world, cls, nl, rng)
    prev = rng.integers(0, N_CLS, size=n)
    x0 = _steady(world, prev, nl, rng) * rng.uniform(0.3, 1.0, size=(n, 1))
    decay = np.exp(-TIMES[None, :] / tau[:, None])  # (S, T)
    x = r[:, :, None] + (x0 - r)[:, :, None] * decay[None]
    x = x * (1 + 0.1 * rng.standard_normal((n, N_SENS, 1)))
    x = x + noise * 0.5 * rng.standard_normal((n, N_SENS, 1))
    x = x + noise * rng.standard_normal(x.shape)
    return x.reshape(n, -1)


def _encode(X, M):
    act = X @ M.T
    idx = np.argpartition(-act, K_WTA, axis=1)[:, :K_WTA]
    code = np.zeros(act.shape, dtype=bool)
    np.put_along_axis(code, idx, True, axis=1)
    return code


def _hebb_fit(codes, y, U, p):
    W = np.zeros((N_CLS, codes.shape[1]), dtype=bool)
    for c in range(N_CLS):
        m = y == c
        W[c] = (codes[m] & (U[m] < p)).any(axis=0)
    return W


def _hebb_predict(codes, W):
    ov = codes.astype(np.float32) @ W.T.astype(np.float32)
    ov = ov / np.sqrt(W.sum(1) + 1.0)[None, :]
    return ov.argmax(1)


def run(config: dict, seed: int) -> dict:
    warnings.filterwarnings("ignore")
    method = config.get("method", "expand_sparsify")
    nl = float(config.get("mixture_nonlinearity", 0.5))
    noise = float(config.get("sensor_noise", 0.2))
    n_per = int(config.get("n_per_class", 20))

    # ---- data (independent of method) ----
    drng = np.random.default_rng(seed)
    world = _make_world(drng)
    yte = np.repeat(np.arange(N_CLS), N_TEST_PER_CLASS)
    Xte = _sample(world, yte, nl, noise, drng)
    ytr = np.repeat(np.arange(N_CLS), n_per)
    Xtr = _sample(world, ytr, nl, noise, drng)

    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd

    # ---- model-side randomness (independent of method for shared parts) ----
    mrng = np.random.default_rng(seed + 10007)
    d = Ztr.shape[1]
    sel = np.argpartition(mrng.random((D_EXP, d)), FAN_IN, axis=1)[:, :FAN_IN]
    M = np.zeros((D_EXP, d))
    np.put_along_axis(M, sel, 1.0, axis=1)
    U = mrng.random((len(ytr), D_EXP))
    perm = mrng.permutation(len(ytr))
    cv_seed = int(mrng.integers(0, 2**31 - 1))

    if method == "linear_svm":
        gs = GridSearchCV(LinearSVC(dual=True, max_iter=3000),
                          {"C": [0.01, 0.1, 1.0]},
                          cv=StratifiedKFold(3, shuffle=True, random_state=cv_seed))
        gs.fit(Ztr, ytr)
        pred = gs.predict(Zte)
        chosen = float(gs.best_params_["C"])
    elif method == "rbf_svm":
        gs = GridSearchCV(SVC(kernel="rbf"),
                          {"C": [1.0, 10.0], "gamma": [0.3 / d, 1.0 / d, 3.0 / d]},
                          cv=StratifiedKFold(3, shuffle=True, random_state=cv_seed))
        gs.fit(Ztr, ytr)
        pred = gs.predict(Zte)
        chosen = float(gs.best_params_["C"])
    elif method == "expand_sparsify":
        Ctr = _encode(Ztr, M)
        Cte = _encode(Zte, M)
        nv = int(0.3 * len(ytr))
        va, tr = perm[:nv], perm[nv:]
        best_p, best_acc = P_GRID[0], -1.0
        for p in P_GRID:
            W = _hebb_fit(Ctr[tr], ytr[tr], U[tr], p)
            acc = float((_hebb_predict(Ctr[va], W) == ytr[va]).mean())
            if acc > best_acc:
                best_acc, best_p = acc, p
        W = _hebb_fit(Ctr, ytr, U, best_p)
        pred = _hebb_predict(Cte, W)
        chosen = float(best_p)
    else:
        raise ValueError(f"unknown method {method}")

    correct = pred == yte
    mix = yte >= N_GAS
    return {
        "mixture_acc": float(correct[mix].mean()),
        "single_acc": float(correct[~mix].mean()),
        "overall_acc": float(correct.mean()),
        "chosen_hparam": chosen,
    }
