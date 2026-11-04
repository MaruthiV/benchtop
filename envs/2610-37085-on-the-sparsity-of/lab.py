import math
import numpy as np

D = 128
DA = 80                 # task A uses coordinates [0, 80)
B_LO = 48               # task B uses coordinates [48, 128); overlap = [48, 80)
K = 4
STEPS = 300
LR_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
N_VAL = 200
N_TEST = 300
REWARD_NOISE = 0.5
INIT_STD = 0.3
SPARSE_THRESH = 0.05


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _gen_task(rng, n, lo, hi, wstar, rho):
    """Sparse binary features; a shared (per-prompt) component cancels in pair differences.
    Uniform draws are thresholded by rho, so data draws are common across density levels."""
    dc = hi - lo
    u_sh = rng.random((n, dc))
    u_own = rng.random((n, K, dc))
    noise = rng.normal(0.0, REWARD_NOISE, (n, K))
    a = rng.integers(0, K, n)
    off = rng.integers(1, K, n)
    b = (a + off) % K
    flip = rng.random(n) < 0.0  # placeholder (kept for stream stability)
    shared = (u_sh < rho).astype(float)
    own = (u_own < rho).astype(float)
    scale = 1.0 / math.sqrt(max(1.0, 2.0 * rho * dc))
    phi = (shared[:, None, :] + own) * scale
    X = np.zeros((n, K, D))
    X[:, :, lo:hi] = phi
    r_true = X @ wstar
    r_noisy = r_true + noise
    idx = np.arange(n)
    a_wins = r_noisy[idx, a] >= r_noisy[idx, b]
    w = np.where(a_wins, a, b)
    l = np.where(a_wins, b, a)
    return X, r_true, w, l


def _train(method, X, w, l, beta, lr, w_idx_unused=None):
    n = X.shape[0]
    idx = np.arange(n)
    Xw = X[idx, w]
    delta = np.zeros(D)
    if method == "sft":
        X2 = X.reshape(n * K, D)
        Xwm = Xw.mean(axis=0)
        # reference logits are constant offsets; handled via theta0 in caller
        return X2, Xwm
    return None


def _fit(method, X, w, l, beta, lr, theta0):
    n = X.shape[0]
    idx = np.arange(n)
    delta = np.zeros(D)
    if method == "sft":
        X2 = X.reshape(n * K, D)
        base = (X2 @ theta0).reshape(n, K)
        Xw_mean = X[idx, w].mean(axis=0)
        for _ in range(STEPS):
            logits = base + (X2 @ delta).reshape(n, K)
            p = _softmax(logits)
            g = -(Xw_mean - (p.reshape(-1) @ X2) / n)
            delta -= lr * g
    else:
        Dm = X[idx, w] - X[idx, l]
        for _ in range(STEPS):
            m = beta * (Dm @ delta)
            g = -(beta / n) * (_sigmoid(-m) @ Dm)
            delta -= lr * g
    return delta


def _coeffs(method, X, w, l, beta, theta0, delta):
    n = X.shape[0]
    idx = np.arange(n)
    if method == "sft":
        p = _softmax(X @ (theta0 + delta))
        return 1.0 - p[idx, w]
    Dm = X[idx, w] - X[idx, l]
    m = beta * (Dm @ delta)
    return beta * _sigmoid(-m)


def _exp_reward(X, r, theta):
    p = _softmax(X @ theta)
    return float((p * r).sum(axis=1).mean())


def _nll(X, w, theta):
    z = X @ theta
    z = z - z.max(axis=1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-lp[np.arange(X.shape[0]), w].mean())


def run(config, seed):
    method = config.get("method", "dpo")
    rho = float(config.get("density", 0.15))
    beta = float(config.get("beta", 0.5))
    n = int(config.get("n_pairs", 200))

    # data rng (independent of method)
    drng = np.random.default_rng(seed)
    wA = drng.normal(0, 1, D)
    wB = drng.normal(0, 1, D)
    wA_full = np.zeros(D); wA_full[:DA] = wA[:DA]
    wB_full = np.zeros(D); wB_full[B_LO:] = wB[B_LO:]
    Xtr, _, w_tr, l_tr = _gen_task(drng, n, 0, DA, wA_full, rho)
    Xva, rva, _, _ = _gen_task(drng, N_VAL, 0, DA, wA_full, rho)
    Xte, rte, _, _ = _gen_task(drng, N_TEST, 0, DA, wA_full, rho)
    XB, _, wB_win, _ = _gen_task(drng, N_TEST, B_LO, D, wB_full, rho)

    # model-init rng
    irng = np.random.default_rng(seed + 10_000)
    theta0 = irng.normal(0, INIT_STD, D)

    # label-noise rng for the random-label null
    w_use, l_use = w_tr, l_tr
    if method == "dpo_random":
        lrng = np.random.default_rng(seed + 20_000)
        sw = lrng.random(n) < 0.5
        w_use = np.where(sw, l_tr, w_tr)
        l_use = np.where(sw, w_tr, l_tr)

    fit_method = "sft" if method == "sft" else "dpo"
    base_val = _exp_reward(Xva, rva, theta0)
    best = None
    for lr in LR_GRID:
        delta = _fit(fit_method, Xtr, w_use, l_use, beta, lr, theta0)
        if not np.all(np.isfinite(delta)):
            continue
        score = _exp_reward(Xva, rva, theta0 + delta) - base_val
        if best is None or score > best[0] + 1e-12:
            best = (score, delta)
    delta = best[1]
    theta = theta0 + delta

    a = _coeffs(fit_method, Xtr, w_use, l_use, beta, theta0, delta)
    a = np.abs(a)
    k = max(1, int(math.ceil(0.1 * n)))
    top = np.sort(a)[::-1][:k].sum() / max(a.sum(), 1e-12)

    dA = np.abs(delta[:DA])
    sparsity = float((dA < SPARSE_THRESH * max(dA.max(), 1e-12)).mean())

    interference = _nll(XB, wB_win, theta) - _nll(XB, wB_win, theta0)
    gain = _exp_reward(Xte, rte, theta) - _exp_reward(Xte, rte, theta0)
    ov = float(np.linalg.norm(delta[B_LO:DA]))

    return {
        "interference": float(interference),
        "top10_mass": float(top),
        "param_sparsity": sparsity,
        "heldout_reward_gain": float(gain),
        "overlap_norm": ov,
    }
