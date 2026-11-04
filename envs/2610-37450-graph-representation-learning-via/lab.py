import numpy as np
from sklearn.linear_model import LogisticRegression

N = 600
Q = 6
AVG_DEG = 10.0
MU_HI = 3.0
CLASS_SHIFT = 0.1
ITERS = 20
TAUS = (0.5, 0.1, 0.25, 0.75, 0.9)   # 0.5 first so ties fall back to mean
C_GRID = (0.1, 1.0, 10.0)


def _make_data(seed, h, p, noise):
    rng = np.random.default_rng([seed, 1])
    y = rng.permutation(np.arange(N) % 2)
    same = y[:, None] == y[None, :]
    half = N / 2.0
    p_in = min(1.0, AVG_DEG * h / half)
    p_out = min(1.0, AVG_DEG * (1 - h) / half)
    prob = np.where(same, p_in, p_out)
    U = rng.random((N, N))
    A = np.triu(U < prob, 1)
    A = A | A.T
    # class-1: mixture with mean 0; class-0: gaussian matched in mean and variance
    mu_lo = -p * MU_HI / (1 - p)
    var1 = 1 + p * (1 - p) * (MU_HI - mu_lo) ** 2
    hi = rng.random((N, Q)) < p
    x1 = np.where(hi, MU_HI, mu_lo) + rng.normal(size=(N, Q))
    x0 = rng.normal(size=(N, Q)) * np.sqrt(var1)
    X = np.where((y == 1)[:, None], x1, x0)
    X = X + CLASS_SHIFT * (2 * y[:, None] - 1)
    X = X + noise * rng.normal(size=(N, Q))
    # split
    srng = np.random.default_rng([seed, 2])
    perm = srng.permutation(N)
    ntr, nva = int(0.6 * N), int(0.2 * N)
    return X, y, A, perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]


def _expectile_agg(X, cols, starts, counts, tau):
    Xj = X[cols]
    num = np.add.reduceat(Xj, starts, axis=0)
    r = num / counts[:, None]  # mean init
    if abs(tau - 0.5) < 1e-12:
        return r
    for _ in range(ITERS):
        rr = np.repeat(r, counts, axis=0)
        w = np.where(Xj >= rr, tau, 1 - tau)
        r = np.add.reduceat(w * Xj, starts, axis=0) / np.add.reduceat(w, starts, axis=0)
    return r


def _fit_eval(F, y, tr, va, te):
    mu = F[tr].mean(0)
    sd = F[tr].std(0) + 1e-8
    Fs = (F - mu) / sd
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=200)
        clf.fit(Fs[tr], y[tr])
        v = clf.score(Fs[va], y[va])
        if best is None or v > best[0]:
            best = (v, clf.score(Fs[te], y[te]), clf.score(Fs[tr], y[tr]))
    return best  # val, test, train


def run(config, seed):
    h = float(config.get("homophily", 0.2))
    p = float(config.get("tail_frac", 0.3))
    noise = float(config.get("noise", 0.0))
    method = config.get("method", "mean")

    X, y, A, tr, va, te = _make_data(seed, h, p, noise)
    A = A | np.eye(N, dtype=bool)
    rows, cols = np.nonzero(A)
    counts = np.bincount(rows, minlength=N)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    if method == "mean":
        taus = (0.5,)
    elif method == "expectile_fixed":
        taus = (0.9,)
    else:
        taus = TAUS

    results = []
    for tau in taus:
        R = _expectile_agg(X, cols, starts, counts, tau)
        F = np.concatenate([X, R], axis=1)
        results.append(_fit_eval(F, y, tr, va, te))
    vals = [r[0] for r in results]
    k = int(np.argmax(vals))  # first max wins ties
    out = {
        "test_acc": float(results[k][1]),
        "val_acc": float(results[k][0]),
        "train_acc": float(results[k][2]),
        "selected_tau": float(taus[k]),
        "oracle_test_acc": float(max(r[1] for r in results)),
    }
    return out
