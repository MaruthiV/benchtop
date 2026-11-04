import numpy as np

D = 16
NMAX_PER_D = 40
COND_SPECTRUM = 100.0
GRID = np.logspace(-4, 1, 12)
FOLDS = 5


def _risk(W, A, Sigma, s):
    d = A.shape[1]
    M = W @ A - np.eye(d)
    return (np.trace(M @ Sigma @ M.T) + s * s * np.trace(W @ W.T)) / d


def _wiener(Ah, Sigma, s):
    m = Ah.shape[0]
    return Sigma @ Ah.T @ np.linalg.inv(Ah @ Sigma @ Ah.T + s * s * np.eye(m))


def _fit_ridge(X, Y, lam):
    # W minimizing sum ||W y - x||^2 + lam ||W||^2 ; X,Y are (n,d)
    G = Y.T @ Y + lam * np.eye(Y.shape[1])
    return np.linalg.solve(G, Y.T @ X).T


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "wiener")
    n = int(round(float(config.get("n_per_d", 5)) * D))
    s = float(config.get("noise", 0.5))
    op_err = float(config.get("operator_error", 0.0))

    # ---- data rng (independent of method) ----
    rd = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rd.standard_normal((D, D)))
    V, _ = np.linalg.qr(rd.standard_normal((D, D)))
    sv = np.logspace(0, -np.log10(COND_SPECTRUM), D)
    A = U @ np.diag(sv) @ V.T
    Q, _ = np.linalg.qr(rd.standard_normal((D, D)))
    eig = np.logspace(0, -1.3, D)
    Sigma = Q @ np.diag(eig) @ Q.T
    E = rd.standard_normal((D, D))
    E = E / np.linalg.norm(E) * np.linalg.norm(A)
    L = Q @ np.diag(np.sqrt(eig))
    Xall = rd.standard_normal((NMAX_PER_D * D, D)) @ L.T
    Nall = rd.standard_normal((NMAX_PER_D * D, D))
    X = Xall[:n]
    Y = X @ A.T + s * Nall[:n]

    # ---- model rng (CV folds only) ----
    rm = np.random.default_rng(seed + 1000003)
    perm = rm.permutation(n)

    Wstar = _wiener(A, Sigma, s)
    bayes = _risk(Wstar, A, Sigma, s)

    if method == "wiener":
        Ah = A + op_err * E
        W = _wiener(Ah, Sigma, s)
    elif method == "ols":
        W = np.linalg.lstsq(Y, X, rcond=None)[0].T
    elif method == "ridge_cv":
        folds = np.array_split(perm, FOLDS)
        errs = np.zeros(len(GRID))
        for f in folds:
            tr = np.setdiff1d(perm, f)
            ntr = len(tr)
            for gi, g in enumerate(GRID):
                Wf = _fit_ridge(X[tr], Y[tr], g * ntr)
                errs[gi] += np.sum((Y[f] @ Wf.T - X[f]) ** 2)
        g = GRID[int(np.argmin(errs))]
        W = _fit_ridge(X, Y, g * n)
    else:
        raise ValueError(method)

    risk = _risk(W, A, Sigma, s)
    excess = risk - bayes
    return {
        "excess_risk": float(excess),
        "test_risk": float(risk),
        "bayes_risk": float(bayes),
        "rel_excess": float(excess / bayes),
    }
