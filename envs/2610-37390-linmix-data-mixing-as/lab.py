import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

NOISE_SD = 0.05
PAIR_SCALE = 1.0


def _project_caps(pi, u):
    """Clip rows to cap u and redistribute the excess proportionally (capped simplex)."""
    pi = pi.copy()
    capped = np.zeros(pi.shape, dtype=bool)
    for _ in range(40):
        over = pi > u + 1e-12
        if not over.any():
            break
        capped |= over
        pi = np.where(capped, u, pi)
        rem = 1.0 - pi.sum(1)
        w = np.where(capped, 0.0, pi)
        s = np.maximum(w.sum(1, keepdims=True), 1e-12)
        pi = np.where(capped, u, pi + rem[:, None] * w / s)
    return pi


def _dirichlet_design(rng, n, prior, conc, d, K):
    """Dirichlet around the tempered prior (tau=2); total concentration log-uniform around conc*d."""
    p = np.sqrt(prior)
    p = p / p.sum()
    gam = conc * d * np.exp(rng.uniform(-np.log(3.0), np.log(3.0), size=n))
    pi = np.empty((n, d))
    for i in range(n):
        a = np.maximum(gam[i] * p, 1e-3)
        pi[i] = rng.dirichlet(a)
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum(1, keepdims=True)
    return _project_caps(pi, 1.0 / K)


def _vertex_design(rng, n, d, K):
    """Greedy fill in a random order with paired reversed order (uniform caps 1/K -> exactly K domains at cap)."""
    X = np.zeros((n, d))
    for i in range(n // 2):
        perm = rng.permutation(d)
        X[2 * i, perm[:K]] = 1.0
        X[2 * i + 1, perm[::-1][:K]] = 1.0
    if n % 2:
        X[-1, rng.permutation(d)[:K]] = 1.0
    return X


def _z(X, beta, W, nu, K):
    X = np.atleast_2d(X)
    lin = X @ beta / np.sqrt(K)
    quad = np.einsum("ij,jk,ik->i", X, W, X) / (2.0 * K)
    return lin + nu * quad


def _perf(X, beta, W, nu, K):
    return 1.0 / (1.0 + np.exp(-_z(X, beta, W, nu, K)))


def _climb(x, beta, W, nu, K):
    x = x.copy()
    for _ in range(150):
        g = W @ x
        I = np.flatnonzero(x == 1)
        O = np.flatnonzero(x == 0)
        dlin = (beta[O][None, :] - beta[I][:, None]) / np.sqrt(K)
        dq = -2 * g[I][:, None] + 2 * g[O][None, :] - 2 * W[np.ix_(I, O)]
        delta = dlin + nu * dq / (2.0 * K)
        j = int(np.argmax(delta))
        if delta.flat[j] <= 1e-12:
            break
        a, b = np.unravel_index(j, delta.shape)
        x[I[a]] = 0.0
        x[O[b]] = 1.0
    return x


def _topk_vertex(coef, K, d):
    x = np.zeros(d)
    x[np.argsort(-coef)[:K]] = 1.0
    return x


def _ols(X, y):
    A = np.hstack([X, np.ones((X.shape[0], 1))])
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    return sol[:-1]


def run(config, seed):
    method = config.get("method", "linmix")
    d = int(config.get("n_domains", 40))
    nu = float(config.get("nonlinearity", 0.0)) * PAIR_SCALE
    conc = float(config.get("concentration", 3.0))
    K = int(max(2, round(d / 4)))
    m = 4 * d

    # ---- data / oracle (identical across methods for a given seed)
    drng = np.random.default_rng(seed)
    prior = np.exp(0.7 * drng.normal(size=d))
    prior /= prior.sum()
    beta = drng.normal(size=d) * (drng.random(d) < 0.6)
    if np.count_nonzero(beta) < 3:
        beta[:3] = drng.normal(size=3) + 1.0
    Wr = drng.normal(size=(d, d))
    W = (Wr + Wr.T) / np.sqrt(2.0)
    np.fill_diagonal(W, 0.0)
    noise = drng.normal(size=m) * NOISE_SD
    starts = [_topk_vertex(beta, K, d)]
    for _ in range(3):
        s = np.zeros(d)
        s[drng.permutation(d)[:K]] = 1.0
        starts.append(s)

    # reference best mix: local search on the noise-free oracle
    ref = 0.0
    for s in starts:
        xb = _climb(s, beta, W, nu, K)
        ref = max(ref, float(_perf(xb, beta, W, nu, K)[0]))

    # ---- design (separate rngs per design type; model init via random_state)
    pool_X = None
    if method == "linmix":
        X = _vertex_design(np.random.default_rng([seed, 1]), m, d, K)
    else:
        pi = _dirichlet_design(np.random.default_rng([seed, 2]), m, prior, conc, d, K)
        X = pi * K
    y = _perf(X, beta, W, nu, K) + noise

    # ---- fit and select
    if method in ("linmix", "dirichlet_ols"):
        coef = _ols(X, y)
        xsel = _topk_vertex(coef, K, d)
    elif method == "dirichlet_gbm":
        gbm = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, max_features=0.5, random_state=seed)
        gbm.fit(X, y)
        pool_pi = _dirichlet_design(np.random.default_rng([seed, 3]), 3000, prior, conc, d, K)
        pool_X = pool_pi * K
        pred = gbm.predict(pool_X)
        top = np.argsort(-pred)[:10]
        xsel = pool_X[top].mean(0)
        coef = _ols(pool_X, pred)  # implied linear attribution of the surrogate
    else:
        raise ValueError("unknown method: %s" % method)

    chosen = float(_perf(xsel, beta, W, nu, K)[0])
    regret = max(0.0, ref - chosen)

    cc = coef - coef.mean()
    bc = beta - beta.mean()
    nz = beta != 0
    sign_agree = float(np.mean(np.sign(cc[nz]) == np.sign(bc[nz])))
    if cc.std() > 1e-12 and bc.std() > 1e-12:
        coef_corr = float(np.corrcoef(cc, bc)[0, 1])
    else:
        coef_corr = 0.0

    return {
        "true_regret": float(regret),
        "sign_agree": sign_agree,
        "coef_corr": coef_corr,
        "chosen_perf": chosen,
        "ref_perf": float(ref),
    }
