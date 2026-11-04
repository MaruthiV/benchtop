import numpy as np
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from scipy.spatial.distance import cdist

N_INIT = 8
N_TOTAL = 36
LS_GRID = [0.1, 0.2, 0.35, 0.6, 1.0]
JIT = 1e-5
THRESH = 0.25
GRID_N = {2: 40, 3: 12}


def sim(X, T):
    t = np.linspace(0.0, 4.0, T)[None, :]
    a = 0.3 + 1.7 * X[:, 0:1]
    f = 0.4 + 1.2 * X[:, 1:2]
    ph = 2 * np.pi * X[:, 2:3] if X.shape[1] > 2 else 0.5
    return np.exp(-a * t) * np.sin(2 * np.pi * f * t + ph)


def kern(A, B, l):
    s = np.sqrt(5.0) * cdist(A, B) / l
    return (1 + s + s * s / 3.0) * np.exp(-s)


def gp_fit(X, Y):
    """Shared-lengthscale Matern-5/2 GP on standardized multi-output Y; lengthscale by marginal likelihood."""
    n, m = Y.shape
    best = None
    for l in LS_GRID:
        K = kern(X, X, l) + JIT * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            continue
        al = cho_solve((L, True), Y)
        lml = -0.5 * np.sum(Y * al) - m * np.sum(np.log(np.diag(L)))
        if best is None or lml > best[0]:
            best = (lml, l, L, al)
    _, l, L, al = best
    return l, L, al


def gp_pred(Xg, X, fit):
    l, L, al = fit
    Ks = kern(Xg, X, l)
    mu = Ks @ al
    v = solve_triangular(L, Ks.T, lower=True)
    var = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-12)
    return mu, var


def surrogate_ll(method, X, Y, z, sigma, Xg):
    """Return (ll mean, ll sd) on grid from data X (n,d), simulator outputs Y (n,T)."""
    if method == "proxy":
        my = Y.mean(0)
        sy = np.maximum(Y.std(0), 1e-6)
        fit = gp_fit(X, (Y - my) / sy)
        mu, var = gp_pred(Xg, X, fit)
        My = mu * sy + my
        r = z[None, :] - My
        ll = -0.5 * (np.sum(r * r, 1) + var * np.sum(sy ** 2)) / sigma ** 2
        sd = np.sqrt(var) * np.sqrt(np.sum((r / sigma ** 2) ** 2 * sy[None, :] ** 2, 1))
        return ll, sd
    ll_tr = -0.5 * np.sum((z[None, :] - Y) ** 2, 1) / sigma ** 2
    m, s = ll_tr.mean(), max(ll_tr.std(), 1e-9)
    fit = gp_fit(X, ((ll_tr - m) / s)[:, None])
    mu, var = gp_pred(Xg, X, fit)
    return mu[:, 0] * s + m, np.sqrt(var) * s


def post_from_ll(ll):
    p = np.exp(ll - ll.max())
    return p / p.sum()


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "loglik")
    sigma = float(config.get("noise", 0.2))
    T = int(config.get("T", 10))
    d = int(config.get("dim", 2))

    rng_data = np.random.default_rng(seed)
    rng_init = np.random.default_rng(seed + 1000003)

    # data
    xstar = rng_data.uniform(0.25, 0.75, size=d)
    z = sim(xstar[None, :], T)[0] + sigma * rng_data.standard_normal(T)

    # grid and reference posterior (exact likelihood)
    n = GRID_N[d]
    ax = (np.arange(n) + 0.5) / n
    Xg = np.stack(np.meshgrid(*([ax] * d), indexing="ij"), -1).reshape(-1, d)
    G = Xg.shape[0]
    Yg = sim(Xg, T)  # cheap enough here; used for the reference and as the "simulator" lookup
    ll_ref = -0.5 * np.sum((z[None, :] - Yg) ** 2, 1) / sigma ** 2
    p_ref = post_from_ll(ll_ref)
    mean_ref = p_ref @ Xg

    perm = rng_init.permutation(G)
    chosen = list(perm[:N_INIT])
    mask = np.zeros(G, bool)
    mask[chosen] = True
    rand_queue = list(perm[N_INIT:])
    gm = "loglik" if method == "random" else method

    tvs = {}
    last_p = None
    for k in range(N_INIT, N_TOTAL + 1):
        idx = np.array(chosen)
        ll, sd = surrogate_ll(gm, Xg[idx], Yg[idx], z, sigma, Xg)
        last_p = post_from_ll(ll)
        tvs[k] = 0.5 * np.abs(last_p - p_ref).sum()
        if k == N_TOTAL:
            break
        if method == "random":
            nxt = rand_queue.pop(0)
        else:
            s2 = np.minimum(sd ** 2, 50.0) + 1e-12
            rel = ll - ll.max()
            acq = 2 * rel + 2 * s2 + np.log(-np.expm1(-s2))
            acq[mask] = -np.inf
            nxt = int(np.argmax(acq))
        chosen.append(nxt)
        mask[nxt] = True

    calls = float(N_TOTAL)
    for k in range(N_INIT, N_TOTAL + 1):
        if tvs[k] < THRESH:
            calls = float(k)
            break

    return {
        "tv_final": float(tvs[N_TOTAL]),
        "tv_at_20": float(tvs[20]),
        "calls_to_thresh": calls,
        "mean_err": float(np.linalg.norm(last_p @ Xg - mean_ref)),
    }
