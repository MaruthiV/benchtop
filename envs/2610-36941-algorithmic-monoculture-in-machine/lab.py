import numpy as np
from scipy.optimize import minimize

K = 9
N_FARMS = 1000
D = 3
RHO = 0.25
LAM = 20.0
T_EVAL = 100
BENCH_DOF = 150
GAP_SPAN = {"small": 0.08, "large": 1.0}


def _counts(q, n):
    raw = q * n
    c = np.floor(raw).astype(int)
    rem = n - c.sum()
    if rem > 0:
        order = np.argsort(-(raw - c), kind="stable")
        c[order[:rem]] += 1
    return c


def _model_loss(q, r, kappa, n):
    sd = np.sqrt(r)
    corr = (1 - kappa) * np.eye(K) + kappa * np.ones((K, K))
    sig = RHO * (sd[:, None] * corr * sd[None, :])
    return 0.5 * q @ r + 0.5 * LAM * (q @ sig @ q + (1 - RHO) / n * (q @ r))


def _portfolio(r_hat, kappa, delta, n):
    rmin = r_hat.min()
    cap = (1 + delta) * rmin
    scale = 1.0 / rmin
    cons = [
        {"type": "eq", "fun": lambda q: q.sum() - 1.0},
        {"type": "ineq", "fun": lambda q: (cap - q @ r_hat) * scale},
    ]
    mono = np.zeros(K)
    mono[np.argmin(r_hat)] = 1.0
    best, best_val = mono, _model_loss(mono, r_hat, kappa, n)
    for x0 in (np.ones(K) / K, 0.5 * mono + 0.5 * np.ones(K) / K, mono):
        res = minimize(lambda q: _model_loss(q, r_hat, kappa, n) * scale, x0,
                       jac=None, method="SLSQP", bounds=[(0, 1)] * K,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
        q = np.clip(res.x, 0, None)
        q = q / q.sum()
        if q @ r_hat <= cap * (1 + 1e-6):
            v = _model_loss(q, r_hat, kappa, n)
            if v < best_val - 1e-12:
                best, best_val = q, v
    return best


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "monoculture")
    kappa = float(config.get("kappa", 0.0))
    gap = config.get("gap", "small")
    delta = float(config.get("cap", 0.1))
    n = N_FARMS

    # data: true model risks (independent of method)
    rng_d = np.random.default_rng(seed)
    rmin_true = 0.1 * np.exp(0.2 * rng_d.standard_normal())
    u = np.sort(rng_d.uniform(0, 1, K - 1))
    mult = np.concatenate([[1.0], 1 + GAP_SPAN[gap] * (0.1 + 0.9 * u)])
    r_true = rmin_true * mult[rng_d.permutation(K)]

    # benchmark estimate of risks (noisy), shared across methods
    rng_b = np.random.default_rng([seed, 1])
    r_hat = r_true * rng_b.chisquare(BENCH_DOF, K) / BENCH_DOF

    # fresh evaluation draws, shared across methods
    rng_e = np.random.default_rng([seed, 2])
    Zb = rng_e.standard_normal((T_EVAL, K, D))
    Cb = rng_e.standard_normal((T_EVAL, 1, D))
    U0 = rng_e.standard_normal((T_EVAL, n, D))

    # allocation
    if method == "monoculture":
        q = np.zeros(K)
        q[np.argmin(r_hat)] = 1.0
    elif method == "uniform":
        q = np.ones(K) / K
    else:
        q = _portfolio(r_hat, kappa, delta, n)

    cnt = _counts(q, n)
    idx = np.repeat(np.arange(K), cnt)

    zstd = np.sqrt(1 - kappa) * Zb + np.sqrt(kappa) * Cb
    z = zstd * np.sqrt(RHO * r_true / D)[None, :, None]
    eps = z[:, idx, :] + U0 * np.sqrt((1 - RHO) * r_true[idx] / D)[None, :, None]
    private = 0.5 * (eps ** 2).sum(axis=2).mean(axis=1)
    system = 0.5 * LAM * (eps.mean(axis=1) ** 2).sum(axis=1)
    qa = cnt / n
    regret = float(qa @ r_true / r_true.min() - 1.0)
    return {
        "social_loss": float((private + system).mean()),
        "private_loss": float(private.mean()),
        "system_loss": float(system.mean()),
        "regret": regret,
        "cap_ok": float(regret <= delta + 1e-9),
    }
