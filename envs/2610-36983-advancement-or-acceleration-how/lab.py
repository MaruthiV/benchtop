import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

D = 20
LAM = 1e-2
PROBE_LAM = 5e-2
N_TARGET_LABELED = 40
N_TEST = 4000
KEEP_FRAC = 0.5


def _fit(X, y, lam, maxiter=150):
    n, d = X.shape

    def fg(w):
        z = X @ w
        loss = np.mean(np.logaddexp(0.0, -y * z)) + 0.5 * lam * (w @ w)
        g = X.T @ (-y * expit(-y * z)) / n + lam * w
        return loss, g

    res = minimize(fg, np.zeros(d), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    return res.x


def _logloss(w, X, y):
    return float(np.mean(np.logaddexp(0.0, -y * (X @ w))))


def _acc(w, X, y):
    return float(np.mean(np.sign(X @ w) == y))


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "uniform")
    phi = float(config.get("gap_deg", 0)) * np.pi / 180.0
    n = int(config.get("n_pool", 200))
    noise = float(config.get("label_noise", 0.0))

    # ---- data (independent of method) ----
    rng = np.random.default_rng(seed)
    B = rng.standard_normal(D)
    B /= np.linalg.norm(B)
    u = rng.standard_normal(D)
    u -= (u @ B) * B
    u /= np.linalg.norm(u)
    Bt = np.cos(phi) * B + np.sin(phi) * u

    def noisy(clean):
        flip = rng.random(len(clean)) < noise
        return np.where(flip, -clean, clean)

    Xp = rng.standard_normal((n, D))
    yp = noisy(np.sign(Xp @ B))                 # training pool, source teacher
    Xl = rng.standard_normal((N_TARGET_LABELED, D))
    yl = noisy(np.sign(Xl @ Bt))                # small labeled target set
    Xt = rng.standard_normal((N_TEST, D))
    yt_clean = np.sign(Xt @ Bt)
    yt_noisy = noisy(yt_clean)

    # ---- baseline fit (also used for the Delta-test diagnostic) ----
    w_base = _fit(Xp, yp, LAM)
    delta = abs(_logloss(w_base, Xt, yt_noisy) - _logloss(w_base, Xp, yp))

    kept = n
    if method == "uniform":
        w = w_base
    elif method == "random_select":
        r = np.random.default_rng(seed + 1000)
        k = int(KEEP_FRAC * n)
        idx = r.choice(n, size=k, replace=False)
        w = _fit(Xp[idx], yp[idx], LAM)
        kept = k
    elif method == "target_select":
        # target probe from small labeled target set
        wp = _fit(Xl, yl, PROBE_LAM)
        s = Xp @ wp
        agree = np.where(yp * s > 0)[0]
        k = int(KEEP_FRAC * n)
        if len(agree) >= 10:
            order = agree[np.argsort(np.abs(s[agree]), kind="stable")]
            idx = order[:min(k, len(order))]
        else:
            idx = np.arange(n)
        w = _fit(Xp[idx], yp[idx], LAM)
        kept = len(idx)
    else:
        raise ValueError(method)

    acc = _acc(w, Xt, yt_clean)
    return {
        "target_acc": acc,
        "target_logloss": _logloss(w, Xt, yt_noisy),
        "baseline_target_acc": _acc(w_base, Xt, yt_clean),
        "delta_test": float(delta),
        "n_kept": float(kept),
    }
