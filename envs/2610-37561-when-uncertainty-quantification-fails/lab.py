import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

N = 5000


def _entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _h_tau(p, tau):
    # Threshold-aware entropy: piecewise-linear warp sending tau -> 0.5,
    # then Shannon entropy. Continuous, monotone within each side of tau,
    # and reduces to H(p) at tau = 0.5.
    p = np.asarray(p, dtype=float)
    q = np.where(p < tau, p / (2 * tau), 0.5 + (p - tau) / (2 * (1 - tau)))
    return _entropy(q)


def _thr(u, c):
    return np.quantile(u, c)


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "global_entropy")
    pi = float(config.get("minority_frac", 0.1))
    c = float(config.get("coverage", 0.8))
    sep = float(config.get("class_sep", 1.0))

    # Data (depends only on seed and data-related config, not on method)
    X, y = make_classification(
        n_samples=N, n_features=10, n_informative=4, n_redundant=2,
        n_clusters_per_class=1, weights=[1 - pi, pi], flip_y=0.01,
        class_sep=sep, random_state=seed)
    rng = np.random.default_rng(seed + 12345)
    perm = rng.permutation(N)
    ntr, nca = int(0.6 * N), int(0.2 * N)
    tr, ca, te = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]

    clf = LogisticRegression(max_iter=200)
    clf.fit(X[tr], y[tr])
    tau = float(np.clip(y[tr].mean(), 0.01, 0.99))  # decision threshold = prior

    p_ca = clf.predict_proba(X[ca])[:, 1]
    p_te = clf.predict_proba(X[te])[:, 1]
    pred_ca = (p_ca >= tau).astype(int)
    pred_te = (p_te >= tau).astype(int)

    if method == "ccr_h_tau":
        score = lambda p: _h_tau(p, tau)
    else:
        score = _entropy
    u_ca, u_te = score(p_ca), score(p_te)

    g_thr = _thr(u_ca, c)
    if method == "global_entropy":
        accept = u_te <= g_thr
    else:
        accept = np.zeros(len(te), dtype=bool)
        for k in (0, 1):
            m_ca = pred_ca == k
            m_te = pred_te == k
            thr = _thr(u_ca[m_ca], c) if m_ca.sum() >= 5 else g_thr
            accept[m_te] = u_te[m_te] <= thr

    yt = y[te]
    mino, majo = yt == 1, yt == 0
    min_def = float(1 - accept[mino].mean()) if mino.any() else 0.0
    maj_def = float(1 - accept[majo].mean()) if majo.any() else 0.0
    am = accept & mino
    min_sens = float(((pred_te == 1) & am).sum() / am.sum()) if am.sum() > 0 else 0.0
    aj = accept & majo
    maj_spec = float(((pred_te == 0) & aj).sum() / aj.sum()) if aj.sum() > 0 else 0.0

    return {
        "deferral_gap": abs(min_def - maj_def),
        "minority_deferral": min_def,
        "majority_deferral": maj_def,
        "minority_sens_accepted": min_sens,
        "balanced_acc_accepted": 0.5 * (min_sens + maj_spec),
    }
