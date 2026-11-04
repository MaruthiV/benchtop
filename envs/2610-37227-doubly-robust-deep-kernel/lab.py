import math
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

_DIG = load_digits().data.astype(np.float64) / 16.0
_ACTIVE = np.where(_DIG.std(0) > 0.05)[0]

D_F = 64          # random Fourier features per kernel component
W_CHAR = 0.2      # weight of the characteristic (raw-pixel) component in the deep kernel
LAM_J = 1e-3      # regulariser in the power criterion
RIDGE = 1.0
N_PERM = 100
STEPS = 60
R_ALT = 6
R_NULL = 3
DELTA = 0.6       # shift norm (pixel units, on /16 scale)


def _gen(rng, n, conf, k, delta):
    idx = rng.integers(0, _DIG.shape[0], n)
    img = _DIG[idx]
    X = rng.normal(size=(n, 3))
    B = 0.15 * rng.normal(size=(3, 64))
    pix = rng.choice(_ACTIVE, size=k, replace=False)
    v = np.zeros(64)
    v[pix] = 1.0 / math.sqrt(k)
    e = 1.0 / (1.0 + np.exp(-conf * X[:, 0]))
    T = (rng.random(n) < e).astype(np.float64)
    Y = img + X @ B + 0.05 * rng.normal(size=(n, 64)) + delta * T[:, None] * v[None, :]
    return X, T, Y


def _design(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def _fit_B(Phi, T, Z):
    Bs = []
    p = Z.shape[1]
    for a in (0.0, 1.0):
        msk = T == a
        Za = Z[msk]
        A = Za.T @ Za + RIDGE * torch.eye(p, dtype=Z.dtype)
        Bs.append(torch.linalg.solve(A, Za.T @ Phi[msk]))
    return Bs


def _g(Phi, T, Z, e, Bs):
    m0 = Z @ Bs[0]
    m1 = Z @ Bs[1]
    mT = torch.where(T[:, None] == 1, m1, m0)
    w = T / e - (1 - T) / (1 - e)
    return w[:, None] * (Phi - mT) + (m1 - m0)


def _crit(Phi, T, Z, e):
    Bs = _fit_B(Phi, T, Z)
    g = _g(Phi, T, Z, e, Bs)
    m = g.shape[0] // 2
    g1 = g[:m].mean(0)
    g2 = g[m:2 * m]
    d = (g1 * g2.mean(0)).sum()
    s2 = ((g2 @ g1 - d) ** 2).mean()
    return d / torch.sqrt(s2 + LAM_J)


def _perm_p(G, signs):
    # G: (m_total, D) numpy; studentised cross statistic with sign-flip calibration
    m = G.shape[0] // 2
    G1, G2 = G[:m], G[m:2 * m]

    def stat(S):
        a = S[:, :m] @ G1 / m
        b = S[:, m:2 * m] @ G2 / m
        d = (a * b).sum(1)
        c = (a @ G2.T) * S[:, m:2 * m]
        s = np.sqrt(((c - d[:, None]) ** 2).mean(1)) + 1e-12
        return math.sqrt(m) * d / s

    obs = stat(np.ones((1, 2 * m)))[0]
    null = stat(signs[:, :2 * m])
    return (1 + np.sum(null >= obs)) / (N_PERM + 1)


def _rff(Y, W, b, h):
    return math.sqrt(2.0 / W.shape[1]) * torch.cos(Y @ W / h + b)


def _one_rep(X, T, Y, Wr, br, Wd, bd, signs, tseed, method):
    n = X.shape[0]
    h = n // 2
    Xtr, Xte = X[:h], X[h:2 * h]
    Ttr, Tte = T[:h], T[h:2 * h]
    Ytr, Yte = Y[:h], Y[h:2 * h]
    lr = LogisticRegression(C=1.0).fit(Xtr, Ttr)
    etr = np.clip(lr.predict_proba(Xtr)[:, 1], 0.05, 0.95)
    ete = np.clip(lr.predict_proba(Xte)[:, 1], 0.05, 0.95)
    Ztr = torch.tensor(_design(Xtr)); Zte = torch.tensor(_design(Xte))
    Ttr_t = torch.tensor(Ttr); Tte_t = torch.tensor(Tte)
    etr_t = torch.tensor(etr); ete_t = torch.tensor(ete)
    Ytr_t = torch.tensor(Ytr); Yte_t = torch.tensor(Yte)
    # median heuristic bandwidth on train outcomes
    dm = torch.cdist(Ytr_t, Ytr_t)
    med = float(torch.median(dm[dm > 0]))
    Wr_t = torch.tensor(Wr); br_t = torch.tensor(br)

    if method == "fixed_median":
        feat = lambda Yt: _rff(Yt, Wr_t, br_t, med)
    elif method == "bw_opt":
        best, besth = -1e18, med
        with torch.no_grad():
            for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
                J = float(_crit(_rff(Ytr_t, Wr_t, br_t, med * mult), Ttr_t, Ztr, etr_t))
                if J > best:
                    best, besth = J, med * mult
        feat = lambda Yt: _rff(Yt, Wr_t, br_t, besth)
    else:
        torch.manual_seed(tseed)
        net = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16)).double()
        sc = float(Ytr_t.std()) + 1e-8
        Wd_t = torch.tensor(Wd); bd_t = torch.tensor(bd)
        ws = math.sqrt(1 - W_CHAR); wc = math.sqrt(W_CHAR)

        def feat(Yt):
            z = net(Yt / sc)
            return torch.cat([ws * _rff(z, Wd_t, bd_t, 2.0), wc * _rff(Yt, Wr_t, br_t, med)], 1)

        opt = torch.optim.Adam(net.parameters(), lr=0.02)
        for _ in range(STEPS):
            opt.zero_grad()
            loss = -_crit(feat(Ytr_t), Ttr_t, Ztr, etr_t)
            loss.backward()
            opt.step()

    with torch.no_grad():
        Bs = _fit_B(feat(Ytr_t), Ttr_t, Ztr)
        G = _g(feat(Yte_t), Tte_t, Zte, ete_t, Bs).numpy()
    return _perm_p(G, signs)


def run(config: dict, seed: int) -> dict:
    torch.set_num_threads(1)
    method = config.get("method", "deep")
    k = int(config.get("shift_dims", 1))
    conf = float(config.get("confounding", 1.0))
    n = int(config.get("n", 160))
    n -= n % 4

    def reps(delta, off, R):
        ps = []
        for r in range(R):
            drng = np.random.default_rng([seed, off, r, 1])
            X, T, Y = _gen(drng, n, conf, k, delta)
            mrng = np.random.default_rng([seed, off, r, 2])
            Wr = mrng.normal(size=(64, D_F)); br = mrng.uniform(0, 2 * math.pi, D_F)
            Wd = mrng.normal(size=(16, D_F)); bd = mrng.uniform(0, 2 * math.pi, D_F)
            prng = np.random.default_rng([seed, off, r, 3])
            signs = prng.choice([-1.0, 1.0], size=(N_PERM, n))
            ps.append(_one_rep(X, T, Y, Wr, br, Wd, bd, signs, seed * 1000 + off * 100 + r, method))
        return np.array(ps)

    pa = reps(DELTA, 0, R_ALT)
    pn = reps(0.0, 1, R_NULL)
    return {
        "power": float(np.mean(pa < 0.05)),
        "type1": float(np.mean(pn < 0.05)),
        "mean_p_alt": float(np.mean(pa)),
        "mean_p_null": float(np.mean(pn)),
    }
