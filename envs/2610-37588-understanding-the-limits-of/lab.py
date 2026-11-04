import numpy as np

# Constants (no randomness at module level)
K = 4                                    # discrete actions
ANGLES = np.array([0.3, -0.3, 0.9, -0.9])  # true rotation angle per action
D_OBS = 6                                # observation dimension
LATENT = 2
N_TRAIN = 160                            # training transitions per trial (40 / action)
N_EVAL = 200                             # held-out rollouts per trial
H_MAX = 20
N_TRIALS = 80                            # independent (data, model-pair) trials per run
TOL = 1e-9


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


R_ALL = _rot(ANGLES)  # (K, 2, 2)


def _fit_faithful(Z0, Z1, acts):
    """Equivariant-structured latent dynamics: one orthogonal map per action (Procrustes)."""
    G = np.zeros((K, 2, 2))
    for k in range(K):
        m = acts == k
        M = Z1[m].T @ Z0[m]
        U, _, Vt = np.linalg.svd(M)
        G[k] = U @ Vt
    return G


def _fit_unfaithful(Z0, Z1, acts):
    """Unconstrained per-action linear map fit by least squares on one-step pairs only."""
    G = np.zeros((K, 2, 2))
    for k in range(K):
        m = acts == k
        A0, A1 = Z0[m], Z1[m]
        W = np.linalg.solve(A0.T @ A0 + 1e-6 * np.eye(2), A0.T @ A1)  # z1 ~ z0 @ W
        G[k] = W.T
    return G


def _rollout(G, z, act_seq, H):
    for t in range(H):
        z = np.einsum('nij,nj->ni', G[act_seq[:, t]], z)
    return z


def _nmse(pred, target):
    den = np.sum((target - target.mean(0)) ** 2) + 1e-12
    return float(np.sum((pred - target) ** 2) / den)


def run(config, seed):
    method = config["method"]
    H = int(config["horizon"])
    sigma = float(config["obs_noise"])
    alpha = float(config["nonlinearity"])

    # Data rng: everything about the environment and samples. It never depends on `method`.
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D_OBS, LATENT)))
    A = 1.5 * Q
    B = 0.8 * rng.standard_normal((D_OBS, LATENT))

    def g(S):
        return S @ A.T + alpha * np.tanh(S @ B.T)

    ok_one, ok_lrc = [], []
    lr_one, lr_lrc = [], []

    base_acts = np.tile(np.arange(K), N_TRAIN // K)

    for _ in range(N_TRIALS):
        # ---- training transitions (true state hidden from the model) ----
        S0 = rng.standard_normal((N_TRAIN, 2))
        acts = rng.permutation(base_acts)
        n0 = rng.standard_normal((N_TRAIN, D_OBS))
        n1 = rng.standard_normal((N_TRAIN, D_OBS))
        S1 = np.einsum('nij,nj->ni', R_ALL[acts], S0)
        X0 = g(S0) + sigma * n0
        X1 = g(S1) + sigma * n1

        # ---- learned encoder: unsupervised whitened top-2 PCA of observations ----
        Xc = np.vstack([X0, X1])
        mu = Xc.mean(0)
        evals, evecs = np.linalg.eigh(np.cov(Xc.T))
        Wenc = evecs[:, -LATENT:] / np.sqrt(evals[-LATENT:] + 1e-12)

        def h(X):
            return (X - mu) @ Wenc

        Z0, Z1 = h(X0), h(X1)

        # ---- two latent dynamics models on the same encoder ----
        GA = _fit_faithful(Z0, Z1, acts)
        GB = _fit_unfaithful(Z0, Z1, acts)

        # ---- held-out evaluation draws (fixed size regardless of H) ----
        Se = rng.standard_normal((N_EVAL, 2))
        aseq = rng.integers(0, K, size=(N_EVAL, H_MAX))
        ne0 = rng.standard_normal((N_EVAL, D_OBS))
        ne1 = rng.standard_normal((N_EVAL, D_OBS))
        neH = rng.standard_normal((N_EVAL, D_OBS))

        z0 = h(g(Se) + sigma * ne0)

        # one-step held-out error
        s1 = np.einsum('nij,nj->ni', R_ALL[aseq[:, 0]], Se)
        t1 = h(g(s1) + sigma * ne1)
        eA1 = _nmse(_rollout(GA, z0, aseq, 1), t1)
        eB1 = _nmse(_rollout(GB, z0, aseq, 1), t1)

        # latent rollout consistency at horizon H (1 - R^2 style normalised error)
        sH = Se
        for t in range(H):
            sH = np.einsum('nij,nj->ni', R_ALL[aseq[:, t]], sH)
        tH = h(g(sH) + sigma * neH)
        eAH = _nmse(_rollout(GA, z0, aseq, H), tH)
        eBH = _nmse(_rollout(GB, z0, aseq, H), tH)

        def pick(ea, eb):
            d = eb - ea
            if abs(d) < TOL * max(1.0, abs(ea) + abs(eb)) or abs(d) < 1e-10:
                return 0.5
            return 1.0 if d > 0 else 0.0

        ok_one.append(pick(eA1, eB1))
        ok_lrc.append(pick(eAH, eBH))
        lr_one.append(np.log((eB1 + 1e-12) / (eA1 + 1e-12)))
        lr_lrc.append(np.log((eBH + 1e-12) / (eAH + 1e-12)))

    acc_one = float(np.mean(ok_one))
    acc_lrc = float(np.mean(ok_lrc))
    primary = acc_lrc if method == "lrc" else acc_one

    return {
        "rank_acc": float(primary),
        "acc_onestep": acc_one,
        "acc_lrc": acc_lrc,
        "logratio_onestep": float(np.mean(lr_one)),
        "logratio_lrc": float(np.mean(lr_lrc)),
    }
