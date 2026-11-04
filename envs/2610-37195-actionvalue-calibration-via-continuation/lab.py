import numpy as np

T_TASKS = 4
N_ACT = 3
HORIZON = 6
GAMMA = 0.95
P_REWARD = 0.4


def _build_and_rollout(rng, S, eps, G):
    T, A, H = T_TASKS, N_ACT, HORIZON
    # MDP tables (one MDP per task); time-layered states so exact Q is well defined
    m = rng.integers(0, S, size=(T, H, S, A))
    R = rng.random((T, H, S, A, S)) * (rng.random((T, H, S, A, S)) < P_REWARD)
    R[:, H - 1] *= 2.0  # larger terminal-step reward
    # rollout randomness (drawn identically for every method)
    acts = rng.integers(0, A, size=(T, G, H))
    u = rng.random((T, G, H))
    rs = rng.integers(0, S, size=(T, G, H))

    # exact Q under uniform behaviour policy by backward DP
    P = (1.0 - eps) * np.eye(S)[m] + eps / S
    er = (P * R).sum(-1)
    Q = np.zeros((T, H, S, A))
    Vn = np.zeros((T, S))
    for t in range(H - 1, -1, -1):
        Q[:, t] = er[:, t] + GAMMA * np.einsum('tsap,tp->tsa', P[:, t], Vn)
        Vn = Q[:, t].mean(-1)

    ti = np.arange(T)[:, None]
    s = np.zeros((T, G, H + 1), dtype=np.int64)
    r = np.zeros((T, G, H))
    for t in range(H):
        a = acts[:, :, t]
        main = m[ti, t, s[:, :, t], a]
        ns = np.where(u[:, :, t] < eps, rs[:, :, t], main)
        r[:, :, t] = R[ti, t, s[:, :, t], a, ns]
        s[:, :, t + 1] = ns
    return Q, s[:, :, :H], acts, r


def _estimate(method, s, acts, r, S):
    T, G, H = r.shape
    A = N_ACT
    if method == "gigpo":
        ret = np.zeros((T, G, H))
        run = np.zeros((T, G))
        for t in range(H - 1, -1, -1):
            run = r[:, :, t] + GAMMA * run
            ret[:, :, t] = run
        return ret, 1.0
    # matching keys on incoming (prev state, prev action, state), within task and layer
    sprev = np.full((T, G, H), -1, dtype=np.int64)
    aprev = np.zeros((T, G, H), dtype=np.int64)
    sprev[:, :, 1:] = s[:, :, :-1]
    aprev[:, :, 1:] = acts[:, :, :-1]
    tid = np.arange(T).reshape(T, 1, 1)
    tt = np.arange(H).reshape(1, 1, H)
    key = ((((tid * H + tt) * (S + 1) + (sprev + 1)) * A + aprev) * S + s).ravel()
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n_vis = cnt[inv].astype(float)
    alpha = np.ones_like(n_vis) if method == "pooled" else n_vis / (n_vis + 1.0)
    alpha = alpha.reshape(T, G, H)
    rflat_shape = (T, G, H)
    Vs = [np.zeros(rflat_shape)]
    V = Vs[0]
    for b in range(1, H):
        Vnext = np.zeros(rflat_shape)
        Vnext[:, :, :-1] = V[:, :, 1:]
        Y = r + GAMMA * Vnext
        mu = np.bincount(inv, weights=Y.ravel()) / cnt
        V = (1 - alpha) * Y + alpha * mu[inv].reshape(rflat_shape)
        Vs.append(V)
    Qt = r.copy()
    for t in range(H - 1):
        Qt[:, :, t] = r[:, :, t] + GAMMA * Vs[H - t - 1][:, :, t + 1]
    return Qt, float(n_vis.mean())


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "concal")
    S = int(config.get("states_per_step", 8))
    eps = float(config.get("noise", 0.0))
    G = int(config.get("group_size", 8))
    T, A, H = T_TASKS, N_ACT, HORIZON

    rng = np.random.default_rng(seed)  # data rng only; methods use no randomness
    Q, s, acts, r = _build_and_rollout(rng, S, eps, G)
    est, pool = _estimate(method, s, acts, r, S)

    tid = np.arange(T).reshape(T, 1, 1)
    tt = np.arange(H).reshape(1, 1, H)
    idx = (((tid * H + tt) * S + s) * A + acts).ravel()
    size = T * H * S * A
    sums = np.bincount(idx, weights=est.ravel(), minlength=size)
    cn = np.bincount(idx, minlength=size)
    E = (sums / np.maximum(cn, 1)).reshape(T, H, S, A)
    obs = (cn > 0).reshape(T, H, S, A)

    errs, taus = [], []
    for a1 in range(A):
        for a2 in range(a1 + 1, A):
            ok = obs[..., a1] & obs[..., a2]
            de = (E[..., a1] - E[..., a2])[ok]
            dq = (Q[..., a1] - Q[..., a2])[ok]
            errs.append(np.abs(de - dq))
            taus.append(np.sign(de) * np.sign(dq))
    errs = np.concatenate(errs)
    taus = np.concatenate(taus)
    if errs.size == 0:
        return {"diff_mae": 0.0, "kendall_tau": 0.0, "n_pairs": 0.0, "mean_pool_size": float(pool)}
    return {
        "diff_mae": float(errs.mean()),
        "kendall_tau": float(taus.mean()),
        "n_pairs": float(errs.size),
        "mean_pool_size": float(pool),
    }
