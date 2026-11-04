import math
import numpy as np

E = 80
C = 5
R = 4
DIM = 16
EPOCHS = 10
BATCHES = 4
LR = 0.08
GAMMA = 1.0
BETA0 = 1.5
CAP_MULT = 4
N_TEST = 300


def _build_world(seed, p):
    rng = np.random.default_rng(seed)
    cluster = np.arange(E) // (E // C)
    pop = np.exp(0.8 * rng.normal(size=E))
    pop = pop / pop.mean()
    compat = np.zeros((R, E, E), dtype=bool)
    for r in range(R):
        perm = rng.permutation(C)
        compat[r] = cluster[None, :] == perm[cluster][:, None]
    a = rng.normal(size=(R, E, 3))
    b = rng.normal(size=(R, E, 3))
    u_true = rng.random((R, E, E))
    u_obs = rng.random((R, E, E))
    test_perm_key = rng.random(R * E * E)
    dot = np.einsum('rhk,rtk->rht', a, b)
    q = pop[None, :, None] * pop[None, None, :] * np.exp(0.7 * dot)
    q = np.where(compat, q, 0.0)
    q = q * (0.2 / q[compat].mean())
    q = np.clip(q, 0, 0.9)
    true = compat & (u_true < q)
    obs = true & (u_obs < p)
    test = true & ~obs
    idx = np.argwhere(test)
    if len(idx) > N_TEST:
        keys = test_perm_key[idx[:, 0] * E * E + idx[:, 1] * E + idx[:, 2]]
        idx = idx[np.argsort(keys)[:N_TEST]]
    return true, obs, idx


def _budgets(obs, tr, K):
    n = len(tr)
    d_tail = obs.sum(axis=2)
    d_head = obs.sum(axis=1)
    d = 0.5 * (d_tail[tr[:, 0], tr[:, 1]] + d_head[tr[:, 0], tr[:, 2]])
    d = np.maximum(d, 1.0)
    w = (d.min() / d) ** GAMMA
    target = K * n
    cap = CAP_MULT * K
    lo, hi = 0.0, 1000.0 * K
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        bud = np.clip(np.rint(np.maximum(1.0, mid * w)), 1, cap)
        if bud.sum() > target:
            hi = mid
        else:
            lo = mid
    return np.clip(np.rint(np.maximum(1.0, lo * w)), 1, cap).astype(np.int64)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _score_grads(model, h, r, t):
    """Return score and gradients of score wrt h, r, t."""
    if model == "TransE":
        d = h + r - t
        nrm = np.sqrt((d ** 2).sum(-1) + 1e-9)
        s = 3.0 - nrm
        g = d / nrm[..., None]
        return s, -g, -g, g
    s = (h * r * t).sum(-1)
    return s, r * t, h * t, h * r


def run(config, seed):
    method = config.get("method", "uniform")
    p = float(config.get("p", 0.6))
    K = int(config.get("K", 8))
    model = config.get("model", "TransE")

    true, obs, test_idx = _build_world(seed, p)
    tr = np.argwhere(obs)
    n = len(tr)
    W = CAP_MULT * K

    if method == "radar":
        bud = _budgets(obs, tr, K)
    else:
        bud = np.full(n, K, dtype=np.int64)

    rng_i = np.random.default_rng(seed + 1000)
    rng_s = np.random.default_rng(seed + 2000)
    if model == "TransE":
        ent = rng_i.normal(size=(E, DIM)) / math.sqrt(DIM)
        rel = rng_i.normal(size=(R, DIM)) / math.sqrt(DIM)
    else:
        ent = 0.3 * rng_i.normal(size=(E, DIM))
        rel = 0.3 * rng_i.normal(size=(R, DIM)) + 0.5

    params = [ent, rel]
    m_ = [np.zeros_like(x) for x in params]
    v_ = [np.zeros_like(x) for x in params]
    step = 0

    ar = np.arange(W)[None, :]
    fn_cnt = 0.0
    val_cnt = 0.0

    for ep in range(EPOCHS):
        perm = rng_s.permutation(n)
        for bi in np.array_split(perm, BATCHES):
            B = len(bi)
            ne = rng_s.integers(0, E, size=(B, W))
            side = rng_s.random((B, W)) < 0.5
            r_, h_, t_ = tr[bi, 0], tr[bi, 1], tr[bi, 2]
            nh = np.where(side, h_[:, None], ne)
            nt = np.where(side, ne, t_[:, None])
            rr = np.broadcast_to(r_[:, None], (B, W))
            valid = (ar < bud[bi][:, None]) & ~obs[rr, nh, nt]
            vf = valid.astype(np.float64)
            fn_cnt += float((true[rr, nh, nt] & valid).sum())
            val_cnt += float(vf.sum())

            sp, gph, gpr, gpt = _score_grads(model, ent[h_], rel[r_], ent[t_])
            sn, gnh, gnr, gnt = _score_grads(model, ent[nh], rel[rr], ent[nt])

            if method == "uniform":
                w = vf
            else:
                s = sn
                nv = np.maximum(vf.sum(1, keepdims=True), 1.0)
                mean = (s * vf).sum(1, keepdims=True) / nv
                var = (((s - mean) ** 2) * vf).sum(1, keepdims=True) / nv
                sd = np.sqrt(var + 1e-6)
                if method == "selfadv":
                    logits = 1.0 * s
                else:
                    z = (s - mean) / sd
                    zm = np.where(valid, z, -1e9)
                    zm = zm - zm.max(1, keepdims=True)
                    ez = np.exp(zm) * vf
                    pz = ez / np.maximum(ez.sum(1, keepdims=True), 1e-12)
                    H = -(pz * np.log(pz + 1e-12) * vf).sum(1, keepdims=True)
                    c = 1.0 - H / np.log(np.maximum(nv, 2.0))
                    sh = np.clip(sd / np.median(sd), 0.25, 4.0)
                    beta = BETA0 * (0.5 + sh) * (0.5 + np.maximum(c, 0.0))
                    logits = beta * z
                lm = np.where(valid, logits, -1e9)
                lm = lm - lm.max(1, keepdims=True)
                e = np.exp(lm) * vf
                sm = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
                w = sm * nv * vf

            # gradients of mean loss
            cp = (-(1.0 - _sigmoid(sp)) / B)[:, None]           # (B,1)
            cn = (w * _sigmoid(sn) / (K * B))[..., None]        # (B,W,1)

            gE = np.zeros_like(ent)
            gR = np.zeros_like(rel)
            np.add.at(gE, h_, cp * gph)
            np.add.at(gE, t_, cp * gpt)
            np.add.at(gR, r_, cp * gpr)
            np.add.at(gE, nh.ravel(), (cn * gnh).reshape(-1, DIM))
            np.add.at(gE, nt.ravel(), (cn * gnt).reshape(-1, DIM))
            np.add.at(gR, rr.ravel(), (cn * gnr).reshape(-1, DIM))
            if model == "DistMult":
                np.add.at(gE, h_, 2e-3 * ent[h_] / B)
                np.add.at(gE, t_, 2e-3 * ent[t_] / B)

            step += 1
            for i, (x, g) in enumerate(zip(params, [gE, gR])):
                m_[i] = 0.9 * m_[i] + 0.1 * g
                v_[i] = 0.999 * v_[i] + 0.001 * g * g
                mh = m_[i] / (1 - 0.9 ** step)
                vh = v_[i] / (1 - 0.999 ** step)
                x -= LR * mh / (np.sqrt(vh) + 1e-8)

            if model == "TransE":
                nrm = np.maximum(np.linalg.norm(ent, axis=1, keepdims=True), 1.0)
                ent /= nrm

    # evaluation: filtered ranking against the closed world
    r_, h_, t_ = test_idx[:, 0], test_idx[:, 1], test_idx[:, 2]
    if model == "TransE":
        q1 = (ent[h_] + rel[r_])[:, None, :] - ent[None, :, :]
        tail_s = 3.0 - np.sqrt((q1 ** 2).sum(-1))
        q2 = (ent[t_] - rel[r_])[:, None, :] - ent[None, :, :]
        head_s = 3.0 - np.sqrt((q2 ** 2).sum(-1))
    else:
        tail_s = (ent[h_] * rel[r_]) @ ent.T
        head_s = (ent[t_] * rel[r_]) @ ent.T
    ar_n = np.arange(len(test_idx))
    tmask = true[r_, h_, :].copy()
    tmask[ar_n, t_] = False
    hmask = true[r_, :, t_].copy()
    hmask[ar_n, h_] = False
    tail_s = np.where(tmask, -1e9, tail_s)
    head_s = np.where(hmask, -1e9, head_s)
    tr_rank = 1 + (tail_s > tail_s[ar_n, t_][:, None]).sum(1)
    hr_rank = 1 + (head_s > head_s[ar_n, h_][:, None]).sum(1)
    ranks = np.concatenate([tr_rank, hr_rank]).astype(np.float64)
    mrr = float((1.0 / ranks).mean())
    hits10 = float((ranks <= 10).mean())

    return {
        "mrr": mrr,
        "hits10": hits10,
        "fn_rate": fn_cnt / max(val_cnt, 1.0),
        "neg_used_per_pos": val_cnt / (EPOCHS * n),
        "n_train": float(n),
    }
