import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- constants (deterministic) ----------------
N_CELLS = 16
N_SYM = 4
N_SAMPLES = 16
LATENT = N_CELLS * N_SYM                   # 64
IN_DIM = N_CELLS * (N_SYM + 1) + LATENT    # 80 input features + 64 latent
HIDDEN = 128
BATCH = 128
TRAIN_STEPS = 250
N_TRAIN = 512
N_TEST = 256
N_TEST_SQUARES = 96
MAX_DESC_STEPS = 8      # energy training: up to 8 descent steps = 16 passes
MAX_BLOCK_STEPS = 16    # baseline training: up to 16 block passes
ALPHA_EVAL = 0.5
STOP_TAU = 0.1          # stop when ||grad|| < tau * ||grad at step 0||
LR = 2e-3


def _enumerate_latin4():
    """Enumerate all 576 4x4 Latin squares by backtracking (deterministic)."""
    grid = [-1] * 16
    out = []

    def rec(p):
        if p == 16:
            out.append(list(grid))
            return
        r, c = divmod(p, 4)
        for v in range(4):
            ok = True
            for j in range(c):
                if grid[r * 4 + j] == v:
                    ok = False
                    break
            if ok:
                for i in range(r):
                    if grid[i * 4 + c] == v:
                        ok = False
                        break
            if ok:
                grid[p] = v
                rec(p + 1)
                grid[p] = -1

    rec(0)
    return np.array(out, dtype=np.int64)


_SQF = _enumerate_latin4()           # (576, 16)
assert _SQF.shape == (576, 16)


# ---------------- puzzle generation ----------------
def _counts(masks, sol):
    """Number of Latin squares consistent with the givens, for each blank mask row."""
    givens = ~masks
    mism = (_SQF[None] != sol[None, None]) & givens[:, None, :]
    return (~mism.any(axis=2)).sum(axis=1)


def _valid_rows(mask, sol):
    givens = ~mask
    ok = (_SQF[:, givens] == sol[givens]).all(axis=1)
    return _SQF[ok]


def _make_puzzles(rng, pool, k, unique, n):
    masks, sols = [], []
    while len(masks) < n:
        sol = _SQF[pool[rng.integers(len(pool))]]
        if unique:
            m = np.zeros(N_CELLS, dtype=bool)
            m[rng.choice(N_CELLS, k, replace=False)] = True
            # restore givens until the completion is unique (monotone, so this terminates)
            while _counts(m[None], sol)[0] > 1:
                m[rng.choice(np.flatnonzero(m))] = False
            masks.append(m)
            sols.append(sol)
        else:
            T = 256
            keys = rng.random((T, N_CELLS))
            M = np.zeros((T, N_CELLS), dtype=bool)
            np.put_along_axis(M, np.argsort(keys, axis=1)[:, :k], True, axis=1)
            good = np.flatnonzero(_counts(M, sol) >= 2)
            for g in good:
                masks.append(M[g])
                sols.append(sol)
    return masks[:n], sols[:n]


def _pack(masks, sols):
    n = len(masks)
    X = np.zeros((n, N_CELLS, N_SYM + 1), dtype=np.float32)
    vlists = []
    blanks = np.zeros(n)
    for i, (m, s) in enumerate(zip(masks, sols)):
        g = ~m
        X[i, np.flatnonzero(g), s[g]] = 1.0
        X[i, np.flatnonzero(m), N_SYM] = 1.0
        vlists.append(_valid_rows(m, s))
        blanks[i] = m.sum()
    cnt = np.array([len(v) for v in vlists], dtype=np.int64)
    V = np.zeros((n, int(cnt.max()), N_CELLS), dtype=np.int64)
    for i, v in enumerate(vlists):
        V[i, :len(v)] = v
        V[i, len(v):] = v[0]  # padding duplicates a valid row; masked by cnt
    return X.reshape(n, -1), V, cnt, blanks


# ---------------- models ----------------
def _build(out_dim, seed):
    # first two layers are identical across methods for the same seed
    torch.manual_seed(seed + 101)
    return nn.Sequential(
        nn.Linear(IN_DIM, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, out_dim),
    )


def _block(net, z, x):
    return z + net(torch.cat([z, x], dim=1))


def _energy(net, z, x):
    return net(torch.cat([z, x], dim=1)).squeeze(-1)


def _is_valid(z, V, cnt):
    pred = z.detach().reshape(-1, N_CELLS, N_SYM).argmax(-1).numpy()   # (B,16)
    match = (pred[:, None, :] == V).all(-1)                             # (B,Vmax)
    within = np.arange(V.shape[1])[None, :] < cnt[:, None]
    return (match & within).any(1)


def _train_direct(net, Xtr, Vtr, cnt_tr, seed, rng):
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed + 202)
    Xt = torch.from_numpy(Xtr)
    for _ in range(TRAIN_STEPS):
        idx = rng.integers(0, Xtr.shape[0], BATCH)
        vi = (rng.random(BATCH) * cnt_tr[idx]).astype(np.int64)
        y = torch.from_numpy(Vtr[idx, vi])                 # (B,16) one sampled valid completion
        x = Xt[torch.from_numpy(idx)]
        n = int(rng.integers(1, MAX_BLOCK_STEPS + 1))
        z = torch.randn(BATCH, LATENT, generator=gen)
        for _ in range(n):
            z = _block(net, z, x)
        loss = F.cross_entropy(z.reshape(-1, N_SYM), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()


def _train_energy(E, Xtr, Vtr, cnt_tr, seed, rng):
    opt = torch.optim.Adam(E.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed + 202)
    Xt = torch.from_numpy(Xtr)
    for _ in range(TRAIN_STEPS):
        idx = rng.integers(0, Xtr.shape[0], BATCH)
        vi = (rng.random(BATCH) * cnt_tr[idx]).astype(np.int64)
        Vb = Vtr[idx]
        cb = cnt_tr[idx]
        y = torch.from_numpy(Vtr[idx, vi])
        x = Xt[torch.from_numpy(idx)]
        n = int(rng.integers(1, MAX_DESC_STEPS + 1))
        alpha = float(rng.choice(np.array([0.25, 0.5, 1.0])))
        # exploration: descend the current energy from noise (negatives / extra modes)
        z = torch.randn(BATCH, LATENT, generator=gen)
        for _ in range(n):
            zz = z.detach().requires_grad_(True)
            (gr,) = torch.autograd.grad(_energy(E, zz, x).sum(), zz)
            z = (zz - alpha * gr).detach()
        valid = torch.from_numpy(_is_valid(z, Vb, cb))
        zpos = F.one_hot(y, N_SYM).float().reshape(BATCH, LATENT) * 2.0 - 1.0
        ep = _energy(E, zpos, x)
        en = _energy(E, z, x)
        # contrastive logistic loss: valid states low energy, invalid states high energy
        loss = (F.softplus(ep).mean()
                + torch.where(valid, F.softplus(en), F.softplus(-en)).mean()
                + 0.01 * (ep.pow(2).mean() + en.pow(2).mean()))
        opt.zero_grad()
        loss.backward()
        opt.step()


def _descend(E, Xr, Z, steps, alpha, tau):
    """Gradient descent on the energy. Each step costs 2 passes (forward + gradient)."""
    z = Z.clone()
    n = z.shape[0]
    active = torch.ones(n, dtype=torch.bool)
    passes = torch.zeros(n)
    g0 = None
    for _ in range(steps):
        zz = z.detach().requires_grad_(True)
        (gr,) = torch.autograd.grad(_energy(E, zz, Xr).sum(), zz)
        passes += 2.0 * active.float()
        gn = gr.norm(dim=1)
        if g0 is None:
            g0 = gn.clamp_min(1e-12)
        if tau is not None:
            active = active & (gn >= tau * g0)
        z = (zz - alpha * gr * active.float().unsqueeze(1)).detach()
    return z.detach(), passes.numpy(), (~active).numpy()


def _evaluate(method, net, Xte, Vte, cnt_te, budget, seed):
    P = Xte.shape[0]
    gen = torch.Generator().manual_seed(seed + 303)
    Z = torch.randn(P * N_SAMPLES, LATENT, generator=gen)   # shared across methods
    Xr = torch.from_numpy(Xte).repeat_interleave(N_SAMPLES, dim=0)

    if method == "direct_output":
        with torch.no_grad():
            z = Z
            for _ in range(budget):
                z = _block(net, z, Xr)
        z = z.detach()
        passes = np.full(P * N_SAMPLES, float(budget))
        stopped = np.zeros(P * N_SAMPLES, dtype=bool)
    else:
        z, passes, stopped = _descend(net, Xr, Z, budget // 2, ALPHA_EVAL, STOP_TAU)

    preds = z.reshape(P, N_SAMPLES, N_CELLS, N_SYM).argmax(-1).numpy()   # (P,S,16)
    match = (preds[:, :, None, :] == Vte[:, None, :, :]).all(-1)          # (P,S,Vmax)
    exact = match[:, 0, :].any(-1).mean()                                 # sample 0 only
    found = match.any(axis=1)                                             # (P,Vmax)
    within = np.arange(Vte.shape[1])[None, :] < cnt_te[:, None]
    coverage = ((found & within).sum(1) / cnt_te).mean()
    return {
        "exact_match_accuracy_at_matched_compute": float(exact),
        "coverage": float(coverage),
        "mean_passes": float(np.mean(passes)),
        "early_stop_frac": float(np.mean(stopped)),
    }


def _parse_blanks(v):
    if isinstance(v, str):
        return int(v.split()[0])
    return int(v)


def _parse_unique(v):
    s = str(v)
    if s == "unique solution":
        return True
    if s == "multiple solutions":
        return False
    raise ValueError(f"unknown solution_multiplicity: {v}")


# ---------------- entry point ----------------
def run(config: dict, seed: int) -> dict:
    torch.set_num_threads(1)
    seed = int(seed)
    method = str(config["method"])
    if method not in ("direct_output", "energy_descent"):
        raise ValueError(f"unknown method: {method}")
    k = _parse_blanks(config["puzzle_difficulty"])
    budget = int(config["inference_step_budget"])
    unique = _parse_unique(config["solution_multiplicity"])

    # Data and splits depend only on (seed, context), never on method.
    split_rng = np.random.default_rng([seed, 3])
    perm = split_rng.permutation(576)
    test_pool = perm[:N_TEST_SQUARES]
    train_pool = perm[N_TEST_SQUARES:]

    data_rng = np.random.default_rng([seed, 4])
    tr_m, tr_s = _make_puzzles(data_rng, train_pool, k, unique, N_TRAIN)
    te_m, te_s = _make_puzzles(data_rng, test_pool, k, unique, N_TEST)
    Xtr, Vtr, cnt_tr, _ = _pack(tr_m, tr_s)
    Xte, Vte, cnt_te, blanks_te = _pack(te_m, te_s)

    train_rng = np.random.default_rng([seed, 5])
    if method == "direct_output":
        net = _build(LATENT, seed)
        _train_direct(net, Xtr, Vtr, cnt_tr, seed, train_rng)
    else:
        net = _build(1, seed)
        _train_energy(net, Xtr, Vtr, cnt_tr, seed, train_rng)

    out = _evaluate(method, net, Xte, Vte, cnt_te, budget, seed)
    out["mean_blanks"] = float(blanks_te.mean())
    return out
