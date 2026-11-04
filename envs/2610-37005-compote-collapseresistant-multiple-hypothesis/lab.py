import math
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

N_REGIONS = 8
ITEMS_PER_REGION = 8
N_ITEMS = N_REGIONS * ITEMS_PER_REGION
HIST = 3
N_TRAIN = 2000
N_TEST = 1000
STEPS = 300
HIDDEN = 64
LR = 0.02
ANNEAL_FRAC = 0.7
TAU0, TAU1 = 10.0, 0.01


def _make_data(seed, k_true):
    rng = np.random.default_rng(seed)
    ang = 2 * math.pi * np.arange(N_REGIONS) / N_REGIONS
    centers = 3.0 * np.stack([np.cos(ang), np.sin(ang)], 1)
    item_emb = np.zeros((N_ITEMS, 2))
    for r in range(N_REGIONS):
        item_emb[r * ITEMS_PER_REGION:(r + 1) * ITEMS_PER_REGION] = (
            centers[r] + 0.5 * rng.standard_normal((ITEMS_PER_REGION, 2)))
    # per source region: random ordering of target regions (nested over k_true)
    perms = np.stack([rng.permutation(N_REGIONS) for _ in range(N_REGIONS)])
    # per target region popularity ranks
    pop = np.stack([rng.permutation(ITEMS_PER_REGION) for _ in range(N_REGIONS)])
    w = 1.0 / (np.arange(ITEMS_PER_REGION) + 1.0)
    w = w / w.sum()
    cw = np.cumsum(w)

    def sample(n):
        src = rng.integers(0, N_REGIONS, n)
        hist = rng.integers(0, ITEMS_PER_REGION, (n, HIST)) + src[:, None] * ITEMS_PER_REGION
        u1 = rng.random(n)
        u2 = rng.random(n)
        slot = np.minimum((u1 * k_true).astype(int), k_true - 1)
        tr = perms[src, slot]
        rank = np.minimum(np.searchsorted(cw, u2), ITEMS_PER_REGION - 1)
        tgt = tr * ITEMS_PER_REGION + pop[tr, rank]
        x = item_emb[hist].reshape(n, HIST * 2)
        x = x + 0.1 * rng.standard_normal(x.shape)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(tgt, dtype=torch.long)

    xtr, ytr = sample(N_TRAIN)
    xte, yte = sample(N_TEST)
    return xtr, ytr, xte, yte


def run(config, seed):
    method = config.get("method", "compote")
    k_true = int(config.get("k_true", 2))
    H = int(config.get("n_hyp", 4))
    xtr, ytr, xte, yte = _make_data(seed, k_true)

    d_in = HIST * 2
    N = N_ITEMS
    if method == "single":
        heads = 1
        hid = int(round((HIDDEN * d_in + (HIDDEN + 1) * H * N - N) / (d_in + N)))
    else:
        heads = H
        hid = HIDDEN

    torch.manual_seed(seed + 100003)
    trunk = torch.nn.Linear(d_in, hid)
    out = torch.nn.Linear(hid, heads * N)
    params = list(trunk.parameters()) + list(out.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    def fwd(x):
        return F.log_softmax(out(torch.tanh(trunk(x))).view(-1, heads, N), dim=-1)

    B = xtr.shape[0]
    ar = torch.arange(B)
    for t in range(STEPS):
        logp = fwd(xtr)
        lp = logp[ar, :, ytr]  # (B, heads)
        if heads == 1 or method == "uniform":
            g = torch.full_like(lp, 1.0 / heads)
        elif method == "wta":
            g = F.one_hot(lp.detach().argmax(1), heads).float()
        else:  # compote: annealed
            frac = min(1.0, t / (ANNEAL_FRAC * STEPS))
            tau = TAU0 * (TAU1 / TAU0) ** frac
            g = torch.softmax(lp.detach() / tau, dim=1)
        loss = -(g * lp).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        # held-out evaluation
        p = fwd(xte).exp()
        smax = p.max(1).values
        spool = p.mean(1)

        def recall(s, k):
            top = torch.topk(s, k, dim=1).indices
            return (top == yte[:, None]).any(1).float().mean().item()

        # collapse diagnostics on training data
        lp = fwd(xtr)[ar, :, ytr]
        win = lp.argmax(1)
        share = torch.bincount(win, minlength=heads).float() / B
        active = float((share >= 0.05).sum().item())
        ent = float(-(share[share > 0] * share[share > 0].log()).sum().item())

    return {
        "recall5": float(recall(smax, 5)),
        "recall10": float(recall(smax, 10)),
        "recall5_pooled": float(recall(spool, 5)),
        "active_heads": active,
        "win_entropy": ent,
    }
