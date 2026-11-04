import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

torch.set_num_threads(1)

K = 8            # channel / resource types
T = 20           # events per history
TP = 12          # prefix length (older 60%)
HID = 32
N_POOL = 1200    # labeled pool (before label fraction)
N_PRE = 1000     # unlabeled pretraining histories
N_TEST = 3000
BUDGET = 0.005
EPS = 1.0        # profile-match threshold (L1 on log-profile)
TAU = 0.2
LAM_ACT = 0.30
LAM_DISP = 0.05
SIGMA0 = 0.02
PRE_EPOCHS = 5
PRE_BS = 64
FT_STEPS = 60
FT_BS = 128
LR = 3e-3


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gen(rng, n, rho, domain):
    """Synthetic payment histories. Hidden risk r drives channel switching
    (resource usage); activity (amount/timing) correlates with r by rho."""
    r = rng.standard_normal(n)
    act = rho * r + math.sqrt(max(1 - rho ** 2, 0.0)) * rng.standard_normal(n)
    if domain == "same":
        slope, b0, alpha, a_off, d_off = 1.5, -0.3, 1.5, 0.0, 0.0
    else:
        slope, b0, alpha, a_off, d_off = 1.0, 0.2, 0.4, 0.6, 0.4
    psw = _sigmoid(slope * r + b0)
    pref = rng.dirichlet(alpha * np.ones(K), size=n)
    cum = np.cumsum(pref, axis=1)
    chan = np.zeros((n, T), dtype=np.int64)
    u0 = rng.random(n)
    chan[:, 0] = np.minimum((cum < u0[:, None]).sum(1), K - 1)
    for t in range(1, T):
        sw = rng.random(n) < psw
        u = rng.random(n)
        new = np.minimum((cum < u[:, None]).sum(1), K - 1)
        chan[:, t] = np.where(sw, new, chan[:, t - 1])
    la = 3.0 + a_off + 0.7 * act[:, None] + 0.5 * rng.standard_normal((n, T))
    ld = 1.0 + d_off - 0.5 * act[:, None] + 0.5 * rng.standard_normal((n, T))
    logit = 2.5 * r - 2.5
    y = (rng.random(n) < _sigmoid(logit)).astype(np.float32)
    onehot = np.eye(K, dtype=np.float32)[chan]
    X = np.concatenate([(la - 3.0)[..., None], (ld - 1.0)[..., None], onehot], axis=2).astype(np.float32)
    return dict(X=X, y=y, la=la, ld=ld, chan=chan)


def future_summary(d):
    sl = slice(TP, T)
    la, ld, ch = d["la"][:, sl], d["ld"][:, sl], d["chan"][:, sl]
    sw = (ch[:, 1:] != ch[:, :-1]).mean(1)
    occ = np.stack([(ch == k).mean(1) for k in range(K)], 1)
    distinct = (occ > 0).sum(1) / K
    S = np.column_stack([la.mean(1), la.std(1), ld.mean(1), ld.std(1), sw, distinct, occ])
    pi = np.column_stack([np.log1p(np.exp(la).sum(1)),
                          np.full(len(la), math.log(1 + la.shape[1])),
                          np.log1p(np.exp(ld).sum(1))])
    return S.astype(np.float32), pi.astype(np.float32)


def activity_target(d):
    la, ld = d["la"], d["ld"]
    A = np.column_stack([la.mean(1), la.std(1), ld.mean(1), ld.std(1), np.log1p(np.exp(la).sum(1))])
    return A.astype(np.float32)


def standardize(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-6)


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2 + K, HID, batch_first=True)

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]


def pretrain(enc, qp, qf, qa, pdata, matched, seed):
    X = torch.tensor(pdata["X"][:, :TP])
    S, pi = future_summary(pdata)
    S = torch.tensor(standardize(S))
    pi = torch.tensor(pi)
    A = torch.tensor(standardize(activity_target(pdata)))
    params = list(enc.parameters()) + list(qp.parameters()) + list(qf.parameters()) + list(qa.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    rb = np.random.default_rng(seed + 5)
    n = X.shape[0]
    for _ in range(PRE_EPOCHS):
        perm = rb.permutation(n)
        for s in range(0, n - 8 + 1, PRE_BS):
            idx = torch.tensor(perm[s:s + PRE_BS])
            B = len(idx)
            r = enc(X[idx])
            zp = F.normalize(qp(r), dim=1)
            zf = F.normalize(qf(S[idx]), dim=1)
            logits = zp @ zf.T / TAU
            if matched:
                d = torch.cdist(pi[idx], pi[idx], p=1)
                M = d <= EPS
                M = M | torch.eye(B, dtype=torch.bool)
                only_self = (M.sum(1) == 1)
                M[only_self] = True
                logits = logits.masked_fill(~M, float("-inf"))
            l_pfc = F.cross_entropy(logits, torch.arange(B))
            l_act = F.smooth_l1_loss(qa(r), A[idx])
            l_disp = torch.clamp(SIGMA0 - r.std(0).mean(), min=0.0)
            loss = l_pfc + LAM_ACT * l_act + LAM_DISP * l_disp
            opt.zero_grad()
            loss.backward()
            opt.step()


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "scratch")
    frac = float(config.get("label_frac", 0.25))
    rho = float(config.get("act_corr", 0.0))
    pdom = config.get("pretrain_domain", "same")

    # data (independent of method)
    rng = np.random.default_rng(seed)
    pool = gen(rng, N_POOL, rho, "same")
    test = gen(rng, N_TEST, rho, "same")
    prng = np.random.default_rng(seed + 1000003)
    pre = gen(prng, N_PRE, rho, pdom)  # drawn for every arm so streams match

    n_lab = int(round(frac * N_POOL))
    Xl = torch.tensor(pool["X"][:n_lab])
    yl = torch.tensor(pool["y"][:n_lab])
    Xt = torch.tensor(test["X"])
    yt = test["y"]

    # identical inits across arms
    torch.manual_seed(seed + 1)
    enc = Enc()
    qp = nn.Linear(HID, 16)
    qf = nn.Sequential(nn.Linear(6 + K, 32), nn.ReLU(), nn.Linear(32, 16))
    qa = nn.Linear(HID, 5)
    torch.manual_seed(seed + 2)
    head = nn.Linear(HID, 1)

    if method in ("pfc_matched", "pfc_random"):
        pretrain(enc, qp, qf, qa, pre, method == "pfc_matched", seed)

    # fine-tune whole encoder + head, class-balanced loss
    npos = float(yl.sum())
    nneg = float(n_lab - npos)
    w1 = n_lab / (2 * npos) if npos > 0 else 1.0
    w0 = n_lab / (2 * nneg) if nneg > 0 else 1.0
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=LR)
    rf = np.random.default_rng(seed + 7)
    bs = min(FT_BS, n_lab)
    for _ in range(FT_STEPS):
        idx = torch.tensor(rf.choice(n_lab, size=bs, replace=False))
        lg = head(enc(Xl[idx])).squeeze(1)
        yb = yl[idx]
        wt = torch.where(yb > 0.5, torch.tensor(w1), torch.tensor(w0))
        loss = (F.binary_cross_entropy_with_logits(lg, yb, reduction="none") * wt).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        sc = head(enc(Xt)).squeeze(1).numpy()
    k = max(1, int(round(BUDGET * N_TEST)))
    top = np.argsort(-sc)[:k]
    tp = float(yt[top].sum())
    npos_t = float(yt.sum())
    return {
        "ap": float(average_precision_score(yt, sc)),
        "recall_budget": tp / max(npos_t, 1.0),
        "precision_budget": tp / k,
        "auc": float(roc_auc_score(yt, sc)),
        "test_prevalence": npos_t / N_TEST,
    }
