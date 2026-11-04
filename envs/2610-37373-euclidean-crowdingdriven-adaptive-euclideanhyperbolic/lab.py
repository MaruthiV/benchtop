import math
import numpy as np
import torch
from functools import lru_cache
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

torch.set_num_threads(1)

STEPS = 80
LR = 0.08
MAX_TRAIN_PAIRS = 2000
MAX_TEST_PAIRS = 6000


@lru_cache(maxsize=None)
def _tree(b, L):
    parents = [-1]
    depth = [0]
    frontier = [0]
    for lvl in range(1, L + 1):
        nxt = []
        for p in frontier:
            for _ in range(b):
                parents.append(p)
                depth.append(lvl)
                nxt.append(len(parents) - 1)
        frontier = nxt
    n = len(parents)
    rows = np.arange(1, n)
    cols = np.array(parents[1:])
    A = coo_matrix((np.ones(n - 1), (rows, cols)), shape=(n, n)).tocsr()
    D = shortest_path(A, directed=False, unweighted=True)
    n1 = (D <= 1).sum(1) - 1
    n2 = (D <= 2).sum(1) - 1
    logr = np.log(n2 / np.maximum(n1, 1))
    return D, np.array(depth), logr


def _eucl(x, I, J):
    diff = x[I] - x[J]
    return torch.sqrt((diff ** 2).sum(-1) + 1e-12)


def _exp0(v):
    nrm = torch.sqrt((v ** 2).sum(-1, keepdim=True) + 1e-12)
    t = torch.tanh(nrm).clamp(max=1 - 1e-9)
    return t * v / nrm


def _hyp(h, I, J):
    x, y = h[I], h[J]
    num = 2 * ((x - y) ** 2).sum(-1)
    den = (1 - (x ** 2).sum(-1)) * (1 - (y ** 2).sum(-1))
    arg = 1 + num / den.clamp(min=1e-12)
    return torch.acosh(arg.clamp(min=1 + 1e-7))


def run(config, seed):
    method = config["method"]
    b = int(config["branching"])
    L = int(config["depth"])
    d = int(config["dim"])
    D, depth, logr = _tree(b, L)
    n = D.shape[0]

    # data rng: pair split (identical across methods)
    drng = np.random.default_rng(seed)
    iu0, iu1 = np.triu_indices(n, 1)
    perm = drng.permutation(len(iu0))
    half = len(perm) // 2
    ntr = min(half, MAX_TRAIN_PAIRS)
    tr = perm[:ntr]
    te = perm[half:half + MAX_TEST_PAIRS]
    # model-init rng (identical across methods)
    irng = np.random.default_rng(seed + 10007)
    Z0 = 0.1 * irng.standard_normal((n, d))

    def pairs(idx):
        return (torch.as_tensor(iu0[idx]), torch.as_tensor(iu1[idx]),
                torch.as_tensor(D[iu0[idx], iu1[idx]], dtype=torch.float64))

    Itr, Jtr, dtr = pairs(tr)
    Ite, Jte, dte_t = pairs(te)
    leaves = np.where(depth == L)[0]
    la, lb = np.triu_indices(len(leaves), 1)
    Ilf = torch.as_tensor(leaves[la])
    Jlf = torch.as_tensor(leaves[lb])

    use_e = method in ("euclidean", "fusion")
    use_h = method in ("hyperbolic", "fusion")
    params = []
    xE = vH = gw = gb = None
    log_s = torch.zeros((), dtype=torch.float64, requires_grad=True)
    params.append(log_s)
    if use_e:
        xE = torch.tensor(Z0, dtype=torch.float64, requires_grad=True)
        params.append(xE)
    if use_h:
        vH = torch.tensor(Z0, dtype=torch.float64, requires_grad=True)
        params.append(vH)
    if method == "fusion":
        gw = torch.ones((), dtype=torch.float64, requires_grad=True)
        gb = torch.zeros((), dtype=torch.float64, requires_grad=True)
        params += [gw, gb]
    lr_c = torch.as_tensor(logr - logr.mean(), dtype=torch.float64)

    def dist(I, J):
        if method == "euclidean":
            return _eucl(xE, I, J)
        h = _exp0(vH)
        if method == "hyperbolic":
            return _hyp(h, I, J)
        g = torch.sigmoid(gw * lr_c + gb)
        gij = 0.5 * (g[I] + g[J])
        return (1 - gij) * _eucl(xE, I, J) + gij * _hyp(h, I, J)

    opt = torch.optim.Adam(params, lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        de = dist(Itr, Jtr)
        loss = (((de / (torch.exp(log_s) * dtr)) - 1) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        s = torch.exp(log_s)

        def mult(I, J, dt):
            ratio = (dist(I, J) / (s * dt)).clamp(min=1e-6)
            return torch.maximum(ratio, 1.0 / ratio), ratio

        m_te, r_te = mult(Ite, Jte, dte_t)
        m_tr, _ = mult(Itr, Jtr, dtr)
        dl = dist(Ilf, Jlf)
        sep = float(dl.min() / dl.median().clamp(min=1e-12))
        res = {
            "avg_distortion": float(m_te.mean()),
            "max_distortion": float(m_te.max()),
            "mean_rel_error": float((r_te - 1).abs().mean()),
            "leaf_sep_norm": sep,
            "train_avg_distortion": float(m_tr.mean()),
        }
    for k, v in res.items():
        if not math.isfinite(v):
            res[k] = 1e6
    return res
