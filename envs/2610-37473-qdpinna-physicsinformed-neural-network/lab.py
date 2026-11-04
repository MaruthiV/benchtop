import math
from functools import lru_cache

import numpy as np
from scipy.linalg import solve_banded
import torch
import torch.nn as nn

LAM = 1.0          # hazard rate of the change point
N_REF = 800        # fine reference grid
STEPS = 200        # fixed PINN training budget
HIDDEN = 16


def _fd_solve(N, c, rho, lam=LAM):
    """Policy iteration for min{ A V + c*p, G - V } = 0 on a uniform grid.
    A = (rho/2) p^2 (1-p)^2 d^2/dp^2 + lam (1-p) d/dp (upwind), G = 1-p,
    V(1) = G(1) = 0. Returns grid, value, boundary (midpoint of last
    continuation node and first stopping node)."""
    p = np.linspace(0.0, 1.0, N)
    h = 1.0 / (N - 1)
    a = 0.5 * rho * p ** 2 * (1 - p) ** 2
    b = lam * (1 - p)
    main = -2 * a / h ** 2 - b / h
    up = a / h ** 2 + b / h
    lo = a / h ** 2
    main[-1] = 0.0
    up[-1] = 0.0
    lo[-1] = 0.0
    G = 1 - p
    stop = np.ones(N, dtype=bool)
    V = G.copy()
    for _ in range(100):
        d = stop.astype(float)
        cont = 1.0 - d
        ab = np.zeros((3, N))
        ab[0, 1:] = cont[:-1] * up[:-1]
        ab[1] = d + cont * main
        ab[2, :-1] = cont[1:] * lo[1:]
        rhs = np.where(stop, G, -c * p)
        V = solve_banded((1, 1), ab, rhs)
        AV = main * V
        AV[:-1] += up[:-1] * V[1:]
        AV[1:] += lo[1:] * V[:-1]
        new_stop = (G - V) <= (AV + c * p)
        new_stop[-1] = True
        if np.array_equal(new_stop, stop):
            break
        stop = new_stop
    k = int(np.argmax(stop))
    if k == 0:
        bnd = 0.0
    else:
        bnd = 0.5 * (p[k - 1] + p[k])
    return p, V, float(bnd)


@lru_cache(maxsize=None)
def _reference(c, rho):
    return _fd_solve(N_REF, c, rho)


def _pinn(c, rho, ncol, seed, complementarity):
    torch.set_num_threads(1)
    # data rng (independent of method)
    drng = np.random.default_rng(seed)
    pi_col = drng.uniform(0.005, 0.995, size=ncol).astype(np.float32)
    # init rng (independent of method)
    irng = np.random.default_rng(seed + 777)
    s0 = 0.5 + 0.2 * (irng.random() - 0.5)
    torch.manual_seed(seed + 1000003)
    net = nn.Sequential(nn.Linear(1, HIDDEN), nn.Tanh(),
                        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                        nn.Linear(HIDDEN, 1))
    raw = nn.Parameter(torch.tensor(math.log(s0 / (1 - s0)), dtype=torch.float32))

    def V_fn(x):
        return (1 - x) * net(x[:, None])[:, 0]

    opt = torch.optim.Adam([{"params": list(net.parameters()), "lr": 1e-2},
                            {"params": [raw], "lr": 2e-2}])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.05 ** (1.0 / STEPS))
    p = torch.tensor(pi_col)
    G = 1 - p
    a = 0.5 * rho * p ** 2 * (1 - p) ** 2
    bdrift = LAM * (1 - p)
    for _ in range(STEPS):
        opt.zero_grad()
        s = torch.sigmoid(raw)
        x = torch.cat([p, s.reshape(1)])
        V = V_fn(x)
        Vp = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        Vpp = torch.autograd.grad(Vp.sum(), x, create_graph=True)[0]
        Vc, Vpc, Vppc = V[:-1], Vp[:-1], Vpp[:-1]
        Vs, Vps = V[-1], Vp[-1]
        RC = a * Vppc + bdrift * Vpc + c * p
        RS = Vc - G
        mC = (p < s.detach()).float()
        mS = 1 - mC
        loss = (RC ** 2 * mC).mean() + (RS ** 2 * mS).mean() \
            + (Vs - (1 - s)) ** 2 + (Vps + 1.0) ** 2
        if complementarity:
            tC = torch.relu(-RC) * mC
            tS = torch.relu(RS) * mS
            u, w = RC, -RS
            phi = u + w - torch.sqrt(u ** 2 + w ** 2 + 1e-12)
            loss = loss + (tC ** 2).mean() + (tS ** 2).mean() + (phi ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        s_fin = float(torch.sigmoid(raw))
        pe = torch.tensor(np.linspace(0.0025, 0.9975, 200).astype(np.float32))
        Ve = V_fn(pe).numpy().astype(float)
    return s_fin, Ve


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "pinn_qd")
    c = float(config.get("c", 1.0))
    sigma = float(config.get("sigma", 1.0))
    ncol = int(config.get("n_col", 256))
    rho = 1.0 / sigma ** 2

    pref, Vref, bref = _reference(c, rho)
    p_eval = np.linspace(0.0025, 0.9975, 200)
    Vref_e = np.interp(p_eval, pref, Vref)

    if method == "fd_coarse":
        pg, Vg, bnd = _fd_solve(ncol, c, rho)
        Ve = np.interp(p_eval, pg, Vg)
    elif method == "pinn_qd":
        bnd, Ve = _pinn(c, rho, ncol, seed, True)
    elif method == "pinn_no_complementarity":
        bnd, Ve = _pinn(c, rho, ncol, seed, False)
    else:
        raise ValueError("unknown method %s" % method)

    return {
        "boundary_abs_error": float(abs(bnd - bref)),
        "value_mae": float(np.mean(np.abs(Ve - Vref_e))),
        "pi_star": float(bnd),
        "ref_boundary": float(bref),
    }
