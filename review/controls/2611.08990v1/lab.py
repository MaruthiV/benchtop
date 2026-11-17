"""Benchtop lab for arXiv:2611.08990: action-centered vs separate-arm contrast learning under a budget.

Miniature of the claim that action-centered contrast estimation stays unbiased under a
misspecified shared baseline, while separate-arm raw-reward regression can acquire contrast bias
when arm-specific designs differ (here through cost-dependent logging and adaptive selection).

Simplifications relative to the paper, identical across methods:
  * no offline doubly-robust warm start; a method-independent pilot of N0 logged rounds stands in
  * no projection onto the parameter ball, no cross-fitting, no randomized farthest-pair tie-break
    beyond the active-set rule below
  * one shared exploration rule (active-set farthest-pair mixture with a cost-dependent logging
    law) and one shared primal-dual affordable-action controller. Only the contrast estimator
    differs between methods.

All learners are closed-form ridge with Sherman-Morrison updates. run() is deterministic in
(config, seed). All randomness comes from three seeded generators: init_rng (pilot), data_rng
(main stream) and eval_rng (held-out contexts). None of them depends on the method, so the arms
share contexts, noise and uniform action draws (common random numbers).
"""
import numpy as np

D = 5            # context dimension
K = 3            # priced actions; index 0 is the free control
T_MAIN = 2000    # adaptive rounds
N0 = 30          # pilot rounds (logging law only)
ALPHA = 0.5      # ridge strength, same for both estimators
BETA = 1.0       # active-set width multiplier
ETA = 0.01       # primal-dual step size
EPS = 0.2        # weight of the cost-dependent logging law in the policy
F_CLIP = 12.0    # clip for the shared baseline estimate
N_EVAL = 1000    # fresh held-out contexts for CATE error

# true linear CATEs; row 0 is the control with zero effect
THETA = np.zeros((K + 1, D))
THETA[1] = [0.6, -0.3, 0.2, 0.0, 0.4]
THETA[2] = [-0.2, 0.5, 0.1, 0.3, -0.1]
THETA[3] = [0.1, 0.1, -0.5, 0.4, 0.2]
BASE_COST = np.array([0.0, 0.5, 1.0, 1.5])

MISSPEC = {"none": 0.0, "mild": 1.0, "strong": 3.0}
LOGGING = {"uniform": 0.0, "moderate": 1.5, "strong": 3.0}
BUDGET_RATE = {"loose": 1.5, "binding": 0.4}
METHODS = ("separate_arm", "action_centered")


def _lookup(table, key, name):
    try:
        return table[key]
    except KeyError:
        raise ValueError(f"unknown {name} level {key!r}; expected one of {sorted(table)}")


def _baseline(X, amp):
    # shared, action-independent outcome baseline f(x); amp=0 gives a linear baseline
    lin = 0.5 * X[:, 0] - 0.3 * X[:, 2]
    nl = np.sin(2.0 * X[:, 0]) + 1.5 * (X[:, 1] ** 2 - 0.3) + 0.8 * X[:, 2] * X[:, 3]
    return lin + amp * nl


def _costs(X):
    # per-context cost of each action; control column is zero
    mult = 1.0 + 0.4 * np.tanh(X[:, 1])
    return mult[:, None] * BASE_COST[None, :]


def _oracle_value(tau, cost, budget_total):
    """Fluid-LP value on the realized contexts, via its convex Lagrangian dual in lambda."""
    def dual(lam):
        return float(np.max(tau - lam * cost, axis=1).sum() + lam * budget_total)

    lo = 0.0
    hi = max(1.0, float(tau.max()) / float(cost[:, 1:].min()) + 1.0)
    for _ in range(120):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if dual(m1) < dual(m2):
            hi = m2
        else:
            lo = m1
    return min(dual(0.0), dual(0.5 * (lo + hi)))


def run(config, seed):
    method = config.get("method", "action_centered")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    amp = _lookup(MISSPEC, config.get("misspec", "mild"), "misspec")
    gamma = _lookup(LOGGING, config.get("logging", "moderate"), "logging")
    rate = _lookup(BUDGET_RATE, config.get("budget", "binding"), "budget")
    seed = int(seed)

    init_rng = np.random.default_rng([seed, 101])   # pilot draws (model warm start)
    data_rng = np.random.default_rng([seed, 202])   # main stream draws
    eval_rng = np.random.default_rng([seed, 303])   # held-out contexts

    N = N0 + T_MAIN
    budget_total = rate * N

    # all data-related randomness is drawn up front, independent of the method
    X = np.vstack([init_rng.uniform(-1, 1, (N0, D)), data_rng.uniform(-1, 1, (T_MAIN, D))])
    noise = np.concatenate([init_rng.uniform(-0.8, 0.8, N0), data_rng.uniform(-0.8, 0.8, T_MAIN)])
    U = np.concatenate([init_rng.uniform(size=N0), data_rng.uniform(size=T_MAIN)])
    X_eval = eval_rng.uniform(-1, 1, (N_EVAL, D))

    tau = np.zeros((N, K + 1))
    tau[:, 1:] = X @ THETA[1:].T
    fbase = _baseline(X, amp)
    cost = _costs(X)
    Zfull = np.zeros((N, K + 1, K * D))   # z_t(a) = e_a (x) X_t, control row is zero
    for k in range(1, K + 1):
        Zfull[:, k, (k - 1) * D:k * D] = X

    # shared design (centered Gram) and per-arm raw ridge statistics (inverses kept online)
    Vinv = np.eye(K * D) / ALPHA
    b = np.zeros(K * D)
    Ginv = np.tile(np.eye(D) / ALPHA, (K + 1, 1, 1))
    h = np.zeros((K + 1, D))
    lam = 0.0
    S = 0.0
    achieved = 0.0

    for t in range(N):
        x = X[t]
        c = cost[t]
        Z = Zfull[t]
        aff = c <= (budget_total - S) + 1e-12
        aff[0] = True

        w = np.einsum("kij,kj->ki", Ginv, h)
        fhat = float(np.clip(w[0] @ x, -F_CLIP, F_CLIP))
        if method == "action_centered":
            th = Vinv @ b
        else:
            th = (w[1:] - w[0]).ravel()
        s = Z @ th - lam * c

        # active set: keep a if no affordable a' beats it beyond the joint width
        dZ = Z[None, :, :] - Z[:, None, :]
        M = np.sqrt(np.maximum(np.einsum("abk,kl,abl->ab", dZ, Vinv, dZ), 0.0))
        viol = (s[None, :] - s[:, None]) - 2.0 * BETA * M
        ok = (viol <= 1e-12) | (~aff)[None, :]
        active = aff & ok.all(axis=1)

        Q = np.zeros(K + 1)
        idx = np.flatnonzero(active)
        if idx.size == 0:
            Q[int(np.argmax(np.where(aff, s, -np.inf)))] = 1.0
        elif idx.size == 1:
            Q[idx[0]] = 1.0
        else:
            sub = M[np.ix_(idx, idx)].copy()
            np.fill_diagonal(sub, -1.0)
            i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
            Q[idx[i]] = 0.5
            Q[idx[j]] = 0.5

        # cost-dependent logging law over the affordable set
        wl = np.exp(-gamma * c) * aff
        L = wl / wl.sum()
        pi = L if t < N0 else (1.0 - EPS) * Q + EPS * L

        cum = np.cumsum(pi)
        a = min(int(np.searchsorted(cum, U[t] * cum[-1], side="right")), K)

        cost_a = c[a]
        y = fbase[t] + tau[t, a] + noise[t]
        achieved += tau[t, a]
        S += cost_a

        # action-centered update (valid for any predictable baseline fhat)
        zbar = pi @ Z
        u = Z[a] - zbar
        g = y - fhat
        Vu = Vinv @ u
        Vinv -= np.outer(Vu, Vu) / (1.0 + u @ Vu)
        b += u * g

        # separate-arm raw-reward update for the chosen arm
        gx = Ginv[a] @ x
        Ginv[a] -= np.outer(gx, gx) / (1.0 + x @ gx)
        h[a] += x * y

        # primal-dual controller
        lam = max(0.0, lam + ETA * (cost_a - rate))

    w = np.einsum("kij,kj->ki", Ginv, h)
    if method == "action_centered":
        th_hat = (Vinv @ b).reshape(K, D)
    else:
        th_hat = w[1:] - w[0]
    err = th_hat - THETA[1:]
    param_err = float(np.mean(np.linalg.norm(err, axis=1)))
    rmse = float(np.sqrt(np.mean((X_eval @ err.T) ** 2)))

    opt = _oracle_value(tau, cost, budget_total)
    gap = (opt - achieved) / N

    return {
        "value_gap": float(gap),
        "cate_param_err": param_err,
        "cate_rmse": rmse,
        "budget_violation": float(max(0.0, S - budget_total) / budget_total),
        "spend_rate": float(S / N),
    }

_original_run = run

def run(config, seed):
    if config.get('probe') == 'oracle':
        from scipy.optimize import linprog
        tau = np.array([[0., .4, -.2, .8], [0., -.1, .6, .2], [0., .3, .2, .5]])
        cost = np.array([[0., .5, 1., 1.5], [0., .4, .8, 1.2], [0., .6, 1.2, 1.8]])
        budget = float(config['budget'])
        eq = np.zeros((3, 12))
        for t in range(3): eq[t, 4*t:4*t+4] = 1
        fit = linprog(-tau.ravel(), A_ub=cost.reshape(1,-1), b_ub=[budget], A_eq=eq, b_eq=np.ones(3), bounds=(0,None), method='highs')
        assert fit.success
        return {'oracle_lp_absolute_error': abs(_oracle_value(tau,cost,budget) + fit.fun)}
    import sys
    checks = {'moment_error':0.,'selected_feature_error':0.,'steps':0.}
    def trace(frame, event, arg):
        if frame.f_code is _original_run.__code__ and event == 'line' and frame.f_lineno == 174:
            l=frame.f_locals
            pi,Z=l['pi'],l['Z']; centered=Z-pi@Z
            residual=l['fbase'][l['t']]-l['fhat']+l['tau'][l['t']]
            lhs=np.einsum('a,ai,a->i',pi,centered,residual)
            M=np.einsum('a,ai,aj->ij',pi,centered,centered)
            checks['moment_error']=max(checks['moment_error'],float(np.max(np.abs(lhs-M@THETA[1:].ravel()))))
            checks['selected_feature_error']=max(checks['selected_feature_error'],float(np.max(np.abs(l['u']-centered[l['a']]))))
            checks['steps']+=1
        return trace
    sys.settrace(trace)
    try: result=_original_run(config,seed)
    finally: sys.settrace(None)
    return {**result,**checks}
