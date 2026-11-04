import math
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

D = 10          # feature dimension
N_SRC = 400     # source training points
N_BATCH = 24    # batches per stream
BS = 64         # batch size
NQ = 32         # query pool size per batch (rest is deployment set)
PRIOR = 1.0     # Dirichlet prior per class
LABEL_COST = 0.004   # label price in batch-accuracy units (PORTAL stop rule)
N_CAND = 3      # source, batch-norm, mean-logit ensemble


def _make_stream(config, seed):
    rng = np.random.default_rng(seed)  # data rng (independent of method)
    arrival = config.get("arrival", "mixed")
    s = float(config.get("shift", 0.5))
    mu = 0.4 * rng.choice([-1.0, 1.0], D)
    ys = rng.permutation(np.arange(N_SRC) % 2)
    Xs = (2 * ys - 1)[:, None] * mu + rng.normal(size=(N_SRC, D))
    scale = np.clip(1 + 0.4 * s * rng.uniform(-1, 1, D), 0.3, None)
    offset = 0.7 * s * rng.normal(size=D)
    batches = []
    for k in range(N_BATCH):
        if arrival == "mixed":
            p1 = 0.5
        else:
            p1 = 0.9 if (k // 4) % 2 == 1 else 0.1
        y = (rng.random(BS) < p1).astype(int)
        x = (2 * y - 1)[:, None] * mu + rng.normal(size=(BS, D))
        x = x * scale + offset
        qperm = rng.permutation(BS)
        batches.append((x, y, qperm[:NQ], qperm[NQ:]))
    return Xs, ys, batches


def _candidates(x, w, b, src_mean, src_std):
    z = (x - x.mean(0)) / (x.std(0) + 1e-6) * src_std + src_mean
    d_src = x @ w + b
    d_bn = z @ w + b
    d_ens = 0.5 * (d_src + d_bn)
    return np.stack([(d_src > 0), (d_bn > 0), (d_ens > 0)]).astype(int)


def _mus(alpha, w, pred):
    S = alpha.shape[0]
    pi = alpha / alpha.sum(1, keepdims=True)
    return (w[:, None] * pi[np.arange(S)[:, None], pred]).sum(0)


def _portal(P, y, qorder, E, budget):
    A = P.shape[0]
    codes = np.zeros(P.shape[1], dtype=int)
    for a in range(A):
        codes += P[a] << a
    uniq, cnt = np.unique(codes[E], return_counts=True)
    S = len(uniq)
    w = cnt / len(E)
    pred = np.array([[(u >> a) & 1 for a in range(A)] for u in uniq])
    remaining = [[i for i in qorder if codes[i] == u] for u in uniq]
    alpha = np.full((S, 2), PRIOR)
    used = 0

    def max_adv(a, j, b_rem):
        ss = [s for s in range(S) if pred[s, a] != pred[s, j]]
        if not ss:
            return -1.0
        dd = {s: alpha[s, pred[s, a]] - alpha[s, pred[s, j]] for s in ss}
        n = {s: alpha[s].sum() for s in ss}
        k = {s: 0 for s in ss}
        F = sum(w[s] * dd[s] / n[s] for s in ss)
        for _ in range(b_rem):
            best, bs_ = 0.0, None
            for s in ss:
                if k[s] < len(remaining[s]):
                    inc = w[s] * (n[s] - dd[s]) / ((n[s] + k[s]) * (n[s] + k[s] + 1))
                    if inc > best:
                        best, bs_ = inc, s
            if bs_ is None or best <= 1e-15:
                break
            F += best
            k[bs_] += 1
        return F

    for step in range(budget):
        mu = _mus(alpha, w, pred)
        j = int(np.argmax(mu))  # ties -> lowest index (incumbent first)
        b_rem = budget - step
        reach = [a for a in range(A) if a != j and max_adv(a, j, b_rem) > 1e-12]
        if not reach:
            break
        pi = alpha / alpha.sum(1, keepdims=True)
        n_s = alpha.sum(1)

        def vpair(s, a):
            pa, pj = pred[s, a], pred[s, j]
            if pa == pj:
                return 0.0
            return pi[s, pa] + pi[s, pj] - (pi[s, pa] - pi[s, pj]) ** 2

        U_sum = 0.0
        for a in range(A):
            if a == j:
                continue
            d = mu[j] - mu[a]
            s2 = sum(w[s] ** 2 / (n_s[s] + 1) * vpair(s, a) for s in range(S))
            U_sum += (math.sqrt(d * d + s2) - d) / 2
        U = min(1 - mu[j], U_sum)
        if U <= LABEL_COST:
            break
        best_key, best_s = None, None
        mx = mu.max()
        for s in range(S):
            if not remaining[s]:
                continue
            if not any(pred[s, a] != pred[s, j] for a in reach):
                continue
            V = -mx
            for c in range(2):
                al = alpha.copy()
                al[s, c] += 1
                V += pi[s, c] * _mus(al, w, pred).max()
            T = max(w[s] ** 2 * vpair(s, a) / ((n_s[s] + 1) * (n_s[s] + 2))
                    for a in reach)
            key = (round(V, 12), T)
            if best_key is None or key > best_key:
                best_key, best_s = key, s
        if best_s is None:
            break
        idx = remaining[best_s].pop(0)
        alpha[best_s, y[idx]] += 1
        used += 1
    mu = _mus(alpha, w, pred)
    return int(np.argmax(mu)), used


def run(config, seed):
    warnings.filterwarnings("ignore")
    method = config.get("method", "portal")
    budget = int(config.get("budget", 6))
    Xs, ys, batches = _make_stream(config, seed)
    clf = LogisticRegression(C=1.0, max_iter=200).fit(Xs, ys)
    w = clf.coef_[0]
    b = float(clf.intercept_[0])
    src_mean, src_std = Xs.mean(0), Xs.std(0)
    rng_q = np.random.default_rng(seed + 1_000_003)  # query randomness, separate

    accs, orac, labels, chose_bn = [], [], [], []
    for (x, y, Q, E) in batches:
        P = _candidates(x, w, b, src_mean, src_std)
        qorder = [int(i) for i in rng_q.permutation(Q)]
        acc_E = np.array([(P[a][E] == y[E]).mean() for a in range(N_CAND)])
        if method == "always_bn":
            j, used = 1, 0
        elif method == "uniform_erm":
            qi = np.array(qorder[:budget])
            errs = [(P[a][qi] != y[qi]).sum() for a in range(N_CAND)]
            j, used = int(np.argmin(errs)), len(qi)
        elif method == "portal":
            j, used = _portal(P, y, qorder, E, budget)
        else:
            raise ValueError(method)
        accs.append(acc_E[j])
        orac.append(acc_E.max())
        labels.append(used)
        chose_bn.append(1.0 if j == 1 else 0.0)

    accs = np.array(accs)
    return {
        "deployed_acc": float(accs.mean()),
        "oracle_gap": float(np.mean(orac) - accs.mean()),
        "labels_used": float(np.mean(labels)),
        "frac_bn": float(np.mean(chose_bn)),
        "p10_batch_acc": float(np.percentile(accs, 10)),
    }
