import itertools
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# variable layout: 0-2 cheap informative, 3 expensive informative, 4-5 noise
P = 6
EXP = 3
CHEAP_COEF = np.array([0.7, 0.55, 0.4])
CHEAP_HARM = 0.001
INTERCEPT = -1.3
N_TRAIN = 1000
N_TEST = 4000


def _nb(y, p, pt):
    pred = p >= pt
    n = len(y)
    tp = np.sum(pred & (y == 1)) / n
    fp = np.sum(pred & (y == 0)) / n
    return tp - fp * pt / (1 - pt)


def _fit_predict(Xtr, ytr, Xte, cols):
    if len(cols) == 0:
        return np.full(len(Xte), ytr.mean())
    m = LogisticRegression(C=1e4, max_iter=200)
    m.fit(Xtr[:, cols], ytr)
    return m.predict_proba(Xte[:, cols])[:, 1]


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "cost_nb")
    harm_exp = float(config.get("expensive_harm", 0.02))
    beta = float(config.get("expensive_coef", 0.2))
    pt = float(config.get("pt", 0.2))

    # data rng (independent of method)
    rng = np.random.default_rng(seed)
    coef = np.concatenate([CHEAP_COEF, [beta], [0.0, 0.0]])
    X = rng.standard_normal((N_TRAIN + N_TEST, P))
    logit = INTERCEPT + X @ coef
    y = (rng.random(N_TRAIN + N_TEST) < 1 / (1 + np.exp(-logit))).astype(int)
    Xtr, ytr = X[:N_TRAIN], y[:N_TRAIN]
    Xte, yte = X[N_TRAIN:], y[N_TRAIN:]

    harms = np.full(P, CHEAP_HARM)
    harms[EXP] = harm_exp

    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr))
    cache = {}

    def cv(cols):
        key = tuple(cols)
        if key in cache:
            return cache[key]
        oof = np.zeros(N_TRAIN)
        for tr, va in folds:
            oof[va] = _fit_predict(Xtr[tr], ytr[tr], Xtr[va], list(cols))
        pc = np.clip(oof, 1e-6, 1 - 1e-6)
        ll = -np.mean(ytr * np.log(pc) + (1 - ytr) * np.log(1 - pc))
        nb = _nb(ytr, oof, pt)
        cache[key] = (nb, ll)
        return cache[key]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "cost_nb":
            best, best_val = (), cv(())[0]
            for k in range(1, P + 1):
                for cols in itertools.combinations(range(P), k):
                    val = cv(cols)[0] - harms[list(cols)].sum()
                    if val > best_val + 1e-12:
                        best, best_val = cols, val
            sel = list(best)
        elif method == "backward_elim":
            cur = list(range(P))
            cur_ll = cv(tuple(cur))[1]
            while cur:
                cands = []
                for v in cur:
                    c = tuple(x for x in cur if x != v)
                    cands.append((cv(c)[1], v, c))
                ll, v, c = min(cands)
                if ll <= cur_ll:
                    cur, cur_ll = list(c), ll
                else:
                    break
            sel = cur
        elif method == "lasso":
            m = LogisticRegressionCV(Cs=10, cv=folds, penalty="l1", solver="liblinear",
                                     scoring="neg_log_loss", max_iter=200)
            m.fit(Xtr, ytr)
            sel = [int(j) for j in np.flatnonzero(np.abs(m.coef_[0]) > 1e-8)]
        else:
            raise ValueError(method)

        p_te = _fit_predict(Xtr, ytr, Xte, sel)

    nb_raw = _nb(yte, p_te, pt)
    total_harm = float(harms[sel].sum()) if sel else 0.0
    auc = roc_auc_score(yte, p_te) if len(sel) > 0 else 0.5
    return {
        "adj_net_benefit": float(nb_raw - total_harm),
        "raw_net_benefit": float(nb_raw),
        "n_selected": float(len(sel)),
        "expensive_selected": float(EXP in sel),
        "auc": float(auc),
    }
