import math

import numpy as np
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Module-level constants only (no randomness).
_DIAB = load_diabetes()
_DIAB_X = np.asarray(_DIAB.data[:, :5], dtype=float)  # first 5 of 10 columns
_DIAB_Y = np.asarray(_DIAB.target, dtype=float)
_DIAB_Y_MU = float(_DIAB_Y.mean())
_DIAB_Y_SD = float(_DIAB_Y.std())

DEGREES = tuple(range(1, 8))
N_FEATURES = 5
RIDGE_ALPHA = 1.0
N_FOLDS = 5
TEST_FRAC = 0.2
# method -> accuracy weight w (None means accuracy-only selection)
METHOD_WEIGHTS = {"accuracy_only": None, "sqs_w0.25": 0.25, "sqs_w0.75": 0.75}


def _make_split(config, seed):
    # One data stream per seed. It is consumed in a fixed order that does not
    # depend on method, so all method arms see identical data and splits.
    rng = np.random.default_rng([int(seed), 0])
    n = int(config["n_samples"])
    noise = float(config["noise"])
    if config["dataset"] == "load_diabetes":
        if n > len(_DIAB_Y):
            raise ValueError("load_diabetes has only 442 rows")
        idx = rng.permutation(len(_DIAB_Y))[:n]
        X = _DIAB_X[idx]
        y = (_DIAB_Y[idx] - _DIAB_Y_MU) / _DIAB_Y_SD  # unit-sd target
    elif config["dataset"] == "make_friedman1":
        X, y = make_friedman1(
            n_samples=n, n_features=N_FEATURES, noise=0.0, random_state=int(seed)
        )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
    else:
        raise ValueError(f"unknown dataset: {config['dataset']}")
    y = y + noise * rng.standard_normal(n)  # additive Gaussian target noise
    perm = rng.permutation(n)
    n_test = int(round(TEST_FRAC * n))
    te, tr = perm[:n_test], perm[n_test:]
    return X[tr], y[tr], X[te], y[te]


def _pipe(degree):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        Ridge(alpha=RIDGE_ALPHA),
    )


def _cv_rmse(X, y, degree, splits):
    errs = []
    for tr, va in splits:
        m = _pipe(degree).fit(X[tr], y[tr])
        r = m.predict(X[va]) - y[va]
        errs.append(np.sqrt(np.mean(r * r)))
    return float(np.mean(errs))


def _test_rmse(Xtr, ytr, Xte, yte, degree):
    m = _pipe(degree).fit(Xtr, ytr)
    r = m.predict(Xte) - yte
    return float(np.sqrt(np.mean(r * r)))


def _minmax(v):
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0.0:
        return np.ones_like(v)
    return (v - lo) / span


def run(config: dict, seed: int) -> dict:
    method = config["method"]
    if method not in METHOD_WEIGHTS:
        raise ValueError(f"unknown method: {method}")
    w = METHOD_WEIGHTS[method]

    Xtr, ytr, Xte, yte = _make_split(config, seed)
    p = Xtr.shape[1]
    splits = list(
        KFold(n_splits=N_FOLDS, shuffle=True, random_state=int(seed)).split(Xtr)
    )

    cv = np.array([_cv_rmse(Xtr, ytr, d, splits) for d in DEGREES])
    # PolynomialFeatures(include_bias=False) yields comb(p+d, d)-1 terms;
    # adding the Ridge intercept gives comb(p+d, d) fitted coefficients.
    params = np.array([math.comb(p + d, d) for d in DEGREES], dtype=float)

    acc_idx = int(np.argmin(cv))
    if w is None:
        sel = acc_idx
    else:
        A = _minmax(-cv)       # normalized accuracy (1 - RMSE), min-max
        C = _minmax(-params)   # normalized simplicity (-params), min-max
        score = np.power(A, w) * np.power(C, 1.0 - w)
        sel = int(np.argmax(score))  # ties go to the lower degree

    rmse_sel = _test_rmse(Xtr, ytr, Xte, yte, DEGREES[sel])
    rmse_acc = rmse_sel if sel == acc_idx else _test_rmse(
        Xtr, ytr, Xte, yte, DEGREES[acc_idx]
    )
    regret = (rmse_sel - rmse_acc) / rmse_acc
    param_ratio = params[sel] / params[acc_idx]

    return {
        "rmse_regret": float(regret),
        "param_ratio": float(param_ratio),
        "selected_degree": float(DEGREES[sel]),
        "test_rmse_selected": float(rmse_sel),
        "test_rmse_acc_only": float(rmse_acc),
    }
