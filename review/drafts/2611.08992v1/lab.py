import math
import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor

# Module-level constants only (no randomness).
N_POOL = 1000
N_TEST = 2000
HIDDEN = (32,)
MAX_ITER = 150
BATCH = 100
LR = 0.01
SCALE_LO = 0.5
SCALE_HI = 2.0
N_GRID = 7
SCALE_GRID = np.logspace(math.log10(SCALE_LO), math.log10(SCALE_HI), N_GRID)
SHIFTS = {"in_dist": 1.0, "ood_100x": 100.0}
METHODS = ("raw", "scale_avg", "invariant")
INIT_SALT = 7919


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _target(u):
    # Depends only on direction of the input, so it is invariant to positive rescaling.
    theta = np.arctan2(u[:, 1], u[:, 0])
    return np.sin(2.0 * theta) + 0.5 * np.cos(3.0 * theta)


def _mlp(random_state):
    # Identical hyperparameters and fixed step budget for every predictor.
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation="tanh",
        solver="adam",
        learning_rate_init=LR,
        batch_size=BATCH,
        max_iter=MAX_ITER,
        shuffle=True,
        tol=0.0,
        n_iter_no_change=10**9,
        early_stopping=False,
        random_state=random_state,
    )


def _mse(pred, y):
    return float(np.mean((pred - y) ** 2))


def run(config, seed):
    method = str(config["method"])
    n = int(config["n_train"])
    shift = str(config["shift"])
    noise = float(config["noise"])
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if shift not in SHIFTS:
        raise ValueError(f"unknown shift: {shift}")
    if not (2 <= n <= N_POOL):
        raise ValueError(f"n_train out of range: {n}")

    # Data stream: fixed draw order, independent of config, so every arm
    # sees the same pool, noise draws, and test points for a given seed.
    rng_data = np.random.default_rng(seed)
    g_tr = rng_data.standard_normal((N_POOL, 2))
    z_tr = rng_data.uniform(0.0, 1.0, N_POOL)
    eps_tr = rng_data.standard_normal(N_POOL)
    g_te = rng_data.standard_normal((N_TEST, 2))
    z_te = rng_data.uniform(0.0, 1.0, N_TEST)

    # Log-uniform scales in [SCALE_LO, SCALE_HI]; training subsets are prefixes
    # of the same pool, so n_train levels share data.
    scale_tr = SCALE_LO * (SCALE_HI / SCALE_LO) ** z_tr
    x_tr = (scale_tr[:, None] * g_tr)[:n]
    u_tr = _unit(x_tr)
    y_tr = _target(u_tr) + noise * eps_tr[:n]

    scale_te = SCALE_LO * (SCALE_HI / SCALE_LO) ** z_te
    y_te = _target(_unit(g_te))  # clean target for held-out scoring

    # Model-init stream: separate from data, fixed per seed.
    rng_init = np.random.default_rng([seed, INIT_SALT])
    seed_raw, seed_inv = (int(v) for v in rng_init.integers(0, 2**31 - 1, size=2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = _mlp(seed_raw).fit(x_tr, y_tr)
        inv = _mlp(seed_inv).fit(u_tr, y_tr)

    def predict(name, x):
        if name == "raw":
            return raw.predict(x)
        if name == "scale_avg":
            # Post-hoc average of the raw model over a normalized log-uniform
            # grid of scales spanning the training interval.
            return np.mean([raw.predict(c * x) for c in SCALE_GRID], axis=0)
        return inv.predict(_unit(x))

    # Score every predictor on both shift levels; select by config afterward.
    scores = {}
    for sname, mult in SHIFTS.items():
        x_te = mult * (scale_te[:, None] * g_te)
        for name in METHODS:
            scores[(name, sname)] = _mse(predict(name, x_te), y_te)

    test_mse = scores[(method, shift)]
    raw_mse = scores[("raw", shift)]
    return {
        "test_mse": test_mse,
        "raw_test_mse": raw_mse,
        "invariant_test_mse": scores[("invariant", shift)],
        "in_dist_test_mse": scores[(method, "in_dist")],
        "gain_over_raw": raw_mse - test_mse,
    }
