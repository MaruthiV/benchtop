import numpy as np

# Benchtop lab: raw-error stacking vs student-relative weighting for a ridge student.
#
# Population: X ~ N(0, I_D). Target f(x) = x^T beta. Two linear weak supervisors
# g_j(x) = x^T b_j with b_j = beta + a_j. Both raw error norms equal RHO and their
# raw error inner product is zero. Supervisor 2 puts a fraction `frac` of its squared
# error in coordinate K, which lies outside the student's representation S = span(e_0..e_{K-1}).
#
# Student: ridge regression on the first K features of an unlabeled pool, fit to the
# pseudo-labels of each convex aggregate. Ridge is linear in the labels, so the student
# coefficient is c = L b_w with L = embed(M^{-1} Z^T X_u) and M = Z^T Z + lam I.
#
# Metric: the population excess risk is ||c[:K] - beta[:K]||^2 (the representation floor
# ||Q beta||^2 is removed). It is computed exactly for the fitted student on fresh x, so no
# training point is scored. The primary metric is excess(method weights) / excess(oracle).
#
# Methods (both see the same anchors, which are clean labels):
#   raw_error_stacking: convex weights minimizing the anchor-estimated raw error Gram
#                       (forecast combination). It ignores the student.
#   student_relative:   convex weights minimizing the student's excess objective, using L
#                       from the unlabeled pool and the target projection beta[:K] estimated
#                       by OLS on the anchors.
#   oracle (reference): the best convex weight with the true beta.
#
# Common random numbers: task, unlabeled, and anchor draws come from separate streams and do
# not depend on method, frac, lam, or n_unl. The unlabeled pool is nested (first n rows used).
# There are no trained initializations. All convex 2-supervisor problems are 1-D quadratics
# and are solved exactly.

D = 10                 # input dimension; X ~ N(0, I_D)
K = 5                  # student representation S = span(e_0..e_{K-1}); e_K.. are invisible
N_POOL = 1000          # unlabeled pool; a run with n uses the first n rows
N_ANCHOR = 60          # clean labeled anchors, shared by both methods
RHO = 1.0              # raw error norm of each supervisor
ANCHOR_NOISE_SD = 0.5  # anchor label noise
_FLOOR = 1e-10
_TINY = 1e-12
_METHODS = ("raw_error_stacking", "student_relative")


def _clip01(t):
    return float(min(1.0, max(0.0, t)))


def _best_convex_t(u0, du):
    """Exact argmin over t in [0, 1] of ||u0 + t*du||^2."""
    den = float(du @ du)
    if den < _TINY:
        return 0.5
    return _clip01(-float(u0 @ du) / den)


def run(config, seed):
    method = str(config["method"])
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}")
    frac = float(config["student_invisible_error_fraction"])
    lam = float(config["ridge_regularization"])
    n_unl = int(config["unlabeled_sample_size"])
    if not 0.0 <= frac <= 1.0:
        raise ValueError("student_invisible_error_fraction must lie in [0, 1]")
    if lam <= 0.0 or not 1 <= n_unl <= N_POOL:
        raise ValueError("ridge_regularization must be > 0 and unlabeled_sample_size in [1, N_POOL]")

    # Independent streams. None of them depend on method, frac, lam, or n_unl.
    ss = np.random.SeedSequence(int(seed)).spawn(3)
    rng_task = np.random.default_rng(ss[0])
    rng_unl = np.random.default_rng(ss[1])
    rng_anc = np.random.default_rng(ss[2])

    beta = rng_task.standard_normal(D)
    sign = 1.0 if rng_task.random() < 0.5 else -1.0
    X_pool = rng_unl.standard_normal((N_POOL, D))
    X_anc = rng_anc.standard_normal((N_ANCHOR, D))
    noise = ANCHOR_NOISE_SD * rng_anc.standard_normal(N_ANCHOR)
    y_anc = X_anc @ beta + noise

    # Supervisor errors. Raw Gram is diag(RHO^2, RHO^2) with zero cross term.
    a1 = np.zeros(D)
    a1[0] = RHO                                   # fully student-visible error
    a2 = np.zeros(D)
    a2[1] = RHO * np.sqrt(1.0 - frac)             # visible part
    a2[K] = sign * RHO * np.sqrt(frac)            # student-invisible part
    b1 = beta + a1
    b2 = beta + a2

    # Student operator from the unlabeled pool: ridge coefficient c = L @ b for labels X_u @ b.
    Xu = X_pool[:n_unl]
    Z = Xu[:, :K]
    M = Z.T @ Z + lam * np.eye(K)
    L = np.zeros((D, D))
    L[:K, :] = np.linalg.solve(M, Z.T @ Xu)

    # Aggregate b_t = t*b1 + (1-t)*b2, student excess(t) = ||u0 + t*du||^2 (exact).
    u0 = (L @ b2 - beta)[:K]
    du = (L @ (b1 - b2))[:K]

    def excess_of_t(t):
        v = u0 + t * du
        return float(v @ v)

    t_oracle = _best_convex_t(u0, du)
    ex_oracle = excess_of_t(t_oracle)
    ex_clean = float(np.sum(((L @ beta) - beta)[:K] ** 2))  # clean-label student excess

    if method == "raw_error_stacking":
        r1 = y_anc - X_anc @ b1
        r2 = y_anc - X_anc @ b2
        S11 = float(r1 @ r1) / N_ANCHOR
        S22 = float(r2 @ r2) / N_ANCHOR
        S12 = float(r1 @ r2) / N_ANCHOR
        den = S11 - 2.0 * S12 + S22
        t = 0.5 if den < _TINY else _clip01((S22 - S12) / den)
    else:
        beta_hat = np.linalg.solve(X_anc.T @ X_anc + 1e-8 * np.eye(D), X_anc.T @ y_anc)
        u0_hat = (L @ b2 - beta_hat)[:K]
        t = _best_convex_t(u0_hat, du)

    ex_method = excess_of_t(t)
    ratio = (ex_method + _FLOOR) / (ex_oracle + _FLOOR)

    return {
        "relative_excess_risk_ratio": float(ratio),
        "excess_method": float(ex_method),
        "excess_oracle": float(ex_oracle),
        "excess_clean_student": float(ex_clean),
        "weight_on_sup1": float(t),
    }
