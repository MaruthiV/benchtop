import numpy as np

# Miniature of the collective self-reinforcement model (binary problems, K=3 majority vote).
# Each problem i has logit z_i = u_i + lam * v, where v is one shared parameter.
# Training is minibatch REINFORCE on the majority-vote pseudo-reward. No labels are used.
K = 3
ETA = 6.0
T_STEPS = 2500
BATCH = 16
Q_CORRECT = 0.65   # prob. that a problem's initial lean points to the true answer
BETA = 0.75        # marginal-entropy bonus weight (above the theoretical critical value 1/2)
N_TEST = 300


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _make_problems(rng, n, a):
    y = rng.choice(np.array([-1.0, 1.0]), size=n)       # hidden ground-truth answer
    right = rng.random(n) < Q_CORRECT
    lean = np.where(right, y, -y)                        # initial preferred answer (balanced population)
    mag = a * rng.uniform(0.8, 1.2, size=n)              # initial confidence |z|
    return y, lean * mag


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "plain")
    n = int(config.get("N", 100))
    lam_cfg = float(config.get("lam", 0.4))
    a = float(config.get("a", 0.5))

    lam = 0.0 if method == "no_shared" else lam_cfg
    use_bonus = method == "entropy_bonus"

    # Separate rngs: data, model init, and sampling noise. Independent of the method.
    data_rng = np.random.default_rng(seed)
    init_rng = np.random.default_rng([seed, 1])
    samp_rng = np.random.default_rng([seed, 2])

    y, u = _make_problems(data_rng, n, a)
    y_t, u_t = _make_problems(data_rng, N_TEST, a)       # fresh problems for a diagnostic

    v = 0.01 * init_rng.standard_normal()                # tiny shared-mode seed
    u = u.copy()
    B = min(BATCH, n)

    for _ in range(T_STEPS):
        idx = samp_rng.integers(0, n, size=B)
        uni = samp_rng.random((B, K))                    # always drawn, so streams match across arms
        z = u[idx] + lam * v
        p = _sig(z)
        k = (uni < p[:, None]).sum(axis=1)               # number of "+" answers among K samples
        maj_plus = k >= 2
        # REINFORCE with r_j = 1[y_j == majority]; d log pi(+) = 1-p, d log pi(-) = -p
        g = np.where(maj_plus, k * (1.0 - p) / K, -(K - k) * p / K)
        if use_bonus:
            pb = min(max(p.mean(), 1e-3), 1 - 1e-3)
            g = g + BETA * np.log((1.0 - pb) / pb) * p * (1.0 - p)
        u += (ETA / B) * np.bincount(idx, weights=g, minlength=n)
        v += (ETA * lam / B) * g.sum()

    z = u + lam * v
    p = _sig(z)
    # Expected accuracy over fresh samples on the problems; true labels were never used in training.
    accuracy = float(np.mean(_sig(y * z)))
    frac_plus = float(np.mean(z > 0))
    answer_share = max(frac_plus, 1.0 - frac_plus)
    self_reward = float(np.mean(p ** 2 + (1 - p) ** 2))
    fresh_acc = float(np.mean(_sig(y_t * (u_t + lam * v))))
    return {
        "accuracy": accuracy,
        "answer_share": float(answer_share),
        "self_reward": self_reward,
        "shared_shift": float(abs(lam * v)),
        "fresh_problem_accuracy": fresh_acc,
    }
