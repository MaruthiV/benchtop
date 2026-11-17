"""Benchtop lab: lower-star filtration (LSF) vs. alternative filtrations (LpF, SMF)
for persistence-based classification of two annulus populations.

Pipeline per (config, seed):
  1. Sample two classes of 2-D point clouds: annuli with inner radii 0.5 and 0.8
     (outer radius 1), plus Gaussian jitter. Train and test clouds are drawn fresh.
  2. For each cloud, build a fixed complex: symmetric 6-NN graph (1-skeleton) plus
     its clique 2-simplices when max_dim == 2. The complex depends only on the
     points, so it is identical across methods.
  3. Assign filtration values to simplices with the chosen rule. f(v) is the
     distance from vertex v to the cloud centroid.
       LSF : max_v f(v)                            (maximum-vertex rule, baseline)
       LpF : (sum_v f(v)^p)^(1/p), p = 2           (L_p aggregate over all vertices)
       SMF : T * log(sum_v exp(f(v)/T)), T = 0.1   (softmax / smooth max over all vertices)
     All three are monotone (a face never enters after a coface), and on vertices
     all three reduce to f(v). The exact paper formulas are not in the excerpt, so
     these are faithful small-scale readings of the descriptions.
  4. Compute H1 persistence by standard GF(2) column reduction over the sorted
     filtration. Essential H1 classes die at the end of their own filtration.
  5. Summarise H1 with three features (total lifetime, count of lifetimes above a
     threshold, max lifetime), standardise them on training clouds, and classify
     with nearest centroid. The primary metric is accuracy on fresh test clouds.
"""
import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Fixed constants (module level, no randomness)
K_NEIGHBORS = 6
N_TRAIN_PER_CLASS = 25
N_TEST_PER_CLASS = 50
LP_P = 2.0
SMF_TEMP = 0.1
LIFETIME_THRESHOLD = 0.05
ANNULI = ((0.5, 1.0), (0.8, 1.0))  # (inner, outer) radius for class 0 and class 1
METHODS = ("LSF", "LpF", "SMF")


def _sample_clouds(rng, n_points, noise):
    """Draw train and test clouds in a fixed order. Every random draw happens
    regardless of method or noise level, so the data are common random numbers."""

    def draw(count_per_class):
        clouds, labels = [], []
        for cls, (r_in, r_out) in enumerate(ANNULI):
            for _ in range(count_per_class):
                theta = rng.uniform(0.0, 2.0 * math.pi, n_points)
                radius = np.sqrt(rng.uniform(r_in ** 2, r_out ** 2, n_points))
                jitter = rng.standard_normal((n_points, 2))
                pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
                clouds.append(pts + noise * jitter)
                labels.append(cls)
        return clouds, np.array(labels)

    tr_X, tr_y = draw(N_TRAIN_PER_CLASS)
    te_X, te_y = draw(N_TEST_PER_CLASS)
    return tr_X, tr_y, te_X, te_y


def _complex(pts, max_dim):
    """Fixed complex from the point geometry: symmetric kNN graph and, if
    max_dim >= 2, its 3-cliques as triangles. Vertices are 0..n-1."""
    n = pts.shape[0]
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    k = min(K_NEIGHBORS, n - 1)
    nn = np.argsort(D, axis=1, kind="stable")[:, :k]
    edge_set = set()
    for i in range(n):
        for j in nn[i]:
            j = int(j)
            edge_set.add((i, j) if i < j else (j, i))
    adj = [set() for _ in range(n)]
    for a, b in edge_set:
        adj[a].add(b)
        adj[b].add(a)
    edges = sorted(edge_set)
    tris = []
    if max_dim >= 2:
        for a, b in edges:
            for c in adj[a] & adj[b]:
                if c > b:
                    tris.append((a, b, c))
        tris.sort()
    return edges, tris


def _aggregate(method, f):
    """Filtration value of a simplex from its vertex values f (list of floats)."""
    if method == "LSF":
        return max(f)
    if method == "LpF":
        return sum(x ** LP_P for x in f) ** (1.0 / LP_P)
    if method == "SMF":
        m = max(f)
        return m + SMF_TEMP * math.log(sum(math.exp((x - m) / SMF_TEMP) for x in f))
    raise ValueError(f"unknown method: {method}")


def _h1_lifetimes(method, vals, edges, tris):
    """H1 lifetimes via standard GF(2) column reduction on the sorted filtration.
    Essential H1 classes die at the end of this filtration."""
    n = len(vals)
    simplices = [(vals[i], 0, (i,)) for i in range(n)]
    for a, b in edges:
        simplices.append((_aggregate(method, [vals[a], vals[b]]), 1, (a, b)))
    for a, b, c in tris:
        simplices.append((_aggregate(method, [vals[a], vals[b], vals[c]]), 2, (a, b, c)))
    # Sort by value, then dimension, so every face precedes its cofaces.
    simplices.sort(key=lambda s: (s[0], s[1], s[2]))
    index = {s[2]: idx for idx, s in enumerate(simplices)}
    N = len(simplices)
    end = simplices[-1][0]

    reduced = [None] * N
    pivot_owner = {}
    for j, (_, dim, verts) in enumerate(simplices):
        if dim == 0:
            reduced[j] = set()
            continue
        col = set()
        for drop in range(len(verts)):
            face = verts[:drop] + verts[drop + 1:]
            col ^= {index[face]}
        while col:
            low = max(col)
            k = pivot_owner.get(low)
            if k is None:
                break
            col ^= reduced[k]
        reduced[j] = col
        if col:
            pivot_owner[max(col)] = j

    creators = set()
    lifetimes = []
    for j in range(N):
        if reduced[j]:
            i = max(reduced[j])
            creators.add(i)
            if simplices[i][1] == 1:  # finite H1 pair (edge born, triangle kills)
                lifetimes.append(max(simplices[j][0] - simplices[i][0], 0.0))

    n_essential = 0
    for i in range(N):
        if simplices[i][1] == 1 and not reduced[i] and i not in creators:
            lifetimes.append(max(end - simplices[i][0], 0.0))
            n_essential += 1
    return lifetimes, n_essential


def _summarise(lifetimes):
    if not lifetimes:
        return [0.0, 0.0, 0.0]
    total = float(sum(lifetimes))
    count = float(sum(1 for L in lifetimes if L > LIFETIME_THRESHOLD))
    return [total, count, float(max(lifetimes))]


def run(config, seed):
    method = str(config["method"])
    n_points = int(config["n_points"])
    noise = float(config["noise"])
    max_dim = int(config["max_dim"])
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")

    # Data RNG. Nearest-centroid has no random initialisation, so this is the only RNG.
    # Any future model init should use np.random.default_rng([seed, 1]).
    rng_data = np.random.default_rng(seed)
    tr_X, tr_y, te_X, te_y = _sample_clouds(rng_data, n_points, noise)

    def featurize(clouds):
        feats = np.zeros((len(clouds), 3))
        n_h1 = np.zeros(len(clouds))
        n_ess = np.zeros(len(clouds))
        for idx, pts in enumerate(clouds):
            edges, tris = _complex(pts, max_dim)
            vals = np.linalg.norm(pts - pts.mean(axis=0), axis=1).tolist()
            lifetimes, ess = _h1_lifetimes(method, vals, edges, tris)
            feats[idx] = _summarise(lifetimes)
            n_h1[idx] = len(lifetimes)
            n_ess[idx] = ess
        return feats, n_h1, n_ess

    Ftr, h_tr, e_tr = featurize(tr_X)
    Fte, h_te, e_te = featurize(te_X)

    # Standardise on train only.
    mu = Ftr.mean(axis=0)
    sd = Ftr.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Ztr = (Ftr - mu) / sd
    Zte = (Fte - mu) / sd

    centroids = np.stack([Ztr[tr_y == c].mean(axis=0) for c in (0, 1)])

    def predict(Z):
        d = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        return d.argmin(axis=1)

    train_acc = float((predict(Ztr) == tr_y).mean())
    test_acc = float((predict(Zte) == te_y).mean())

    all_h1 = np.concatenate([h_tr, h_te])
    all_ess = np.concatenate([e_tr, e_te])
    total_h1 = float(all_h1.sum())
    ess_frac = float(all_ess.sum() / total_h1) if total_h1 > 0 else 0.0

    return {
        "accuracy": test_acc,
        "train_accuracy": train_acc,
        "mean_h1_features": float(all_h1.mean()),
        "essential_h1_fraction": ess_frac,
    }
