import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_D = load_digits()
_X = _D.data.astype(np.float64)
_Y = _D.target.astype(np.int64)
EMB_DIM = 32
STEPS_PER_TASK = 80
REPLAY_PER_TASK = 20


def _make_tasks(Z, y, T, grouping):
    # order classes along first principal axis of class means
    means = np.stack([Z[y == c].mean(0) for c in range(10)])
    c0 = means - means.mean(0)
    u = np.linalg.svd(c0, full_matrices=False)[2][0]
    order = list(np.argsort(c0 @ u))
    if grouping == "clustered":
        tasks = [list(map(int, a)) for a in np.array_split(order, T)]
    else:
        tasks = [[int(order[i]) for i in range(T_i, 10, T)] for T_i in range(T)]
    return tasks


def _fit_adapter(Xt, yt):
    cls = np.unique(yt)
    if len(cls) == 1:
        return ("const", int(cls[0]))
    m = LogisticRegression(C=1.0, max_iter=200)
    m.fit(Xt, yt)
    return ("lr", m)


def _adapter_predict(ad, X):
    if ad[0] == "const":
        return np.full(len(X), ad[1])
    return ad[1].predict(X)


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "prototype_routing")
    T = int(config.get("task_count", 5))
    grouping = config.get("task_grouping", "clustered")
    cs = int(config.get("chunk_size", 32))

    # ---- data (independent of method) ----
    Xtr, Xte, ytr, yte = train_test_split(_X, _Y, test_size=0.4, random_state=seed, stratify=_Y)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=EMB_DIM, random_state=seed).fit(sc.transform(Xtr))
    Ztr = pca.transform(sc.transform(Xtr))
    Zte = pca.transform(sc.transform(Xte))
    tasks = _make_tasks(Ztr, ytr, T, grouping)
    tr_idx = [np.where(np.isin(ytr, t))[0] for t in tasks]
    te_idx = [np.where(np.isin(yte, t))[0] for t in tasks]

    drng = np.random.default_rng(seed + 1000)
    replay_sel = [drng.permutation(ix)[:REPLAY_PER_TASK] for ix in tr_idx]
    brng = np.random.default_rng(seed + 2000)
    chunks = []  # (task, indices)
    for k, ix in enumerate(te_idx):
        p = brng.permutation(ix)
        for s in range(0, len(p), cs):
            chunks.append((k, p[s:s + cs]))

    per_task_correct = np.zeros(T)
    per_task_n = np.zeros(T)
    route_ok = 0

    if method == "shared_replay":
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(EMB_DIM, 64), nn.ReLU(), nn.Linear(64, 10))
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        lossf = nn.CrossEntropyLoss()
        Zt = torch.tensor(Ztr, dtype=torch.float32)
        yt = torch.tensor(ytr)
        for k in range(T):
            ix = list(tr_idx[k])
            for j in range(k):
                ix += list(replay_sel[j])
            ix = np.array(ix)
            xb, yb = Zt[ix], yt[ix]
            for _ in range(STEPS_PER_TASK):
                opt.zero_grad()
                lossf(net(xb), yb).backward()
                opt.step()
        with torch.no_grad():
            pred = net(torch.tensor(Zte, dtype=torch.float32)).argmax(1).numpy()
        for k, ix in chunks:
            per_task_correct[k] += (pred[ix] == yte[ix]).sum()
            per_task_n[k] += len(ix)
        route_acc = float("nan")
    else:
        adapters = [_fit_adapter(Ztr[ix], ytr[ix]) for ix in tr_idx]
        protos = np.stack([Ztr[ix].mean(0) for ix in tr_idx])
        protos_n = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
        for k, ix in chunks:
            if method == "oracle_routing":
                r = k
            else:
                e = Zte[ix].mean(0)
                e = e / (np.linalg.norm(e) + 1e-12)
                r = int(np.argmax(protos_n @ e))
            route_ok += int(r == k)
            pr = _adapter_predict(adapters[r], Zte[ix])
            per_task_correct[k] += (pr == yte[ix]).sum()
            per_task_n[k] += len(ix)
        route_acc = route_ok / len(chunks)

    acc_t = per_task_correct / np.maximum(per_task_n, 1)
    return {
        "final_average_accuracy": float(acc_t.mean()),
        "routing_acc": float(route_acc) if route_acc == route_acc else -1.0,
        "first_task_acc": float(acc_t[0]),
        "last_task_acc": float(acc_t[-1]),
        "worst_task_acc": float(acc_t.min()),
    }
