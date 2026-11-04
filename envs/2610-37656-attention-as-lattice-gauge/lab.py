import numpy as np
import itertools
from scipy.stats import rankdata

N_ENT = 30
BETA = 8.0


def _build_group(name):
    if name == "Z4":
        d = 4
        mats = [np.roll(np.eye(d), s, axis=0) for s in range(4)]
    else:
        d = 3 if name == "S3" else 4
        mats = []
        for perm in itertools.permutations(range(d)):
            P = np.zeros((d, d))
            for a, b in enumerate(perm):
                P[b, a] = 1.0
            mats.append(P)
    return np.array(mats), d


def _snap(M, G):
    sc = np.einsum('...ab,mab->...m', M, G)
    return np.argmax(sc, axis=-1)


def _auc(pos_score, labels):
    # AUC with positive class = labels True; higher score -> more positive
    labels = labels.astype(bool)
    npos = labels.sum()
    nneg = (~labels).sum()
    if npos == 0 or nneg == 0:
        return 0.5
    r = rankdata(pos_score)
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def run(config, seed):
    method = config.get("method", "holonomy")
    c = float(config.get("corruption", 0.2))
    p = float(config.get("edge_prob", 0.3))
    gname = config.get("group", "S4")
    n = N_ENT

    G, d = _build_group(gname)
    m = len(G)
    abelian = (gname == "Z4")
    I = np.eye(d)

    # ---- data (independent rng; unaffected by method) ----
    rng = np.random.default_rng(seed)
    gi = rng.integers(0, m, size=n)
    Aup = np.triu(rng.random((n, n)) < p, k=1)
    A = Aup | Aup.T
    corr_draw = rng.random((n, n)) < c
    rand_draw = rng.integers(0, m - 1, size=(n, n))

    Ttrue = np.einsum('iab,jcb->ijac', G[gi], G[gi])
    tidx = _snap(Ttrue, G)
    newidx = rand_draw + (rand_draw >= tidx)
    Cup = Aup & corr_draw
    Tup = np.where(Cup[..., None, None], G[newidx], Ttrue)
    Tfull = np.where(Aup[..., None, None], Tup, 0.0)
    Tfull = Tfull + np.where(Aup.T[..., None, None], np.transpose(Tup, (1, 0, 3, 2)), 0.0)
    T = np.where(A[..., None, None], Tfull, I)
    Csym = Cup | Cup.T

    # ---- triangles ----
    ii = np.arange(n)
    order = (ii[:, None, None] < ii[None, :, None]) & (ii[None, :, None] < ii[None, None, :])
    tri = A[:, :, None] & A[None, :, :] & A[:, None, :] & order
    # tri[i,j,k] = A[i,j] & A[j,k] & A[i,k]
    tri = (A[:, :, None] & A[None, :, :].transpose(0, 1, 2) * 0 == 0) if False else (
        A[:, :, None] & A[:, None, :] & A[None, :, :]) & order
    P = np.einsum('ijab,jkbc->ijkac', T, T)
    trH = np.einsum('ijkac,kica->ijk', P, T) / d
    lab = (Csym[:, :, None] | Csym[:, None, :] | Csym[None, :, :])
    labels = lab[tri]
    s_hol = trH[tri]
    ntri = int(tri.sum())

    sgn = np.round(np.linalg.det(T))
    if abelian:
        s_com = s_hol
    else:
        dh = sgn[:, :, None] * sgn[None, :, :] * sgn[:, None, :]
        s_com = dh[tri]
    auc_hol = _auc(-s_hol, labels)
    auc_com = _auc(-s_com, labels)

    # ---- unobserved relation recovery (one round of composition) ----
    iu, ju = np.nonzero(np.triu(~A, k=1))
    if len(iu) > 0:
        # candidates C[i,j,k] = T_ik T_kj
        Cc = np.einsum('ikab,kjbc->ijkac', T, T)
        V = A[:, :, None] & A.T[None, :, :].transpose(0, 1, 2) if False else (
            A[:, None, :] & A.T[None, :, :]) if False else None
        V = A[:, None, :] & A.T[:, :, None].transpose(1, 0, 2) if False else None
        # V[i,j,k] = A[i,k] & A[k,j]
        V = A[:, None, :] & A.T[None, :, :]
        V = A[:, None, :] & A[None, :, :].transpose(0, 2, 1) if False else V
        V = np.einsum('ik,kj->ijk', A.astype(float), A.astype(float)) * 0 + (
            A[:, None, :] & A.T[None, :, :]) if False else (A[:, None, :] & A[:, :, None].transpose(1, 0, 2).transpose(0, 1, 2) if False else None)
        V = (A[:, None, :] & A.T[:, :, None].transpose(0, 1, 2).transpose(0, 1, 2) if False else None)
        V = A[:, None, :] & A[None, :, :].transpose(1, 0, 2) if False else None
        # explicit: V[i,j,k] = A[i,k] and A[k,j]
        V = A[:, None, :] & np.swapaxes(A, 0, 1)[None, :, :] if False else np.einsum(
            'ik,kj->ijk', A.astype(int), A.astype(int)) > 0
        Vs = V[iu, ju]                      # (q, n)
        Cs = Cc[iu, ju]                     # (q, n, d, d)
        truth = tidx_pair = _snap(Ttrue[iu, ju], G)
        Cf = Cs.reshape(len(iu), n, d * d)
        if method == "holonomy" or abelian:
            gram = np.einsum('qka,qla->qkl', Cf, Cf) / d
            Vf = Vs.astype(float)
            pair = Vf[:, :, None] * Vf[:, None, :] * (1 - np.eye(n))[None]
            cnt = pair.sum(axis=2)
            s = (gram * pair).sum(axis=2) / np.maximum(cnt, 1.0)
            logit = np.where(Vs, BETA * s, -np.inf)
            mx = np.max(np.where(Vs, logit, -1e9), axis=1, keepdims=True)
            w = np.where(Vs, np.exp(logit - mx), 0.0)
            wsum = w.sum(axis=1, keepdims=True)
            alpha = w / np.maximum(wsum, 1e-12)
            agg = np.einsum('qk,qkab->qab', alpha, Cs)
            pred = _snap(agg, G)
        else:
            dets = np.round(np.linalg.det(Cs))
            vote = (dets * Vs).sum(axis=1)
            want = np.where(vote >= 0, 1.0, -1.0)
            gdet = np.round(np.linalg.det(G))
            pred = np.array([int(np.argmax(gdet == w_)) for w_ in want])
        has = Vs.any(axis=1)
        correct = np.where(has, (pred == truth).astype(float), 1.0 / m)
        rec = float(correct.mean())
    else:
        rec = 0.0

    auc = auc_hol if method == "holonomy" else auc_com
    return {
        "auc": float(auc),
        "auc_holonomy": float(auc_hol),
        "auc_commuting": float(auc_com),
        "recovery_acc": float(rec),
        "n_triangles": float(ntri),
    }
