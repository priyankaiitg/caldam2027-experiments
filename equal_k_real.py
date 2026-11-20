"""Real data: the hierarchy with threshold delta = t (|T| - 1), and flat k-means with as many clusters as the
hierarchy returned. Reads the count matrices under ../../src. Writes equal_k_real_results.txt.
The runs with t = inf replay the random stream of real_data.py, so they are the depth-3 hierarchies of Table 1.
Everything else uses a second stream. k-means++ updates the minimum distance incrementally and Lloyd's algorithm
computes distances in blocks of centres; both give the same values as real_data.py and only bound the memory."""
import numpy as np

SRC = "../../src/cikm/src/"
K = 10
REPS = 10
rng = np.random.default_rng(2027)   # the stream of real_data.py
rng2 = np.random.default_rng(2028)  # everything else


def load():
    esm = np.loadtxt(SRC + "knome/monthlyCounts10.csv")[:, 1:]
    se = np.loadtxt(SRC + "stackexchange/0/monthlyCounts10.csv", delimiter=",")[:, 2:]
    lkml = np.loadtxt(SRC + "lkml/recursive/0/monthlyCounts10_2001.csv", delimiter=",")
    return {"ESM": esm, "StackExchange": se, "LKML": lkml}


def cost(X, lab):
    return sum(((X[lab == j] - X[lab == j].mean(0)) ** 2).sum() for j in np.unique(lab))


def kmeanspp(X, k, g):
    C = [X[g.integers(len(X))]]
    d = ((X - C[0]) ** 2).sum(1)
    for _ in range(k - 1):
        c = X[g.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[g.integers(len(X))]
        C.append(c)
        d = np.minimum(d, ((X - c) ** 2).sum(1))
    return np.array(C)


def dist(X, C, block=16):
    return np.concatenate([((X[:, None] - C[None, j:j + block]) ** 2).sum(-1) for j in range(0, len(C), block)], 1)


def lloyd(X, k, restarts, g):
    best = None
    for _ in range(restarts):
        C = kmeanspp(X, k, g)
        for _ in range(200):
            a = dist(X, C).argmin(1)
            C2 = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
            if np.allclose(C2, C):
                break
            C = C2
        c = cost(X, a)
        if best is None or c < best[0]:
            best = (c, a)
    return best[1]


def diversity(X):
    """Weight of a minimum spanning tree over cosine similarities (Prim), as in run_experiments.py."""
    m = len(X)
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    W = U @ U.T
    in_tree = np.zeros(m, bool)
    in_tree[0] = True
    best = W[0].copy()
    total = 0.0
    for _ in range(m - 1):
        b = np.where(in_tree, np.inf, best)
        j = b.argmin()
        total += b[j]
        in_tree[j] = True
        best = np.minimum(best, W[j])
    return total


def hierarchy(X, idx, depth, out, t, g, max_depth=3):
    """Algorithm 1 with at most three levels. Clusters with at most K users are not split. The root uses 10
    restarts, deeper calls use 5. For t = inf this draws random numbers exactly as real_data.py does."""
    if len(idx) <= K or depth == max_depth:
        out.append(idx)
        return
    a = lloyd(X[idx], K, 10 if depth == 0 else 5, g)
    parts = [idx[a == j] for j in np.unique(a)]
    if len(parts) == 1:
        out.append(idx)
        return
    for p in parts:
        if len(p) > K and (t == np.inf or diversity(X[p]) <= t * (len(p) - 1)):
            hierarchy(X, p, depth + 1, out, t, g, max_depth)
        else:
            out.append(p)


def summary(v):
    return f"mean={np.mean(v):.2f} min={np.min(v):.2f} max={np.max(v):.2f}"


lines = []
for name, X in load().items():
    n = len(X)
    for t in (np.inf, 0.0, 0.1, 0.25):
        hc, hl, hs, fl, fs = [], [], [], [], []
        for _ in range(REPS):
            out = []
            hierarchy(X, np.arange(n), 0, out, t, rng if t == np.inf else rng2)
            sz = np.array(list(map(len, out)))
            f = np.bincount(lloyd(X, len(sz), 5, rng2))
            f = f[f > 0]
            hc.append(len(sz)); hl.append(sz.max() / n); hs.append((sz == 1).sum())
            fl.append(f.max() / n); fs.append((f == 1).sum())
        lines.append(f"{name} t={t}: hierarchy clusters {summary(hc)}; largest {summary(hl)}; singletons {summary(hs)}")
        lines.append(f"{name} t={t}: flat k-means, same k: largest {summary(fl)}; singletons {summary(fs)}")
        print(lines[-2]); print(lines[-1], flush=True)

open("equal_k_real_results.txt", "w").write("\n".join(lines) + "\n")
