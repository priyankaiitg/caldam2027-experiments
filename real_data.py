"""Real-data statistics for the CALDAM 2027 draft. Reads the count matrices under ../../src. Writes real_results.txt."""
import numpy as np

SRC = "../../src/cikm/src/"
K = 10
REPS = 10
rng = np.random.default_rng(2027)


def load():
    esm = np.loadtxt(SRC + "knome/monthlyCounts10.csv")[:, 1:]
    se = np.loadtxt(SRC + "stackexchange/0/monthlyCounts10.csv", delimiter=",")[:, 2:]
    lkml = np.loadtxt(SRC + "lkml/recursive/0/monthlyCounts10_2001.csv", delimiter=",")
    return {"ESM": esm, "StackExchange": se, "LKML": lkml}


def cost(X, lab):
    return sum(((X[lab == j] - X[lab == j].mean(0)) ** 2).sum() for j in np.unique(lab))


def kmeanspp(X, k):
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    return np.array(C)


def lloyd(X, k, restarts):
    best = None
    for _ in range(restarts):
        C = kmeanspp(X, k)
        for _ in range(200):
            a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C2 = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
            if np.allclose(C2, C):
                break
            C = C2
        c = cost(X, a)
        if best is None or c < best[0]:
            best = (c, a)
    return best[1]


def hierarchy(X, idx, depth, levels, max_depth):
    """One nested hierarchy. levels[d] collects the clusters present after d levels of splitting.
    Clusters with at most K users are not split. The root uses 10 restarts, deeper calls use 5."""
    levels[depth].append(idx)
    parts = []
    if len(idx) > K and depth < max_depth:
        a = lloyd(X[idx], K, restarts=10 if depth == 0 else 5)
        parts = [idx[a == j] for j in np.unique(a)]
    if len(parts) <= 1:
        for d in range(depth + 1, max_depth + 1):
            levels[d].append(idx)
        return
    for p in parts:
        hierarchy(X, p, depth + 1, levels, max_depth)


def fit(v, xmin):
    """Tail index of P[V >= v] = ((v - 0.5) / (xmin - 0.5))^-alpha on integers v >= xmin, and its KS distance."""
    t = v[v >= xmin]
    a = len(t) / np.log(t / (xmin - 0.5)).sum()
    u = np.unique(t)
    emp = np.searchsorted(t, u, side="right") / len(t)
    model = 1 - ((u + 0.5) / (xmin - 0.5)) ** -a
    return a, np.abs(emp - model).max(), len(t)


def tail_index(v):
    """Clauset-Shalizi-Newman: maximum likelihood for each cut-off, cut-off chosen by KS distance.
    Also returns the range of the estimate over all admissible cut-offs (tail of at least 50 users)."""
    v = np.sort(v.astype(float))
    fits = []
    for xmin in np.unique(v):
        a, ks, nt = fit(v, xmin)
        if nt < 50:
            break
        fits.append((ks, a, xmin, nt))
    best = min(fits)
    alphas = [f[1] for f in fits]
    return best, min(alphas), max(alphas)


lines = []
for name, X in load().items():
    n, N = X.shape
    vol = X.sum(1)
    (ks, a, xmin, nt), amin, amax = tail_index(vol)
    lines.append(f"{name}: n={n} N={N} volume min={vol.min():.0f} median={np.median(vol):.0f} max={vol.max():.0f}")
    lines.append(f"  tail index alpha={a:.2f} (s.e. {a / np.sqrt(nt):.2f}), xmin={xmin:.0f}, tail size={nt}, KS={ks:.3f}; range over cut-offs {amin:.2f} to {amax:.2f}")
    stats = {"depth 1 largest": [], "depth 1 singletons": [], "depth 2 largest": [], "depth 3 largest": [], "depth 3 clusters": [], "depth 3 singletons": []}
    for _ in range(REPS):
        levels = {d: [] for d in range(4)}
        hierarchy(X, np.arange(n), 0, levels, 3)
        for d in (1, 2, 3):
            sz = np.array(list(map(len, levels[d])))
            stats[f"depth {d} largest"].append(sz.max() / n)
            if d == 1:
                stats["depth 1 singletons"].append((sz == 1).sum())
            if d == 3:
                stats["depth 3 clusters"].append(len(sz))
                stats["depth 3 singletons"].append((sz == 1).sum())
    for key, val in stats.items():
        lines.append(f"  {key}: mean={np.mean(val):.2f} min={np.min(val):.2f} max={np.max(val):.2f} over {REPS} runs")

saved = np.loadtxt(SRC + "knome/kmeans11.csv")[:, 0].astype(int)
s = np.sort(np.bincount(saved)[1:])[::-1]
lines.append(f"ESM local search baseline (11 medians, saved assignment): sizes={s.tolist()} largest={s[0] / len(saved):.2f} singletons={(s == 1).sum()}")

text = "\n".join(lines)
print(text)
open("real_results.txt", "w").write(text + "\n")
