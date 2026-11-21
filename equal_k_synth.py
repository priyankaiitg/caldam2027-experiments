"""Synthetic data: the hierarchy with threshold delta = t (|T| - 1), and flat k-means with as many clusters as
the hierarchy returned. Writes equal_k_synth_results.txt.
The first stream replays run_experiments.py, so the samples and the hierarchies with delta = 0 and delta = inf
are those of the paper's synthetic results table. The hierarchies with t = 0.1 and t = 0.25 and all flat runs with the same k use a second
stream on the same samples."""
import numpy as np

N, K, T, N_USERS = 28, 10, 40, 400
ALPHAS = [1.0, 1.5, 2.5, 4.0]
rng = np.random.default_rng(2027)   # the stream of run_experiments.py
rng2 = np.random.default_rng(2028)  # everything else


def cost(Z):
    return float((np.abs(Z - Z.mean(0)) ** 2).sum()) if len(Z) else 0.0


def kmeanspp(X, k, g):
    C = [X[g.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min((np.abs(X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[g.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[g.integers(len(X))])
    return np.array(C)


def lloyd(X, k, restarts, g, it=100):
    best = None
    for _ in range(restarts):
        C = kmeanspp(X, k, g)
        for _ in range(it):
            a = (np.abs(X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C2 = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
            if np.allclose(C2, C):
                break
            C = C2
        c = sum(cost(X[a == j]) for j in range(k))
        if best is None or c < best[0]:
            best = (c, a)
    return best[1]


def diversity(X):
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


def hierarchical(X, idx, depth, out, t, g, max_depth=3):
    """As in run_experiments.py, with the threshold t (|T| - 1). For t = 0 and t = inf the comparison is the
    same as with delta = 0 and delta = inf there."""
    if len(idx) <= K or depth == max_depth:
        out.append(idx)
        return
    a = lloyd(X[idx], K, 3, g)
    parts = [idx[a == j] for j in range(K) if (a == j).any()]
    if len(parts) == 1:
        out.append(idx)
        return
    for p in parts:
        if len(p) > 1 and diversity(X[p]) <= (np.inf if t == np.inf else t * (len(p) - 1)):
            hierarchical(X, p, depth + 1, out, t, g, max_depth)
        else:
            out.append(p)


def sample(alpha, n=N_USERS):
    v = rng.pareto(alpha, n) + 1
    X = np.array([rng.poisson(vi * rng.dirichlet(np.full(N, 0.3)) * N / 4) for vi in v]).astype(float)
    return X[X.sum(1) > 0]


# Replay the checks at the start of run_experiments.py, which draw from the first stream.
X = sample(1.2)
state = rng.bit_generator.state
lloyd(X, K, 1, rng)
rng.bit_generator.state = state
lloyd(np.fft.fft(X, axis=1), K, 1, rng)

lines = ["alpha | t | hierarchy: clusters, largest, singletons | flat k-means with the same k: largest, singletons"]
check = []
for alpha in ALPHAS:
    res = {t: ([], []) for t in (0.0, 0.1, 0.25, np.inf)}
    lf = []
    for _ in range(T):
        X = sample(alpha)
        m = len(X)
        sizes = np.bincount(lloyd(X, K, 10, rng), minlength=K)
        lf.append(sizes.max() / m)
        outs = {}
        for t in (0.0, np.inf):
            outs[t] = []
            hierarchical(X, np.arange(m), 0, outs[t], t, rng)
        for t in (0.1, 0.25):
            outs[t] = []
            hierarchical(X, np.arange(m), 0, outs[t], t, rng2)
        for t, out in outs.items():
            s = np.array(list(map(len, out)))
            g = np.bincount(lloyd(X, len(s), 5, rng2))
            g = g[g > 0]
            res[t][0].append((len(s), s.max() / m, (s == 1).sum()))
            res[t][1].append((g.max() / m, (g == 1).sum()))
    check.append(f"{alpha}: flat k=10 largest {np.mean(lf):.2f}")
    for t in (0.0, 0.1, 0.25, np.inf):
        h, f = np.array(res[t][0]), np.array(res[t][1])
        lines.append(f"{alpha} | {t} | {h[:, 0].mean():.1f}, {h[:, 1].mean():.2f}, {h[:, 2].mean():.1f} | {f[:, 0].mean():.2f}, {f[:, 1].mean():.1f}")
        print(lines[-1], flush=True)
print("\n".join(check))
open("equal_k_synth_results.txt", "w").write("\n".join(lines) + "\n")
