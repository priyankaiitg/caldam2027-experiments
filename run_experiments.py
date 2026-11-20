"""Synthetic experiments for the CALDAM 2027 draft. Writes results.txt."""
import numpy as np

N, K, T, N_USERS = 28, 10, 40, 400
ALPHAS = [1.0, 1.5, 2.5, 4.0]
rng = np.random.default_rng(2027)


def cost(Z):
    return float((np.abs(Z - Z.mean(0)) ** 2).sum()) if len(Z) else 0.0


def kmeanspp(X, k):
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min((np.abs(X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    return np.array(C)


def lloyd(X, k, restarts, it=100):
    best = None
    for _ in range(restarts):
        C = kmeanspp(X, k)
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


REV = (-np.arange(N)) % N


def diversity(X):
    # Cosine similarity of the count vectors, as in cosinemst.m of the original code.
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


def hierarchical(X, idx, depth, out, delta, max_depth=3):
    if len(idx) <= K or depth == max_depth:
        out.append(idx)
        return
    a = lloyd(X[idx], K, restarts=3)
    parts = [idx[a == j] for j in range(K) if (a == j).any()]
    if len(parts) == 1:
        out.append(idx)
        return
    for p in parts:
        if len(p) > 1 and diversity(X[p]) <= delta:
            hierarchical(X, p, depth + 1, out, delta, max_depth)
        else:
            out.append(p)


def sample(alpha, n=N_USERS):
    v = rng.pareto(alpha, n) + 1
    X = np.array([rng.poisson(vi * rng.dirichlet(np.full(N, 0.3)) * N / 4) for vi in v]).astype(float)
    return X[X.sum(1) > 0]


def consistency_paper(P):
    c = np.fft.fft(P)
    c = c / np.abs(c).max()
    return 1 / np.std(c, ddof=1)


def consistency_closed(P):
    return np.sqrt((N - 1) / N) * P.sum() / np.sqrt((P ** 2).sum() - P[0] ** 2)


lines = []
X = sample(1.2)
rng_state = rng.bit_generator.state
a_time = lloyd(X, K, restarts=1)
rng.bit_generator.state = rng_state
a_freq = lloyd(np.fft.fft(X, axis=1), K, restarts=1)
lines.append(f"Parseval: identical partitions time vs DFT (same seeds): {np.array_equal(a_time, a_freq)}")
ok = [p for p in X if (p ** 2).sum() - p[0] ** 2 > 0]
err = max(abs(consistency_paper(p) - consistency_closed(p)) for p in ok)
lines.append(f"Consistency closed form: max abs error over {len(ok)} users = {err:.2e}")
lines.append(f"Consistency of constant series = {consistency_paper(np.ones(N)):.4f}, sqrt(N) = {np.sqrt(N):.4f}")
F = np.fft.fft(X, axis=1)
bil_err = np.max(np.abs(F @ F.T - N * (X @ X[:, REV].T)))
lines.append(f"Unconjugated dot identity F x . F y = N <x, R y>: max abs error = {bil_err:.2e}")
lines.append("")
lines.append("alpha | flat: largest frac | flat: singletons | hier(delta=0): #clusters, largest frac | hier(delta=inf): #clusters, largest frac")
for alpha in ALPHAS:
    lf, sg, h0n, h0l, hin, hil = [], [], [], [], [], []
    for _ in range(T):
        X = sample(alpha)
        m = len(X)
        sizes = np.bincount(lloyd(X, K, restarts=10), minlength=K)
        lf.append(sizes.max() / m)
        sg.append((sizes == 1).sum())
        for delta, nn, ll in ((0.0, h0n, h0l), (np.inf, hin, hil)):
            out = []
            hierarchical(X, np.arange(m), 0, out, delta)
            nn.append(len(out))
            ll.append(max(map(len, out)) / m)
    lines.append(f"{alpha} | {np.mean(lf):.2f} | {np.mean(sg):.1f} | {np.mean(h0n):.1f}, {np.mean(h0l):.2f} | {np.mean(hin):.1f}, {np.mean(hil):.2f}")

lines.append("")
lines.append("Scaling in n (flat k-means, 5 restarts, 20 draws): mean number of users outside the largest cluster")
lines.append("alpha | " + " | ".join(f"n={n}" for n in (200, 800, 3200, 12800)))
for alpha in (1.2, 4.0):
    row = []
    for n in (200, 800, 3200, 12800):
        outside = []
        for _ in range(20):
            X = sample(alpha, n)
            sizes = np.bincount(lloyd(X, K, restarts=5), minlength=K)
            outside.append(len(X) - sizes.max())
        row.append(f"{np.mean(outside):.1f}")
    lines.append(f"{alpha} | " + " | ".join(row))

text = "\n".join(lines)
print(text)
with open("results.txt", "w") as f:
    f.write(text + "\n")
