"""Zipfian volumes: user of rank r has volume 5 * (n / r)^s, so the least active user has volume 5.
For s = 0.5 the total number of posts is about 10 n. Writes zipf_results.txt."""
import numpy as np

N, K, DRAWS = 28, 10, 20
rng = np.random.default_rng(2027)


def cost(X, lab):
    return sum(((X[lab == j] - X[lab == j].mean(0)) ** 2).sum() for j in np.unique(lab))


def kmeanspp(X, k):
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    return np.array(C)


def lloyd(X, k, restarts=5):
    best = None
    for _ in range(restarts):
        C = kmeanspp(X, k)
        for _ in range(100):
            a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C2 = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
            if np.allclose(C2, C):
                break
            C = C2
        c = cost(X, a)
        if best is None or c < best[0]:
            best = (c, a)
    return best[1]


def sample(s, n):
    v = 5.0 * (n / np.arange(1, n + 1)) ** s
    X = np.array([rng.poisson(vi * rng.dirichlet(np.full(N, 0.3))) for vi in v]).astype(float)
    return X[X.sum(1) > 0]


lines = ["s | alpha=1/s | " + " | ".join(f"n={n}: outside (share), total posts/n" for n in (100, 1000, 10000))]
for s in (0.25, 0.5, 0.75, 1.0):
    row = []
    for n in (100, 1000, 10000):
        outside, share, ratio = [], [], []
        for _ in range(DRAWS):
            X = sample(s, n)
            sizes = np.bincount(lloyd(X, K), minlength=K)
            outside.append(len(X) - sizes.max())
            share.append(1 - sizes.max() / len(X))
            ratio.append(X.sum() / len(X))
        row.append(f"{np.mean(outside):.1f} ({np.mean(share):.3f}), {np.mean(ratio):.0f}")
    lines.append(f"{s} | {1 / s:.2f} | " + " | ".join(row))
text = "\n".join(lines)
print(text)
open("zipf_results.txt", "w").write(text + "\n")
