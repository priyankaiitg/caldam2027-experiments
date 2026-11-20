"""Exact optimal k-means in one dimension (dynamic programming) on Pareto volumes.
Counts the users outside the largest optimal cluster. Writes exact_1d_results.txt."""
import numpy as np

ALPHA, DRAWS = 1.2, 20
SIZES = (200, 800, 3200, 12800)
KS = (2, 3, 5, 10)
rng = np.random.default_rng(2027)


def exact_kmeans_sizes(v, kmax):
    """v sorted ascending. Returns {k: cluster sizes of an optimal k-clustering} for k = 1..kmax."""
    n = len(v)
    p1 = np.concatenate([[0.0], np.cumsum(v)])
    p2 = np.concatenate([[0.0], np.cumsum(v * v)])

    def seg(m, i):  # cost of v[m:i] as one cluster; m may be an array
        cnt = i - m
        s = p1[i] - p1[m]
        return (p2[i] - p2[m]) - s * s / cnt

    D = np.full((kmax + 1, n + 1), np.inf)
    arg = np.zeros((kmax + 1, n + 1), dtype=int)
    D[0, 0] = 0.0
    for j in range(1, kmax + 1):
        for i in range(j, n + 1):
            m = np.arange(j - 1, i)
            c = D[j - 1, m] + seg(m, i)
            b = int(c.argmin())
            D[j, i] = c[b]
            arg[j, i] = m[b]
    out = {}
    for k in range(1, kmax + 1):
        sizes, i = [], n
        for j in range(k, 0, -1):
            m = arg[j, i]
            sizes.append(i - m)
            i = m
        out[k] = sizes
    return out


lines = [f"alpha={ALPHA}, exact 1-D k-means, mean users outside the largest optimal cluster over {DRAWS} draws (min-max)",
         "k | " + " | ".join(f"n={n}" for n in SIZES)]
res = {k: {n: [] for n in SIZES} for k in KS}
for n in SIZES:
    for _ in range(DRAWS):
        v = np.sort(rng.pareto(ALPHA, n) + 1)
        sizes = exact_kmeans_sizes(v, max(KS))
        for k in KS:
            res[k][n].append(n - max(sizes[k]))
for k in KS:
    lines.append(f"{k} | " + " | ".join(f"{np.mean(res[k][n]):.1f} ({np.min(res[k][n])}-{np.max(res[k][n])})" for n in SIZES))
lines.append("standard error of the mean")
for k in KS:
    lines.append(f"{k} | " + " | ".join(f"{np.std(res[k][n], ddof=1) / np.sqrt(DRAWS):.1f}" for n in SIZES))
text = "\n".join(lines)
print(text)
open("exact_1d_results.txt", "w").write(text + "\n")
