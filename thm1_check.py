"""Check of Theorem 1: Lloyd's algorithm from the same random data points, on counts and on their DFT.
Writes thm1_results.txt."""
import numpy as np

N, K, N_USERS, SAMPLES = 28, 10, 400, 20
rng = np.random.default_rng(2027)


def lloyd_from(X, init, scale=1.0, decimals=None):
    """decimals: round the distances before the argmin, so that exact ties are broken by lowest index in both domains."""
    C = X[init].copy()
    a = None
    for _ in range(200):
        D = (np.abs(X[:, None] - C[None]) ** 2).sum(-1) / scale
        if decimals is not None:
            D = np.round(D, decimals)
        a2 = D.argmin(1)
        if a is not None and np.array_equal(a, a2):
            break
        a = a2
        C = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(K)])
    return a


same = same_ties = tied = 0
for _ in range(SAMPLES):
    v = rng.pareto(1.2, N_USERS) + 1
    X = np.array([rng.poisson(vi * rng.dirichlet(np.full(N, 0.3)) * N / 4) for vi in v]).astype(float)
    X = X[X.sum(1) > 0]
    init = rng.choice(len(X), K, replace=False)
    F = np.fft.fft(X, axis=1)
    same += np.array_equal(lloyd_from(X, init), lloyd_from(F, init, scale=N))
    same_ties += np.array_equal(lloyd_from(X, init, decimals=6), lloyd_from(F, init, scale=N, decimals=6))
    D0 = np.sort(((X[:, None] - X[init][None]) ** 2).sum(-1), 1)
    tied += int((D0[:, 0] == D0[:, 1]).sum())
text = (f"Theorem 1 check, random data points as initial centres, {SAMPLES} samples:\n"
        f"  identical final partitions, plain floating point: {same}\n"
        f"  identical final partitions, distances rounded to 1e-6 so that ties break the same way: {same_ties}\n"
        f"  mean number of users exactly tied between their two nearest initial centres: {tied / SAMPLES:.1f}")
print(text)
open("thm1_results.txt", "w").write(text + "\n")
