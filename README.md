# Experiments for "Why Flat k-Means Collapses on Long-Tailed Activity Data"

Scripts, results and generated tables for the CALDAM 2027 paper by Priyanka Sinha, Dilys Thomas,
Nikolaos Kekatos, Antonio Bruto da Costa and Pabitra Mitra.

Requirements: Python 3 with NumPy. Every script uses a fixed seed and writes a results file next to itself.

| Script | Output | What it does |
|---|---|---|
| `run_experiments.py` | `results.txt` | Synthetic Pareto data: flat k-means, the hierarchy with delta = 0 and delta = infinity, scaling in n, and checks of the closed forms |
| `thm1_check.py` | `thm1_results.txt` | Lloyd's algorithm on the counts and on their DFT from the same start |
| `exact_1d.py` | `exact_1d_results.txt` | Exact one-dimensional k-means by dynamic programming |
| `zipf_experiment.py` | `zipf_results.txt` | Deterministic Zipfian volumes |
| `equal_k_synth.py` | `equal_k_synth_results.txt` | Hierarchy with threshold t(|T|-1) against flat k-means with the same number of clusters, synthetic data |
| `real_data.py` | `real_results.txt` | Tail index and hierarchy on the three real datasets |
| `equal_k_real.py` | `equal_k_real_results.txt` | The same comparison at an equal number of clusters, real data |
| `make_tables.py` | `table.tex`, `table_scaling.tex` | Tables from `results.txt` |
| `make_equal_k_tables.py` | `table_equal_k.tex` | Table from the two `equal_k_*_results.txt` files |

## Supplement

`supplement.pdf` (source: `supplement.tex`) contains the proofs of Propositions 1, 3 and 4, Corollary 2 and
Theorem 4, which the paper omits for lack of space. All other proofs are in the paper.

## Data

`real_data.py` and `equal_k_real.py` read three monthly count matrices (ESM, StackExchange, LKML) from
`../../src/cikm/src/`. These matrices are not part of this repository, so the two scripts cannot be run from
it. Their results files are included. All other scripts generate their own data.

## Implementation notes

These notes describe the Octave, C++ and R code that accompanies the earlier paper (Sinha, Dey, Mitra,
Thomas: "A Hierarchical Clustering Algorithm for Characterizing Social Media Users", WWW 2020 Companion),
which is the method this paper analyses. That code is not part of this repository.

**Clustering.**
- Lloyd's algorithm. The number of clusters is an argument; the earlier paper reports k = 10.
- The initial centres are k data points chosen at random.
- The distance from a point to a centre is the sum of squared moduli of the coefficient differences, and a
  centre is the complex mean of its members.
- Iteration stops when the cost improves by less than 10^-3.
- The released file checks that its input is real, so this check has to be removed before it accepts
  spectra. The iteration itself is as described.
- By default an empty cluster raises an error. The optional fallback, which should move the farthest point
  into the empty cluster, always moves the last point. The scripts in this repository keep the old centre
  instead.

**Diversity index.**
- For a cluster T, form the complete graph on T with edge weights equal to the cosine similarity
  `w_ij = <u_i, u_j> / (|u_i| |u_j|)`, where `u_i` is either the count vector or its DFT.
- A C++ program runs Kruskal's algorithm with edges in increasing order and returns the total weight I(T)
  of the minimum spanning tree.
- Octave's `dot` conjugates its first argument, so by Parseval's identity both choices of `u_i` give the
  same weight. Without the conjugation the product would compare month t of one user with month N - t of
  the other, and the index would no longer measure co-activity.
- The variant for spectra also takes the modulus, which changes nothing for per-user spectra of counts.
- Both variants skip the first column of their input, which holds a label.
- The C++ program stores at most 10^6 edges, so it cannot handle a cluster with more than about 1400 users.
- The similarity files that could be matched to saved clusters agree with the cosine similarity of the raw
  counts to machine precision (2.2 x 10^-16).

**Consistency score.**
- For a complex vector v of length N with mean m, let `s(v)^2 = sum_f |v_f - m|^2 / (N - 1)`. This is what
  Octave's `std` computes for complex input.
- The score of a centre c is `1 / s(c / max_f |c_f|)`.

**Baselines.**
- DTW clustering used the R package `dtwclust` in hierarchical mode with 30 clusters.
- Local search used the code of O'Callaghan et al. (ICDE 2002). That code minimises the sum of squared
  Euclidean distances to the chosen medians, which is the k-means objective restricted to centres at data
  points.
