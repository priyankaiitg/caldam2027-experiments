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

## Data

`real_data.py` and `equal_k_real.py` read three monthly count matrices (ESM, StackExchange, LKML) from
`../../src/cikm/src/`. These matrices are not part of this repository, so the two scripts cannot be run from
it. Their results files are included. All other scripts generate their own data.
