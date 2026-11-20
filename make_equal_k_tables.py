"""Writes table_equal_k.tex from equal_k_real_results.txt and equal_k_synth_results.txt."""
import re


def t_cell(t):
    return r"$\infty$" if t == "inf" else t


real = {}
for l in open("equal_k_real_results.txt"):
    name, t = re.match(r"(\S+) t=(\S+):", l).groups()
    mean = dict(re.findall(r"(clusters|largest|singletons) mean=([\d.]+)", l))
    lo, hi = re.search(r"largest mean=[\d.]+ min=([\d.]+) max=([\d.]+)", l).groups()
    mean["largest"] += f" [{lo}, {hi}]"
    real.setdefault((name, t), {})["flat" if "flat" in l else "hier"] = mean
rows = []
ORDER = {"0.0": 0, "0.1": 1, "0.25": 2, "inf": 3}
for (name, t), r in sorted(real.items(), key=lambda kv: (list(dict.fromkeys(k[0] for k in real)).index(kv[0][0]), ORDER[kv[0][1]])):
    h, f = r["hier"], r["flat"]
    rows.append(f"{name.replace('StackExchange', 'StackEx.')} & {t_cell(t)} & {float(h['clusters']):.1f} & {h['largest']} & {float(h['singletons']):.1f} & {f['largest']} & {float(f['singletons']):.1f} \\\\")
synth = []
for l in list(open("equal_k_synth_results.txt"))[1:]:
    a, t, h, f = [c.strip() for c in l.split("|")]
    hc, hl, hs = [c.strip() for c in h.split(",")]
    fl, fs = [c.strip() for c in f.split(",")]
    synth.append(f"$\\alpha={a}$ & {t_cell(t)} & {hc} & {hl} & {hs} & {fl} & {fs} \\\\")

head = r"""\begin{tabular}{llccccc}
\toprule
& & \multicolumn{3}{c}{Hierarchy} & \multicolumn{2}{c}{Flat, same $k$} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Data & $t$ & clusters & largest & singletons & largest & singletons \\
\midrule
"""
open("table_equal_k.tex", "w").write(r"""\begin{table}[h]
\centering
\caption{The hierarchy with threshold $\delta=t(|T|-1)$ and at most three levels, and flat $k$-means with as many clusters as the hierarchy returned in the same run. ``Largest'' is the share of users in the largest cluster. Means over $10$ runs for the real data, with the range in brackets, and over $40$ draws for the synthetic data ($n=400$, $N=28$).}\label{tab:equalk}
\setlength{\tabcolsep}{4pt}\small
""" + head + "\n".join(rows) + "\n\\midrule\n" + "\n".join(synth) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
