"""Writes table.tex and table_scaling.tex from results.txt."""
rows = [l.strip() for l in open("results.txt")]
i = next(j for j, l in enumerate(rows) if l.startswith("alpha | flat"))
synth = []
for l in rows[i + 1:i + 5]:
    a, fl, sg, h0, hi = [c.strip() for c in l.split("|")]
    synth.append(f"{a} & {fl} & {sg} & {h0.split(',')[0]} & {h0.split(',')[1].strip()} & {hi.split(',')[0]} & {hi.split(',')[1].strip()} \\\\")
j = next(q for q, l in enumerate(rows) if l.startswith("alpha | n="))
sizes = [int(c.split("=")[1]) for c in rows[j].split("|")[1:]]


def share(x):
    return f"{x:.2f}" if x >= 0.1 else (f"{x:.3f}" if x >= 0.01 else f"{x:.4f}")


scal = []
for l in rows[j + 1:j + 3]:
    cells = [c.strip() for c in l.split("|")]
    scal.append(f"{cells[0]} & " + " & ".join(f"{c} ({share(float(c) / n)})" for c, n in zip(cells[1:], sizes)) + " \\\\")

open("table.tex", "w").write(r"""\begin{table}[h]
\centering
\caption{Synthetic data ($n=400$, $N=28$, $k=10$, means over $40$ draws). ``Largest'' is the share of users in the largest cluster, and ``clusters'' is the number of final clusters.}\label{tab:synth}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Flat $k$-means} & \multicolumn{2}{c}{Hierarchy, $\delta=0$} & \multicolumn{2}{c}{Hierarchy, $\delta=\infty$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
$\alpha$ & largest & singletons & clusters & largest & clusters & largest \\
\midrule
""" + "\n".join(synth) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
open("table_scaling.tex", "w").write(r"""\begin{table}[h]
\centering
\caption{Mean number of users outside the largest flat $k$-means cluster ($N=28$, Poisson counts, $k=10$, five restarts, $20$ draws). In parentheses, this mean divided by $n$.}\label{tab:scaling}
\begin{tabular}{lcccc}
\toprule
$\alpha$ & """ + " & ".join(f"$n={n}$" for n in sizes) + r""" \\
\midrule
""" + "\n".join(scal) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
