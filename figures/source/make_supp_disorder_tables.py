"""LaTeX tables for the SI disorder-check section (2026-10-21), written to output/si/disorder_tables.tex
from data/disorder_check/bender_disorder_check.csv (disorder_check.py analyze) and the BENDER ensembles."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fig_disagreement as fd  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

d = pd.read_csv(paths.DISORDER / "bender_disorder_check.csv").set_index("acc")
b, mr, fm = fd.load()
have = d.dropna(subset=["full_w20"]).copy()
have["full"] = have.full_w20.astype(bool)
pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
L = []

L.append(r"""\begin{table}[H]
  \centering
  \caption{Disorder of the BENDER chains by three criteria. All chains are disordered over their full length by
  MobiDB-lite, the selection criterion. The metapredict V2 column gives the chains forming one disordered region,
  and the AlphaFold2 columns apply the Human-IDRome criterion (20-residue windowed pLDDT) to the chains with an
  AlphaFold Database model, giving the chains forming one disordered region, chains with disordered regions
  covering at least 90\% of their length, residues in disordered regions and chains with none. The final column
  gives the chains more than half covered by UniProt domain-type annotations ($f_\mathrm{domain} > 0.5$).}
  \label{tab:disorder_groups}
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{lrrrrrrrr}
    \toprule
     & & metapredict & \multicolumn{5}{c}{AlphaFold2 (Human-IDRome criterion)} & \\
    \cmidrule(lr){3-3}\cmidrule(lr){4-8}
    Taxon & Chains & whole (\%) & models & whole (\%) & $\geq$90\% (\%) & residues (\%) & none (\%) & $f_\mathrm{dom} > 0.5$ (\%) \\
    \midrule""")
for g in fd.ORDER + ["All"]:
    s = d if g == "All" else d[d.group == g]
    h = have if g == "All" else have[have.group == g]
    row = (f"    {'\\textbf{All}' if g == 'All' else g} & {len(s):,} & {pct(s.full_V2.astype(bool).mean())} & {len(h):,} & "
           f"{pct(h.full.mean())} & {pct((h.frac_idr_w20 >= 0.9).mean())} & "
           f"{pct(np.average(h.frac_idr_w20, weights=h.n))} & {pct((h.n_idr_w20 == 0).mean())} & "
           f"{pct((s.f_domain > 0.5).mean())} \\\\")
    if g == "All":
        L.append(r"    \midrule")
    L.append(row)
L.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")

dnu = mr["nu"] - b["nu"]
ra = b["f_R"] + b["f_aro"]
cls = (ra > ra.quantile(0.9)) & (b["f_K"] < b["f_K"].median())
L.append(r"""\begin{table}[H]
  \centering
  \caption{The comparison between force fields on all chains and on the chains that do or do not pass the AlphaFold2
  criterion of disorder, with the \textit{contested} class as defined on all 10,282 chains. Here $r(\nu)$ is the Pearson
  correlation of $\nu$ between CALVADOS-2 and MPIPI-Recharged, $\Delta\nu$ is MPIPI-Recharged minus CALVADOS-2,
  and $\rho$ is the Spearman correlation of $\Delta\nu$ with Aromatic fraction and with mean hydropathy.}
  \label{tab:disorder_robust}
  \small
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{lrrrrrrrr}
    \toprule
     & & & & \multicolumn{2}{c}{\textit{Contested} class} & & \multicolumn{2}{c}{$\rho(\Delta\nu, \cdot)$} \\
    \cmidrule(lr){5-6}\cmidrule(lr){8-9}
    Chains & $n$ & $r(\nu)$ & $\langle\Delta\nu\rangle$ & $n$ & $\langle\Delta\nu\rangle$ & $|\Delta\nu| > 0.1$ & Aromatic & hydropathy \\
    \midrule""")
for name, idx in (("All", b.index), ("AlphaFold2, one region", have.index[have.full]),
                  ("AlphaFold2, at least partly folded", have.index[~have.full])):
    x, y, dd, c = b.loc[idx, "nu"], mr.loc[idx, "nu"], dnu.loc[idx], cls.loc[idx]
    L.append(f"    {name} & {len(idx):,} & {np.corrcoef(x, y)[0, 1]:.3f} & ${dd.mean():+.4f}$ & {int(c.sum())} & "
             f"${dd[c].mean():+.3f}$ & {pct((dd.abs() > 0.1).mean())}\\% & "
             f"${stats.spearmanr(b.loc[idx, 'f_aro'], dd)[0]:+.2f}$ & "
             f"${stats.spearmanr(b.loc[idx, 'hydropathy_avg'], dd)[0]:+.2f}$ \\\\")
L.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")

L.append(r"""\begin{table}[H]
  \centering
  \caption{Agreement of $\nu$ between force fields by taxon, on all chains and split by the AlphaFold2 criterion,
  with the median Lysine fraction of the two subsets.}
  \label{tab:disorder_groups_r}
  \small
  \begin{tabular}{lrrrrrr}
    \toprule
     & All & \multicolumn{2}{c}{One disordered region} & \multicolumn{2}{c}{At least partly folded} & median $f_K$ \\
    \cmidrule(lr){3-4}\cmidrule(lr){5-6}
    Taxon & $r(\nu)$ & $n$ & $r(\nu)$ & $n$ & $r(\nu)$ & partly folded / one region \\
    \midrule""")
for g in fd.ORDER:
    ia = b.index[b.kingdom == g]
    i1 = have.index[have.full & (have.group == g)]
    i0 = have.index[~have.full & (have.group == g)]
    rr = lambda idx: np.corrcoef(b.loc[idx, "nu"], mr.loc[idx, "nu"])[0, 1]  # noqa: E731
    L.append(f"    {g} & {rr(ia):.3f} & {len(i1):,} & {rr(i1):.3f} & {len(i0):,} & {rr(i0):.3f} & "
             f"{b.loc[i0, 'f_K'].median():.3f} / {b.loc[i1, 'f_K'].median():.3f} \\\\")
L.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")
out = paths.out("si", "disorder_tables.tex")
out.write_text("% Generated by figures/source/make_supp_disorder_tables.py; do not edit by hand.\n" + "\n".join(L))
print("wrote", out)
