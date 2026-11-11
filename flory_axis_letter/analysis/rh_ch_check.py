"""Two further checks for the Flory-axis letter (Section S9, Table S10).

1. Measured hydrodynamic radii. Table S1 of Marsh and Forman-Kay (Biophys. J. 2010, 98, 2383) lists 32 IDPs with
   Rh measured by PFG-NMR (12) or size-exclusion chromatography (20) and their sequences
   (data/flory_axis_letter/marsh_2010_table_s1.csv). Compactness is the residual of ln Rh from a linear fit against
   ln N across the 32, so it needs no conversion from Rh to the radius of gyration. The nonlinear models of the SAXS
   check (ten descriptors, trained on nu_eff of the BENDER ensembles) predict nu_eff for each protein, and the test is
   the Spearman rho between prediction and compactness, for all 32 and for the 26 that share no 20-residue stretch
   with any BENDER sequence.
2. Charge-hydropathy plot (Uversky, Gillespie and Fink, Proteins 2000, 41, 415). Mean hydropathy is the Kyte-Doolittle
   scale normalized to 0 to 1 and averaged over windows of five residues, mean net charge is |NCPR|, and a sequence
   lies on the folded side when its hydropathy exceeds (|NCPR| + 1.151) / 2.785.

Usage: python rh_ch_check.py <bio.csv> <marsh_2010_table_s1.csv> <outdir>   (writes <outdir>/tab_rh.tex)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc import KD, desc  # noqa: E402

R0 = 0.1963
F = ['FCR', 'NCPR', 'SCD', 'f_aro', 'f_P', 'f_G', 'hydropathy_avg', 'f_K', 'f_R', 'logN']


def ch_hydropathy(s):
    h = np.array([(KD.get(c, 0) + 4.5) / 9 for c in s])
    return h.mean() if len(h) < 5 else np.convolve(h, np.ones(5) / 5, mode='valid').mean()


def bender_relatives(marsh_seqs, bender_seqs, k=20):
    kmers = set()
    for s in bender_seqs:
        kmers.update(s[i:i + k] for i in range(len(s) - k + 1))
    return [any(s[i:i + k] in kmers for i in range(len(s) - k + 1)) for s in marsh_seqs]


def main(bio, marsh, out):
    d = pd.read_csv(bio, low_memory=False)
    d = d[d.nu_ok_cal.astype(bool) & d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig', 'nu', 'rg_orig', 'rg', 'sequence'])
    d['logN'] = np.log(d.length)
    sc = 10 if d.rg_orig.median() > 10 else 1

    # 1. hydrodynamic radii
    m = pd.read_csv(marsh)
    x = pd.concat([m, pd.DataFrame([desc(s) for s in m.seq])], axis=1)
    x['Nseq'] = x.seq.str.len()
    b, a = np.polyfit(np.log(x.Nseq), np.log(x.Rh), 1)
    x['compactness'] = np.log(x.Rh) - (a + b * np.log(x.Nseq))
    x['relative'] = bender_relatives(x.seq, d.sequence)
    print(f'Rh: {len(x)} IDPs ({(x.method == "PFG-NMR").sum()} PFG-NMR), ln Rh = {a:.2f} + {b:.3f} ln N; '
          f'{x.relative.sum()} share a 20-residue stretch with BENDER: {", ".join(x.name[x.relative])}')
    for col, lab in [('rg_orig', 'C2'), ('rg', 'MR')]:
        y = np.log(d[col] / sc / R0) / np.log(d.length)
        x['pred_' + lab] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(d[F], y).predict(x[F])
        for sub, mk in [('all', np.ones(len(x), bool)), ('no relatives', ~x.relative.values)]:
            r = spearmanr(x['pred_' + lab][mk], x.compactness[mk])
            print(f'  {lab} model vs compactness, {sub:12s} n={mk.sum():2d}  rho={r.correlation:+.2f}  p={r.pvalue:.3f}')
    keep = ~x.relative
    r = spearmanr(x.NCPR.abs()[keep], x.compactness[keep]); print(f'  |NCPR| alone, no relatives: rho={r.correlation:+.2f} p={r.pvalue:.3f}')
    aN = x.NCPR.abs()
    r = spearmanr(x.FCR[aN <= 0.1], x.compactness[aN <= 0.1])
    print(f'  compacting branch: {int((aN <= 0.1).sum())} with |NCPR| <= 0.1, rho(FCR, compactness) = {r.correlation:+.2f} p = {r.pvalue:.2f}; '
          f'charge-rich (FCR > 0.35) and |NCPR| <= 0.1: {", ".join(x.name[(x.FCR > 0.35) & (aN <= 0.1)])}')

    def f(v, nd=3):
        s = f'{v:.{nd}f}'
        return s.replace('-', '$-$') if s.startswith('-') else s
    rows = [[r.name.replace('alpha', r'$\alpha$').replace('gamma', r'$\gamma$'), r.method, r.Nseq, f(r.Rh, 1), f(r.FCR, 2), f(r.NCPR, 2),
             f(r.compactness), f(r.pred_C2), f(r.pred_MR), 'yes' if r.relative else 'no'] for r in x.sort_values('compactness').itertuples()]
    body = '\n'.join(' & '.join(str(c) for c in row) + r' \\' for row in rows)
    tex = (r'\begin{table}[htbp]' + '\n' + r'\centering' + '\n'
           + r'\caption{The 32 IDPs with measured hydrodynamic radii from Marsh and Forman-Kay, ordered from the most compact to the most expanded for their length. $R_h$ in \AA, compactness as the residual of $\ln R_h$ from a fit against $\ln N$ (positive values are more expanded), $\nu_\mathrm{eff}$ predicted by the nonlinear models trained on \textsc{calvados-2} (C2) and Mpipi-Recharged (MR), and whether the protein shares a 20-residue stretch with any BENDER sequence.}' + '\n'
           + r'\label{tab:rh}' + '\n' + r'\footnotesize\setlength{\tabcolsep}{3.5pt}' + '\n' + r'\begin{tabular}{llrrrrrrrc}' + '\n' + r'\toprule' + '\n'
           + r'Protein & Method & $N$ & $R_h$ & FCR & NCPR & Compactness & C2 model & MR model & In BENDER \\' + '\n' + r'\midrule' + '\n'
           + body + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n')
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'tab_rh.tex'), 'w').write(tex)

    # 2. charge-hydropathy plot
    d['H'] = d.sequence.map(ch_hydropathy)
    d['folded_side'] = d.H > (d.NCPR.abs() + 1.151) / 2.785
    cr = d[(d.FCR > 0.35) & (d.NCPR.abs() <= 0.05)]
    print(f'Charge-hydropathy: {len(d)} IDPs, {100 * (1 - d.folded_side.mean()):.1f}% on the disordered side; '
          f'{len(cr)} charge-rich and nearly neutral (FCR > 0.35, |NCPR| <= 0.05), {100 * cr.folded_side.mean():.1f}% on the folded side, '
          f'median nu {cr.nu_orig.median():.3f} (C2) and {cr.nu.median():.3f} (MR) against {d.nu_orig.median():.3f} and {d.nu.median():.3f} overall; '
          f'compact IDPs (nu < 0.5, C2) on the folded side {100 * d[d.nu_orig < 0.5].folded_side.mean():.1f}%')


if __name__ == '__main__':
    main(*sys.argv[1:4])
