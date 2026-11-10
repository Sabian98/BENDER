"""
Tables and figures of the Supporting Information for the JPC Letter.
Usage: python -I make_si.py <bio.csv> <mpipi_results_v2.csv> <saxs63_with_map.csv>
                            <saxs_meta (traj_vs_saxs_nu.csv)> <go_enrichment_expanded.tsv> <figures dir> <outdir>
Writes tab_*.tex (booktabs) and figS*.pdf into <outdir>. Every number comes from the input files or from the CSVs
that make_figures.py wrote into <figures dir>.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold

plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'cm', 'font.size': 7.5, 'axes.labelsize': 7.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 300})
TAXA = ['Bacteria', 'Fungi', 'Mammals', 'Plants', 'Protists', 'Viruses']
ORGANISMS = {'Bacteria': 654, 'Plants': 224, 'Fungi': 413, 'Mammals': 155, 'Protists': 246, 'Viruses': 464}
CAL, RCH, GGL, INK, MUTED = '#6FA3D2', '#E3A35A', '#3A6E38', '#333333', '#777777'
FF = [('nu_orig', 'CALVADOS-2', CAL, '-'), ('nu', 'Mpipi-Recharged', RCH, '--')]
DESC = ['FCR', 'NCPR', 'SCD', 'SHD', 'f_aro', 'f_P', 'f_G', 'hydropathy_avg', 'f_K', 'f_R', 'logN']
BINS = np.arange(0.38, 0.76, 0.02)
EDGES = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 1.0)]
ELAB = ['0 to 0.05', '0.05 to 0.1', '0.1 to 0.2', '0.2 to 0.35', 'above 0.35']


def tex_table(path, caption, label, header, rows, colspec, small=True, note=None, longtable=False):
    body = '\n'.join(' & '.join(str(c) for c in r) + r' \\' for r in rows)
    if longtable:
        s = (r'{\scriptsize\setlength{\tabcolsep}{2.2pt}' + '\n' + r'\begin{longtable}{' + colspec + '}\n' + r'\caption{' + caption + r'}\label{' + label
             + r'}\\' + '\n' + r'\toprule' + '\n' + ' & '.join(header) + r' \\' + '\n' + r'\midrule' + '\n' + r'\endfirsthead'
             + '\n' + r'\toprule' + '\n' + ' & '.join(header) + r' \\' + '\n' + r'\midrule' + '\n' + r'\endhead' + '\n'
             + body + '\n' + r'\bottomrule' + '\n' + r'\end{longtable}' + '\n}')
    else:
        s = (r'\begin{table}[htbp]' + '\n' + r'\centering' + '\n' + r'\caption{' + caption + r'}' + '\n' + r'\label{' + label
             + '}\n' + (r'\small' + '\n' if small and len(header) <= 5 else r'\footnotesize\setlength{\tabcolsep}{4pt}' + '\n') + r'\begin{tabular}{' + colspec + '}\n' + r'\toprule' + '\n'
             + ' & '.join(header) + r' \\' + '\n' + r'\midrule' + '\n' + body + '\n' + r'\bottomrule' + '\n'
             + r'\end{tabular}' + '\n' + (r'\par\smallskip\footnotesize ' + note + '\n' if note else '') + r'\end{table}')
    open(path, 'w').write(s + '\n')


def f(x, nd=3):
    if pd.isna(x):
        return '--'
    s = f'{x:.{nd}f}'
    return s.replace('-', '$-$') if s.startswith('-') else s


def main(bio, gg, saxs, meta, go, figdir, out):
    os.makedirs(out, exist_ok=True)
    raw = pd.read_csv(bio)
    d = raw[raw.nu_ok_cal.astype(bool) & raw.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig', 'nu']).copy()
    d['logN'] = np.log(d.length); d['aN'] = d.NCPR.abs(); d = d.dropna(subset=DESC)
    g = pd.read_csv(gg); g = g[g.nu_r2_mpipi >= 0.99][['UniProt_ID', 'nu_mpipi']]
    G = d.merge(g, on='UniProt_ID')

    # S1 data set
    rows = []
    for k in TAXA:
        rows.append([k, f'{(raw.kingdom == k).sum():,}', f'{ORGANISMS[k]:,}', f'{(d.kingdom == k).sum():,}',
                     f'{(G.kingdom == k).sum():,}', (d[d.kingdom == k].nu_orig > 0.63).sum(), (d[d.kingdom == k].nu > 0.63).sum()])
    rows.append(['Total', f'{len(raw):,}', f'{sum(ORGANISMS.values()):,}', f'{len(d):,}', f'{len(G):,}',
                 (d.nu_orig > 0.63).sum(), (d.nu > 0.63).sum()])
    tex_table(f'{out}/tab_dataset.tex', 'Sequences per taxon. Analyzed sequences pass the fit of $\\nu$ in both \\textsc{calvados-2} and Mpipi-Recharged, and expanded IDPs have $\\nu > 0.63$ in the force field named.',
              'tab:dataset', ['Taxon', 'Sequences', 'Organisms', 'Analyzed', 'Mpipi-GG', 'Expanded (C2)', 'Expanded (MR)'],
              rows, 'lrrrrrr')

    # S2 stage markers
    m = pd.read_csv(f'{figdir}/fig2a_stage_markers.csv')
    rows = [[r.taxon, r.ff, f'{f(r.fcr_min, 2)} ({f(r.fcr_min_value, 3)})', f'{r.ncpr_onset:.3f} ({r.ncpr_lo:.3f} to {r.ncpr_hi:.3f})',
             f'{r.scd_onset:.3f} ({r.scd_lo:.3f} to {r.scd_hi:.3f})', f(r.lys_over_arg, 2)] for r in m.itertuples()]
    tex_table(f'{out}/tab_stages.tex', 'Positions along $\\nu$ of the four markers of the charge transition for each taxon and force field. Onsets are medians of 200 bootstrap hinge fits with 95\\% intervals in parentheses. The FCR minimum and the crossing of Lysine over Arginine are the centers of the binned values, with the mean FCR at the minimum in parentheses.',
              'tab:stages', ['Taxon', 'Force field', 'FCR minimum', '$|$NCPR$|$ onset', 'SCD onset', 'Lys over Arg'], rows, 'llcccc')

    # S3 curve agreement
    ca = pd.read_csv(f'{figdir}/fig3a_curve_agreement.csv', index_col=0)
    names = {'Aro': 'Aromatic', 'Hyd': 'Hydropathy', '|NCPR|': '$|$NCPR$|$'}
    rows = [[i] + [f(v, 2) for v in r] for i, r in ca.iterrows()]
    tex_table(f'{out}/tab_curves.tex', 'Pearson $r$ between decile curves of each descriptor in \\textsc{calvados-2} and Mpipi-Recharged per taxon and pooled, and between \\textsc{calvados-2} and Mpipi-GG for the shared subset (eight bins).',
              'tab:curves', ['Taxon'] + [names.get(c, c) for c in ca.columns], rows, 'l' + 'c' * len(ca.columns))

    # S4 branch slopes
    rows = []
    for col, ffn, data in [('nu_orig', 'CALVADOS-2', d), ('nu', 'Mpipi-Recharged', d), ('nu_orig', 'CALVADOS-2, GG subset', G), ('nu_mpipi', 'Mpipi-GG', G)]:
        cells = []
        for lo, hi in EDGES:
            s = data[(data.aN > lo) & (data.aN <= hi)] if lo > 0 else data[data.aN <= hi]
            r = spearmanr(s.FCR, s[col]); cells.append(f'{f(r.correlation, 2)} ({len(s)})')
        rows.append([ffn] + cells)
    tex_table(f'{out}/tab_branches.tex', 'Spearman $\\rho$ between FCR and $\\nu$ within bins of $|$NCPR$|$, with the number of IDPs in parentheses. Negative values mean that added charge compacts chains and positive values mean that it expands them.',
              'tab:branches', ['Force field'] + ELAB, rows, 'lccccc')

    # S5 diagram of states
    def reg(r):
        if r.FCR < 0.25: return 'Weak PA'
        if r.FCR <= 0.35: return 'Janus'
        if r.aN <= 0.35: return 'Strong PA'
        return 'Polyanion' if r.NCPR < 0 else 'Polycation'
    d['reg'] = d.apply(reg, axis=1); R = ['Weak PA', 'Janus', 'Strong PA', 'Polyanion', 'Polycation']
    occ = pd.crosstab(d.kingdom, d.reg, normalize='index').reindex(index=TAXA, columns=R) * 100
    rows = []
    for k in TAXA:
        row = [k]
        for r_ in R:
            sub = d[(d.kingdom == k) & (d.reg == r_)]
            row.append(f'{occ.loc[k, r_]:.1f} ({f(sub.nu_orig.median(), 3)}, {f(sub.nu.median(), 3)})' if len(sub) >= 5 else f'{occ.loc[k, r_]:.1f}')
        rows.append(row)
    tex_table(f'{out}/tab_states.tex', 'Share of IDPs (\\%) in each region of the diagram of states, with the median $\\nu$ in \\textsc{calvados-2} and Mpipi-Recharged in parentheses where a cell holds at least five IDPs. PA denotes polyampholytes.',
              'tab:states', ['Taxon'] + R, rows, 'lccccc')

    # S6 models
    base = 'FCR + I(NCPR**2) + SCD + SHD + f_aro + f_P + f_G + hydropathy_avg + f_K + f_R + logN'
    rows = []
    for col, ffn, _, _ in FF:
        m0 = smf.ols(f'{col} ~ {base}', d).fit(); m1 = smf.ols(f'{col} ~ {base} + C(kingdom, Treatment("Mammals"))', d).fit()
        p = cross_val_predict(HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0), d[DESC], d[col],
                              cv=KFold(5, shuffle=True, random_state=0))
        r2g = 1 - np.var(d[col] - p) / np.var(d[col])
        rows.append([ffn, f(m0.rsquared, 3), f(m1.rsquared, 3), f'{m1.compare_f_test(m0)[1]:.1e}', f(r2g, 3)])
    tex_table(f'{out}/tab_models.tex', 'Models of $\\nu$ on eleven sequence descriptors. Linear $R^2$ without and with a taxon factor, the $F$ test between them, and the five-fold cross-validated $R^2$ of the nonlinear (gradient-boosting) model without a taxon label.',
              'tab:models', ['Force field', 'Linear $R^2$', 'Linear $R^2$ + taxon', '$F$ test $p$', 'Nonlinear CV $R^2$'], rows, 'lcccc')
    e = pd.read_csv(f'{figdir}/fig4e_taxon_offsets.csv'); fl = pd.read_csv(f'{figdir}/fig4f_leave_one_taxon_out.csv')
    rows = []
    for k in TAXA:
        row = [k]
        for ffn in ['CALVADOS-2', 'Mpipi-Recharged']:
            lin = e[(e.ff == ffn) & (e.model == 'Linear')][k].iloc[0]; flx = e[(e.ff == ffn) & (e.model == 'Flexible')][k].iloc[0]
            q = fl[(fl.ff == ffn) & (fl.taxon == k)].iloc[0]
            row += [f(1000 * lin, 1), f(1000 * flx, 1), f(q.within, 3), f(q.held_out, 3)]
        rows.append(row)
    tex_table(f'{out}/tab_transfer.tex', 'Mean prediction error of each taxon ($\\nu \\times 10^3$, observed minus predicted, with each IDP predicted by a model not trained on it) under the linear model and the nonlinear model, and $R^2$ of the nonlinear model from cross-validation within the taxon and for the taxon held out of training.',
              'tab:transfer', ['Taxon', 'C2 linear', 'C2 nonlinear', 'C2 within', 'C2 held out', 'MR linear', 'MR nonlinear', 'MR within', 'MR held out'],
              rows, 'lcccccccc', note='C2, \\textsc{calvados-2}. MR, Mpipi-Recharged.')

    # S7 SAXS constructs
    x = pd.read_csv(saxs); mm = pd.read_csv(meta)[['traj_id', 'protein_name']].drop_duplicates('traj_id')
    x = x.merge(mm, on='traj_id', how='left').sort_values(['taxon', 'traj_id'])
    def clean(s):
        s = str(s)
        if len(s) > 26:
            s = s[:26].rsplit(' ', 1)[0].rstrip(' (,') + '...'
        return s.replace('&', '\\&').replace('_', '\\_').replace('%', '\\%').replace('#', '\\#')
    rows = [[r.sasbdb_id, clean(r.protein_name), r.taxon.replace('Other vertebrate', 'Vertebrate'), 'yes' if r.set == 'strict' else 'no', r.n_res, f(r.saxs_rg_nm, 2), f(r.nueff_saxs, 3), f(r.nueff_cal_sim, 3),
             f(r.nueff_rech_sim, 3), f(r.pred_nueff_cal, 3), f(r.pred_nueff_rech, 3), f(r.FCR, 2), f(r.NCPR, 2)] for r in x.itertuples()]
    tex_table(f'{out}/tab_saxs.tex', 'The 63 SAXS constructs. Measured radius of gyration (nm) and effective exponent, $\\nu_\\mathrm{eff}$ from direct simulation in \\textsc{calvados-2} (C2) and Mpipi-Recharged (MR), $\\nu_\\mathrm{eff}$ from the nonlinear model trained on each force field, and the charge descriptors of each construct. Strict constructs meet both disorder criteria, and the others lack a UniProt mapping.',
              'tab:saxs', ['SASBDB', 'Protein', 'Taxon', 'Strict', '$N$', '$R_g$', '$\\nu_\\mathrm{eff}$', 'C2 sim.', 'MR sim.', 'C2 model', 'MR model', 'FCR', 'NCPR'],
              rows, 'lllclrrrrrrrr', longtable=True)

    # S8 GO
    G8 = pd.read_csv(go, sep='\t'); rows = []
    for ffn in ['CALVADOS-2', 'Mpipi-Recharged']:
        s = G8[(G8.ff == ffn) & (G8.group == 'expanded_basic') & (G8.q < 0.05)].sort_values('q').head(12)
        for r in s.itertuples():
            rows.append([ffn, r.term.replace('&', '\\&'), f'{r.k}/{r.n}', f(r.frac_bg, 3), f(r.odds, 1), f'{r.q:.1e}'])
    tex_table(f'{out}/tab_go.tex', 'Gene Ontology terms (BP, MF, and CC) and UniProt keywords (KW) enriched among annotated expanded basic IDPs ($\\nu > 0.63$, NCPR above 0.1) against all annotated IDPs, with the share of the group, the share of the background, the odds ratio, and the Benjamini--Hochberg $q$. Only 8 (\\textsc{calvados-2}) and 7 (Mpipi-Recharged) expanded acidic IDPs carry any annotation, too few to test.',
              'tab:go', ['Force field', 'Term', 'Group', 'Background', 'Odds ratio', '$q$'], rows, 'llcccc')

    # Figure S1, extra descriptors per taxon
    extra = [('f_K', 'Lysine'), ('f_R', 'Arginine'), ('f_aro', 'Aromatic'), ('f_P', 'Proline'), ('f_G', 'Glycine'), ('hydropathy_avg', 'Hydropathy')]
    fig, axes = plt.subplots(len(extra), 6, figsize=(7.0, 7.1), sharex=True, sharey='row')
    for j, k in enumerate(TAXA):
        sub = d[d.kingdom == k]
        for i, (v, lab) in enumerate(extra):
            ax = axes[i, j]; ax.axvspan(0.49, 0.60, color='#EFEFEF', lw=0, zorder=0)
            ax.axvline(0.588, color=MUTED, lw=0.6, ls=':')
            for col, ffn, c, ls in FF:
                b = pd.cut(sub[col], BINS); q = sub.groupby(b, observed=True)[v].agg(['mean', 'sem', 'size']); q = q[q['size'] >= 10]
                xs = np.array([t.mid for t in q.index])
                ax.fill_between(xs, q['mean'] - q['sem'], q['mean'] + q['sem'], color=c, alpha=0.22, lw=0)
                ax.plot(xs, q['mean'], ls=ls, color=c, lw=1.2)
            if i == 0: ax.set_title(k, fontsize=7.5, color=INK)
            if j == 0: ax.set_ylabel(lab)
            ax.set_xlim(0.40, 0.72); ax.set_xticks([0.45, 0.55, 0.65])
    fig.legend(handles=[Line2D([], [], color=CAL, lw=1.2, label='CALVADOS-2'), Line2D([], [], color=RCH, lw=1.2, ls='--', label='Mpipi-Recharged')],
               loc='lower center', ncol=2, frameon=False, fontsize=6.5)
    fig.supxlabel(r'Flory exponent $\nu$', y=0.04, fontsize=7.5); fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.3)
    fig.savefig(f'{out}/figS1_extra_descriptors.pdf'); plt.close(fig)

    # Figure S2, Mpipi-GG against CALVADOS-2 on the shared subset
    rows3 = [('FCR', 'FCR'), ('aN', '$|$NCPR$|$'.replace('$|$', '|')), ('SCD', 'SCD')]
    fig, axes = plt.subplots(3, 6, figsize=(7.0, 3.9), sharex=True, sharey='row')
    G['aN'] = G.NCPR.abs()
    for j, k in enumerate(TAXA):
        sub = G[G.kingdom == k]
        for i, (v, lab) in enumerate(rows3):
            ax = axes[i, j]
            for col, c, ls, lbl in [('nu_orig', CAL, '-', 'CALVADOS-2'), ('nu_mpipi', GGL, '-.', 'Mpipi-GG')]:
                b = pd.cut(sub[col], np.arange(0.38, 0.76, 0.03)); q = sub.groupby(b, observed=True)[v].agg(['mean', 'size']); q = q[q['size'] >= 5]
                ax.plot([t.mid for t in q.index], q['mean'], ls=ls, color=c, lw=1.2, marker='o', ms=2)
            if i == 0: ax.set_title(f'{k} (n = {len(sub)})', fontsize=7, color=INK)
            if j == 0: ax.set_ylabel(lab)
            ax.set_xlim(0.40, 0.72); ax.set_xticks([0.45, 0.55, 0.65])
    fig.legend(handles=[Line2D([], [], color=CAL, lw=1.2, label='CALVADOS-2'), Line2D([], [], color=GGL, lw=1.2, ls='-.', label='Mpipi-GG')],
               loc='lower center', ncol=2, frameon=False, fontsize=6.5)
    fig.supxlabel(r'Flory exponent $\nu$', y=0.05, fontsize=7.5); fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.6, w_pad=0.3)
    fig.savefig(f'{out}/figS2_mpipi_gg.pdf'); plt.close(fig)
    print('SI tables and figures written to', out)


if __name__ == '__main__':
    main(*sys.argv[1:8])
