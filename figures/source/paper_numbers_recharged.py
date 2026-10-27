"""
paper_numbers_recharged.py — every number quoted in the revised text (2026-10-20): 10,282 BENDER sequences (the
300-protist top-up included) and MPIPI-Recharged on all of them, with MPIPI-GG kept as the 957-sequence subset.

  python3 paper_numbers_recharged.py [--bio PATH] [--recharged PATH] [--gg PATH] [--pae-c2 PATH] [--pae-gg PATH]
                                     [--protists300 PATH]

Definitions follow paper_numbers.py (length model: ln N and (ln N)^2; group adds six-group dummies; matched-length
bins 30-50, 50-70, 70-100, 100-150). Delta-nu is always MPIPI-Recharged (or MPIPI-GG) minus CALVADOS-2.
MPIPI-Recharged nu comes from analysis/mpipi_recharged/recharged_nu.py (bender_common.flory, all production frames, chain
made whole), the same estimator as CALVADOS-2 and MPIPI-GG.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

DEFAULTS = dict(
    bio=str(paths.BIO),
    recharged=str(paths.RECHARGED / 'recharged_300K_nu.csv'),
    gg=str(paths.GG_RESULTS),
    pae_c2=str(paths.PAE / 'calvados_vs_pae_results_v2.tsv'),
    pae_gg=str(paths.PAE / 'mpipi_vs_pae_results_v2.tsv'),
    protists300=str(paths.INPUTS / 'protist_300_cdhit40.fasta'),  # FASTA as simulated
)
ORDER = ['Mammals', 'Fungi', 'Plants', 'Protists', 'Bacteria', 'Viruses']
EUK = ['Mammals', 'Fungi', 'Plants', 'Protists']
BINS = ((30, 50), (50, 70), (70, 100), (100, 151))
FEATS = ['FCR', 'NCPR', 'SCD', 'SHD', 'lambda_avg', 'hydropathy_avg', 'f_aro', 'f_R', 'f_K', 'f_E', 'f_D', 'f_G', 'f_P',
         'f_S', 'f_Q', 'f_N', 'f_H', 'f_A', 'f_V', 'f_I', 'f_L', 'f_M', 'f_T', 'f_C', 'f_F', 'f_W', 'f_Y']


def r2(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - np.var(y - X @ b) / np.var(y)


def length_group(y, n, g):
    L = np.log(n)
    base = r2(y, [L, L ** 2])
    return base, r2(y, [L, L ** 2] + [(g == k).astype(float) for k in ORDER[1:]]) - base


def resid(y, n):
    L = np.log(n)
    X = np.column_stack([np.ones_like(L), L, L ** 2])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def agree(x, y):
    """Pearson r, mean and SD of y - x, 95% limits of agreement."""
    d = y - x
    return (f"r {stats.pearsonr(x, y)[0]:.3f} | rho {stats.spearmanr(x, y)[0]:.3f} | mean {d.mean():+.4f} | "
            f"SD {d.std():.4f} | LoA [{d.mean() - 1.96 * d.std():+.3f}, {d.mean() + 1.96 * d.std():+.3f}] | "
            f"MAE {d.abs().mean():.4f}")


def gaps(nu, n, g):
    out = []
    for lo, hi in BINS:
        s = (n >= lo) & (n < hi)
        m = pd.Series(nu[s]).groupby(g[s]).median()
        out.append(m.max() - m.min())
    return out


def cv_r2(X, y, k=5, seed=0):
    folds = np.random.default_rng(seed).integers(0, k, len(y))
    pred = np.zeros_like(y)
    for f in range(k):
        tr, te = folds != f, folds == f
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ w
    return 1 - np.var(y - pred) / np.var(y)


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    ap.add_argument('--pae-mr', default=None, help='per-sequence MPIPI-Recharged PAE results (uniprot_id, pearson_r, status)')
    ap.add_argument('--features-mr', default=None, help='recharged_300K_features.csv (Re, asphericity, A0, network, contacts)')
    a = ap.parse_args()

    b = pd.read_csv(a.bio, low_memory=False)
    b = b[b['kingdom'].isin(ORDER)].set_index('UniProt_ID')
    mr = pd.read_csv(a.recharged).set_index('UniProt_ID').reindex(b.index)
    new300 = {l[1:].split()[0] for l in open(a.protists300) if l.startswith('>')}
    assert len(new300) == 300, len(new300)
    n, g = b['n_residues'].to_numpy(float), b['kingdom'].to_numpy()
    out = []
    P = lambda k, v: out.append((k, v))

    # ---------------------------------------------------------------- composition
    P('== Composition', '')
    P('sequences', len(b))
    P('organisms (unique taxonomy_id)', b['taxonomy_id'].nunique())
    for k in ORDER:
        s = b[b['kingdom'] == k]
        P(f'  {k}', f"{len(s)} sequences, {s['taxonomy_id'].nunique()} organisms")
    t = b[b.index.isin(new300)]
    P('protist top-up', f"{len(t)} sequences, {t['taxonomy_id'].nunique()} organisms; "
                        f"cluster ids shared with the rest: {len(set(t['cluster_id']) & set(b.loc[~b.index.isin(new300), 'cluster_id']))}")
    P('MPIPI-Recharged rows / missing', f"{mr['nu'].notna().sum()} / {mr['nu'].isna().sum()}")
    P('MPIPI-Recharged max bond (nm)', f"{mr['max_bond_nm'].max():.3f}")
    P('MPIPI-Recharged production frames', f"{int(mr['n_frames'].min())}-{int(mr['n_frames'].max())}")

    # ---------------------------------------------------------------- data quality
    P('== Data quality', '')
    for lab, r2c, bse in (('CALVADOS-2', b['nu_fit_r2'], b['nu_block_se']), ('MPIPI-Recharged', mr['nu_r2'], mr['nu_block_se'])):
        P(f'{lab}: median fit R2 / >=0.99 / >0.95', f"{r2c.median():.4f} / {(r2c >= 0.99).mean():.1%} / {(r2c > 0.95).mean():.1%}")
        P(f'{lab}: count R2 < 0.99', int((r2c < 0.99).sum()))
        P(f'{lab}: R2 < 0.99 by group', ', '.join(f"{k} {v:.1%}" for k, v in (r2c < 0.99).groupby(b['kingdom']).mean().reindex(ORDER).items()))
        P(f'{lab}: median block SE nu', f"{bse.median():.4f}")
    P('CALVADOS-2 median |half1 - half2|', f"{(b['nu_half1'] - b['nu_half2']).abs().median():.3f}")
    P('MPIPI-Recharged median |half1 - half2|', f"{(mr['nu_half1'] - mr['nu_half2']).abs().median():.3f}")
    P('both R2 < 0.99', int(((b['nu_fit_r2'] < 0.99) & (mr['nu_r2'] < 0.99)).sum()))
    P('nu < 0.35: CALVADOS-2 / Recharged', f"{int((b['nu'] < 0.35).sum())} / {int((mr['nu'] < 0.35).sum())}")
    p = b['kingdom'] == 'Protists'
    for lab, s in (('original 763', p & ~b.index.isin(new300)), ('top-up 300', p & b.index.isin(new300))):
        P(f'protists {lab}: C2 R2<0.99 / MR R2<0.99', f"{(b.loc[s, 'nu_fit_r2'] < 0.99).mean():.1%} / {(mr.loc[s, 'nu_r2'] < 0.99).mean():.1%}")

    # ---------------------------------------------------------------- results (CALVADOS-2 and Recharged)
    P('== Results', '')
    med = lambda c: b.groupby('kingdom')[c].median()
    P('median length B / V; eukaryotes', f"{med('n_residues')['Bacteria']:.0f} / {med('n_residues')['Viruses']:.0f}; "
                                         f"{med('n_residues')[EUK].min():.0f}-{med('n_residues')[EUK].max():.0f}")
    P('median Rg C2 B / V (nm)', f"{med('rg')['Bacteria']:.2f} / {med('rg')['Viruses']:.2f}")
    P('median nu C2 by group', ', '.join(f"{k} {v:.3f}" for k, v in med('nu').reindex(ORDER).items()))
    P('median nu MR by group', ', '.join(f"{k} {v:.3f}" for k, v in mr['nu'].groupby(b['kingdom']).median().reindex(ORDER).items()))
    P('Spearman nu vs length C2 / MR', f"{stats.spearmanr(b['nu'], n)[0]:.2f} / {stats.spearmanr(mr['nu'], n)[0]:.2f}")
    P('share <= 150 residues', f"{(n <= 150).mean():.1%}")
    P('matched-length nu gap C2', ' '.join(f'{x:.3f}' for x in gaps(b['nu'].to_numpy(), n, g)))
    P('matched-length nu gap MR', ' '.join(f'{x:.3f}' for x in gaps(mr['nu'].to_numpy(), n, g)))
    s = (n >= 100) & (n <= 150)
    P('100-150 median nu by group C2', ', '.join(f"{k} {v:.3f}" for k, v in b.loc[s].groupby('kingdom')['nu'].median().reindex(ORDER).items()))
    P('100-150 median nu by group MR', ', '.join(f"{k} {v:.3f}" for k, v in mr.loc[s, 'nu'].groupby(b.loc[s, 'kingdom']).median().reindex(ORDER).items()))
    s2 = s & ~(p & ~b.index.isin(new300))
    P('MR 100-150 gap without the original 763 protists', f"{mr.loc[s2, 'nu'].groupby(b.loc[s2, 'kingdom']).median().pipe(lambda x: x.max() - x.min()):.3f}")
    for lab, y in (('nu C2', b['nu']), ('nu MR', mr['nu']), ('Rg C2', b['rg']), ('Rg MR', mr['rg_mean']),
                   ('g_eff C2', b['global_efficiency']), ('assortativity C2', b['degree_assortativity'])):
        x, e = length_group(y.to_numpy(float), n, g)
        P(f'{lab}: length R2 / + group', f"{x:.1%} / {e:.2%}")
    P('share nu > 0.588 C2', ', '.join(f"{k} {v:.1%}" for k, v in (b['nu'] > 0.588).groupby(b['kingdom']).mean().reindex(ORDER).items()))
    P('share nu > 0.588 MR', ', '.join(f"{k} {v:.1%}" for k, v in (mr['nu'] > 0.588).groupby(b['kingdom']).mean().reindex(ORDER).items()))
    P('median FCR by group', f"{med('FCR').min():.2f}-{med('FCR').max():.2f}")
    P('mean stickiness: group medians', f"{med('lambda_avg').min():.2f}-{med('lambda_avg').max():.2f}")
    P('median fragmentation by group', f"{med('fragmentation_index').min():.3f}-{med('fragmentation_index').max():.3f}")
    P('median g_eff by group', ', '.join(f"{k} {v:.3f}" for k, v in med('global_efficiency').reindex(ORDER).items()))
    P('median clustering by group', f"{med('avg_clustering').min():.3f}-{med('avg_clustering').max():.3f}")
    P('median transitivity by group', f"{med('transitivity').min():.3f}-{med('transitivity').max():.3f}")
    P('Spearman assortativity vs length', f"{stats.spearmanr(b['degree_assortativity'], n)[0]:.2f}")
    P('median assortativity by group', ', '.join(f"{k} {v:.3f}" for k, v in med('degree_assortativity').reindex(ORDER).items()))

    # ---------------------------------------------------------------- cross-force-field, all sequences
    P('== CALVADOS-2 vs MPIPI-Recharged (all)', '')
    P('nu', agree(b['nu'], mr['nu']))
    P('nu r by group', ', '.join(f"{k} {stats.pearsonr(b.loc[g == k, 'nu'], mr.loc[g == k, 'nu'])[0]:.3f}" for k in ORDER))
    P('nu r, length removed', f"{stats.pearsonr(resid(b['nu'].to_numpy(), n), resid(mr['nu'].to_numpy(), n))[0]:.3f}")
    d = mr['nu'] - b['nu']
    P('share of var(dnu) from sampling (block SE)', f"{((b['nu_block_se'] ** 2 + mr['nu_block_se'] ** 2).mean() / d.var()):.1%}")
    ok = (b['nu_fit_r2'] >= 0.99) & (mr['nu_r2'] >= 0.99)
    P(f'nu, both fits R2 >= 0.99 (n={int(ok.sum())})', agree(b.loc[ok, 'nu'], mr.loc[ok, 'nu']))
    P('Rg', agree(b['rg'], mr['rg_mean']))
    rel = (mr['rg_mean'] - b['rg']) / b['rg']
    P('Rg relative difference MR vs C2: mean / median', f"{rel.mean():+.1%} / {rel.median():+.1%}")
    P('|dnu| > 0.10', f"{int((d.abs() > 0.1).sum())} ({(d.abs() > 0.1).mean():.1%})")
    for lab, s in (('original 763', p & ~b.index.isin(new300)), ('top-up 300', p & b.index.isin(new300))):
        P(f'protists {lab}: mean / SD dnu', f"{d[s].mean():+.4f} / {d[s].std():.4f}")
    q = (n >= 100) & (n <= 150) & p
    c = q & (d < -0.05)
    P('protists 100-150 with dnu < -0.05', f"{int(c.sum())} of {int(q.sum())}; from the original 763: {int((c & ~b.index.isin(new300)).sum())}")
    P('  their composition vs other protists 100-150 (f_aro, f_R, f_K)',
      ' | '.join(f"{f}: {b.loc[c, f].mean():.3f} vs {b.loc[q & ~c, f].mean():.3f}" for f in ('f_aro', 'f_R', 'f_K')))
    P('  their median nu C2 / MR', f"{b.loc[c, 'nu'].median():.3f} / {mr.loc[c, 'nu'].median():.3f}")
    P('Spearman dnu with f_aro / hydropathy / FCR / lambda', ' / '.join(f"{stats.spearmanr(b[f], d)[0]:+.2f}" for f in ('f_aro', 'hydropathy_avg', 'FCR', 'lambda_avg')))
    X = b[FEATS].copy()
    X['logN'], X['absNCPR'] = np.log(n), b['NCPR'].abs()
    X = ((X - X.mean()) / X.std()).fillna(0).to_numpy()
    P('5-fold CV R2 of dnu from composition, charge and ln N (linear)', f"{cv_r2(X, d.to_numpy()):.3f}")
    G = np.column_stack([(g == k).astype(float) for k in ORDER[1:]])
    comp = r2(d.to_numpy(), [X])
    P('in-sample R2 of dnu: group only / composition / composition + group',
      f"{r2(d.to_numpy(), [G]):.3f} / {comp:.3f} / {r2(d.to_numpy(), [X, G]):.3f}")
    P('mean |dnu| by group (Recharged)', ', '.join(f"{k} {v:.3f}" for k, v in d.abs().groupby(b['kingdom']).mean().reindex(ORDER).items()))
    ra = b['f_R'] + b['f_aro']
    sel = (ra > ra.quantile(0.9)) & (b['f_K'] < b['f_K'].median())
    P('Arg+Aromatic top decile and Lys below median: n / mean dnu / rest', f"{int(sel.sum())} / {d[sel].mean():+.4f} / {d[~sel].mean():+.4f}")

    # composition model with squared and pairwise terms on the main axes (CV), then the contested class by group
    key = [FEATS.index(f) for f in ('f_aro', 'f_R', 'f_K', 'hydropathy_avg', 'f_H', 'FCR')] + [X.shape[1] - 2]  # + ln N
    extra = []
    for i, c1 in enumerate(key):
        extra.append(X[:, c1] ** 2)
        extra += [X[:, c1] * X[:, c2] for c2 in key[i + 1:]]
    Xq = np.column_stack([X] + extra)
    folds = np.random.default_rng(0).integers(0, 5, len(d))
    def cvq(M):
        pred = np.zeros(len(d))
        for f in range(5):
            tr, te = folds != f, folds == f
            A = np.column_stack([np.ones(tr.sum()), M[tr]])
            w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ d.to_numpy()[tr])
            pred[te] = np.column_stack([np.ones(te.sum()), M[te]]) @ w
        return 1 - np.var(d.to_numpy() - pred) / np.var(d.to_numpy()), pred
    rq, pq = cvq(Xq)
    rqg, _ = cvq(np.column_stack([Xq, G]))
    P('5-fold CV R2 of dnu, composition with squared and pairwise terms / + group', f"{rq:.3f} / {rqg:.3f}")
    cls = sel.to_numpy()
    res = d.to_numpy() - pq
    P('contested class (Arg+Aromatic top decile, Lys below median): share by group',
      ', '.join(f"{k} {v:.1%}" for k, v in pd.Series(cls).groupby(g).mean().reindex(ORDER).items()))
    P('  mean dnu inside class by group', ', '.join(f"{k} {v:+.3f}" for k, v in pd.Series(d.to_numpy()[cls]).groupby(g[cls]).mean().reindex(ORDER).items()))
    P('  mean dnu outside class by group', ', '.join(f"{k} {v:+.3f}" for k, v in pd.Series(d.to_numpy()[~cls]).groupby(g[~cls]).mean().reindex(ORDER).items()))
    P('  composition-adjusted residual inside class by group', ', '.join(f"{k} {v:+.3f}" for k, v in pd.Series(res[cls]).groupby(g[cls]).mean().reindex(ORDER).items()))
    P('  Kruskal-Wallis p, residual inside class by group', f"{stats.kruskal(*[res[cls & (g == k)] for k in ORDER]).pvalue:.1e}")
    eta = lambda v, gg: 1 - np.var(v - pd.Series(v).groupby(gg).transform('mean').to_numpy()) / np.var(v)
    P('  share of within-class variance explained by group: raw / residual', f"{eta(d.to_numpy()[cls], g[cls]):.3f} / {eta(res[cls], g[cls]):.3f}")
    P('share |dnu| > 0.10 by group', ', '.join(f"{k} {v:.1%}" for k, v in (d.abs() > 0.1).groupby(b['kingdom']).mean().reindex(ORDER).items()))

    # ---------------------------------------------------------------- other properties, CALVADOS-2 vs MPIPI-Recharged
    if a.features_mr:
        fm = pd.read_csv(a.features_mr).set_index('UniProt_ID').reindex(b.index)
        P('== Other properties, CALVADOS-2 vs MPIPI-Recharged (all)', '')
        P('features rows ok', int((fm['status'] == 'ok').sum()))
        for c in ('ree', 'delta', 'A0', 'fragmentation_index', 'global_efficiency', 'avg_clustering', 'transitivity',
                  'degree_assortativity', 'pi_pi_contact_freq', 'cation_pi_contact_freq'):
            x, y = b[c].to_numpy(float), fm[c].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if c in ('pi_pi_contact_freq', 'cation_pi_contact_freq'):
                pairs = 'n_aromatic_pairs' if c.startswith('pi') else 'n_cation_pi_pairs'
                m &= b[pairs].fillna(0).to_numpy() > 0
            byg = [stats.pearsonr(x[m & (g == k)], y[m & (g == k)])[0] for k in ORDER]
            P(f'  {c}', f"n {m.sum()} | median C2 {np.median(x[m]):.3f} MR {np.median(y[m]):.3f} | r {stats.pearsonr(x[m], y[m])[0]:.2f} | "
                        f"rho {stats.spearmanr(x[m], y[m])[0]:.2f} | r length removed {stats.pearsonr(resid(x[m], n[m]), resid(y[m], n[m]))[0]:.2f} | "
                        f"r by group {min(byg):.2f}-{max(byg):.2f}")
        for c, pairs in (('pi_pi_contact_freq', 'n_aromatic_pairs'), ('cation_pi_contact_freq', 'n_cation_pi_pairs')):
            m = (b[pairs].fillna(0) > 0) & fm[c].notna() & b[c].notna()
            ratio = np.log((fm.loc[m, c] + 1e-4) / (b.loc[m, c] + 1e-4))
            P(f'  Spearman dnu vs log contact ratio MR/C2, {c}', f"{stats.spearmanr(d[m], ratio)[0]:+.2f} (n {int(m.sum())})")
        for lab, y in (('g_eff MR', fm['global_efficiency']), ('assortativity MR', fm['degree_assortativity'])):
            x_, e_ = length_group(y.to_numpy(float), n, g)
            P(f'{lab}: length R2 / + group', f"{x_:.1%} / {e_:.2%}")

    # ---------------------------------------------------------------- three force fields on the MPIPI-GG subset
    P('== MPIPI-GG subset (957)', '')
    gg = pd.read_csv(a.gg).set_index('UniProt_ID')
    gg = gg[(gg['status'] == 'ok') & gg['in_paper_set'] & gg.index.isin(b.index)]
    ids = gg.index
    P('n', len(ids))
    P('C2 vs GG (GG - C2)', agree(b.loc[ids, 'nu'], gg['nu_mpipi']))
    P('C2 vs MR (MR - C2)', agree(b.loc[ids, 'nu'], mr.loc[ids, 'nu']))
    P('GG vs MR (MR - GG)', agree(gg['nu_mpipi'], mr.loc[ids, 'nu']))
    dg = gg['nu_mpipi'] - b.loc[ids, 'nu']
    P('GG: share of var(dnu) from sampling', f"{((b.loc[ids, 'nu_block_se'] ** 2 + gg['nu_block_se_mpipi'] ** 2).mean() / dg.var()):.0%}")
    P('GG nu r by group', ', '.join(f"{k} {stats.pearsonr(b.loc[ids][b.loc[ids, 'kingdom'] == k]['nu'], gg.loc[b.loc[ids, 'kingdom'] == k, 'nu_mpipi'])[0]:.3f}" for k in ORDER))
    P('GG per-group n', ', '.join(f"{k} {v}" for k, v in b.loc[ids, 'kingdom'].value_counts().reindex(ORDER).items()))

    # ---------------------------------------------------------------- PAE
    P('== AF2 PAE', '')
    pc = pd.read_csv(a.pae_c2, sep='\t').set_index('uniprot_id')
    pc = pc[(pc['status'] == 'ok') & pc.index.isin(b.index)]
    P('CALVADOS-2 sequences with PAE', f"{len(pc)} (top-up protists without PAE: {len(new300 - set(pc.index))})")
    pr = pc['pearson_r']
    kk = b.loc[pc.index, 'kingdom']
    P('CALVADOS-2 median r; by group', f"{pr.median():.3f}; " + ', '.join(f"{k} {v:.3f}" for k, v in pr.groupby(kk).median().reindex(ORDER).items()))
    pg = pd.read_csv(a.pae_gg, sep='\t').set_index('uniprot_id')
    pg = pg[(pg['status'] == 'ok') & pg.index.isin(ids) & pg.index.isin(pc.index)]
    pr2, pg2 = pc.loc[pg.index, 'pearson_r'], pg['pearson_r']
    vir = b.loc[pg.index, 'kingdom'] == 'Viruses'
    P('GG subset with PAE: n / C2 median / GG median', f"{len(pg)} / {pr2.median():.3f} / {pg2.median():.3f}")
    P('  viruses: n / C2 / GG', f"{int(vir.sum())} / {pr2[vir].median():.3f} / {pg2[vir].median():.3f}")
    dd = pr2 - pg2
    P('  C2 - GG: mean / LoA / share GG better / |diff| < 0.04',
      f"{dd.mean():+.3f} / [{dd.mean() - 1.96 * dd.std():+.3f}, {dd.mean() + 1.96 * dd.std():+.3f}] / {(dd < 0).mean():.0%} / {(dd.abs() < 0.04).mean():.0%}")
    if a.pae_mr:
        pm = pd.read_csv(a.pae_mr, sep='\t' if a.pae_mr.endswith('.tsv') else ',').set_index('uniprot_id')
        pm = pm[(pm['status'] == 'ok') & pm.index.isin(pc.index)]
        c2, m2 = pc.loc[pm.index, 'pearson_r'], pm['pearson_r']
        km = b.loc[pm.index, 'kingdom']
        P('Recharged with PAE (of the CALVADOS-2 PAE set)', f"{len(pm)} of {len(pc)}")
        P('  median r C2 / MR on that set', f"{c2.median():.3f} / {m2.median():.3f}")
        P('  MR median r by group', ', '.join(f"{k} {v:.3f}" for k, v in m2.groupby(km).median().reindex(ORDER).items()))
        P('  C2 median r by group (same set)', ', '.join(f"{k} {v:.3f}" for k, v in c2.groupby(km).median().reindex(ORDER).items()))
        dm = c2 - m2
        P('  C2 - MR: mean / LoA / share MR better / |diff| < 0.04',
          f"{dm.mean():+.3f} / [{dm.mean() - 1.96 * dm.std():+.3f}, {dm.mean() + 1.96 * dm.std():+.3f}] / {(dm < 0).mean():.0%} / {(dm.abs() < 0.04).mean():.0%}")
        P('  Pearson r between C2 and MR per-sequence PAE r', f"{stats.pearsonr(c2, m2)[0]:.3f}")

    w = max(len(k) for k, _ in out)
    for k, v in out:
        print(k if v == '' else f'{k:<{w}}  {v}')


if __name__ == '__main__':
    main()
