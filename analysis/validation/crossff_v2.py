"""
crossff_v2.py — Task 12 downstream: CALVADOS-2 vs MPIPI-GG on the same sequences (v2 features).

Replaces MPIPI/compare_contact_classes.py (retired). Inputs are the v2 tables, so both force fields
use identical rules: production frames, unwrapped MPIPI coordinates, primary ν, per-pair contact
frequency (Cα–Cα < 0.8 nm, |i−j| ≥ 3; no composition normalization).

Outputs (in --out):
  crossff_nu_agreement_v2.csv     ν agreement (Pearson, Spearman, mean offset, MAE)
  crossff_direction_v2.csv        taxon × contact class: rank-biserial (compact vs expanded, each FF's own
                                  ν), sign agreement, per-sequence cross-FF Spearman of the class frequency
  crossff_binned_v2.csv           per taxon × ν bin means for SCD, NCPR, FCR, Lys- and Arg-mediated
                                  frequency, both force fields (Fig. 7 data)

Usage: python3 crossff_v2.py --cal Analysis_300K_v2/scalar_features_v2.csv \
           --mpipi mpipi_results_v2/mpipi_results_v2.csv --out DIR
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared modules (bender_common, ...)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # this folder (also flat on HPC)
import bender_common as bc  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

BIO_CSV = str(paths.BIO)
CLASSES = {
    'Lys-mediated':   ['Lys_Tyr', 'Lys_Phe', 'Lys_Trp', 'Lys_Glu', 'Lys_Asp', 'Lys_Lys', 'Lys_Ser'],
    'Charge–charge':  ['Arg_Glu', 'Arg_Asp', 'Lys_Glu', 'Lys_Asp'],
    'Cation–π':       ['Arg_Tyr', 'Arg_Phe', 'Arg_Trp', 'Lys_Tyr', 'Lys_Phe', 'Lys_Trp'],
    'Arg-mediated':   ['Arg_Tyr', 'Arg_Phe', 'Arg_Trp', 'Arg_Glu', 'Arg_Asp', 'Arg_Arg', 'Arg_Ser'],
}


def class_freq(df):
    out = pd.DataFrame(index=df.index)
    for c, pairs in CLASSES.items():
        cols = [f'freq_{p}' for p in pairs if f'freq_{p}' in df.columns]
        out[c] = df[cols].mean(axis=1)
    return out


def rank_biserial(a, b):
    """P(compact > expanded) − P(compact < expanded); positive = contact loss on expansion."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, len(a), len(b)
    u = stats.mannwhitneyu(a, b, alternative='two-sided')
    return 2 * u.statistic / (len(a) * len(b)) - 1, u.pvalue, len(a), len(b)


def main(cal_csv, mpipi_csv, out, six_lineages=False):
    os.makedirs(out, exist_ok=True)
    cal = pd.read_csv(cal_csv)
    cal = cal[cal['status'] == 'ok'].drop_duplicates('UniProt_ID', keep='last').set_index('UniProt_ID')
    mp = pd.read_csv(mpipi_csv)
    mp = mp[(mp['status'] == 'ok') & (mp['in_paper_set'])].set_index('UniProt_ID')
    bio = pd.read_csv(BIO_CSV, usecols=['UniProt_ID', 'kingdom', 'sequence', 'nu']).set_index('UniProt_ID')
    ids = mp.index.intersection(cal.index)
    if six_lineages:
        n0 = len(ids)
        ids = ids[ids.astype(str).isin(bc.six_lineage_ids())]
        print(f'six lineages only: {len(ids)} of {n0} paired sequences kept')
    complete = cal.loc[ids, 'n_frames_dcd'] >= cal.loc[ids, 'n_residues'].map(bc.expected_frames)
    ok_fit = (cal.loc[ids, 'nu_r2'] >= bc.NU_R2_MIN) & (mp.loc[ids, 'nu_r2_mpipi'] >= bc.NU_R2_MIN) & complete
    n_pair, n_fit = len(ids), int(ok_fit.sum())
    ids = ids[ok_fit.values]
    kingdom = bio['kingdom'].reindex(ids)
    print(f'{n_pair} sequences in both force fields; {n_fit} with R² ≥ {bc.NU_R2_MIN} in both and a complete CALVADOS-2 trajectory')

    # ── ν agreement ─────────────────────────────────────────────────────
    nc, nm = cal.loc[ids, 'nu'], mp.loc[ids, 'nu_mpipi']
    agree = [{'scope': 'all', 'n': len(ids), 'pearson_r': stats.pearsonr(nc, nm)[0],
              'spearman_rho': stats.spearmanr(nc, nm)[0], 'mean_offset_mpipi_minus_cal': float((nm - nc).mean()),
              'mae': float((nm - nc).abs().mean())}]
    for k in bc.SIX_LINEAGES:
        m = kingdom == k
        agree.append({'scope': k, 'n': int(m.sum()), 'pearson_r': stats.pearsonr(nc[m], nm[m])[0],
                      'spearman_rho': stats.spearmanr(nc[m], nm[m])[0],
                      'mean_offset_mpipi_minus_cal': float((nm[m] - nc[m]).mean()),
                      'mae': float((nm[m] - nc[m]).abs().mean())})
    agree = pd.DataFrame(agree)
    agree.to_csv(os.path.join(out, 'crossff_nu_agreement_v2.csv'), index=False)

    # ── direction agreement per taxon × class ────────────────────────────
    cf_c, cf_m = class_freq(cal.loc[ids]), class_freq(mp.loc[ids])
    rows = []
    for k in bc.SIX_LINEAGES:
        m = (kingdom == k).values
        zc = nc[m].map(bc.zone); zm = nm[m].map(bc.zone)
        for c in CLASSES:
            rc, pc, nc1, nc2 = rank_biserial(cf_c[c][m][zc == 'compact'], cf_c[c][m][zc == 'expanded'])
            rm, pm, nm1, nm2 = rank_biserial(cf_m[c][m][zm == 'compact'], cf_m[c][m][zm == 'expanded'])
            both = cf_c[c][m].notna() & cf_m[c][m].notna()
            # threshold-free direction: Spearman ρ of class frequency against each force field's own ν
            okc, okm = cf_c[c][m].notna(), cf_m[c][m].notna()
            sc = stats.spearmanr(nc[m][okc], cf_c[c][m][okc]) if okc.sum() > 10 else (np.nan, np.nan)
            sm = stats.spearmanr(nm[m][okm], cf_m[c][m][okm]) if okm.sum() > 10 else (np.nan, np.nan)
            rows.append({'lineage': k, 'contact_class': c,
                         'rho_nu_calvados': sc[0], 'p_rho_calvados': sc[1], 'n_cal': int(okc.sum()),
                         'rho_nu_mpipi': sm[0], 'p_rho_mpipi': sm[1], 'n_mpipi': int(okm.sum()),
                         'rho_sign_agree': bool(np.sign(sc[0]) == np.sign(sm[0]))
                         if np.isfinite(sc[0]) and np.isfinite(sm[0]) else np.nan,
                         'rrb_calvados': rc, 'p_calvados': pc, 'n_compact_cal': nc1, 'n_expanded_cal': nc2,
                         'rrb_mpipi': rm, 'p_mpipi': pm, 'n_compact_mpipi': nm1, 'n_expanded_mpipi': nm2,
                         'sign_agree': bool(np.sign(rc) == np.sign(rm)) if np.isfinite(rc) and np.isfinite(rm) else np.nan,
                         'per_sequence_spearman_cal_vs_mpipi': stats.spearmanr(cf_c[c][m][both], cf_m[c][m][both])[0]
                         if both.sum() > 5 else np.nan})
    d = pd.DataFrame(rows)
    d['p_holm_calvados'] = bc.holm(d['p_calvados'].values)
    d['p_holm_mpipi'] = bc.holm(d['p_mpipi'].values)
    d['p_holm_rho_calvados'] = bc.holm(d['p_rho_calvados'].values)
    d['p_holm_rho_mpipi'] = bc.holm(d['p_rho_mpipi'].values)
    d.to_csv(os.path.join(out, 'crossff_direction_v2.csv'), index=False)

    # ── binned trajectories (Fig. 7 data) ────────────────────────────────
    desc = pd.DataFrame([bc.charge_descriptors(s) for s in bio['sequence'].reindex(ids)], index=ids)
    parts = []
    for ff, nu, cf in (('CALVADOS-2', nc, cf_c), ('MPIPI-GG', nm, cf_m)):
        t = pd.concat([desc, cf[['Lys-mediated', 'Arg-mediated']]], axis=1)
        t['nu'] = nu; t['kingdom'] = kingdom; t['nu_bin'] = bc.nu_bin(nu)
        g = t.groupby(['kingdom', 'nu_bin'])
        agg = g[['SCD', 'NCPR', 'FCR', 'Lys-mediated', 'Arg-mediated']].agg(['mean', 'sem'])
        agg.columns = [f'{a}_{b}' for a, b in agg.columns]
        agg['n'] = g.size()
        agg['force_field'] = ff
        parts.append(agg.reset_index())
    pd.concat(parts).to_csv(os.path.join(out, 'crossff_binned_v2.csv'), index=False)

    n_ok = int(d['sign_agree'].sum()); n_tot = int(d['sign_agree'].notna().sum())
    print(agree.round(3).to_string(index=False))
    with pd.option_context('display.width', 200):
        print(d[['lineage', 'contact_class', 'rho_nu_calvados', 'rho_nu_mpipi', 'rho_sign_agree',
                 'p_holm_rho_calvados', 'p_holm_rho_mpipi', 'rrb_calvados', 'rrb_mpipi', 'sign_agree',
                 'per_sequence_spearman_cal_vs_mpipi']].round(3).to_string(index=False))
    n_rho = int(d['rho_sign_agree'].sum()); n_rho_tot = int(d['rho_sign_agree'].notna().sum())
    both_sig = d[(d['p_holm_rho_calvados'] < 0.05) & (d['p_holm_rho_mpipi'] < 0.05)]
    print(f'direction agreement, Spearman ρ vs own ν (primary): {n_rho}/{n_rho_tot}; '
          f'among cells significant in both force fields: {int(both_sig["rho_sign_agree"].sum())}/{len(both_sig)}')
    print(f'direction agreement, compact vs expanded rank-biserial (secondary): {n_ok}/{n_tot} defined cells')
    bc.write_provenance(os.path.join(out, 'crossff_direction_v2.csv'), __file__,
                        {'classes': CLASSES, 'zones': [bc.ZONE_COMPACT_MAX, bc.ZONE_EXPANDED_MIN],
                         'NU_R2_MIN': bc.NU_R2_MIN, 'n_sequences': len(ids), 'six_lineages': six_lineages}, [cal_csv, mpipi_csv, BIO_CSV])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cal', required=True)
    ap.add_argument('--mpipi', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--six_lineages', action='store_true',
                    help='keep only sequences whose BENDER_BIO kingdom is one of the six lineages')
    a = ap.parse_args()
    main(a.cal, a.mpipi, a.out, a.six_lineages)
