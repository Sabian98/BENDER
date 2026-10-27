"""
update_canonical_csvs.py — write v2 results into the canonical BENDER-BIO tables.

User decision (2026-09-27): update the canonical CSVs in place; primary ν goes in `nu`.
Every file that is overwritten is first copied to <name>_v1<ext> beside it (never overwritten if
that backup already exists). Derived v1 tables with no v2 counterpart are renamed to *_v1 so that
nothing stale keeps a canonical name. Dry run by default; pass --apply to write.

Targets
  BENDER_BIO.csv                                ensemble columns ← Task 1 v2; FCR/NCPR/SCD/f_pos/f_neg
                                                recomputed from sequence (residue charges only)
  MPIPI/mpipi_results/mpipi_results_raw.csv     Task 12 v2 (unwrapped, all frames, primary ν); eff_* dropped
  MPIPI/mpipi_results/mpipi_aggregated.csv      re-binned
  MPIPI/mpipi_results/mpipi_nu_comparison.csv   v2 ν in both force fields
  ARG:LYS/substitution experiment/results/delta_nu_full.csv   one row per sequence × variant (v1 had 4
                                                identical rows each), variant ν from Task 2 v2

Usage: python3 update_canonical_csvs.py [--apply] [--only bio,mpipi,substitution]

The Contact Results tables it used to write (contact_results_raw/aggregated.csv) were exact copies of the
freq_*/npairs_* columns of Analysis_300K_v2_graph and were recycled on 2026-09-30; that step was removed.
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared modules (bender_common, ...)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # this folder (also flat on HPC)
import bender_common as bc  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

ROOT = str(paths.BIO.parent)
BIO = f'{ROOT}/BENDER_BIO.csv'
T1 = str(paths.WORK / 'calvados2_300K' / 'scalar_features_v2.csv')
MP2 = f'{ROOT}/MPIPI/mpipi_results_v2/mpipi_results_v2.csv'
SUB = f'{ROOT}/ARG:LYS/substitution experiment'
T2 = f'{SUB}/results_v2/variant_features_v2.csv'
MPR = f'{ROOT}/MPIPI/mpipi_results'

APPLY = False
LOG = []


def v1_name(path):
    stem, ext = os.path.splitext(path)
    return f'{stem}_v1{ext}'


def backup(path):
    dst = v1_name(path)
    if os.path.exists(dst):
        LOG.append(f'backup exists, kept: {dst}')
        return
    LOG.append(f'backup: {os.path.basename(path)} -> {os.path.basename(dst)}')
    if APPLY:
        shutil.copy2(path, dst)


def retire(path):
    """Stale derived table/folder with no v2 counterpart: rename to *_v1."""
    if not os.path.exists(path):
        return
    dst = v1_name(path) if os.path.isfile(path) else path.rstrip('/') + '_v1'
    LOG.append(f'retire (stale v1 output): {os.path.relpath(path, ROOT)} -> {os.path.basename(dst)}')
    if APPLY and not os.path.exists(dst):
        os.rename(path, dst)


def write(df, path, inputs, params):
    old = pd.read_csv(path, nrows=0).columns.tolist() if os.path.exists(path) else []
    dropped = [c for c in old if c not in df.columns]
    added = [c for c in df.columns if c not in old]
    LOG.append(f'write {os.path.relpath(path, ROOT)}: {len(df)} rows, {len(df.columns)} cols'
               f'{"; dropped " + str(len(dropped)) + " (" + ", ".join(dropped[:6]) + ("…" if len(dropped) > 6 else "") + ")" if dropped else ""}'
               f'{"; added " + ", ".join(added[:8]) + ("…" if len(added) > 8 else "") if added else ""}')
    if APPLY:
        if os.path.exists(path):
            backup(path)
        df.to_csv(path, index=False)
        bc.write_provenance(path, __file__, params, inputs)
    elif os.path.exists(path):
        backup(path)


def _t1():
    t = pd.read_csv(T1)
    t = t[t['status'] == 'ok'].drop_duplicates('UniProt_ID', keep='last').set_index('UniProt_ID')
    return t


def _desc(seqs):
    rows = []
    for s in seqs:
        d = bc.charge_descriptors(s)
        q = bc.residue_charges(s)
        d['f_pos'] = float((q > 0).mean()); d['f_neg'] = float((q < 0).mean())
        rows.append(d)
    return pd.DataFrame(rows, index=seqs.index)


# ══════════════════════════════════════════════════════════════════════
def update_bio():
    bio = pd.read_csv(BIO)
    t1 = _t1()
    idx = bio['UniProt_ID']
    has = idx.isin(t1.index)
    LOG.append(f'BENDER_BIO: {len(bio)} rows, {int(has.sum())} with re-extracted ensemble values'
               + (f'; {int((~has).sum())} WITHOUT a result — check before applying' if (~has).any() else ''))
    new = bio.copy()
    m = {'nu': 'nu', 'A0': 'A0', 'rg': 'rg', 'rg_err': 'rg_err', 'ree': 'ree', 'ree_err': 'ree_err',
         'delta': 'delta', 'delta_err': 'delta_err', 'nu_err': 'nu_block_se', 'nu_fit_r2': 'nu_r2'}
    for dst, src in m.items():
        vals = idx.map(t1[src])
        new.loc[has, dst] = vals[has]
    new.loc[has, 'A0_err'] = np.nan            # ν uncertainty is the block SE; no curve-fit A0 error
    for c in ['nu_fit_err', 'nu_block_se', 'nu_legacy', 'nu_legacy_r2', 'nu_half1', 'nu_half2',
              'n_frames_dcd', 'n_frames_production', 'n_frames_used']:
        new[c] = idx.map(t1[c])
    new['nu_r2_pass'] = new['nu_fit_r2'] >= bc.NU_R2_MIN
    new['zone'] = new['nu'].map(bc.zone)
    exp = new['n_residues'].map(bc.expected_frames)
    new['expected_frames_dcd'] = exp
    new['truncated_trajectory'] = (new['n_frames_dcd'] < exp).astype(bool)
    new['analysis_ok'] = new['nu_r2_pass'] & ~new['truncated_trajectory']
    new['network_features_note'] = 'contact graphs with |i-j| >= 2; recomputation with |i-j| >= 3 pending'
    LOG.append(f'  truncated trajectories: {int(new["truncated_trajectory"].sum())}')
    LOG.append(f'  R² < {bc.NU_R2_MIN}: {int((~new["nu_r2_pass"]).sum())}')
    d = _desc(bio.set_index('UniProt_ID')['sequence'])
    for c in ['FCR', 'NCPR', 'SCD', 'f_pos', 'f_neg']:
        vals = idx.map(d[c])
        new[c] = vals.values
    write(new, BIO, [BIO, T1], {'nu': 'primary (user decision 2026-09-27)', 'nu_err': '5-block SE',
                                'charges': 'residue-only, no termini'})
    return new


def _aggregate(df, cols, nu_col='nu'):
    df = df.copy(); df['nu_bin'] = bc.nu_bin(df[nu_col])
    rows = []
    for (k, nb), g in df.groupby(['kingdom', 'nu_bin']):
        r = {'kingdom': k, 'nu_bin': nb, 'n_sequences': len(g)}
        for c in cols:
            v = g[c].dropna()
            r[f'mean_{c}'] = v.mean() if len(v) else np.nan
            r[f'se_{c}'] = v.sem() if len(v) > 1 else np.nan
            if c.startswith('raw_'):
                r[f'n_{c}'] = len(v)
        rows.append(r)
    return pd.DataFrame(rows)


def update_mpipi(bio_new):
    old = pd.read_csv(f'{MPR}/mpipi_results_raw.csv')
    v2 = pd.read_csv(MP2)
    v2 = v2[v2['status'] == 'ok'].set_index('UniProt_ID')
    ids = old['UniProt_ID']
    miss = int((~ids.isin(v2.index)).sum())
    b = bio_new.set_index('UniProt_ID')
    out = old[['UniProt_ID', 'kingdom', 'n_residues', 'sequence']].copy()
    out['rg_mean_A'] = ids.map(v2['rg_mean_A'])
    out['rg_std_A'] = ids.map(10 * v2['rg_err_nm'] * np.sqrt(v2['n_frames']))
    out['rg_mean_nm'] = out['rg_mean_A'] / 10
    out['rg_nm_mpipi'] = ids.map(v2['rg_nm']); out['rg_err_nm_mpipi'] = ids.map(v2['rg_err_nm'])
    out['ree_nm_mpipi'] = ids.map(v2['ree_nm']); out['ree_err_nm_mpipi'] = ids.map(v2['ree_err_nm'])
    out['delta_mpipi'] = ids.map(v2['delta']); out['delta_err_mpipi'] = np.nan
    out['n_prod_frames'] = ids.map(v2['n_frames'])
    d = _desc(old.set_index('UniProt_ID')['sequence'])
    for c in ['SCD', 'NCPR', 'FCR']:
        out[c] = ids.map(d[c]).values
    for c in ['f_R', 'f_K', 'f_aro']:
        out[c] = old[c]
    out['nu_mpipi'] = ids.map(v2['nu_mpipi'])
    out['nu_err_mpipi'] = ids.map(v2['nu_block_se_mpipi'])
    out['A0_mpipi'] = ids.map(v2['A0_mpipi']); out['A0_err_mpipi'] = np.nan
    out['nu_fit_r2'] = ids.map(v2['nu_r2_mpipi']); out['n_nu_frames'] = ids.map(v2['n_frames'])
    out['nu_legacy_mpipi'] = ids.map(v2['nu_legacy_mpipi'])
    out['nu_bin'] = bc.nu_bin(out['nu_mpipi'])
    out['status'] = np.where(out['nu_mpipi'].notna(), 'ok', 'missing')
    out['nu_method'] = 'RMS distance per pair, mean per separation, fit s >= 5; unwrapped; all production frames'
    out['contact_source'] = 'per-pair contact frequency (Ca-Ca < 0.8 nm, |i-j| >= 3); no composition normalisation'
    out['nu_half1'] = ids.map(v2['nu_half1']); out['nu_half2'] = ids.map(v2['nu_half2'])
    out['max_bond_raw_nm'] = ids.map(v2['max_bond_raw_nm']); out['max_bond_nm'] = ids.map(v2['max_bond_nm'])
    for c in [c for c in v2.columns if c.startswith('freq_') and not c.endswith(('_h1', '_h2'))]:
        out['raw_' + c[5:]] = ids.map(v2[c])
    LOG.append(f'MPIPI: {len(old)} rows, {miss} missing from v2 run')
    write(out, f'{MPR}/mpipi_results_raw.csv', [MP2, f'{MPR}/mpipi_results_raw.csv'],
          {'frames': 'all (production only)', 'unwrapped': True, 'nu': 'primary'})
    agg = _aggregate(out.rename(columns={'nu_mpipi': 'nu_use'}),
                     ['rg_mean_nm', 'n_residues', 'nu_use', 'nu_err_mpipi', 'A0_mpipi', 'nu_fit_r2', 'n_nu_frames',
                      'SCD', 'NCPR', 'FCR', 'f_R', 'f_K', 'f_aro'] +
                     [c for c in out.columns if c.startswith('raw_')], nu_col='nu_use')
    agg = agg.rename(columns=lambda c: c.replace('nu_use', 'nu_mpipi'))
    write(agg, f'{MPR}/mpipi_aggregated.csv', [f'{MPR}/mpipi_results_raw.csv'], {'bin_width': bc.NU_BIN_WIDTH})
    t1 = _t1()
    comp = pd.DataFrame({'UniProt_ID': ids, 'kingdom': old['kingdom'], 'nu_mpipi': out['nu_mpipi'],
                         'nu_calvados': ids.map(t1['nu'])})
    write(comp, f'{MPR}/mpipi_nu_comparison.csv', [MP2, T1], {'nu': 'primary, both force fields'})
    retire(f'{MPR}/mpipi_contact_summary.csv')
    retire(f'{MPR}/contact_class_comparison')


def update_substitution():
    path = f'{SUB}/results/delta_nu_full.csv'
    old = pd.read_csv(path)
    one = old.drop_duplicates(['UniProt_ID', 'variant'])
    LOG.append(f'substitution: v1 {len(old)} rows = {len(one)} unique sequence × variant')
    v = pd.read_csv(T2)
    v = v[v['status'] == 'ok'].set_index(['UniProt_ID', 'variant'])
    t1 = _t1()
    out = one[['protein_name', 'UniProt_ID', 'variant', 'kingdom', 'n_arg', 'n_lys', 'n_aro', 'length']].copy()
    key = list(zip(out['UniProt_ID'], out['variant']))
    out['nu_mutant'] = [v['nu'].get(k, np.nan) for k in key]
    out['nu_err_mutant'] = [v['nu_block_se'].get(k, np.nan) for k in key]
    out['nu_fit_r2'] = [v['nu_r2'].get(k, np.nan) for k in key]
    out['nu_legacy_mutant'] = [v['nu_legacy'].get(k, np.nan) for k in key]
    out['n_frames'] = [v['n_frames_dcd'].get(k, np.nan) for k in key]
    out['A0'] = [v['A0'].get(k, np.nan) for k in key]; out['A0_err'] = np.nan
    out['n_residues'] = out['length']
    out['nu_wildtype'] = out['UniProt_ID'].map(t1['nu'])
    out['nu_wt'] = out['nu_wildtype']
    out['nu_err_wildtype'] = out['UniProt_ID'].map(t1['nu_block_se'])
    out['delta_nu'] = out['nu_mutant'] - out['nu_wildtype']
    out['delta_nu_err'] = np.sqrt(out['nu_err_mutant'] ** 2 + out['nu_err_wildtype'] ** 2)
    z = out['nu_wildtype'].map(bc.zone)
    out['zone'] = z
    out['nu_zone'] = z.map({'compact': 'below_threshold', 'transition': 'near_threshold',
                            'expanded': 'above_threshold'})
    out['status'] = np.where(out['nu_mutant'].notna(), 'OK', 'missing_v2')
    write(out, path, [T2, T1, path], {'note': 'Δν uses the Task 1 wild type (different run); prefer the '
                                               'within-sequence contrasts in results_v2/contrasts'})
    for stale in ['paired_comparison.csv', 'variant_contacts.csv', 'scalar_features.csv', 'shuffle_contacts.csv',
                  'shuffle_contacts_analyzed.csv', 'zone_variant_summary.csv', 'predicted_ratios.csv']:
        retire(f'{SUB}/results/{stale}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--only', default='bio,mpipi,substitution')
    a = ap.parse_args()
    APPLY = a.apply
    only = set(a.only.split(','))
    bio_new = update_bio() if 'bio' in only else pd.read_csv(BIO)
    if 'mpipi' in only:
        update_mpipi(bio_new)
    if 'substitution' in only:
        update_substitution()
    print(('APPLIED' if APPLY else 'DRY RUN (nothing written)') + '\n' + '\n'.join(LOG))
