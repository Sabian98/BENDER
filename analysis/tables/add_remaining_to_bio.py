"""
add_remaining_to_bio.py — complete BENDER_BIO.csv to all 11,704 BENDER sequences (2026-09-27).

1. Appends the 1,432 sequences that had no row (non-mammal animals, archaea and 53 sequences filed under
   other curation groups). Their ensemble and network values come from traj_analysis_calvados_v2.py
   run --graph (same protocol as every other row), written to Analysis_300K_v2_graph_remaining.
   kingdom = curation group (the BENDER_300K folder, as for existing rows); organism and taxonomy_id
   from BENDER_300K_lineages.csv. Columns that cannot be reproduced for them (names, clusters, qcdpred,
   old contact scores, *_orig) are left blank.
2. Adds `lineage` (UniProt lineage: six major lineages or Archaea) for every row.
3. Recomputes every sequence descriptor for every row with one definition per column
   (bender_common charge descriptors; sequence_descriptors for composition, residue sets,
   CALVADOS-2 lambda, hydropathy, kappa, SHD).

Dry run by default; --apply copies BENDER_BIO.csv to a backup beside it (never overwritten) and
writes the table with provenance.

--descriptors-only (the mode used, 2026-09-27): the dataset paper keeps the 10,272 sequences of the six
main groups, so no rows are added; only step 3 runs (backup BENDER_BIO_before_descriptors.csv).
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
import sequence_descriptors as sd  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

BIO = str(paths.BIO)
BACKUP = BIO.replace('.csv', '_before_remaining.csv')
LINEAGES = str(paths.LINEAGES)
T_REST = str(paths.WORK / 'calvados2_300K_remaining' / 'scalar_features_v2.csv')
NETWORK = ['fragmentation_index', 'global_efficiency', 'avg_clustering', 'transitivity', 'degree_assortativity']


def new_rows(bio, lin):
    t = pd.read_csv(T_REST)
    t = t[t['status'] == 'ok'].drop_duplicates('UniProt_ID', keep='last')
    missing = lin[~lin['uniprot_id'].isin(bio['UniProt_ID'])]
    got = t[t['UniProt_ID'].isin(missing['uniprot_id'])].set_index('UniProt_ID')
    lacking = sorted(set(missing['uniprot_id']) - set(got.index))
    m = missing.set_index('uniprot_id').loc[got.index]
    new = pd.DataFrame(index=got.index)
    new['UniProt_ID'] = got.index
    new['organism'] = m['organism'].values
    new['taxonomy_id'] = m['taxid'].values
    new['kingdom'] = m['folder'].values
    new['sequence'] = got['sequence_top'].values
    new['length'] = got['n_residues'].values
    new['n_residues'] = got['n_residues'].values
    same = {'rg': 'rg', 'rg_err': 'rg_err', 'ree': 'ree', 'ree_err': 'ree_err', 'delta': 'delta',
            'delta_err': 'delta_err', 'nu': 'nu', 'A0': 'A0', 'nu_err': 'nu_block_se', 'nu_fit_r2': 'nu_r2'}
    for dst, src in same.items():
        new[dst] = got[src].values
    for c in ['nu_fit_err', 'nu_block_se', 'nu_legacy', 'nu_legacy_r2', 'nu_half1', 'nu_half2',
              'n_frames_dcd', 'n_frames_production', 'n_frames_used', 'n_frames_graph']:
        new[c] = got[c].values
    for c in NETWORK:
        new[c] = got[c].values
        new[f'{c}_std'] = got[f'{c}_std'].values
    new['A0_err'] = np.nan
    new['source'] = bio['source'].mode()[0]
    new['network_features_method'] = bio['network_features_method'].mode()[0]
    new['nu_r2_pass'] = new['nu_fit_r2'] >= bc.NU_R2_MIN
    new['zone'] = new['nu'].map(bc.zone)
    new['expected_frames_dcd'] = new['n_residues'].map(bc.expected_frames)
    new['truncated_trajectory'] = (new['n_frames_dcd'] < new['expected_frames_dcd']).astype(bool)
    new['analysis_ok'] = new['nu_r2_pass'] & ~new['truncated_trajectory']
    return new.reset_index(drop=True), lacking


def recompute_descriptors(df):
    rows = []
    for s in df['sequence']:
        d = bc.charge_descriptors(s)
        q = bc.residue_charges(s)
        d['f_pos'], d['f_neg'] = float((q > 0).mean()), float((q < 0).mean())
        d.update(sd.descriptors(s))
        rows.append(d)
    desc = pd.DataFrame(rows, index=df.index)
    for c in desc.columns:
        df[c] = desc[c]
    return df, list(desc.columns)


def main(apply, descriptors_only=False):
    bio = pd.read_csv(BIO, low_memory=False)
    if descriptors_only:
        return harmonise(bio, apply)
    lin = pd.read_csv(LINEAGES)
    new, lacking = new_rows(bio, lin)
    print(f'BENDER_BIO: {len(bio)} rows; adding {len(new)}; still without a trajectory result: {len(lacking)} {lacking[:10]}')
    out = pd.concat([bio, new.reindex(columns=bio.columns)], ignore_index=True)
    out = out[bio.columns]
    out['lineage'] = out['UniProt_ID'].map(lin.set_index('uniprot_id')['lineage'])
    assert out['lineage'].notna().all() and out['UniProt_ID'].is_unique
    assert (out['sequence'].str.len() == out['n_residues']).all()
    before = out[['kappa', 'SHD', 'lambda_avg', 'f_hydrophobic', 'f_polar', 'f_disorder_promoting',
                  'f_order_promoting']].iloc[:len(bio)].copy()
    out, cols = recompute_descriptors(out)
    print(f'recomputed for all {len(out)} rows: {", ".join(cols)}')
    for c in before.columns:
        a, b = before[c].to_numpy(float), out[c].iloc[:len(bio)].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        print(f'  {c:22s} existing rows: r(old, new) {np.corrcoef(a[ok], b[ok])[0, 1]:+.3f}, '
              f'max |diff| {np.abs(a[ok] - b[ok]).max():.3f}')
    print(out.groupby('lineage').size().to_string())
    print('flags among added rows: fit R2 < 0.99', int((~new['nu_r2_pass']).sum()),
          '| truncated', int(new['truncated_trajectory'].sum()))
    if not apply:
        print('dry run: nothing written (pass --apply)')
        return
    if not os.path.exists(BACKUP):
        shutil.copy2(BIO, BACKUP)
    out.to_csv(BIO, index=False)
    bc.write_provenance(BIO, __file__, {
        'step': 'added the 1,432 sequences without a row; lineage column; all descriptors recomputed',
        'descriptors': 'sequence_descriptors.py (Dunker sets, CALVADOS-2 lambda, KD hydropathy, '
                       'Das-Pappu kappa, Zheng SHD) + bender_common charge descriptors',
        'backup': os.path.basename(BACKUP)}, [BACKUP, T_REST, LINEAGES])
    print(f'wrote {BIO}: {len(out)} rows (backup {os.path.basename(BACKUP)})')


def harmonise(bio, apply):
    backup = BIO.replace('.csv', '_before_descriptors.csv')
    old = bio.copy()
    out, cols = recompute_descriptors(bio.copy())
    print(f'{len(out)} rows; recomputed: {", ".join(cols)}')
    for c in ['FCR', 'NCPR', 'SCD', 'f_R', 'hydropathy_avg', 'lambda_avg', 'kappa', 'SHD', 'f_hydrophobic',
              'f_polar', 'f_disorder_promoting', 'f_order_promoting', 'f_pi']:
        a, b = old[c].to_numpy(float), out[c].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        changed = int((~np.isclose(a, b, atol=1e-6, equal_nan=True)).sum())
        print(f'  {c:22s} rows changed {changed:5d}' + (f' | r(old, new) {np.corrcoef(a[ok], b[ok])[0, 1]:+.3f}' if changed else ''))
    print('kappa NaN (no charged residue):', int(out['kappa'].isna().sum()))
    if not apply:
        print('dry run: nothing written (pass --apply)')
        return
    if not os.path.exists(backup):
        shutil.copy2(BIO, backup)
    out.to_csv(BIO, index=False)
    bc.write_provenance(BIO, __file__, {
        'step': 'all sequence descriptors recomputed with one definition per column (no rows added)',
        'descriptors': 'sequence_descriptors.py (Dunker sets, CALVADOS-2 lambda, KD hydropathy, '
                       'Das-Pappu kappa, Zheng SHD) + bender_common charge descriptors',
        'backup': os.path.basename(backup)}, [backup])
    print(f'wrote {BIO} (backup {os.path.basename(backup)})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--descriptors-only', action='store_true')
    a = ap.parse_args()
    main(a.apply, a.descriptors_only)
