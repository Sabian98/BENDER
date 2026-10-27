"""
pae_separation_check.py: is the agreement of simulated distance fluctuations with AlphaFold2 PAE more than a shared
dependence on sequence separation? (BENDER JCTC, 2026-10-21)

The PAE metric of pae_v2.py is the per-sequence Pearson r between the SD of every inter-residue distance (pairs
i < j, 1,000 production frames) and PAE[i, j]. Both grow with the separation s = j - i, so part of r could come from
separation alone. Per sequence and force field this script reports
  r_raw       the pae_v2 metric (checked against the stored values)
  r_sep_only  r between PAE and the separation profile of the SDs (the mean SD at each s), i.e. what separation
              alone gives
  r_within    r between the SD and PAE residuals after removing each one's mean at each separation (the
              within-separation agreement, free of the shared trend)
  pae_sep_r2  share of the PAE variance explained by separation alone
on a stratified random sample of the 9,982 sequences with a PAE matrix (200 per group, seed 20261021), for
CALVADOS-2 (BENDER_300K) and MPIPI-Recharged (MPIPI Recharged/300K), with pae_v2's frames and PAE loader.

    python3 pae_separation_check.py [--per-group 200] [--workers 6]
Output: BENDER-BIO/PAE/pae_analysis/tsv/pae_separation_check.tsv (+ provenance)
"""
import argparse
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bender_common as bc  # noqa: E402
import pae_v2  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

if platform.system() == 'Darwin':
    mp.set_start_method('fork', force=True)

OUT = f'{pae_v2.TSV_DIR}/pae_separation_check.tsv'
RECHARGED_ROOT = str(paths.TRAJ_RECHARGED)
ORDER = ['Mammals', 'Fungi', 'Plants', 'Protists', 'Bacteria', 'Viruses']


def metrics(xyz, pae):
    n = min(xyz.shape[1], pae.shape[0])
    ii, jj, m1, m2 = bc._pair_moments(xyz[:, :n], 1)
    f = xyz.shape[0]
    sd = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0) * f / max(f - 1, 1))
    p = pae[:n, :n][ii, jj]
    s = jj - ii
    cnt = np.bincount(s)
    sd_s = np.bincount(s, sd) / np.maximum(cnt, 1)
    p_s = np.bincount(s, p) / np.maximum(cnt, 1)
    rs, rp = sd - sd_s[s], p - p_s[s]
    out = dict(r_raw=stats.pearsonr(sd, p)[0], r_sep_only=stats.pearsonr(sd_s[s], p)[0],
               pae_sep_r2=1 - np.var(rp) / np.var(p), n_pairs=len(s))
    out['r_within'] = stats.pearsonr(rs, rp)[0] if np.std(rs) > 0 and np.std(rp) > 0 else np.nan
    return out


def run_one(args):
    uid, group, cal_dir = args
    rows = []
    pae = pae_v2.load_pae(uid)
    if pae is None:
        return [dict(uniprot_id=uid, group=group, ff=ff, status='no_pae') for ff in ('CALVADOS-2', 'MPIPI-Recharged')]
    for ff, loc in (('CALVADOS-2', cal_dir), ('MPIPI-Recharged', f'{RECHARGED_ROOT}/{uid}')):
        try:
            xyz, _, _ = bc.load_calvados(f'{loc}/{uid}.dcd', f'{loc}/top.pdb', k=pae_v2.K_FRAMES)
            bond = float(np.linalg.norm(np.diff(xyz, axis=1), axis=2).max())
            if bond > 0.5:
                rows.append(dict(uniprot_id=uid, group=group, ff=ff, status=f'broken chain {bond:.2f} nm'))
                continue
            rows.append(dict(uniprot_id=uid, group=group, ff=ff, status='ok', n_residues=xyz.shape[1],
                             **metrics(xyz, pae)))
        except Exception as e:
            rows.append(dict(uniprot_id=uid, group=group, ff=ff, status=f'failed: {e!r}'[:200]))
    return rows


def main(per_group, workers):
    bio = pd.read_csv(pae_v2.BIO_CSV, usecols=['UniProt_ID', 'kingdom'])
    bio = bio[bio['kingdom'].isin(ORDER)].set_index('UniProt_ID')
    stored = pd.read_csv(f'{pae_v2.TSV_DIR}/calvados_vs_pae_results_v2.tsv', sep='\t').set_index('uniprot_id')
    ok = stored.index[(stored['status'] == 'ok') & stored.index.isin(bio.index)]
    lin = pd.read_csv(pae_v2.LINEAGES).set_index('uniprot_id')
    rng = np.random.default_rng(20261021)
    pick = []
    for g in ORDER:
        ids = [u for u in ok if bio.at[u, 'kingdom'] == g]
        pick += [(u, g) for u in rng.choice(ids, min(per_group, len(ids)), replace=False)]
    jobs = [(u, g, f"{pae_v2.CAL_ROOT}/{lin.at[u, 'folder']}/{u}") for u, g in pick]
    print(f'{len(jobs)} sequences, {workers} workers', flush=True)
    t0, rows = time.time(), []
    with mp.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows += r
            if i % 100 == 0:
                print(f'[{i}/{len(jobs)}] {(time.time() - t0) / 60:.1f} min', flush=True)
    df = pd.DataFrame(rows)
    c2 = df[(df['ff'] == 'CALVADOS-2') & (df['status'] == 'ok')].set_index('uniprot_id')
    chk = (c2['r_raw'] - stored.loc[c2.index, 'pearson_r']).abs().max()
    print(f'check: CALVADOS-2 r_raw vs stored pae_v2 values, max |diff| {chk:.2e}')
    df.to_csv(OUT, sep='\t', index=False)
    bc.write_provenance(OUT, __file__, {'per_group': per_group, 'seed': 20261021, 'K_FRAMES': pae_v2.K_FRAMES,
                                        'pairs': 'all i < j, as pae_v2', 'max_abs_diff_vs_stored_r': float(chk)},
                        [pae_v2.PAE_DIR, pae_v2.BIO_CSV, f'{pae_v2.TSV_DIR}/calvados_vs_pae_results_v2.tsv'])
    okd = df[df['status'] == 'ok']
    print(df['status'].value_counts().to_string())
    summ = okd.groupby(['ff', 'group'])[['r_raw', 'r_sep_only', 'r_within', 'pae_sep_r2']].median().round(3)
    print(summ.to_string())
    print(okd.groupby('ff')[['r_raw', 'r_sep_only', 'r_within', 'pae_sep_r2']].median().round(3).to_string())
    w = okd.pivot(index='uniprot_id', columns='ff', values='r_within').dropna()
    print(f"r_within, CALVADOS-2 vs Mpipi-Recharged: Pearson {stats.pearsonr(w.iloc[:, 0], w.iloc[:, 1])[0]:.3f}, "
          f"median difference {np.median(w['CALVADOS-2'] - w['MPIPI-Recharged']):+.3f}, n {len(w)}")
    print(f'wrote {OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-group', type=int, default=200)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    main(a.per_group, a.workers)
