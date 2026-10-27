"""
mpipi_analysis_v2.py — Task 12 of the reanalysis plan (MPIPI-GG re-extraction).

For every run directory (seq.dat + traj.xtc):
  * all frames are production (the dump is delayed until equilibration ends) — nothing discarded
  * coordinates are unwrapped from minimum-image bond vectors before any geometry
  * geometry, primary ν (+R², block SE), legacy ν — identical estimators to CALVADOS (bender_common)
  * per-pair contact frequencies (Cα–Cα < 0.8 nm, |i−j| ≥ 3; no composition normalization)
  * ν and contacts on each half of production (convergence)

Output: <out>/mpipi_results_v2.csv (+ provenance).  v1 descriptor columns (SCD, NCPR, FCR, f_R,
f_K, f_aro, kingdom) and the paper-set membership come from mpipi_results_raw.csv.

Usage: python3 mpipi_analysis_v2.py --out DIR [--workers 6]
"""

import argparse
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared modules (bender_common, ...)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # this folder (also flat on HPC)
import bender_common as bc  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

if platform.system() == 'Darwin':
    mp.set_start_method('fork', force=True)

RUNS_DIR = str(paths.TRAJ_GG)
V1_RAW   = str(paths.WORK / 'mpipi_gg' / 'mpipi_results_raw.csv')
PAIRS    = list(dict.fromkeys(bc.CONTACT_PAIRS + [('R', 'K')]))


def analyse(uid):
    d = os.path.join(RUNS_DIR, uid)
    seq_dat, xtc = os.path.join(d, 'seq.dat'), os.path.join(d, 'traj.xtc')
    if not (os.path.exists(seq_dat) and os.path.exists(xtc)):
        return {'UniProt_ID': uid, 'status': 'missing'}
    try:
        t0 = time.time()
        seq = bc.read_mpipi_sequence(seq_dat)
        xyz, info = bc.load_mpipi(xtc, n_res=len(seq))
        if info['n_frames_total'] < 100:
            return {'UniProt_ID': uid, 'status': f"too_few_frames:{info['n_frames_total']}"}
        row = {'UniProt_ID': uid, 'n_residues': len(seq), 'sequence': seq,
               'n_frames': info['n_frames_total'], 'box_nm': info['box_nm'],
               'max_bond_raw_nm': info['max_bond_raw_nm'], 'max_bond_nm': info['max_bond_nm']}
        g = bc.geometry(xyz)
        row.update({'rg_nm': g['rg'], 'rg_err_nm': g['rg_err'], 'ree_nm': g['ree'],
                    'ree_err_nm': g['ree_err'], 'delta': g['delta'], 'rg_mean_A': 10 * g['rg']})
        f = bc.flory(xyz)
        row.update({'nu_mpipi': f['nu'], 'A0_mpipi': f['A0'], 'nu_fit_err_mpipi': f['nu_fit_err'],
                    'nu_r2_mpipi': f['nu_r2'], 'nu_block_se_mpipi': f['nu_block_se'],
                    'nu_legacy_mpipi': f['nu_legacy'], 'nu_legacy_r2_mpipi': f['nu_legacy_r2']})
        h = len(xyz) // 2
        row['nu_half1'] = bc.nu_primary_from_xyz(xyz[:h])['nu']
        row['nu_half2'] = bc.nu_primary_from_xyz(xyz[h:])['nu']
        F = bc.contact_frequency_map(xyz)
        row.update(bc.pair_type_table(F, seq, PAIRS))
        iu = np.triu_indices(len(seq), k=bc.CONTACT_MIN_SEP)
        row['contacts_per_frame'] = float(F[iu].sum())
        for tag, Fh in (('h1', bc.contact_frequency_map(xyz[:h])), ('h2', bc.contact_frequency_map(xyz[h:]))):
            for a, b in PAIRS:
                row[f'freq_{bc.AA1_TO_NAME[a]}_{bc.AA1_TO_NAME[b]}_{tag}'] = bc.pair_type_frequency(Fh, seq, a, b)[0]
        row['seconds'] = round(time.time() - t0, 1)
        row['status'] = 'ok'
        return row
    except Exception as e:
        return {'UniProt_ID': uid, 'status': f'failed: {e!r}'[:300]}


def main(out, workers):
    os.makedirs(out, exist_ok=True)
    uids = sorted(u for u in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, u)))
    print(f'{len(uids)} run directories', flush=True)
    with mp.Pool(workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(analyse, uids), 1):
            rows.append(r)
            if i % 100 == 0:
                print(f'[{i}/{len(uids)}]', flush=True)
    df = pd.DataFrame(rows)
    v1 = pd.read_csv(V1_RAW)
    carry = ['UniProt_ID', 'kingdom', 'SCD', 'NCPR', 'FCR', 'f_R', 'f_K', 'f_aro']
    v1c = v1[carry].copy()
    v1c['in_paper_set'] = True          # the 992 sequences analyzed in the paper
    df = df.merge(v1c, on='UniProt_ID', how='left')
    df['in_paper_set'] = df['in_paper_set'].fillna(False).astype(bool)
    front = ['UniProt_ID', 'kingdom', 'status', 'in_paper_set', 'n_residues', 'n_frames', 'nu_mpipi',
             'nu_r2_mpipi', 'nu_block_se_mpipi', 'nu_legacy_mpipi', 'rg_mean_A']
    df = df[[c for c in front if c in df.columns] + [c for c in df.columns if c not in front]]
    path = os.path.join(out, 'mpipi_results_v2.csv')
    df.to_csv(path, index=False)
    bc.write_provenance(path, __file__, {'frames': 'all (xtc is production only)', 'unwrap': True,
                                         'pairs': [''.join(p) for p in PAIRS], 'workers': workers},
                        [RUNS_DIR, V1_RAW])
    ok = df[df['status'] == 'ok']
    print(f'ok {len(ok)} / {len(df)}; in paper set: {int(ok["in_paper_set"].sum())}')
    print(df['status'].value_counts().head(10))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    main(a.out, a.workers)
