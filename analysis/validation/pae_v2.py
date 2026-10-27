"""
pae_v2.py — agreement of CALVADOS-2 and MPIPI-GG ensembles with AlphaFold2 PAE (2026-09-27).

Same metric as BENDER-BIO/PAE/paper_analysis.py: per sequence, the Pearson r between the standard
deviation of each inter-residue distance (all pairs i < j) over the trajectory and the AF2 PAE entry
PAE[i, j]. Fixes two problems of the original:
  * MPIPI-GG distances were computed on coordinates wrapped into the periodic box (978 of the 992
    paper-set chains cross the box), inflating the distance SDs;
  * CALVADOS-2 distances included the equilibration half of each trajectory.
Frames follow bender_common: CALVADOS-2 production half, MPIPI-GG all frames unwrapped, 1,000 evenly
spaced frames for both.

Usage: python3 pae_v2.py calvados|mpipi [--workers 6] [--ids IDS.txt]
Outputs: BENDER-BIO/PAE/pae_analysis/tsv/{calvados,mpipi}_vs_pae_results_v2.tsv (+ provenance)
"""

import argparse
import gzip
import json
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared modules (bender_common, ...)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # this folder (also flat on HPC)
import bender_common as bc  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

if platform.system() == 'Darwin':
    mp.set_start_method('fork', force=True)

PAE_DIR = str(paths.WORK / 'pae_files')   # AlphaFold2 PAE JSON files from the AlphaFold Database
TSV_DIR = str(paths.PAE)
BIO_CSV = str(paths.BIO)
CAL_ROOT = str(paths.TRAJ_C2)
MPIPI_ROOT = str(paths.TRAJ_GG)
K_FRAMES = 1000


def load_pae(uid):
    """PAE matrix from <uid>_pae.json (optionally gzipped), as in paper_analysis.load_pae_matrix."""
    path = f'{PAE_DIR}/{uid}_pae.json'
    if not os.path.exists(path):
        return None
    raw = open(path, 'rb').read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    data = json.loads(raw)
    if isinstance(data, list):
        data = data[0]
    return np.array(data['predicted_aligned_error'], dtype=np.float64) if 'predicted_aligned_error' in data else None


def correlate(uid, xyz, pae):
    n = min(xyz.shape[1], pae.shape[0])
    ii, jj, m1, m2 = bc._pair_moments(xyz[:, :n], 1)
    f = xyz.shape[0]
    sd = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0) * f / max(f - 1, 1))
    p = pae[:n, :n][ii, jj]
    r, pval = stats.pearsonr(sd, p)
    return {'uniprot_id': uid, 'pearson_r': float(r), 'p_value': float(pval), 'n_residues': int(xyz.shape[1]),
            'n_pae': int(pae.shape[0]), 'n_frames': int(f), 'n_pairs': int(len(ii)),
            'mean_sd': float(sd.mean()), 'mean_pae': float(p.mean()),
            'status': 'ok' if xyz.shape[1] == pae.shape[0] else 'length_mismatch'}   # never compare a truncated chain


def run_one(args):
    ff, uid, loc = args
    try:
        pae = load_pae(uid)
        if pae is None:
            return {'uniprot_id': uid, 'status': 'no_pae'}
        if ff == 'calvados':
            xyz, _, _ = bc.load_calvados(f'{loc}/{uid}.dcd', f'{loc}/top.pdb', k=K_FRAMES)
        else:
            seq = bc.read_mpipi_sequence(f'{loc}/seq.dat')
            xyz, info = bc.load_mpipi(f'{loc}/traj.xtc', n_res=len(seq), k=K_FRAMES)
            assert info['max_bond_nm'] < 0.5, f"broken chain after unwrapping ({info['max_bond_nm']:.2f} nm)"
        return correlate(uid, xyz, pae)
    except Exception as e:
        return {'uniprot_id': uid, 'status': f'failed: {e!r}'[:200]}


def main(ff, workers, ids_file):
    bio = pd.read_csv(BIO_CSV, usecols=['UniProt_ID', 'kingdom'])
    if ff == 'calvados':
        uids = bio['UniProt_ID'].tolist()
        locs = {u: f"{CAL_ROOT}/{g}/{u}" for u, g in zip(bio['UniProt_ID'], bio['kingdom'])}   # <group>/<ID>
    else:
        m = pd.read_csv(str(paths.GG_RESULTS),
                        usecols=['UniProt_ID', 'status', 'in_paper_set'])
        uids = m.loc[(m['status'] == 'ok') & m['in_paper_set'], 'UniProt_ID'].tolist()
        locs = {u: f'{MPIPI_ROOT}/{u}' for u in uids}
    if ids_file:
        keep = {l.strip() for l in open(ids_file) if l.strip()}
        uids = [u for u in uids if u in keep]
    print(f'{ff}: {len(uids)} sequences, {workers} workers', flush=True)
    t0, rows = time.time(), []
    with mp.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, [(ff, u, locs[u]) for u in uids]), 1):
            rows.append(r)
            if i % 250 == 0:
                print(f'[{i}/{len(uids)}] {(time.time() - t0) / 60:.1f} min', flush=True)
    df = pd.DataFrame(rows)
    out = f'{TSV_DIR}/{ff}_vs_pae_results_v2.tsv'
    if ids_file and os.path.exists(out):
        # subset run: replace only these rows, keeping a dated copy of the previous table next to it
        import datetime, shutil
        shutil.copy2(out, out + '.before_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        old = pd.read_csv(out, sep='\t')
        df = pd.concat([old[~old['uniprot_id'].isin(set(df['uniprot_id']))], df], ignore_index=True)
    df.to_csv(out, sep='\t', index=False)
    bc.write_provenance(out, __file__, {'ff': ff, 'K_FRAMES': K_FRAMES,
                                        'frames': 'production half' if ff == 'calvados' else 'all, unwrapped',
                                        'sd': 'ddof=1 over frames, all pairs i<j'}, [PAE_DIR, BIO_CSV])
    ok = df[df['status'] == 'ok']
    print(f'wrote {out}: {len(ok)} ok of {len(df)}; median r {ok["pearson_r"].median():.3f}; '
          f'median mean_sd {ok["mean_sd"].median():.2f} nm')
    print(df['status'].value_counts().head().to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ff', choices=['calvados', 'mpipi'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--ids', default=None)
    a = ap.parse_args()
    main(a.ff, a.workers, a.ids)
