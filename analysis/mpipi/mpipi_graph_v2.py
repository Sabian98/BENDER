"""
mpipi_graph_v2.py — contact-network features for the MPIPI-GG trajectories, with the CALVADOS-2
protocol of traj_analysis_calvados_v2.py --graph (constants and feature code imported from there):

  * frames: every traj.xtc frame is production (the dump starts after equilibration); the chain
    is unwrapped from minimum-image bond vectors, then K_FRAMES frames are taken evenly and
    GRAPH_FRAMES of those evenly, exactly as for CALVADOS-2
  * graphs: Cα–Cα < 0.8 nm with |i−j| >= 3 (bender_common.adjacency)
  * features: graph_features.graph_features_from_adj (fragmentation index, global efficiency,
    average clustering, transitivity, degree assortativity; frame means and s.d.)

Usage
  python3 mpipi_graph_v2.py run   [--workers 6] [--limit N]   -> MPIPI Results/mpipi_graph_features_v2.csv
  python3 mpipi_graph_v2.py merge                             -> adds the network columns to mpipi_results_v2.csv
                                                                  (original copied beside it first)
"""

import argparse
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared modules (bender_common, ...)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # this folder (also flat on HPC)
import bender_common as bc  # noqa: E402
from graph_features import graph_features_from_adj  # noqa: E402  (v1 feature definitions)
from traj_analysis_calvados_v2 import K_FRAMES, GRAPH_FRAMES, _append  # noqa: E402  (same protocol)
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

if platform.system() == 'Darwin':
    mp.set_start_method('fork', force=True)

RUNS_DIR = str(paths.TRAJ_GG)
RESULTS = str(paths.DATA / 'mpipi_gg')
OUT_CSV = os.path.join(RESULTS, 'mpipi_graph_features_v2.csv')
MAIN_CSV = os.path.join(RESULTS, 'mpipi_results_v2.csv')
FEATURES = ['fragmentation_index', 'global_efficiency', 'avg_clustering', 'transitivity', 'degree_assortativity']
METHOD = (f'contact graphs (Ca-Ca < {bc.CONTACT_CUTOFF_NM} nm, |i-j| >= {bc.CONTACT_MIN_SEP}) on '
          f'{GRAPH_FRAMES} evenly spaced production frames of unwrapped coordinates; frame means and s.d.')


def analyse(uid):
    t0 = time.time()
    d = os.path.join(RUNS_DIR, uid)
    seq = bc.read_mpipi_sequence(os.path.join(d, 'seq.dat'))
    xyz, info = bc.load_mpipi(os.path.join(d, 'traj.xtc'), n_res=len(seq), k=K_FRAMES)
    xg = xyz[bc.even_indices(len(xyz), GRAPH_FRAMES)]
    feats, _ = graph_features_from_adj(bc.adjacency(xg), seq)
    row = {'UniProt_ID': uid, 'n_residues': len(seq), 'n_frames_xtc': info['n_frames_total'],
           'n_frames_graph': len(xg), 'max_bond_nm': round(bc.max_bond(xg), 4)}
    row.update({k: feats[k] for k in FEATURES})
    row.update({f'{k}_std': feats[f'{k}_std'] for k in FEATURES})
    row['seconds'] = round(time.time() - t0, 1)
    row['status'] = 'ok'
    return row


def _worker(uid):
    try:
        return analyse(uid)
    except Exception as e:  # keep going; record the failure
        return {'UniProt_ID': uid, 'status': f'failed: {e!r}'[:300], 'trace': traceback.format_exc()[-500:]}


def run(workers=6, limit=None):
    uids = sorted(u for u in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, u))
                  and os.path.exists(os.path.join(RUNS_DIR, u, 'traj.xtc'))
                  and os.path.exists(os.path.join(RUNS_DIR, u, 'seq.dat')))
    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV, usecols=['UniProt_ID', 'status'])
        done = set(prev.loc[prev['status'] == 'ok', 'UniProt_ID'].astype(str))
    todo = [u for u in uids if u not in done][:limit]
    print(f'{len(uids)} runs with traj.xtc and seq.dat, {len(done)} done, {len(todo)} to run '
          f'with {workers} workers', flush=True)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_worker, todo), 1):
            _append(OUT_CSV, row)
            if i % 50 == 0 or row.get('status') != 'ok':
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] {row.get('status')}:{row['UniProt_ID']}  ({el / 60:.1f} min, "
                      f"~{el / i * (len(todo) - i) / 60:.0f} min left)", flush=True)
    bc.write_provenance(OUT_CSV, __file__, {'K_FRAMES': K_FRAMES, 'GRAPH_FRAMES': GRAPH_FRAMES,
                                            'contact_cutoff_nm': bc.CONTACT_CUTOFF_NM,
                                            'contact_min_sep': bc.CONTACT_MIN_SEP, 'unwrap': True,
                                            'frames': 'all xtc frames are production', 'workers': workers},
                        [RUNS_DIR])
    df = pd.read_csv(OUT_CSV)
    print(df['status'].value_counts().head())


def merge():
    g = pd.read_csv(OUT_CSV)
    g = g[g['status'] == 'ok'].drop_duplicates('UniProt_ID', keep='last')
    assert (g['max_bond_nm'] < 0.5).all(), 'broken chain after unwrapping'
    main = pd.read_csv(MAIN_CSV)
    backup = MAIN_CSV.replace('.csv', '_before_network.csv')
    if not os.path.exists(backup):
        shutil.copy2(MAIN_CSV, backup)
    cols = FEATURES + [f'{k}_std' for k in FEATURES] + ['n_frames_graph']
    main = main.drop(columns=[c for c in cols + ['network_features_method'] if c in main.columns])
    main = main.merge(g[['UniProt_ID'] + cols], on='UniProt_ID', how='left')
    main['network_features_method'] = np.where(main['global_efficiency'].notna(), METHOD, '')
    main.to_csv(MAIN_CSV, index=False)
    bc.write_provenance(MAIN_CSV, __file__, {'step': 'network features merged', 'method': METHOD},
                        [backup, OUT_CSV])
    ok = main[(main['status'] == 'ok')]
    print(f'mpipi_results_v2.csv: {len(main)} rows; network features for {int(ok["global_efficiency"].notna().sum())} '
          f'of {len(ok)} ok runs ({int(ok.loc[ok["in_paper_set"], "global_efficiency"].notna().sum())} of '
          f'{int(ok["in_paper_set"].sum())} in the paper set); backup: {os.path.basename(backup)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'merge'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    run(a.workers, a.limit) if a.mode == 'run' else merge()
