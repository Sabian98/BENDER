"""
traj_analysis_calvados_v2.py — Task 1 (+ Task 9 inputs) of the reanalysis plan.

One pass per wild-type CALVADOS-2 trajectory, using bender_common:
  * production = second half of the DCD, K frames sampled evenly across it
  * geometry (Rg, Ree, asphericity; unweighted)
  * primary ν (RMS per pair → mean per s → fit s ≥ 5), R², curve-fit SE, 5-block SE
  * legacy ν (v1 estimator) as nu_legacy, with R²
  * convergence (Task 9): primary ν and per-pair contact frequencies on each half of production
  * per-pair contact frequencies (Cα–Cα < 0.8 nm, |i−j| ≥ 3) for the 23 v1 pair types
  * optional (--graph): v1 graph features on GRAPH_FRAMES frames, with |i−j| ≥ 3 graphs

Outputs (never overwrites v1):
  <out>/scalar_features_v2.csv          one row per sequence (appended, resumable)
  <out>/npz/<UID>.npz                   contact-frequency map (float16) + RMS-vs-s profile
  <out>/scalar_features_v2.csv.provenance.json
Then `build` merges the table into BENDER_BIO_v2.csv (v1 columns kept; new columns added).

Usage
  python3 traj_analysis_calvados_v2.py run   --out DIR [--ids IDS.txt] [--workers 4] [--graph]
  python3 traj_analysis_calvados_v2.py build --out DIR
"""

import argparse
import os
import sys
import time
import traceback
import multiprocessing as mp
import platform

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bender_common as bc  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

if platform.system() == 'Darwin':
    mp.set_start_method('fork', force=True)

BASE_DIR   = str(paths.TRAJ_C2)
BIO_CSV    = str(paths.BIO)
K_FRAMES   = 1000      # evenly spaced production frames for ν, geometry and contacts
GRAPH_FRAMES = 50

PARAMS = dict(K_FRAMES=K_FRAMES, GRAPH_FRAMES=GRAPH_FRAMES, production='frames[n//2:]',
              subsample='even stride max(1, n_prod//K)', nu_primary='RMS per pair, mean per s, fit s>=5',
              nu_legacy='mean distance per pair, all pairs s>=5', blocks=bc.N_BLOCKS,
              contact_cutoff_nm=bc.CONTACT_CUTOFF_NM, contact_min_sep=bc.CONTACT_MIN_SEP,
              convergence='ν and contacts on first vs second half of production (contiguous)')


def index_trajectories(base_dir=BASE_DIR):
    """uid -> directory, preferring lineage folders over 'Other' (which duplicates them)."""
    idx = {}
    groups = sorted(g for g in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, g)))
    groups = [g for g in groups if g != 'Other'] + (['Other'] if 'Other' in groups else [])
    for g in groups:
        for uid in os.listdir(os.path.join(base_dir, g)):
            d = os.path.join(base_dir, g, uid)
            if uid not in idx and os.path.isdir(d):
                idx[uid] = d
    return idx


def analyse(uid, d, npz_dir, do_graph=False):
    dcd, top = os.path.join(d, f'{uid}.dcd'), os.path.join(d, 'top.pdb')
    if not os.path.exists(dcd) or not os.path.exists(top):
        return {'UniProt_ID': uid, 'status': 'missing'}
    t0 = time.time()
    xyz, seq, info = bc.load_calvados(dcd, top, k=K_FRAMES)
    return analyse_xyz(uid, xyz, seq, info, npz_dir, os.path.basename(os.path.dirname(d)), do_graph, t0)


def analyse_xyz(uid, xyz, seq, info, npz_dir, folder, do_graph=False, t0=None):
    """Everything after loading (shared with restart-piece concatenation in extract_nu_mutants_v2)."""
    t0 = time.time() if t0 is None else t0
    row = {'UniProt_ID': uid, 'folder': folder, 'n_residues': len(seq),
           'sequence_top': seq, 'n_frames_dcd': info['n_frames_total'],
           'n_frames_production': info['n_frames_production'], 'n_frames_used': info['n_frames_used'],
           'stride': info['stride'], 'max_bond_nm': bc.max_bond(xyz)}
    row.update(bc.geometry(xyz))
    row.update(bc.flory(xyz))
    h = len(xyz) // 2
    row['nu_half1'] = bc.nu_primary_from_xyz(xyz[:h])['nu']
    row['nu_half2'] = bc.nu_primary_from_xyz(xyz[h:])['nu']

    F = bc.contact_frequency_map(xyz)
    row.update(bc.pair_type_table(F, seq))
    F1, F2 = bc.contact_frequency_map(xyz[:h]), bc.contact_frequency_map(xyz[h:])
    for tag, Fh in (('h1', F1), ('h2', F2)):
        for a, b in bc.CONTACT_PAIRS:
            f, _ = bc.pair_type_frequency(Fh, seq, a, b)
            row[f'freq_{bc.AA1_TO_NAME[a]}_{bc.AA1_TO_NAME[b]}_{tag}'] = f
    iu = np.triu_indices(len(seq), k=bc.CONTACT_MIN_SEP)
    row['contacts_per_frame'] = float(F[iu].sum())

    # RMS-vs-separation profile (for re-fitting without re-reading trajectories)
    ii, jj, mean_d, msq = bc._pair_moments(xyz, 1)
    sep = jj - ii
    s_vals = np.arange(1, len(seq))
    rms_s = np.array([np.sqrt(msq[sep == s]).mean() for s in s_vals])
    mean_s = np.array([mean_d[sep == s].mean() for s in s_vals])
    np.savez_compressed(os.path.join(npz_dir, f'{uid}.npz'), contact_freq=F.astype(np.float16),
                        s=s_vals, rms_per_s=rms_s, mean_per_s=mean_s, sequence=np.array(list(seq)))

    if do_graph:
        from graph_features import graph_features_from_adj  # v1 feature definitions
        xg = xyz[bc.even_indices(len(xyz), GRAPH_FRAMES)]
        seq_graph, res_graph = graph_features_from_adj(bc.adjacency(xg), seq)
        # per-residue graph features (betweenness, degree, ... and per-amino-acid CV arrays)
        res_dir = os.path.join(os.path.dirname(npz_dir), 'npz_residue')
        os.makedirs(res_dir, exist_ok=True)
        np.savez_compressed(os.path.join(res_dir, f'{uid}_residue_features.npz'), **res_graph)
        row.update({f'{k}': v for k, v in seq_graph.items()})
        row['n_frames_graph'] = len(xg)
    row['seconds'] = round(time.time() - t0, 1)
    row['status'] = 'ok'
    return row


def _worker(args):
    uid, d, npz_dir, do_graph = args
    try:
        return analyse(uid, d, npz_dir, do_graph)
    except Exception as e:  # keep going; record the failure
        return {'UniProt_ID': uid, 'status': f'failed: {e!r}'[:300],
                'trace': traceback.format_exc()[-500:]}


def _append(csv_path, row, tries=6):
    """Single writer (the parent process): append one row, keeping the header's column order.
    (File locks are unreliable on the SMB share, so workers never write.) Transient SMB errors
    (e.g. 'Bad file descriptor', seen once at 288 K) are retried with back-off instead of ending the run."""
    for k in range(tries):
        try:
            return _append_once(csv_path, row)
        except OSError as e:
            if k == tries - 1:
                raise
            print(f'  append retry {k + 1} after {e!r}', flush=True)
            time.sleep(5 * (k + 1))


def _append_once(csv_path, row):
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        extra = [c for c in df.columns if c not in cols]
        if extra:  # new columns (e.g. first failure row carries 'trace'): rewrite with widened header
            old = pd.read_csv(csv_path)
            pd.concat([old, df], ignore_index=True).to_csv(csv_path, index=False)
            return
        df.reindex(columns=cols).to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def run(out, ids_file=None, workers=4, do_graph=False, limit=None, base_dir=BASE_DIR):
    os.makedirs(os.path.join(out, 'npz'), exist_ok=True)
    csv_path = os.path.join(out, 'scalar_features_v2.csv')
    if ids_file:
        ids = [l.strip() for l in open(ids_file) if l.strip()]
    else:  # default: the sequences in BENDER_BIO.csv (the paper's analysis set)
        ids = pd.read_csv(BIO_CSV, usecols=['UniProt_ID'])['UniProt_ID'].astype(str).tolist()
    # flat layout (BENDER_<T>/<ID>/, the other temperatures) or group folders (BENDER_300K/<group>/<ID>/)
    flat = any(os.path.isdir(os.path.join(base_dir, u)) for u in ids[:50])
    if flat:
        idx = {u: os.path.join(base_dir, u) for u in ids if os.path.isdir(os.path.join(base_dir, u))}
    else:
        idx = index_trajectories(base_dir)
    print(f'layout: {"flat" if flat else "group folders"}; {len(idx)} trajectory folders found', flush=True)
    done = set()
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path, usecols=['UniProt_ID', 'status'])
        done = set(prev.loc[prev['status'] == 'ok', 'UniProt_ID'].astype(str))
    todo = [u for u in ids if u not in done]
    missing = [u for u in todo if u not in idx]
    todo = [u for u in todo if u in idx][:limit]
    print(f'{len(ids)} requested, {len(done)} done, {len(missing)} without a trajectory folder, '
          f'{len(todo)} to run with {workers} workers', flush=True)
    if missing:
        with open(os.path.join(out, 'missing_trajectories.txt'), 'w') as fh:
            fh.write('\n'.join(missing))
    args = [(u, idx[u], os.path.join(out, 'npz'), do_graph) for u in todo]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_worker, args), 1):
            _append(csv_path, row)
            msg = (f"ok:{row['UniProt_ID']}:{row['nu']:.3f}" if row.get('status') == 'ok'
                   else f"{row.get('status')}:{row['UniProt_ID']}")
            if i % 25 == 0 or not msg.startswith('ok'):
                el = time.time() - t0
                print(f'[{i}/{len(args)}] {msg}  ({el / 60:.1f} min, ~{el / i * (len(args) - i) / 3600:.1f} h left)',
                      flush=True)
    bc.write_provenance(csv_path, __file__, {**PARAMS, 'graph': do_graph, 'workers': workers, 'base_dir': base_dir},
                        [BIO_CSV, base_dir] + ([ids_file] if ids_file else []))


def build(out):
    """BENDER_BIO_v2.csv: v1 table + v2 ensemble columns (v1 ensemble columns renamed *_v1)."""
    v2 = pd.read_csv(os.path.join(out, 'scalar_features_v2.csv'))
    v2 = v2[v2['status'] == 'ok'].drop_duplicates('UniProt_ID', keep='last')
    bio = pd.read_csv(BIO_CSV)
    ens = ['rg', 'rg_err', 'ree', 'ree_err', 'delta', 'delta_err', 'nu', 'nu_err', 'A0', 'A0_err']
    bio = bio.rename(columns={c: f'{c}_v1' for c in ens if c in bio.columns})
    keep = ['UniProt_ID', 'n_frames_dcd', 'n_frames_production', 'n_frames_used', 'rg', 'rg_err',
            'ree', 'ree_err', 'delta', 'delta_err', 'nu', 'A0', 'nu_fit_err', 'nu_r2', 'nu_block_se',
            'nu_legacy', 'A0_legacy', 'nu_legacy_r2', 'nu_half1', 'nu_half2', 'contacts_per_frame']
    keep += [c for c in v2.columns if c.startswith('freq_') and not c.endswith(('_h1', '_h2'))]
    merged = bio.merge(v2[keep], on='UniProt_ID', how='left')
    merged['nu_r2_pass'] = merged['nu_r2'] >= bc.NU_R2_MIN
    merged['expected_frames_dcd'] = merged['n_residues'].map(bc.expected_frames)
    merged['truncated_trajectory'] = merged['n_frames_dcd'] < merged['expected_frames_dcd']
    merged['analysis_ok'] = merged['nu_r2_pass'] & ~merged['truncated_trajectory']
    merged['zone'] = merged['nu'].map(bc.zone)
    path = os.path.join(out, 'BENDER_BIO_v2.csv')
    merged.to_csv(path, index=False)
    bc.write_provenance(path, __file__, {'step': 'build', 'NU_R2_MIN': bc.NU_R2_MIN},
                        [BIO_CSV, os.path.join(out, 'scalar_features_v2.csv')])
    n_ok = merged['nu'].notna().sum()
    print(f'BENDER_BIO_v2.csv: {len(merged)} rows, {n_ok} with ν, {int(merged["nu_r2_pass"].sum())} pass '
          f'R² ≥ {bc.NU_R2_MIN}, {int(merged["truncated_trajectory"].sum())} truncated, '
          f'{int(merged["analysis_ok"].sum())} analysis_ok')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'build'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--ids', default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--graph', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--base-dir', default=BASE_DIR, help='trajectory root, laid out as <group>/<ID>/')
    a = ap.parse_args()
    if a.mode == 'run':
        run(a.out, a.ids, a.workers, a.graph, a.limit, a.base_dir)
    else:
        build(a.out)
