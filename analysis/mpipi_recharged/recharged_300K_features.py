"""
recharged_300K_features.py — per-sequence BENDER properties for the MPIPI-Recharged 300 K runs that so far exist only
for CALVADOS-2 (BENDER_BIO.csv): Re, asphericity, A0 (with nu), the five contact-network properties with frame SDs,
pi-pi and cation-pi contact frequencies, and the distance-SD vs AF2 PAE Pearson r. Every definition is imported:

  frames      recharged_io.load_run: bender_common.load_calvados, production = frames[n//2:], K_FRAMES = 1000
              evenly spaced; the chain is rebuilt whole from minimum-image bonds if any bond exceeds 1 nm (as
              recharged_nu.py); max bond recorded per run, a run is flagged if it is not below 0.5 nm (pae_v2 assert)
  geometry    bender_common.geometry (rg, ree, delta; SE = frame SD / sqrt(frames)) on the 1,000 frames
  nu, A0      bender_common.flory on the 1,000 frames (BENDER_BIO: nu_err = nu_block_se; A0 has no error there)
  contacts    bender_common.contact_frequency_map / pair_type_table (C-alpha < 0.8 nm, |i-j| >= 3), the 23 v1 pair
              types + Arg-Lys (mpipi_analysis_v2.PAIRS)
  pi          bender_bio_csv/update_pi_contacts.py (dynamic_pi_score definitions: F/W/Y pairs; Arg-aromatic pairs)
  network     graph_features.graph_features_from_adj on bender_common.adjacency of GRAPH_FRAMES = 50 of the 1,000
              frames (traj_analysis_calvados_v2 / mpipi_graph_v2 protocol); frame means and SDs
  PAE         validation/pae_v2.load_pae + correlate (SD ddof=1 of every i<j distance over the 1,000 frames vs PAE)

The adjacency stack is built frame-chunk by frame-chunk (identical result, bounded memory for long chains).

Modes
  run       --ids FILE | all; resumable (appends to <work>/rows_300K.csv; workers never write)
  check     the same feature code on (a) the MPIPI-GG trajectories with the MPIPI-GG frame rules
            (mpipi_analysis_v2: all xtc frames; mpipi_graph_v2 / pae_v2: 1,000 frames) against mpipi_results_v2.csv,
            mpipi_graph_features_v2.csv and mpipi_vs_pae_results_v2.tsv, and (b) the CALVADOS-2 trajectories
            against BENDER_BIO.csv and calvados_vs_pae_results_v2.tsv
  finalize  <out>/recharged_300K_features.csv (one row per run, failed runs listed) + provenance
"""
import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the code folders imported below

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # analysis/
for p in (os.path.join(BR, 'tables'), os.path.join(BR, 'validation'), os.path.join(BR, 'mpipi'),
          os.path.join(BR, 'mpipi_recharged'), BR):
    sys.path.insert(0, p)
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)
if 'tqdm' not in sys.modules:   # dynamic_pi_score imports tqdm at module level; the md env has none, nothing here uses it
    try:
        import tqdm  # noqa: F401
    except ImportError:
        sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda x, *a, **k: x)

import bender_common as bc  # noqa: E402
from graph_features import graph_features_from_adj  # noqa: E402
from traj_analysis_calvados_v2 import K_FRAMES, GRAPH_FRAMES, _append  # noqa: E402
import pae_v2  # noqa: E402
import update_pi_contacts as UP  # noqa: E402
import mpipi_analysis_v2 as MA  # noqa: E402
import mpipi_graph_v2 as MG  # noqa: E402
import recharged_io as R  # noqa: E402

RUNS = str(paths.TRAJ_RECHARGED.parent)   # <RUNS>/300K/<ID>/
TEMP = 300
OUT = str(paths.RECHARGED)
WORK = str(paths.WORK / 'mpipi_recharged_features_300K')
ROWS = os.path.join(WORK, 'rows_300K.csv')
FINAL = os.path.join(OUT, f'recharged_{TEMP}K_features.csv')
NU_CSV = os.path.join(OUT, f'recharged_{TEMP}K_nu.csv')
PAE_TOPUP = os.path.join(WORK, 'recharged_vs_pae_protists300.tsv')   # written by pae_topup_protists300.py
PAE_TSV = os.path.join(OUT, f'recharged_{TEMP}K_vs_pae_results_v2.tsv')   # calvados_vs_pae_results_v2.tsv layout
BIO = bc.BIO_CSV
GG_RUNS = str(paths.TRAJ_GG)
GG_RES = str(paths.DATA / 'mpipi_gg')
CAL_ROOT = str(paths.TRAJ_C2)
PAIRS = MA.PAIRS
FEATURES = MG.FEATURES
MAX_BOND_OK = 0.5
assert K_FRAMES == R.K_FRAMES == pae_v2.K_FRAMES == 1000 and GRAPH_FRAMES == 50


def adjacency_chunked(xyz, max_elems=4e7):
    """bender_common.adjacency over frame chunks (same boolean stack, bounded memory)."""
    n = xyz.shape[1]
    c = max(1, int(max_elems // (n * n * 3)))
    return np.concatenate([bc.adjacency(xyz[a:a + c]) for a in range(0, len(xyz), c)], axis=0)


def pi_contacts(F, seq):
    """update_pi_contacts.compute on a contact-frequency map in memory."""
    aro = [i for i, a in enumerate(seq) if a in UP.AROMATIC]
    cat = [i for i, a in enumerate(seq) if a in UP.CATION_PI]
    pp, npp = UP.pair_mean(F, aro, aro) if len(aro) >= 2 else (0.0, 0)
    cp, ncp = UP.pair_mean(F, cat, aro) if cat and aro else (0.0, 0)
    s = UP._seq_pi_score(seq)
    return {'pi_pi_contact_freq': pp, 'cation_pi_contact_freq': cp, 'delta_pi_pi_score': pp - s,
            'delta_cation_pi_score': cp - s, 'n_aromatic_pairs': npp, 'n_cation_pi_pairs': ncp}


def features(uid, seq, xyz, xyz_geo=None, pae=None, npz_path=None):
    """All per-sequence properties. xyz: the K_FRAMES frames (network, PAE). xyz_geo: frames for geometry, nu and
    contacts (CALVADOS-2 and MPIPI-Recharged: the same K_FRAMES frames; MPIPI-GG: all xtc frames)."""
    xg = xyz if xyz_geo is None else xyz_geo
    row = {'n_frames_used': int(len(xg)), 'max_bond_nm': round(bc.max_bond(xg), 4)}
    row.update(bc.geometry(xg))
    f = bc.flory(xg)
    row.update({k: f[k] for k in ('nu', 'A0', 'nu_fit_err', 'nu_r2', 'nu_block_se', 'nu_legacy', 'A0_legacy',
                                  'nu_legacy_r2')})
    row['nu_err'] = f['nu_block_se']                                  # BENDER_BIO convention
    F = bc.contact_frequency_map(xg)
    row.update(pi_contacts(F, seq))
    row.update(bc.pair_type_table(F, seq, PAIRS))
    iu = np.triu_indices(len(seq), k=bc.CONTACT_MIN_SEP)
    row['contacts_per_frame'] = float(F[iu].sum())

    xgr = xyz[bc.even_indices(len(xyz), GRAPH_FRAMES)]
    seq_graph, res_graph = graph_features_from_adj(adjacency_chunked(xgr), seq)
    row.update({k: seq_graph[k] for k in FEATURES})
    row.update({f'{k}_std': seq_graph[f'{k}_std'] for k in FEATURES})
    row['n_frames_graph'] = int(len(xgr))

    if pae is None:
        row['pae_status'] = 'no_pae'
    else:
        c = pae_v2.correlate(uid, xyz, pae)
        row.update({'pae_r': c['pearson_r'], 'pae_p': c['p_value'], 'n_pae': c['n_pae'], 'pae_n_pairs': c['n_pairs'],
                    'pae_n_frames': c['n_frames'], 'mean_sd': c['mean_sd'], 'mean_pae': c['mean_pae'],
                    'pae_status': 'ok' if c['n_pae'] == len(seq) else f"length_mismatch:{c['n_pae']}"})

    if npz_path:
        ii, jj, m1, m2 = bc._pair_moments(xyz, 1)
        fr = xyz.shape[0]
        sd = np.zeros((len(seq), len(seq)))
        sd[ii, jj] = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0) * fr / max(fr - 1, 1))
        sd += sd.T
        sep = jj - ii
        s_vals = np.arange(1, len(seq))
        tmp = npz_path + '.tmp.npz'
        np.savez_compressed(tmp, contact_freq=F.astype(np.float16), dist_sd=sd.astype(np.float16), s=s_vals,
                            rms_per_s=np.array([np.sqrt(m2[sep == s]).mean() for s in s_vals]),
                            mean_per_s=np.array([m1[sep == s].mean() for s in s_vals]), sequence=np.array(list(seq)),
                            **{f'res_{k}': v for k, v in res_graph.items() if k != 'sequence'})
        os.replace(tmp, npz_path)
    return row


def analyse_recharged(args):
    uid, d, npz_dir = args
    t0 = time.time()
    try:
        xyz, seq, info = R.load_run(d, uid)            # 1,000 production frames, chain whole, 'X' check
        row = {'UniProt_ID': uid, 'n_residues': len(seq), 'sequence_top': seq,
               'n_frames_dcd': info['n_frames_total'], 'n_frames_production': info['n_frames_production'],
               'expected_frames_dcd': bc.expected_frames(len(seq)), 'stride': info['stride'],
               'production_start': info['production_start'], 'box_nm': R.box_nm(len(seq)),
               'max_bond_raw_nm': round(info['max_bond_raw_nm'], 4),
               'unwrapped': bool(info['max_bond_raw_nm'] > 1.0)}
        row.update(features(uid, seq, xyz, pae=pae_v2.load_pae(uid),
                            npz_path=os.path.join(npz_dir, f'{uid}.npz') if npz_dir else None))
        row['status'] = 'ok' if row['max_bond_nm'] < MAX_BOND_OK else f"broken_chain:{row['max_bond_nm']:.3f}nm"
    except Exception as e:
        row = {'UniProt_ID': uid, 'status': f'failed: {e!r}'[:300], 'trace': traceback.format_exc()[-500:]}
    row['seconds'] = round(time.time() - t0, 1)
    return row


def run(ids_file=None, workers=6, limit=None, rows=ROWS, npz=True, seed=0):
    os.makedirs(WORK, exist_ok=True)
    npz_dir = os.path.join(WORK, 'npz') if npz else None
    if npz_dir:
        os.makedirs(npz_dir, exist_ok=True)
    dirs = R.run_dirs(RUNS, TEMP, ids_file)
    done = set()
    if os.path.exists(rows):
        prev = pd.read_csv(rows, usecols=['UniProt_ID', 'status'])
        done = set(prev.loc[~prev['status'].astype(str).str.startswith('failed'), 'UniProt_ID'].astype(str))
    todo = [u for u in dirs if u not in done]
    np.random.default_rng(seed).shuffle(todo)       # long chains spread over the run (memory, honest ETA)
    todo = todo[:limit]
    print(f'{len(dirs)} runs with {TEMP}K/<ID>/<ID>.dcd + top.pdb, {len(done)} done, {len(todo)} to run, '
          f'{workers} workers -> {rows}', flush=True)
    t0 = time.time()
    with mp.get_context('fork').Pool(workers, maxtasksperchild=50) as pool:
        for i, row in enumerate(pool.imap_unordered(analyse_recharged, [(u, dirs[u], npz_dir) for u in todo]), 1):
            _append(rows, row)
            if i % 100 == 0 or row.get('status') != 'ok':
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] {row.get('status')}:{row['UniProt_ID']}  ({el / 60:.1f} min, "
                      f"~{el / i * (len(todo) - i) / 3600:.1f} h left)", flush=True)
    print(f'run finished in {(time.time() - t0) / 60:.1f} min', flush=True)


# ── check: the same feature code reproduces the existing MPIPI-GG and CALVADOS-2 outputs ─────────────────────────
def check_one(args):
    uid, ff, kingdom = args
    t0 = time.time()
    try:
        pae = pae_v2.load_pae(uid)
        if ff == 'mpipi_gg':
            d = os.path.join(GG_RUNS, uid)
            seq = bc.read_mpipi_sequence(os.path.join(d, 'seq.dat'))
            xyz_all, _ = bc.load_mpipi(os.path.join(d, 'traj.xtc'), n_res=len(seq))          # mpipi_analysis_v2
            xyz, info = bc.load_mpipi(os.path.join(d, 'traj.xtc'), n_res=len(seq), k=K_FRAMES)  # graph_v2, pae_v2
            row = features(uid, seq, xyz, xyz_geo=xyz_all, pae=pae)
        else:
            d = os.path.join(CAL_ROOT, kingdom, uid)
            xyz, seq, info = bc.load_calvados(os.path.join(d, f'{uid}.dcd'), os.path.join(d, 'top.pdb'), k=K_FRAMES)
            row = features(uid, seq, xyz, pae=pae)
        row.update({'UniProt_ID': uid, 'ff': ff, 'status': 'ok'})
    except Exception as e:
        row = {'UniProt_ID': uid, 'ff': ff, 'status': f'failed: {e!r}'[:300]}
    row['seconds'] = round(time.time() - t0, 1)
    return row


def check(ids_file, workers, out_dir, n_cal=12):
    ids = [l.strip() for l in open(ids_file) if l.strip()]
    bio = pd.read_csv(BIO, low_memory=False).set_index('UniProt_ID')
    mres = pd.read_csv(os.path.join(GG_RES, 'mpipi_results_v2.csv')).set_index('UniProt_ID')
    gg957 = set(mres.index[(mres['status'] == 'ok') & mres['in_paper_set'] & mres.index.isin(bio.index)])
    gg957 = {u for u in gg957 if bio.at[u, 'kingdom'] in bc.SIX_LINEAGES}
    gg_ids = [u for u in ids if u in gg957]
    cal_ids = (gg_ids + [u for u in ids if u not in gg957])[:n_cal]
    print(f'{len(gg957)} sequences in the Mpipi-GG 957 set; checking {len(gg_ids)} Mpipi-GG and {len(cal_ids)} '
          f'CALVADOS-2 trajectories', flush=True)
    jobs = [(u, 'mpipi_gg', None) for u in gg_ids] + [(u, 'calvados', bio.at[u, 'kingdom']) for u in cal_ids]
    with mp.get_context('fork').Pool(workers) as pool:
        rows = pool.map(check_one, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'check_rows.csv'), index=False)

    # references
    mg = pd.read_csv(os.path.join(GG_RES, 'mpipi_graph_features_v2.csv'))
    mg = mg[mg['status'] == 'ok'].drop_duplicates('UniProt_ID', keep='last').set_index('UniProt_ID')
    mpae = pd.read_csv(f'{pae_v2.TSV_DIR}/mpipi_vs_pae_results_v2.tsv', sep='\t').set_index('uniprot_id')
    cpae = pd.read_csv(f'{pae_v2.TSV_DIR}/calvados_vs_pae_results_v2.tsv', sep='\t').set_index('uniprot_id')
    freq_cols = [c for c in df.columns if c.startswith('freq_')]
    gg_map = {'rg': ('res', 'rg_nm'), 'rg_err': ('res', 'rg_err_nm'), 'ree': ('res', 'ree_nm'),
              'ree_err': ('res', 'ree_err_nm'), 'delta': ('res', 'delta'), 'nu': ('res', 'nu_mpipi'),
              'A0': ('res', 'A0_mpipi'), 'nu_block_se': ('res', 'nu_block_se_mpipi'),
              'contacts_per_frame': ('res', 'contacts_per_frame'), 'pae_r': ('pae', 'pearson_r'),
              'mean_sd': ('pae', 'mean_sd'), **{c: ('res', c) for c in freq_cols},
              **{k: ('graph', k) for k in FEATURES}, **{f'{k}_std': ('graph', f'{k}_std') for k in FEATURES}}
    cal_map = {k: ('bio', k) for k in ['rg', 'rg_err', 'ree', 'ree_err', 'delta', 'delta_err', 'nu', 'nu_err', 'A0',
                                       'pi_pi_contact_freq', 'cation_pi_contact_freq', 'delta_pi_pi_score',
                                       'delta_cation_pi_score', 'n_aromatic_pairs', 'n_cation_pi_pairs']
               + FEATURES + [f'{k}_std' for k in FEATURES]}
    cal_map.update({'pae_r': ('pae', 'pearson_r'), 'mean_sd': ('pae', 'mean_sd')})
    out = []
    for ff, cmap in (('mpipi_gg', gg_map), ('calvados', cal_map)):
        sub = df[(df['ff'] == ff) & (df['status'] == 'ok')].set_index('UniProt_ID')
        refs = {'res': mres, 'graph': mg, 'pae': mpae} if ff == 'mpipi_gg' else {'bio': bio, 'pae': cpae}
        for col, (src, rcol) in cmap.items():
            ref = refs[src]
            u = [x for x in sub.index if x in ref.index]
            a = sub.loc[u, col].astype(float).to_numpy()
            b = ref.loc[u, rcol].astype(float).to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            both_nan = int((~np.isfinite(a) & ~np.isfinite(b)).sum())
            dif = np.abs(a[ok] - b[ok])
            rel = dif / np.maximum(np.abs(b[ok]), 1e-12)
            out.append({'ff': ff, 'property': col, 'reference': rcol, 'n': int(ok.sum()), 'both_nan': both_nan,
                        'one_nan': int(len(u) - ok.sum() - both_nan),
                        'max_abs_diff': float(dif.max()) if len(dif) else np.nan,
                        'max_rel_diff': float(rel.max()) if len(rel) else np.nan})
    rep = pd.DataFrame(out)
    rep.to_csv(os.path.join(out_dir, 'check_report.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(rep.to_string(index=False))
    print(df['status'].value_counts().to_string())


# ── finalize ─────────────────────────────────────────────────────────────────────────────────────────────────────
FRONT = ['UniProt_ID', 'kingdom', 'status', 'n_residues', 'n_frames_dcd', 'n_frames_production', 'n_frames_used',
         'expected_frames_dcd', 'stride', 'box_nm', 'max_bond_raw_nm', 'max_bond_nm', 'unwrapped',
         'rg', 'rg_err', 'ree', 'ree_err', 'delta', 'delta_err', 'nu', 'nu_err', 'A0', 'nu_fit_err', 'nu_r2',
         'nu_block_se', 'nu_legacy', 'A0_legacy', 'nu_legacy_r2']
FRONT += FEATURES + [f'{k}_std' for k in FEATURES] + ['n_frames_graph']
FRONT += ['pi_pi_contact_freq', 'cation_pi_contact_freq', 'delta_pi_pi_score', 'delta_cation_pi_score',
          'n_aromatic_pairs', 'n_cation_pi_pairs', 'pae_r', 'pae_p', 'pae_status', 'n_pae', 'pae_n_pairs',
          'pae_n_frames', 'mean_sd', 'mean_pae', 'contacts_per_frame']


def finalize():
    rows = pd.read_csv(ROWS, low_memory=False)
    rows['_ok'] = rows['status'] == 'ok'
    rows = rows.sort_values('_ok').drop_duplicates('UniProt_ID', keep='last').drop(columns='_ok')  # ok row wins
    rows['pae_source'] = np.where(rows.get('pae_status', pd.Series(index=rows.index)).notna(), 'run', '')
    if os.path.exists(PAE_TOPUP):   # pae_topup_protists300.py recharged: PAE files fetched after the run
        top = pd.read_csv(PAE_TOPUP, sep='\t')
        top = top[top['status'] == 'ok'].set_index('uniprot_id')
        fill = rows['status'].eq('ok') & rows['pae_status'].eq('no_pae') & rows['UniProt_ID'].isin(top.index)
        u = rows.loc[fill, 'UniProt_ID']
        for col, src in (('pae_r', 'pearson_r'), ('pae_p', 'p_value'), ('n_pae', 'n_pae'), ('pae_n_pairs', 'n_pairs'),
                         ('pae_n_frames', 'n_frames'), ('mean_sd', 'mean_sd'), ('mean_pae', 'mean_pae')):
            rows.loc[fill, col] = top.loc[u, src].to_numpy()
        rows.loc[fill, 'pae_status'] = np.where(rows.loc[fill, 'n_pae'] == rows.loc[fill, 'n_residues'], 'ok',
                                                'length_mismatch:' + rows.loc[fill, 'n_pae'].astype(str))
        rows.loc[fill, 'pae_source'] = os.path.basename(PAE_TOPUP)
        print(f'PAE columns filled from {os.path.basename(PAE_TOPUP)} for {int(fill.sum())} runs')
    bio = pd.read_csv(BIO, usecols=['UniProt_ID', 'kingdom', 'sequence'])
    nu = pd.read_csv(NU_CSV, usecols=['UniProt_ID'])
    universe = sorted(set(bio['UniProt_ID']) | set(nu['UniProt_ID']) | set(rows['UniProt_ID']))
    df = pd.DataFrame({'UniProt_ID': universe}).merge(rows, on='UniProt_ID', how='left')
    df['status'] = df['status'].fillna('missing: no row (run not found or not analysed)')
    df = df.merge(bio, on='UniProt_ID', how='left')
    df['sequence_matches_bio'] = df['sequence_top'] == df['sequence']
    # BENDER_BIO keeps UniProt's unknown residue X; both the CALVADOS-2 and the MPIPI-Recharged topologies have Gly there
    x_gly = [isinstance(a, str) and isinstance(b, str) and len(a) == len(b) and a != b and
             all(p == q or (q == 'X' and p == 'G') for p, q in zip(a, b))
             for a, b in zip(df['sequence_top'], df['sequence'])]
    df['sequence_note'] = np.where(x_gly, 'BENDER_BIO X simulated as Gly (as in CALVADOS-2)', '')
    bad_seq = df['status'].eq('ok') & ~df['sequence_matches_bio'] & ~np.array(x_gly)
    df.loc[bad_seq, 'status'] = 'sequence_mismatch_with_BENDER_BIO'
    df['truncated_trajectory'] = df['n_frames_dcd'] < df['expected_frames_dcd']
    df['nu_r2_pass'] = df['nu_r2'] >= bc.NU_R2_MIN
    df['network_features_method'] = np.where(df['global_efficiency'].notna(), MG.METHOD, '')
    df = df.drop(columns=['sequence', 'sequence_top', 'trace'], errors='ignore')
    cols = [c for c in FRONT if c in df.columns]
    df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(FINAL, index=False)
    failed = df[df['status'] != 'ok'][['UniProt_ID', 'kingdom', 'status']]
    failed.to_csv(os.path.join(WORK, f'failed_{TEMP}K.tsv'), sep='\t', index=False)
    params = {'temperature_K': TEMP, 'K_FRAMES': K_FRAMES, 'GRAPH_FRAMES': GRAPH_FRAMES,
              'production': 'frames[n//2:] (recharged_io.load_run -> bender_common.load_calvados)',
              'subsample': 'even stride max(1, n_prod//K)',
              'whole_chain': 'unwrap_chain with box_nm(N) if raw max bond > 1.0 nm (recharged_nu.py rule); '
                             f'status ok requires max bond < {MAX_BOND_OK} nm',
              'geometry': 'bender_common.geometry on the 1,000 frames; *_err = frame SD / sqrt(frames)',
              'nu_A0': 'bender_common.flory on the 1,000 frames; nu_err = nu_block_se (BENDER_BIO); '
                       'A0 has no error (BENDER_BIO A0_err is empty)',
              'contacts': f'C-alpha < {bc.CONTACT_CUTOFF_NM} nm, |i-j| >= {bc.CONTACT_MIN_SEP}; '
                          'per-pair frequency, pair types = mpipi_analysis_v2.PAIRS',
              'pi_contacts': 'bender_bio_csv/update_pi_contacts.py (dynamic_pi_score: F/W/Y; Arg-aromatic)',
              'network': MG.METHOD + ' (graph_features.graph_features_from_adj)',
              'pae': 'validation/pae_v2.correlate: SD (ddof=1) of every i<j distance over the 1,000 frames vs AF2 '
                     'PAE[i, j]; pae_v2.load_pae',
              'npz': 'features_300K/npz/<ID>.npz: contact_freq, dist_sd (float16), RMS/mean distance per s, '
                     'res_* per-residue graph features',
              'imports': {m.__name__: os.path.abspath(m.__file__) for m in
                          (bc, pae_v2, UP, MA, MG, R, sys.modules['graph_features'],
                           sys.modules['traj_analysis_calvados_v2'], sys.modules['pi_contacts'])},
              'n_rows': int(len(df)), 'n_ok': int((df['status'] == 'ok').sum())}
    params['pae_topup'] = f'{PAE_TOPUP} (rows whose PAE file was fetched after the run; column pae_source)'
    side = bc.write_provenance(FINAL, __file__, params, [RUNS, ROWS, BIO, NU_CSV, pae_v2.PAE_DIR] +
                               ([PAE_TOPUP] if os.path.exists(PAE_TOPUP) else []))
    print(f'wrote {FINAL}: {len(df)} rows, {params["n_ok"]} ok; provenance {os.path.basename(side)}')
    print(df['status'].value_counts().to_string())

    # per-sequence PAE table in the calvados_vs_pae_results_v2.tsv layout (pae_v2.correlate keys)
    pae = pd.DataFrame({'uniprot_id': df['UniProt_ID'], 'pearson_r': df['pae_r'], 'p_value': df['pae_p'],
                        'n_residues': df['n_residues'], 'n_pae': df['n_pae'], 'n_frames': df['pae_n_frames'],
                        'n_pairs': df['pae_n_pairs'], 'mean_sd': df['mean_sd'], 'mean_pae': df['mean_pae'],
                        'status': np.where(df['status'] != 'ok', df['status'], df['pae_status'])})
    for c in ('n_residues', 'n_pae', 'n_frames', 'n_pairs'):
        pae[c] = pae[c].astype('Int64')
    pae.to_csv(PAE_TSV, sep='\t', index=False)
    bc.write_provenance(PAE_TSV, __file__, {'ff': 'mpipi_recharged', 'K_FRAMES': K_FRAMES,
                                            'frames': 'production half, chain whole (recharged_io.load_run)',
                                            'sd': 'ddof=1 over frames, all pairs i<j (pae_v2.correlate)',
                                            'status': 'ok, no_pae (no AF2 PAE file), length_mismatch:<n_pae>, or the '
                                                      'run status when the run was not ok',
                                            'source': FINAL}, [FINAL, pae_v2.PAE_DIR])
    print(f'wrote {PAE_TSV}:', pae['status'].value_counts().to_dict())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'check', 'finalize'])
    ap.add_argument('--ids', default=None)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--rows', default=ROWS, help='rows CSV (run); a pilot can write elsewhere')
    ap.add_argument('--no-npz', action='store_true')
    ap.add_argument('--check-out', default=None)
    a = ap.parse_args()
    if a.mode == 'run':
        run(a.ids, a.workers, a.limit, a.rows, not a.no_npz)
    elif a.mode == 'check':
        os.makedirs(a.check_out, exist_ok=True)
        check(a.ids, a.workers, a.check_out)
    else:
        finalize()
