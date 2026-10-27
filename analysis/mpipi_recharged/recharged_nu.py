"""
recharged_nu.py — compaction of the MPIPI-Recharged (OpenMpipi) 300 K runs: the scaling exponent nu per protein, with
the same estimator as CALVADOS-2 (BENDER_BIO) and MPIPI-GG (mpipi_temperature_hpc.py): bender_common.flory on all
production frames (second half of the DCD), chain made whole across the periodic box first.

    python recharged_nu.py --runs ROOT --temp 300 --out DIR [--workers N]

ROOT holds <T>K/<ID>/<ID>.dcd + top.pdb (run_openmpipi_idp.py --outdir). Output: DIR/recharged_<T>K_nu.csv, one row
per protein (nu, A0, nu_fit_err, nu_r2, nu_block_se, nu_legacy, nu_legacy_r2, nu on each half, <Rg>); resumable.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import traceback  # noqa: E402

import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import recharged_io as R  # noqa: E402

bc = R.bc


def one(args):
    uid, d = args
    try:
        xyz, seq, info = bc.load_calvados(os.path.join(d, f'{uid}.dcd'), os.path.join(d, 'top.pdb'))
        if bc.max_bond(xyz) > 1.0:
            xyz = bc.unwrap_chain(xyz, R.box_nm(len(seq)))
        f = bc.flory(xyz)
        h = len(xyz) // 2
        row = {'UniProt_ID': uid, 'n_residues': len(seq), 'n_frames': len(xyz),
               **{k: f[k] for k in ('nu', 'A0', 'nu_fit_err', 'nu_r2', 'nu_block_se', 'nu_legacy', 'nu_legacy_r2')},
               'nu_half1': bc.nu_primary_from_xyz(xyz[:h])['nu'], 'nu_half2': bc.nu_primary_from_xyz(xyz[h:])['nu'],
               'rg_mean': bc.geometry(xyz)['rg'], 'max_bond_nm': round(bc.max_bond(xyz), 4)}
        return row, None
    except Exception as ex:
        return {'UniProt_ID': uid}, f'{ex!r} | {traceback.format_exc()[-300:]}'.replace('\n', ' ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--temp', type=float, default=300)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'recharged_{int(a.temp)}K_nu.csv')
    dirs = R.run_dirs(a.runs, a.temp)
    done = set(pd.read_csv(path, usecols=['UniProt_ID']).UniProt_ID) if os.path.exists(path) else set()
    todo = [(u, d) for u, d in dirs.items() if u not in done]
    print(f'{len(dirs)} runs, {len(done)} done, {len(todo)} to run', flush=True)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for i, (row, err) in enumerate(pool.imap_unordered(one, todo), 1):
            if err:
                print(f"failed {row['UniProt_ID']}: {err[:200]}", flush=True)
                continue
            pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
            if i % 500 == 0:
                print(f'{i}/{len(todo)}', flush=True)
    bc.write_provenance(path, __file__, {'temperature_K': a.temp, 'frames': 'all production frames (second half)',
                                         'estimator': 'bender_common.flory'}, [a.runs])
    print('finished', path, flush=True)


if __name__ == '__main__':
    main()
