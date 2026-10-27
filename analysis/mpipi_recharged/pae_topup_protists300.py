"""
pae_topup_protists300.py — AF2 PAE agreement for the 300 top-up protists (protist_300_cdhit40.json), which have no
row in calvados_vs_pae_results_v2.tsv (that table covers the original 9,982 six-group sequences).

  fetch      AF2 PAE JSON from the AlphaFold DB for every accession without <acc>_pae.json in pae_v2.PAE_DIR
             (prediction API -> paeDocUrl, fallback files/AF-<acc>-F1-predicted_aligned_error_v4.json). Each file is
             downloaded into a fresh staging folder, parsed and checked (N x N matrix, AF sequence = BENDER_BIO
             sequence), and copied into PAE_DIR only when no file of that name exists. Manifest: fetch_manifest.tsv.
  calvados   pae_v2.run_one('calvados', ...) on trajectories/calvados2/300K/Protists/<ID>/ (production half, 1,000 frames)
             -> BENDER-BIO/PAE/pae_analysis/tsv/calvados_vs_pae_results_v2_protists300.tsv (new file; the existing
             tsv is not touched)
  recharged  the same correlate() on the MPIPI-Recharged runs (recharged_io.load_run frames, as
             recharged_300K_features.py) -> Analysis_FDT_RECHARGED/features_300K/recharged_vs_pae_protists300.tsv,
             which recharged_300K_features.py finalize uses to fill the PAE columns of rows that had no PAE file

    python pae_topup_protists300.py fetch|calvados|recharged [--workers N]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recharged_300K_features as RF  # noqa: E402  (sets up the shared imports)
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

bc, pae_v2, R = RF.bc, RF.pae_v2, RF.R
FASTA_300 = str(paths.INPUTS / 'protist_300_cdhit40.fasta')   # the 300 protists added to BENDER
CAL_TSV = f'{pae_v2.TSV_DIR}/calvados_vs_pae_results_v2_protists300.tsv'
RCH_TSV = os.path.join(RF.WORK, 'recharged_vs_pae_protists300.tsv')
MANIFEST = os.path.join(RF.WORK, 'pae_fetch_protists300', 'fetch_manifest.tsv')
API = 'https://alphafold.ebi.ac.uk/api/prediction/{acc}'
FALLBACK = 'https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-predicted_aligned_error_v4.json'


def accessions():
    return [line[1:].split()[0] for line in open(FASTA_300) if line.startswith('>')]


def _get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'bender-pae-topup'}),
                                        timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def fetch():
    bio = pd.read_csv(RF.BIO, usecols=['UniProt_ID', 'sequence']).set_index('UniProt_ID')['sequence']
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    rows = []
    for acc in accessions():
        dest = f'{pae_v2.PAE_DIR}/{acc}_pae.json'
        if os.path.exists(dest):
            rows.append({'acc': acc, 'status': 'already_on_disk'})
            continue
        row = {'acc': acc}
        stage = tempfile.mkdtemp(prefix=f'pae_{acc}_', dir=os.path.dirname(MANIFEST))   # own empty folder per file
        try:
            url, af_seq, version = FALLBACK.format(acc=acc), None, None
            try:
                meta = json.loads(_get(API.format(acc=acc)))
                ent = next(e for e in meta if e.get('uniprotAccession') == acc)
                url, af_seq, version = ent['paeDocUrl'], ent.get('uniprotSequence') or ent.get('sequence'), \
                    ent.get('latestVersion')
            except Exception as e:
                row['api_note'] = f'{e!r}'[:120]
            raw = _get(url)
            path = os.path.join(stage, f'{acc}_pae.json')
            open(path, 'wb').write(raw)
            data = json.loads(raw)
            d0 = data[0] if isinstance(data, list) else data
            pae = np.asarray(d0['predicted_aligned_error'], float)
            n = len(bio.get(acc, ''))
            row.update({'url': url, 'af_version': version, 'bytes': len(raw),
                        'sha256': hashlib.sha256(raw).hexdigest(), 'n_pae': pae.shape[0], 'n_bio': n,
                        'square': pae.ndim == 2 and pae.shape[0] == pae.shape[1],
                        'af_seq_matches_bio': (af_seq == bio.get(acc)) if af_seq else None})
            ok = row['square'] and pae.shape[0] == n and row['af_seq_matches_bio'] is not False
            if ok and not os.path.exists(dest):
                shutil.copyfile(path, dest)
                row['status'] = 'fetched'
            else:
                row['status'] = 'rejected: ' + ('length/shape mismatch' if not (row['square'] and pae.shape[0] == n)
                                                else 'AF sequence differs from BENDER_BIO')
        except Exception as e:
            row['status'] = f'failed: {e!r}'[:200]
        finally:
            row['when'] = dt.datetime.now().isoformat(timespec='seconds')
            rows.append(row)
            print(row['acc'], row['status'], flush=True)
    m = pd.DataFrame(rows)
    m.to_csv(MANIFEST, sep='\t', index=False)
    print(m['status'].value_counts().to_string())


def calvados(workers):
    ids = accessions()
    jobs = [('calvados', u, f'{RF.CAL_ROOT}/Protists/{u}') for u in ids]
    with mp.get_context('fork').Pool(workers) as pool:
        df = pd.DataFrame(pool.map(pae_v2.run_one, jobs, chunksize=1))
    df.to_csv(CAL_TSV, sep='\t', index=False)
    bc.write_provenance(CAL_TSV, __file__, {'ff': 'calvados', 'K_FRAMES': pae_v2.K_FRAMES, 'frames': 'production half',
                                            'sd': 'ddof=1 over frames, all pairs i<j (pae_v2.correlate)',
                                            'ids': FASTA_300, 'traj': f'{RF.CAL_ROOT}/Protists/<ID>'},
                        [pae_v2.PAE_DIR, FASTA_300])
    print(f'wrote {CAL_TSV}'); print(df['status'].value_counts().to_string())
    ok = df[df['status'] == 'ok']
    if len(ok):
        print(f'median r {ok["pearson_r"].median():.3f} over {len(ok)}')


def _rch_one(uid):
    try:
        pae = pae_v2.load_pae(uid)
        if pae is None:
            return {'uniprot_id': uid, 'status': 'no_pae'}
        xyz, seq, info = R.load_run(os.path.join(RF.RUNS, f'{RF.TEMP}K', uid), uid)
        assert bc.max_bond(xyz) < RF.MAX_BOND_OK, f'broken chain ({bc.max_bond(xyz):.2f} nm)'
        return pae_v2.correlate(uid, xyz, pae)
    except Exception as e:
        return {'uniprot_id': uid, 'status': f'failed: {e!r}'[:200]}


def recharged(workers):
    with mp.get_context('fork').Pool(workers) as pool:
        df = pd.DataFrame(pool.map(_rch_one, accessions(), chunksize=1))
    df.to_csv(RCH_TSV, sep='\t', index=False)
    bc.write_provenance(RCH_TSV, __file__, {'ff': 'mpipi_recharged', 'K_FRAMES': RF.K_FRAMES,
                                            'frames': 'production half, chain whole (recharged_io.load_run)',
                                            'sd': 'ddof=1 over frames, all pairs i<j (pae_v2.correlate)',
                                            'ids': FASTA_300}, [pae_v2.PAE_DIR, FASTA_300, RF.RUNS])
    print(f'wrote {RCH_TSV}'); print(df['status'].value_counts().to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['fetch', 'calvados', 'recharged'])
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    {'fetch': lambda: fetch(), 'calvados': lambda: calvados(a.workers), 'recharged': lambda: recharged(a.workers)}[a.mode]()
