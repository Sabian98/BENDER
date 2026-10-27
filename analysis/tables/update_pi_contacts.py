"""
update_pi_contacts.py — recompute the pi-pi and cation-pi contact columns of BENDER_BIO.csv (2026-09-27).

Definitions follow Scripts/dynamic_pi_score.py, the script that produced the original columns:
  pi_pi_contact_freq      mean per-pair contact frequency over all aromatic (F, W, Y) pairs
                          (0 when fewer than two aromatic residues)
  cation_pi_contact_freq  same over Arg–aromatic pairs (Lys excluded, as in the original)
  delta_*_score           contact frequency minus the sequence pi_score (_seq_pi_score)
The original values used every trajectory frame (equilibration included) and pairs with |i−j| >= 2,
which counts the always-in-contact i,i+2 neighbors. Here they come from the reanalysis contact maps
(Analysis_300K_v2_graph/npz: 1,000 production frames, Cα–Cα < 0.8 nm, |i−j| >= 3, the bender_common rule).
Adds n_aromatic_pairs and n_cation_pi_pairs so that 0 for "no eligible pair" can be told apart.

Dry run by default; --apply copies BENDER_BIO.csv to BENDER_BIO_before_pi.csv (never overwritten).
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
from pi_contacts import _seq_pi_score, AROMATIC, CATION_PI  # noqa: E402  (original definitions)
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

BIO = str(paths.BIO)
NPZ = str(paths.WORK / 'calvados2_300K' / 'npz')   # contact maps written by traj_analysis_calvados_v2.py --graph


def pair_mean(F, idx_a, idx_b):
    """Mean contact frequency over unique pairs (i in a, j in b, |i-j| >= CONTACT_MIN_SEP)."""
    pairs = {(min(i, j), max(i, j)) for i in idx_a for j in idx_b
             if i != j and abs(i - j) >= bc.CONTACT_MIN_SEP}
    if not pairs:
        return 0.0, 0
    ii, jj = np.array(sorted(pairs)).T
    return float(F[ii, jj].mean()), len(pairs)


def compute(uid, seq):
    F = np.load(f'{NPZ}/{uid}.npz')['contact_freq'].astype(np.float64)
    assert F.shape == (len(seq), len(seq)), (uid, F.shape, len(seq))
    aro = [i for i, a in enumerate(seq) if a in AROMATIC]
    cat = [i for i, a in enumerate(seq) if a in CATION_PI]
    pp, npp = pair_mean(F, aro, aro) if len(aro) >= 2 else (0.0, 0)
    cp, ncp = pair_mean(F, cat, aro) if cat and aro else (0.0, 0)
    s = _seq_pi_score(seq)
    return {'pi_pi_contact_freq': pp, 'cation_pi_contact_freq': cp,
            'delta_pi_pi_score': pp - s, 'delta_cation_pi_score': cp - s,
            'n_aromatic_pairs': npp, 'n_cation_pi_pairs': ncp}


def main(apply):
    bio = pd.read_csv(BIO, low_memory=False)
    rows = [compute(u, s) for u, s in zip(bio['UniProt_ID'], bio['sequence'])]
    new = pd.DataFrame(rows, index=bio.index)
    for c in ['pi_pi_contact_freq', 'cation_pi_contact_freq']:
        a, b = bio[c].to_numpy(float), new[c].to_numpy(float)
        print(f'{c:24s} old median {np.median(a):.4f} max {a.max():.3f} | new median {np.median(b):.4f} '
              f'max {b.max():.3f} | r(old, new) {np.corrcoef(a, b)[0, 1]:+.3f}')
    print('sequences with no aromatic pair:', int((new['n_aromatic_pairs'] == 0).sum()),
          '| no Arg-aromatic pair:', int((new['n_cation_pi_pairs'] == 0).sum()))
    if not apply:
        print('dry run: nothing written (pass --apply)')
        return
    backup = BIO.replace('.csv', '_before_pi.csv')
    if not os.path.exists(backup):
        shutil.copy2(BIO, backup)
    for c in new.columns:
        bio[c] = new[c]
    bio.to_csv(BIO, index=False)
    bc.write_provenance(BIO, __file__, {'step': 'pi-pi / cation-pi contact columns recomputed from v2 contact maps',
                                        'definition': 'dynamic_pi_score.py; |i-j| >= 3; production frames'},
                        [backup, NPZ])
    print(f'wrote {BIO} (backup {os.path.basename(backup)})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    main(ap.parse_args().apply)
