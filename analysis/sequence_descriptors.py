"""
sequence_descriptors.py — sequence descriptors for BENDER_BIO, one definition per column (user
decision 2026-09-27: recompute every row; kappa and SHD reimplemented from their publications).

Charge descriptors (FCR, NCPR, SCD, f_pos, f_neg) come from bender_common (residue charges only).

Definitions
  f_X                  count(X)/N for each amino acid
  f_aro                F, W, Y
  f_hydrophobic        A, I, L, M, F, V, W
  f_polar              S, T, N, Q
  f_disorder_promoting A, R, G, Q, S, P, E, K   (Dunker et al. 2001)
  f_order_promoting    W, C, F, I, Y, V, L, N   (Dunker et al. 2001)
  f_pi                 F, R, W, Y
  f_pi_extended        F, H, N, Q, R, W, Y
  hydropathy_avg       mean Kyte–Doolittle hydropathy
  lambda_avg, _std     mean and s.d. (ddof = 0) of CALVADOS-2 residue stickiness lambda
  kappa                Das & Pappu, PNAS 110, 13392 (2013): mean over blob sizes g = 5 and 6 of
                       delta/delta_max, with delta = mean over the N-g+1 sliding blobs of
                       (sigma_blob - sigma)^2, sigma = (f+ - f-)^2 / (f+ + f-) (0 for a blob without
                       charges); delta_max is the largest delta over sequences of the same composition
                       with all positive and all negative residues each in one contiguous block and
                       the neutral residues split between the start, the middle and the end (searched
                       exhaustively for neutral segments shorter than g + 1; longer segments behave
                       identically). NaN when the sequence has no charged residue.
  SHD                  Zheng et al., J. Phys. Chem. Lett. 11, 3408 (2020):
                       (1/N) sum_{i<j} (lambda_i + lambda_j) |i - j|^-1, with CALVADOS-2 lambda.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

AA = 'ACDEFGHIKLMNPQRSTVWY'
RESIDUES_CSV = str(paths.CALVADOS_RESIDUES)
LAMBDA = pd.read_csv(RESIDUES_CSV).set_index('one')['lambdas'].to_dict()
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9,
          M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
SETS = {'f_aro': 'FWY', 'f_hydrophobic': 'AILMFVW', 'f_polar': 'STNQ',
        'f_disorder_promoting': 'ARGQSPEK', 'f_order_promoting': 'WCFIYVLN',
        'f_pi': 'FRWY', 'f_pi_extended': 'FHNQRWY'}
BLOBS = (5, 6)


# ── kappa ─────────────────────────────────────────────────────────────
def _delta(q: np.ndarray, g: int) -> float:
    """Das–Pappu delta for a charge vector q (+1, -1, 0) and blob size g."""
    n = len(q)
    pos, neg = (q > 0).astype(float), (q < 0).astype(float)
    def sig(fp, fn):
        tot = fp + fn
        return np.divide((fp - fn) ** 2, tot, out=np.zeros_like(tot), where=tot > 0)
    s_all = sig(np.array([pos.mean()]), np.array([neg.mean()]))[0]
    if n < g:
        return 0.0
    w = np.ones(g)
    fp = np.convolve(pos, w, 'valid') / g
    fn = np.convolve(neg, w, 'valid') / g
    return float(np.mean((sig(fp, fn) - s_all) ** 2))


def _delta_max(n_pos: int, n_neg: int, n_neu: int, g: int) -> float:
    best = 0.0
    small = range(0, g + 2)
    for first, second in ((1, -1), (-1, 1)):
        for a in small:                      # neutrals before the first charge block
            for b in small:                  # neutrals between the blocks
                for c_rest in (True, False):
                    if a + b > n_neu:
                        continue
                    rest = n_neu - a - b
                    # remaining neutrals go to the end, or (if c_rest False) to the middle
                    mid = b + (0 if c_rest else rest)
                    end = rest if c_rest else 0
                    q = np.concatenate([np.zeros(a), np.full(n_pos if first > 0 else n_neg, first),
                                        np.zeros(mid), np.full(n_neg if first > 0 else n_pos, second),
                                        np.zeros(end)])
                    best = max(best, _delta(q, g))
    return best


def kappa(seq: str) -> float:
    q = np.array([1 if c in 'KR' else -1 if c in 'DE' else 0 for c in seq])
    n_pos, n_neg = int((q > 0).sum()), int((q < 0).sum())
    if n_pos + n_neg == 0:
        return float('nan')
    vals = []
    for g in BLOBS:
        dmax = _delta_max(n_pos, n_neg, len(q) - n_pos - n_neg, g)
        vals.append(_delta(q, g) / dmax if dmax > 0 else 0.0)
    return float(np.mean(vals))


# ── SHD ───────────────────────────────────────────────────────────────
def shd(seq: str) -> float:
    lam = np.array([LAMBDA[c] for c in seq])
    n = len(lam)
    i, j = np.triu_indices(n, 1)
    return float(((lam[i] + lam[j]) / (j - i)).sum() / n)


# ── all non-charge descriptors ────────────────────────────────────────
def descriptors(seq: str) -> dict:
    n = len(seq)
    out = {f'f_{a}': seq.count(a) / n for a in AA}
    out.update({k: sum(seq.count(a) for a in v) / n for k, v in SETS.items()})
    lam = np.array([LAMBDA[c] for c in seq])
    out['lambda_avg'] = float(lam.mean())
    out['lambda_std'] = float(lam.std())
    out['hydropathy_avg'] = float(np.mean([KD[c] for c in seq]))
    out['kappa'] = kappa(seq)
    out['SHD'] = shd(seq)
    return out


if __name__ == '__main__':
    # reference points from Das & Pappu (2013): sv1 = (EK)25 has kappa 0.0009, sv30 = E25K25 has kappa 1.0
    print('sv1  (EK)25 ', round(kappa('EK' * 25), 4))
    print('sv30 E25K25 ', round(kappa('E' * 25 + 'K' * 25), 4))
    print('K only      ', round(kappa('K' * 10 + 'G' * 40), 4))
    print('no charges  ', kappa('GS' * 20))
