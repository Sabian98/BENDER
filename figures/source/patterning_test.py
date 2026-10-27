"""Does sticker patterning explain the part of dnu that composition leaves unexplained? (2026-10-26)

Adds Das-Pappu-style blob patterning parameters to the compositional model of make_fig_disagreement.py (which already
contains SCD and SHD, the charge and hydropathy patterning descriptors) and compares five-fold cross-validated R^2.
Omega for a residue class is the mean squared deviation of its fraction in blobs of 5 and 6 residues from the
sequence-wide fraction, normalized by the most segregated arrangement of the same composition (0 evenly mixed,
1 fully clustered). Classes: aromatic (FWY), sticker (FWYR) and basic (RK).

Result (2026-10-26): CV R^2 0.483 -> 0.484 (+0.14 points); Spearman of the composition-model residual with
Omega_sticker -0.09 overall and +0.07 in the contested class.

Usage:  ~/miniforge3/envs/openmpipi/bin/python patterning_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fig_disagreement as fd  # noqa: E402

CLASSES = {"omega_aro": "FWY", "omega_sticker": "FWYR", "omega_RK": "RK"}


def omega(seq, members, blobs=(5, 6)):
    x = np.array([c in members for c in seq], float)
    n, f = len(x), x.mean()
    if f in (0.0, 1.0) or n < 7:
        return 0.0
    vals = []
    for g in blobs:
        def dev(arr):
            return np.mean((np.convolve(arr, np.ones(g) / g, mode="valid") - f) ** 2)
        seg = np.zeros(n)
        seg[: int(round(f * n))] = 1
        dmax = dev(seg)
        vals.append(dev(x) / dmax if dmax > 0 else 0.0)
    return float(np.mean(vals))


def main():
    b, mr, _ = fd.load()
    d = (mr["nu"] - b["nu"]).astype(float)
    seq = next(c for c in b.columns if c.lower() in ("sequence", "seq", "fasta"))
    for name, members in CLASSES.items():
        b[name] = [omega(s, members) for s in b[seq]]
    r2 = lambda p: 1 - np.var(d.to_numpy() - p) / np.var(d.to_numpy())  # noqa: E731
    base = fd.composition_model(b, d)
    feats = fd.FEATS
    fd.FEATS = feats + list(CLASSES)
    aug = fd.composition_model(b, d)
    fd.FEATS = feats
    res = d - base
    cls = ((b.f_R + b.f_aro) >= (b.f_R + b.f_aro).quantile(0.9)) & (b.f_K < b.f_K.median())
    print({c: round(b[c].corr(d, method="spearman"), 3) for c in CLASSES})
    print(f"CV R2 {r2(base):.3f} -> {r2(aug):.3f} (+{100 * (r2(aug) - r2(base)):.2f} points); residual vs omega_sticker "
          f"{res.corr(b.omega_sticker, method='spearman'):+.3f} overall, "
          f"{res[cls].corr(b.omega_sticker[cls], method='spearman'):+.3f} in the contested class (n {int(cls.sum())})")


if __name__ == "__main__":
    main()
