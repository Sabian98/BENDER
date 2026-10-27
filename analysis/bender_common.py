"""
bender_common.py — shared definitions for the BENDER reanalysis (v2).

Every v2 script imports frame selection, the ν estimators, contact rules,
zone boundaries and statistics from here, so they cannot diverge again.

Conventions (reanalysis plan, 2026-09-27)
-----------------------------------------
Frames
    CALVADOS-2 DCDs include equilibration: production = frames[n//2:].
    MPIPI-GG traj.xtc files are production only: use every frame.
    n is always read from the file.
Subsampling
    Even stride across production: stride = max(1, n_prod // K). Never a tail window.
Unwrapping
    MPIPI-GG coordinates are wrapped into the periodic box. Every frame is rebuilt
    from minimum-image bond vectors before any geometry. CALVADOS DCDs are not wrapped.
ν, primary estimator (matches calvados.analysis.fit_scaling_exp)
    RMS distance per residue pair over frames, averaged per sequence separation s,
    then A0·s^ν fitted for s ≥ 5. (calvados fits s > 5; on the reference the two differ by 0.001.)
    Pair distances come from mdtraj.compute_distances.
ν, legacy estimator (the v1 BENDER values)
    Mean distance per pair, every pair with s ≥ 5 weighted equally.
Contacts
    Cα–Cα < 0.8 nm, |i−j| ≥ 3. Metric = per-pair contact frequency: fraction of frames
    in contact, averaged over all pairs of a residue-type pair. No division by
    composition.
Zones
    compact ν < 0.54, transition 0.54–0.63, expanded ν > 0.63.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from itertools import combinations_with_replacement

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════
CONTACT_CUTOFF_NM = 0.8
CONTACT_MIN_SEP   = 3          # |i-j| >= 3 for contacts and contact graphs
NU_S_MIN          = 5          # primary ν: fit separations s >= 5 (plan; calvados uses s > 5, Δν ≈ 0.001)
NU_LEGACY_S_MIN   = 5          # legacy ν: pairs with s >= 5
NU_R2_MIN         = 0.99       # exclude sequences whose primary fit has R² below this
N_BLOCKS          = 5          # block standard error of ν

ZONE_COMPACT_MAX    = 0.54
ZONE_EXPANDED_MIN   = 0.63
NU_BIN_WIDTH        = 0.02
NU_BIN_EDGES        = np.round(np.arange(0.30, 0.80 + 1e-9, NU_BIN_WIDTH), 4)
MIN_N_PER_BIN       = 10

N_BOOT              = 10_000
PSEUDOCOUNT         = 1e-4     # for log-ratios of per-pair contact frequencies

SIX_LINEAGES = ['Bacteria', 'Fungi', 'Mammals', 'Plants', 'Protists', 'Viruses']
BIO_CSV = str(paths.BIO)


def six_lineage_ids(bio_csv: str = BIO_CSV) -> set:
    """UniProt IDs whose current BENDER_BIO kingdom is one of the six lineages (the BENDER analysis set).
    Sequences outside them (other animals, or never in BENDER_BIO) are simulated but not analyzed."""
    import pandas as pd
    b = pd.read_csv(bio_csv, usecols=['UniProt_ID', 'kingdom'])
    return set(b.loc[b['kingdom'].isin(SIX_LINEAGES), 'UniProt_ID'].astype(str))

AA3_TO_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
            'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
            'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
            'TYR': 'Y', 'VAL': 'V'}
AA1_TO_NAME = {'R': 'Arg', 'K': 'Lys', 'E': 'Glu', 'D': 'Asp', 'Y': 'Tyr', 'F': 'Phe',
               'W': 'Trp', 'S': 'Ser', 'H': 'His', 'G': 'Gly', 'P': 'Pro', 'A': 'Ala',
               'Q': 'Gln', 'N': 'Asn', 'T': 'Thr', 'L': 'Leu', 'I': 'Ile', 'V': 'Val',
               'M': 'Met', 'C': 'Cys'}

# The 23 residue-type pairs used in the v1 contact tables
CONTACT_PAIRS = [
    ('R', 'Y'), ('R', 'F'), ('R', 'W'), ('K', 'Y'), ('K', 'F'), ('K', 'W'),
    ('R', 'E'), ('R', 'D'), ('K', 'E'), ('K', 'D'), ('R', 'R'), ('K', 'K'),
    ('R', 'S'), ('K', 'S'), ('Y', 'Y'), ('F', 'F'), ('Y', 'F'), ('E', 'E'),
    ('D', 'D'), ('E', 'Y'), ('D', 'Y'), ('E', 'F'), ('Y', 'S'),
]
# Partner set for Arg-vs-Lys comparisons (Methods): aromatic + charged + Ser
ARG_LYS_PARTNERS = ['Y', 'F', 'W', 'E', 'D', 'S']


def expected_frames(n_residues: int) -> int:
    """Frames a complete CALVADOS-2 300 K DCD should hold (prepare.py): ≤196 → 10k, ≤312 → 15k, else 20k."""
    return 10000 if n_residues <= 196 else 15000 if n_residues <= 312 else 20000


def zone(nu: float) -> str | None:
    if nu is None or not np.isfinite(nu):
        return None
    if nu < ZONE_COMPACT_MAX:
        return 'compact'
    if nu <= ZONE_EXPANDED_MIN:
        return 'transition'
    return 'expanded'


def nu_bin(nu):
    """Bin center on the shared 0.02 grid (edges at multiples of 0.02 from 0.30)."""
    nu = np.asarray(nu, dtype=float)
    idx = np.floor((nu - NU_BIN_EDGES[0]) / NU_BIN_WIDTH)
    return np.round(NU_BIN_EDGES[0] + (idx + 0.5) * NU_BIN_WIDTH, 4)


# ══════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════
def even_indices(n: int, k: int | None) -> np.ndarray:
    """Evenly spaced indices across n frames; all frames if k is None or k >= n."""
    if k is None or k >= n:
        return np.arange(n)
    stride = max(1, n // k)
    return np.arange(0, n, stride)


def load_calvados(dcd: str | list[str], top: str, k: int | None = None,
                  production: bool = True):
    """
    Load a CALVADOS-2 trajectory (one DCD, or restart pieces in order).
    Returns (xyz [F,N,3] in nm, sequence, info dict).
    """
    import mdtraj as md
    dcds = [dcd] if isinstance(dcd, str) else list(dcd)
    traj = md.load(dcds, top=top)
    ca = traj.topology.select('name CA')
    if len(ca) == 0:
        ca = np.array([next(iter(r.atoms)).index for r in traj.topology.residues])
    traj = traj.atom_slice(ca)
    n_total = traj.n_frames
    start = n_total // 2 if production else 0
    xyz = traj.xyz[start:]
    idx = even_indices(len(xyz), k)
    seq = ''.join(AA3_TO_1.get(r.name[:3].upper(), 'X') for r in traj.topology.residues)
    info = {'n_frames_total': int(n_total), 'n_frames_production': int(len(xyz)),
            'n_frames_used': int(len(idx)), 'stride': int(idx[1] - idx[0]) if len(idx) > 1 else 1,
            'production_start': int(start), 'dcd_files': dcds}
    return xyz[idx].astype(np.float64), seq, info


def read_mpipi_sequence(seq_dat: str) -> str:
    with open(seq_dat) as fh:
        toks = fh.read().split()
    seqs = [t for t in toks if t.isalpha() and len(t) > 5]
    if not seqs:
        raise ValueError(f'no sequence found in {seq_dat}')
    return max(seqs, key=len).upper()


def load_mpipi(xtc: str, n_res: int | None = None, k: int | None = None, unwrap: bool = True):
    """
    Load an MPIPI-GG traj.xtc (production frames only — do NOT discard half).
    Coordinates are unwrapped with minimum-image bond vectors.
    Returns (xyz [F,N,3] in nm, info dict).
    """
    import mdtraj as md
    with md.formats.XTCTrajectoryFile(xtc) as fh:
        xyz, time, step, box = fh.read()
    xyz = xyz.astype(np.float64)
    if n_res is not None and xyz.shape[1] != n_res:
        raise ValueError(f'{xtc}: {xyz.shape[1]} atoms vs sequence length {n_res}')
    info = {'n_frames_total': int(len(xyz)), 'n_frames_production': int(len(xyz)),
            'production_start': 0, 'box_nm': float(np.median(box[:, 0, 0])),
            'max_bond_raw_nm': float(max_bond(xyz))}
    if unwrap:
        xyz = unwrap_chain(xyz, box)
    info['max_bond_nm'] = float(max_bond(xyz))
    idx = even_indices(len(xyz), k)
    info['n_frames_used'] = int(len(idx))
    info['stride'] = int(idx[1] - idx[0]) if len(idx) > 1 else 1
    return xyz[idx], info


def unwrap_chain(xyz: np.ndarray, box) -> np.ndarray:
    """
    Rebuild each frame from minimum-image bond vectors.
    xyz: (F, N, 3); box: scalar L, (3,) lengths, or (F, 3, 3) box vectors (orthorhombic).
    """
    box = np.asarray(box, dtype=float)
    if box.ndim == 3:
        L = np.stack([box[:, 0, 0], box[:, 1, 1], box[:, 2, 2]], axis=1)[:, None, :]
    elif box.ndim == 1:
        L = box[None, None, :]
    else:
        L = float(box)
    b = np.diff(xyz, axis=1)
    b = b - L * np.round(b / L)
    first = xyz[:, :1, :]
    return first + np.concatenate([np.zeros_like(first), np.cumsum(b, axis=1)], axis=1)


def max_bond(xyz: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xyz, axis=1), axis=2).max())


# ══════════════════════════════════════════════════════════════════════
# Geometry and ν
# ══════════════════════════════════════════════════════════════════════
def geometry(xyz: np.ndarray) -> dict:
    """Unweighted Rg, end-to-end distance and asphericity (nm), frame means and SEs."""
    r = xyz - xyz.mean(axis=1, keepdims=True)
    T = np.einsum('fni,fnj->fij', r, r) / xyz.shape[1]
    ev = np.linalg.eigvalsh(T)
    tr = ev.sum(axis=1)
    rg = np.sqrt(tr)
    delta = 1.5 * ((ev - ev.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) / (tr ** 2 + 1e-12)
    ree = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=1)
    se = lambda x: float(x.std() / np.sqrt(len(x)))
    return {'rg': float(rg.mean()), 'rg_err': se(rg), 'ree': float(ree.mean()),
            'ree_err': se(ree), 'delta': float(delta.mean()), 'delta_err': se(delta)}


_TOP_CACHE: dict = {}


def _ca_topology(n: int):
    """Minimal one-chain, one-CA-per-residue mdtraj topology (cached by length)."""
    import mdtraj as md
    if n not in _TOP_CACHE:
        top = md.Topology()
        ch = top.add_chain()
        for _ in range(n):
            top.add_atom('CA', md.element.carbon, top.add_residue('GLY', ch))
        _TOP_CACHE[n] = top
    return _TOP_CACHE[n]


def as_mdtraj(xyz: np.ndarray):
    import mdtraj as md
    return md.Trajectory(xyz.astype(np.float32), _ca_topology(xyz.shape[1]))


def _pair_distances(xyz: np.ndarray, s_min: int):
    """All pairs with separation >= s_min; distances (F, P) in nm via mdtraj.compute_distances.
    Coordinates must already be whole (unwrap MPIPI-GG first), so no periodic imaging.
    Memory grows as F x P; use _pair_moments for long chains."""
    import mdtraj as md
    n = xyz.shape[1]
    ii, jj = np.triu_indices(n, k=s_min)
    d = md.compute_distances(as_mdtraj(xyz), np.stack([ii, jj], axis=1), periodic=False)
    return ii, jj, d.astype(np.float64)


def _pair_moments(xyz: np.ndarray, s_min: int, max_elems: float = 2e7):
    """Per-pair frame means of d and d² (s >= s_min), accumulated over frame chunks
    with mdtraj.compute_distances so memory stays bounded for long chains."""
    import mdtraj as md
    n = xyz.shape[1]
    ii, jj = np.triu_indices(n, k=s_min)
    pairs = np.stack([ii, jj], axis=1)
    s1 = np.zeros(len(ii)); s2 = np.zeros(len(ii))
    chunk = max(1, int(max_elems // max(1, len(ii))))
    for a in range(0, xyz.shape[0], chunk):
        d = md.compute_distances(as_mdtraj(xyz[a:a + chunk]), pairs, periodic=False).astype(np.float64)
        s1 += d.sum(axis=0); s2 += (d ** 2).sum(axis=0)
    f = xyz.shape[0]
    return ii, jj, s1 / f, s2 / f


def _power_fit(s, y):
    popt, pcov = curve_fit(lambda x, a0, v: a0 * np.power(x, v), s, y,
                           p0=[0.5, 0.55], maxfev=20000)
    pred = popt[0] * np.power(s, popt[1])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(popt[0]), float(popt[1]), float(np.sqrt(pcov[1, 1])), r2


def nu_primary_from_xyz(xyz: np.ndarray, s_min: int = NU_S_MIN) -> dict:
    """RMS distance per pair → mean per separation → fit s ≥ s_min."""
    ii, jj, _, msq = _pair_moments(xyz, s_min)
    rms = np.sqrt(msq)
    sep = jj - ii
    s_vals = np.unique(sep)
    per_s = np.array([rms[sep == s].mean() for s in s_vals])
    a0, nu, nu_fit_err, r2 = _power_fit(s_vals.astype(float), per_s)
    return {'nu': nu, 'A0': a0, 'nu_fit_err': nu_fit_err, 'nu_r2': r2}


def nu_legacy_from_xyz(xyz: np.ndarray, s_min: int = NU_LEGACY_S_MIN) -> dict:
    """v1 BENDER estimator: mean distance per pair, all pairs with s ≥ s_min."""
    ii, jj, md_, _ = _pair_moments(xyz, s_min)
    a0, nu, nu_fit_err, r2 = _power_fit((jj - ii).astype(float), md_)
    return {'nu_legacy': nu, 'A0_legacy': a0, 'nu_legacy_r2': r2}


def block_se(xyz: np.ndarray, fn, key: str, n_blocks: int = N_BLOCKS) -> float:
    """Standard error of an estimator across contiguous blocks of frames."""
    blocks = np.array_split(np.arange(xyz.shape[0]), n_blocks)
    vals = [fn(xyz[b])[key] for b in blocks if len(b) >= 2]
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def flory(xyz: np.ndarray) -> dict:
    """Primary ν with R² and block SE, plus the legacy ν."""
    out = nu_primary_from_xyz(xyz)
    out['nu_block_se'] = block_se(xyz, nu_primary_from_xyz, 'nu')
    out.update(nu_legacy_from_xyz(xyz))
    return out


# ══════════════════════════════════════════════════════════════════════
# Contacts
# ══════════════════════════════════════════════════════════════════════
def contact_frequency_map(xyz: np.ndarray, cutoff: float = CONTACT_CUTOFF_NM,
                          min_sep: int = CONTACT_MIN_SEP, box=None) -> np.ndarray:
    """
    (N, N) fraction of frames with Cα–Cα < cutoff, zero for |i−j| < min_sep.
    If box (scalar L or (3,)) is given, distances use the minimum image.
    """
    n = xyz.shape[1]
    freq = np.zeros((n, n))
    chunk = max(1, int(1.5e7 // (n * n)))
    for a in range(0, xyz.shape[0], chunk):
        diff = xyz[a:a + chunk, :, None, :] - xyz[a:a + chunk, None, :, :]
        if box is not None:
            L = np.asarray(box, dtype=float)
            diff = diff - L * np.round(diff / L)
        freq += (np.linalg.norm(diff, axis=3) < cutoff).sum(axis=0)
    freq /= xyz.shape[0]
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    freq[sep < min_sep] = 0.0
    return freq


def pair_type_frequency(freq: np.ndarray, seq: str, a: str, b: str) -> tuple[float, int]:
    """
    Mean per-pair contact frequency over all residue pairs of types (a, b) with
    |i−j| ≥ CONTACT_MIN_SEP (unordered pairs; same-type pairs counted once).
    Returns (mean frequency, number of pairs); (nan, 0) if no eligible pairs.
    """
    s = np.array(list(seq))
    n = len(s)
    ia, ib = np.where(s == a)[0], np.where(s == b)[0]
    if len(ia) == 0 or len(ib) == 0:
        return np.nan, 0
    I, J = np.meshgrid(ia, ib, indexing='ij')
    I, J = I.ravel(), J.ravel()
    keep = np.abs(I - J) >= CONTACT_MIN_SEP
    if a == b:
        keep &= I < J
    I, J = I[keep], J[keep]
    if len(I) == 0:
        return np.nan, 0
    assert freq.shape[0] == n
    return float(freq[I, J].mean()), int(len(I))


def pair_type_table(freq: np.ndarray, seq: str, pairs=CONTACT_PAIRS) -> dict:
    out = {}
    for a, b in pairs:
        f, npair = pair_type_frequency(freq, seq, a, b)
        key = f'{AA1_TO_NAME[a]}_{AA1_TO_NAME[b]}'
        out[f'freq_{key}'] = f
        out[f'npairs_{key}'] = npair
    return out


def adjacency(xyz: np.ndarray, cutoff: float = CONTACT_CUTOFF_NM,
              min_sep: int = CONTACT_MIN_SEP) -> np.ndarray:
    """(F, N, N) boolean contact graphs with |i−j| ≥ min_sep."""
    diff = xyz[:, :, None, :] - xyz[:, None, :, :]
    adj = np.linalg.norm(diff, axis=3) < cutoff
    sep = np.abs(np.arange(xyz.shape[1])[:, None] - np.arange(xyz.shape[1])[None, :])
    adj &= (sep >= min_sep)[None]
    return adj


# ══════════════════════════════════════════════════════════════════════
# Arg vs Lys summaries
# ══════════════════════════════════════════════════════════════════════
def pooled_ratio(num, den) -> float:
    """sum/sum ratio over sequences (ignores pairs where either side is missing)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    return float(num[ok].sum() / den[ok].sum())


def log_ratio(num, den, pseudocount: float = PSEUDOCOUNT):
    return np.log((np.asarray(num, float) + pseudocount) / (np.asarray(den, float) + pseudocount))


# ══════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════
def wilcoxon_paired(x, y=None) -> dict:
    x = np.asarray(x, float)
    d = x if y is None else x - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if len(d) < 5 or np.all(d == 0):
        return {'n': int(len(d)), 'median_diff': float(np.median(d)) if len(d) else np.nan,
                'mean_diff': float(d.mean()) if len(d) else np.nan, 'W': np.nan, 'p': np.nan}
    w = stats.wilcoxon(d)
    return {'n': int(len(d)), 'median_diff': float(np.median(d)), 'mean_diff': float(d.mean()),
            'frac_positive': float((d > 0).mean()), 'W': float(w.statistic), 'p': float(w.pvalue)}


def holm(pvals) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (NaNs passed through)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def bootstrap_ci(values, stat=np.mean, n_boot: int = N_BOOT, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Sequence-level percentile bootstrap CI of stat(values)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    bs = np.apply_along_axis(stat, 1, v[idx]) if stat is not np.mean else v[idx].mean(axis=1)
    return (float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2))))


def bootstrap_ratio_ci(num, den, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """CI of a pooled sum/sum ratio, resampling sequences."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


# ══════════════════════════════════════════════════════════════════════
# Provenance
# ══════════════════════════════════════════════════════════════════════
def _sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for block in iter(lambda: fh.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def write_provenance(out_path: str, script: str, params: dict, inputs: list[str]) -> str:
    """
    Write <out_path>.provenance.json next to an output: git commit (if the script
    is in a repository), SHA-256 of the script and of bender_common.py, parameters,
    and the input file list (with sizes and modification times).
    """
    script = os.path.abspath(script)
    try:
        commit = subprocess.run(['git', '-C', os.path.dirname(script), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    inp = []
    for p in inputs:
        try:
            st = os.stat(p)
            inp.append({'path': p, 'bytes': st.st_size,
                        'mtime': _dt.datetime.fromtimestamp(st.st_mtime).isoformat()})
        except OSError:
            inp.append({'path': p, 'missing': True})
    record = {
        'output': os.path.abspath(out_path),
        'created': _dt.datetime.now().isoformat(timespec='seconds'),
        'git_commit': commit,
        'script': script, 'script_sha256': _sha256(script),
        'bender_common_sha256': _sha256(os.path.abspath(__file__)),
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'params': params, 'n_inputs': len(inp), 'inputs': inp,
    }
    side = out_path + '.provenance.json'
    with open(side, 'w') as fh:
        json.dump(record, fh, indent=2, default=str)
    return side


# ══════════════════════════════════════════════════════════════════════
# Sequence charge descriptors (residue charges only; no terminal charges)
# ══════════════════════════════════════════════════════════════════════
def residue_charges(seq: str) -> np.ndarray:
    """+1 for K/R, −1 for D/E, 0 otherwise (His neutral). Termini are not charged."""
    return np.array([1.0 if c in 'KR' else -1.0 if c in 'DE' else 0.0 for c in seq.upper()])


def charge_descriptors(seq: str) -> dict:
    """
    FCR  = (K+R+D+E)/N
    NCPR = (K+R−D−E)/N
    SCD  = (1/N) Σ_{i<j} q_i q_j |i−j|^{1/2}   (Sawle & Ghosh 2015 sign convention:
           more negative = opposite charges more segregated; positive = like charges dominate)
    """
    q = residue_charges(seq)
    n = len(q)
    i, j = np.triu_indices(n, 1)
    return {'FCR': float(np.abs(q).mean()), 'NCPR': float(q.mean()),
            'SCD': float((q[i] * q[j] * np.sqrt(j - i)).sum() / n)}
