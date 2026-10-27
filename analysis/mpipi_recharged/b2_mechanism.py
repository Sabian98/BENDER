"""
b2_mechanism.py — do the residue-pair potentials of CALVADOS-2 and MPIPI-Recharged explain the per-sequence
force-field disagreement dnu = nu(MPIPI-Recharged) - nu(CALVADOS-2) across the 10,282 BENDER sequences?

  1. B2 of the non-bonded pair potential for all 210 residue pairs in both force fields, 300 K, 150 mM
       B2 = -2 pi int_0^inf (exp(-u(r)/kT) - 1) r^2 dr
     CALVADOS-2: Ashbaugh-Hatch (lambda, sigma from residues_CALVADOS2.csv; eps_lj, cutoff_lj, cutoff_yu,
       fixed_lambda = calvados.inputmodels.SimulationInput defaults, which prepare.py does not override) plus
       Debye-Hueckel (calvados.interactions.genParamsDH; charges calvados.sequence.get_qs, flexhis, pH 7.5).
     MPIPI-Recharged: Wang-Frenkel plus pair-specific Yukawa from OpenMpipi data_files/recharged_params.txt,
       laid out as in run_openmpipi_idp.py apply_phospho (W [z, y, x], A [y, x]); kappa from
       OpenMpipi.system_building.calculate_debye_length; Wang-Frenkel zero beyond 3 sigma (and the 2.5 nm force
       cutoff), Yukawa truncated (unshifted) at 3.5 nm, as get_mpipi_system builds them.
     Both are checked first against the force field's own OpenMM energy (Reference platform) for six residue
     pairs at 11 distances; the script stops if any energy differs by more than 1e-6 kJ/mol.
     Residue-level only: CALVADOS-2 terminal charges (charge_termini both) have no MPIPI-Recharged counterpart
     and are not in the table.
  2. Pair classes (exclusive; the 'other' class takes the rest): aro-aro (F, W, Y), Arg-aro, Lys-aro,
     charge +/- (R, K with D, E), charge like (R/K with R/K, D/E with D/E), aliph-aliph (A, I, L, M, V),
     GSP-GSP (G, S, P), other.
  3. Per-sequence features: mean dB2 per residue pair, overall and by class (class terms sum to the overall), and
     the mean B2 in each force field. Pair weights:
       pairs3   every pair i < j with j - i >= 3 (primary; the separation used for the contact maps)
       comp     composition products x_a x_b (no sequence order)
       contact  pairs j - i >= 3 weighted by (j - i)^-3/2 (ideal-chain contact probability)
     The three X residues are glycine, as in BENDER_BIO and both simulations.
  4. 5-fold CV R2 of dnu, with the folds, ridge and standardization of paper_numbers_recharged.py (whose cv_r2 and
     FEATS are imported); the composition baselines (0.362 linear, 0.483 squared and pairwise) are recomputed and
     must match. Also the contested class (Arg + aromatic top decile, Lys below median): out-of-fold predictions of
     the global models scored inside the class, and models refit inside the class.

    python b2_mechanism.py [--out DIR] [--bio CSV] [--recharged CSV] [--numbers PY]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import openmm as mm  # noqa: E402
from openmm import unit  # noqa: E402
from scipy import constants as sc  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.integrate import simpson  # noqa: E402

import calvados  # noqa: E402
from calvados import interactions as cint  # noqa: E402
from calvados.inputmodels import SimulationInput  # noqa: E402
from calvados.sequence import get_qs  # noqa: E402
import OpenMpipi  # noqa: E402
from OpenMpipi.system_building import calculate_debye_length, get_mpipi_system  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)

DEFAULTS = dict(
    out=str(paths.B2),
    bio=str(paths.BIO),
    recharged=str(paths.RECHARGED / 'recharged_300K_nu.csv'),
    numbers=str(paths.FIGURE_SOURCE / 'paper_numbers_recharged.py'),
)
RESIDUES = str(paths.CALVADOS_RESIDUES)
MPIPI_PARAMS = os.path.join(os.path.dirname(OpenMpipi.__file__), 'data_files', 'recharged_params.txt')

T, IONIC, PH = 300.0, 0.150, 7.5
KT = sc.R * T / 1000.0                       # kJ/mol
AA = 'ACDEFGHIKLMNPQRSTVWY'                  # BENDER_BIO / CALVADOS order
NONSTD = {'O': 'K', 'U': 'C', 'B': 'D', 'Z': 'E', 'J': 'L', 'X': 'G'}   # CALVADOS IDP/prepare.py mapping
MP_ORDER = 'MGKTRADEYVLQWFSHNPCI'            # OpenMpipi mapping_dict indices 0-19 ('pM' 0 ... 'pI' 19)
ARO, ALIPH, GSP, POS, NEG = set('FWY'), set('AILMV'), set('GSP'), set('RK'), set('DE')
CLASSES = ['aro-aro', 'Arg-aro', 'Lys-aro', 'His-aro', 'charge +/-', 'charge like', 'His-acid', 'His-base',
           'aliph-aliph', 'GSP-GSP', 'other']
SCHEMES = ['pairs3', 'comp', 'contact']
MIN_SEP = 3


def pair_class(a, b):
    s = {a, b}
    if a in ARO and b in ARO:
        return 'aro-aro'
    if s & ARO and s - ARO in ({'R'}, {'K'}, {'H'}):
        return {'R': 'Arg-aro', 'K': 'Lys-aro', 'H': 'His-aro'}[(s - ARO).pop()]
    if 'H' in s and s - {'H'} <= NEG and s != {'H'}:
        return 'His-acid'
    if 'H' in s and s - {'H'} <= POS:
        return 'His-base'                         # HR, HK and HH
    if (a in POS and b in NEG) or (a in NEG and b in POS):
        return 'charge +/-'
    if (a in POS and b in POS) or (a in NEG and b in NEG):
        return 'charge like'
    if a in ALIPH and b in ALIPH:
        return 'aliph-aliph'
    if a in GSP and b in GSP:
        return 'GSP-GSP'
    return 'other'


# ============================================================================ CALVADOS-2
_F = SimulationInput.model_fields
CAL = dict(eps_lj_kcal=_F['eps_lj'].default, cutoff_lj=_F['cutoff_lj'].default, cutoff_yu=_F['cutoff_yu'].default,
           fixed_lambda=_F['fixed_lambda'].default)
CAL['eps_lj'] = CAL['eps_lj_kcal'] * 4.184   # calvados.sim.Sim: kcal -> kJ/mol
CAL['eps_yu'], CAL['k_yu'] = cint.genParamsDH(T, IONIC)
RES = pd.read_csv(RESIDUES).set_index('one')
QCAL = dict(zip(AA, get_qs(AA, flexhis=True, pH=PH, residues=RES)[0]))


def u_cal(r, a, b):
    """(Ashbaugh-Hatch, Debye-Hueckel) in kJ/mol, as calvados.interactions.init_ah/yu_interactions."""
    s = 0.5 * (RES.loc[a, 'sigmas'] + RES.loc[b, 'sigmas'])
    lam = 0.5 * (RES.loc[a, 'lambdas'] + RES.loc[b, 'lambdas'])
    rc, ry, k = CAL['cutoff_lj'], CAL['cutoff_yu'], CAL['k_yu']
    shift = (s / rc) ** 12 - (s / rc) ** 6
    with np.errstate(over='ignore', invalid='ignore'):
        x6 = (s / r) ** 6
        lj = x6 * x6 - x6
        ah = CAL['eps_lj'] * np.where(r < 2 ** (1 / 6) * s, 4 * (lj - lam * shift) + (1 - lam), 4 * lam * (lj - shift))
    ah = np.where(r < rc, ah, 0.0)
    yu = QCAL[a] * QCAL[b] * CAL['eps_yu'] * (np.exp(-k * r) / r - np.exp(-k * ry) / ry)
    return ah, np.where(r < ry, yu, 0.0)


# ============================================================================ MPIPI-Recharged
_NB = np.loadtxt(MPIPI_PARAMS)
_N0 = 21
assert _NB.size == 4 * _N0 * _N0, 'recharged_params.txt: 21 types expected'
MPW, MPA = _NB[:-_N0 * _N0].reshape(3, _N0, _N0), _NB[-_N0 * _N0:].reshape(_N0, _N0)   # [z, y, x], [y, x]
MPI = {c: i for i, c in enumerate(MP_ORDER)}
MP = dict(kappa=1.0 / calculate_debye_length(T, IONIC * 1000), cutoff_wf_force=2.5, cutoff_yukawa=3.5)


def u_mp(r, a, b):
    """(Wang-Frenkel, Yukawa) in kJ/mol, as OpenMpipi.system_building.get_mpipi_system."""
    i, j = MPI[a], MPI[b]
    eps, sig, mu = MPW[0, j, i], MPW[1, j, i], np.floor(MPW[2, j, i])
    rc = 3 * sig
    alpha = 2 * 3 ** (2 * mu) * (3 / (2 * (3 ** (2 * mu) - 1))) ** 3
    with np.errstate(over='ignore', invalid='ignore'):
        wf = eps * alpha * ((sig / r) ** (2 * mu) - 1) * ((rc / r) ** (2 * mu) - 1) ** 2
    wf = np.where((r < rc) & (r < MP['cutoff_wf_force']), wf, 0.0)
    yk = MPA[j, i] / r * np.exp(-MP['kappa'] * r)
    return wf, np.where(r < MP['cutoff_yukawa'], yk, 0.0)


# ============================================================================ check against OpenMM
def _energies(system, dists):
    L = 20.0
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L))
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName('Reference'))
    out = []
    for d in dists:
        ctx.setPositions([mm.Vec3(1, 1, 1), mm.Vec3(1 + d, 1, 1)] * unit.nanometer)
        out.append([ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    for g in (0, 1)])
    return np.array(out)


def openmm_cal(a, b, dists):
    ah, yu = cint.init_nonbonded_interactions(CAL['eps_lj'], CAL['cutoff_lj'], CAL['eps_yu'], CAL['k_yu'],
                                              CAL['cutoff_yu'], CAL['fixed_lambda'])
    system = mm.System()
    for c in (a, b):                              # as calvados.sim.Sim.add_particles for a protein
        system.addParticle(RES.loc[c, 'MW'])
        ah.addParticle([RES.loc[c, 'sigmas'] * unit.nanometer, RES.loc[c, 'lambdas'], 1])
        yu.addParticle([QCAL[c]])
    system.addForce(ah)
    system.addForce(yu)
    return _energies(system, dists)


def openmm_mp(a, b, dists):
    top = mm.app.Topology()
    for k, c in enumerate((a, b)):                # one bead per chain: no bond, no exclusion
        res = top.addResidue(c, top.addChain(id=f'c{k}'))
        top.addAtom('p' + c, mm.app.Element.getBySymbol('C'), res)
    pos = np.array([[1.0, 1, 1], [1.0 + max(dists), 1, 1]])
    return _energies(get_mpipi_system(pos, top, {}, T, IONIC * 1000, CM_remover=False, periodic=True), dists)


CHECK_PAIRS = [('R', 'Y'), ('K', 'E'), ('G', 'S'), ('H', 'D'), ('I', 'L'), ('W', 'W')]
CHECK_D = np.array([0.45, 0.55, 0.62, 0.70, 0.85, 1.0, 1.4, 1.9, 2.4, 3.0, 3.6])


def verify():
    rows = []
    for a, b in CHECK_PAIRS:
        for ff, ref, mine in (('CALVADOS-2', openmm_cal, u_cal), ('MPIPI-Recharged', openmm_mp, u_mp)):
            e = ref(a, b, CHECK_D)
            m = np.column_stack(mine(CHECK_D, a, b))
            for d, (e0, e1), (m0, m1) in zip(CHECK_D, e, m):
                rows.append(dict(ff=ff, pair=a + b, r_nm=d, openmm_short=e0, here_short=m0, openmm_elec=e1,
                                 here_elec=m1, abs_diff=max(abs(e0 - m0), abs(e1 - m1))))
    v = pd.DataFrame(rows)
    assert v['abs_diff'].max() < 1e-6, v.sort_values('abs_diff').tail()
    return v


# ============================================================================ B2
def b2(ufun, a, b, which='total', r0=0.05, dr=1e-4):
    """B2 in nm^3. Below r0 the Boltzmann factor is 0 (u > 1e3 kT for every pair), integrated analytically; above,
    Simpson on segments that end at every cutoff so that truncations fall on grid points."""
    if ufun is u_cal:
        edges = sorted({r0, CAL['cutoff_lj'], CAL['cutoff_yu']})
    else:
        edges = sorted({r0, 3 * MPW[1, MPI[b], MPI[a]], MP['cutoff_wf_force'], MP['cutoff_yukawa']})
    edges = [e for e in edges if e >= r0]
    assert sum(ufun(np.array([r0]), a, b))[0] > 1e3 * KT
    tot = 2 * np.pi * r0 ** 3 / 3
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(np.ceil((hi - lo) / dr))
        n += n % 2                                      # even number of intervals for Simpson
        r = np.linspace(lo, hi - 1e-12, n + 1)          # just inside the segment, so each cutoff is respected
        s, e = ufun(r, a, b)
        u = s + e if which == 'total' else s
        with np.errstate(over='ignore'):
            f = np.exp(-u / KT) - 1.0
        tot -= 2 * np.pi * simpson(f * r * r, x=r)
    return tot


def pair_table():
    rows = []
    for i, a in enumerate(AA):
        for b in AA[i:]:
            row = dict(res1=a, res2=b, pair=a + b, pair_class=pair_class(a, b))
            for ff, uf in (('C2', u_cal), ('MR', u_mp)):
                for w in ('total', 'short'):
                    row[f'B2_{ff}' + ('' if w == 'total' else f'_{w}')] = b2(uf, a, b, w)
            row['dB2'] = row['B2_MR'] - row['B2_C2']
            row['dB2_short'] = row['B2_MR_short'] - row['B2_C2_short']
            row['q_C2'] = QCAL[a] * QCAL[b]
            row['A_MR'] = MPA[MPI[b], MPI[a]]
            row['lambda_C2'] = 0.5 * (RES.loc[a, 'lambdas'] + RES.loc[b, 'lambdas'])
            row['sigma_C2'] = 0.5 * (RES.loc[a, 'sigmas'] + RES.loc[b, 'sigmas'])
            row['eps_MR'], row['sigma_MR'], row['mu_MR'] = (MPW[k, MPI[b], MPI[a]] for k in range(3))
            rows.append(row)
    return pd.DataFrame(rows)


def b2_convergence():
    """Largest change in B2 (nm^3) over a few pairs when dr is halved and the hard-core start moved to 0.03 nm."""
    return max(abs(b2(uf, a, b) - b2(uf, a, b, r0=0.03, dr=5e-5))
               for uf in (u_cal, u_mp) for a, b in CHECK_PAIRS + [('D', 'R'), ('H', 'H')])


# ============================================================================ per-sequence features
def clean(seq):
    return ''.join(NONSTD.get(c, c) for c in seq.upper() if c in AA or c in NONSTD)


def pair_matrices(seq, wtab):
    """Symmetric 20x20 pair-weight matrices (sum = total weight) for the three schemes."""
    idx = np.array([AA.index(c) for c in seq])
    n = len(idx)
    O = np.zeros((n, 20))
    O[np.arange(n), idx] = 1.0
    x = O.mean(0)
    P = np.cumsum(O, 0)
    C = P[:-MIN_SEP].T @ O[MIN_SEP:]                  # residue a at i, b at j, j - i >= MIN_SEP
    S3 = 0.5 * (C + C.T)
    Wm = wtab[:n, :n]
    Sc = O.T @ Wm @ O / 2.0                           # Wm counts each unordered pair twice
    return {'pairs3': S3, 'comp': np.outer(x, x), 'contact': Sc}


def features(seqs, pt):
    D = np.zeros((20, 20))
    B = {'C2': np.zeros((20, 20)), 'MR': np.zeros((20, 20))}
    M = {k: np.zeros((20, 20)) for k in CLASSES}
    for _, r in pt.iterrows():
        i, j = AA.index(r['res1']), AA.index(r['res2'])
        for (a_, b_) in ((i, j), (j, i)):
            D[a_, b_] = r['dB2']
            B['C2'][a_, b_], B['MR'][a_, b_] = r['B2_C2'], r['B2_MR']
            M[r['pair_class']][a_, b_] = 1.0
    nmax = max(len(s) for s in seqs)
    sep = np.abs(np.subtract.outer(np.arange(nmax), np.arange(nmax))).astype(float)
    with np.errstate(divide='ignore'):
        wtab = np.where(sep >= MIN_SEP, sep ** -1.5, 0.0)
    rows = []
    for s in seqs:
        mats = pair_matrices(s, wtab)
        row = {}
        for sch, S in mats.items():
            w = S / S.sum()
            row[f'{sch}_dB2'] = (w * D).sum()
            row[f'{sch}_B2_C2'] = (w * B['C2']).sum()
            row[f'{sch}_B2_MR'] = (w * B['MR']).sum()
            for k in CLASSES:
                row[f'{sch}_dB2[{k}]'] = (w * D * M[k]).sum()
                row[f'{sch}_frac[{k}]'] = (w * M[k]).sum()
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ regression
def load_numbers(path):
    spec = importlib.util.spec_from_file_location('paper_numbers_recharged', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def std(M):
    M = np.asarray(M, float)
    return np.nan_to_num((M - M.mean(0)) / M.std(0, ddof=1))


def cv_pred(M, y, folds, ridge):
    """Out-of-fold predictions; same solver as paper_numbers_recharged.py (cv_r2 ridge 1e-3, cvq ridge 1e-2)."""
    pred = np.zeros(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        A = np.column_stack([np.ones(tr.sum()), M[tr]])
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.column_stack([np.ones(te.sum()), M[te]]) @ w
    return pred


def r2_of(y, pred):
    return 1 - np.var(y - pred) / np.var(y)


def quad(M, cols):
    """Squares and pairwise products of the listed columns (as paper_numbers_recharged.py builds Xq)."""
    extra = []
    for i, c1 in enumerate(cols):
        extra.append(M[:, c1] ** 2)
        extra += [M[:, c1] * M[:, c2] for c2 in cols[i + 1:]]
    return np.column_stack([M] + extra)


def gbm_pred(M, y, folds):
    from sklearn.ensemble import HistGradientBoostingRegressor
    pred = np.zeros(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
                                          l2_regularization=1.0, random_state=0)
        pred[te] = m.fit(M[tr], y[tr]).predict(M[te])
    return pred


# ============================================================================ provenance
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(output, inputs, params):
    st = [(p, os.stat(p)) for p in inputs]
    prov = {
        'output': output,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': None,
        'script': os.path.abspath(__file__),
        'script_sha256': sha256(os.path.abspath(__file__)),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {'calvados': getattr(calvados, '__version__', None) or _dist_version('calvados'),
                     'openmpipi': _dist_version('openmpipi'), 'openmm': mm.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__},
        'params': params,
        'n_inputs': len(inputs),
        'inputs': [{'path': p, 'bytes': s.st_size, 'sha256': sha256(p) if os.path.isfile(p) else None,
                    'mtime': datetime.datetime.fromtimestamp(s.st_mtime).isoformat()} for p, s in st],
    }
    with open(output + '.provenance.json', 'w') as fh:
        json.dump(prov, fh, indent=2)


def _dist_version(name):
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


# ============================================================================ main
N_SEEDS = 20          # fold seeds for the spread of CV R2 (seed 0 = the folds of paper_numbers_recharged.py)


def pair_contributions(seqs, pt):
    """Per-sequence contribution of each of the 210 unordered pairs to the pairs3 mean dB2, and the pair weights
    (n x 210 each, columns in the order of pair_table)."""
    iu = np.triu_indices(20)
    assert list(pt['pair']) == [AA[i] + AA[j] for i, j in zip(*iu)]
    dB = pt['dB2'].to_numpy()
    out, wout = np.zeros((len(seqs), len(pt))), np.zeros((len(seqs), len(pt)))
    for k, seq in enumerate(seqs):
        S = _pairs3(seq)
        S = S / S.sum()
        wout[k] = np.where(iu[0] == iu[1], S[iu], 2 * S[iu])
        out[k] = wout[k] * dB
    return out, wout


def _pairs3(seq):
    idx = np.array([AA.index(c) for c in seq])
    O = np.zeros((len(idx), 20))
    O[np.arange(len(idx)), idx] = 1.0
    C = np.cumsum(O, 0)[:-MIN_SEP].T @ O[MIN_SEP:]
    return 0.5 * (C + C.T)


def paragraph(pt, ct, reg, d, cls):
    """3-4 sentence draft for the JCTC Results; every number is taken from the tables computed above."""
    c = ct.set_index('pair_class')
    R = reg.set_index(['model', 'weights'])
    ar = pt[pt['pair_class'].isin(['aro-aro', 'Arg-aro'])]
    s1 = R.loc[('dB2 + ln N', 'pairs3')]
    s9 = R.loc[('dB2, B2_C2, ln N with squares and products', 'pairs3')]
    qd = R.loc[('composition squared and pairwise (paper)', 'composition')]
    both = R.loc[('composition squared and pairwise + 9-term B2 model', 'composition+pairs3')]
    pct = lambda x: f'{100 * x:.0f}\\%'  # noqa: E731
    shared = (s9['cv_r2'] + qd['cv_r2'] - both['cv_r2']) / max(s9['cv_r2'], qd['cv_r2'])
    shared = round(20 * shared) / 20                     # to the nearest 5%
    off_all, off_ra = c['contested_minus_rest'].sum(), c.loc['Arg-aro', 'contested_minus_rest']
    assert off_all < 0 and abs(off_ra / off_all - 1) < 0.1, 'the "all of it from Arg-aromatic" wording no longer holds'
    num = lambda x: f'${x:.1f}$'  # noqa: E731
    return (
        'To ask whether these differences are already written into the potentials, we computed the second virial '
        'coefficient $B_2$ of the non-bonded pair potential of each force field for all 210 residue pairs at 300 K and '
        '150 mM. Arginine--aromatic and aromatic--aromatic pairs are close to ideal under CALVADOS-2 '
        f"($|B_2| \\le {ar['B2_C2'].abs().max():.2f}$ nm$^3$) but strongly attractive under Mpipi-Recharged ($B_2$ from "
        f"{num(ar['B2_MR'].max())} to {num(ar['B2_MR'].min())} nm$^3$), whereas lysine--aromatic and aliphatic pairs are "
        'more repulsive under MPIPI-Recharged, and histidine, nearly neutral in CALVADOS-2 at pH 7.5 but partially '
        'charged in MPIPI-Recharged, becomes attractive to acidic and repulsive to basic residues. Averaged over the '
        f"residue pairs of each sequence, the difference $\\Delta B_2$ predicts {pct(s1['cv_r2'])} of the variance of "
        '$\\Delta\\nu$ under the same cross-validation, as much as the linear compositional model, and together with '
        f"the CALVADOS-2 $B_2$ and their squares and products it predicts {pct(s9['cv_r2'])} with {int(s9['n_terms'])} "
        f"terms, as much as the {int(qd['n_terms'])}-term compositional model; combined, the two reach "
        f"{pct(both['cv_r2'])}, so that about {pct(shared)} of what either explains is shared. In the contested class "
        f"the mean $\\Delta B_2$ is lower than in the other sequences by {abs(off_all):.3f} nm$^3$, all of it from the "
        'arginine--aromatic term, and the $B_2$ model predicts $\\Delta\\nu$ there slightly better than composition '
        f"(out-of-fold $R^2 = {s9['contested_oof_r2']:.2f}$ against {qd['contested_oof_r2']:.2f}); the compositional "
        'dependence of the disagreement is therefore largely written into the pair potentials, above all into the '
        'arginine--aromatic and aromatic--aromatic attractions that MPIPI-Recharged strengthens.')


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f'--{k}', default=v)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = lambda name: os.path.join(a.out, name)  # noqa: E731
    pn = load_numbers(a.numbers)

    # ------------------------------------------------------------ 1. check, then B2 for 210 pairs
    chk = verify()
    chk.to_csv(P('b2_openmm_check.csv'), index=False)
    conv = b2_convergence()
    assert conv < 1e-5, conv
    pt = pair_table()

    # ------------------------------------------------------------ data (as paper_numbers_recharged.py)
    b = pd.read_csv(a.bio, low_memory=False)
    b = b[b['kingdom'].isin(pn.ORDER)].set_index('UniProt_ID')
    mr = pd.read_csv(a.recharged).set_index('UniProt_ID').reindex(b.index)
    assert mr['nu'].notna().all()
    n, g = b['n_residues'].to_numpy(float), b['kingdom'].to_numpy()
    d = (mr['nu'] - b['nu']).to_numpy()
    noise = ((b['nu_block_se'] ** 2 + mr['nu_block_se'] ** 2).mean() / np.var(d, ddof=1))   # as paper_numbers_recharged
    seqs = [clean(s) for s in b['sequence']]
    assert all(len(s) == k for s, k in zip(seqs, n)), 'cleaned sequence length differs from n_residues'
    n_x = sum(s.count('X') for s in b['sequence'])
    ra = b['f_R'] + b['f_aro']
    cls = ((ra > ra.quantile(0.9)) & (b['f_K'] < b['f_K'].median())).to_numpy()

    # ------------------------------------------------------------ 3. per-sequence features
    F = features(seqs, pt)
    F.index = b.index
    contrib, wts = pair_contributions(seqs, pt)
    assert np.allclose(contrib.sum(1), F['pairs3_dB2'].to_numpy())
    tot = F['pairs3_dB2'].to_numpy()
    pt['mean_pair_weight'] = wts.mean(0)
    pt['mean_contrib'] = contrib.mean(0)
    pt['var_share'] = [np.cov(c, tot)[0, 1] / tot.var(ddof=1) for c in contrib.T]
    pt['mean_contrib_contested'] = contrib[cls].mean(0)
    pt['mean_contrib_rest'] = contrib[~cls].mean(0)
    pt.to_csv(P('b2_pairs_300K_150mM.csv'), index=False, float_format='%.6g')

    out = pd.DataFrame({'UniProt_ID': b.index, 'kingdom': g, 'n_residues': n.astype(int),
                        'nu_calvados2': b['nu'].to_numpy(), 'nu_recharged': mr['nu'].to_numpy(), 'dnu': d,
                        'contested_class': cls}).set_index('UniProt_ID')
    out = out.join(F)
    out.to_csv(P('b2_features_per_sequence.csv'), float_format='%.6g')

    # ------------------------------------------------------------ 2. pair classes
    crow = []
    for k in CLASSES:
        s = pt[pt['pair_class'] == k]
        f = F[f'pairs3_dB2[{k}]'].to_numpy()
        crow.append({'pair_class': k, 'n_pairs': len(s), 'pairs': ' '.join(s['pair']),
                     'mean_B2_C2': s['B2_C2'].mean(), 'mean_B2_MR': s['B2_MR'].mean(),
                     'mean_dB2_per_pair': s['dB2'].mean(), 'min_dB2': s['dB2'].min(), 'max_dB2': s['dB2'].max(),
                     'pair_min': s.loc[s['dB2'].idxmin(), 'pair'], 'pair_max': s.loc[s['dB2'].idxmax(), 'pair'],
                     'mean_pair_weight': F[f'pairs3_frac[{k}]'].mean(),
                     'mean_contrib': f.mean(), 'sd_contrib': f.std(ddof=1),
                     'var_share': np.cov(f, tot)[0, 1] / tot.var(ddof=1),
                     'mean_contrib_contested': f[cls].mean(), 'mean_contrib_rest': f[~cls].mean(),
                     'contested_minus_rest': f[cls].mean() - f[~cls].mean(),
                     'spearman_with_dnu': stats.spearmanr(f, d)[0], 'pearson_with_dnu': stats.pearsonr(f, d)[0]})
    ct = pd.DataFrame(crow)
    ct.to_csv(P('b2_pair_classes.csv'), index=False, float_format='%.6g')

    # ------------------------------------------------------------ 4. regression
    X = b[pn.FEATS].copy()                          # paper_numbers_recharged.py composition matrix, verbatim
    X['logN'], X['absNCPR'] = np.log(n), b['NCPR'].abs()
    X = ((X - X.mean()) / X.std()).fillna(0).to_numpy()
    key = [pn.FEATS.index(f) for f in ('f_aro', 'f_R', 'f_K', 'hydropathy_avg', 'f_H', 'FCR')] + [X.shape[1] - 2]
    Xq = quad(X, key)
    folds0 = np.random.default_rng(0).integers(0, 5, len(d))
    paper_lin = pn.cv_r2(X, d)
    assert np.isclose(r2_of(d, cv_pred(X, d, folds0, 1e-3)), paper_lin)
    assert round(paper_lin, 3) == 0.362 and round(r2_of(d, cv_pred(Xq, d, folds0, 1e-2)), 3) == 0.483, \
        'composition baselines do not reproduce paper_numbers_recharged.py'
    L = np.log(n)[:, None]
    ncls = int(cls.sum())
    fcls = np.random.default_rng(0).integers(0, 5, ncls)
    seeds = [np.random.default_rng(s).integers(0, 5, len(d)) for s in range(N_SEEDS)]

    def B(sch):
        cl = [f'{sch}_dB2[{k}]' for k in CLASSES]
        fr = [f'{sch}_frac[{k}]' for k in CLASSES]
        dq = std(np.column_stack([F[f'{sch}_dB2'], F[f'{sch}_B2_C2'], L]))
        return [
            ('ln N only', std(L), 1e-3),
            ('dB2 + ln N', std(np.column_stack([F[f'{sch}_dB2'], L])), 1e-3),
            ('B2_C2 + B2_MR + ln N', std(np.column_stack([F[f'{sch}_B2_C2'], F[f'{sch}_B2_MR'], L])), 1e-3),
            ('dB2, B2_C2, ln N with squares and products', quad(dq, [0, 1, 2]), 1e-2),
            ('class dB2 (11) + ln N', std(np.column_stack([F[cl], L])), 1e-3),
            ('class pair fractions (11, no B2) + ln N', std(np.column_stack([F[fr], L])), 1e-3),
        ]

    models = [(m, 'pairs3', M, r) for m, M, r in B('pairs3')]
    models += [('composition linear (paper)', 'composition', X, 1e-3),
               ('composition squared and pairwise (paper)', 'composition', Xq, 1e-2)]
    cl3 = std(F[[f'pairs3_dB2[{k}]' for k in CLASSES]].to_numpy())
    models += [('composition linear + class dB2', 'composition+pairs3', np.column_stack([X, cl3]), 1e-3),
               ('composition squared and pairwise + class dB2', 'composition+pairs3', np.column_stack([Xq, cl3]), 1e-2)]
    b9 = quad(std(np.column_stack([F['pairs3_dB2'], F['pairs3_B2_C2'], L])), [0, 1, 2])
    models += [('composition squared and pairwise + 9-term B2 model', 'composition+pairs3', np.column_stack([Xq, b9]),
                1e-2)]
    for sch in ('comp', 'contact'):
        models += [(m, sch, M, r) for m, M, r in B(sch) if m != 'ln N only']

    rows = []
    for name, sch, M, ridge in models:
        p0 = cv_pred(M, d, folds0, ridge)
        rs = [r2_of(d, cv_pred(M, d, f, ridge)) for f in seeds]
        yc = d[cls]
        pc = cv_pred(std(M[cls]), yc, fcls, ridge)
        rows.append({'model': name, 'weights': sch, 'learner': 'ridge (paper solver)', 'n_terms': M.shape[1],
                     'ridge': ridge, 'cv_r2': r2_of(d, p0), 'cv_r2_mean_20seeds': np.mean(rs),
                     'cv_r2_sd_20seeds': np.std(rs, ddof=1),
                     'contested_oof_r2': 1 - np.mean((yc - p0[cls]) ** 2) / np.var(yc),
                     'contested_mean_pred': p0[cls].mean(), 'contested_mean_obs': yc.mean(),
                     'rest_mean_pred': p0[~cls].mean(), 'rest_mean_obs': d[~cls].mean(),
                     'contested_refit_cv_r2': r2_of(yc, pc)})
    cl_raw = F[[f'pairs3_dB2[{k}]' for k in CLASSES]].to_numpy()
    for name, M in (('GBM composition (29 features)', X),
                    ('GBM dB2, B2_C2, ln N', np.column_stack([F['pairs3_dB2'], F['pairs3_B2_C2'], L])),
                    ('GBM class dB2 (11) + B2_C2 + ln N', np.column_stack([cl_raw, F['pairs3_B2_C2'], L]))):
        p0 = gbm_pred(M, d, folds0)
        yc = d[cls]
        rows.append({'model': name, 'weights': 'composition' if 'composition' in name else 'pairs3',
                     'learner': 'HistGradientBoosting', 'n_terms': M.shape[1], 'ridge': np.nan,
                     'cv_r2': r2_of(d, p0), 'contested_oof_r2': 1 - np.mean((yc - p0[cls]) ** 2) / np.var(yc),
                     'contested_mean_pred': p0[cls].mean(), 'contested_mean_obs': yc.mean(),
                     'rest_mean_pred': p0[~cls].mean(), 'rest_mean_obs': d[~cls].mean(),
                     'contested_refit_cv_r2': r2_of(yc, gbm_pred(M[cls], yc, fcls))})
    # within one force field: how much of nu itself does the mean pair B2 explain?
    for lab, y, col in (('CALVADOS-2', b['nu'].to_numpy(), 'pairs3_B2_C2'), ('MPIPI-Recharged', mr['nu'].to_numpy(),
                                                                            'pairs3_B2_MR')):
        for name, M, ridge in ((f'within {lab}: nu from B2 + ln N, squares and products',
                                quad(std(np.column_stack([F[col], L])), [0, 1]), 1e-2),
                               (f'within {lab}: nu from composition linear', X, 1e-3)):
            rows.append({'model': name, 'weights': 'pairs3' if 'B2' in name else 'composition',
                         'learner': 'ridge (paper solver)', 'n_terms': M.shape[1], 'ridge': ridge,
                         'cv_r2': r2_of(y, cv_pred(M, y, folds0, ridge))})
    reg = pd.DataFrame(rows)
    reg.to_csv(P('b2_regression.csv'), index=False, float_format='%.4f')

    # ------------------------------------------------------------ summary
    R = reg.set_index(['model', 'weights'])
    get = lambda m, w='pairs3', c='cv_r2': R.loc[(m, w), c]  # noqa: E731
    sp = lambda x: stats.spearmanr(x, d)[0]  # noqa: E731
    L_ = []
    W = L_.append
    W('B2 mechanism test: residue-pair second virial coefficients of CALVADOS-2 and MPIPI-Recharged vs the per-sequence')
    W(f'disagreement dnu = nu(Mpipi-Recharged) - nu(CALVADOS-2), {len(d)} BENDER sequences ({datetime.date.today()})')
    W('')
    W('== Settings')
    W(f'T {T:g} K, ionic strength {IONIC * 1000:g} mM, kT {KT:.4f} kJ/mol. B2 = -2 pi int (exp(-u/kT) - 1) r^2 dr, nm^3; '
      'dB2 = B2(MPIPI-Recharged) - B2(CALVADOS-2), so dB2 < 0 means more attractive under MPIPI-Recharged.')
    W(f"CALVADOS-2 (calvados {_dist_version('calvados')}): Ashbaugh-Hatch eps = {CAL['eps_lj_kcal']:g} kcal/mol = "
      f"{CAL['eps_lj']:.4f} kJ/mol, lambda and sigma arithmetic means, cutoff {CAL['cutoff_lj']:g} nm (shifted); "
      f"Debye-Hueckel eps_yu {CAL['eps_yu']:.4f} kJ/mol nm, kappa {CAL['k_yu']:.4f} /nm, cutoff {CAL['cutoff_yu']:g} nm "
      f"(shifted); charges R, K +1, D, E -1, H {QCAL['H']:.4f} (pH {PH:g}). Terminal charges are not residue-pair terms "
      'and are left out (CALVADOS-2 charges both termini, MPIPI-Recharged neither).')
    W(f"Mpipi-Recharged (OpenMpipi {_dist_version('openmpipi')}): Wang-Frenkel (pair eps, sigma, mu; nu = 1; zero beyond "
      f"3 sigma) plus Yukawa A_ij exp(-kappa r)/r, kappa {MP['kappa']:.4f} /nm, truncated without shift at "
      f"{MP['cutoff_yukawa']:g} nm. The A table gives His an effective charge of about "
      f"+{np.sqrt(MPA[MPI['H'], MPI['H']] / MPA[MPI['K'], MPI['K']]):.2f} (A_HH / A_KK = "
      f"{MPA[MPI['H'], MPI['H']] / MPA[MPI['K'], MPI['K']]:.3f}); Arg-acid A is "
      f"{MPA[MPI['R'], MPI['D']] / MPA[MPI['K'], MPI['D']]:.2f} times Lys-acid A.")
    W(f'Check against OpenMM (Reference platform, two beads): {len(CHECK_PAIRS)} pairs x {len(CHECK_D)} distances '
      f'({CHECK_D.min():g}-{CHECK_D.max():g} nm) per force field, max |difference| {chk["abs_diff"].max():.1e} kJ/mol '
      f'(b2_openmm_check.csv). B2 integration: halving dr and starting at 0.03 nm changes B2 by at most {conv:.1e} nm^3.')
    W('CALVADOS IDP/prepare.py sets neither eps_lj nor the cutoffs, so the package defaults above apply (assumes the '
      'calvados on the cluster has the defaults of the local install).')
    W(f'Sequences: BENDER_BIO six groups; {n_x} X residues taken as Gly (as BENDER_BIO and both simulations).')
    W('')
    W('== Pair classes (pairs3 weights). mean dB2 = unweighted mean over the pairs of the class (nm^3); contribution = '
      'class term of the per-sequence mean dB2 per residue pair; var share = cov(class term, total) / var(total).')
    W(f"{'class':12s} {'pairs':>5s} {'B2 C2':>7s} {'B2 MR':>7s} {'mean dB2':>8s} {'range (pair)':>27s} "
      f"{'contrib':>8s} {'var share':>9s} {'contested':>9s} {'rest':>8s} {'diff':>8s} {'rho dnu':>7s}")
    for _, r in ct.iterrows():
        rng_ = f"{r['min_dB2']:+.2f} ({r['pair_min']}) .. {r['max_dB2']:+.2f} ({r['pair_max']})"
        W(f"{r['pair_class']:12s} {r['n_pairs']:5d} {r['mean_B2_C2']:+7.3f} {r['mean_B2_MR']:+7.3f} "
          f"{r['mean_dB2_per_pair']:+8.3f} {rng_:>27s} {r['mean_contrib']:+8.4f} {r['var_share']:9.3f} "
          f"{r['mean_contrib_contested']:+9.4f} {r['mean_contrib_rest']:+8.4f} {r['contested_minus_rest']:+8.4f} "
          f"{r['spearman_with_dnu']:+7.2f}")
    W(f"{'total':12s} {len(pt):5d} {'':7s} {'':7s} {pt['dB2'].mean():+8.3f} {'':>27s} {tot.mean():+8.4f} {1:9.3f} "
      f"{tot[cls].mean():+9.4f} {tot[~cls].mean():+8.4f} {tot[cls].mean() - tot[~cls].mean():+8.4f} {sp(tot):+7.2f}")
    oth = pt[pt['pair_class'] == 'other']
    has = lambda S: oth['pair'].apply(lambda p: bool(set(p) & S))  # noqa: E731
    aro_x, ali_x = has(ARO), has(ALIPH) & ~has(ARO)
    W(f"'other' ({len(oth)} pairs, var share {oth['var_share'].sum():.3f}): pairs with one aromatic "
      f"{oth.loc[aro_x, 'var_share'].sum():.3f} (mean dB2 {oth.loc[aro_x, 'dB2'].mean():+.3f}), with one aliphatic "
      f"{oth.loc[ali_x, 'var_share'].sum():.3f} (mean dB2 {oth.loc[ali_x, 'dB2'].mean():+.3f}), the rest "
      f"{oth.loc[~aro_x & ~ali_x, 'var_share'].sum():.3f} (mean dB2 {oth.loc[~aro_x & ~ali_x, 'dB2'].mean():+.3f}).")
    W('Most negative dB2: ' + ', '.join(f"{r['pair']} {r['dB2']:+.2f}" for _, r in pt.nsmallest(10, 'dB2').iterrows()))
    W('Most positive dB2: ' + ', '.join(f"{r['pair']} {r['dB2']:+.2f}" for _, r in pt.nlargest(10, 'dB2').iterrows()))
    W('Largest var share (pairs): ' + ', '.join(f"{r['pair']} {r['var_share']:+.3f}" for _, r in
                                                pt.reindex(pt['var_share'].abs().sort_values(ascending=False).index)
                                                .head(10).iterrows()))
    W('')
    W('== Per-sequence weights')
    W(f'Primary: every residue pair with |i - j| >= {MIN_SEP} (pairs3). Sensitivity: composition products x_a x_b (comp) '
      f'and |i - j|^-3/2 contact weights (contact). Pearson r of the per-sequence mean dB2, pairs3 vs comp '
      f"{stats.pearsonr(F['pairs3_dB2'], F['comp_dB2'])[0]:.4f}, pairs3 vs contact "
      f"{stats.pearsonr(F['pairs3_dB2'], F['contact_dB2'])[0]:.4f}.")
    W(f"Mean dB2 per pair: {tot.mean():+.4f} nm^3 (contested class {tot[cls].mean():+.4f}, rest {tot[~cls].mean():+.4f}); "
      f"Pearson r with dnu {stats.pearsonr(tot, d)[0]:.3f}, Spearman {sp(tot):.3f}; inside the contested class "
      f"r {stats.pearsonr(tot[cls], d[cls])[0]:.3f}.")
    W('')
    W(f'== Regression of dnu (5-fold CV R2; folds and solver of paper_numbers_recharged.py, seed 0; mean +/- SD over '
      f'{N_SEEDS} fold seeds; contested class n = {ncls}, mean dnu {d[cls].mean():+.4f})')
    W('   oof in class = out-of-fold predictions of the all-sequence model scored inside the class (R2 about the class mean);')
    W('   refit in class = the same model refit on the class alone, 5-fold CV inside it.')
    W(f"{'model':52s} {'weights':18s} {'terms':>5s} {'CV R2':>6s} {'20 seeds':>13s} {'oof in class':>12s} "
      f"{'pred mean':>9s} {'refit in class':>14s}")
    for _, r in reg.iterrows():
        if r['model'].startswith('within'):
            continue
        sd = '' if pd.isna(r.get('cv_r2_mean_20seeds')) else f"{r['cv_r2_mean_20seeds']:.3f}+/-{r['cv_r2_sd_20seeds']:.3f}"
        W(f"{r['model']:52s} {r['weights']:18s} {r['n_terms']:5d} {r['cv_r2']:6.3f} {sd:>13s} "
          f"{r['contested_oof_r2']:12.3f} {r['contested_mean_pred']:+9.4f} {r['contested_refit_cv_r2']:14.3f}")
    W('Within one force field (pairs3 mean B2 vs composition, CV R2 of nu itself): ' + '; '.join(
        f"{r['model'].replace('within ', '')} {r['cv_r2']:.3f}" for _, r in reg.iterrows() if r['model'].startswith('within')))
    W('')
    s1, s9 = get('dB2 + ln N'), get('dB2, B2_C2, ln N with squares and products')
    cl11, fr11 = get('class dB2 (11) + ln N'), get('class pair fractions (11, no B2) + ln N')
    lin, qd = get('composition linear (paper)', 'composition'), get('composition squared and pairwise (paper)', 'composition')
    W('== Reading')
    both = get('composition squared and pairwise + 9-term B2 model', 'composition+pairs3')
    W(f'A single physical feature, the pair-weighted mean dB2, with ln N predicts {s1:.3f} of var(dnu), as much as the '
      f'29-term composition model ({lin:.3f}); with the CALVADOS-2 mean B2 and squares and products (9 terms) it reaches '
      f'{s9:.3f}, the 57-term composition model being {qd:.3f}. The two overlap but are not the same: together they give '
      f'{both:.3f} (shared {s9 + qd - both:.3f}, about {(s9 + qd - both) / max(s9, qd):.0%} of either). Splitting dB2 '
      f'into 11 chemical classes gives {cl11:.3f}, against {fr11:.3f} for the same class pair fractions without B2 '
      'weights, so the B2 weights, not the grouping of pairs, carry the signal. Weighting pairs by ideal-chain contact '
      f"probability lowers every B2 model (dB2 + ln N {get('dB2 + ln N', 'contact'):.3f}): the disagreement follows "
      'interactions between all residue pairs, not sequence-local ones. A gradient-boosting check gives the same picture '
      f"(3 B2 features {get('GBM dB2, B2_C2, ln N'):.3f}, 29 composition features "
      f"{get('GBM composition (29 features)', 'composition'):.3f}). Mean pair B2 explains less of nu itself within "
      'either force field than composition does (see the within-force-field line above), so B2 describes the '
      'disagreement better than it describes nu. In the contested class the 9-term B2 '
      f"model scores {get('dB2, B2_C2, ln N with squares and products', c='contested_oof_r2'):.3f} out of fold "
      f"(composition {get('composition squared and pairwise (paper)', 'composition', 'contested_oof_r2'):.3f}) and "
      f"predicts a class mean of {get('dB2, B2_C2, ln N with squares and products', c='contested_mean_pred'):+.3f} "
      f'(observed {d[cls].mean():+.3f}).')
    W('')
    W('== Caveats')
    W(f'Sampling noise (block SE of both nu) is {noise:.1%} of var(dnu), so the ceiling for any model is about '
      f'{1 - noise:.2f}; the best models here reach {both:.2f}, leaving about {1 - noise - both:.2f} systematic but '
      'unexplained (sequence patterning, terminal charges, many-body effects are candidates).')
    W('Part of the B2-composition overlap is built in: the mean dB2 is the pair fractions weighted by fixed dB2 values. '
      'The informative result is that fixed weights from the potentials, with 2-9 fitted coefficients, match fits of '
      '29-57 coefficients, and beat the unweighted class fractions.')
    W('This is still a regression on dnu; the parameter-swap simulation is the causal test. B2 is a dilute two-body '
      'measure of each pair potential and ignores chain connectivity and many-body screening.')
    W('CALVADOS-2 terminal charges (+1 N, -1 C) are not in the residue-pair table; the termini control runs will show '
      'their share of dnu.')
    W('')
    W('== Draft mechanism paragraph (JCTC Results)')
    W(paragraph(pt, ct, reg, d, cls))
    W('')
    W('== Columns')
    W('b2_pairs_300K_150mM.csv: B2_C2, B2_MR, dB2 (nm^3, full potential); *_short, short-range term only (AH or WF); '
      'q_C2, product of CALVADOS-2 charges; A_MR, MPIPI-Recharged Yukawa amplitude (kJ/mol nm); lambda_C2, sigma_C2, '
      'eps_MR (kJ/mol), sigma_MR (nm), mu_MR, pair parameters; mean_pair_weight, mean share of the pair among the '
      'pairs3 residue pairs of a sequence; mean_contrib, its mean term in the per-sequence mean dB2; var_share, '
      'cov(term, total) / var(total); mean_contrib_contested / _rest, by contested class.')
    W('b2_features_per_sequence.csv: <weights>_dB2, _B2_C2, _B2_MR (mean per residue pair, nm^3); '
      '<weights>_dB2[class] (class term, sums to <weights>_dB2); <weights>_frac[class] (share of pairs in the class).')
    W('b2_regression.csv: as the table above; contested_mean_pred / rest_mean_pred are out-of-fold means.')
    W('')
    W('Outputs: b2_pairs_300K_150mM.csv (210 pairs), b2_pair_classes.csv, b2_features_per_sequence.csv, '
      'b2_regression.csv, b2_openmm_check.csv, this file; each CSV has a .provenance.json.')
    text = '\n'.join(L_)
    with open(P('b2_summary.txt'), 'w') as fh:
        fh.write(text + '\n')
    print(text)

    params = {'temperature_K': T, 'ionic_M': IONIC, 'pH': PH, 'kT_kJmol': KT,
              'calvados2': {k: float(v) for k, v in CAL.items()} | {'q_His': float(QCAL['H']),
                                                                    'residues': RESIDUES},
              'mpipi_recharged': MP | {'params': MPIPI_PARAMS, 'type_order': MP_ORDER,
                                       'layout': 'W = nb[:-441].reshape(3,21,21) [z,y,x]; A = nb[-441:].reshape(21,21)'},
              'b2': 'Simpson, dr <= 1e-4 nm, segments split at every cutoff; Boltzmann factor 0 below 0.05 nm',
              'b2_convergence_nm3': conv,
              'classes': {k: ' '.join(pt.loc[pt['pair_class'] == k, 'pair']) for k in CLASSES},
              'weights': {'pairs3': f'all i<j with j-i >= {MIN_SEP}', 'comp': 'x_a x_b',
                          'contact': f'(j-i)^-1.5 for j-i >= {MIN_SEP}'},
              'regression': 'paper_numbers_recharged.py: 5-fold, folds rng(0).integers(0,5,n), ridge 1e-3 (linear) '
                            'or 1e-2 (squared/pairwise) incl. intercept; features standardised',
              'contested_class': 'f_R + f_aro > 90th percentile and f_K < median (n %d)' % ncls,
              'sampling_share_var_dnu': float(noise),
              'nonstandard_residues': NONSTD,
              'numbers_script': a.numbers, 'numbers_script_sha256': sha256(a.numbers)}
    ins_pairs = [RESIDUES, MPIPI_PARAMS]
    ins_seq = ins_pairs + [a.bio, a.recharged]
    write_provenance(P('b2_pairs_300K_150mM.csv'), ins_seq, params)
    write_provenance(P('b2_openmm_check.csv'), ins_pairs, params)
    for f in ('b2_pair_classes.csv', 'b2_features_per_sequence.csv', 'b2_regression.csv', 'b2_summary.txt'):
        write_provenance(P(f), ins_seq + [a.numbers], params)


if __name__ == '__main__':
    main()
