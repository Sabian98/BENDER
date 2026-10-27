"""
make_swap_tables.py: the parameter-swap tables for MPIPI-Recharged (run locally; writes the CSVs next to this file).

Each swap makes one class of MPIPI-Recharged pair interactions match CALVADOS-2, leaving everything else unchanged:
  swap_aro    Arg-aromatic and aromatic-aromatic pairs (RF RW RY FF FW FY WW WY YY): the Wang-Frenkel epsilon of each
              pair is set so that the pair's full B2 (Wang-Frenkel + Yukawa) equals its CALVADOS-2 B2
              (Ashbaugh-Hatch + Debye-Hueckel) at 300 K and 150 mM.
  swap_aliph  aliphatic pairs (A I L M V, all 15): the same B2 matching.
  swap_his    histidine charge: every Yukawa amplitude involving His is scaled by q_C2(His) / q_MR(His), where
              q_C2 = +0.0307 (CALVADOS-2, pH 7.5) and q_MR = sqrt(A_HH / A_KK) is the effective MPIPI-Recharged
              His charge (about +0.51).
  swap_all    the three together.
  swap_gsp    glycine, serine and proline pairs (GG GS GP SS SP PP): the same B2 matching. Added 2026-10-22 for the
              SAXS benchmark, where MPIPI-Recharged is too expanded for Gly/Pro-rich chains and these pairs are less
              attractive than in CALVADOS-2.
  swap_gspaliph  every pair among G S P A I L M V (36): the Gly/Ser/Pro pairs, the aliphatic pairs and the pairs
              between them, which together make up elastin-like chains such as (VPGVG)n. Added 2026-10-22.
B2 and both pair potentials come from b2_mechanism.py (BENDER-MT/.../Analysis_FDT_RECHARGED/b2_mechanism), whose
potentials were checked against OpenMM to 1e-8 kJ/mol. Here every swap table is checked again: the edited
OpenMM system (mpipi_swap.apply_swap on get_mpipi_system) must reproduce the edited analytic potential.

    ~/miniforge3/envs/openmpipi/bin/python make_swap_tables.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpipi_swap import apply_swap  # noqa: E402

B2_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'analysis', 'mpipi_recharged',
                         'b2_mechanism.py')
spec = importlib.util.spec_from_file_location('b2_mechanism', B2_SCRIPT)
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

ARO_PAIRS = ['RF', 'RW', 'RY', 'FF', 'FW', 'FY', 'WW', 'WY', 'YY']
ALIPH = 'AILMV'
ALIPH_PAIRS = [a + b for i, a in enumerate(ALIPH) for b in ALIPH[i:]]
GSP = 'GSP'
GSP_PAIRS = [a + b for i, a in enumerate(GSP) for b in GSP[i:]]
GSP_ALIPH_PAIRS = [a + b for a in GSP for b in ALIPH]
CHECK_D = np.array([0.42, 0.5, 0.58, 0.66, 0.75, 0.9, 1.1, 1.5, 2.0, 2.45, 3.0, 3.4])


def set_eps(a, b, eps):
    i, j = bm.MPI[a], bm.MPI[b]
    bm.MPW[0, j, i] = bm.MPW[0, i, j] = eps


def match_eps(a, b):
    """Wang-Frenkel epsilon giving B2_MR(a, b) = B2_C2(a, b).

    Because epsilon scales both the repulsive core and the attraction of the Wang-Frenkel form, B2 is not monotonic
    in epsilon: it rises from a soft core at small epsilon to a maximum (about 0.15-0.26 kJ/mol here) and then falls
    as attraction takes over. The match is made on the falling branch, which keeps a full excluded-volume core and
    contains the original epsilon of every swapped pair; the target must lie below the branch maximum."""
    i, j = bm.MPI[a], bm.MPI[b]
    eps0 = bm.MPW[0, j, i]
    target = bm.b2(bm.u_cal, a, b)
    b2_0 = bm.b2(bm.u_mp, a, b)
    f = lambda e: (set_eps(a, b, e), bm.b2(bm.u_mp, a, b))[1] - target
    scan = np.geomspace(1e-3, max(3 * eps0, 1.0), 80)
    vals = [f(e) for e in scan]
    k = int(np.argmax(vals))
    if vals[k] < 0:
        raise RuntimeError(f'{a}{b}: B2 {target:.3f} is above the Wang-Frenkel maximum {vals[k] + target:.3f}')
    lo, hi = scan[k], scan[k] * 2        # falling branch: f(lo) >= 0; raise hi until f(hi) < 0
    while f(hi) > 0:
        hi *= 2
        if hi > 1e3:
            raise RuntimeError(f'{a}{b}: cannot reach B2 {target:.3f} on the falling branch')
    for _ in range(80):                  # bisection in log epsilon
        mid = np.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    eps = np.sqrt(lo * hi)
    b2_new = f(eps) + target
    set_eps(a, b, eps0)                  # restore
    return dict(pair=a + b, a=a, b=b, eps_mr_original=eps0, eps_new=eps, b2_c2=target, b2_mr_original=b2_0,
                b2_mr_new=b2_new)


def openmm_two_beads(a, b, swap):
    top = bm.mm.app.Topology()
    for k, c in enumerate((a, b)):
        res = top.addResidue(c, top.addChain(id=f'c{k}'))
        top.addAtom('p' + c, bm.mm.app.Element.getBySymbol('C'), res)
    pos = np.array([[1.0, 1, 1], [1.0 + CHECK_D.max(), 1, 1]])
    system = bm.get_mpipi_system(pos, top, {}, bm.T, bm.IONIC * 1000, CM_remover=False, periodic=True)
    apply_swap(system, swap, verbose=False)
    return bm._energies(system, CHECK_D)


def check(swap, pairs):
    """Edited OpenMM energies against the edited analytic potential, for the swapped pairs and two untouched ones."""
    W0, A0 = bm.MPW.copy(), bm.MPA.copy()
    for r in swap.itertuples():           # edit the analytic tables the same way
        if r.kind == 'wf_eps':
            set_eps(r.a, r.b, r.value)
        else:
            i = bm.MPI[r.a]
            bm.MPA[i, :] *= r.value
            bm.MPA[:, i] *= r.value
    worst = 0.0
    for p in pairs + ['GS', 'KE']:
        e = openmm_two_beads(p[0], p[1], swap)
        m = np.column_stack(bm.u_mp(CHECK_D, p[0], p[1]))
        # relative to max(1 kJ/mol, |E|): steep cores reach 1e17 kJ/mol at 0.42 nm (Ile-Ile, mu = 12)
        worst = max(worst, float((np.abs(e - m) / np.maximum(1.0, np.abs(m))).max()))
    bm.MPW[:], bm.MPA[:] = W0, A0
    assert worst < 1e-9, f'OpenMM and analytic potentials differ by {worst:.2e} (relative)'
    return worst


def main():
    rows = [match_eps(p[0], p[1]) for p in ARO_PAIRS + ALIPH_PAIRS + GSP_PAIRS + GSP_ALIPH_PAIRS]
    t = pd.DataFrame(rows)
    t['class'] = ['aro' if p in ARO_PAIRS else 'aliph' if p in ALIPH_PAIRS else 'gsp' if p in GSP_PAIRS
                  else 'gsp-aliph' for p in t['pair']]
    q_c2 = float(bm.QCAL['H'])
    q_mr = float(np.sqrt(bm.MPA[bm.MPI['H'], bm.MPI['H']] / bm.MPA[bm.MPI['K'], bm.MPI['K']]))
    his = pd.DataFrame([dict(kind='yk_scale', a='H', b='', value=q_c2 / q_mr)])
    mk = lambda *cls: pd.DataFrame(dict(kind='wf_eps', a=t.loc[t['class'].isin(cls), 'a'],
                                        b=t.loc[t['class'].isin(cls), 'b'], value=t.loc[t['class'].isin(cls), 'eps_new']))
    swaps = {'aro': mk('aro'), 'aliph': mk('aliph'), 'his': his}
    swaps['all'] = pd.concat([swaps['aro'], swaps['aliph'], his], ignore_index=True)
    swaps['gsp'] = mk('gsp')
    swaps['gspaliph'] = mk('gsp', 'aliph', 'gsp-aliph')
    # B2 of the His pairs after the charge swap, for the record
    i = bm.MPI['H']
    A0 = bm.MPA.copy()
    bm.MPA[i, :] *= q_c2 / q_mr
    bm.MPA[:, i] *= q_c2 / q_mr
    his_rows = [dict(pair='H' + x, b2_c2=bm.b2(bm.u_cal, 'H', x), b2_mr_new=bm.b2(bm.u_mp, 'H', x)) for x in 'DEKRH']
    bm.MPA[:] = A0
    his_rows = [dict(r, b2_mr_original=bm.b2(bm.u_mp, 'H', r['pair'][1])) for r in his_rows]
    for name, s in swaps.items():
        pairs = [r.a + r.b for r in s.itertuples() if r.kind == 'wf_eps'] or ['HD', 'HK', 'HH']
        worst = check(s, pairs)
        s.to_csv(os.path.join(HERE, f'swap_{name}.csv'), index=False)
        print(f'swap_{name}.csv: {len(s)} edits; OpenMM check max relative |dE| {worst:.1e}')
    t.to_csv(os.path.join(HERE, 'swap_pairs_b2.csv'), index=False)
    pd.DataFrame(his_rows).to_csv(os.path.join(HERE, 'swap_his_b2.csv'), index=False)
    pd.set_option('display.width', 160)
    print(t[['pair', 'class', 'eps_mr_original', 'eps_new', 'b2_c2', 'b2_mr_original', 'b2_mr_new']].round(4).to_string(index=False))
    print(f'His charge: CALVADOS-2 {q_c2:.4f}, Mpipi-Recharged effective {q_mr:.4f}, Yukawa scale {q_c2 / q_mr:.4f}')
    print(pd.DataFrame(his_rows).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
