"""
mpipi_swap.py: parameter-swap edits to an OpenMpipi (MPIPI-Recharged) system, for the causal test of the
CALVADOS-2 / MPIPI-Recharged disagreement (BENDER JCTC, 2026-10-20).

A swap table (CSV) lists edits to the 21-type interaction tables that get_mpipi_system builds:
  kind=wf_eps     a, b, value   set the Wang-Frenkel epsilon of residue pair (a, b) (kJ/mol), both orders
  kind=yk_scale   a, -, value   multiply every Yukawa amplitude involving residue a by value (A_aa by value^2)
Residues are one-letter codes. make_swap_tables.py writes the tables; apply_swap() is used by
run_openmpipi_swap.py and by make_swap_tables.py's check against OpenMM energies, so both see the same edits.
"""
import numpy as np
import openmm as mm
import pandas as pd

MP_ORDER = 'MGKTRADEYVLQWFSHNPCI'   # OpenMpipi mapping_dict indices 0-19 ('pM' 0 ... 'pI' 19); 20 = RNA
N0 = 21


def _tables(system):
    wf = next(f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce) and f.getNumTabulatedFunctions()
              and f.getTabulatedFunctionName(0) == 'wf_table')
    yk = next(f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce) and f.getNumTabulatedFunctions()
              and f.getTabulatedFunctionName(0) == 'A_table')
    return wf, yk


def apply_swap(system, swap, verbose=True):
    """Edit the wf_table and A_table of an OpenMpipi system in place. swap: DataFrame or CSV path."""
    if isinstance(swap, str):
        swap = pd.read_csv(swap)
    wf, yk = _tables(system)
    xs, ys, zs, wv = wf.getTabulatedFunction(0).getFunctionParameters()
    ax, ay, av = yk.getTabulatedFunction(0).getFunctionParameters()
    if (xs, ys, zs, ax, ay) != (N0, N0, 3, N0, N0):
        raise RuntimeError('OpenMpipi tables are not the 21-type tables this swap expects; check the version')
    W = np.array(wv).reshape(3, N0, N0)          # [z, y, x]; z = 0 epsilon, 1 sigma, 2 mu
    A = np.array(av).reshape(N0, N0)             # [y, x]
    idx = {c: i for i, c in enumerate(MP_ORDER)}
    for row in swap.itertuples():
        if row.kind == 'wf_eps':
            i, j = idx[row.a], idx[row.b]
            old = W[0, j, i]
            W[0, j, i] = W[0, i, j] = row.value
            if verbose:
                print(f'  swap: Wang-Frenkel eps {row.a}{row.b} {old:.4f} -> {row.value:.4f} kJ/mol', flush=True)
        elif row.kind == 'yk_scale':
            i = idx[row.a]
            A[i, :] *= row.value
            A[:, i] *= row.value
            if verbose:
                print(f'  swap: Yukawa amplitudes of {row.a} x{row.value:.4f} ({row.a}{row.a} x{row.value ** 2:.4f})',
                      flush=True)
        else:
            raise ValueError(f'unknown swap kind {row.kind}')
    wf.getTabulatedFunction(0).setFunctionParameters(N0, N0, 3, W.ravel())
    yk.getTabulatedFunction(0).setFunctionParameters(N0, N0, A.ravel())
    return system
