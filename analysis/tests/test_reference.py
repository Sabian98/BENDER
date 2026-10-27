"""
Reference checks for bender_common (reanalysis plan, 2026-09-27).
Tolerance ±0.002 unless stated.  Run:  python3 -m pytest -q tests/
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bender_common as bc  # noqa: E402

REF = os.path.join(os.path.dirname(HERE), 'reference')
CAL_DCD = os.path.join(REF, 'A0A5J4X754_K_to_R.dcd')
CAL_TOP = os.path.join(REF, 'top.pdb')
MP_DIR = os.path.join(REF, 'A0A0E1W5D6')
TOL = 0.002


@pytest.fixture(scope='module')
def cal_prod():
    return bc.load_calvados(CAL_DCD, CAL_TOP)


@pytest.fixture(scope='module')
def mp_raw():
    import mdtraj as md
    with md.formats.XTCTrajectoryFile(os.path.join(MP_DIR, 'traj.xtc')) as fh:
        xyz, _, _, box = fh.read()
    return xyz.astype(float), box


# ── CALVADOS-2: A0A5J4X754_K_to_R ─────────────────────────────────────
def test_calvados_frames(cal_prod):
    xyz, seq, info = cal_prod
    assert info['n_frames_total'] == 10000
    assert info['n_frames_production'] == 5000
    assert len(seq) == xyz.shape[1] == 106
    assert bc.max_bond(xyz) < 0.5          # DCDs are not wrapped


def test_calvados_nu_primary(cal_prod):
    xyz, _, _ = cal_prod
    assert abs(bc.nu_primary_from_xyz(xyz)['nu'] - 0.630) <= TOL


def test_calvados_nu_block_se(cal_prod):
    xyz, _, _ = cal_prod
    assert abs(bc.block_se(xyz, bc.nu_primary_from_xyz, 'nu') - 0.009) <= 0.002


def test_calvados_nu_legacy_production_200():
    xyz, _, _ = bc.load_calvados(CAL_DCD, CAL_TOP, k=200)
    assert len(xyz) == 200
    assert abs(bc.nu_legacy_from_xyz(xyz)['nu_legacy'] - 0.649) <= TOL


def test_calvados_nu_legacy_all_frames_200():
    xyz, _, _ = bc.load_calvados(CAL_DCD, CAL_TOP, k=200, production=False)
    assert abs(bc.nu_legacy_from_xyz(xyz)['nu_legacy'] - 0.662) <= TOL


def test_calvados_arg_aromatic_frequency(cal_prod):
    xyz, seq, _ = cal_prod
    F = bc.contact_frequency_map(xyz)
    parts = [bc.pair_type_frequency(F, seq, 'R', a) for a in 'YFW']
    num = sum(f * n for f, n in parts if n)
    den = sum(n for _, n in parts)
    assert abs(num / den - 0.0047) <= 0.0002


def test_calvados_i_i2_edge_share(cal_prod):
    xyz, _, _ = cal_prod
    adj = bc.adjacency(xyz[::50], min_sep=2)
    n = xyz.shape[1]
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    share = adj[:, sep == 2].sum() / adj.sum()
    assert abs(share - 0.90) <= 0.015


def test_contact_graph_excludes_i_i2(cal_prod):
    xyz, _, _ = cal_prod
    adj = bc.adjacency(xyz[:5])
    n = xyz.shape[1]
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    assert not adj[:, sep < 3].any()


# ── MPIPI-GG: A0A0E1W5D6 ──────────────────────────────────────────────
def test_mpipi_is_wrapped_raw(mp_raw):
    xyz, _ = mp_raw
    assert len(xyz) == 5002
    assert bc.max_bond(xyz) > 10          # raw file has broken chains


def test_mpipi_unwrap(mp_raw):
    xyz, box = mp_raw
    u = bc.unwrap_chain(xyz, box)
    assert bc.max_bond(u) < 0.5
    assert abs(10 * bc.geometry(u)['rg'] - 22.6) <= 0.1
    assert abs(10 * bc.geometry(xyz)['rg'] - 50.1) <= 0.2


def test_mpipi_loader_keeps_all_frames():
    seq = bc.read_mpipi_sequence(os.path.join(MP_DIR, 'seq.dat'))
    xyz, info = bc.load_mpipi(os.path.join(MP_DIR, 'traj.xtc'), n_res=len(seq))
    assert info['n_frames_production'] == info['n_frames_total'] == 5002
    assert info['max_bond_nm'] < 0.5


def test_mpipi_nu_primary(mp_raw):
    xyz, box = mp_raw
    u = bc.unwrap_chain(xyz, box)
    assert abs(bc.nu_primary_from_xyz(u)['nu'] - 0.495) <= TOL
    assert abs(bc.nu_primary_from_xyz(xyz)['nu'] - 0.270) <= TOL
    assert abs(bc.block_se(u, bc.nu_primary_from_xyz, 'nu') - 0.011) <= 0.002


def test_mpipi_nu_legacy_200(mp_raw):
    xyz, box = mp_raw
    i = bc.even_indices(len(xyz), 200)
    u = bc.unwrap_chain(xyz, box)
    raw_fit, unw_fit = bc.nu_legacy_from_xyz(xyz[i]), bc.nu_legacy_from_xyz(u[i])
    assert abs(raw_fit['nu_legacy'] - 0.451) <= TOL
    assert abs(unw_fit['nu_legacy'] - 0.453) <= TOL
    assert abs(raw_fit['nu_legacy_r2'] - 0.81) <= 0.01
    assert abs(unw_fit['nu_legacy_r2'] - 0.95) <= 0.01


def test_mpipi_contacts_minimum_image(mp_raw):
    xyz, box = mp_raw
    i = bc.even_indices(len(xyz), 200)
    u = bc.unwrap_chain(xyz, box)
    F_mi = bc.contact_frequency_map(xyz[i], box=50.0)
    F_u = bc.contact_frequency_map(u[i])
    assert np.allclose(F_mi, F_u)
    assert abs(F_u.sum() / 2 - 14.1) <= 0.15


# ── helpers ───────────────────────────────────────────────────────────
def test_holm():
    adj = bc.holm([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def test_zones():
    assert bc.zone(0.539) == 'compact'
    assert bc.zone(0.54) == 'transition' and bc.zone(0.63) == 'transition'
    assert bc.zone(0.631) == 'expanded'
