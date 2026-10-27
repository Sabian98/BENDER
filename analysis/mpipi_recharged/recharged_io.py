"""Loading MPIPI-Recharged runs (run_openmpipi_idp.py output) with the frame rules of every BENDER analysis.

Production frames are the second half of the DCD (bender_common.load_calvados), K_FRAMES of them evenly spaced, and
a chain split by the periodic box is rebuilt whole from minimum-image bonds before any geometry is computed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # analysis/
import bender_common as bc  # noqa: E402

K_FRAMES = 1000
EQUIL_FRAC = 0.5    # first half of the DCD frames discarded as equilibration (run_openmpipi_idp.py)


def box_nm(n_res: int) -> float:
    """Cubic box edge used by run_openmpipi_idp.py."""
    return float(int(n_res / 5) + 20 if n_res > 300 else 50)


def run_dirs(root: str, T: float, ids_file: str = None) -> dict:
    """{UniProt ID: run folder} for every finished run in <root>/<T>K/."""
    tdir = os.path.join(root, f'{int(T)}K')
    names = [l.strip() for l in open(ids_file) if l.strip()] if ids_file else sorted(os.listdir(tdir))
    return {u: os.path.join(tdir, u) for u in names
            if os.path.isfile(os.path.join(tdir, u, f'{u}.dcd')) and os.path.isfile(os.path.join(tdir, u, 'top.pdb'))}


def load_run(run_dir: str, uid: str):
    """(xyz nm [K_FRAMES, N, 3], sequence, info) with the chain whole."""
    xyz, seq, info = bc.load_calvados(os.path.join(run_dir, f'{uid}.dcd'), os.path.join(run_dir, 'top.pdb'), k=K_FRAMES)
    if info['production_start'] != int(info['n_frames_total'] * EQUIL_FRAC):
        raise ValueError(f"production starts at frame {info['production_start']} of {info['n_frames_total']}, "
                         f'expected a {EQUIL_FRAC:.0%} equilibration discard')
    info['max_bond_raw_nm'] = bc.max_bond(xyz)
    if info['max_bond_raw_nm'] > 1.0:                 # a chain split by the periodic box: rebuild it whole
        xyz = bc.unwrap_chain(xyz, box_nm(len(seq)))
    info['max_bond_nm'] = bc.max_bond(xyz)
    if 'X' in seq:
        raise ValueError(f'unknown residue name in top.pdb: {seq}')
    return xyz, seq, info
