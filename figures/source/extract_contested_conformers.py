"""Cache 2D projections of real conformers of one contested protist from both
force fields, for the cross-force-field disagreement figure, panel (f).

As extract_n0aza6_conformers.py: frames from the production half (the first
half is discarded, as in all BENDER analyses), eight evenly spaced conformers
per force field, superposed on the conformer whose Rg is closest to that force
field's median and projected onto its two principal axes. The MPIPI-Recharged
chain is rebuilt whole from minimum-image bonds if the box split it (as
recharged_io.load_run). Rg of every production frame is kept for the
density strip.

MPIPI-GG conformers are added when the protein is in the MPIPI-GG subset (2026-10-26).

Usage:  python extract_contested_conformers.py [--id A0A7S2N1D9] [--group Protists]
"""
import argparse
import json
import sys
from pathlib import Path

import mdtraj as md
import numpy as np
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

sys.dont_write_bytecode = True
sys.path.insert(0, str(paths.ANALYSIS))
import bender_common as bc  # noqa: E402

HERE = Path(__file__).resolve().parent
CALVADOS_RUNS = paths.TRAJ_C2
RECHARGED_RUNS = paths.TRAJ_RECHARGED
GG_RUNS = paths.TRAJ_GG


def box_nm(n_res):
    """Cubic box edge of the MPIPI-Recharged runs (recharged_io.box_nm)."""
    return float(int(n_res / 5) + 20 if n_res > 300 else 50)


def conformers(run, uid, unwrap_box=None, n=8, xtc=None, top=None, production_only=False):
    traj = md.load(str(xtc), top=str(top)) if xtc else md.load(str(run / f"{uid}.dcd"), top=str(run / "top.pdb"))
    seq = "".join(md.core.residue_names._AMINO_ACID_CODES.get(r.name, "X")
                  for r in traj.topology.residues)
    prod = traj if production_only else traj[traj.n_frames // 2:]
    xyz = prod.xyz.astype(float)
    raw = bc.max_bond(xyz)
    if unwrap_box is not None and raw > 1.0:
        xyz = bc.unwrap_chain(xyz, unwrap_box)
    rg = np.sqrt(((xyz - xyz.mean(1, keepdims=True)) ** 2).sum(2).mean(1))
    frames = np.linspace(0, len(xyz) - 1, n).astype(int)
    ref = int(np.argmin(np.abs(rg - np.median(rg))))
    pick = [ref] + [f for f in frames if f != ref][:n - 1]
    sel = md.Trajectory(xyz[pick], prod.topology)
    sel.superpose(sel, 0)
    c = sel.xyz - sel.xyz[0].mean(0)
    _, _, vt = np.linalg.svd(c[0], full_matrices=False)
    proj = c @ vt[:2].T
    return dict(n_res=int(traj.n_residues), production_frames=int(len(xyz)),
                max_bond_raw_nm=round(raw, 3), max_bond_nm=round(bc.max_bond(xyz), 3),
                rg_median_nm=float(np.median(rg)), rg_frames_nm=rg.round(3).tolist(),
                conformers_nm=proj.round(4).tolist()), seq


def main(uid, group):
    c2, s1 = conformers(CALVADOS_RUNS / group / uid, uid)
    n_res = c2["n_res"]
    mr, s2 = conformers(RECHARGED_RUNS / uid, uid, unwrap_box=box_nm(n_res))
    assert s1 == s2, "the two force fields simulated different sequences"
    record = {"protein": uid, "group": group, "sequence": s1, "calvados2": c2, "recharged": mr}
    gg_dir = GG_RUNS / uid
    if (gg_dir / "traj.xtc").exists():
        # MPIPI-GG (LAMMPS): the xtc holds production frames only (dump delayed to the second half of the run);
        # the CALVADOS-2 topology has the same 118 one-bead residues
        gg, _ = conformers(None, uid, unwrap_box=50.0, xtc=gg_dir / "traj.xtc",
                           top=CALVADOS_RUNS / group / uid / "top.pdb", production_only=True)
        record["mpipi_gg"] = gg
    out = HERE / f"{uid.lower()}_conformers.json"
    json.dump(record, open(out, "w"))
    for k, v in (("CALVADOS-2", c2), ("MPIPI-Recharged", mr), ("MPIPI-GG", record.get("mpipi_gg"))):
        if v is None:
            continue
        print(f"{k}: {v['production_frames']} production frames, median Rg {v['rg_median_nm']:.2f} nm, "
              f"max bond {v['max_bond_raw_nm']} -> {v['max_bond_nm']} nm")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="A0A7S2N1D9")
    ap.add_argument("--group", default="Protists")
    a = ap.parse_args()
    main(a.id, a.group)
