"""Cache 2D projections of real N0AZA6 conformers for Fig. 1 panel b.

Takes eight evenly spaced frames from the production half (the first half is
discarded, as in all BENDER analyses) of the CALVADOS-2 trajectory,
superposes them on the frame whose Rg is closest to the ensemble median, and
projects every conformer onto the two principal axes of that reference.
"""
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

RUN = paths.TRAJ_C2 / "Bacteria" / "N0AZA6"
OUT = Path(__file__).resolve().parent / "n0aza6_conformers.json"

traj = md.load(RUN / "N0AZA6.dcd", top=RUN / "top.pdb")
prod = traj[traj.n_frames // 2:]
rg = md.compute_rg(prod)
frames = np.linspace(0, prod.n_frames - 1, 8).astype(int)
ref = int(np.argmin(np.abs(rg - np.median(rg))))
sel = prod[[ref] + [f for f in frames if f != ref][:7]]
sel.superpose(sel, 0)
xyz = sel.xyz - sel.xyz[0].mean(0)
_, _, vt = np.linalg.svd(xyz[0] - xyz[0].mean(0), full_matrices=False)
proj = xyz @ vt[:2].T                       # (8, N, 2), nm
ree = np.linalg.norm(prod.xyz[:, -1] - prod.xyz[:, 0], axis=1)
json.dump({"protein": "N0AZA6", "n_res": int(traj.n_residues),
           "frames_used": int(prod.n_frames),
           "rg_median_nm": float(np.median(rg)), "ree_median_nm": float(np.median(ree)),
           "conformers_nm": proj.round(4).tolist()}, open(OUT, "w"))
print(OUT.name, "| residues", traj.n_residues, "| production frames", prod.n_frames,
      "| median Rg %.2f nm, Ree %.2f nm" % (np.median(rg), np.median(ree)),
      "| 2D span", np.ptp(proj.reshape(-1, 2), 0).round(2))
