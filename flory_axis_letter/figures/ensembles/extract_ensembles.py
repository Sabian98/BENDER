"""Extract aligned ensembles (20 evenly spaced production frames) for the representative IDPs in both force fields.
Usage: python -I extract_ensembles.py <BENDER_300K dir> <MPIPI Recharged/300K dir> <outdir>"""
import os, sys, numpy as np, mdtraj as md
CAL, RCH, OUT = sys.argv[1:4]
os.makedirs(f'{OUT}/frames', exist_ok=True)
REPS = [('A0A540LLY7', 'Plants', 'compact'), ('A0A0C3P091', 'Fungi', 'minimum'),
        ('Q95UN5', 'Protists', 'basic'), ('A0A091CXV9', 'Mammals', 'acidic')]
for uid, tax, tag in REPS:
    for ff, path in [('cal', f'{CAL}/{tax}/{uid}'), ('rch', f'{RCH}/{uid}')]:
        t = md.load(f'{path}/{uid}.dcd', top=f'{path}/top.pdb')
        prod = t[len(t) // 2:]
        idx = np.linspace(0, len(prod) - 1, 20).astype(int)
        rg_all = md.compute_rg(prod); rg = rg_all.mean()
        rep = idx[np.argmin(np.abs(rg_all[idx] - rg))]          # frame closest to the mean Rg goes first
        idx = np.r_[rep, idx[idx != rep]][:10]
        e = prod[idx]
        e.center_coordinates()
        e.superpose(e, 0)
        e.save_pdb(f'{OUT}/{tag}_{uid}_{ff}.pdb')
        for k in range(len(e)):                                  # one file per frame, frame 1 is the representative
            e[k].save_pdb(f'{OUT}/frames/{tag}_{uid}_{ff}_f{k + 1:02d}.pdb')
        print(uid, tax, ff, 'frames', len(t), 'residues', t.n_residues, 'mean Rg (nm)', round(float(rg), 2))
