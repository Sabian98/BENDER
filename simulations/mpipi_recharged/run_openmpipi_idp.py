#!/usr/bin/env python
"""
Single-chain MPIPI-Recharged simulation of one IDP with OpenMpipi (OpenMM),
run under the same conditions as the CALVADOS IDP pipeline
(Scripts/CALVADOS IDP/prepare.py):

  - same FASTA / accession handling and non-standard residue mapping
  - same step budget, save interval and frame count (length-adaptive)
  - same cubic periodic box (50 nm, or N/5 + 20 nm above 300 residues)
  - 10 fs time step, Langevin-middle integrator, friction 0.01 / ps
  - 150 mM ionic strength, temperature from --temp
  - analysis with calvados.analysis.save_conf_prop (same Rg / Ree / nu,
    CALVADOS masses), discarding the first 50% of frames

Model differences that cannot be matched: MPIPI-Recharged has no terminal
charges and no pH dependence (CALVADOS runs used charge_termini='both', pH 7.5).

Phosphorylation (--phospho, added 2026-10-13): the letters B (pSer) and J (pThr)
of the CALVADOS phospho FASTA files are kept as phosphate-like Glu beads, the
representation of the PAGE4 Mpipi runs (GAIVI_PAGE4/mpipi/run_openmpipi_page4.py):
each gets its own particle type with Glu's Wang-Frenkel parameters and Glu's
Yukawa amplitudes times --phospho-scale (default 2, a fully phosphorylated site;
pairs of two such residues: times both scales). Without --phospho, B and J are
still replaced by D and L as before, so always pass it for phosphorylated runs.
Added 2026-10-15: Z (pTyr) is kept as a Tyr bead with its own type, Tyr's Wang-Frenkel
parameters and Glu's Yukawa amplitudes times --phospho-scale, so the aromatic ring keeps
its stickiness and gains the phosphate charge; top.pdb names these residues PTR. B and J
are handled exactly as before (systems verified identical). Without --phospho, Z -> E.

Output (same layout as CALVADOS):
  <outdir>/<T>K/<name>/<name>.dcd, top.pdb
  <outdir>/<T>K/data/<name>/conf_prop.csv (+ rgs.npy, rees.npy, ...)

The run is split into 10 segments; progress is saved after each, so re-running
the same command after a timeout resumes from the last finished segment.
All paths come from arguments; the script never changes directory.
"""
import os
from argparse import ArgumentParser
import json

import mdtraj as md
import numpy as np
import openmm as mm
import pandas as pd
from openmm import app, unit
from Bio import SeqIO

from OpenMpipi import IDP, get_mpipi_system

parser = ArgumentParser()
parser.add_argument('--name', required=True, type=str)
parser.add_argument('--fasta', required=True, type=str)
parser.add_argument('--outdir', type=str, default=os.path.expanduser('~/bender_runs/mpipi_recharged'),
                    help='root for <outdir>/<T>K/<name> and <outdir>/<T>K/data/<name>')
parser.add_argument('--temp', type=float, default=300.0)
parser.add_argument('--ionic', type=float, default=0.15, help='ionic strength, M')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--platform', default='CUDA', choices=['CUDA', 'CPU'])
parser.add_argument('--scale', type=float, default=1.0,
                    help='multiply the step budget (testing only; keep 1 for production)')
parser.add_argument('--phospho', action='store_true',
                    help='keep B (pSer) and J (pThr) as phosphate-like Glu beads instead of replacing them by D and L. '
                         'Added 2026-10-13.')
parser.add_argument('--phospho-scale', type=float, default=2.0,
                    help='Yukawa scale of the phosphate-like beads (2 = fully phosphorylated, as in the PAGE4 runs)')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
residues_file = f'{script_dir}/input/residues_CALVADOS2.csv'
fasta_path = os.path.abspath(os.path.expanduser(args.fasta))

# --- name and sequence: identical handling to CALVADOS prepare.py ---------
parts = args.name.split('|')
first = parts[0].strip()
if first in ('sp', 'tr') and len(parts) >= 2:
    input_name = parts[1].strip()
else:
    input_name = first.split()[0]
sysname = input_name

_NONSTANDARD_AA = {'O': 'K', 'U': 'C', 'B': 'D', 'Z': 'E', 'J': 'L', 'X': 'G'}
_VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')
_PHOSPHO = {'B': 'pSer', 'J': 'pThr', 'Z': 'pTyr'}   # with --phospho: phosphate-like beads (B, J 2026-10-13; Z 2026-10-15)
phospho_sites = []                          # 0-based positions of the phosphate-like beads
ptyr_sites = set()                          # the subset that are pTyr (Tyr bead with phosphate Yukawa)


def clean_sequence(seq):
    seq = seq.upper().replace('*', '')
    cleaned, replacements = [], {}
    for aa in seq:
        if aa in _VALID_AA:
            cleaned.append(aa)
        elif args.phospho and aa in _PHOSPHO:
            phospho_sites.append(len(cleaned))
            if aa == 'Z':
                ptyr_sites.add(len(cleaned))
                cleaned.append('Y')
            else:
                cleaned.append('E')
        elif aa in _NONSTANDARD_AA:
            replacements[aa] = _NONSTANDARD_AA[aa]
            cleaned.append(_NONSTANDARD_AA[aa])
    if replacements:
        print(f"  WARNING: non-standard residues replaced: {replacements}")
    return ''.join(cleaned)


seq = None
for record in SeqIO.parse(fasta_path, 'fasta'):
    rec_parts = record.id.split('|')
    rec_first = rec_parts[0].strip()
    record_id = rec_parts[1].strip() if rec_first in ('sp', 'tr') and len(rec_parts) >= 2 else rec_first
    if record_id == input_name:
        seq = clean_sequence(str(record.seq))
        break
if not seq:
    raise ValueError(f"Protein '{input_name}' not found in {fasta_path}")
N_res = len(seq)

# --- step budget: identical arithmetic to CALVADOS prepare.py --------------
if N_res <= 150:
    target_time_ns = 200
else:
    target_time_ns = int(200 * (N_res / 150) ** 1.5)
if target_time_ns <= 300:
    target_frames = 10000
elif target_time_ns <= 600:
    target_frames = 15000
else:
    target_frames = 20000
time_per_frame_ps = (target_time_ns * 1000) / target_frames
steps_per_frame = int(time_per_frame_ps / 0.02)  # nominal 20 fs, as in BENDER
timestep_ps = 0.01                               # real step, as in CALVADOS 2
if args.scale != 1.0:
    steps_per_frame = max(1, int(steps_per_frame * args.scale))
N_save = steps_per_frame
total_steps = target_frames * N_save
equilibration_frames = int(target_frames * 0.5)

n_segments = 10
frames_per_segment = target_frames // n_segments
segment_steps = frames_per_segment * N_save

L = int(N_res / 5) + 20 if N_res > 300 else 50  # nm, cubic box
temp = args.temp
csx = args.ionic * 1000  # mM for OpenMpipi

print(f"Configuring {sysname}:")
print(f"  Residues: {N_res}")
print(f"  Simulated time: {total_steps * timestep_ps / 1000:.1f} ns (10 fs step)")
print(f"  Frames: {target_frames}, every {N_save * timestep_ps:.1f} ps ({N_save} steps)")
print(f"  Discarded as equilibration: first {equilibration_frames} frames")
print(f"  Box: {L} nm, T = {temp} K, ionic = {args.ionic} M")
if args.phospho:
    print(f"  Phosphate-like beads (Glu for pSer/pThr, Tyr for pTyr; Yukawa x{args.phospho_scale:g}): "
          f"{[f'{seq[i]}{i + 1}' for i in phospho_sites] or 'none in this sequence'}")

temp_tag = f'{int(temp)}K'
path = f'{args.outdir.rstrip("/")}/{temp_tag}/{sysname}'
data_path = f'{args.outdir.rstrip("/")}/{temp_tag}/data/{sysname}'
os.makedirs(path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)


# --- starting conformation: self-avoiding random walk at the box center ----
# min_dist 0.6 nm: closer non-bonded contacts sit on the steep Wang-Frenkel
# wall and minimization cannot resolve them (NaN for long chains)
def saw_coords(n, bond=0.381, min_dist=0.6, rng=None, max_tries=200):
    rng = rng or np.random.default_rng()
    while True:
        xyz = np.zeros((n, 3))
        ok = True
        for i in range(1, n):
            for _ in range(max_tries):
                v = rng.normal(size=3)
                trial = xyz[i - 1] + bond * v / np.linalg.norm(v)
                if i < 2 or np.min(np.linalg.norm(xyz[:i - 1] - trial, axis=1)) > min_dist:
                    xyz[i] = trial
                    break
            else:
                ok = False
                break
        if ok:
            return xyz - xyz.mean(axis=0)


chain = IDP(sysname, seq)
topology = chain.topology
state_xml, progress_json = f'{path}/state.xml', f'{path}/progress.json'
resuming = os.path.exists(state_xml) and os.path.exists(progress_json)

positions = saw_coords(N_res, rng=np.random.default_rng(args.seed)) + L / 2
box = [mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L)] * unit.nanometer
topology.setPeriodicBoxVectors(box)
system = get_mpipi_system(positions, topology, {sysname: []}, temp, csx,
                          CM_remover=False, periodic=True)
system.setDefaultPeriodicBoxVectors(*box)


# --- phosphate-like residues (--phospho), as in the PAGE4 Mpipi runs --------
def apply_phospho(system, sites, seq):
    """Give each listed residue (a Glu bead) its own type: Glu's Wang-Frenkel parameters, Yukawa amplitudes x scale.
    sites: [(0-based index, scale)]. Copied from GAIVI_PAGE4/mpipi/run_openmpipi_page4.py (2026-10-10)."""
    for i, _ in sites:
        want = 'Y' if i in ptyr_sites else 'E'
        if seq[i] != want:
            raise ValueError(f'residue {i + 1} is {seq[i]}, expected a {want} bead')
    wf = next(f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce) and f.getNumTabulatedFunctions()
              and f.getTabulatedFunctionName(0) == 'wf_table')
    yu = next(f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce) and f.getNumTabulatedFunctions()
              and f.getTabulatedFunctionName(0) == 'A_table')
    n0, glu, tyr = 21, 7, 8                        # OpenMpipi type table size, Glu's and Tyr's index (mapping_dict 'pE', 'pY')
    nb = np.loadtxt(os.path.join(os.path.dirname(__import__('OpenMpipi').__file__), 'data_files', 'recharged_params.txt'))
    W0, A0 = nb[:-n0 * n0].reshape(3, n0, n0), nb[-n0 * n0:].reshape(n0, n0)   # [z, y, x] and [y, x]
    # guard against a different OpenMpipi version: the tables in the system must be the 21-type tables read here
    xs, ys, zs, wv = wf.getTabulatedFunction(0).getFunctionParameters()
    ax, ay, av = yu.getTabulatedFunction(0).getFunctionParameters()
    if (xs, ys, zs, ax, ay) != (n0, n0, 3, n0, n0) or not (np.allclose(wv, W0.ravel()) and np.allclose(av, A0.ravel())):
        raise RuntimeError('OpenMpipi tables differ from recharged_params.txt (21 types expected); check the version')
    base = [tyr if i in ptyr_sites else glu for i, _ in sites]
    if any(wf.getParticleParameters(i)[0] != b or yu.getParticleParameters(i)[0] != b for (i, _), b in zip(sites, base)):
        raise RuntimeError('a phosphosite is not on the expected particle type')
    n = n0 + len(sites)
    mw = np.r_[np.arange(n0), base]                                # Wang-Frenkel: Glu (pSer, pThr) or Tyr (pTyr)
    ma = np.r_[np.arange(n0), np.full(len(sites), glu)]           # Yukawa: Glu, scaled
    sc = np.r_[np.ones(n0), [s for _, s in sites]]
    W = W0[:, mw][:, :, mw]
    A = A0[np.ix_(ma, ma)] * np.outer(sc, sc)
    wf.getTabulatedFunction(0).setFunctionParameters(n, n, 3, W.ravel())
    yu.getTabulatedFunction(0).setFunctionParameters(n, n, A.ravel())
    for k, (i, s) in enumerate(sites):
        _, g = wf.getParticleParameters(i)
        wf.setParticleParameters(i, [n0 + k, g])
        yu.setParticleParameters(i, [n0 + k])
        print(f'  phosphate-like: residue {i + 1} ({"pTyr, Tyr Wang-Frenkel" if i in ptyr_sites else "Glu bead"}) '
              f'-> type {n0 + k}, Yukawa x{s:g}')
    return sites


if phospho_sites:
    apply_phospho(system, [(i, args.phospho_scale) for i in phospho_sites], seq)

# top.pdb in CALVADOS format (3-letter residue names, CA beads) for the analysis
three = dict(pd.read_csv(residues_file)[['one', 'three']].values)
if not resuming:
    top_cal = app.Topology()
    c = top_cal.addChain()
    for k_, aa in enumerate(seq):
        res = top_cal.addResidue('PTR' if k_ in ptyr_sites else three[aa], c)
        top_cal.addAtom('CA', app.Element.getBySymbol('C'), res)
    atoms = list(top_cal.atoms())
    for i in range(N_res - 1):
        top_cal.addBond(atoms[i], atoms[i + 1])
    top_cal.setPeriodicBoxVectors(box)
    with open(f'{path}/top.pdb', 'w') as fh:
        app.PDBFile.writeFile(top_cal, positions * unit.nanometer, fh)

integrator = mm.LangevinMiddleIntegrator(temp * unit.kelvin, 0.01 / unit.picosecond,
                                         timestep_ps * unit.picosecond)
if args.seed is not None:
    integrator.setRandomNumberSeed(args.seed)
platform = mm.Platform.getPlatformByName(args.platform)
props = {'Precision': 'mixed'} if args.platform == 'CUDA' else {}
simulation = app.Simulation(topology, system, integrator, platform, props)


def save_progress(progress):
    simulation.saveState(state_xml + '.tmp')
    os.replace(state_xml + '.tmp', state_xml)
    with open(progress_json + '.tmp', 'w') as fh:
        json.dump(progress, fh)
    os.replace(progress_json + '.tmp', progress_json)


if resuming:
    with open(progress_json) as fh:
        progress = json.load(fh)
    simulation.loadState(state_xml)
    print(f"Resuming {sysname}: {progress['segments_done']}/{n_segments} segments done", flush=True)
else:
    progress = {'segments_done': 0}
    simulation.context.setPositions(positions * unit.nanometer)
    simulation.minimizeEnergy()
    if args.seed is not None:
        simulation.context.setVelocitiesToTemperature(temp * unit.kelvin, args.seed)
    else:
        simulation.context.setVelocitiesToTemperature(temp * unit.kelvin)

log = open(f'{path}/{sysname}.log', 'a')
simulation.reporters.append(app.StateDataReporter(
    log, max(N_save, segment_steps // 10), step=True, speed=True, elapsedTime=True,
    potentialEnergy=True, temperature=True, separator='\t'))

print("STARTING SIMULATION", flush=True)
for s in range(progress['segments_done'], n_segments):
    dcd = app.DCDReporter(f'{path}/seg_{s:02d}.dcd', N_save)  # rerun segment overwrites
    simulation.reporters.append(dcd)
    simulation.step(segment_steps)
    simulation.reporters.remove(dcd)
    del dcd  # closes the file
    progress['segments_done'] = s + 1
    save_progress(progress)
    print(f"  segment {s + 1}/{n_segments} done", flush=True)
log.close()

# --- join segments into <name>.dcd, as CALVADOS writes it ------------------
segs = [f'{path}/seg_{s:02d}.dcd' for s in range(n_segments)]
if all(os.path.exists(f) for f in segs):
    traj = md.load(segs, top=f'{path}/top.pdb')
    assert traj.n_frames == target_frames, (traj.n_frames, target_frames)
    # write under a temporary name first: <name>.dcd marks the protein as done
    traj.save_dcd(f'{path}/{sysname}.tmp.dcd')
    os.replace(f'{path}/{sysname}.tmp.dcd', f'{path}/{sysname}.dcd')
    for f in segs:
        os.remove(f)

# --- analysis: CALVADOS' own function, same arguments as prepare.py --------
from calvados.analysis import save_conf_prop  # noqa: E402

save_conf_prop(path=path, name=sysname, residues_file=residues_file,
               output_path=data_path, start=equilibration_frames, is_idr=True, select='all')
print(open(f'{data_path}/conf_prop.csv').read())
