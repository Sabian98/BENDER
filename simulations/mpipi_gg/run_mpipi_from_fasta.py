"""
MPIPI-GG simulation setup from FASTA
Mirrors the CALVADOS IDRome adaptive protocol:
  - <=150 res: 200 ns
  - >150 res: scales as (N/150)^p, p = --time-exponent (2.0 for the BENDER-BIO runs; 1.5 matches CALVADOS)
  - 50% equilibration discarded
  - 10,000 production frames
  - 150 mM implicit salt, 300K, Langevin

Usage:
    python3 run_mpipi_from_fasta.py --fasta my.fasta --name PROTEIN1
    python3 run_mpipi_from_fasta.py --fasta my.fasta --name PROTEIN1 --outdir ~/runs/PROTEIN1

Dependencies:
    pip install biopython
"""

import os, math, random
from argparse import ArgumentParser
from Bio import SeqIO

# ── Argument parsing ───────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument('--fasta',  required=True,  type=str, help='Input FASTA file')
parser.add_argument('--name',   required=True,  type=str, help='Sequence ID in FASTA')
parser.add_argument('--outdir', default=None,   type=str, help='Output directory (default: runs/<name>)')
parser.add_argument('--time-exponent', default=2.0, type=float,
                    help='Exponent p in 200*(N/150)^p ns for N > 150 (default 2.0; CALVADOS uses 1.5)')
args = parser.parse_args()

# ── Read sequence ──────────────────────────────────────────────────────────
input_name = args.name.split('|')[0].strip()
sequence = None
for record in SeqIO.parse(args.fasta, "fasta"):
    if record.id.split('|')[0].strip() == input_name:
        sequence = str(record.seq).upper()
        break

if sequence is None:
    raise ValueError(f"'{input_name}' not found in {args.fasta}")

N_res = len(sequence)
print(f"\nProtein:   {input_name}")
print(f"Sequence:  {sequence}")
print(f"Residues:  {N_res}")

# ── Adaptive simulation time (mirrors CALVADOS IDRome) ─────────────────────
if N_res <= 150:
    target_time_ns = 200
else:
    target_time_ns = int(200 * (N_res / 150) ** args.time_exponent)

if target_time_ns <= 300:
    target_frames = 10000
elif target_time_ns <= 600:
    target_frames = 15000
else:
    target_frames = 20000

# MPIPI-GG timestep = 20 fs
timestep_ns   = 20e-6
time_per_frame_ns = target_time_ns / target_frames
steps_per_frame   = int(time_per_frame_ns / timestep_ns)
total_steps       = target_frames * steps_per_frame
equil_frames      = target_frames // 2
equil_steps       = equil_frames * steps_per_frame

print(f"\nSimulation parameters:")
print(f"  Target time:     {target_time_ns} ns (time exponent {args.time_exponent})")
print(f"  Total frames:    {target_frames}")
print(f"  Steps/frame:     {steps_per_frame:,}")
print(f"  Total steps:     {total_steps:,}")
print(f"  Equilibration:   {target_time_ns//2} ns ({equil_steps:,} steps) — discarded")
print(f"  Production:      {target_time_ns//2} ns ({total_steps-equil_steps:,} steps)")
print(f"  Est. walltime:   ~{total_steps/1900/3600:.1f} hours at ~1900 steps/sec (CPU)")

# ── Box size ───────────────────────────────────────────────────────────────
L_A = int(N_res / 5 + 20) * 10 if N_res > 300 else 500
print(f"  Box size:        {L_A} Å ({L_A/10:.0f} nm)")

# ── MPIPI-GG type mapping ──────────────────────────────────────────────────
AA_TO_GLOBAL = {
    'M':1,  'G':2,  'K':3,  'T':4,  'R':5,
    'A':6,  'D':7,  'E':8,  'Y':9,  'V':10,
    'L':11, 'Q':12, 'W':13, 'F':14, 'S':15,
    'H':16, 'N':17, 'P':18, 'C':19, 'I':20
}
AA_MASS = {
    'M':131.197, 'G':57.052,  'K':128.174, 'T':101.105, 'R':156.188,
    'A':71.078,  'D':115.089, 'E':129.116, 'Y':163.176, 'V':99.132,
    'L':113.159, 'Q':128.131, 'W':186.213, 'F':147.177, 'S':87.078,
    'H':137.141, 'N':114.104, 'P':97.116,  'C':103.144, 'I':113.159
}
CHARGED_POS = {3, 5, 16}   # K, R, H (+0.375 in Mpipi-GGv23; the SAXS benchmark runs left His uncharged)
CHARGED_NEG = {7, 8}        # D, E

# Check all residues are recognized
for aa in sequence:
    if aa not in AA_TO_GLOBAL:
        raise ValueError(f"Unrecognized residue: '{aa}'. Only standard 20 AAs supported.")

types_used      = sorted(set(AA_TO_GLOBAL[aa] for aa in sequence))
global_to_local = {t: i+1 for i, t in enumerate(types_used)}
local_to_global = {i+1: t for i, t in enumerate(types_used)}
type_to_aa      = {v: k for k, v in AA_TO_GLOBAL.items()}

local_charged_pos = {global_to_local[g] for g in CHARGED_POS if g in global_to_local}
local_charged_neg = {global_to_local[g] for g in CHARGED_NEG if g in global_to_local}

# ── Self-avoiding random walk starting config ──────────────────────────────
random.seed(42)
bond_len = 3.81
min_dist = 3.0
margin   = 50.0
center   = L_A / 2.0

def dist3(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

positions = [(center, center, center)]
for i in range(1, N_res):
    placed = False
    for _ in range(10000):
        theta = math.acos(2*random.random() - 1)
        phi   = 2 * math.pi * random.random()
        nx = positions[-1][0] + bond_len * math.sin(theta) * math.cos(phi)
        ny = positions[-1][1] + bond_len * math.sin(theta) * math.sin(phi)
        nz = positions[-1][2] + bond_len * math.cos(theta)
        if not (margin < nx < L_A-margin and
                margin < ny < L_A-margin and
                margin < nz < L_A-margin):
            continue
        if all(dist3((nx,ny,nz), p) >= min_dist for p in positions):
            positions.append((nx, ny, nz))
            placed = True
            break
    if not placed:
        # Fallback: extend along z
        px, py, pz = positions[-1]
        positions.append((px, py, min(pz + bond_len, L_A - margin)))
        print(f"  Warning: fallback placement for bead {i}")

# ── Output directory ───────────────────────────────────────────────────────
outdir = args.outdir or os.path.join(
    'runs', input_name)
os.makedirs(outdir, exist_ok=True)
print(f"\nOutput directory: {outdir}")

# ── Write seq.dat ──────────────────────────────────────────────────────────
lines = [
    f"Mpipi-GG: {input_name} | {sequence}", "",
    f"{N_res} atoms", f"{N_res-1} bonds", "",
    f"{len(types_used)} atom types", "1 bond types", "",
    f"0.0 {L_A} xlo xhi",
    f"0.0 {L_A} ylo yhi",
    f"0.0 {L_A} zlo zhi", "",
    "Masses", ""
]
for t in types_used:
    aa = type_to_aa[t]
    lines.append(f"{global_to_local[t]} {AA_MASS[aa]}  # {aa}")
lines += ["", "Atoms", ""]
for i, (aa, (x,y,z)) in enumerate(zip(sequence, positions)):
    lines.append(
        f"{i+1} 1 {global_to_local[AA_TO_GLOBAL[aa]]} 0.0 {x:.4f} {y:.4f} {z:.4f}")
lines += ["", "Bonds", ""]
for i in range(N_res-1):
    lines.append(f"{i+1} 1 {i+1} {i+2}")

seq_dat = os.path.join(outdir, "seq.dat")
with open(seq_dat, "w") as f:
    f.write("\n".join(lines))
print(f"Written: {seq_dat}")

# ── Read MPIPI-GG force field template ────────────────────────────────────
ff_paths = [
    os.path.join(os.path.dirname(__file__), "mpipi_gg.in"),
    os.environ.get("MPIPI_FF", ""),
]
ff_template = None
for p in ff_paths:
    if p and os.path.exists(p):
        ff_template = p
        break

if ff_template is None:
    raise FileNotFoundError(
        "mpipi_gg.in not found. Set MPIPI_FF env var or place it in "
        "~/mpipi_sim/protein/mpipi_gg.in")

with open(ff_template) as f:
    template = f.read()

# Extract pair_coeff lines for types used in this sequence
wf_coeffs = {}
for line in template.split('\n'):
    if not line.startswith('pair_coeff') or 'wf/cut' not in line:
        continue
    parts = line.split()
    try:
        gi, gj = int(parts[1]), int(parts[2])
        if gi in global_to_local and gj in global_to_local:
            li = global_to_local[gi]
            lj = global_to_local[gj]
            key = (min(li,lj), max(li,lj))
            params = ' '.join(parts[4:]).split('#')[0].strip()
            wf_coeffs[key] = params
    except (ValueError, IndexError):
        pass

# ── Write run.in ──────────────────────────────────────────────────────────
seed = random.randint(10000, 99999)
run_lines = [
    f"# Mpipi-GG: {input_name} | {N_res} res | {target_time_ns} ns",
    "# Force field: Mpipi-GGv23-r1 (Lotthammer et al. Nat Methods 2024)",
    "units real",
    "dimension 3",
    "boundary p p p",
    "atom_style full",
    "bond_style harmonic",
    "dielectric 80.0",
    "",
    "read_data seq.dat",
    "",
    "# Partial charges (Mpipi convention)",
]
for lt in sorted(local_charged_pos):
    gt = local_to_global[lt]
    q  = 0.375 if gt == 16 else 0.75
    run_lines.append(f"set type {lt} charge {q}  # {type_to_aa[gt]}")
for lt in sorted(local_charged_neg):
    gt = local_to_global[lt]
    run_lines.append(f"set type {lt} charge -0.75  # {type_to_aa[gt]}")

run_lines += [
    "",
    "# Bond parameters: harmonic, k=9.6 kcal/mol/A^2, r0=3.81 A",
    "bond_coeff 1 9.600 3.81",
    "",
]
charged_locals = local_charged_pos | local_charged_neg
if charged_locals:
    run_lines += [
        "# Pair style: WF potential + Coulomb/Debye (150mM salt, kappa=0.126)",
        "pair_style hybrid/overlay wf/cut 25.0 coul/debye 0.126 0.0",
        "",
        "# WF pair interactions (MPIPI-GG parameters)",
    ]
    for (li, lj), params in sorted(wf_coeffs.items()):
        run_lines.append(f"pair_coeff {li} {lj} wf/cut {params}")
    run_lines.append("")
    run_lines.append("# Coulombic interactions between charged residues")
    for li in sorted(charged_locals):
        for lj in sorted(charged_locals):
            if li <= lj:
                run_lines.append(f"pair_coeff {li} {lj} coul/debye 35.")
else:
    run_lines += [
        "# Pair style: WF potential only (no charged residues)",
        "pair_style wf/cut 25.0",
        "",
        "# WF pair interactions (MPIPI-GG parameters)",
    ]
    for (li, lj), params in sorted(wf_coeffs.items()):
        run_lines.append(f"pair_coeff {li} {lj} {params}")

run_lines += [
    "",
    "# FENE special bonds (excludes 1-2 bonded pairs from non-bonded)",
    "special_bonds fene",
    "",
    "# Communication cutoff > pair cutoff (prevents bond atoms missing error)",
    "comm_modify cutoff 40.0",
    "comm_style brick",
    "",
    "# Neighbor list",
    "neighbor 3.5 bin",
    "neigh_modify every 1 delay 0 check yes",
    "",
    "# Energy minimization (remove bad contacts from starting config)",
    "minimize 0.0 0.00000001 1000 100000",
    "",
    "# MD settings",
    "timestep 20",
    "timer normal",
    "",
    "# NVE + Langevin thermostat = NVT at 300K",
    "fix fxnve    all nve",
    f"fix fxlange  all langevin 300 300 5000.0 {seed}",
    "",
    f"# Save trajectory every {steps_per_frame} steps ({time_per_frame_ns*1000:.0f} ps)",
    f"# Start saving after equilibration ({equil_steps} steps = {target_time_ns//2} ns)",
    f"dump 1 all xtc {steps_per_frame} traj.xtc",
    "dump_modify 1 sort id",
    f"dump_modify 1 delay {equil_steps}",
    "",
    "# Radius of gyration output",
    "compute rad all gyration",
    f"fix fxrg all ave/time {steps_per_frame} 1 {steps_per_frame} "
    f"c_rad start {equil_steps} file Rg.out",
    "",
    f"thermo {steps_per_frame}",
    "thermo_style custom step pe press ke temp lx ly lz spcpu",
    "thermo_modify flush yes",
    "",
    f"# Initialize velocities at 300K",
    f"velocity all create 300.0 {seed} rot yes dist gaussian",
    "",
    f"restart {total_steps//2} restart",
    f"run {total_steps}",
]

run_in = os.path.join(outdir, "run.in")
with open(run_in, "w") as f:
    f.write("\n".join(run_lines))
print(f"Written: {run_in}")

# ── Write SLURM submission script ──────────────────────────────────────────
# NOTE: No GPU requested. The wf/cut pair style (EXTRA-PAIR package) has no
# KOKKOS/CUDA kernel, so MPIPI-GG runs almost entirely on CPU regardless of
# GPU allocation. Requesting a GPU here would only risk a hard crash on
# GAIVI's older cards (TitanX/1080Ti are compute capability < 7.0, while our
# lmp binary was built for compute capability 8.0 - Kokkos refuses to run
# and aborts with a core dump). CPU-only avoids this entirely and frees up
# GPU slots on the cluster for jobs that actually need them (e.g. CALVADOS).
walltime_hrs = int(total_steps / 1900 / 3600 * 1.3) + 1
if walltime_hrs <= 24:
    partition = "Quick"
else:
    partition = "Extended"

slurm = f"""#!/bin/bash -l
#SBATCH --job-name=mpipi_{input_name[:8]}
#SBATCH -p {partition}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o mpipi_%j.log
#SBATCH -e mpipi_%j.err

# ── Environment (no CUDA/GPU needed - CPU-only run) ────────────────────────
source /apps/gaivi-environment.sh
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate gmx-cuda 2>/dev/null || true

export PATH=$HOME/lammps-gpu/bin:$PATH

# ── Run ───────────────────────────────────────────────────────────────────
cd {outdir}

echo "======================================================"
echo "  Mpipi-GG simulation (CPU)"
echo "  Protein:   {input_name}"
echo "  Sequence:  {sequence}"
echo "  Residues:  {N_res}"
echo "  Target:    {target_time_ns} ns ({total_steps:,} steps)"
echo "  Node:      $(hostname)"
echo "  Started:   $(date)"
echo "======================================================"

srun lmp -in run.in
EXIT=$?

[ $EXIT -eq 0 ] && rm -f restart.* && echo "Cleaned restart files"

echo "======================================================"
echo "  Done: $(date)"
echo "  Exit: $EXIT"
echo "======================================================"
exit $EXIT
"""

slurm_script = os.path.join(outdir, "run.sh")
with open(slurm_script, "w") as f:
    f.write(slurm)
os.chmod(slurm_script, 0o755)
print(f"Written: {slurm_script}")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════╗
║  Setup complete!                                         ║
║  Protein:    {input_name:<44} ║
║  Residues:   {N_res:<3}  |  Target: {target_time_ns:<6} ns                ║
║  Steps:      {total_steps:<12,}                            ║
║  Partition:  {partition:<8}  |  No time limit                   ║
╚══════════════════════════════════════════════════════════╝

To submit:
  cd {outdir}
  sbatch run.sh

To submit from anywhere:
  sbatch {outdir}/run.sh
""")