import os
from calvados.cfg import Config, Job, Components
import numpy as np
from argparse import ArgumentParser
from Bio import SeqIO  # Required to read the fasta sequence length

parser = ArgumentParser()
parser.add_argument('--name', nargs='?', required=True, type=str)
parser.add_argument(
    '--fasta',
    type=str,
    default=None,
    help=(
        "Explicit path to a FASTA file. Overrides --chunk-idx and the "
        "CALVADOS_FASTA / CALVADOS_CHUNK_IDX environment variables when set."
    ),
)
parser.add_argument(
    '--chunk-idx',
    type=str,
    default=None,
    help=(
        "Chunk index used to locate input/combined_chunk_<idx>.fasta. "
        "Falls back to the CALVADOS_CHUNK_IDX environment variable, then "
        "to the legacy 'idr.fasta' if neither is set."
    ),
)
parser.add_argument(
    '--outdir',
    type=str,
    default=os.path.expanduser('~/bender_runs/calvados2'),
    help="Root for run and data directories (<outdir>/<T>K/<name>, <outdir>/<T>K/data/<name>).",
)
parser.add_argument(
    '--temp',
    type=float,
    default=300.0,
    help="Simulation temperature in K (default: 300). Used in Config and output path.",
)
parser.add_argument(
    '--ph',
    type=float,
    default=7.5,
    help="pH (default: 7.5, as in all earlier runs). Only the histidine charge depends on it in CALVADOS-2.",
)
parser.add_argument(
    '--ionic',
    type=float,
    default=0.15,
    help="Ionic strength in M (default: 0.15, as in all BENDER runs).",
)
parser.add_argument(
    '--charge-termini',
    choices=['both', 'N', 'C', 'none'],
    default='both',
    help=(
        "Terminal charges (default: both, as in all earlier runs). 'none' matches MPIPI-Recharged, which has no "
        "terminal charges; used for the termini control runs. Added 2026-10-20."
    ),
)
parser.add_argument(
    '--residues',
    type=str,
    default=None,
    help=(
        "Residues CSV (default: input/residues_CALVADOS2.csv, as in all earlier runs). A relative path is taken "
        "relative to this folder. Letters defined in this file are kept in the sequence instead of being mapped to "
        "a standard residue, e.g. B = pSer and J = pThr in input_phospho/residues_phospho.csv. Added 2026-10-12."
    ),
)
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Residues file (default unchanged) ---
if args.residues is None:
    residues_file = f'{script_dir}/input/residues_CALVADOS2.csv'
elif os.path.isabs(os.path.expanduser(args.residues)):
    residues_file = os.path.expanduser(args.residues)
else:
    residues_file = os.path.join(script_dir, args.residues)
if not os.path.isfile(residues_file):
    raise FileNotFoundError(f"Residues file not found: {residues_file}")
with open(residues_file) as _fh:
    _header = _fh.readline().strip().split(',')
    _one = _header.index('one')
    _DEFINED_AA = {line.split(',')[_one].strip() for line in _fh if line.strip()}
print(f"Using residues: {residues_file}")

# --- Resolve FASTA path ---
# Priority: --fasta flag  >  CALVADOS_FASTA env var  >  --chunk-idx flag  >
#           CALVADOS_CHUNK_IDX env var  >  legacy idr.fasta
explicit_fasta = args.fasta or os.environ.get("CALVADOS_FASTA")
chunk_idx = args.chunk_idx or os.environ.get("CALVADOS_CHUNK_IDX")
if explicit_fasta is not None:
    fasta_path = os.path.abspath(os.path.expanduser(explicit_fasta))
elif chunk_idx is not None:
    fasta_path = f'{script_dir}/input/combined_chunk_{chunk_idx}.fasta'
else:
    fasta_path = f'{script_dir}/input/idr.fasta'

if not os.path.isfile(fasta_path):
    raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

print(f"Using FASTA: {fasta_path}")

# Normalize the input name: handle both UniProt format (sp|ACC|..., tr|ACC|...)
# and plain format (ACC | description | ...).
# e.g. "sp|Q9NPI6|DPYL2_HUMAN" -> "Q9NPI6"
#      "A0A8S5UG11 | Myoviridae sp. ctshb19 | ..." -> "A0A8S5UG11"
parts = args.name.split('|')
first = parts[0].strip()
if first in ('sp', 'tr') and len(parts) >= 2:
    input_name = parts[1].strip()
else:
    input_name = first.split()[0]  # ">ACC Rhizaria | ..." -> "ACC"
sysname = input_name

# --- Logic Implementation: Calculate Simulation Time based on Sequence ---
# 1. Extract the sequence for the specific protein name to get its length.
#    Biopython sets record.id to the token before the first whitespace, but
#    FASTA headers that use '|' as a delimiter may embed the accession before
#    the first '|' in record.description.  We therefore normalize every
#    record id the same way we normalized the input name.
# Map non-standard amino acids to nearest standard equivalent for CALVADOS
_NONSTANDARD_AA = {
    'O': 'K',  # pyrrolysine → lysine
    'U': 'C',  # selenocysteine → cysteine
    'B': 'D',  # Asp/Asn ambiguity → aspartate
    'Z': 'E',  # Glu/Gln ambiguity → glutamate
    'J': 'L',  # Leu/Ile ambiguity → leucine
    'X': 'G',  # unknown → glycine
}
_VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

def clean_sequence(seq):
    seq = seq.upper().replace('*', '')
    cleaned = []
    replacements = {}
    for aa in seq:
        if aa in _VALID_AA or aa in _DEFINED_AA:     # letters defined in the residues file are kept (e.g. B, J)
            cleaned.append(aa)
        elif aa in _NONSTANDARD_AA:
            sub = _NONSTANDARD_AA[aa]
            replacements[aa] = sub
            cleaned.append(sub)
        # silently drop anything else (gaps, etc.)
    if replacements:
        print(f"  WARNING: non-standard residues replaced: {replacements}")
    return ''.join(cleaned)

N_res = 0
matched_seq = None
for record in SeqIO.parse(fasta_path, "fasta"):
    rec_parts = record.id.split('|')
    rec_first = rec_parts[0].strip()
    if rec_first in ('sp', 'tr') and len(rec_parts) >= 2:
        record_id = rec_parts[1].strip()
    else:
        record_id = rec_first
    if record_id == input_name:
        matched_seq = clean_sequence(str(record.seq))
        N_res = len(matched_seq)
        break

if N_res == 0:
    raise ValueError(f"Protein '{input_name}' not found in {fasta_path}")

# 2. ADAPTIVE SIMULATION TIME: Scale based on sequence length
# NOTE (2026-09-28): the numbers below are a step *budget* written as if CALVADOS used a 20 fs step.
# CALVADOS 2 actually integrates at 10 fs (every BENDER DCD header), so each run is half the nominal time:
# 100 ns for ≤150 residues, 100·(N/150)^1.5 ns above. The arithmetic is kept unchanged so new runs have exactly
# the same steps and frames as BENDER; the times printed below are the real ones.

if N_res <= 150:
    target_time_ns = 200  # 200 ns for small sequences
else:
    # N^1.5 scaling: more time for larger sequences, but not quadratic
    scaling_factor = (N_res / 150) ** 1.5
    target_time_ns = int(200 * scaling_factor)

# Strategy: Aim for reasonable frame count that scales with simulation time
# Base: 10,000 frames for 200 ns (20 ps per frame)
# Scale frames for longer simulations to maintain temporal resolution
if target_time_ns <= 300:
    target_frames = 10000  # Standard resolution
elif target_time_ns <= 600:
    target_frames = 15000  # Higher resolution for longer sims
else:
    target_frames = 20000  # Even higher resolution

# Calculate time per frame in ps
time_per_frame_ps = (target_time_ns * 1000) / target_frames

# Step budget per frame, computed with the nominal 20 fs used for all BENDER runs (do not change: keeps runs identical)
nominal_timestep_ps = 0.02
steps_per_frame = int(time_per_frame_ps / nominal_timestep_ps)

# Real CALVADOS 2 time step (default; confirmed in the DCD headers)
timestep_ps = 0.01

# Total simulation steps
total_steps = target_frames * steps_per_frame

# Set saving interval
N_save = steps_per_frame

# Real simulated time (what the trajectory actually covers)
actual_time_ns = total_steps * timestep_ps / 1000
actual_ps_per_frame = N_save * timestep_ps

# EQUILIBRATION: Discard first 50% of simulation for equilibration
equilibration_fraction = 0.5
equilibration_time_ns = actual_time_ns * equilibration_fraction
equilibration_frames = int(target_frames * equilibration_fraction)

print(f"Configuring {sysname}:")
print(f"  Residues: {N_res}")
print(f"  Scaling: {'Baseline (≤150 res)' if N_res <= 150 else f'N^1.5 scaling (>{N_res}/150)^1.5'}")
print(f"  Simulated time: {actual_time_ns:.1f} ns (10 fs step)")
print(f"  Total frames: {target_frames}")
print(f"  Time per frame: {actual_ps_per_frame:.1f} ps")
print(f"  Save interval (steps): {N_save}")
print(f"  Total steps: {total_steps:,}")
print(f"  Expected simulation time: {(total_steps * timestep_ps / 1000):.1f} ns")
print(f"  Equilibration: First {equilibration_time_ns:.1f} ns ({equilibration_frames} frames) will be discarded")
print(f"  Production analysis: {actual_time_ns - equilibration_time_ns:.1f} ns ({target_frames - equilibration_frames} frames)")
# ------------------------------------------------------------------------

# set the side length of the cubic box
# Dynamic box sizing based on sequence length
if N_res > 300:
    L = int(N_res / 5) + 20  # Heuristic to scale box with length
else:
    L = 50
# residues_file is set near the top (--residues; default input/residues_CALVADOS2.csv)
_custom = [f'{c}{i + 1}' for i, c in enumerate(matched_seq) if c not in _VALID_AA]
if _custom:
    print(f"  Custom residues kept (defined in the residues file): {_custom}")

temp = args.temp
print(f"  Temperature: {temp} K")
print(f"  pH: {args.ph}")
print(f"  Ionic strength: {args.ionic} M")
print(f"  Charged termini: {args.charge_termini}")

config = Config(
    # GENERAL
    sysname = sysname,
    box = [L, L, L],
    temp = temp,
    ionic = args.ionic,
    pH = args.ph,
    topol = 'center',

    # RUNTIME SETTINGS - Adaptive based on sequence length
    wfreq = N_save,           # Save interval (steps)
    steps = total_steps,      # Total steps (scaled)
    runtime = 0, 
    platform = 'CUDA',
    restart = 'checkpoint',
    frestart = 'restart.chk',
    verbose = True,
)

# PATH - Create protein-specific directories using os.makedirs to safely
# handle paths with spaces or other special characters (avoids shell-splitting
# issues that arise when using subprocess.run with shell=True).
outdir = args.outdir.rstrip('/')
temp_tag = f'{int(temp)}K'
path = f'{outdir}/{temp_tag}/{sysname:s}'
data_path = f'{outdir}/{temp_tag}/data/{sysname:s}'
os.makedirs(path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

# Write a cleaned single-record FASTA whose header is exactly `sysname`.
# CALVADOS keys records by Biopython's record.id (the token before the first
# whitespace) and looks the component up by its exact name.  Headers in the
# combined FASTA look like ">ACCESSION|..." (no space before the '|'), so
# record.id would be "ACCESSION|..." and the lookup by the stripped accession
# would raise a KeyError.  Pointing ffasta at this clean, one-record file makes
# the id match the component name exactly.
component_fasta = f'{path}/{sysname:s}.fasta'
with open(component_fasta, 'w') as fh:
    fh.write(f'>{sysname:s}\n{matched_seq:s}\n')

# Analysis with automatic equilibration discard (50% of simulation)
# start = frame number to begin analysis (discarding equilibration)
analyses = f"""
from calvados.analysis import save_conf_prop
save_conf_prop(path="{path:s}",name="{sysname:s}",residues_file="{residues_file:s}",output_path="{data_path:s}",start={equilibration_frames},is_idr=True,select='all')
"""

config.write(path, name='config.yaml', analyses=analyses)

components = Components(
    molecule_type = 'protein',
    nmol = 1, 
    restraint = False, 
    charge_termini = args.charge_termini, 
    fresidues = residues_file,
    ffasta = component_fasta,
)

components.add(name=input_name)
components.write(path, name='components.yaml')

print(f"\nSetup complete!")
print(f"  FASTA used: {fasta_path}")
print(f"  Component FASTA (cleaned): {component_fasta}")
print(f"  Simulation directory: {path}")
print(f"  Data output directory: {data_path}")
print(f"  Analysis will use frames {equilibration_frames}-{target_frames} ({equilibration_time_ns:.1f}-{actual_time_ns:.1f} ns)")