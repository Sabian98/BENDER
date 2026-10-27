#!/usr/bin/env bash
# =============================================================================
# INSTALLER: Run this once to (re)write all pipeline scripts to
# ~/mpipi_sim/scripts/. Safe to re-run any time - just overwrites the helper
# scripts, never touches ~/mpipi_sim/runs/ or any simulation output.
#
# Usage: bash install_scripts.sh
#
# NOTE: This installer writes batch_submit.sh, monitor.sh, analyze.py, and
# check_env.sh. run_mpipi_from_fasta.py is NOT included here (it's long) -
# copy/scp it separately into ~/mpipi_sim/scripts/.
#
# NOTE ON CPU vs GPU: all scripts here are CPU-only. The wf/cut pair style
# MPIPI-GG depends on has no KOKKOS/CUDA kernel, and GAIVI's GPU fleet mixes
# old cards (TitanX/1080Ti, compute capability < 7.0) with new ones
# (A40/A100, 8.0+) - our lmp binary targets 8.0 and hard-crashes on the old
# cards. Requesting no GPU sidesteps this entirely. See README.md for details.
# =============================================================================

SCRIPTS="$HOME/mpipi_sim/scripts"
mkdir -p "$SCRIPTS"

echo "Installing MPIPI-GG pipeline scripts to $SCRIPTS ..."

# ── batch_submit.sh (SLURM array job, CPU-only, chunk auto-resubmit) ──────
cat > "$SCRIPTS/batch_submit.sh" << 'EOF'
#!/bin/bash
# =============================================================================
# batch_submit.sh
#
# Submit MPIPI-GG simulations as a SLURM array job (one task per protein),
# CPU-only. Optionally schedules a self-resubmission job after completion,
# mirroring the CALVADOS chunk-based pipeline.
#
# Usage:
#   bash batch_submit.sh <fasta_file> [chunk_idx]
# =============================================================================

FASTA_FILE="$1"
CHUNK_IDX="${2:-0}"

[ -z "$FASTA_FILE" ] && echo "Usage: $0 <fasta_file> [chunk_idx]" && exit 1
[ ! -f "$FASTA_FILE" ] && echo "ERROR: $FASTA_FILE not found" && exit 1

FASTA_FILE="$(realpath "$FASTA_FILE")"
RUNS_DIR="$HOME/mpipi_sim/runs"
SETUP_SCRIPT="$HOME/mpipi_sim/scripts/run_mpipi_from_fasta.py"
SCRIPT_PATH="$(realpath "$0")"
WORK_DIR="$(pwd)"
NUM_PROTEINS=$(grep -c ">" "$FASTA_FILE")

echo "=============================================="
echo "  MPIPI-GG batch submission (CPU-only)"
echo "  FASTA:    $FASTA_FILE"
echo "  Proteins: $NUM_PROTEINS"
echo "  Chunk:    $CHUNK_IDX"
echo "=============================================="

echo "Pre-generating simulation files..."

mapfile -t NAMES < <(grep "^>" "$FASTA_FILE" \
    | sed 's/>//' | awk '{print $1}' | cut -d'|' -f1 | tr -d ' ')

for name in "${NAMES[@]}"; do
    outdir="$RUNS_DIR/$name"
    if [[ -f "$outdir/run.in" ]] && [[ -f "$outdir/seq.dat" ]]; then
        echo "  skip: $name"
    else
        python3 "$SETUP_SCRIPT" \
            --fasta "$FASTA_FILE" \
            --name  "$name" \
            --outdir "$outdir" 2>&1 | grep -E "Written|Error" || true
    fi
done

echo "Setup complete for $NUM_PROTEINS proteins."

TMPSCRIPT=$(mktemp /tmp/mpipi_array_XXXXXX.sh)

cat > "$TMPSCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=mpipi_chunk${CHUNK_IDX}
#SBATCH -p Quick
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --array=1-${NUM_PROTEINS}
#SBATCH -o ${RUNS_DIR}/chunk${CHUNK_IDX}_%a_%j.out
#SBATCH -e ${RUNS_DIR}/chunk${CHUNK_IDX}_%a_%j.err

source /apps/gaivi-environment.sh
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate gmx-cuda 2>/dev/null || true

export PATH=\$HOME/lammps-gpu/bin:\$PATH

NAME=\$(grep "^>" "${FASTA_FILE}" \
    | sed 's/>//' | awk '{print \$1}' | cut -d'|' -f1 | tr -d ' ' \
    | sed -n "\${SLURM_ARRAY_TASK_ID}p")

OUTDIR="${RUNS_DIR}/\${NAME}"

echo "Task \${SLURM_ARRAY_TASK_ID}/${NUM_PROTEINS} | \${NAME} | \$(hostname) | \$(date)"

cd "\$OUTDIR"
srun lmp -in run.in
EXIT=\$?

[ \$EXIT -eq 0 ] && rm -f restart.* && echo "Cleaned restart files"
echo "Done: \$(date) | Exit: \$EXIT"
exit \$EXIT
EOF

echo ""
echo "Submitting SLURM array job (1-${NUM_PROTEINS}, CPU-only)..."
ARRAY_JOB_ID=$(sbatch --parsable "$TMPSCRIPT")
rm -f "$TMPSCRIPT"

echo "Array job submitted: ${ARRAY_JOB_ID}"
echo "  Tasks: 1-${NUM_PROTEINS}"
echo ""

if [[ "$CHUNK_IDX" != "0" ]]; then
    NEXT_CHUNK=$((CHUNK_IDX + 1))
    NEXT_FASTA="$(dirname "$FASTA_FILE")/combined_chunk_${NEXT_CHUNK}.fasta"
    RESUBMIT_TIME=$(date -d '+30 hours' '+%Y-%m-%dT%H:%M:%S')

    if [[ -f "$NEXT_FASTA" ]]; then
        RESUBMIT_JOB_ID=$(sbatch --parsable \
            --job-name=resubmit_chunk${CHUNK_IDX} \
            --dependency=afterany:${ARRAY_JOB_ID} \
            --begin=${RESUBMIT_TIME} \
            --ntasks=1 --mem=1GB \
            --output=${RUNS_DIR}/resubmit_chunk${CHUNK_IDX}_%j.log \
            --wrap="cd ${WORK_DIR} && bash ${SCRIPT_PATH} ${NEXT_FASTA} ${NEXT_CHUNK}")
        echo "Resubmitter: ${RESUBMIT_JOB_ID}"
        echo "  -> next chunk: $NEXT_FASTA"
        echo "  -> starts: $RESUBMIT_TIME"
    else
        echo "  (no next chunk found at ${NEXT_FASTA} - no resubmission scheduled)"
    fi
fi

echo ""
echo "Monitor: bash ~/mpipi_sim/scripts/monitor.sh"
EOF
chmod +x "$SCRIPTS/batch_submit.sh"
echo "✅ batch_submit.sh (CPU-only array job)"

# ── monitor.sh ────────────────────────────────────────────────────────────
cat > "$SCRIPTS/monitor.sh" << 'EOF'
#!/bin/bash
# Monitor MPIPI-GG simulation progress
# Usage: bash monitor.sh

echo "=== SLURM Jobs ==="
squeue -u $USER -o "%.10i %.15j %.8T %.10M %.6D %R"

echo ""
echo "=== Simulation Progress ==="
printf "%-20s %12s %10s %10s %10s\n" "Name" "Step" "Progress" "Rg_frames" "Last_Rg(A)"
echo "$(printf '%0.s-' {1..70})"

for rundir in ~/mpipi_sim/runs/*/; do
    [ ! -d "$rundir" ] && continue
    name=$(basename "$rundir")
    logfile=$(ls "$rundir"/*.out "$rundir"/mpipi_*.log 2>/dev/null | tail -1)
    rg_file="$rundir/Rg.out"

    if [ ! -f "$logfile" ]; then
        printf "%-20s %s\n" "$name" "not started"
        continue
    fi

    last_step=$(grep "^[ ]*[0-9]" "$logfile" 2>/dev/null | tail -1 | awk '{print $1}')
    total_steps=$(grep "^run " "$rundir/run.in" 2>/dev/null | awk '{print $2}')

    if [ -n "$last_step" ] && [ -n "$total_steps" ] && [ "$total_steps" -gt 0 ]; then
        pct=$(awk "BEGIN {printf \"%.1f%%\", $last_step/$total_steps*100}")
    else
        pct="?"
    fi

    if [ -f "$rg_file" ]; then
        n_frames=$(grep -v "^#" "$rg_file" | grep -c "[0-9]" 2>/dev/null || echo 0)
        last_rg=$(grep -v "^#" "$rg_file" | tail -1 | awk '{print $2}' 2>/dev/null || echo "-")
    else
        n_frames=0
        last_rg="-"
    fi

    printf "%-20s %12s %10s %10s %10s\n" \
        "$name" "${last_step:-?}" "$pct" "$n_frames" "$last_rg"
done
EOF
chmod +x "$SCRIPTS/monitor.sh"
echo "✅ monitor.sh"

# ── analyze.py ────────────────────────────────────────────────────────────
cat > "$SCRIPTS/analyze.py" << 'EOF'
"""
Analyze MPIPI-GG simulation results
Computes Rg statistics and checks convergence

Usage:
    python3 analyze.py --rundir ~/mpipi_sim/runs/PROTEIN1
    python3 analyze.py --runsdir ~/mpipi_sim/runs
    python3 analyze.py --runsdir ~/mpipi_sim/runs --csv results.csv
"""
import os
import numpy as np
from argparse import ArgumentParser


def analyze_rg(rundir):
    name = os.path.basename(rundir.rstrip('/'))
    rg_file = os.path.join(rundir, "Rg.out")

    progress = "unknown"
    logfiles = [f for f in os.listdir(rundir)
                if (f.startswith("mpipi_") and f.endswith(".log"))
                or f.endswith(".out")]
    if logfiles:
        log = os.path.join(rundir, sorted(logfiles)[-1])
        try:
            with open(log) as f:
                for line in reversed(f.readlines()):
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        progress = f"step {int(parts[0]):,}"
                        break
        except Exception:
            pass

    if not os.path.exists(rg_file):
        return {'name': name, 'status': 'running', 'progress': progress,
                'n': 0, 'mean': None, 'std': None}

    rg = []
    with open(rg_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    rg.append(float(parts[1]))
                except ValueError:
                    pass

    if not rg:
        return {'name': name, 'status': 'no_data', 'progress': progress,
                'n': 0, 'mean': None, 'std': None}

    rg = np.array(rg)
    mid = len(rg) // 2
    first_half  = rg[:mid].mean() if mid > 0 else np.nan
    second_half = rg[mid:].mean() if mid > 0 else np.nan
    converged   = abs(first_half - second_half) < 2.0 if mid > 0 else False

    return {
        'name':       name,
        'status':     'done',
        'progress':   progress,
        'n':          len(rg),
        'mean':       float(rg.mean()),
        'std':        float(rg.std()),
        'min':        float(rg.min()),
        'max':        float(rg.max()),
        'first_half': float(first_half),
        'second_half':float(second_half),
        'converged':  converged,
    }


def print_result(r):
    if r['status'] in ('running', 'no_data'):
        print(f"\n  {r['name']}: {r['status']} ({r['progress']})")
        return
    conv = "YES" if r['converged'] else "NO"
    print(f"""
{'='*55}
  {r['name']}
{'='*55}
  Progress:    {r['progress']}
  Frames:      {r['n']}
  Rg mean:     {r['mean']:.2f} +/- {r['std']:.2f} A
  Rg range:    {r['min']:.2f} - {r['max']:.2f} A
  1st half:    {r['first_half']:.2f} A
  2nd half:    {r['second_half']:.2f} A
  Converged:   {conv}""")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--rundir',  type=str, default=None)
    parser.add_argument('--runsdir', type=str, default=None)
    parser.add_argument('--csv',     type=str, default=None,
                        help='Save summary to CSV file')
    args = parser.parse_args()

    results = []

    if args.rundir:
        r = analyze_rg(args.rundir)
        print_result(r)
        results.append(r)

    elif args.runsdir:
        entries = sorted(os.listdir(args.runsdir))
        for d in entries:
            rd = os.path.join(args.runsdir, d)
            if os.path.isdir(rd):
                r = analyze_rg(rd)
                print_result(r)
                results.append(r)

        done = [r for r in results if r['status'] == 'done']
        if done:
            means = [r['mean'] for r in done]
            n_conv = sum(1 for r in done if r['converged'])
            print(f"""
{'='*55}
  SUMMARY - {len(done)}/{len(results)} complete
{'='*55}
  Mean Rg (all):  {np.mean(means):.2f} A
  Rg range:       {min(means):.2f} - {max(means):.2f} A
  Converged:      {n_conv}/{len(done)}""")

    if args.csv and results:
        import csv
        fieldnames = ['name', 'status', 'n', 'mean', 'std',
                      'min', 'max', 'first_half', 'second_half', 'converged']
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
        print(f"\nSaved: {args.csv}")

    if not args.rundir and not args.runsdir:
        parser.print_help()
EOF
echo "✅ analyze.py"

# ── check_env.sh ──────────────────────────────────────────────────────────
cat > "$SCRIPTS/check_env.sh" << 'EOF'
#!/bin/bash
# Check that the MPIPI-GG environment is correctly set up.
# This pipeline is CPU-only (see README "IMPORTANT: This pipeline runs on
# CPU, not GPU"), so this script does NOT check for CUDA/GPU availability -
# only that lmp exists and the helper scripts are in place.

echo "=== MPIPI-GG Environment Check (CPU-only pipeline) ==="
echo ""

ok=0; fail=0

check() {
    local label="$1"; local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo "  [OK]   $label"
        ((ok++))
    else
        echo "  [FAIL] $label"
        ((fail++))
    fi
}

check "LAMMPS binary exists"        "[ -f '$HOME/lammps-gpu/bin/lmp' ]"
check "lmp in PATH"                 "which lmp"
check "gmx-cuda conda env exists"   "conda env list | grep -q gmx-cuda"
check "MPIPI-GG force field"        "[ -f '$HOME/mpipi_sim/protein/mpipi_gg.in' ]"
check "run_mpipi_from_fasta.py"     "[ -f '$HOME/mpipi_sim/scripts/run_mpipi_from_fasta.py' ]"
check "batch_submit.sh"             "[ -f '$HOME/mpipi_sim/scripts/batch_submit.sh' ]"
check "analyze.py"                  "[ -f '$HOME/mpipi_sim/scripts/analyze.py' ]"
check "monitor.sh"                  "[ -f '$HOME/mpipi_sim/scripts/monitor.sh' ]"

echo ""
echo "  Active conda env: ${CONDA_DEFAULT_ENV:-none}"
echo "  lmp path:         $(which lmp 2>/dev/null || echo 'not in PATH')"
echo ""
echo "  Result: $ok passed, $fail failed"

if [ $fail -eq 0 ]; then
    echo ""
    echo "  All checks passed. Ready to run simulations."
    echo ""
    echo "  Quick start:"
    echo "    bash ~/mpipi_sim/scripts/batch_submit.sh <your.fasta>"
else
    echo ""
    echo "  Some checks failed. Try: source ~/.bashrc"
fi
EOF
chmod +x "$SCRIPTS/check_env.sh"
echo "✅ check_env.sh"

echo ""
echo "=== All scripts installed to $SCRIPTS ==="
ls -la "$SCRIPTS/"
echo ""
echo "Run environment check:"
echo "  bash $SCRIPTS/check_env.sh"