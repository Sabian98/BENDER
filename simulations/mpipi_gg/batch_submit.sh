#!/bin/bash
# =============================================================================
# batch_submit.sh
#
# Submit MPIPI-GG simulations as a SLURM array job (one task per protein),
# CPU-only. Optionally schedules a self-resubmission job after completion,
# mirroring the CALVADOS chunk-based pipeline.
#
# NOTE ON CPU vs GPU:
#   The wf/cut pair style (EXTRA-PAIR package) has no KOKKOS/CUDA kernel, so
#   MPIPI-GG runs almost entirely on CPU regardless of GPU allocation.
#   GAIVI's GPU fleet is a mix of TitanX/1080Ti (compute capability < 7.0)
#   and A40/A100 (8.0+). Our lmp binary targets compute capability 8.0, so
#   requesting a GPU risks landing on an old card and hard-crashing
#   ("Kokkos::Cuda::initialize ERROR: likely mismatch of architecture").
#   CPU-only avoids this entirely and frees GPU slots for jobs that actually
#   benefit from them (e.g. CALVADOS/OpenMM).
#
# Usage:
#   bash batch_submit.sh <fasta_file> [chunk_idx]
#
# Examples:
#   bash batch_submit.sh input/my_proteins.fasta
#   bash batch_submit.sh simulations/mpipi_gg/input/compact_mpipi_300K.fasta
#   bash batch_submit.sh simulations/mpipi_gg/input/matched_pairs_340.fasta 1

#
# Output:
#   ~/bender_runs/mpipi_gg/<protein_name>/   <- one directory per protein
#       seq.dat, run.in, Rg.out, traj.xtc
#   ~/bender_runs/mpipi_gg/chunk<N>_<task>_<jobid>.out/.err
# =============================================================================

FASTA_FILE="$1"
CHUNK_IDX="${2:-0}"

[ -z "$FASTA_FILE" ] && echo "Usage: $0 <fasta_file> [chunk_idx]" && exit 1
[ ! -f "$FASTA_FILE" ] && echo "ERROR: $FASTA_FILE not found" && exit 1

FASTA_FILE="$(realpath "$FASTA_FILE")"
RUNS_DIR="${RUNS_DIR:-$HOME/bender_runs/mpipi_gg}"
mkdir -p "$RUNS_DIR"
SETUP_SCRIPT="$HOME/mpipi_sim/scripts/run_mpipi_from_fasta.py"
SCRIPT_PATH="$(realpath "$0")"
NUM_PROTEINS=$(grep -c ">" "$FASTA_FILE")

echo "=============================================="
echo "  MPIPI-GG batch submission (CPU-only)"
echo "  FASTA:    $FASTA_FILE"
echo "  Proteins: $NUM_PROTEINS"
echo "  Chunk:    $CHUNK_IDX"
echo "  Time exponent (N>150): ${MPIPI_TIME_EXPONENT:-2.0}"
echo "=============================================="

# ── Pre-generate simulation files on login node ───────────────────────────
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
            --outdir "$outdir" \
            --time-exponent "${MPIPI_TIME_EXPONENT:-2.0}" 2>&1 | grep -E "Written|Error" || true
    fi
done

echo "Setup complete for $NUM_PROTEINS proteins."

# ── Write array job script to a temp file ──────────────────────────────────
# (temp file avoids heredoc/variable-expansion issues seen with inline sbatch)
TMPSCRIPT=$(mktemp /tmp/mpipi_array_XXXXXX.sh)

cat > "$TMPSCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=mpipi_chunk${CHUNK_IDX}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
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

# ── Auto-resubmit next chunk ───────────────────────────────────────────────
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
            --wrap="bash '${SCRIPT_PATH}' '${NEXT_FASTA}' ${NEXT_CHUNK}")
        echo "Resubmitter: ${RESUBMIT_JOB_ID}"
        echo "  -> next chunk: $NEXT_FASTA"
        echo "  -> starts: $RESUBMIT_TIME"
    else
        echo "  (no next chunk found at ${NEXT_FASTA} - no resubmission scheduled)"
    fi
fi

echo ""
echo "Monitor: bash ~/mpipi_sim/scripts/monitor.sh"