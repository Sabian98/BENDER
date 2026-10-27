#!/bin/bash
# =============================================================================
# resubmit_incomplete.sh
#
# Scans all 4 temperature run folders and resubmits any protein simulation
# that did not complete (no "Total wall time" in log.lammps).
#
# Skips runs whose log.lammps was modified in the last 60 minutes (likely
# still running). Delete that log manually to force a resubmit.
#
# Usage:
#   bash resubmit_incomplete.sh
# =============================================================================

RUNS_BASE="${RUNS_BASE:-$HOME/bender_runs/mpipi_gg}"
TEMPS=("278k_runs" "288k_runs" "310k_runs" "320k_runs")

INCOMPLETE_LIST="$RUNS_BASE/incomplete_runs.txt"
> "$INCOMPLETE_LIST"

echo "=============================================="
echo "  MPIPI-GG resubmission scan"
echo "=============================================="

total_checked=0
total_complete=0
total_running=0
total_incomplete=0

for temp_dir in "${TEMPS[@]}"; do
    folder="$RUNS_BASE/$temp_dir"
    if [ ! -d "$folder" ]; then
        echo "  $temp_dir: not found, skipping"
        continue
    fi

    n_complete=0
    n_running=0
    n_incomplete=0

    for protdir in "$folder"/*/; do
        [ -d "$protdir" ] || continue
        [ -f "$protdir/run.in" ]  || continue
        ((total_checked++))

        log="$protdir/log.lammps"

        # Complete
        if grep -q "Total wall time" "$log" 2>/dev/null; then
            ((n_complete++)); ((total_complete++))
            continue
        fi

        # Likely still running (log touched in last 60 min)
        if [ -f "$log" ] && [ -n "$(find "$log" -mmin -60 2>/dev/null)" ]; then
            ((n_running++)); ((total_running++))
            continue
        fi

        # Incomplete / failed / never ran
        echo "$protdir" >> "$INCOMPLETE_LIST"
        ((n_incomplete++)); ((total_incomplete++))
    done

    echo "  $temp_dir: $n_complete complete | $n_running running | $n_incomplete incomplete"
done

echo "----------------------------------------------"
echo "  Total checked:    $total_checked"
echo "  Complete:         $total_complete"
echo "  Running (skip):   $total_running"
echo "  To resubmit:      $total_incomplete"
echo "=============================================="

if [ "$total_incomplete" -eq 0 ]; then
    echo "Nothing to resubmit. All done!"
    rm -f "$INCOMPLETE_LIST"
    exit 0
fi

echo ""
echo "Submitting SLURM array job for $total_incomplete proteins..."

TMPSCRIPT=$(mktemp "$RUNS_BASE/mpipi_resub_XXXXXX.sh")

cat > "$TMPSCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=mpipi_resub
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --array=1-${total_incomplete}

PROTDIR=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$INCOMPLETE_LIST")
NAME=\$(basename "\$PROTDIR")
TEMP=\$(basename \$(dirname "\$PROTDIR"))

echo "======================================================"
echo "  Task \${SLURM_ARRAY_TASK_ID}/${total_incomplete} | \$TEMP | \$NAME"
echo "  Node: \$(hostname) | \$(date)"
echo "======================================================"

rm -f "\$PROTDIR"/restart.*

srun lmp -in "\$PROTDIR/run.in"
EXIT=\$?

[ \$EXIT -eq 0 ] && rm -f "\$PROTDIR"/restart.* && echo "Cleaned restart files"
echo "Done: \$(date) | Exit: \$EXIT"
exit \$EXIT
EOF

JOB_ID=$(sbatch --parsable "$TMPSCRIPT")
rm -f "$TMPSCRIPT"

echo "Array job submitted: $JOB_ID (tasks 1-$total_incomplete)"
echo "Incomplete list saved: $INCOMPLETE_LIST"
echo ""
echo "Monitor:  squeue -u \$USER"
