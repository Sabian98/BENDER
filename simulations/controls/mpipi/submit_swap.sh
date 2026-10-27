#!/bin/bash
# =============================================================================
# submit_swap.sh: MPIPI-Recharged parameter-swap runs for the BENDER JCTC paper (2026-10-21)
#
# Runs every sequence of ../inputs/swap_set_300K.fasta under each swap table (swap_<name>.csv in this folder)
# with run_openmpipi_swap.py, at 300 K and 150 mM, with the BENDER MPIPI-Recharged protocol.
#
# Usage (activate the calvados env first; jobs inherit it):
#   bash submit_swap.sh                          # all four swaps: aro his aliph all
#   bash submit_swap.sh aro his                  # selected swaps
#   options: --outroot ~/bender_runs/controls     (default; runs go to <outroot>/<prefix>_<name>/300K/<ID>/)
#            --exclude GPU4,GPU6,GPU8            (nodes to keep jobs off, as in submit_openmpipi.sh; '' for none)
#            --fasta FILE                        (default ../inputs/swap_set_300K.fasta)
#            --prefix NAME                       (default swap; 2026-10-22: bremer2022_swap for the hnRNPA1 variants,
#                                                 saxs_swap for the SAXS benchmark with the gsp and gspaliph swaps)
#   e.g. bash submit_swap.sh aro --fasta ../inputs/bremer2022_a1lcd_300K.fasta --prefix bremer2022_swap
#
# Only sequences without <outroot>/swap_<name>/300K/<ID>/<ID>.dcd are submitted, and unfinished runs resume
# from their last saved segment, so re-running this script retries leftovers.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FASTA="$(cd "$SCRIPT_DIR/../inputs" && pwd)/swap_set_300K.fasta"
OUT_ROOT="~/bender_runs/controls"
EXCLUDE="GPU4,GPU6,GPU8"
PREFIX="swap"
SWAPS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --outroot) OUT_ROOT="${2%/}"; shift 2 ;;
        --exclude) EXCLUDE="$2";      shift 2 ;;
        --fasta)   FASTA="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
        --prefix)  PREFIX="$2";       shift 2 ;;
        *)         SWAPS+=("$1");     shift ;;
    esac
done
[[ ${#SWAPS[@]} -eq 0 ]] && SWAPS=(aro his aliph all)
[[ -f "$FASTA" ]] || { echo "ERROR: $FASTA not found"; exit 1; }

for S in "${SWAPS[@]}"; do
    TABLE="$SCRIPT_DIR/swap_${S}.csv"
    [[ -f "$TABLE" ]] || { echo "ERROR: $TABLE not found"; exit 1; }
    ROOT="$OUT_ROOT/${PREFIX}_${S}"
    LOGDIR="$ROOT/300K/logs"
    mkdir -p "$LOGDIR"
    LIST="$LOGDIR/pending_$(date +%Y%m%d_%H%M%S).txt"
    : > "$LIST"
    N_TOTAL=0
    while read -r NAME; do
        N_TOTAL=$((N_TOTAL + 1))
        [[ -f "$ROOT/300K/$NAME/$NAME.dcd" ]] || echo "$NAME" >> "$LIST"
    done < <(grep '^>' "$FASTA" | sed 's/^>//' | cut -d'|' -f1)
    N_PENDING=$(wc -l < "$LIST" | tr -d ' ')
    echo "=== ${PREFIX}_${S}: pending $N_PENDING / $N_TOTAL -> $ROOT"
    (( N_PENDING == 0 )) && { rm -f "$LIST"; continue; }

    JOB_ID=$(sbatch --parsable \
        --job-name="${PREFIX}_${S}" \
        --output="$LOGDIR/sim_%a.out" \
        --error="$LOGDIR/sim_%a.err" \
        --array="1-${N_PENDING}" ${EXCLUDE:+--exclude="$EXCLUDE"} <<EOF
#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

NAME=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${LIST}")
RUNDIR="${ROOT}/300K/\$NAME"
DATADIR="${ROOT}/300K/data/\$NAME"
if [[ -f "\$RUNDIR/\$NAME.dcd" ]]; then echo "\$NAME already done"; exit 0; fi
echo "\$NAME ${PREFIX}_${S} on \$(hostname)"

python "${SCRIPT_DIR}/run_openmpipi_swap.py" --name "\$NAME" --fasta "${FASTA}" --temp 300 --ionic 0.15 \
    --outdir "${ROOT}" --swap "${TABLE}"

# clean up only finished runs, so unfinished ones keep state.xml and segments to resume from
if [[ -f "\$DATADIR/conf_prop.csv" ]]; then
    find "\$DATADIR" -type f ! -name "conf_prop.csv" -delete
    find "\$RUNDIR" -type f ! \( -name "*.dcd" -o -name "top.pdb" \) -delete
fi
EOF
    )
    echo "  submitted $JOB_ID ($N_PENDING tasks)"
done
