#!/bin/bash
# =============================================================================
# submit_openmpipi.sh — MPIPI-Recharged (OpenMpipi) twin of submit_calvados.sh
#
# The BENDER set is split into input/combined_chunk_<N>.fasta (800 sequences
# each, 13 chunks). Only ONE chunk is in the queue at a time: when a chunk is
# finished (after up to MAX_ATTEMPTS rounds of retrying failures), a small
# follow-up job submits the next chunk automatically.
#
# Usage (activate the calvados env first; jobs inherit it):
#   bash submit_openmpipi.sh                  # start at chunk 1, chain through all chunks
#   bash submit_openmpipi.sh 5                # start at chunk 5, chain onwards
#   bash submit_openmpipi.sh 5 --no-chain     # chunk 5 only
#   bash submit_openmpipi.sh <file.fasta>     # any FASTA (chains only if named combined_chunk_<N>)
#   options: --temps "288 300" --outroot ~/bender_runs/mpipi_recharged --ionic 0.09   (ionic strength in M, default 0.15)
#            --phospho          keep B (pSer) and J (pThr) as phosphate-like Glu beads (run_openmpipi_idp.py
#                               --phospho); always pass it for the phospho FASTA files, without it B -> D and
#                               J -> L. Added 2026-10-13.
#            --exclude GPU4,GPU6,GPU8   nodes to keep jobs off (default GPU4,GPU6,GPU8, as in submit_calvados.sh;
#                               '' for none). Added 2026-10-13.
#
# Temperature: --temps overrides; otherwise taken from the filename
# (…_310K….fasta → 310), default 300. Several temps = several parallel chains.
#
# Output : <outroot>/<T>K/<name>/        (<name>.dcd + top.pdb)
#          <outroot>/<T>K/data/<name>/   (conf_prop.csv)
# Logs   : <outroot>/<T>K/logs/
# <outroot> defaults to ~/bender_runs/mpipi_recharged.
#
# Only proteins without <outroot>/<T>K/<name>/<name>.dcd are submitted;
# unfinished runs resume from their last saved segment.
# =============================================================================

set -euo pipefail

BATCH_SIZE=800
MAX_ATTEMPTS=3
OUT_ROOT="${OUT_ROOT:-$HOME/bender_runs/mpipi_recharged}"

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# --- Args ---
FASTA_ARG="1"
TEMPS_OVERRIDE=""
ATTEMPT=1
IONIC=0.15
CHAIN=1
PHOSPHO_OPT=""
EXCLUDE="GPU4,GPU6,GPU8"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --temps)    TEMPS_OVERRIDE="$2"; shift 2 ;;
        --outroot)  OUT_ROOT="${2%/}";   shift 2 ;;
        --attempt)  ATTEMPT="$2";        shift 2 ;;
        --ionic)    IONIC="$2";          shift 2 ;;
        --no-chain) CHAIN=0;             shift ;;
        --phospho)  PHOSPHO_OPT="--phospho"; shift ;;
        --exclude)  EXCLUDE="$2";        shift 2 ;;
        *)          FASTA_ARG="$1";      shift ;;
    esac
done

if [[ "$FASTA_ARG" =~ ^[0-9]+$ ]]; then
    FASTA="$SCRIPT_DIR/input/combined_chunk_${FASTA_ARG}.fasta"
else
    FASTA="$FASTA_ARG"
fi
[[ -f "$FASTA" ]] || { echo "ERROR: $FASTA not found"; exit 1; }
FASTA="$(realpath "$FASTA")"
TAG="$(basename "$FASTA")"; TAG="${TAG%.*}"

# Next chunk in the chain (only for combined_chunk_<N> files)
NEXT_FASTA=""
if (( CHAIN )) && [[ "$TAG" =~ ^combined_chunk_([0-9]+)$ ]]; then
    CAND="$(dirname "$FASTA")/combined_chunk_$(( BASH_REMATCH[1] + 1 )).fasta"
    [[ -f "$CAND" ]] && NEXT_FASTA="$CAND"
fi

extract_temp() {
    if [[ "$1" =~ ([0-9]+)K ]]; then echo "${BASH_REMATCH[1]}"; else echo "300"; fi
}
TEMPS="${TEMPS_OVERRIDE:-$(extract_temp "$TAG")}"

# Accession per header: sp|ACC|..., tr|ACC|..., "ACC|Group|N=..", "ACC | ..." → ACC
fasta_names() {
    grep '^>' "$1" | sed 's/^>//' | awk -F'|' '{
        f1=$1; gsub(/^[ \t]+|[ \t]+$/,"",f1)
        id=(f1=="sp" || f1=="tr") ? $2 : f1
        gsub(/^[ \t]+/,"",id); split(id,a,/[ \t]+/); print a[1]
    }'
}

for TEMP in $TEMPS; do
    LOGDIR="$OUT_ROOT/${TEMP}K/logs"
    mkdir -p "$LOGDIR"

    # Proteins still missing <name>/<name>.dcd at this temperature
    LIST="$LOGDIR/pending_${TAG}_${TEMP}K_attempt${ATTEMPT}_$(date +%Y%m%d_%H%M%S).txt"
    : > "$LIST"
    N_TOTAL=0
    while read -r NAME; do
        N_TOTAL=$((N_TOTAL + 1))
        [[ -f "$OUT_ROOT/${TEMP}K/$NAME/$NAME.dcd" ]] || echo "$NAME" >> "$LIST"
    done < <(fasta_names "$FASTA")
    N_PENDING=$(wc -l < "$LIST" | tr -d ' ')

    echo ""
    echo "=== $TAG @ ${TEMP}K  (attempt $ATTEMPT/$MAX_ATTEMPTS) ==="
    echo "  FASTA   : $FASTA"
    echo "  Pending : $N_PENDING / $N_TOTAL"

    if (( N_PENDING == 0 )); then
        echo "  All done — nothing to submit."
        rm -f "$LIST"
        if [[ -n "$NEXT_FASTA" ]]; then
            echo "  Moving on to $(basename "$NEXT_FASTA")"
            bash "$SCRIPT_PATH" "$NEXT_FASTA" --temps "$TEMP" --ionic "$IONIC" --outroot "$OUT_ROOT" ${PHOSPHO_OPT} --exclude "$EXCLUDE"
        fi
        continue
    fi

    NUM_BATCHES=$(( (N_PENDING + BATCH_SIZE - 1) / BATCH_SIZE ))
    BATCH_JOB_IDS=()

    for (( b=1; b<=NUM_BATCHES; b++ )); do
        START=$(( (b - 1) * BATCH_SIZE + 1 ))
        END=$(( b * BATCH_SIZE < N_PENDING ? b * BATCH_SIZE : N_PENDING ))
        COUNT=$(( END - START + 1 ))
        OFFSET=$(( START - 1 ))

        JOB_ID=$(sbatch --parsable \
            --job-name="MPI_${TAG}_${TEMP}K_b${b}" \
            --output="$LOGDIR/sim_${TAG}_${TEMP}K_b${b}_%a.out" \
            --error="$LOGDIR/sim_${TAG}_${TEMP}K_b${b}_%a.err" \
            --array="1-${COUNT}" ${EXCLUDE:+--exclude="$EXCLUDE"} <<EOF
#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

NAME=\$(sed -n "\$(( SLURM_ARRAY_TASK_ID + ${OFFSET} ))p" "${LIST}")
RUNDIR="${OUT_ROOT}/${TEMP}K/\$NAME"
DATADIR="${OUT_ROOT}/${TEMP}K/data/\$NAME"

if [[ -f "\$RUNDIR/\$NAME.dcd" ]]; then
    echo "\$NAME @ ${TEMP}K already done"; exit 0
fi
echo "\$NAME @ ${TEMP}K on \$(hostname)"

python "${SCRIPT_DIR}/run_openmpipi_idp.py" --name "\$NAME" --fasta "${FASTA}" --temp ${TEMP} --ionic ${IONIC} --outdir "${OUT_ROOT}" ${PHOSPHO_OPT}

# Clean up only finished runs, so unfinished ones keep state.xml / segments to resume from
if [[ -f "\$DATADIR/conf_prop.csv" ]]; then
    find "\$DATADIR" -type f ! -name "conf_prop.csv" -delete
    find "\$RUNDIR" -type f ! \( -name "*.dcd" -o -name "top.pdb" \) -delete
fi
EOF
        )
        echo "  Batch $b/$NUM_BATCHES (${COUNT} tasks) → $JOB_ID"
        BATCH_JOB_IDS+=("$JOB_ID")
    done

    # Follow-up after this chunk: retry leftovers, or (last attempt) start the next chunk
    CHAIN_OPT=""; (( CHAIN )) || CHAIN_OPT="--no-chain"
    if (( ATTEMPT < MAX_ATTEMPTS )); then
        FOLLOW="bash '${SCRIPT_PATH}' '${FASTA}' --temps ${TEMP} --ionic ${IONIC} --outroot '${OUT_ROOT}' --attempt $((ATTEMPT + 1)) ${CHAIN_OPT} ${PHOSPHO_OPT} --exclude '${EXCLUDE}'"
        WHAT="retries leftovers of $TAG"
    elif [[ -n "$NEXT_FASTA" ]]; then
        FOLLOW="bash '${SCRIPT_PATH}' '${NEXT_FASTA}' --temps ${TEMP} --ionic ${IONIC} --outroot '${OUT_ROOT}' ${PHOSPHO_OPT} --exclude '${EXCLUDE}'"
        WHAT="last attempt for $TAG; then submits $(basename "$NEXT_FASTA")"
    else
        FOLLOW=""
        echo "  Last attempt, no further chunk — check $LOGDIR for failures."
    fi

    if [[ -n "$FOLLOW" ]]; then
        DEP="afterany:$(IFS=:; echo "${BATCH_JOB_IDS[*]}")"
        RID=$(sbatch --parsable \
            --job-name="next_${TAG}_${TEMP}K" \
            --output="$LOGDIR/followup_${TAG}_${TEMP}K_%j.out" \
            --error="$LOGDIR/followup_${TAG}_${TEMP}K_%j.err" \
            --dependency="$DEP" \
            --ntasks=1 --mem=1G ${EXCLUDE:+--exclude="$EXCLUDE"} \
            --wrap="$FOLLOW")
        echo "  Follow-up → $RID ($WHAT)"
    fi
done

echo ""
echo "Done."
