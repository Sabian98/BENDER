#!/bin/bash
# =============================================================================
# submit_calvados.sh
#
# Usage:
#   bash submit_calvados.sh                                  # every <script dir>/input/*.fasta
#   bash submit_calvados.sh <file.fasta>                     # one FASTA, temp from its name
#   bash submit_calvados.sh <file.fasta> --temps "278 288 300 310 320" --outroot ~/bender_runs/calvados2_protists
#   bash submit_calvados.sh <file.fasta> --ph 7.0          # pH (default 7.5, as in all earlier runs)
#   bash submit_calvados.sh <file.fasta> --ionic 0.09      # ionic strength in M (default 0.15, as in all earlier runs)
#   bash submit_calvados.sh <file.fasta> --residues input_phospho/residues_phospho.csv
#                                                          # residues CSV passed to prepare.py (default: input/residues_CALVADOS2.csv);
#                                                          # letters it defines (B = pSer, J = pThr) are kept. Added 2026-10-12.
#   bash submit_calvados.sh <file.fasta> --charge-termini none --outroot ~/bender_runs/termini_control
#                                                          # terminal charges passed to prepare.py (default both, as in all
#                                                          # earlier runs; none matches MPIPI-Recharged). Added 2026-10-20.
#   bash submit_calvados.sh <file.fasta> --exclude GPU4,GPU6,GPU8
#                                                          # nodes to keep jobs off (default GPU4,GPU6,GPU8, which misbehave:
#                                                          # GPU6 never writes to /data, GPU8 fails every job; '' for none)
#
# Temperature: --temps overrides; otherwise taken from the filename
# (truncated_310K_rerun.fasta → 310), default 300.
#
# Output : <outroot>/<T>K/<name>/        (.dcd + top.pdb)
#          <outroot>/<T>K/data/<name>/   (conf_prop.csv)
# Logs   : <outroot>/<T>K/logs/
# <outroot> defaults to ~/bender_runs/calvados2.
#
# Only proteins without conf_prop.csv are submitted. After each FASTA×temp
# finishes, a resubmitter retries the leftovers (up to MAX_ATTEMPTS rounds).
# =============================================================================

set -euo pipefail

BATCH_SIZE=800
MAX_ATTEMPTS=3
OUT_ROOT="${OUT_ROOT:-$HOME/bender_runs/calvados2}"

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# --- Args ---
FASTA_ARG=""
TEMPS_OVERRIDE=""
ATTEMPT=1
PH=7.5
IONIC=0.15
RESIDUES=""
TERMINI="both"
EXCLUDE="GPU4,GPU6,GPU8"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --temps)    TEMPS_OVERRIDE="$2"; shift 2 ;;
        --outroot)  OUT_ROOT="${2%/}";   shift 2 ;;
        --attempt)  ATTEMPT="$2";        shift 2 ;;
        --ph)       PH="$2";             shift 2 ;;
        --ionic)    IONIC="$2";          shift 2 ;;
        --residues) RESIDUES="$2";       shift 2 ;;
        --charge-termini) TERMINI="$2";  shift 2 ;;
        --exclude)  EXCLUDE="$2";        shift 2 ;;
        *)          FASTA_ARG="$1";      shift ;;
    esac
done
RES_OPT=""
if [[ -n "$RESIDUES" ]]; then
    [[ "$RESIDUES" = /* ]] || RESIDUES="$SCRIPT_DIR/$RESIDUES"
    [[ -f "$RESIDUES" ]] || { echo "ERROR: residues file $RESIDUES not found"; exit 1; }
    RES_OPT="--residues '$RESIDUES'"
fi

if [[ -z "$FASTA_ARG" ]]; then
    mapfile -t FASTA_FILES < <(ls "$SCRIPT_DIR"/input/*.fasta 2>/dev/null | sort)
    [[ ${#FASTA_FILES[@]} -eq 0 ]] && { echo "ERROR: no .fasta in $SCRIPT_DIR/input/"; exit 1; }
elif [[ "$FASTA_ARG" == *.fasta || "$FASTA_ARG" == *.fa || "$FASTA_ARG" == */* ]]; then
    FASTA_FILES=("$FASTA_ARG")
else
    FASTA_FILES=("$SCRIPT_DIR/input/combined_chunk_${FASTA_ARG}.fasta")
fi

extract_temp() {
    if [[ "$1" =~ ([0-9]+)K ]]; then echo "${BASH_REMATCH[1]}"; else echo "300"; fi
}

# Accession per header: sp|ACC|..., tr|ACC|..., "ACC | ...", "ACC Rhizaria | ..." → ACC
fasta_names() {
    grep '^>' "$1" | sed 's/^>//' | awk -F'|' '{
        f1=$1; gsub(/^[ \t]+|[ \t]+$/,"",f1)
        id=(f1=="sp" || f1=="tr") ? $2 : f1
        gsub(/^[ \t]+/,"",id); split(id,a,/[ \t]+/); print a[1]
    }'
}

for FASTA in "${FASTA_FILES[@]}"; do
    [[ -f "$FASTA" ]] || { echo "SKIP: $FASTA not found"; continue; }
    FASTA="$(realpath "$FASTA")"
    TAG="$(basename "$FASTA")"; TAG="${TAG%.*}"
    TEMPS="${TEMPS_OVERRIDE:-$(extract_temp "$TAG")}"

    for TEMP in $TEMPS; do
        LOGDIR="$OUT_ROOT/${TEMP}K/logs"
        mkdir -p "$LOGDIR"

        # Proteins still missing conf_prop.csv at this temperature
        LIST="$LOGDIR/pending_${TAG}_${TEMP}K_attempt${ATTEMPT}_$(date +%Y%m%d_%H%M%S).txt"
        : > "$LIST"
        N_TOTAL=0
        while read -r NAME; do
            N_TOTAL=$((N_TOTAL + 1))
            [[ -f "$OUT_ROOT/${TEMP}K/data/$NAME/conf_prop.csv" ]] || echo "$NAME" >> "$LIST"
        done < <(fasta_names "$FASTA")
        N_PENDING=$(wc -l < "$LIST" | tr -d ' ')

        echo ""
        echo "=== $TAG @ ${TEMP}K  (attempt $ATTEMPT/$MAX_ATTEMPTS) ==="
        echo "  FASTA   : $FASTA"
        echo "  Pending : $N_PENDING / $N_TOTAL"

        if (( N_PENDING == 0 )); then
            echo "  All done — nothing to submit."
            rm -f "$LIST"
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
                --job-name="CAL_${TAG}_${TEMP}K_b${b}" \
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

if [[ -f "\$DATADIR/conf_prop.csv" ]]; then
    echo "\$NAME @ ${TEMP}K already done"; exit 0
fi
echo "\$NAME @ ${TEMP}K on \$(hostname)"

python "${SCRIPT_DIR}/prepare.py" --name "\$NAME" --fasta "${FASTA}" --temp ${TEMP} --ph ${PH} --ionic ${IONIC} --charge-termini ${TERMINI} --outdir "${OUT_ROOT}" ${RES_OPT}
python "\$RUNDIR/run.py" --path "\$RUNDIR"

if [ -d "\$DATADIR" ]; then
    find "\$DATADIR" -type f ! -name "conf_prop.csv" -delete
    find "\$DATADIR" -mindepth 1 -type d -empty -delete
fi
if [ -d "\$RUNDIR" ]; then
    find "\$RUNDIR" -type f ! \( -name "*.dcd" -o -name "top.pdb" \) -delete
    find "\$RUNDIR" -mindepth 1 -type d -empty -delete
fi
EOF
            )
            echo "  Batch $b/$NUM_BATCHES (${COUNT} tasks) → $JOB_ID"
            BATCH_JOB_IDS+=("$JOB_ID")
        done

        if (( ATTEMPT < MAX_ATTEMPTS )); then
            DEP="afterany:$(IFS=:; echo "${BATCH_JOB_IDS[*]}")"
            RID=$(sbatch --parsable \
                --job-name="resub_${TAG}_${TEMP}K" \
                --output="$LOGDIR/resubmit_${TAG}_${TEMP}K_%j.out" \
                --error="$LOGDIR/resubmit_${TAG}_${TEMP}K_%j.err" \
                --dependency="$DEP" \
                --ntasks=1 --mem=1G ${EXCLUDE:+--exclude="$EXCLUDE"} \
                --wrap="bash '${SCRIPT_PATH}' '${FASTA}' --temps ${TEMP} --ph ${PH} --ionic ${IONIC} --charge-termini ${TERMINI} --outroot '${OUT_ROOT}' --attempt $((ATTEMPT + 1)) --exclude '${EXCLUDE}'${RESIDUES:+ --residues '${RESIDUES}'}")
            echo "  Resubmitter → $RID (retries leftovers after ${BATCH_JOB_IDS[*]})"
        else
            echo "  Last attempt — no further resubmission. Check $LOGDIR for failures."
        fi
    done
done

echo ""
echo "Done."
