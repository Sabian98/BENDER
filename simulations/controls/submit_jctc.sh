#!/bin/bash
# =============================================================================
# submit_jctc.sh: the control and parameter-swap run sets of the BENDER paper. See README.md.
#
# Usage (activate an environment with OpenMM, CALVADOS and OpenMpipi first; jobs inherit it):
#   bash submit_jctc.sh                    # every set
#   bash submit_jctc.sh bremer_c2 swap     # selected sets
# Sets: bremer_c2 bremer_mr termini termini_his swap bremer_swap saxs_swap
#
# The CALVADOS-2 and MPIPI-Recharged sets use the submitters in ../calvados2 and ../mpipi_recharged;
# override with CALVADOS_DIR=... OPENMPIPI_DIR=... bash submit_jctc.sh
# Every set has its own output root, because the termini and swap sets reuse BENDER accession names.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CALVADOS_DIR="${CALVADOS_DIR:-$HERE/../calvados2}"
OPENMPIPI_DIR="${OPENMPIPI_DIR:-$HERE/../mpipi_recharged}"
DATA_ROOT="${DATA_ROOT:-$HOME/bender_runs/controls}"          # CALVADOS-2 runs
GENERAL_ROOT="${GENERAL_ROOT:-$HOME/bender_runs/controls}"  # MPIPI-Recharged runs

SETS=("$@")
[[ ${#SETS[@]} -eq 0 ]] && SETS=(bremer_c2 bremer_mr termini termini_his swap bremer_swap saxs_swap)

grep -q -- '--charge-termini' "$CALVADOS_DIR/submit_calvados.sh" \
    || { echo "ERROR: $CALVADOS_DIR/submit_calvados.sh has no --charge-termini"; exit 1; }

for SET in "${SETS[@]}"; do
    echo ""
    echo "##### $SET"
    case "$SET" in
        bremer_c2)
            bash "$CALVADOS_DIR/submit_calvados.sh" "$HERE/../inputs/bremer2022_a1lcd_300K.fasta" \
                --outroot "$DATA_ROOT/bremer2022" ;;
        bremer_mr)
            bash "$OPENMPIPI_DIR/submit_openmpipi.sh" "$HERE/../inputs/bremer2022_a1lcd_300K.fasta" \
                --outroot "$GENERAL_ROOT/bremer2022" ;;
        termini)
            bash "$CALVADOS_DIR/submit_calvados.sh" "$HERE/../inputs/termini_control_300K.fasta" \
                --charge-termini none --outroot "$DATA_ROOT/termini_control" ;;
        termini_his)
            bash "$CALVADOS_DIR/submit_calvados.sh" "$HERE/../inputs/termini_control_300K.fasta" \
                --charge-termini none --ph 6.0 --outroot "$DATA_ROOT/termini_his_control" ;;
        swap)
            bash "$HERE/mpipi/submit_swap.sh" --outroot "$GENERAL_ROOT" ;;
        bremer_swap)
            bash "$HERE/mpipi/submit_swap.sh" aro --fasta "$HERE/../inputs/bremer2022_a1lcd_300K.fasta" \
                --prefix bremer2022_swap --outroot "$GENERAL_ROOT" ;;
        saxs_swap)
            bash "$HERE/mpipi/submit_swap.sh" gsp gspaliph --fasta "$HERE/../inputs/saxs_swap_300K.fasta" \
                --prefix saxs_swap --outroot "$GENERAL_ROOT" ;;
        *)
            echo "unknown set $SET (bremer_c2 bremer_mr termini termini_his swap bremer_swap saxs_swap)"; exit 1 ;;
    esac
done
