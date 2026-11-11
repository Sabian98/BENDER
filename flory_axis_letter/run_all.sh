#!/usr/bin/env bash
# Reproduce every number, figure and SI table of the Flory-axis letter from data/.
# Run from the repository root:  bash flory_axis_letter/run_all.sh
# Outputs go to $BENDER_OUT/flory_axis_letter (default ./output/flory_axis_letter).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
L=$ROOT/flory_axis_letter
DATA=${BENDER_DATA:-$ROOT/data}
OUT=${BENDER_OUT:-$ROOT/output}/flory_axis_letter
PY=${PYTHON:-python}
mkdir -p "$OUT/logs"
BIO=$OUT/bio.csv
GG=$DATA/mpipi_gg/mpipi_results_v2.csv

echo "1. Input table"
$PY "$L/prepare_inputs.py"

echo "2. Analyses (printed results in $OUT/logs)"
for s in stages dp letter_nums transfer sizectrl resid gbmresid; do
  $PY "$L/analysis/$s.py" "$BIO" > "$OUT/logs/$s.out"
done
for s in curves crossff; do
  $PY "$L/analysis/$s.py" "$BIO" "$GG" > "$OUT/logs/$s.out"
done

echo "3. SAXS map"
$PY "$L/analysis/saxs_map.py" "$BIO" "$DATA/flory_axis_letter/saxs_eval_63.csv" "$OUT" > "$OUT/logs/saxs_map.out"

echo "3b. Hydrodynamic radii and charge-hydropathy plot (Section S9, Table S10)"
$PY "$L/analysis/rh_ch_check.py" "$BIO" "$DATA/flory_axis_letter/marsh_2010_table_s1.csv" "$OUT/si" > "$OUT/logs/rh_ch_check.out"

echo "4. Gene Ontology enrichment"
# data/flory_axis_letter/uniprot_annotations.tsv was retrieved on 2026-11-08 with
#   python flory_axis_letter/analysis/fetch_go.py "$BIO" data/flory_axis_letter/uniprot_annotations.tsv
$PY "$L/analysis/go_enrich.py" "$BIO" "$DATA/flory_axis_letter/uniprot_annotations.tsv" "$OUT/go.tsv" > "$OUT/logs/go_enrich.out"

echo "5. Figures 2 to 5, SI and TOC"
$PY "$L/figures/make_figures.py" "$BIO" "$GG" "$OUT/saxs63_with_map.csv" "$OUT/go.tsv" "$OUT"
$PY "$L/figures/make_si.py" "$BIO" "$GG" "$OUT/saxs63_with_map.csv" "$DATA/saxs/traj_vs_saxs_nu.csv" "$OUT/go.tsv" "$OUT" "$OUT/si"
$PY "$L/figures/make_toc_icons.py" "$OUT" "$BIO"

# Figure 1 needs the trajectories from Hugging Face and ChimeraX, so it is not run here:
#   python flory_axis_letter/figures/ensembles/extract_ensembles.py <CALVADOS-2 300K dir> <MPIPI-Recharged 300K dir> "$OUT/ensembles"
#   cd "$OUT/ensembles" && ChimeraX --nostatus --exit "$L/figures/ensembles/render_ensembles.cxc"
#   python flory_axis_letter/figures/make_fig_ensembles.py "$BIO" "$OUT"
echo "Done: $OUT"
