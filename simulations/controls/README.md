# Control and parameter-swap simulations

These runs test where the two force fields disagree and whether the disagreement can be traced to specific
residue-pair terms. Every set uses the BENDER protocol (300 K, 150 mM, the length-dependent step budget, first half
of each trajectory discarded) except where the table says otherwise.

| Set | Force field | Sequences | Change from BENDER | Runs |
|---|---|---|---|---|
| `bremer_c2` | CALVADOS-2 | 38 hnRNPA1 LCD variants | none | 38 |
| `bremer_mr` | MPIPI-Recharged | the same 38 | none | 38 |
| `termini` | CALVADOS-2 | 300 BENDER sequences | uncharged termini | 300 |
| `termini_his` | CALVADOS-2 | the same 300 | uncharged termini and pH 6.0 | 300 |
| `swap` | MPIPI-Recharged | 300 BENDER sequences | one class of pair interactions set to CALVADOS-2 (four swaps) | 1,200 |
| `bremer_swap` | MPIPI-Recharged | the 38 hnRNPA1 LCD variants | Arginine–Aromatic and Aromatic–Aromatic pairs matched to CALVADOS-2 | 38 |
| `saxs_swap` | MPIPI-Recharged | the 77 SAXS benchmark constructs | Gly/Ser/Pro pairs, or Gly/Ser/Pro and aliphatic pairs, matched to CALVADOS-2 | 154 |

## Inputs (`../inputs/`)

**hnRNPA1 LCD variants.** `bremer2022_a1lcd_300K.fasta` holds the 38 variants of Bremer et al. (Nat. Chem. 2022),
each 137 residues including the N-terminal GS left after tag cleavage, taken from the paper's Supplementary Table 1.
`bremer2022_a1lcd_300K.tsv` maps the run names (for example `A1LCD_p7R_m10D`) to the construct names and gives the
measured SAXS Rg and apparent ν for the 20 variants that have them. CALVADOS-2 was parameterized partly on variants of
this domain, so this comparison tests MPIPI-Recharged but cannot validate CALVADOS-2.

**Termini and histidine controls.** `termini_control_300K.fasta` holds 300 BENDER sequences, 50 per taxon (25 from the
top decile of |Δν| within the taxon and 25 drawn at random from the rest, none longer than 400 residues; seed
20261020). The `termini` set removes the terminal charges, which MPIPI-Recharged does not have. The `termini_his` set
also lowers the pH to 6.0, where the CALVADOS-2 histidine charge (+0.5) matches the effective charge of about +0.51 in
the MPIPI-Recharged Yukawa table (at pH 7.5 it is +0.03).

**Parameter swap.** `swap_set_300K.fasta` holds 300 BENDER sequences of at most 250 residues (seed 20261021), and
`swap_set_300K.tsv` gives each sequence's group, ν in both force fields and Δν.

| Group | Sequences | Mean Δν | Selection |
|---|---|---|---|
| `contested` | 100 | −0.041 | top decile of Arginine + Aromatic fraction, and Lysine fraction below the median |
| `his_rich` | 50 | +0.005 | top decile of Histidine fraction |
| `aliph_rich` | 50 | +0.021 | top decile of A + I + L + M + V fraction |
| `control` | 100 | +0.006 | none of the three |

**SAXS swap.** `saxs_swap_300K.fasta` holds the exact sequences of the MPIPI-Recharged SAXS runs, with the Gly/Pro-rich
third first and each header tagged `|GP=high|mid|low`. `saxs_benchmark_300K.tsv` gives each construct's subset, its
Gly, Pro and Ser fractions, its G + P third and the measured and unswapped Rg in all three force fields.

## Swaps (`mpipi/`)

Each swap changes one class of MPIPI-Recharged interactions to match CALVADOS-2 and leaves everything else unchanged.

| Swap | Pairs changed | How |
|---|---|---|
| `swap_aro` | RF RW RY FF FW FY WW WY YY | Wang–Frenkel ε set so that the pair's B2 equals its CALVADOS-2 B2 |
| `swap_aliph` | the 15 pairs of A I L M V | the same B2 matching |
| `swap_his` | every pair with His | Yukawa amplitudes of His scaled by 0.0602, giving it the CALVADOS-2 charge |
| `swap_all` | all three | |
| `swap_gsp` | GG GS GP SS SP PP | the same B2 matching |
| `swap_gspaliph` | all 36 pairs among G S P A I L M V | the same B2 matching |

B2 values use the full pair potentials (Wang–Frenkel plus Yukawa, and Ashbaugh–Hatch plus Debye–Hückel) at 300 K and
150 mM, from `analysis/mpipi_recharged/b2_mechanism.py`. In the Wang–Frenkel form ε scales both the repulsive core and
the attraction, so B2 rises to a maximum (at ε of about 0.14–0.26 kJ/mol) and then falls. Matching is done on the
falling branch, which keeps a full excluded-volume core. Every swapped pair matches its target B2 to at least four
decimals (`mpipi/swap_pairs_b2.csv`), and the edited OpenMM systems reproduce the edited analytic potentials to a
relative 1e-13 (`mpipi/make_swap_tables.py`).

| File | Contents |
|---|---|
| `submit_jctc.sh` | submits the sets: `bash submit_jctc.sh [bremer_c2 bremer_mr termini termini_his swap bremer_swap saxs_swap]` |
| `mpipi/submit_swap.sh` | the swap job arrays: `bash mpipi/submit_swap.sh [aro his aliph all gsp gspaliph]` |
| `mpipi/run_openmpipi_swap.py` | `simulations/mpipi_recharged/run_openmpipi_idp.py` plus `--swap TABLE`; without `--swap` it is identical |
| `mpipi/mpipi_swap.py` | `apply_swap`, which edits the `wf_table` and `A_table` after `get_mpipi_system` builds them |
| `mpipi/make_swap_tables.py` | writes the `swap_*.csv` tables and checks them against OpenMM |
| `mpipi/swap_*.csv` | swap tables (rows `wf_eps a b value` or `yk_scale a - value`) |
| `mpipi/swap_pairs_b2.csv`, `mpipi/swap_his_b2.csv` | B2 of each swapped pair before and after the swap |

## Running

The scripts were written for a SLURM cluster (USF GAIVI) and need an environment with OpenMM, CALVADOS and OpenMpipi.
Set `CALVADOS_DIR` and `OPENMPIPI_DIR` if `simulations/calvados2` and `simulations/mpipi_recharged` are elsewhere, and
`DATA_ROOT` and `GENERAL_ROOT` to choose the output roots (default `~/bender_runs/controls`). Keep these roots separate
from the BENDER runs, since the termini and swap sets reuse BENDER accession names.

ν and Rg of the finished runs are computed with the BENDER estimators by `figures/source/analyze_jctc_runs.py`, which
reads the runs from `trajectories/controls/` and writes the per-set tables in `data/controls/`.
