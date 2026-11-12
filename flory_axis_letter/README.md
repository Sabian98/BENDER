# Taking Charge across the Tree of Life: Universal Flory Scaling of Intrinsically Disordered Proteins

Code for the letter by T. Rahman and V. N. Uversky (manuscript for *The Journal of Physical Chemistry Letters*). The
letter orders the IDPs of BENDER from six taxa by their Flory exponent ν in CALVADOS-2, Mpipi-Recharged and Mpipi-GG
and asks how charge architecture changes along this axis in each taxon.

Everything runs from the tables in `data/` of this repository, so no trajectories are needed except for Figure 1.

```bash
conda activate bender
bash flory_axis_letter/run_all.sh          # writes output/flory_axis_letter/
```

## Inputs

| File | Contents |
|---|---|
| `data/bender_bio/BENDER_BIO.csv` | sequences, taxa, descriptors and CALVADOS-2 ν (`nu_orig`) |
| `data/mpipi_recharged/recharged_300K_features.csv` | Mpipi-Recharged ν (`nu`) and its fit |
| `data/mpipi_gg/mpipi_results_v2.csv` | Mpipi-GG ν for the shared subset |
| `data/saxs/traj_vs_saxs_nu.csv` | SAXS construct metadata |
| `data/flory_axis_letter/saxs_eval_63.csv` | the 63 SASBDB constructs of at most 256 residues, with measured and simulated radii of gyration |
| `data/flory_axis_letter/marsh_2010_table_s1.csv` | the 32 IDPs with measured hydrodynamic radii and sequences from Table S1 of Marsh and Forman-Kay, *Biophys. J.* 2010, 98, 2383 |
| `data/flory_axis_letter/exclude_rice_japonica.txt` | 1,422 *Oryza sativa* subsp. *japonica* entries removed from every analysis (1,254 of them in the analyzed set), from `analysis/rice_ref_match.py` |
| `data/flory_axis_letter/rice_ref_match.tsv` | every BENDER *japonica* entry with its UniProt status and whether it matches the current rice reference proteome |
| `data/flory_axis_letter/uniprot_annotations.tsv` | Gene Ontology terms and UniProt keywords, retrieved 2026-11-08 with `analysis/fetch_go.py` |

`prepare_inputs.py` joins the first two tables into `bio.csv`, removes the withdrawn rice entries and marks the
sequences whose Flory fit reaches R² ≥ 0.99 in each force field. Every script then reads `bio.csv`.

The rice entries are removed because UniProt deleted them in January 2026 ("not part of a reference proteome") and
they share no 20-residue segment with any protein of the current rice reference proteome (UP000059680), whereas 88%
of the rice entries still in UniProt match it exactly. To redo the check:

```bash
curl -L -o UP000059680.fasta.gz "https://rest.uniprot.org/uniprotkb/stream?format=fasta&compressed=true&query=(proteome:UP000059680)"
gunzip UP000059680.fasta.gz
python flory_axis_letter/analysis/rice_ref_match.py output/flory_axis_letter/bio.csv UP000059680.fasta output/rice_ref
```

(run it on a `bio.csv` built before the exclusion, for example with the exclusion file temporarily emptied).

## Scripts

| Script | Output | In the letter |
|---|---|---|
| `prepare_inputs.py` | `bio.csv` | data set |
| `analysis/rice_ref_match.py` | UniProt status and reference-proteome match of every rice entry (`rice_ref_match.tsv`, `exclude_rice.txt`) | Computational Methods |
| `analysis/stages.py` | stage markers per taxon in both force fields (FCR minimum, onsets of the rise in \|NCPR\| and SCD, Lysine over Arginine crossing) | Figure 3, Table S3 |
| `analysis/letter_nums.py` | occupancy χ², branch correlations, FCR minima, share of between-taxon variance in mean ν explained by occupancy | text, Figure 5 |
| `analysis/dp.py` | occupancy of the diagram of states and median ν per region | Figure 5, Table S5 |
| `analysis/curves.py` | decile-curve agreement between force fields | Figure 4, Table S4 |
| `analysis/crossff.py` | Mpipi-GG branch check and transfer of the map between force fields | Figure 4, Table S2 |
| `analysis/transfer.py`, `sizectrl.py` | within-taxon and leave-one-taxon-out prediction of ν, and a size-matched control | Figure 5, Table S7 |
| `analysis/resid.py`, `gbmresid.py` | taxon offsets of linear and flexible models | Figure 5, Tables S6 and S7 |
| `analysis/desc.py` | sequence descriptors for constructs outside BENDER | used by `saxs_map.py` |
| `analysis/saxs_map.py` | sequence-only map of ν_eff against SAXS (`saxs63_with_map.csv`) | Figure 4, Section S7, Table S8 |
| `analysis/rh_ch_check.py` | nonlinear model against measured hydrodynamic radii, and the charge-hydropathy plot (`tab_rh.tex`) | Section S9, Table S10 |
| `analysis/fetch_go.py`, `go_enrich.py` | Gene Ontology and keyword enrichment (`go.tsv`) | Figure 5, Table S9 |
| `figures/make_figures.py` | Figures 2 to 5 | |
| `figures/make_si.py` | Figures S1 and S2, Tables S1 to S9 | |
| `figures/make_toc_icons.py` | TOC graphic (taxon icons in `figures/taxon_icons/`) | |
| `figures/ensembles/`, `figures/make_fig_ensembles.py` | Figure 1 from the trajectories (needs ChimeraX, see `run_all.sh`) | Figure 1 |

The printed results of each analysis are written to `output/flory_axis_letter/logs/`.
