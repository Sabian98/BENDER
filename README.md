# BENDER

Code and per-sequence data for **"Conformational Ensembles of Disordered Proteins Across the Tree of Life"**
(T. Rahman, submitted to the *Journal of Chemical Theory and Computation*).

BENDER holds coarse-grained ensembles of 10,282 fully disordered proteins from six taxa (mammals, fungi,
plants, protists, bacteria and viruses), simulated with CALVADOS-2 and MPIPI-Recharged, with a 957-sequence subset
also simulated with MPIPI-GG. This repository contains the simulation inputs and run scripts, the analysis code that
turns trajectories into per-sequence properties, the per-sequence tables themselves and the scripts that draw every
figure and generated table of the paper.

## Data

| What | Where |
|---|---|
| Per-sequence properties, force-field agreement, PAE agreement, disorder checks, SAXS benchmark | `data/` in this repository (see `data/README.md`) |
| CALVADOS-2 trajectories (300 K) | [huggingface.co/datasets/taseef/BENDER](https://huggingface.co/datasets/taseef/BENDER) |
| MPIPI-GG trajectories and the per-sequence properties of both MPIPI models | [huggingface.co/datasets/taseef/BENDER-MPIPI](https://huggingface.co/datasets/taseef/BENDER-MPIPI) |

Sequences are distributed with the data because many of the UniProt accessions have since become obsolete.

## Layout

```
paths.py              every input and output location (override with environment variables, see below)
data/                 per-sequence tables used by the paper
simulations/          inputs and run scripts
  calvados2/            CALVADOS-2 (prepare.py, SLURM submitter)
  mpipi_recharged/      MPIPI-Recharged with OpenMpipi in OpenMM
  mpipi_gg/             MPIPI-GG with LAMMPS
  controls/             hnRNPA1 variants, termini and histidine controls, parameter swaps (see its README)
  inputs/               FASTA and TSV inputs of the control and swap sets
analysis/             trajectories to per-sequence properties
  bender_common.py      shared rules: frame selection, chain unwrapping, Flory fits, contacts
  traj_analysis_calvados_v2.py, mpipi/, mpipi_recharged/   per-force-field pipelines
  validation/           agreement with AlphaFold2 PAE and between force fields
  saxs/                 SAXS benchmark (77 constructs)
  tables/               assembly of the per-sequence tables, NCBI taxonomy check
  tests/                reference checks on two bundled trajectories
figures/source/       one script per figure or generated table
```

## Reproducing the figures

The figure scripts read only `data/` and write to `output/`:

```bash
conda env create -f environment.yml
conda activate bender
cd figures/source
python make_fig1_overview.py
```

| Figure | Script |
|---|---|
| TOC graphic | `make_toc_ring.py` |
| Fig. 1, BENDER from one protein to the dataset | `make_fig1_overview.py` |
| Fig. 2, cross-force-field disagreement | `make_fig_disagreement.py` |
| Fig. 3, residue-pair B2 mechanism | `make_fig_b2_mechanism.py` |
| Fig. 4, taxa | `make_fig_lineages.py` |
| Fig. 5, parameter swaps and hnRNPA1 variants | `make_fig_tests.py` |
| Fig. 6, chain length | `make_fig_length.py` |
| SI, taxonomic composition | `make_supp_species_sunburst.py` |
| SI, disorder checks | `make_supp_disorder.py`, `make_supp_disorder_tables.py` |
| SI, AlphaFold2 PAE agreement | `make_fig3_pae_validation.py` |
| SI, property distributions and Flory fits | `make_supp_distributions.py` |
| SI, SAXS benchmark and Kratky profiles | `make_supp_saxs_f4p.py plot` |
| SI, Gly/Ser/Pro swap on the SAXS benchmark | `make_fig_saxs_swap.py` |
| SI tables of the swaps and hnRNPA1 variants | `make_supp_tests_tables.py` |
| Numbers quoted in the text | `paper_numbers_recharged.py` |

Run from this repository, every figure above is pixel-identical to the one in the paper. The Kratky profiles also
need the experimental SAXS curves, which are not redistributed here. Download them from
[SASBDB](https://www.sasbdb.org) as `<SASBDB code>.dat` (codes in `data/saxs/traj_vs_saxs_rg.csv`) into
`data/saxs/sasbdb/`, or point `BENDER_SASBDB` at them.

## Recomputing the tables from trajectories

The analysis scripts need the trajectories, laid out under `trajectories/` as described in `paths.py`. Set
`BENDER_TRAJ` if they are elsewhere, and `BENDER_WORK` for intermediate outputs (per-frame features and contact maps).
For example, CALVADOS-2 properties at 300 K come from

```bash
python analysis/traj_analysis_calvados_v2.py run --graph --workers 8 --out work/calvados2_300K
```

and MPIPI-Recharged properties from `analysis/mpipi_recharged/recharged_nu.py` and `recharged_300K_features.py`.
`analysis/tests/test_reference.py` checks the shared rules against two bundled reference trajectories.

## Running simulations

`simulations/` holds the scripts as run on a SLURM cluster (USF GAIVI). The full sequence set is
`data/sequences/bender_10282.fasta`. Output roots default to `~/bender_runs/` and can be changed with `--outdir`,
`--outroot` or the variables named in each submitter.

| Force field | Software | Reference |
|---|---|---|
| CALVADOS-2 | [CALVADOS](https://github.com/KULL-Centre/CALVADOS) in OpenMM (`prepare.py` predates release 0.9, which rejects its `runtime = 0` setting) | Tesei and Lindorff-Larsen, *Open Res. Eur.* 2023 |
| MPIPI-Recharged | OpenMpipi in OpenMM | R. Tejedor et al., *ACS Cent. Sci.* 2025 |
| MPIPI-GG | LAMMPS | Joseph et al., *Nat. Comput. Sci.* 2021; Lotthammer et al., *Nat. Methods* 2024 |

## Environment variables

| Variable | Default | Contents |
|---|---|---|
| `BENDER_DATA` | `data/` | per-sequence tables |
| `BENDER_OUT` | `output/` | figures and generated tables |
| `BENDER_TRAJ` | `trajectories/` | trajectories from Hugging Face |
| `BENDER_WORK` | `work/` | intermediate outputs of the trajectory analyses |
| `BENDER_SASBDB` | `data/saxs/sasbdb/` | experimental SAXS profiles |

## License

Code is released under the MIT License (`LICENSE`). The data in `data/` are released under CC BY 4.0, as on
Hugging Face.

## Citation

T. Rahman, Conformational Ensembles of Disordered Proteins Across the Tree of Life, submitted to *J. Chem. Theory
Comput.* (2026).
