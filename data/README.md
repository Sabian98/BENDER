# Per-sequence data

All tables are keyed by UniProt accession. Ensembles are at 300 K and 150 mM ionic strength, and every property is
computed on the production half of each trajectory. ν is the apparent Flory scaling exponent, Rg the radius of
gyration, Ree the end-to-end distance and Δν the MPIPI-Recharged value minus the CALVADOS-2 value.

| Path | Rows | Contents |
|---|---|---|
| `bender_bio/BENDER_BIO.csv` | 10,282 | sequence, organism, NCBI taxonomy ID and group (`kingdom`), composition and patterning descriptors (fractions, κ, SCD, SHD, λ), and the CALVADOS-2 ensemble properties (ν, Rg, Ree, asphericity, contact-network properties, π–π and cation–π contact frequencies) |
| `mpipi_recharged/recharged_300K_nu.csv` | 10,282 | MPIPI-Recharged ν and its fit statistics, split-half ν and mean Rg |
| `mpipi_recharged/recharged_300K_features.csv` | 10,282 | MPIPI-Recharged Rg, Ree, asphericity, ν, contact-network properties, contact frequencies and PAE agreement |
| `mpipi_recharged/recharged_300K_vs_pae_results_v2.tsv` | 10,282 | MPIPI-Recharged agreement with AlphaFold2 PAE |
| `mpipi_recharged/b2_mechanism/` | | B2 of all 210 residue pairs in both force fields, pair-class summaries, per-sequence mean ΔB2 and the B2 regression models |
| `mpipi_gg/mpipi_results_v2.csv` | 1,307 | MPIPI-GG properties; `in_paper_set` marks the 957 sequences of BENDER |
| `pae/` | | CALVADOS-2 and MPIPI-GG agreement with AlphaFold2 PAE (Pearson r between inter-residue distance fluctuations and PAE) |
| `disorder_check/` | 10,282 | metapredict V2 and V3 disorder, AlphaFold2 pLDDT, folded-domain fraction and UniProt status of every sequence (`bender_disorder_check.csv`), with the per-residue scores and retrieved records behind it |
| `saxs/` | 77 | SAXS benchmark: simulated against measured Rg and ν for each construct in the three force fields, overlap with CALVADOS-2 training data, and the simulated scattering profiles (`figures/*_iq_cache.json`) |
| `controls/` | | ν and Rg of the control and parameter-swap runs (hnRNPA1 variants, termini and histidine controls, swaps), one file per force field and set |
| `sequences/` | 10,282 | all sequences as FASTA and TSV |
| `taxonomy/` | | NCBI lineages of every organism and the group assignment of every sequence |

The `delta_pi_pi_score` and `delta_cation_pi_score` columns are legacy columns that the paper does not use.

These data are released under CC BY 4.0.
