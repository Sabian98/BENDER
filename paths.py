"""Where the BENDER code finds its inputs and writes its outputs.

Every script imports this module, so the locations below are the only ones to change. Each root can also be set
with an environment variable.

  BENDER_DATA     per-sequence tables shipped in this repository (default: ./data)
  BENDER_TRAJ     trajectories downloaded from Hugging Face, needed only to recompute the tables
                  (default: ./trajectories), laid out as
                    calvados2/300K/<taxon>/<UniProt ID>/            CALVADOS-2 (top.pdb, <ID>.dcd)
                    mpipi_recharged/300K/<UniProt ID>/              MPIPI-Recharged (top.pdb, <ID>.dcd)
                    mpipi_gg/300K/<UniProt ID>/                     MPIPI-GG (traj.xtc)
                    controls/{calvados2,recharged}/<set>/300K/<ID>/ control and swap runs
                    saxs/{calvados2,mpipi_gg,recharged}/<run>/      the 77 SAXS constructs
  BENDER_WORK     intermediate outputs of the trajectory analyses (per-frame features, contact maps, AlphaFold2
                  PAE files) that the table-building scripts in analysis/tables read (default: ./work)
  BENDER_SASBDB   experimental SAXS profiles, <SASBDB code>.dat, downloaded from https://www.sasbdb.org
                  (default: ./data/saxs/sasbdb)
  BENDER_OUT      figures and tables written by the scripts (default: ./output)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BENDER_DATA", ROOT / "data"))
TRAJ = Path(os.environ.get("BENDER_TRAJ", ROOT / "trajectories"))
SASBDB = Path(os.environ.get("BENDER_SASBDB", DATA / "saxs" / "sasbdb"))
OUT = Path(os.environ.get("BENDER_OUT", ROOT / "output"))
WORK = Path(os.environ.get("BENDER_WORK", ROOT / "work"))

ANALYSIS = ROOT / "analysis"
FIGURE_SOURCE = ROOT / "figures" / "source"
INPUTS = ROOT / "simulations" / "inputs"
CALVADOS_RESIDUES = ROOT / "simulations" / "calvados2" / "input" / "residues_CALVADOS2.csv"

# per-sequence tables
BIO = DATA / "bender_bio" / "BENDER_BIO.csv"                  # sequences, composition, CALVADOS-2 properties
GG_RESULTS = DATA / "mpipi_gg" / "mpipi_results_v2.csv"       # MPIPI-GG properties of the 957-sequence subset
RECHARGED = DATA / "mpipi_recharged"                          # MPIPI-Recharged properties, PAE agreement, B2
B2 = RECHARGED / "b2_mechanism"
PAE = DATA / "pae"                                            # CALVADOS-2 and MPIPI-GG agreement with AF2 PAE
SAXS = DATA / "saxs"                                          # SAXS benchmark (77 constructs)
CONTROLS = DATA / "controls"                                  # hnRNPA1 variants, termini controls, swaps
DISORDER = DATA / "disorder_check"                            # metapredict, pLDDT and domain checks
TAXONOMY = DATA / "taxonomy" / "ncbi_taxonomy_cache.json"     # NCBI lineages of every organism
LINEAGES = DATA / "taxonomy" / "BENDER_300K_lineages.csv"     # taxon and run folder of every sequence
SEQUENCES = DATA / "sequences"                                # all 10,282 sequences (FASTA and TSV)

# trajectories
TRAJ_C2 = TRAJ / "calvados2" / "300K"
TRAJ_RECHARGED = TRAJ / "mpipi_recharged" / "300K"
TRAJ_GG = TRAJ / "mpipi_gg" / "300K"
TRAJ_CONTROLS = TRAJ / "controls"
TRAJ_SAXS = TRAJ / "saxs"


def out(*parts):
    """A file under BENDER_OUT, creating its folder."""
    p = OUT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def outdir(*parts):
    """A folder under BENDER_OUT, created if needed."""
    p = OUT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
