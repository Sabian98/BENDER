"""Per-sequence property table for the six-lineage distribution figures.

Everything comes from data/bender_bio/BENDER_BIO.csv (the 10,282 BENDER sequences
of the six groups: reanalysed 300 K ensembles, primary nu, contact graphs with
|i-j| >= 3, and sequence descriptors recomputed with one definition each).
The group is the `kingdom` column (Mammals, Fungi, Plants, Protists, Bacteria,
Viruses). FCR and the mean CALVADOS-2 stickiness lambda are checked against the
sequence; an unknown residue X (three of the added protists) counts as glycine,
as it does in BENDER_BIO (lambda_avg, f_G). The analysis_ok flag (complete
trajectory and nu fit R^2 >= 0.99) is carried through for reference; the
figures use every sequence.

Output: lineage_properties.csv (next to this script).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

HERE = Path(__file__).resolve().parent
BIO = paths.BIO
RESIDUES = paths.CALVADOS_RESIDUES
SIX = ["Mammals", "Fungi", "Plants", "Protists", "Bacteria", "Viruses"]
ENSEMBLE = ["rg", "ree", "delta", "nu", "A0", "global_efficiency", "fragmentation_index",
            "avg_clustering", "transitivity", "degree_assortativity"]

LAMBDA = pd.read_csv(RESIDUES).set_index("one")["lambdas"].to_dict()
LAMBDA["X"] = LAMBDA["G"]


def main():
    df = pd.read_csv(BIO, low_memory=False)
    df = df[df["kingdom"].isin(SIX)].copy()           # the six BENDER groups
    df["uniprot_id"], df["lineage"] = df["UniProt_ID"], df["kingdom"]

    seq = df["sequence"]
    assert (seq.str.len() == df["length"]).all() and (df["length"] == df["n_residues"]).all(), \
        "sequence length != simulated residues"
    fcr = seq.map(lambda s: sum(s.count(a) for a in "KRDE") / len(s))
    print(f"FCR: max |(K+R+D+E)/N - BENDER_BIO| = {(fcr - df['FCR']).abs().max():.2e}")
    assert (fcr - df["FCR"]).abs().max() < 1e-9
    lam = seq.map(lambda s: np.mean([LAMBDA[a] for a in s]))
    assert (lam - df["lambda_avg"]).abs().max() < 1e-9, "stored lambda_avg is not CALVADOS-2"

    keep = ["uniprot_id", "lineage", "length", "FCR", "lambda_avg"] + ENSEMBLE + ["analysis_ok"]
    df[keep].to_csv(HERE / "lineage_properties.csv", index=False)
    print("wrote lineage_properties.csv:", len(df), "sequences,", int(df["analysis_ok"].sum()), "analysis_ok")
    print(df.groupby("lineage")["analysis_ok"].agg(["size", "sum"]).reindex(SIX).to_string())


if __name__ == "__main__":
    main()
