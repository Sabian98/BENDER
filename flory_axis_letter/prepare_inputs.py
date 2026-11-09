"""Build the per-sequence input table of the Flory-axis letter from the tables in data/.

The letter compares CALVADOS-2 and MPIPI-Recharged on the same sequences. This script joins the CALVADOS-2 table
(data/bender_bio/BENDER_BIO.csv, where nu_orig and rg_orig are the CALVADOS-2 values) with the MPIPI-Recharged
features (data/mpipi_recharged/recharged_300K_features.csv, where nu and rg are the MPIPI-Recharged values) and marks
the sequences whose Flory fit reaches R^2 >= 0.99 in each force field (nu_ok_cal, nu_ok_rech).

Usage: python prepare_inputs.py            writes <BENDER_OUT>/flory_axis_letter/bio.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, OUT  # noqa: E402

R2_MIN = 0.99


def main():
    cal = pd.read_csv(DATA / "bender_bio" / "BENDER_BIO.csv", low_memory=False)
    rch = pd.read_csv(DATA / "mpipi_recharged" / "recharged_300K_features.csv", low_memory=False)
    # Ensemble columns that both tables share (nu, rg, A0, contact-network properties) take the MPIPI-Recharged
    # values; the CALVADOS-2 values stay in the *_orig columns and nu_fit_r2.
    cal = cal.drop(columns=[c for c in rch.columns if c in cal.columns and c != "UniProt_ID"])
    bio = cal.merge(rch, on="UniProt_ID", how="left", validate="one_to_one")
    bio = bio.assign(nu_ok_cal=bio["nu_fit_r2"] >= R2_MIN, nu_ok_rech=bio["nu_r2"] >= R2_MIN,
                     label_source="Mpipi-Recharged 300K")
    out = OUT / "flory_axis_letter"
    out.mkdir(parents=True, exist_ok=True)
    bio.to_csv(out / "bio.csv", index=False)
    both = (bio.nu_ok_cal & bio.nu_ok_rech).sum()
    print(f"wrote {out / 'bio.csv'}: {len(bio):,} sequences, {both:,} pass the fit in both force fields")


if __name__ == "__main__":
    main()
