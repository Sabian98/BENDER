"""Residue sets and sequence pi score behind the pi-pi and cation-pi contact columns of BENDER_BIO.csv.

The columns were first computed by dynamic_pi_score.py, which is not part of this release. Its residue sets are
restated here: aromatic residues are F, W and Y, and cation-pi pairs are Arg-aromatic pairs (Lys excluded). Only the
legacy delta_*_score columns (contact frequency minus the sequence pi score) need the score itself, and the paper uses
none of them. For BENDER sequences the score is recovered from the released table as pi_pi_contact_freq minus
delta_pi_pi_score; other sequences raise KeyError.
"""
import functools

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # analysis/
import bender_paths as paths  # noqa: E402

AROMATIC = set("FWY")
CATION_PI = set("R")


@functools.lru_cache(maxsize=1)
def _scores():
    b = pd.read_csv(paths.BIO, usecols=["sequence", "pi_pi_contact_freq", "delta_pi_pi_score"], low_memory=False)
    return dict(zip(b["sequence"], b["pi_pi_contact_freq"] - b["delta_pi_pi_score"]))


def _seq_pi_score(seq: str) -> float:
    return float(_scores()[seq])
