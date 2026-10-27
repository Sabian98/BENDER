"""metapredict disorder check of the 10,282 BENDER chains (2026-10-21).

Runs metapredict (Emenecker et al.; package 3.1.1, legacy network version='V2' as requested, and the current V3 as a
sensitivity check) on every BENDER sequence with metapredict's own IDR calling: the version's default disorder
threshold, IDRs of at least 12 residues (X, in 3 chains, read as G), folded domains of at least 50 residues, gaps of up to 10 residues closed.
A chain is fully disordered when metapredict returns one IDR covering the whole sequence.

Needs the `metapredict` conda env (PyTorch):  ~/miniforge3/envs/metapredict/bin/python run_metapredict.py
Writes data/disorder_check/metapredict.csv and metapredict_scores_{V2,V3}.npz (per-residue scores).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import metapredict as meta
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

HERE = Path(__file__).resolve().parent
DATA = paths.DISORDER
FASTA = DATA / "bender_10282.fasta"


def read_fasta(path):
    ids, seqs, groups = [], [], []
    for block in path.read_text().split(">")[1:]:
        head, *body = block.strip().splitlines()
        acc, group = head.split()[:2]
        ids.append(acc)
        groups.append(group)
        seqs.append("".join(body))
    return ids, groups, seqs


def main():
    ids, groups, seqs = read_fasta(FASTA)
    assert len(ids) == 10282
    print("X -> G in", sum("X" in q for q in seqs), "sequences (as in BENDER_BIO feature tables)")
    seqs = [q.replace("X", "G") for q in seqs]
    out = pd.DataFrame({"acc": ids, "group": groups, "n": [len(s) for s in seqs]}).set_index("acc")
    for version in ("V2", "V3"):
        res = meta.predict_disorder(seqs, version=version, return_domains=True, show_progress_bar=False)
        scores, rows = {}, []
        for acc, seq, r in zip(ids, seqs, res):
            sc = np.asarray(r.disorder, float)
            idr = [(int(a), int(b)) for a, b in r.disordered_domain_boundaries]
            fold = [int(b) - int(a) for a, b in r.folded_domain_boundaries]
            in_idr = sum(b - a for a, b in idr)
            rows.append({f"mean_{version}": sc.mean(), f"frac_idr_{version}": in_idr / len(seq),
                         f"n_idr_{version}": len(idr), f"full_{version}": len(idr) == 1 and idr[0] == (0, len(seq)),
                         f"longest_folded_{version}": max(fold, default=0)})
            scores[acc] = sc.astype(np.float32)
        thr = res[0].disorder_threshold if hasattr(res[0], "disorder_threshold") else None
        out = out.join(pd.DataFrame(rows, index=ids))
        np.savez_compressed(DATA / f"metapredict_scores_{version}.npz", **scores)
        f = out[f"full_{version}"]
        print(f"{version} (threshold {thr}): fully disordered {f.sum():,} ({100 * f.mean():.1f}%); "
              f">= 90% in IDRs {100 * (out[f'frac_idr_{version}'] >= 0.9).mean():.1f}%; "
              f"residues in IDRs {100 * np.average(out[f'frac_idr_{version}'], weights=out.n):.1f}%; "
              f"no IDR {100 * (out[f'n_idr_{version}'] == 0).mean():.1f}%", flush=True)
    out.to_csv(DATA / "metapredict.csv")
    print(out.groupby("group")[["full_V2", "full_V3"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
