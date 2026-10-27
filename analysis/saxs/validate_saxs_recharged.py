#!/usr/bin/env python3
"""Validate the MPIPI-Recharged runs of the 77 SAXS benchmark constructs (added 2026-10-21).

Follows validate_saxs_constructs.py, whose helpers (nu, top_sequence, expected_frames, protein_key, R0,
TARGET_FRAMES) are imported rather than copied. The MPIPI-Recharged DCDs are written like CALVADOS-2 DCDs
(run_openmpipi_idp.py: same frame schedule, first half equilibration), so they are read with
bender_common.load_calvados: production half, 200 evenly spaced frames, unweighted C-alpha Rg. A chain is
rebuilt whole from minimum-image bonds if any bond exceeds 1 nm (recharged_io.load_run rule) and the
run is rejected unless its largest bond is below 0.5 nm (MAX_BOND_NM, as for MPIPI-GG).

SAXS condition, SAXS Rg, N, set, group and CALVADOS-2 overlap per construct are taken from traj_vs_saxs_nu.csv
and traj_vs_saxs_rg.csv (the CALVADOS-2 / MPIPI-GG validation), so all three force fields are compared against
the same experimental values. The simulated sequence must equal the CALVADOS-2 top.pdb sequence.

Usage:  python -B validate_saxs_recharged.py            (writes recharged_vs_saxs.csv + .provenance.json)
Writes only NEW files; refuses to overwrite unless --force.
"""
import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the code folders imported below

import argparse  # noqa: E402
import collections  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # analysis/
sys.path.insert(0, BR)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bender_common as bc  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)
import validate_saxs_constructs as V  # noqa: E402  (nu, top_sequence, expected_frames, protein_key, R0, TARGET_FRAMES)

BENCH = paths.SAXS
RUNS = paths.TRAJ_SAXS / "recharged"
TRAJ_DIR = V.TRAJ_DIR  # CALVADOS-2 runs (top.pdb gives the simulated sequence)
MAP_TSV = paths.ROOT / "simulations" / "mpipi_recharged" / "input" / "saxs_benchmark_300K.tsv"
OUT = BENCH / "recharged_vs_saxs.csv"
MAX_BOND_NM = V.MAX_BOND_NM   # 0.5 nm
UNWRAP_ABOVE_NM = 1.0         # recharged_io.load_run / recharged_nu.py
EQUIL_FRAC = 0.5
GROUP = {"Human": "Human", "Unassigned": "Human"}  # mapping TSV group of the two unassigned constructs


def box_nm(n):
    """Cubic box edge of run_openmpipi_idp.py (and CALVADOS prepare.py)."""
    return float(int(n / 5) + 20 if n > 300 else 50)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def analyse(traj_id, n_expected_res):
    d = RUNS / traj_id
    xyz, seq, info = bc.load_calvados(str(d / f"{traj_id}.dcd"), str(d / "top.pdb"), k=V.TARGET_FRAMES)
    if info["production_start"] != int(info["n_frames_total"] * EQUIL_FRAC):
        raise ValueError(f"{traj_id}: production starts at {info['production_start']} of {info['n_frames_total']}")
    seq_c2 = V.top_sequence(TRAJ_DIR / traj_id / "top.pdb")
    if seq != seq_c2:
        raise ValueError(f"{traj_id}: Mpipi-Recharged sequence ({len(seq)} aa) differs from CALVADOS-2 ({len(seq_c2)} aa)")
    if len(seq) != n_expected_res:
        raise ValueError(f"{traj_id}: {len(seq)} residues, benchmark lists {n_expected_res}")
    raw = bc.max_bond(xyz)
    unwrapped = raw > UNWRAP_ABOVE_NM
    if unwrapped:
        xyz = bc.unwrap_chain(xyz, box_nm(len(seq)))
    mb = bc.max_bond(xyz)
    if mb >= MAX_BOND_NM:
        raise ValueError(f"{traj_id}: chain broken (max bond {mb:.3f} nm after unwrapping)")
    g = bc.geometry(xyz)
    rees = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=1)
    return {"recharged_n_frames_total": info["n_frames_total"],
            "recharged_n_frames_expected": V.expected_frames(len(seq)),
            "recharged_production_start": info["production_start"],
            "recharged_n_frames_used": info["n_frames_used"],
            "recharged_max_bond_raw_nm": round(raw, 3), "recharged_unwrapped": bool(unwrapped),
            "recharged_max_bond_nm": round(mb, 3),
            "recharged_rg_nm": round(g["rg"], 4), "recharged_rg_sem_nm": round(g["rg_err"], 4),
            "recharged_ree_nm": round(g["ree"], 4), "recharged_ree_std_nm": round(float(rees.std()), 4)}


def stats(d, rg_col, dnu_col):
    rs, rx, dnu, n = (d[c].to_numpy(float) for c in (rg_col, "saxs_rg_nm", dnu_col, "n_res"))
    fit = lambda rg: np.polyfit(np.log(n), np.log(rg), 1)[0]  # noqa: E731
    return {"n": len(d), "proteins": len({V.protein_key(r) for r in d.fillna("").to_dict("records")}),
            "mare_pct": 100 * np.mean(np.abs(rs - rx) / rx), "r_rg": np.corrcoef(rs, rx)[0, 1],
            "within_005_pct": 100 * np.mean(np.abs(dnu) < 0.05), "within_010_pct": 100 * np.mean(np.abs(dnu) < 0.10),
            "mean_dnu": dnu.mean(), "sd_dnu": dnu.std(ddof=1) if len(d) > 1 else float("nan"),
            "nu_fit_sim": fit(rs) if len(d) > 1 else float("nan"), "nu_fit_saxs": fit(rx) if len(d) > 1 else float("nan")}


FFS = (("CALVADOS-2", "traj_rg_nm", "delta_nu"), ("MPIPI-GG", "mpipi_rg_nm", "delta_nu_mpipi"),
       ("MPIPI-Recharged", "recharged_rg_nm", "delta_nu_recharged"))


def report(d):
    lines = []
    p = lambda s="": lines.append(s)  # noqa: E731
    primary = d["set"] == "strict"
    # tab:saxs_subsets: "independent" = primary with no overlap at all (validate_saxs_constructs "STRICT never seen",
    # 53); "overlapping training data" = any set, training or training_overlap ("training overlap only", 10)
    independent = primary & (d["calvados2_overlap"] == "none")
    training = d["calvados2_overlap"].isin(V.SEEN_IN_TRAINING)
    subsets = [("All constructs", d), ("Primary set", d[primary]),
               ("Primary, independent of CALVADOS-2", d[independent]),
               ("Overlapping CALVADOS-2 training data", d[training]),
               ("No UniProt mapping", d[~primary]),
               ("Primary, simulated before rules were fixed", d[primary & (d["source"] == "existing")]),
               ("Primary, simulated after rules were fixed", d[primary & (d["source"] == "new")]),
               ("Viral, primary", d[primary & (d["taxon"] == "Virus")]),
               ("Viral, all", d[d["taxon"] == "Virus"]),
               ("Non-viral, all", d[d["taxon"] != "Virus"]),
               ("Non-viral, primary", d[primary & (d["taxon"] != "Virus")])]
    p("Subset summaries (MARE %, r(Rg), |dnu|<0.05 %, <0.10 %, mean dnu ± SD (ddof 1), nu fit sim / SAXS)")
    for name, s in subsets:
        p(f"\n{name}: n = {len(s)}")
        for ff, rg, dn in FFS:
            x = stats(s, rg, dn)
            p(f"  {ff:16s} proteins {x['proteins']:3d} | MARE {x['mare_pct']:5.2f} | r {x['r_rg']:.3f} | "
              f"<0.05 {x['within_005_pct']:5.1f} | <0.10 {x['within_010_pct']:5.1f} | "
              f"mean {x['mean_dnu']:+.4f} ± {x['sd_dnu']:.4f} | fit {x['nu_fit_sim']:.3f} / {x['nu_fit_saxs']:.3f}")
    p("\nPrimary set by group (n, proteins | MARE %, mean dnu for C2 | GG | MR)")
    pr = d[primary]
    for t, cnt in sorted(collections.Counter(pr["taxon"]).items(), key=lambda kv: -kv[1]):
        s = pr[pr["taxon"] == t]
        xs = [stats(s, rg, dn) for _, rg, dn in FFS]
        p(f"  {t:17s} n={len(s):3d} prot={xs[0]['proteins']:3d} | " +
          " | ".join(f"{x['mare_pct']:5.1f} {x['mean_dnu']:+.4f}" for x in xs))
    a = {ff: d[dn].to_numpy(float) for ff, _, dn in FFS}
    p("\nPer-construct correlations of dnu (all 77)")
    for f1, f2 in (("MPIPI-Recharged", "CALVADOS-2"), ("MPIPI-Recharged", "MPIPI-GG"), ("MPIPI-GG", "CALVADOS-2")):
        p(f"  r(dnu {f1}, dnu {f2}) = {np.corrcoef(a[f1], a[f2])[0, 1]:.3f}")
    for f1, f2 in (("MPIPI-Recharged", "CALVADOS-2"), ("MPIPI-Recharged", "MPIPI-GG")):
        rgc = dict((ff, rg) for ff, rg, _ in FFS)
        p(f"  r(Rg {f1}, Rg {f2}) = {np.corrcoef(d[rgc[f1]], d[rgc[f2]])[0, 1]:.3f}")
    p("\nCloser to SAXS (|dnu| strictly smaller)")
    absd = np.vstack([np.abs(a[ff]) for ff, _, _ in FFS])
    for f1, f2 in (("MPIPI-Recharged", "CALVADOS-2"), ("MPIPI-Recharged", "MPIPI-GG"), ("MPIPI-GG", "CALVADOS-2")):
        p(f"  {f1} closer than {f2}: {100 * np.mean(np.abs(a[f1]) < np.abs(a[f2])):.1f}%")
    best = absd.argmin(axis=0)
    ties = int(np.sum((absd == absd.min(axis=0)).sum(axis=0) > 1))
    for k, (ff, _, _) in enumerate(FFS):
        p(f"  closest of three: {ff} {100 * np.mean(best == k):.1f}% ({int(np.sum(best == k))})")
    p(f"  ties for closest: {ties}")
    vir = d["taxon"].to_numpy() == "Virus"
    p(f"  viral constructs (all {vir.sum()}): closest " +
      ", ".join(f"{ff} {int(np.sum(best[vir] == k))}" for k, (ff, _, _) in enumerate(FFS)))
    pv = d[primary & (d["taxon"] == "Virus")]
    pn = d[primary & (d["taxon"] != "Virus")]
    p("\nViral constructs, primary (13): median dnu, mean dnu without the N protein linker (SASDSZ9), "
      "mean dnu relative to the non-viral primary mean of the same force field")
    for ff, _, dn in FFS:
        p(f"  {ff:16s} median {pv[dn].median():+.4f} | mean w/o N linker {pv[pv['traj_id'] != 'SASDSZ9'][dn].mean():+.4f} | "
          f"mean w/o N linker and V protein O55777 {pv[~pv['traj_id'].isin(['SASDSZ9', 'O55777'])][dn].mean():+.4f} | "
          f"viral - non-viral {pv[dn].mean() - pn[dn].mean():+.4f} | non-viral mean {pn[dn].mean():+.4f}")
    absv = np.vstack([np.abs(pv[dn].to_numpy(float)) for _, _, dn in FFS])
    bv = absv.argmin(axis=0)
    p("  closest of three (13 primary viral): " + ", ".join(f"{ff} {int(np.sum(bv == k))}" for k, (ff, _, _) in enumerate(FFS)))
    p(f"  Mpipi-Recharged closer than CALVADOS-2 on {int(np.sum(absv[2] < absv[0]))} of 13")
    absp = np.vstack([np.abs(d[primary][dn].to_numpy(float)) for _, _, dn in FFS])
    bp = absp.argmin(axis=0)
    p("Primary set (63), closest of three: " + ", ".join(f"{ff} {100 * np.mean(bp == k):.1f}% ({int(np.sum(bp == k))})"
                                                       for k, (ff, _, _) in enumerate(FFS)))
    for f1, f2 in (("MPIPI-Recharged", "CALVADOS-2"), ("MPIPI-Recharged", "MPIPI-GG")):
        k1, k2 = [dict((ff, dn) for ff, _, dn in FFS)[f] for f in (f1, f2)]
        p(f"Primary set r(dnu {f1}, dnu {f2}) = {np.corrcoef(d[primary][k1], d[primary][k2])[0, 1]:.3f}")
    p("\nConstructs with |dnu| >= 0.05 in any force field (dnu C2 / GG / MR; Rg SAXS, C2, GG, MR)")
    big = d[(np.abs(d[["delta_nu", "delta_nu_mpipi", "delta_nu_recharged"]]) >= 0.05).any(axis=1)]
    for _, r in big.sort_values("delta_nu_recharged").iterrows():
        p(f"  {r['traj_id']:10s} {str(r['protein_name'])[:34]:34s} {r['taxon']:16s} {r['set']:14s} N={r['n_res']:4d} "
          f"dnu {r['delta_nu']:+.3f} / {r['delta_nu_mpipi']:+.3f} / {r['delta_nu_recharged']:+.3f} | "
          f"Rg {r['saxs_rg_nm']:.2f} {r['traj_rg_nm']:.2f} {r['mpipi_rg_nm']:.2f} {r['recharged_rg_nm']:.2f} T={r['saxs_temperature_c']}")
    p("\nViral constructs (dnu C2 / GG / MR)")
    for _, r in d[vir].sort_values("delta_nu").iterrows():
        p(f"  {r['traj_id']:10s} {str(r['protein_name'])[:40]:40s} {r['set']:14s} N={r['n_res']:4d} "
          f"{r['delta_nu']:+.3f} / {r['delta_nu_mpipi']:+.3f} / {r['delta_nu_recharged']:+.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite recharged_vs_saxs.csv")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        sys.exit(f"{OUT} exists; pass --force to overwrite")

    nu = pd.read_csv(BENCH / "traj_vs_saxs_nu.csv", keep_default_na=False, na_values=[""])
    rg = pd.read_csv(BENCH / "traj_vs_saxs_rg.csv", keep_default_na=False, na_values=[""])
    extra = [c for c in ("saxs_rg_err_nm", "traj_rg_sem_nm", "mpipi_rg_sem_nm") if c in rg]
    d = nu.merge(rg[["traj_id"] + extra], on="traj_id", how="left", validate="one_to_one")
    assert len(d) == 77 and d["flag"].isna().all(), "expected 77 unflagged constructs"
    m = pd.read_csv(MAP_TSV, sep="\t", dtype=str)
    assert set(m["traj_id"]) == set(d["traj_id"]), "run mapping and benchmark constructs differ"
    d = d.merge(m[["traj_id", "group", "sasbdb_id"]].rename(columns={"group": "run_group", "sasbdb_id": "run_sasbdb_id"}),
                on="traj_id", validate="one_to_one")
    assert (d["sasbdb_id"] == d["run_sasbdb_id"]).all(), "SASBDB id differs between run mapping and benchmark"

    rows = []
    for i, r in enumerate(d.itertuples(index=False), 1):
        res = analyse(r.traj_id, int(r.n_res))
        nu_saxs = V.nu(r.saxs_rg_nm, r.n_res)
        nu_mr = V.nu(res["recharged_rg_nm"], r.n_res)
        res.update({"traj_id": r.traj_id, "nu_saxs": round(nu_saxs, 4), "nu_recharged": round(nu_mr, 4),
                    "delta_nu_recharged": round(nu_mr - nu_saxs, 4),
                    "ratio_recharged_saxs": round(res["recharged_rg_nm"] / r.saxs_rg_nm, 4),
                    "abs_diff_recharged_nm": round(abs(res["recharged_rg_nm"] - r.saxs_rg_nm), 4)})
        rows.append(res)
        print(f"  {i}/{len(d)} {r.traj_id} N={r.n_res} Rg {res['recharged_rg_nm']:.3f} (SAXS {r.saxs_rg_nm}) "
              f"dnu {res['delta_nu_recharged']:+.4f} frames {res['recharged_n_frames_total']}/{res['recharged_n_frames_expected']} "
              f"bond {res['recharged_max_bond_nm']}", flush=True)
    mr = pd.DataFrame(rows)
    d = d.drop(columns=["nu_saxs"]).merge(mr, on="traj_id", validate="one_to_one")
    d["nu_saxs_check"] = [round(V.nu(a, b), 4) for a, b in zip(d["saxs_rg_nm"], d["n_res"])]

    absd = np.abs(d[["delta_nu", "delta_nu_mpipi", "delta_nu_recharged"]].to_numpy(float))
    d["closest_to_saxs"] = np.array(["CALVADOS-2", "MPIPI-GG", "MPIPI-Recharged"])[absd.argmin(axis=1)]

    cols = ["traj_id", "uniprot_id", "protein_name", "taxon", "run_group", "set", "source", "calvados2_overlap",
            "sasbdb_id", "saxs_condition_id", "saxs_temperature_c", "n_res",
            "recharged_n_frames_total", "recharged_n_frames_expected", "recharged_production_start",
            "recharged_n_frames_used", "recharged_max_bond_raw_nm", "recharged_unwrapped", "recharged_max_bond_nm",
            "saxs_rg_nm", "saxs_rg_err_nm", "recharged_rg_nm", "recharged_rg_sem_nm", "recharged_ree_nm",
            "recharged_ree_std_nm", "ratio_recharged_saxs", "abs_diff_recharged_nm",
            "nu_saxs", "nu_recharged", "delta_nu_recharged",
            "traj_rg_nm", "nu_traj", "delta_nu", "mpipi_rg_nm", "nu_mpipi", "delta_nu_mpipi", "closest_to_saxs"]
    d = d.drop(columns=["nu_saxs"]).rename(columns={"nu_saxs_check": "nu_saxs"})   # one nu_saxs column (identical values)
    d[cols].to_csv(OUT, index=False)

    text = report(d)
    print("\n" + text)
    (BENCH / "recharged_vs_saxs_summary.txt").write_text(text + "\n")
    short = d[d["recharged_n_frames_total"] != d["recharged_n_frames_expected"]]["traj_id"].tolist()
    prov = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "host": platform.node(),
            "inputs": {"runs": str(RUNS), "calvados2_topologies": str(TRAJ_DIR), "run_mapping": str(MAP_TSV),
                       "traj_vs_saxs_nu.csv": sha256(BENCH / "traj_vs_saxs_nu.csv"),
                       "traj_vs_saxs_rg.csv": sha256(BENCH / "traj_vs_saxs_rg.csv"),
                       "bender_common.py": sha256(Path(BR) / "bender_common.py"),
                       "validate_saxs_constructs.py": sha256(HERE / "validate_saxs_constructs.py")},
            "method": {"frames": "bender_common.load_calvados: frames[n//2:], 200 evenly spaced",
                       "whole_chain": f"unwrap_chain with box (50 nm, or int(N/5)+20 above 300) if raw max bond > "
                                      f"{UNWRAP_ABOVE_NM} nm; rejected unless max bond < {MAX_BOND_NM} nm",
                       "rg": "bender_common.geometry: unweighted C-alpha Rg, frame mean; SE = SD/sqrt(frames)",
                       "nu": f"nu_eff = ln(Rg/{V.R0})/ln N; delta_nu = nu_sim - nu_SAXS (rounded to 4 d.p. as in "
                             "traj_vs_saxs_nu.csv)",
                       "saxs": "saxs_rg_nm, saxs_condition_id, n_res, set, taxon, calvados2_overlap from "
                               "traj_vs_saxs_nu.csv (same SAXS condition as CALVADOS-2 and MPIPI-GG)",
                       "sequence": "MPIPI-Recharged top.pdb sequence equals the CALVADOS-2 top.pdb sequence (asserted)",
                       "closest_to_saxs": "force field with the smallest |delta_nu| of the three"},
            "n_constructs": int(len(d)), "frames_not_expected": short,
            "max_bond_nm_overall": float(d["recharged_max_bond_nm"].max()),
            "n_unwrapped": int(d["recharged_unwrapped"].sum()),
            "outputs": [OUT.name, "recharged_vs_saxs_summary.txt"]}
    (BENCH / "recharged_vs_saxs.provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"\nwrote {OUT} ({len(d)} rows); frames not as expected: {short or 'none'}; "
          f"max bond {prov['max_bond_nm_overall']} nm; unwrapped {prov['n_unwrapped']}")


if __name__ == "__main__":
    main()
