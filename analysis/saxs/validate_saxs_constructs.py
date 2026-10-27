#!/usr/bin/env python3
"""Validate CALVADOS-2 and MPIPI-GG trajectories against construct-matched IDP SASBDB conditions (Rg and nu_eff).

Frame handling and geometry come from bender_common (the BENDER reanalysis module), so both force
fields follow the BENDER conventions: CALVADOS-2 DCDs drop the first half as equilibration; MPIPI-GG
traj.xtc files are production only and are unwrapped from minimum-image bond vectors; 200 evenly
spaced frames; unweighted C-alpha Rg.
"""
import collections, csv, difflib, math, re, statistics, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)
import bender_common as bc  # noqa: E402

EXP = paths.SAXS                      # processed/ holds the SASBDB entry and condition tables
BENCH = paths.SAXS
TRAJ_DIR = paths.TRAJ_SAXS / "calvados2"
MPIPI_DIR = paths.TRAJ_SAXS / "mpipi_gg"
MAX_BOND_NM = 0.5            # an unwrapped MPIPI-GG chain must have no bond longer than this
SKIP_DIRS = {"data", "zscore_validation"}
OVERLAP_CSV = BENCH / "calvados2_overlap.csv"
SEEN_IN_TRAINING = {"training", "training_overlap"}
R0 = 0.1963
B0_REE = 0.38
TARGET_FRAMES = 200
SIM_T_C = 26.85
NU_MONOMER_MAX = 0.7
MIN_IDENTITY = 0.95
ACCEPT = {"IDP": "strict", "idp_kratky_only": "no_uniprot_map", "idp_kratky_only_unaligned": "no_uniprot_map"}
REASON = {"mostly_disordered": "mostly disordered (60-80% of construct)",
          "no_structure_evidence": "no AlphaFold/DisProt support for disorder",
          "not_idp_sequence": "folded or multidomain construct",
          "not_idp_kratky": "compact or aggregated Kratky profile"}
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
         "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
         "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def nu(rg, n):
    return math.log(rg / R0) / math.log(n)


def top_sequence(top):
    seq, seen = [], set()
    for line in open(top):
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[22:27])
            if key not in seen:
                seen.add(key)
                seq.append(THREE.get(line[17:20].strip(), "X"))
    return "".join(seq)


def expected_frames(n):
    t = 200 if n <= 150 else int(200 * (n / 150) ** 1.5)
    return 10000 if t <= 300 else (15000 if t <= 600 else 20000)


def traj_rg(dcd, top):
    """CALVADOS-2: production half, 200 evenly spaced frames (bender_common)."""
    xyz, _, info = bc.load_calvados(str(dcd), str(top), k=TARGET_FRAMES)
    g = bc.geometry(xyz)
    rees = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=1)
    return (g["rg"], g["rg_err"], g["ree"], float(rees.std()), info["n_frames_total"], info["n_frames_used"])


def mpipi_expected_frames(run_in):
    """Frames the LAMMPS input should dump: (run - dump delay) / dump interval + 1."""
    txt = Path(run_in).read_text()
    run = int(re.search(r"^run\s+(\d+)", txt, re.M).group(1))
    every = int(re.search(r"^dump\s+\S+\s+all\s+xtc\s+(\d+)", txt, re.M).group(1))
    delay = int(re.search(r"^dump_modify\s+\S+\s+delay\s+(\d+)", txt, re.M).group(1))
    return (run - delay) // every + 1


def mpipi_rg(name, seq):
    """MPIPI-GG run of the same construct: all frames are production; unwrap, 200 evenly spaced frames.
    Returns (result dict or None, problem string)."""
    d = MPIPI_DIR / name
    if not ((d / "seq.dat").exists() and (d / "traj.xtc").exists()):
        return None, "missing MPIPI-GG run"
    seq_m = bc.read_mpipi_sequence(str(d / "seq.dat"))
    if seq_m != seq:
        return None, f"MPIPI-GG sequence differs from CALVADOS-2 ({len(seq_m)} vs {len(seq)} residues)"
    xyz, info = bc.load_mpipi(str(d / "traj.xtc"), n_res=len(seq), k=TARGET_FRAMES)
    if info["max_bond_nm"] > MAX_BOND_NM:
        return None, f"MPIPI-GG chain broken after unwrapping (max bond {info['max_bond_nm']:.2f} nm)"
    g = bc.geometry(xyz)
    rees = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=1)
    return {"mpipi_rg_nm": g["rg"], "mpipi_rg_sem_nm": g["rg_err"], "mpipi_ree_nm": g["ree"],
            "mpipi_ree_std_nm": float(rees.std()), "mpipi_n_frames_total": info["n_frames_total"],
            "mpipi_n_frames_expected": mpipi_expected_frames(d / "run.in"),
            "mpipi_n_frames_used": info["n_frames_used"], "mpipi_max_bond_nm": round(info["max_bond_nm"], 3)}, ""


def identity(a, b):
    if a == b:
        return 1.0
    if min(len(a), len(b)) / max(len(a), len(b)) < 0.85:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def protein_key(row):
    if row["uniprot_id"]:
        return row["uniprot_id"]
    n = row["protein_name"].lower()
    for k in ("synuclein", "neurofilament", "elastin-like", "low complexity domain", "huntingtin", "protein w", "protocadherin"):
        if k in n:
            return k
    return n


FF_KEYS = {"CALVADOS-2": ("traj_rg_nm", "delta_nu"), "MPIPI-GG": ("mpipi_rg_nm", "delta_nu_mpipi")}


def summarize(rows, label, ff="CALVADOS-2"):
    rg_key, dnu_key = FF_KEYS[ff]
    rows = [r for r in rows if r[rg_key] != ""]
    if len(rows) < 2:
        return
    rs, rx = np.array([r[rg_key] for r in rows]), np.array([r["saxs_rg_nm"] for r in rows])
    dnu = np.array([r[dnu_key] for r in rows])
    n = np.array([r["n_res"] for r in rows])
    fit = lambda rg: np.polyfit(np.log(n), np.log(rg), 1)[0]
    print(f"{label:20s} n={len(rows):3d} proteins={len({protein_key(r) for r in rows}):3d} | "
          f"MARE {100 * np.mean(np.abs(rs - rx) / rx):4.1f}% | r(Rg) {np.corrcoef(rs, rx)[0, 1]:.3f} | "
          f"|dnu|<0.05 {np.mean(np.abs(dnu) < 0.05):4.0%} <0.10 {np.mean(np.abs(dnu) < 0.10):4.0%} | "
          f"mean dnu {dnu.mean():+.4f} ± {dnu.std(ddof=1):.4f} | fit nu sim {fit(rs):.3f} saxs {fit(rx):.3f}")


def main():
    conds = list(csv.DictReader(open(EXP / "processed/sasbdb_conditions.csv")))
    entries = {r["sasbdb_id"]: r for r in csv.DictReader(open(EXP / "processed/sasbdb_entries_screened.csv"))}
    names = sorted(p.name for p in TRAJ_DIR.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in SKIP_DIRS)
    overlap = {r["traj_id"]: r["calvados2_overlap"] for r in csv.DictReader(open(OVERLAP_CSV))} if OVERLAP_CSV.exists() else {}

    rows, excluded, mpipi_problems = [], [], []
    for i, name in enumerate(names, 1):
        d = TRAJ_DIR / name
        dcd, top = d / f"{name}.dcd", d / "top.pdb"
        if not (dcd.exists() and top.exists()):
            excluded.append({"traj_id": name, "reason": "missing dcd or top.pdb", "detail": ""})
            continue
        seq = top_sequence(top)
        matched = [(c, identity(seq, c["sequence"])) for c in conds]
        matched = [(c, s) for c, s in matched if s >= MIN_IDENTITY]
        ok = [(c, s) for c, s in matched if c["idp_call"] in ACCEPT]
        if not ok:
            best = min(matched, key=lambda cs: list(REASON).index(cs[0]["idp_call"]) if cs[0]["idp_call"] in REASON else 99, default=None)
            excluded.append({"traj_id": name,
                             "reason": REASON.get(best[0]["idp_call"], best[0]["idp_call"]) if best else "no matching SAXS construct",
                             "detail": best[0]["condition_id"] if best else ""})
            continue
        c, ident = min(ok, key=lambda cs: (cs[1] < 0.999,
                                           abs(float(cs[0]["temperature_c"]) - SIM_T_C) if cs[0]["temperature_c"] not in ("", "None") else 99,
                                           -cs[1], -int(cs[0]["n_entries"]), cs[0]["condition_id"]))
        n = len(seq)
        rg_saxs = float(c["rg_median_nm"])
        nu_saxs = nu(rg_saxs, n)
        if nu_saxs > NU_MONOMER_MAX:
            excluded.append({"traj_id": name, "reason": f"non-monomeric SAXS (nu_saxs={nu_saxs:.2f})", "detail": c["condition_id"]})
            continue
        rg_sim, rg_sem, ree, ree_std, n_total, n_used = traj_rg(dcd, top)
        idp_entries = [entries[e] for e in c["entries"].split(";") if entries[e].get("tier") in ("disordered", "disordered_by_nu")]
        errs = [float(e["rg_err_nm"]) for e in idp_entries if e.get("rg_err_nm") not in ("", None)]
        nu_sim = nu(rg_sim, n)
        mp, problem = mpipi_rg(name, seq)
        if mp is None:
            mpipi_problems.append((name, problem))
            mp = {k: "" for k in ("mpipi_rg_nm", "mpipi_rg_sem_nm", "mpipi_ree_nm", "mpipi_ree_std_nm", "mpipi_n_frames_total",
                                  "mpipi_n_frames_expected", "mpipi_n_frames_used", "mpipi_max_bond_nm")}
            mp.update({"nu_mpipi": "", "delta_nu_mpipi": "", "ratio_mpipi_saxs": "", "mpipi_ree_rg_ratio": ""})
        else:
            nu_mp = nu(mp["mpipi_rg_nm"], n)
            mp.update({"nu_mpipi": round(nu_mp, 4), "delta_nu_mpipi": round(nu_mp - nu_saxs, 4),
                       "ratio_mpipi_saxs": round(mp["mpipi_rg_nm"] / rg_saxs, 4),
                       "mpipi_ree_rg_ratio": round(mp["mpipi_ree_nm"] / mp["mpipi_rg_nm"], 4)})
            for k in ("mpipi_rg_nm", "mpipi_rg_sem_nm", "mpipi_ree_nm", "mpipi_ree_std_nm"):
                mp[k] = round(mp[k], 4)
        rows.append({**mp,
            "traj_id": name, "uniprot_id": c["uniprot_id"], "protein_name": c["protein_name"][:60], "taxon": c["taxon"],
            "set": ACCEPT[c["idp_call"]], "source": "new" if name.startswith("SASD") else "existing",
            "calvados2_overlap": overlap.get(name, "none"),
            "sasbdb_id": c["construct_id"], "saxs_condition_id": c["condition_id"], "saxs_entries": c["entries"],
            "saxs_n_entries": int(c["n_entries"]), "saxs_temperature_c": c["temperature_c"],
            "construct_identity": round(ident, 3), "n_res": n,
            "n_frames_total": n_total, "n_frames_expected": expected_frames(n), "n_frames_used": n_used,
            "traj_rg_nm": round(rg_sim, 4), "traj_rg_sem_nm": round(rg_sem, 4),
            "saxs_rg_nm": rg_saxs, "saxs_rg_err_nm": round(statistics.median(errs), 4) if errs else "",
            "ratio_traj_saxs": round(rg_sim / rg_saxs, 4), "abs_diff_nm": round(abs(rg_sim - rg_saxs), 4),
            "nu_traj": round(nu_sim, 4), "nu_saxs": round(nu_saxs, 4), "delta_nu": round(nu_sim - nu_saxs, 4),
            "ree_nm": round(ree, 4), "ree_std_nm": round(ree_std, 4), "ree_rg_ratio": round(ree / rg_sim, 4),
            "nu_ree_traj": round(math.log(ree / B0_REE) / math.log(n), 4), "flag": ""})
        print(f"  {i}/{len(names)} {name}", flush=True)

    used = collections.Counter(r["saxs_condition_id"] for r in rows)
    for r in rows:
        if used[r["saxs_condition_id"]] > 1:
            r["flag"] = "shared_saxs_condition"

    rows.sort(key=lambda r: (r["set"], r["taxon"], r["traj_id"]))
    common = ["traj_id", "uniprot_id", "protein_name", "taxon", "set", "source", "calvados2_overlap", "sasbdb_id", "saxs_condition_id",
              "saxs_entries", "saxs_n_entries", "saxs_temperature_c", "construct_identity", "n_res",
              "n_frames_total", "n_frames_expected", "n_frames_used", "traj_rg_nm", "saxs_rg_nm",
              "mpipi_n_frames_total", "mpipi_n_frames_expected", "mpipi_n_frames_used", "mpipi_rg_nm"]
    for fname, extra in (("traj_vs_saxs_nu.csv", ["nu_traj", "nu_saxs", "delta_nu", "nu_mpipi", "delta_nu_mpipi", "flag"]),
                         ("traj_vs_saxs_rg.csv", ["traj_rg_sem_nm", "saxs_rg_err_nm", "ratio_traj_saxs", "abs_diff_nm",
                                                  "mpipi_rg_sem_nm", "ratio_mpipi_saxs", "mpipi_max_bond_nm", "flag"])):
        with open(BENCH / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=common + extra, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    ree_cols = {"traj_id": "traj_id", "uniprot_id": "uniprot_id", "taxon": "taxon", "set": "set",
                "calvados2_overlap": "calvados2_overlap", "n_res": "n_res", "n_frames": "n_frames_used", "rg_nm": "traj_rg_nm",
                "ree_nm": "ree_nm", "ree_std_nm": "ree_std_nm", "ree_rg_ratio": "ree_rg_ratio", "nu_rg_traj": "nu_traj",
                "nu_ree_traj": "nu_ree_traj", "nu_saxs": "nu_saxs", "saxs_rg_nm": "saxs_rg_nm",
                "mpipi_rg_nm": "mpipi_rg_nm", "mpipi_ree_nm": "mpipi_ree_nm", "mpipi_ree_std_nm": "mpipi_ree_std_nm",
                "mpipi_ree_rg_ratio": "mpipi_ree_rg_ratio", "nu_mpipi": "nu_mpipi", "flag": "flag"}
    with open(BENCH / "traj_ree.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ree_cols))
        w.writeheader()
        w.writerows({k: r[v] for k, v in ree_cols.items()} for r in rows)
    with open(BENCH / "validation_excluded.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["traj_id", "reason", "detail"])
        w.writeheader()
        w.writerows(excluded)

    print(f"\ntrajectories: {len(names)} | validated: {len(rows)} | excluded: {len(excluded)} {[(e['traj_id'], e['reason']) for e in excluded]}")
    short = [(r["traj_id"], r["n_frames_total"], r["n_frames_expected"]) for r in rows if r["n_frames_total"] < 0.9 * r["n_frames_expected"]]
    print(f"runs below 90% of expected frames: {short or 'none'}")
    short_mp = [(r["traj_id"], r["mpipi_n_frames_total"], r["mpipi_n_frames_expected"]) for r in rows
                if r["mpipi_rg_nm"] != "" and r["mpipi_n_frames_total"] < 0.9 * r["mpipi_n_frames_expected"]]
    print(f"MPIPI-GG runs: {sum(r['mpipi_rg_nm'] != '' for r in rows)} / {len(rows)} | problems: {mpipi_problems or 'none'} | "
          f"below 90% of expected frames: {short_mp or 'none'} | max bond after unwrapping: "
          f"{max((r['mpipi_max_bond_nm'] for r in rows if r['mpipi_max_bond_nm'] != ''), default='n/a')} nm")
    print(f"shared SAXS conditions: {[k for k, v in used.items() if v > 1] or 'none'}\n")
    clean = [r for r in rows if not r["flag"]]
    for ff in FF_KEYS:
        print(f"--- {ff}")
        summarize([r for r in clean if r["set"] == "strict" and r["source"] == "existing"], "strict / existing", ff)
        summarize([r for r in clean if r["set"] == "strict" and r["source"] == "new"], "strict / new", ff)
        summarize([r for r in clean if r["set"] == "strict"], "STRICT (headline)", ff)
        summarize([r for r in clean if r["set"] == "no_uniprot_map"], "no_uniprot_map", ff)
        summarize(clean, "ALL", ff)
        summarize([r for r in clean if r["set"] == "strict" and r["calvados2_overlap"] not in SEEN_IN_TRAINING], "STRICT excl. training", ff)
        summarize([r for r in clean if r["set"] == "strict" and r["calvados2_overlap"] == "none"], "STRICT never seen", ff)
        summarize([r for r in clean if r["calvados2_overlap"] in SEEN_IN_TRAINING], "training overlap only", ff)
    paired = [r for r in clean if r["mpipi_rg_nm"] != ""]
    if len(paired) > 2:
        dc = np.array([r["delta_nu"] for r in paired]); dm = np.array([r["delta_nu_mpipi"] for r in paired])
        print(f"\npaired constructs {len(paired)}: r(dnu CALVADOS-2, dnu MPIPI-GG) {np.corrcoef(dc, dm)[0, 1]:.3f} | "
              f"MPIPI-GG closer to SAXS for {np.mean(np.abs(dm) < np.abs(dc)):.0%} | "
              f"r(Rg CALVADOS-2, Rg MPIPI-GG) {np.corrcoef([r['traj_rg_nm'] for r in paired], [r['mpipi_rg_nm'] for r in paired])[0, 1]:.3f}")
    print("\nstrict by taxon (mean dnu CALVADOS-2 | MPIPI-GG):")
    for t, cnt in sorted(collections.Counter(r["taxon"] for r in clean if r["set"] == "strict").items(), key=lambda kv: -kv[1]):
        sub = [r for r in clean if r["set"] == "strict" and r["taxon"] == t]
        mp = [r["delta_nu_mpipi"] for r in sub if r["delta_nu_mpipi"] != ""]
        print(f"  {t:17s} n={len(sub):3d} proteins={len({protein_key(r) for r in sub}):3d} mean dnu {np.mean([r['delta_nu'] for r in sub]):+.4f}"
              f" | {np.mean(mp) if mp else float('nan'):+.4f}")
    ratio = np.array([r["ree_rg_ratio"] for r in clean]); nre = np.array([r["nu_ree_traj"] for r in clean])
    print(f"\nRee/Rg: mean {ratio.mean():.3f} ± {ratio.std(ddof=1):.3f} (Gaussian chain sqrt(6) = 2.449) | "
          f"nu_Ree - nu_Rg (sim): {np.mean(nre - np.array([r['nu_traj'] for r in clean])):+.4f}")
    print("\n|dnu| >= 0.05:")
    for r in sorted(clean, key=lambda r: -abs(r["delta_nu"])):
        if abs(r["delta_nu"]) >= 0.05:
            print(f"  {r['traj_id']:10s} {r['protein_name'][:38]:38s} {r['set']:14s} N={r['n_res']:4d} "
                  f"Rg sim {r['traj_rg_nm']:.2f} / saxs {r['saxs_rg_nm']:.2f}  dnu {r['delta_nu']:+.3f}  T={r['saxs_temperature_c']}")
    print("\nMPIPI-GG |dnu| >= 0.05:")
    for r in sorted(paired, key=lambda r: -abs(r["delta_nu_mpipi"])):
        if abs(r["delta_nu_mpipi"]) >= 0.05:
            print(f"  {r['traj_id']:10s} {r['protein_name'][:38]:38s} {r['set']:14s} N={r['n_res']:4d} "
                  f"Rg MPIPI-GG {r['mpipi_rg_nm']:.2f} / saxs {r['saxs_rg_nm']:.2f}  dnu {r['delta_nu_mpipi']:+.3f}"
                  f"  (CALVADOS-2 {r['delta_nu']:+.3f})  T={r['saxs_temperature_c']}")


if __name__ == "__main__":
    main()
