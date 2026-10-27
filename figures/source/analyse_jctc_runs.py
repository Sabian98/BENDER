"""Inventory and analysis of the JCTC follow-up runs (2026-10-21; run package simulations/controls/).

Sets (trajectories/controls/<force field>/<set>/300K/<ID>/<ID>.dcd + top.pdb):
  calvados2/termini_control       300 BENDER sequences, CALVADOS-2 with uncharged termini
  calvados2/termini_his_control   the same, uncharged termini and pH 6.0 (His charge +0.5)
  calvados2/bremer2022            38 hnRNPA1 LCD variants (Bremer et al. 2022)
  recharged/bremer2022            the same 38 in MPIPI-Recharged
  recharged/swap_{aro,aliph,his,all}   300 BENDER sequences, one interaction class set to CALVADOS-2 (B2-matched)
  recharged/bremer2022_swap_aro    the 38 hnRNPA1 LCD variants with the Arg/aromatic swap (added 2026-10-22)
  recharged/saxs_swap_{gsp,gspaliph}  the 77 SAXS benchmark constructs with the Gly/Ser/Pro swap and the
                                 Gly/Ser/Pro + aliphatic swap (added 2026-10-22)

A run is complete when its DCD holds bender_common.expected_frames(N) frames (the BENDER schedule); the count is
read from the DCD header, so runs still being written are listed but not analyzed. Complete runs are analyzed as in
BENDER (traj_analysis_calvados_v2.py): production half, 1,000 evenly spaced frames, bender_common.geometry and
bender_common.flory. MPIPI-Recharged chains are rebuilt whole from minimum-image bonds when a bond exceeds 1 nm and
rejected if the largest bond is 0.5 nm or more after that (validate_saxs_recharged.py rule).

  python analyze_jctc_runs.py inventory
  python analyze_jctc_runs.py analyze      (incremental; cache in data/controls/<ff>_<set>.csv)
  python analyze_jctc_runs.py summary
"""
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

sys.dont_write_bytecode = True
sys.path.insert(0, str(paths.ANALYSIS))
import bender_common as bc  # noqa: E402

ROOT = paths.TRAJ_CONTROLS
INPUTS = paths.INPUTS
OUT = paths.CONTROLS
SETS = [("calvados2", "termini_control"), ("calvados2", "termini_his_control"), ("calvados2", "bremer2022"),
        ("recharged", "bremer2022"), ("recharged", "swap_aro"), ("recharged", "swap_aliph"),
        ("recharged", "swap_his"), ("recharged", "swap_all"), ("recharged", "bremer2022_swap_aro"),
        ("recharged", "saxs_swap_gsp"), ("recharged", "saxs_swap_gspaliph")]
K_FRAMES = 1000
UNWRAP_ABOVE_NM, MAX_BOND_NM = 1.0, 0.5


def dcd_frames(path):
    """Frame count from the CHARMM/OpenMM DCD header (NSET), cross-checked against the file size."""
    with open(path, "rb") as fh:
        head = fh.read(100)
    if len(head) < 12 or head[4:8] != b"CORD":
        return 0
    nset = struct.unpack("<i", head[8:12])[0]
    return nset


def n_atoms(top):
    return sum(1 for line in open(top) if line.startswith(("ATOM", "HETATM")))


def runs(ff, name):
    base = ROOT / ff / name / "300K"
    for d in sorted(base.iterdir()) if base.exists() else []:
        dcd, top = d / f"{d.name}.dcd", d / "top.pdb"
        if not (dcd.exists() and top.exists()):
            yield d.name, None, None, 0, None
            continue
        n = n_atoms(top)
        settled = time.time() - dcd.stat().st_mtime > 120      # not still being written or copied
        yield d.name, dcd, top, dcd_frames(dcd) if settled else 0, n


def box_nm(n):
    return float(int(n / 5) + 20 if n > 300 else 50)


def inventory(verbose=True):
    rows = []
    for ff, name in SETS:
        for rid, dcd, top, frames, n in runs(ff, name):
            exp = bc.expected_frames(n) if n else None
            rows.append({"ff": ff, "set": name, "id": rid, "has_dcd": dcd is not None, "n": n, "frames": frames,
                         "expected": exp, "complete": bool(n and frames >= exp)})
    inv = pd.DataFrame(rows)
    if verbose:
        g = inv.groupby(["ff", "set"]).agg(folders=("id", "size"), with_dcd=("has_dcd", "sum"),
                                           complete=("complete", "sum"))
        print(g.to_string())
    return inv


def analyse():
    OUT.mkdir(parents=True, exist_ok=True)
    inv = inventory(verbose=False)
    for (ff, name), sub in inv[inv.complete].groupby(["ff", "set"]):
        path = OUT / f"{ff}_{name}.csv"
        if path.exists():
            prev = pd.read_csv(path).drop_duplicates("id", keep="last")
            done = set(prev.loc[prev.status.astype(str).isin(["ok", "broken chain"]), "id"])   # errors are retried
        else:
            done = set()
        todo = sub[~sub.id.isin(done)]
        for k, r in enumerate(todo.itertuples(), 1):
            d = ROOT / ff / name / "300K" / r.id
            try:
                xyz, seq, info = bc.load_calvados(str(d / f"{r.id}.dcd"), str(d / "top.pdb"), k=K_FRAMES)
                rec = {"id": r.id, "n": len(seq), "sequence": seq, "frames": info["n_frames_total"]}
                if ff == "recharged":
                    raw = bc.max_bond(xyz)
                    if raw > UNWRAP_ABOVE_NM:
                        xyz = bc.unwrap_chain(xyz, box_nm(len(seq)))
                    rec["max_bond_nm"] = bc.max_bond(xyz)
                    if rec["max_bond_nm"] >= MAX_BOND_NM:
                        rec["status"] = "broken chain"
                        pd.DataFrame([rec]).to_csv(path, mode="a", header=not path.exists(), index=False)
                        continue
                g = bc.geometry(xyz)
                f = bc.flory(xyz)
                rec.update(status="ok", rg=g["rg"], rg_err=g["rg_err"], ree=g["ree"], nu=f["nu"],
                           nu_r2=f["nu_r2"], nu_block_se=f["nu_block_se"])
            except Exception as e:  # noqa: BLE001  (a run being rewritten, a corrupt file)
                rec = {"id": r.id, "status": f"error: {str(e)[:120]}"}
            pd.DataFrame([rec]).to_csv(path, mode="a", header=not path.exists(), index=False)
            if k % 10 == 0:
                print(f"  {ff}/{name}: {k}/{len(todo)}", flush=True)
        print(f"{ff}/{name}: analysed {len(todo)} new runs", flush=True)


def load(ff, name):
    p = OUT / f"{ff}_{name}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["id", "nu", "status"])
    d = pd.read_csv(p).drop_duplicates("id", keep="last")
    return d[d.status == "ok"].set_index("id")


def summary():
    inventory()
    print()
    ref = pd.read_csv(INPUTS / "termini_control_300K.tsv", sep="\t").set_index("UniProt_ID")
    for name, label in (("termini_control", "uncharged termini"), ("termini_his_control", "uncharged termini, pH 6.0")):
        d = load("calvados2", name).join(ref, how="inner")
        if d.empty:
            print(f"{label}: no complete runs yet")
            continue
        shift = d.nu - d.nu_calvados2_charged_termini
        gap0 = (d.nu_recharged - d.nu_calvados2_charged_termini).abs()
        gap1 = (d.nu_recharged - d.nu).abs()
        noise = np.sqrt(2) * d.nu_block_se.median()
        print(f"{label}: {len(d)} complete | shift in nu: mean {shift.mean():+.4f}, median |shift| "
              f"{shift.abs().median():.4f} (noise ~{noise:.4f}) | corr(shift, dnu) "
              f"{np.corrcoef(shift, d.dnu_recharged_minus_c2)[0, 1]:+.2f} | gap to MPIPI-Recharged "
              f"{gap0.mean():.4f} -> {gap1.mean():.4f} ({100 * (1 - gap1.mean() / gap0.mean()):+.0f}% closed)")
        for st, s in d.groupby("stratum"):
            g0 = (s.nu_recharged - s.nu_calvados2_charged_termini).abs().mean()
            g1 = (s.nu_recharged - s.nu).abs().mean()
            print(f"    {st:18s} n {len(s):3d} | mean shift {(s.nu - s.nu_calvados2_charged_termini).mean():+.4f} | "
                  f"gap {g0:.4f} -> {g1:.4f}")
    print()
    sw = pd.read_csv(INPUTS / "swap_set_300K.tsv", sep="\t")
    sw = sw.set_index(sw.columns[0])
    print("swap set columns:", list(sw.columns))
    for name in ("swap_aro", "swap_aliph", "swap_his", "swap_all"):
        d = load("recharged", name).join(sw, how="inner", rsuffix="_ref")
        if d.empty:
            print(f"{name}: no complete runs yet")
            continue
        c2 = next(c for c in sw.columns if "calvados" in c.lower() or c.lower() in ("nu_c2", "nu_calvados2"))
        mr = next(c for c in sw.columns if "recharged" in c.lower() and c.lower().startswith("nu"))
        grp = next(c for c in sw.columns if c.lower() in ("group", "set", "swap_group"))
        print(f"{name}: {len(d)} complete")
        for gname, s in d.groupby(grp):
            dnu0 = (s[mr] - s[c2]).mean()
            dnu1 = (s.nu - s[c2]).mean()
            print(f"    {gname:11s} n {len(s):3d} | dnu unswapped {dnu0:+.4f} -> swapped {dnu1:+.4f}")
    print()
    br = pd.read_csv(INPUTS / "bremer2022_a1lcd_300K.tsv", sep="\t").set_index("run_name")
    for ff in ("calvados2", "recharged"):
        d = load(ff, "bremer2022").join(br, how="inner")
        m = d.dropna(subset=["saxs_Rg_A"])
        if m.empty:
            print(f"bremer {ff}: {len(d)} complete, none with SAXS yet")
            continue
        rg_a = 10 * m.rg
        print(f"bremer {ff}: {len(d)} complete, {len(m)} with SAXS | Rg MARE "
              f"{100 * np.mean(np.abs(rg_a - m.saxs_Rg_A) / m.saxs_Rg_A):.1f}% | r(Rg) "
              f"{np.corrcoef(rg_a, m.saxs_Rg_A)[0, 1]:.3f}" if len(m) > 2 else f"bremer {ff}: {len(m)} with SAXS")


EXPECTED = {("recharged", "bremer2022"): 38, ("recharged", "swap_aro"): 300, ("recharged", "swap_aliph"): 300,
            ("recharged", "swap_his"): 300, ("recharged", "swap_all"): 300, ("recharged", "bremer2022_swap_aro"): 38,
            ("recharged", "saxs_swap_gsp"): 77, ("recharged", "saxs_swap_gspaliph"): 77}


def wait_recharged(poll=180):
    """Analyze new runs as they arrive; return once every MPIPI-Recharged set is complete."""
    while True:
        analyse()
        inv = inventory(verbose=False)
        got = inv[inv.complete].groupby(["ff", "set"]).size()
        left = {k: n - int(got.get(k, 0)) for k, n in EXPECTED.items() if got.get(k, 0) < n}
        print(time.strftime("%H:%M"), "remaining:", left or "none", flush=True)
        if not left:
            analyse()                       # runs that completed during the last pass
            summary()
            return
        time.sleep(poll)


if __name__ == "__main__":
    {"inventory": inventory, "analyse": analyse, "summary": summary, "wait-recharged": wait_recharged}[sys.argv[1]]()
