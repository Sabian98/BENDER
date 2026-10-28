"""SAXS SI figures for the BENDER JCTC draft (2026-10-21): figures4papers palette (palette.py) and Kratky profiles
for all three force fields, without editing the shared SAXS scripts in analysis/saxs.

Stages:
  compute  MPIPI-Recharged I(q) of the 77 SAXS constructs, cached in bender_benchmark/figures/recharged_iq_cache.json
           (a new file next to the CALVADOS-2 and MPIPI-GG caches). Frames are read as validate_saxs_recharged.py
           reads them (production half, 200 evenly spaced frames, chain rebuilt if a bond exceeds 1 nm, rejected if
           the largest bond is 0.5 nm or more), and I(q)/I(0) comes from figure_saxs_validation.debye_iq on the same
           q grid as the other two force fields. Each run's Rg is checked against recharged_vs_saxs.csv.
  plot     (default) summary, three-force-field comparison, representative Kratky panels and the nine Kratky pages,
           written to output/saxs; side CSVs go to output/saxs_side_csv.

Kratky lines: SAXS solid black, CALVADOS-2 dashed navy, MPIPI-GG dotted teal, MPIPI-Recharged dash-dotted red.
Panel text gives Delta nu (simulation minus SAXS) for CALVADOS-2 / MPIPI-GG / MPIPI-Recharged.

Usage:  ~/miniforge3/envs/openmpipi/bin/python -B make_supp_saxs_f4p.py [compute|plot]
"""
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SRC = os.path.dirname(os.path.abspath(__file__))
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)
sys.path[:0] = [str(paths.ANALYSIS), str(paths.ANALYSIS / "saxs"), SRC]
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
import palette as P  # noqa: E402
import figure_saxs_validation as F  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CACHE_MR = F.BENCH / "figures" / "recharged_iq_cache.json"
RECHARGED_CSV = F.BENCH / "recharged_vs_saxs.csv"
OUT = paths.outdir("saxs")
SIDE = paths.outdir("saxs_side_csv")

GG_LINE = P.GG_DARK
F.C_EXP, F.C_SIM = P.INK, P.C2              # CALVADOS-2 navy: teal vs the lighter blue fill is too close under tritan vision
F.C_MP, F.C_MP_LINE = P.GG, GG_LINE
F.INK, F.INK2, F.INK3, F.BAND = P.INK, P.DARK, P.MUTED, P.PALE
F.POINT_C, F.RULE_C = "#999999", P.RULE
matplotlib.rcParams.update({"axes.edgecolor": P.DARK, "axes.labelcolor": P.INK, "xtick.color": P.DARK,
                            "ytick.color": P.DARK})

import figure_saxs_forcefields_recharged as R  # noqa: E402  (imported after F's colors are set)
R.C_MR, R.C_MR_DARK = P.MR, P.MR_DARK
R.FFS = [("CALVADOS-2", P.C2, P.C2_DARK, "traj_rg_nm", "nu_traj", "delta_nu"),
         ("MPIPI-GG", P.GG, GG_LINE, "mpipi_rg_nm", "nu_mpipi", "delta_nu_mpipi"),
         ("MPIPI-Recharged", P.MR, P.MR_DARK, "recharged_rg_nm", "nu_recharged", "delta_nu_recharged")]

# The share scripts start two annotations in lower case; capitalise them here, as with the colors above
_axes_text = matplotlib.axes.Axes.text


def _text(self, x, y, s, *args, **kwargs):
    if isinstance(s, str) and s.startswith(("constructs", "good solvent")):
        s = s[0].upper() + s[1:]
    return _axes_text(self, x, y, s, *args, **kwargs)


matplotlib.axes.Axes.text = _text

# ...write their legend separators without colons or semicolons ("A: x; y" -> "A (x, y)") and the force-field family
# as MPIPI, as in the text, and the deviation from SAXS as δν (Δν is MPIPI-Recharged minus CALVADOS-2)
_set_text = matplotlib.text.Text.set_text


def _set_text_plain(self, s):
    if isinstance(s, str):
        s = s.replace("Mpipi", "MPIPI")
        s = s.replace(r"\Delta\mathsf{\nu}", r"\delta\mathsf{\nu}")  # simulation minus SAXS is δν, as in the SI
        s = re.sub(r"^(MPIPI-GG|MPIPI-Recharged|CALVADOS-2): (.+)$", r"\1 (\2)", s.replace("; ", ", "))
    return _set_text(self, s)


matplotlib.text.Text.set_text = _set_text_plain

# Kratky line styles, one per source; the legend is built from the same dicts
LINES = [("SAXS (SASBDB)", dict(color=P.INK, lw=1.5), 3),
         ("CALVADOS-2", dict(color=P.C2, lw=1.4, ls=(0, (4, 1.6))), 5),
         ("MPIPI-GG", dict(color=GG_LINE, lw=1.8, ls=(0, (0.1, 2.0)), dash_capstyle="round"), 6),
         ("MPIPI-Recharged", dict(color=P.MR, lw=1.3, ls=(0, (6, 1.5, 1, 1.5))), 4)]
DNU_COLS = ("delta_nu", "delta_nu_mpipi", "delta_nu_recharged")


# ---------------------------------------------------------------------------------------------------------------
def compute():
    import bender_common as bc
    import validate_saxs_recharged as VR
    rows = F.load_rows()
    ref = {r["traj_id"]: r for r in csv.DictReader(open(RECHARGED_CSV))}
    cache = json.load(open(CACHE_MR)) if CACHE_MR.exists() else {}
    todo = [r for r in rows if r["traj_id"] not in cache]
    for i, r in enumerate(todo, 1):
        tid = r["traj_id"]
        d = VR.RUNS / tid
        xyz, seq, info = bc.load_calvados(str(d / f"{tid}.dcd"), str(d / "top.pdb"), k=F.TARGET_FRAMES)
        assert info["production_start"] == int(info["n_frames_total"] * VR.EQUIL_FRAC), tid
        assert len(seq) == r["n_res"], f"{tid}: {len(seq)} residues, benchmark lists {r['n_res']}"
        if bc.max_bond(xyz) > VR.UNWRAP_ABOVE_NM:
            xyz = bc.unwrap_chain(xyz, VR.box_nm(len(seq)))
        assert bc.max_bond(xyz) < VR.MAX_BOND_NM, f"{tid}: chain broken"
        rg = bc.geometry(xyz)["rg"]
        assert abs(rg - float(ref[tid]["recharged_rg_nm"])) < 1e-3, f"{tid}: Rg {rg:.4f} vs {ref[tid]['recharged_rg_nm']}"
        cache[tid] = F.debye_iq(xyz)
        json.dump({"q": F.Q_GRID.tolist(), **{k: v for k, v in cache.items() if k != "q"}}, open(CACHE_MR, "w"))
        print(f"  {i}/{len(todo)} {tid}  N={len(seq)}  Rg={rg:.3f} nm", flush=True)
    print("cached", len([k for k in json.load(open(CACHE_MR)) if k != "q"]), "constructs in", CACHE_MR)


# ---------------------------------------------------------------------------------------------------------------
def fmt_dnu(v):
    return "–" if v is None else F.fmt_dnu(v).replace("-", "−")


def kratky_panel(ax, r, caches, entries, title_size=7.6, mark=""):
    code = F.representative_entry(r, entries)
    xb, kb, rg = F.exp_kratky(code)
    ax.axvline(math.sqrt(3), color=P.RULE, lw=1.0, zorder=0)
    ax.plot(xb, kb, zorder=LINES[0][2], **LINES[0][1])
    tops = [np.nanmax(kb[xb <= 7]) if len(kb) else 0]
    for (_, style, z), cache in zip(LINES[1:], caches):
        if r["traj_id"] not in cache:
            continue
        xs = np.array(cache["q"]) * rg
        k = xs ** 2 * np.array(cache[r["traj_id"]])
        ax.plot(xs, k, zorder=z, **style)
        tops.append(np.nanmax(k[xs <= 7]))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, max(2.2, max(tops) * 1.12))
    ax.set_title(f"{F.short_name(r['protein_name'])}{mark}", fontsize=title_size, color=P.INK, pad=3, loc="left")
    ident = f"{r['label']} ({r['traj_id']})" if r["traj_id"] != r["label"] else r["label"]
    dnu = " / ".join(fmt_dnu(r.get(c)) for c in DNU_COLS)
    ax.text(0.97, 0.04, f"{ident}\n{r['taxon']} · $N$ = {r['n_res']}\n$\\Delta\\mathsf{{\\nu}}$ = {dnu}",
            transform=ax.transAxes, fontsize=6.0, color=P.DARK, va="bottom", ha="right", linespacing=1.3)
    return code


def legend(fig, **kw):
    fig.legend([Line2D([0], [0], **s) for _, s, _ in LINES], [n for n, _, _ in LINES], **kw)


def kratky_grid(chunk, caches, entries, marks=None):
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.9))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.11, wspace=0.28, hspace=0.62)
    codes = []
    for i, ax in enumerate(axes.flat):
        if i >= len(chunk):
            ax.axis("off")
            continue
        codes.append(kratky_panel(ax, chunk[i], caches, entries, mark=marks[i] if marks else ""))
        if i % 4 == 0:
            ax.set_ylabel("$(qR_g)^2\\, I(q)/I(0)$")
        if i >= 4 or i + 4 >= len(chunk):
            ax.set_xlabel("$qR_g$")
    return fig, codes


def fig_representative(rows, caches, entries):
    reps = F.pick_representatives(rows)
    fig, codes = kratky_grid(reps, caches, entries)
    legend(fig, loc="center", bbox_to_anchor=(0.875, 0.30), fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_saxs_kratky_representative.{ext}", bbox_inches="tight")
    plt.close(fig)
    return reps, codes


def fig_supplementary(rows, caches, entries, reps):
    rep_ids = {r["traj_id"] for r in reps}
    rest = sorted([r for r in rows if r["traj_id"] not in rep_ids],
                  key=lambda r: (F.TAXON_ORDER.index(r["taxon"]), not r["strict"], r["label"], r["traj_id"]))
    pages = [rest[i:i + 8] for i in range(0, len(rest), 8)]
    manifest = []
    for p, chunk in enumerate(pages, 1):
        marks = [("†" if not r["independent"] else "") + ("§" if not r["strict"] else "") for r in chunk]
        fig, codes = kratky_grid(chunk, caches, entries, marks)
        legend(fig, loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=4, fontsize=7.5)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"figS_saxs_kratky_{p:02d}.{ext}", bbox_inches="tight")
        plt.close(fig)
        manifest += [{"figure": f"S{p}", "traj_id": r["traj_id"], "label": r["label"], "protein": r["protein_name"],
                      "taxon": r["taxon"], "sasbdb_entry_plotted": c} for r, c in zip(chunk, codes)]
    with open(SIDE / "kratky_panel_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    return pages


def plot():
    SIDE.mkdir(parents=True, exist_ok=True)
    F.OUT, F.FIG = OUT, SIDE
    rows = R.load(str(RECHARGED_CSV))              # CALVADOS-2 / MPIPI-GG rows plus the MPIPI-Recharged columns
    caches = [json.load(open(F.CACHE)), json.load(open(F.CACHE_MP)), json.load(open(CACHE_MR))]
    missing = [r["traj_id"] for r in rows if any(r["traj_id"] not in c for c in caches)]
    assert not missing, f"I(q) missing for {missing}"
    entries = {r["sasbdb_id"]: r for r in csv.DictReader(open(F.EXP / "processed/sasbdb_entries_screened.csv"))}
    F.fig_summary(rows)
    reps, codes = fig_representative(rows, caches, entries)
    with open(SIDE / "representative_panels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["taxon", "traj_id", "label", "protein", "n_res", "delta_nu", "delta_nu_mpipi",
                    "delta_nu_recharged", "sasbdb_entry_plotted"])
        for r, c in zip(reps, codes):
            w.writerow([r["taxon"], r["traj_id"], r["label"], r["protein_name"], r["n_res"],
                        *(r[k] for k in DNU_COLS), c])
    pages = fig_supplementary(rows, caches, entries, reps)
    print("representatives:", [(r["taxon"], r["label"]) for r in reps], "pages:", len(pages), [len(p) for p in pages])
    R.main(str(OUT), str(RECHARGED_CSV))


if __name__ == "__main__":
    {"compute": compute, "plot": plot}[sys.argv[1] if len(sys.argv) > 1 else "plot"]()
