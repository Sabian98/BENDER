"""SI figure: IDRome-style disorder check of the BENDER chains (2026-10-21).

(a) Per group, the share of chains whose whole sequence is one IDR by the Human-IDRome criterion (20-residue windowed
    AlphaFold2 pLDDT), whose IDRs cover >= 90%, 50-90% or < 50% of the chain, or that contain no IDR.
(b) Per-chain mean AlphaFold2 pLDDT, fully disordered chains and the rest stacked; dotted lines, 0.7 and 0.8.
(c) Mean dnu (MPIPI-Recharged minus CALVADOS-2) in deciles of arginine plus aromatic fraction, for fully disordered
    chains and for the rest: the chemistry dependence of the disagreement does not rest on chains with folded parts.
(d) Per group, the share of chains with UniProt domain-type annotations (f_domain > 0 and > 0.5).
Input: data/disorder_check/bender_disorder_check.csv (disorder_check.py analyze). Colors from palette.py.

Usage:  python make_supp_disorder.py [--out output/S_disorder_check]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import palette as P  # noqa: E402
import make_fig_disagreement as fd  # noqa: E402  (data loading, styling, icons)
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

CSV = paths.DISORDER / "bender_disorder_check.csv"
CATS = [("Whole chain", lambda d: d.full_w20.astype(bool)),
        (r"$\geq$90%", lambda d: ~d.full_w20.astype(bool) & (d.frac_idr_w20 >= 0.9)),
        ("50–90%", lambda d: (d.frac_idr_w20 >= 0.5) & (d.frac_idr_w20 < 0.9)),
        ("<50%", lambda d: (d.frac_idr_w20 > 0) & (d.frac_idr_w20 < 0.5)),
        ("No IDR", lambda d: d.frac_idr_w20 == 0)]
CAT_COL = ["#3F8F98", "#6BB2BA", "#9FD0D5", "#CFE8EA", P.RULE]


def main(out):
    d = pd.read_csv(CSV).set_index("acc")
    b, mr, fm = fd.load()
    have = d.dropna(subset=["full_w20"])
    W, H = 7.0, 2.55
    fig = plt.figure(figsize=(W, H))
    overlay = fig.add_axes([0, 0, 1, 1], zorder=10)
    overlay.set_xlim(0, W)
    overlay.set_ylim(0, H)
    overlay.axis("off")
    y0, h = 0.42, 1.68
    top = y0 + h + 0.30

    # (a) category shares per group ------------------------------------------------------------------
    ax = fig.add_axes([0.78 / W, y0 / H, 1.30 / W, h / H])
    groups = fd.ORDER + ["All"]
    for i, g in enumerate(groups):
        s = have if g == "All" else have[have.group == g]
        left = 0.0
        for (lab, f), col in zip(CATS, CAT_COL):
            v = 100 * f(s).mean()
            ax.barh(i, v, left=left, height=0.72, color=col, lw=0)
            left += v
        full = 100 * CATS[0][1](s).mean()
        ax.text(full - 2, i, f"{full:.0f}", ha="right", va="center", fontsize=5.4, color="white")
        overlay.text(0.70, (y0 + h * (1 - (i + 0.5) / len(groups))) / 1, f"{g} ({len(s):,})", ha="right",
                     va="center", fontsize=5.8, color=P.INK, fontweight="bold" if g == "All" else "normal")
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Chains (%)", fontsize=6.2)
    fd.clean(ax, left=False)
    xk = 0.78
    for (lab, _), col in zip(CATS, CAT_COL):
        overlay.add_patch(plt.Rectangle((xk, top - 0.20), 0.07, 0.06, color=col, lw=0))
        overlay.text(xk + 0.09, top - 0.17, lab, fontsize=5.2, color=P.DARK, va="center")
        xk += 0.13 + 0.042 * len(lab)
    fd.label(fig, 0.02, top, "a", "Chain covered by IDRs", W, H)

    # (b) mean pLDDT -----------------------------------------------------------------------------------
    ax = fig.add_axes([2.62 / W, y0 / H, 1.20 / W, h / H])
    bins = np.linspace(0.2, 1.0, 33)
    full = have.full_w20.astype(bool)
    ax.hist([have.mean_plddt[full], have.mean_plddt[~full]], bins=bins, stacked=True,
            color=[CAT_COL[0], P.RULE], lw=0)
    for x in (0.7, 0.8):
        ax.axvline(x, color=P.MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_xlabel("Mean pLDDT", fontsize=6.2)
    ax.set_ylabel("Chains", fontsize=6.2)
    ax.text(0.03, 0.97, "Whole chain IDR", transform=ax.transAxes, fontsize=5.4, color=CAT_COL[0], va="top")
    ax.text(0.03, 0.88, "Other", transform=ax.transAxes, fontsize=5.4, color=P.MUTED, va="top")
    fd.clean(ax)
    fd.label(fig, 2.30, top, "b", "AlphaFold2 confidence", W, H)

    # (c) chemistry dependence of dnu in both subsets ----------------------------------------------------
    ax = fig.add_axes([4.18 / W, y0 / H, 1.05 / W, h / H])
    dnu = (mr["nu"] - b["nu"]).loc[have.index]
    ra = (b["f_R"] + b["f_aro"]).loc[have.index]
    edges = np.quantile(ra, np.linspace(0, 1, 11))
    for sub, col, ls, lab in ((full, CAT_COL[0], "-", "Whole chain IDR"), (~full, P.MUTED, (0, (3, 2)), "Other")):
        xs, ms = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = sub & (ra >= lo) & (ra <= hi)
            xs.append(ra[sel].median())
            ms.append(dnu[sel].mean())
        ax.plot(xs, ms, color=col, lw=1.2, ls=ls, marker="o", ms=2.4, label=lab)
    ax.axhline(0, color=P.RULE, lw=0.6, zorder=0)
    ax.set_xlabel("Arg + Aromatic fraction", fontsize=6.2)
    ax.set_ylabel(r"Mean $\Delta$" + fd.NUTXT, fontsize=6.2)
    ax.legend(fontsize=5.2, frameon=False, loc="lower left", handlelength=2.2)
    fd.clean(ax)
    fd.label(fig, 3.80, top, "c", r"$\Delta$" + fd.NUTXT + " in both subsets", W, H)

    # (d) f_domain by group -------------------------------------------------------------------------------
    ax = fig.add_axes([5.98 / W, y0 / H, 0.95 / W, h / H])
    fdd = d.dropna(subset=["f_domain"])
    for i, g in enumerate(groups):
        s = fdd if g == "All" else fdd[fdd.group == g]
        v0, v5 = 100 * (s.f_domain > 0).mean(), 100 * (s.f_domain > 0.5).mean()
        ax.barh(i, v0, height=0.72, color=P.RULE, lw=0)
        ax.barh(i, v5, height=0.72, color=P.DARK, lw=0)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_yticks(range(len(groups)), [g if g != "All" else "All" for g in groups], fontsize=5.6)
    ax.set_xlabel("Chains (%)", fontsize=6.2)
    for k, (col, lab) in enumerate(((P.RULE, r"$f_\mathrm{domain}$ > 0"), (P.DARK, "> 0.5"))):
        xk = 5.98 + k * 0.62
        overlay.add_patch(plt.Rectangle((xk, top - 0.20), 0.07, 0.06, color=col, lw=0))
        overlay.text(xk + 0.09, top - 0.17, lab, fontsize=5.2, color=P.DARK, va="center")
    fd.clean(ax, left=False)
    fd.label(fig, 5.42, top, "d", "UniProt domains", W, H)

    out = Path(out).expanduser()
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), dpi=600, **kw)
    fig.savefig(out.with_suffix(".png"), dpi=300, **kw)
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.out("S_disorder_check")))
    main(ap.parse_args().out)
