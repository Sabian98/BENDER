"""Figure 6 (JCTC, 2026-10-22): the causal test and the experimental test of the disagreement.

(a) Mean dnu (MPIPI-Recharged minus CALVADOS-2) of the four sequence sets of the swap experiment, unswapped (gray)
    and with one class of MPIPI-Recharged interactions B2-matched to CALVADOS-2: arginine-aromatic and aromatic-
    aromatic pairs, aliphatic pairs, histidine charge, or all three; bars, 95% bootstrap intervals over chains;
    replicate runs (His-free chains under the His swap, whose potentials are unchanged) differ by 0.018 per chain.
(b) The 100 contested chains, dnu unswapped against dnu with the aromatic swap.
(c) Simulated against SAXS radius of gyration for the hnRNPA1 LCD variants of Bremer et al. 2022 with SAXS data;
    open symbols, A1-LCD+NLS, the only one outside the CALVADOS-2 training set.
(d) Spearman correlation, across those variants, of Rg with the number of arginine, aromatic, lysine and acidic
    residues: SAXS, MPIPI-Recharged and CALVADOS-2.
Data: data/controls/ (analyze_jctc_runs.py), simulations/inputs/. Styling as make_fig_disagreement.py.

Usage:  python make_fig_tests.py [--out output/tests]
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
import make_fig_disagreement as fd  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

DATA = paths.CONTROLS
INPUTS = paths.INPUTS
SETS = [("contested", "Contested"), ("aliph_rich", "Aliphatic-rich"), ("his_rich", "His-rich"), ("control", "Control")]
SWAPS = [("aro", "Arg/Aromatic pairs", "o", P.ARO, P.INK), ("aliph", "Aliphatic pairs", "s", P.RULE, P.DARK),
         ("his", "His charge", "D", "white", P.DARK), ("all", "All three", "^", P.DARK, P.DARK)]
RESIDUES = [("R", "Arg"), ("FWY", "Aromatic"), ("K", "Lys"), ("DE", "Asp + Glu")]
HELD_OUT = "A1LCD_pNLS"     # the one measured variant outside the CALVADOS-2 training set, drawn open in (c)


def boot_ci(x, B=4000, seed=0):
    x = np.asarray(x, float)
    bs = np.random.default_rng(seed).choice(x, (B, len(x))).mean(1)
    return np.percentile(bs, [2.5, 97.5])


def load_run(ff, name):
    p = DATA / f"{ff}_{name}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["nu", "rg", "sequence"])
    d = pd.read_csv(p).drop_duplicates("id", keep="last")
    return d[d.status == "ok"].set_index("id")


def main(out):
    ref = pd.read_csv(INPUTS / "swap_set_300K.tsv", sep="\t").set_index("UniProt_ID")
    sw = {k: load_run("recharged", f"swap_{k}")["nu"] for k, *_ in SWAPS}
    W, H = 7.0, 2.6
    fig = plt.figure(figsize=(W, H))
    y0, h = 0.42, 1.70
    top = y0 + h + 0.32

    # (a) swaps by sequence set ------------------------------------------------------------------
    ax = fig.add_axes([0.95 / W, y0 / H, 1.75 / W, h / H])
    nohis = ref.index[ref.f_H == 0]
    rep = (sw["his"].reindex(nohis) - ref.nu_recharged.reindex(nohis)).dropna()
    lo, hi = boot_ci(rep)
    for i, (key, lab) in enumerate(SETS):
        idx = ref.index[ref.set == key]
        base = ref.loc[idx, "dnu_recharged_minus_c2"]
        ax.axhspan(i - 0.42, i + 0.42, xmin=0, xmax=1, color=P.PALE, lw=0, zorder=0)
        b_lo, b_hi = boot_ci(base)
        ax.plot([b_lo, b_hi], [i - 0.36] * 2, color=P.MUTED, lw=0.8, zorder=2)
        ax.scatter(base.mean(), i - 0.36, s=18, color=P.MUTED, lw=0, zorder=3)
        for j, (k, slab, mk, fc, ec) in enumerate(SWAPS):
            after = sw[k].reindex(idx) - ref.loc[idx, "nu_calvados2"]
            a_lo, a_hi = boot_ci(after.dropna(), seed=j + 1)
            yy = i - 0.18 + 0.18 * j
            ax.plot([a_lo, a_hi], [yy, yy], color=ec, lw=0.8, zorder=2)
            ax.scatter(after.mean(), yy, s=16, marker=mk, facecolor=fc, edgecolor=ec, lw=0.5, zorder=4)
    ax.axvline(0, color=P.RULE, lw=0.7, zorder=0)
    ax.set_yticks(range(len(SETS)), [lab for _, lab in SETS], fontsize=6.2)
    ax.set_ylim(len(SETS) - 0.5, -0.55)
    ax.set_xlabel(r"Mean $\Delta$" + fd.NUTXT, fontsize=6.4)
    ax.tick_params(axis="y", length=0)
    fd.clean(ax, left=False)
    entries = [("Unswapped", "o", P.MUTED, P.MUTED)] + [(s[1], s[2], s[3], s[4]) for s in SWAPS]
    for (lab, mk, fc, ec), xk in zip(entries, (0.95, 1.62, 2.55, 3.35, 4.00)):
        fig.text((xk + 0.07) / W, (top - 0.20) / H, lab, fontsize=5.2, color=P.DARK, va="center")
        fig.add_artist(plt.Line2D([(xk + 0.02) / W], [(top - 0.20) / H], marker=mk, color=fc, markeredgecolor=ec,
                                  markeredgewidth=0.5, markersize=3.6, transform=fig.transFigure, ls=""))
    fd.label(fig, 0.02, top, "a", "Parameter swap", W, H)

    # (b) contested chains, per chain ---------------------------------------------------------------
    ax = fig.add_axes([3.15 / W, y0 / H, 1.20 / W, h / H])
    c = ref[ref.set == "contested"]
    before = c.dnu_recharged_minus_c2
    after = sw["aro"].reindex(c.index) - c.nu_calvados2
    lim = (min(before.min(), after.min()) - 0.01, max(before.max(), after.max()) + 0.01)
    ax.plot(lim, lim, color=P.RULE, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.axhline(0, color=P.RULE, lw=0.6, zorder=0)
    ax.axvline(0, color=P.RULE, lw=0.6, zorder=0)
    ax.scatter(before, after, s=8, facecolor=P.ARO, edgecolor=P.INK, lw=0.3, zorder=3)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta$" + fd.NUTXT + ", unswapped", fontsize=6.2)
    ax.set_ylabel(r"$\Delta$" + fd.NUTXT + ", Arg/Aromatic swap", fontsize=6.2)
    fd.clean(ax)
    fd.label(fig, 2.80, top, "b", "Contested chains", W, H)

    # (c) Bremer variants against SAXS ------------------------------------------------------------------
    br = pd.read_csv(INPUTS / "bremer2022_a1lcd_300K.tsv", sep="\t").set_index("run_name").dropna(subset=["saxs_Rg_A"])
    mr, c2 = load_run("recharged", "bremer2022"), load_run("calvados2", "bremer2022")
    ax = fig.add_axes([4.75 / W, y0 / H, 1.05 / W, h / H])
    sims = []
    for d, col, mk in ((mr, P.MR, "s"), (c2, P.C2, "o")):
        j = br.join(d[["rg"]], how="inner")
        held = j.index == HELD_OUT
        ax.scatter(j.saxs_Rg_A[~held], 10 * j.rg[~held], s=12, marker=mk, facecolor=col, edgecolor="white", lw=0.3,
                   zorder=3)
        ax.scatter(j.saxs_Rg_A[held], 10 * j.rg[held], s=12, marker=mk, facecolor="white", edgecolor=col, lw=0.8,
                   zorder=4)
        sims += list(10 * j.rg)
    xlim = (br.saxs_Rg_A.min() - 0.5, br.saxs_Rg_A.max() + 0.5)
    ylim = (min(sims) - 0.6, max(sims) + 0.6)
    ends = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    ax.plot(ends, ends, color=P.RULE, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for k, (lab, col, mk) in enumerate((("MPIPI-Recharged", P.MR, "s"), ("CALVADOS-2", P.C2, "o"))):
        yk = (top - 0.13 - 0.09 * k) / H
        fig.add_artist(plt.Line2D([4.80 / W], [yk], marker=mk, color=col, markeredgecolor="white",
                                  markeredgewidth=0.3, markersize=3.4, transform=fig.transFigure, ls=""))
        fig.text(4.86 / W, yk, lab, fontsize=5.0, color=col, va="center", fontweight="bold")
    ax.set_xlabel(r"$R_g$, SAXS (Å)", fontsize=6.2)
    ax.set_ylabel(r"$R_g$, simulation (Å)", fontsize=6.2)
    fd.clean(ax)
    fd.label(fig, 4.35, top, "c", "hnRNPA1 LCD variants", W, H)

    # (d) response to residue changes -------------------------------------------------------------------
    ax = fig.add_axes([6.20 / W, y0 / H, 0.75 / W, h / H])
    seqs = mr.sequence.reindex(br.index)
    for i, (aa, lab) in enumerate(RESIDUES):
        cnt = seqs.map(lambda q: sum(q.count(a) for a in aa))
        vals = [("SAXS", br.saxs_Rg_A, P.MUTED), ("MPIPI-Recharged", 10 * mr.rg.reindex(br.index), P.MR),
                ("CALVADOS-2", 10 * c2.rg.reindex(br.index), P.C2)]
        for j, (name, rg, col) in enumerate(vals):
            ok = rg.notna()
            if ok.sum() < 8:                                  # CALVADOS-2 runs still arriving
                continue
            rho = cnt[ok].corr(rg[ok], method="spearman")
            ax.barh(i + (j - 1) * 0.26, rho, height=0.24, color=col, lw=0)
    ax.axvline(0, color=P.DARK, lw=0.6)
    ax.set_yticks(range(len(RESIDUES)), [lab for _, lab in RESIDUES], fontsize=6.0)
    ax.set_ylim(len(RESIDUES) - 0.5, -0.6)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_xlabel(r"Spearman $\rho$ with $R_g$", fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    fd.clean(ax, left=False)
    for k, (lab, col) in enumerate((("SAXS", P.MUTED), ("MPIPI-Recharged", P.MR), ("CALVADOS-2", P.C2))):
        fig.text(6.20 / W, (top - 0.13 - 0.09 * k) / H, lab, fontsize=5.0, color=col, va="center",
                 fontweight="bold")
    fd.label(fig, 5.85, top, "d", "Response", W, H)

    out = Path(out).expanduser()
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), dpi=600, **kw)
    fig.savefig(out.with_suffix(".png"), dpi=300, **kw)
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.out("tests")))
    main(ap.parse_args().out)
