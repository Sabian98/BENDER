"""Figure 4 (JCTC, 2026-10-21): lineages differ in how much contested chemistry they carry.

(a) Each group in composition space (Arg + aromatic fraction against Lys fraction; density), with the contested
    region (top decile of f_R + f_aro, f_K below the median) outlined and the group's share inside it.
(b) Per group, the share in the contested class and the mean dnu inside it, raw and after the composition model
    (squared and pairwise terms, five-fold cross-validated; make_fig_disagreement.composition_model).
(c) Per group, the share of sequences with |dnu| > 0.1 (protists as one group since 2026-10-22).
(d) A contested protist, A0A7S2N1D9 (Alexandrium andersonii): conformers from both force fields
    on one scale, its Arg and aromatic positions, and Rg over each trajectory (moved from the first Fig. 2).
Shared data, definitions and styling come from make_fig_disagreement.py.

Usage:  python make_fig_lineages.py [--out output/lineages]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fig_disagreement as fd  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)
fig1 = fd.fig1
NUTXT, INK, MUTED, RULE, SEQ = fd.NUTXT, fd.INK, fd.MUTED, fd.RULE, fd.SEQ


def main(out):
    b, mr, fm = fd.load()
    d, cls, ratio, mcpi, t = fd.numbers(b, mr, fm)
    ex = json.load(open(fd.CONFORMERS))
    EX = fd.EXAMPLE
    exc2, exmr = b.loc[EX, 'nu'], mr.loc[EX, 'nu']
    ra = (b['f_R'] + b['f_aro']).to_numpy()
    fk = b['f_K'].to_numpy()
    x90, kmed = np.quantile(ra, 0.9), np.median(fk)
    g = b['kingdom'].to_numpy()

    W, H = 7.0, 4.55
    fig = plt.figure(figsize=(W, H))
    overlay = fig.add_axes([0, 0, 1, 1], zorder=10)
    overlay.set_xlim(0, W)
    overlay.set_ylim(0, H)
    overlay.set_aspect("equal")
    overlay.axis("off")

    # (a) composition space, one small panel per group ---------------------------------------
    ya, sa = 2.78, 1.02
    xe, ye = (0, 0.42), (0, 0.30)
    for i, k in enumerate(fd.ORDER):
        x0 = 0.42 + i * 1.10
        ax = fig.add_axes([x0 / W, ya / H, sa / W, sa / H])
        s = g == k
        ax.hexbin(ra[s], fk[s], gridsize=(20, 15), extent=(*xe, *ye), cmap=fd.DENSITY, norm=LogNorm(1, 60),
                  mincnt=1, linewidths=0.1, edgecolors="face", rasterized=True)
        ax.add_patch(plt.Rectangle((x90, 0), xe[1] - x90, kmed, facecolor="none", edgecolor=fd.RECHARGED_DARK,
                                   lw=0.9, zorder=5))
        ax.text(0.97, 0.96, f"{100 * cls[s].mean():.1f}%", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.4, color=fd.RECHARGED_DARK, fontweight="bold")
        ax.set_xlim(*xe)
        ax.set_ylim(*ye)
        ax.set_xticks([0, 0.2, 0.4], ["0" if i == 0 else "", "0.2", "0.4"])
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.tick_params(labelsize=5.2)
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Lys fraction", fontsize=6)
        ax.set_xlabel("Arg + Aromatic", fontsize=5.8, labelpad=1)
        fd.clean(ax)
        fig.text((x0 + sa / 2 + 0.10) / W, (ya + sa + 0.05) / H, k, ha="center", va="bottom", fontsize=6.4,
                 color=INK)
        fig1.lineage_icon(overlay, k, x0 + sa / 2 - 0.28, ya + sa + 0.12, 0.15)
    fd.label(fig, 0.02, ya + sa + 0.30, "a", "Each taxon in composition space", W, H)
    fig.text((0.42 + 6 * 1.10 - 0.08) / W, (ya + sa + 0.32) / H, "Outline marks the contested region, with the share of each taxon inside",
             ha="right", va="bottom", fontsize=5.4, color=fd.RECHARGED_DARK)

    # (b) contested class by group (share; mean dnu raw and adjusted) ------------------------
    y2, h = 0.40, 1.80
    top2 = y2 + h + 0.20
    rows = fd.ORDER + ['All']
    ylim = (len(rows) - 0.4, -0.6)
    axs = fig.add_axes([0.95 / W, y2 / H, 0.62 / W, h / H])
    axd = fig.add_axes([1.76 / W, y2 / H, 0.87 / W, h / H])
    for i, k in enumerate(rows):
        axs.barh(i, 100 * t.loc[k, 'share'], height=0.56, color=SEQ if k != 'All' else MUTED, lw=0, zorder=2)
        axs.text(100 * t.loc[k, 'share'] + 0.4, i, f"{100 * t.loc[k, 'share']:.1f}", va="center", ha="left",
                 fontsize=5.4, color=INK)
        raw, adj = t.loc[k, 'raw'], t.loc[k, 'adj']
        if np.isfinite(adj):
            axd.annotate("", xy=(adj, i), xytext=(raw, i), zorder=2,
                         arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", color=MUTED, lw=0.6,
                                         shrinkA=2.2, shrinkB=2.6))
            axd.scatter([adj], [i], s=11, facecolor="white", edgecolor=INK, lw=0.7, zorder=3)
        axd.scatter([raw], [i], s=11, color=INK, lw=0, zorder=4)
    for a_ in (axs, axd):
        a_.set_ylim(*ylim)
        a_.set_yticks([])
        a_.axhline(len(fd.ORDER) - 0.5, color=RULE, lw=0.5)
        fd.clean(a_, left=False)
    axs.set_xlim(0, 16)
    axs.set_xticks([0, 5, 10, 15])
    axs.set_xlabel("In class (%)")
    axd.axvline(0, color=INK, lw=0.5, zorder=1)
    axd.set_xlim(-0.145, 0.03)
    axd.set_xticks([-0.1, -0.05, 0])
    axd.set_xticklabels(["−0.10", "−0.05", "0"])
    axd.set_xlabel(f"Δ{NUTXT} in class")
    axd.scatter([-0.135], [-0.95], s=9, color=INK, lw=0, clip_on=False)
    axd.text(-0.127, -0.95, "Raw", fontsize=5.4, va="center", color=INK)
    axd.scatter([-0.085], [-0.95], s=9, facecolor="white", edgecolor=INK, lw=0.6, clip_on=False)
    axd.text(-0.077, -0.95, "Adjusted", fontsize=5.4, va="center", color=INK)
    fd.row_labels(fig, overlay, axs, W, H, 0.68, 0.82)
    fd.label(fig, 0.02, top2, "b", "Contested class by taxon", W, H)

    # (c) large disagreements by group ----------------------------------------------------------
    big = (d.abs() > 0.1).to_numpy()
    rows_c = [(k, g == k) for k in fd.ORDER]
    axc = fig.add_axes([3.62 / W, y2 / H, 0.95 / W, h / H])
    for i, (k, s) in enumerate(rows_c):
        v = 100 * big[s].mean()
        col = fd.RECHARGED if k == "Protists" else SEQ
        axc.barh(i, v, height=0.56, color=col, lw=0, zorder=2)
        axc.text(v + 0.25, i, f"{v:.1f}", va="center", ha="left", fontsize=5.4, color=INK)
        axc.text(-0.4, i, f"{k} ({int(s.sum()):,})", va="center", ha="right", fontsize=5.6, color=INK)
    axc.set_ylim(len(rows_c) - 0.4, -0.6)
    axc.set_yticks([])
    axc.set_xlim(0, 14)
    axc.set_xticks([0, 5, 10])
    axc.set_xlabel(f"|Δ{NUTXT}| > 0.1 (%)")
    fd.clean(axc, left=False)
    fd.label(fig, 2.78, top2, "c", "Large disagreements", W, H)

    # (d) the contested protist (moved from the first Fig. 2f) --------------------------------
    seq = ex["sequence"]
    cc, cm = (np.array(ex[k]["conformers_nm"]) for k in ("calvados2", "recharged"))
    half = 1.04 * max(np.abs(cc).max(), np.abs(cm).max())
    size, fx, fy = 0.90, 4.98, y2 + 0.71
    for j, (conf, col, dark, name, nu_, rg_) in enumerate((
            (cc, fd.CALVADOS, fd.CALVADOS_DARK, "CALVADOS-2", exc2, b.loc[EX, 'rg']),
            (cm, fd.RECHARGED, fd.RECHARGED_DARK, "MPIPI-Recharged", exmr, mr.loc[EX, 'rg_mean']))):
        x0 = fx + j * (size + 0.06)
        a_ = fig.add_axes([x0 / W, fy / H, size / W, size / H])
        fd.conformer_panel(a_, conf, col, dark, seq, half)
        fig.text((x0 + size / 2) / W, (fy + size + 0.03) / H, name, ha="center", va="bottom", fontsize=6.4,
                 color=dark, fontweight="bold")
        fig.text((x0 + size / 2) / W, (fy - 0.03) / H, f"{NUTXT} {nu_:.2f}    $R_g$ {rg_:.1f} nm", ha="center",
                 va="top", fontsize=6.0, color=INK)
        if j == 1:
            a_.plot([half - 5.4, half - 0.4], [-half + 0.4] * 2, color=INK, lw=1.0, solid_capstyle="butt")
            a_.text(half - 2.9, -half + 0.75, "5 nm", ha="center", va="bottom", fontsize=5.4, color=INK)
    bw = 2 * size + 0.06
    ab = fig.add_axes([fx / W, (y2 + 0.40) / H, bw / W, 0.07 / H])
    for i, aa in enumerate(seq):
        col = INK if aa == "R" else fd.ARO if aa in "FWY" else fd.P.OTHER_RES
        ab.add_patch(plt.Rectangle((i, 0), 0.82, 1, color=col, lw=0))
    ab.set_xlim(0, len(seq))
    ab.set_ylim(0, 1)
    ab.axis("off")
    fig.text(fx / W, (y2 + 0.49) / H, f"Sequence, {len(seq)} aa", fontsize=5.4, color=MUTED, va="bottom")
    xt = fx + bw
    for txt, col in (("F W Y", fd.ARO), ("R", INK)):
        fig.text(xt / W, (y2 + 0.49) / H, txt, fontsize=5.8, color=col, va="bottom", ha="right", fontweight="bold")
        xt -= 0.32
    ar = fig.add_axes([fx / W, y2 / H, bw / W, 0.27 / H])
    grid = np.linspace(1.0, 5.5, 300)
    for k, col, dark in (("calvados2", fd.CALVADOS, fd.CALVADOS_DARK), ("recharged", fd.RECHARGED,
                                                                          fd.RECHARGED_DARK)):
        v = np.array(ex[k]["rg_frames_nm"])
        dens = gaussian_kde(v, bw_method=0.25)(grid)
        ar.fill_between(grid, 0, dens, color=col, alpha=0.55, lw=0)
        ar.plot(grid, dens, color=dark, lw=0.6)
    ar.set_xlim(grid[0], grid[-1])
    ar.set_ylim(0, None)
    ar.set_yticks([])
    ar.set_xticks([1, 2, 3, 4, 5])
    ar.set_xlabel("$R_g$ over the trajectory (nm)")
    fd.clean(ar, left=False)
    fd.label(fig, 4.80, top2, "d", f"Contested protist {EX}", W, H)

    out = Path(out).expanduser()
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), dpi=600, **kw)
    fig.savefig(out.with_suffix(".png"), dpi=400, **kw)
    print("wrote", out.with_suffix(".pdf"))
    for k, s in rows_c:
        print(f"  {k:24s} n {int(s.sum()):5d}  |dnu|>0.1 {100 * big[s].mean():.1f}%  in class {100 * cls[s].mean():.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.out("lineages")))
    main(ap.parse_args().out)
