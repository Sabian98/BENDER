"""Figure 3: agreement of force-field ensembles with AlphaFold2 PAE.

(a) CALVADOS-2 on every BENDER sequence with an AF2 PAE matrix, by group.
(b) The BENDER sequences simulated with both CALVADOS-2 and the second force
    field, by group: MPIPI-Recharged on all 9,982 sequences with PAE (default,
    2026-10-20), or MPIPI-GG on its 957-sequence subset (--ff gg, the figure
    of the arXiv version).
(c) Bland-Altman comparison of the two force fields on those sequences.

The metric is the per-sequence Pearson r between inter-residue distance SDs
from the trajectory and the AF2 PAE matrix (analysis/validation/pae_v2.py).
Color encodes force field only, as in Fig. 1 and the SAXS figures (the
soft palette): CALVADOS-2 powder blue, MPIPI-GG peach, MPIPI-Recharged plum,
warm grays otherwise. Lineages are identified by labeled rows, in the tree
order of Fig. 1.

Inputs:
  data/pae/calvados_vs_pae_results_v2.tsv  (analysis/validation/pae_v2.py:
  data/pae/mpipi_vs_pae_results_v2.tsv      production frames; MPIPI-GG unwrapped)
  data/mpipi_recharged/recharged_300K_vs_pae_results_v2.tsv
                                                                   (recharged_300K_features.py: the same pae_v2
                                                                    correlate on 1,000 production frames)
  data/bender_bio/BENDER_BIO.csv                                     (10,282 sequences; kingdom = group)

Top-up tables (same columns, e.g. the 300 protists added in 2026-10) are appended with --extra-calvados and
--extra-second; only status ok rows are kept, and a sequence may be ok in one table only.

Usage:  python make_fig3_pae_validation.py [--ff recharged|gg] [--extra-calvados TSV] [--extra-second TSV]
                                           [--out ../calvados_mpipi_pae_convergence]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

HERE = Path(__file__).resolve().parent
TSV = paths.PAE
BIO_CSV = paths.BIO

RECHARGED_TSV = paths.RECHARGED / "recharged_300K_vs_pae_results_v2.tsv"

import palette as P  # noqa: E402  (shared JCTC scheme after figures4papers, 2026-10-21)
CALVADOS, CALVADOS_DARK = P.C2_FILL, P.C2
MPIPI, MPIPI_DARK = P.GG, P.GG_DARK
RECHARGED, RECHARGED_DARK = P.MR_FILL, P.MR
# second force field: results table, name, color, dark color, Bland-Altman marker size
SECOND = {"recharged": (RECHARGED_TSV, "MPIPI-Recharged", RECHARGED, RECHARGED_DARK, 1.2),
          "gg": (TSV / "mpipi_vs_pae_results_v2.tsv", "MPIPI-GG", MPIPI, MPIPI_DARK, 3.2)}
INK, MUTED, POINTS, RULE = P.INK, P.MUTED, "#999999", P.RULE
BAND = P.PALE   # 95% limits of agreement (cream, as the SAXS box in Fig. 1)
LEGEND_ROOM = 0.42  # rows of space under the last group for the one-line legend of (b)
# same row order as the tree in Fig. 1
ORDER = ["Mammals", "Fungi", "Plants", "Protists", "Bacteria", "Viruses"]

plt.rcParams.update({
    "font.family": "Arial",
    # upright maths in the body sans; Greek letters via \mathsf (DejaVu Sans),
    # whose nu is distinct from a Latin v (Arial's is not)
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "axes.labelsize": 6.6,
    "axes.labelcolor": INK,
})


def read_ok(paths):
    """Rows with status ok from a results table and its top-up tables (same columns); an id may be ok only once."""
    t = pd.concat([pd.read_csv(f, sep="\t") for f in paths], ignore_index=True)
    t = t[t["status"] == "ok"]
    dup = t["uniprot_id"][t["uniprot_id"].duplicated()]
    assert dup.empty, f"ok in more than one table: {sorted(set(dup))[:10]}"
    return t


def load(second_tsv, extra_calvados=(), extra_second=()):
    lin = pd.read_csv(BIO_CSV, usecols=["UniProt_ID", "kingdom"]).rename(
        columns={"UniProt_ID": "uniprot_id", "kingdom": "lineage"})
    calv = read_ok([TSV / "calvados_vs_pae_results_v2.tsv", *extra_calvados])
    calv = calv.merge(lin, on="uniprot_id", how="inner")
    calv = calv[calv["lineage"].isin(ORDER)]                 # the six BENDER groups
    mp = read_ok([second_tsv, *extra_second])
    paired = mp[["uniprot_id", "pearson_r"]].merge(
        calv[["uniprot_id", "pearson_r", "lineage"]], on="uniprot_id",
        suffixes=("_mpipi", "_calvados"))
    print(f"paired sequences in the six groups: {len(paired)} of {len(mp)}")
    return calv, paired


def violin(ax, values, y, color, dark, half=0.36, xlim=(0, 1)):
    """Horizontal violin with median (white dot) and interquartile bar."""
    grid = np.linspace(max(values.min(), xlim[0]), min(values.max(), xlim[1]),
                       200)
    dens = gaussian_kde(values, bw_method=0.25)(grid)
    dens = dens / dens.max() * half
    ax.fill_between(grid, y - dens, y + dens, facecolor=color, alpha=0.45,
                    edgecolor=color, lw=0.6, zorder=2)
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    ax.plot([q1, q3], [y, y], color=dark, lw=1.6, solid_capstyle="butt",
            zorder=3)
    ax.scatter([med], [y], s=9, facecolor="white", edgecolor=dark,
               linewidth=0.7, zorder=4)
    return med


def style(ax, xlim):
    ax.set_xlim(*xlim)
    ax.set_ylim(len(ORDER) - 0.45 + LEGEND_ROOM, -0.55)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.axvline(0.8, color=RULE, lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)


def panel_label(fig, x, y, letter, title, W, H):
    fig.text(x / W, y / H, letter, fontsize=9.5, fontweight="bold",
             color=INK, va="bottom")
    fig.text((x + 0.17) / W, y / H, title, fontsize=6.8, fontweight="bold",
             color=INK, va="bottom")


def main(out, ff, extra_calvados=(), extra_second=()):
    second_tsv, NAME, COL, DARK, ms = SECOND[ff]
    calv, paired = load(second_tsv, extra_calvados, extra_second)
    W, H = 5.5, 2.62
    fig = plt.figure(figsize=(W, H))
    bottom, height = 0.40, 1.92
    ax_a = fig.add_axes([0.80 / W, bottom / H, 1.42 / W, height / H])
    ax_b = fig.add_axes([2.36 / W, bottom / H, 1.50 / W, height / H])
    ax_c = fig.add_axes([4.36 / W, bottom / H, 1.10 / W, height / H])
    top = (bottom + height + 0.08)

    # (a) CALVADOS-2, every sequence with a PAE matrix
    for y, key in enumerate(ORDER):
        v = calv.loc[calv["lineage"] == key, "pearson_r"].to_numpy()
        violin(ax_a, v, y, CALVADOS, CALVADOS_DARK)
        ax_a.text(-0.04, y - 0.1, key, transform=ax_a.get_yaxis_transform(),
                  ha="right", va="center", fontsize=6.8, color=INK)
        ax_a.text(-0.04, y + 0.25, f"{len(v):,}",
                  transform=ax_a.get_yaxis_transform(), ha="right",
                  va="center", fontsize=5.6, color=MUTED)
    style(ax_a, (0, 1))
    ax_a.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    panel_label(fig, 0.05, top, "a",
                f"CALVADOS-2, all {len(calv):,} sequences", W, H)

    # (b) the paired sequences simulated with both force fields
    for y, key in enumerate(ORDER):
        sub = paired[paired["lineage"] == key]
        if sub.empty:
            ax_b.text(0.98, y, f"no {NAME} runs", ha="right", va="center",
                      fontsize=5.8, color=MUTED)
            continue
        violin(ax_b, sub["pearson_r_calvados"].to_numpy(), y - 0.2,
               CALVADOS, CALVADOS_DARK, half=0.19)
        violin(ax_b, sub["pearson_r_mpipi"].to_numpy(), y + 0.2, COL,
               DARK, half=0.19)
    style(ax_b, (0, 1))
    ax_b.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    panel_label(fig, 2.26 - 0.10, top,
                "b", f"{len(paired):,} sequences, both force fields", W, H)
    ax_b.legend(handles=[Patch(facecolor=CALVADOS, alpha=0.5,
                               edgecolor=CALVADOS, lw=0.6, label="CALVADOS-2"),
                         Patch(facecolor=COL, alpha=0.5, edgecolor=COL,
                               lw=0.6, label=NAME)],
                loc="lower left", bbox_to_anchor=(0.0, -0.01), frameon=False, ncol=2,
                columnspacing=1.2, fontsize=5.8, handlelength=1.0, handleheight=0.7,
                borderaxespad=0.2, labelspacing=0.3)

    fig.text((0.80 + 3.86) / 2 / W, 0.10 / H,
             "Pearson $r$ between trajectory distance SD and AF2 PAE",
             ha="center", va="center", fontsize=6.6, color=INK)

    # (c) Bland-Altman on the paired sequences
    a, b = paired["pearson_r_calvados"], paired["pearson_r_mpipi"]
    mean, diff = (a + b) / 2, a - b
    mu, sd = diff.mean(), diff.std()
    lo, hi = mu - 1.96 * sd, mu + 1.96 * sd
    # points colored by which force field agrees better with AF2 PAE
    better = diff > 0
    ax_c.axhspan(lo, hi, color=BAND, lw=0, zorder=0)
    for mask, col in ((better, CALVADOS), (~better, COL)):
        ax_c.scatter(mean[mask], diff[mask], s=ms, color=col, alpha=0.8,
                     lw=0, zorder=2, rasterized=True)
    ax_c.axhline(0, color=RULE, lw=0.6, zorder=1)
    ax_c.text(0.97, 0.975, f"CALVADOS-2 better ({100 * better.mean():.0f}%)",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=5.4, zorder=5,
              color=CALVADOS_DARK)
    ax_c.text(0.03, 0.01, f"{NAME} better ({100 * (~better).mean():.0f}%)",
              transform=ax_c.transAxes, ha="left", va="bottom", fontsize=5.4,
              color=DARK)
    ax_c.axhline(mu, color=INK, lw=0.9, zorder=3)
    for v in (lo, hi):
        ax_c.axhline(v, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=3)
    def signed(x, digits):
        return f"{x:+.{digits}f}".replace("-", "−")
    # y-range follows the data (0.2nd-99.8th percentile) with room for the labels
    q_lo, q_hi = np.quantile(diff, [0.002, 0.998])
    y0, y1 = min(q_lo, lo), max(q_hi, hi)
    span = y1 - y0
    ylim = (y0 - 0.30 * span, y1 + 0.45 * span)
    n_out = int(((diff < ylim[0]) | (diff > ylim[1])).sum())
    ax_c.text(0.03, 0.88, f"Mean {signed(mu, 3)}\n95% limits {signed(lo, 3)}, {signed(hi, 3)}",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=5.4, color=INK,
              linespacing=1.35, zorder=5,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5))
    ax_c.set_xlim(0.25, 1.0)
    ax_c.set_ylim(*ylim)
    print(f"Bland-Altman y-range {ylim[0]:.3f} to {ylim[1]:.3f}; points outside: {n_out}")
    ax_c.set_xticks([0.4, 0.6, 0.8, 1.0])
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    ax_c.set_xlabel("Mean $r$ of both force fields", labelpad=2)
    ax_c.set_ylabel(f"Difference in $r$ (CALVADOS-2 − {NAME})",
                    labelpad=1.5)
    panel_label(fig, 4.02, top, "c", f"CALVADOS-2 vs {NAME}", W, H)

    out = Path(out)
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), dpi=600, **kw)
    fig.savefig(out.with_suffix(".png"), dpi=400, **kw)
    med = calv.groupby("lineage")["pearson_r"].median().round(3).to_dict()
    pm = paired.groupby("lineage")[["pearson_r_calvados", "pearson_r_mpipi"]]\
        .median().round(3)
    print("wrote", out.with_suffix(".pdf"))
    print(f"all: n={len(calv)} median={calv['pearson_r'].median():.3f}", med)
    print(f"paired medians: calv {a.median():.3f} {NAME} {b.median():.3f}")
    print(pm.to_string())
    print(f"Bland-Altman: mean {mu:+.4f}, LoA [{lo:+.3f}, {hi:+.3f}], "
          f"CALVADOS-2 higher r in {100 * better.mean():.1f}%, |d| < 0.04 in {100 * (diff.abs() < 0.04).mean():.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.out("calvados_mpipi_pae_convergence")))
    ap.add_argument("--ff", choices=sorted(SECOND), default="recharged",
                    help="second force field in (b) and (c); gg is the MPIPI-GG subset of the arXiv version")
    ap.add_argument("--extra-calvados", action="append", default=[], metavar="TSV",
                    help="top-up CALVADOS-2 PAE results with the columns of calvados_vs_pae_results_v2.tsv "
                         "(repeatable), e.g. the 300 protists added in 2026-10")
    ap.add_argument("--extra-second", action="append", default=[], metavar="TSV",
                    help="top-up PAE results for the second force field (repeatable)")
    a = ap.parse_args()
    main(a.out, a.ff, a.extra_calvados, a.extra_second)
