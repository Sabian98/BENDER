"""Per-lineage property distributions (supplement) and Flory-fit quality (main).

Six major lineages only, in the tree order of Fig. 1. Style matches Fig. 1 and
the validation figure: soft palette, upright text, horizontal violins with a
white median dot and interquartile bar, lineage identified by labeled rows.
Color encodes data source only: taupe for properties computed from the
sequence, powder blue for properties computed from CALVADOS-2 ensembles.

Inputs:
  lineage_properties.csv            (build_lineage_properties.py, from BENDER_BIO.csv;
                                     all 10,282 sequences, none excluded)
  data/bender_bio/BENDER_BIO.csv         (Flory fit R^2 of the primary nu; 10,282 sequences)

Outputs (paper figures folder):
  S_dist_sequence_geometry.pdf
  S_dist_network.pdf
  nu_fit_r2_per_taxa.pdf
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
from scipy.stats import gaussian_kde
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

HERE = Path(__file__).resolve().parent
FIGS = paths.outdir()
BIO = paths.BIO

ORDER = ["Mammals", "Fungi", "Plants", "Protists", "Bacteria", "Viruses"]
import palette as P  # noqa: E402  (shared JCTC scheme, 2026-10-21)
CALV, CALV_DARK = P.C2_FILL, P.C2     # CALVADOS-2 ensembles
SEQ, SEQ_DARK = P.SEQ, P.SEQ_DARK       # sequence-derived
INK, MUTED, RULE = P.INK, P.MUTED, P.RULE

plt.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.5,
    "xtick.major.size": 2.0, "xtick.labelsize": 6.0, "axes.labelsize": 6.8,
    "axes.labelcolor": INK,
})

# (column, axis label, color pair, log x, reference lines)
SEQ_GEOM = [
    ("length", "Length (residues)", "seq", True, []),
    ("FCR", "Fraction of charged residues", "seq", False, []),
    ("lambda_avg", r"Mean stickiness $\lambda$ (CALVADOS-2)", "seq", False, []),
    ("rg", r"$R_\mathrm{g}$ (nm)", "calv", False, []),
    ("ree", r"$R_\mathrm{e}$ (nm)", "calv", False, []),
    ("nu", r"Scaling exponent $\mathsf{\nu}$", "calv", False,
     [(0.5, "0.50"), (0.588, "0.588")]),
]
NETWORK = [
    ("global_efficiency", r"Global efficiency $g_\mathrm{eff}$", "calv", False, []),
    ("fragmentation_index", "Fragmentation index", "calv", False, []),
    ("avg_clustering", "Average clustering coefficient", "calv", False, []),
    ("transitivity", "Transitivity", "calv", False, []),
    ("degree_assortativity", "Degree assortativity", "calv", False, [(0.0, "0")]),
]
COLOURS = {"seq": (SEQ, SEQ_DARK), "calv": (CALV, CALV_DARK)}


def violin(ax, values, y, color, dark, lo, hi, half=0.38, log=False):
    v = values[(values >= lo) & (values <= hi)]
    t = np.log10(v) if log else v
    grid = np.linspace(t.min(), t.max(), 200)
    dens = gaussian_kde(t, bw_method=0.2)(grid)
    dens = dens / dens.max() * half
    x = 10 ** grid if log else grid
    ax.fill_between(x, y - dens, y + dens, facecolor=color, alpha=0.6,
                    edgecolor=dark, lw=0.5, zorder=2)
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    ax.plot([q1, q3], [y, y], color=dark, lw=1.5, solid_capstyle="butt",
            zorder=3)
    ax.scatter([med], [y], s=8, facecolor="white", edgecolor=dark,
               linewidth=0.6, zorder=4)


def panel(ax, df, col, label, kind, log, refs, show_labels, counts):
    color, dark = COLOURS[kind]
    lo, hi = np.nanpercentile(df[col], [0.5, 99.5])     # clip the far tails
    for x, _ in refs:                                   # keep reference lines
        lo, hi = min(lo, x - 0.02 * (hi - lo)), max(hi, x + 0.02 * (hi - lo))
    for y, key in enumerate(ORDER):
        violin(ax, df.loc[df["lineage"] == key, col].dropna().to_numpy(), y,
               color, dark, lo, hi, log=log)
        if show_labels:
            ax.text(-0.05, y - 0.12, key, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6.6, color=INK)
            ax.text(-0.05, y + 0.26, f"{counts[key]:,}",
                    transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=5.4, color=MUTED)
    for x, text in refs:
        ax.axvline(x, color=MUTED, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
        ax.text(x, -0.62, text, ha="center", va="bottom", fontsize=5.2,
                color=MUTED)
    if log:
        ax.set_xscale("log")
        ax.set_xticks([t for t in (30, 50, 100, 200, 400) if lo <= t <= hi])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(ORDER) - 0.45, -0.6)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(label, labelpad=2)


def grid_figure(df, specs, out, legend):
    counts = df["lineage"].value_counts().to_dict()
    W, H = 5.5, 4.1
    fig = plt.figure(figsize=(W, H))
    left, colw, gap, rowh, top = 0.62, 1.42, 0.30, 1.45, 3.95
    for k, spec in enumerate(specs):
        r, c = divmod(k, 3)
        x0 = left + c * (colw + gap)
        y0 = top - (r + 1) * rowh - r * 0.52
        ax = fig.add_axes([x0 / W, y0 / H, colw / W, rowh / H])
        panel(ax, df, *spec, show_labels=(c == 0), counts=counts)
        fig.text((x0 - 0.02) / W, (y0 + rowh + 0.04) / H, "abcdef"[k],
                 fontsize=9, fontweight="bold", color=INK, va="bottom",
                 ha="right" if c == 0 else "left")
    if legend:
        fig.legend(handles=[Patch(facecolor=SEQ, edgecolor=SEQ_DARK, lw=0.5,
                                  label="From sequence"),
                            Patch(facecolor=CALV, edgecolor=CALV_DARK, lw=0.5,
                                  label="From CALVADOS-2 ensembles")],
                   loc="lower center", bbox_to_anchor=(0.56, 1.0), ncol=2,
                   frameon=False, fontsize=6.2, handlelength=1.2)
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), **kw)
    fig.savefig(out.with_suffix(".png"), dpi=300, **kw)
    plt.close(fig)
    print("wrote", out.name)


R2_CUT = 0.99     # fits below this R^2 are flagged in BENDER_BIO (nu_r2_pass); none are excluded


def convergence_figure(out):
    b = pd.read_csv(BIO, usecols=["UniProt_ID", "kingdom", "nu_fit_r2"]).rename(
        columns={"kingdom": "lineage"})
    b = b[b["lineage"].isin(ORDER)]                   # the six BENDER groups
    W, H = 4.1, 2.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.95 / W, 0.36 / H, 2.55 / W, 1.55 / H])
    lo = 0.95
    for y, key in enumerate(ORDER):
        v = b.loc[b["lineage"] == key, "nu_fit_r2"].to_numpy()
        violin(ax, v, y, CALV, CALV_DARK, lo, 1.0)
        ax.text(-0.04, y - 0.12, key, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.6, color=INK)
        ax.text(-0.04, y + 0.26, f"{len(v):,}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=5.4, color=MUTED)
        ax.text(1.02, y, f"{(v < R2_CUT).sum()}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.0, color=INK)
    ax.text(1.02, -0.75, f"Fit $R^2$ < {R2_CUT}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=5.6, color=MUTED)
    ax.axvline(R2_CUT, color=MUTED, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
    ax.text(R2_CUT, -0.62, f"{R2_CUT}", ha="center", va="bottom", fontsize=5.2, color=MUTED)
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(len(ORDER) - 0.45, -0.6)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax.set_xlabel(r"Flory scaling fit $R^2$ (axis starts at 0.95)", labelpad=2)
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), **kw)
    fig.savefig(out.with_suffix(".png"), dpi=300, **kw)
    plt.close(fig)
    print("wrote", out.name, "| n =", len(b), "| median %.4f" % b["nu_fit_r2"].median(),
          "| >= %.2f: %.1f%%" % (R2_CUT, 100 * (b["nu_fit_r2"] >= R2_CUT).mean()))
    print((b.assign(low=b["nu_fit_r2"] < R2_CUT).groupby("lineage")["low"].agg(["size", "sum", "mean"])
           .reindex(ORDER).round(3).to_string()))


def main():
    df = pd.read_csv(HERE / "lineage_properties.csv")
    df = df[df["lineage"].isin(ORDER)]
    print("distribution figures:", len(df), "sequences")
    grid_figure(df, SEQ_GEOM, FIGS / "S_dist_sequence_geometry",
                legend=True)
    grid_figure(df, NETWORK, FIGS / "S_dist_network", legend=False)
    convergence_figure(FIGS / "nu_fit_r2_per_taxa")


if __name__ == "__main__":
    main()
