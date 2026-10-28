"""Figure 4 (JCTC): the disagreement is written into the pair potentials.
(a) dB2 = B2(MPIPI-Recharged) - B2(CALVADOS-2) for all 210 residue pairs (300 K, 150 mM, full non-bonded
    potentials), lower triangle, residues grouped by chemistry. Diverging scale with symmetric limits: plum, the pair
    is more attractive under MPIPI-Recharged; blue, under CALVADOS-2; gray, no difference.
(b) Mean B2 of each chemical pair class in both force fields (dumbbells), ordered by dB2.
(c, d) The pair potentials u(r) of Arg-Tyr and Leu-Leu in both force fields (non-bonded, 300 K, 150 mM).
(e) Per sequence, dnu against the mean dB2 over its residue pairs |i - j| >= 3 (density; contested class on top).
(f) From potentials to contacts to compaction: per sequence, the Arg-aromatic term of the mean dB2 against the
    log ratio of cation-pi contact frequencies (MPIPI-Recharged / CALVADOS-2), hexbins colored by mean dnu.
(g) Five-fold cross-validated R2 of dnu: composition models against B2 models with far fewer terms, and the
    sampling-noise ceiling (1 - share of var(dnu) from block standard errors).
Pair potentials come from b2_mechanism.py (u_cal, u_mp; checked there against OpenMM to 1e-8 kJ/mol), so this
script needs the openmpipi env (calvados, OpenMpipi). Contact frequencies: BENDER_BIO.csv (CALVADOS-2) and
recharged_300K_features.csv (MPIPI-Recharged).
All values come from the b2_mechanism outputs (data/mpipi_recharged/b2_mechanism/); nothing is
refitted here. Palette as Fig. 1 and Fig. 3: CALVADOS-2 blue, MPIPI-Recharged plum, warm grays, taupe for
sequence-derived quantities.

Usage:  python make_fig_b2_mechanism.py [--b2dir DIR] [--out output/b2_mechanism]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from scipy import stats
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

B2DIR = paths.B2
OUT = paths.out("b2_mechanism")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import palette as P  # noqa: E402  (shared JCTC scheme after figures4papers, 2026-10-21)
CALVADOS, CALVADOS_DARK = P.C2_FILL, P.C2
RECHARGED, RECHARGED_DARK = P.MR_FILL, P.MR
INK, MUTED, RULE = P.INK, P.MUTED, P.RULE
SEQ, SEQ_DARK = P.SEQ, P.SEQ_DARK
NEUTRAL = "#F2F2F2"
DENSITY = P.DENSITY
DIVERGING = P.DIVERGING          # -lim (MPIPI-Recharged stickier, red) ... 0 ... +lim (CALVADOS-2, blue)
GROUPS = [("FWY", "Aro"), ("R", "R"), ("K", "K"), ("H", "H"), ("DE", "Acid"), ("AILMV", "Aliph"), ("GPSTNQC", "Other")]
RES = "".join(g for g, _ in GROUPS)
NUTXT = r"$\mathsf{\nu}$"
NOISE_SHARE = 0.101          # paper_numbers_recharged.py: share of var(dnu) from block standard errors

plt.rcParams.update({
    "font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial", "mathtext.cal": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.labelcolor": INK,
    "text.color": INK,
})


def panel_label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def heatmap(ax, cax, pairs):
    n = len(RES)
    idx = {c: i for i, c in enumerate(RES)}
    M = np.full((n, n), np.nan)
    for r in pairs.itertuples():
        i, j = idx[r.res1], idx[r.res2]
        a, b = max(i, j), min(i, j)
        M[a, b] = r.dB2                                 # lower triangle incl. diagonal
    lim = np.ceil(np.nanmax(np.abs(M)) * 2) / 2         # symmetric limits: equal steps per arm
    im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(0.0, -lim, lim),
                   interpolation="nearest")
    ax.set_xticks(range(n), list(RES), fontsize=5.8)
    ax.set_yticks(range(n), list(RES), fontsize=5.8)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    edges = np.cumsum([len(g) for g, _ in GROUPS])[:-1] - 0.5
    for e in edges:                                    # class separators, inside the triangle only
        ax.plot([e, e], [e, n - 0.5], color="white", lw=1.2)
        ax.plot([-0.5, e], [e, e], color="white", lw=1.2)
    start = 0
    for g, name in GROUPS:                             # class names along the diagonal, multi-residue classes only
        if len(g) > 1:
            mid = start + (len(g) - 1) / 2
            ax.text(mid + 1.1, mid - 1.1, name, fontsize=6, color=MUTED, ha="left", va="bottom")
        start += len(g)
    cb = plt.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.8, length=1.5)
    cb.set_label(r"$\Delta B_2$ (nm$^3$)", fontsize=6.3, labelpad=1)
    cax.text(0.5, 1.04, "Stickier in\nCALVADOS-2", transform=cax.transAxes, fontsize=5.4, color=CALVADOS_DARK,
             ha="center", va="bottom")
    cax.text(0.5, -0.04, "Stickier in\nMPIPI-Recharged", transform=cax.transAxes, fontsize=5.4,
             color=RECHARGED_DARK, ha="center", va="top")
    return lim


def scatter(ax, feat):
    x, y = feat["pairs3_dB2"].to_numpy(), feat["dnu"].to_numpy()
    hb = ax.hexbin(x, y, gridsize=55, cmap=DENSITY, norm=LogNorm(), mincnt=1, linewidths=0, zorder=1)
    c = feat["contested_class"].astype(bool).to_numpy()
    ax.scatter(x[c], y[c], s=3.5, color=RECHARGED, edgecolor="white", linewidth=0.25, zorder=3,
               label=rf"$\mathcal{{Contested}}$ class (n = {c.sum():,})")
    ax.axhline(0, color=RULE, lw=0.6, zorder=0)
    ax.axvline(0, color=RULE, lw=0.6, zorder=0)
    r = stats.pearsonr(x, y)[0]
    ax.text(0.03, 0.97, f"r = {r:.2f}\n{len(x):,} sequences", transform=ax.transAxes, fontsize=6.3, va="top")
    ax.set_xlabel(r"Mean $\Delta B_2$ per residue pair (nm$^3$)")
    ax.set_ylabel(r"$\Delta$" + NUTXT + r" (MPIPI-Recharged $-$ CALVADOS-2)", fontsize=6.5)
    lo, hi = np.percentile(y, [0.3, 99.9])
    ax.set_ylim(lo, hi)
    ax.set_xlim(*np.percentile(x, [0.1, 99.9]))
    ax.legend(loc="lower right", frameon=False, fontsize=5.8, handletextpad=0.2, markerscale=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return r


def models(ax, reg):
    pick = [
        ("Length only", "ln N only", "pairs3", 1, "comp"),
        ("Composition, linear", "composition linear (paper)", "composition", 29, "comp"),
        (r"$\Delta B_2$ + length", "dB2 + ln N", "pairs3", 2, "b2"),
        ("Composition, squared + pairwise", "composition squared and pairwise (paper)", "composition", 57, "comp"),
        (r"$B_2$, squared + products", "dB2, B2_C2, ln N with squares and products", "pairs3", 9, "b2"),
        (r"Composition + $B_2$", "composition squared and pairwise + 9-term B2 model", "composition+pairs3", 66, "both"),
    ]
    color = {"comp": P.RULE, "b2": P.MUTED, "both": P.INK}
    rows = []
    for lab, model, w, terms, fam in pick:
        v = reg[(reg["model"] == model) & (reg["weights"] == w)]["cv_r2"]
        assert len(v) == 1, (model, w)
        rows.append((lab, float(v.iloc[0]), terms, fam))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, terms, fam) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=color[fam], edgecolor="none", zorder=2)
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=6, color=INK)
        ax.text(-0.012, yi, f"{lab}  ({terms})", va="center", ha="right", fontsize=6, color=INK)
    ceil = 1 - NOISE_SHARE
    ax.axvline(ceil, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.text(ceil - 0.02, y.max() + 0.75, "Noise\nceiling", fontsize=5.6, color=MUTED, ha="right", va="center")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, y.max() + 1.4)
    ax.set_yticks([])
    ax.set_xlabel(r"Cross-validated $R^2$ of $\Delta$" + NUTXT)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    return rows


CLASS_NAMES = {"aro-aro": "Aromatic-Aromatic", "Arg-aro": "Arg-Aromatic", "Lys-aro": "Lys-Aromatic",
               "His-aro": "His-Aromatic", "charge +/-": "Opposite charges", "charge like": "Like charges",
               "His-acid": "His-acidic", "His-base": "His-basic", "aliph-aliph": "Aliphatic", "GSP-GSP": "Gly/Ser/Pro",
               "other": "Other"}


def dumbbells(ax, classes):
    c = classes[classes["pair_class"] != "total"].sort_values("mean_dB2_per_pair")
    y = np.arange(len(c))[::-1]
    for yi, r in zip(y, c.itertuples()):
        ax.plot([r.mean_B2_C2, r.mean_B2_MR], [yi, yi], color=RULE, lw=1.4, zorder=1, solid_capstyle="round")
        ax.scatter(r.mean_B2_C2, yi, s=14, color=CALVADOS, edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter(r.mean_B2_MR, yi, s=14, color=RECHARGED, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yticks(y, [CLASS_NAMES[k] for k in c["pair_class"]], fontsize=5.6)
    ax.axvline(0, color=RULE, lw=0.6, zorder=0)
    ax.set_xlabel(r"Mean $B_2$ (nm$^3$)")
    ax.tick_params(axis="y", length=0, pad=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(0.98, 0.98, "CALVADOS-2", transform=ax.transAxes, fontsize=5.6, color=CALVADOS_DARK, ha="right",
            va="top")
    ax.text(0.98, 0.92, "MPIPI-Recharged", transform=ax.transAxes, fontsize=5.6, color=RECHARGED_DARK, ha="right",
            va="top")


def potentials(ax, bm, a, b, ylim, show_x=True):
    r = np.linspace(0.42, 2.2, 600)
    for f, col, name in ((bm.u_cal, CALVADOS_DARK, "CALVADOS-2"), (bm.u_mp, RECHARGED_DARK, "MPIPI-Recharged")):
        u = np.sum(f(r, a, b), axis=0)
        ax.plot(r, u, color=col, lw=1.1)
    ax.axhline(0, color=RULE, lw=0.6, zorder=0)
    ax.axhline(-bm.KT, color=RULE, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.set_ylim(*ylim)
    ax.set_xlim(0.4, 2.0)
    ax.text(0.96, 0.92, f"{a}\u2013{b}", transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
            fontweight="bold")
    ax.set_ylabel(r"$u(r)$ (kJ/mol)", fontsize=6)
    if show_x:
        ax.set_xlabel(r"$r$ (nm)")
    else:
        ax.set_xticklabels([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def contacts(ax, cax, feat):
    bio = pd.read_csv(paths.BIO, usecols=["UniProt_ID", "cation_pi_contact_freq",
                                                                  "n_cation_pi_pairs"]).set_index("UniProt_ID")
    fm = pd.read_csv(paths.RECHARGED / "recharged_300K_features.csv",
                     usecols=["UniProt_ID", "cation_pi_contact_freq"]).set_index("UniProt_ID")
    f = feat.set_index("UniProt_ID")
    m = bio.reindex(f.index)["n_cation_pi_pairs"].fillna(0) > 0
    x = f.loc[m, "pairs3_dB2[Arg-aro]"].to_numpy()
    ratio = np.log10((fm.reindex(f.index)["cation_pi_contact_freq"][m] + 1e-4) /
                     (bio.reindex(f.index)["cation_pi_contact_freq"][m] + 1e-4)).to_numpy()
    dnu = f.loc[m, "dnu"].to_numpy()
    lim = 0.08
    hb = ax.hexbin(x, ratio, C=dnu, reduce_C_function=np.mean, gridsize=40, cmap=DIVERGING,
                   norm=TwoSlopeNorm(0.0, -lim, lim), mincnt=3, linewidths=0)
    ax.axhline(0, color=RULE, lw=0.6, zorder=0)
    rr = stats.spearmanr(x, ratio)[0]
    ax.text(0.03, 0.03, f"$\\rho$ = {rr:.2f}\n{m.sum():,} sequences", transform=ax.transAxes, fontsize=5.6,
            ha="left", va="bottom")
    ax.set_xlabel(r"Arg-Aromatic term of mean $\Delta B_2$ (nm$^3$)", fontsize=6)
    ax.set_ylabel("Cation-pi contacts, log$_{10}$ MR / C2", fontsize=6)
    ax.set_xlim(np.percentile(x, 0.5), np.percentile(x, 99.8))
    ax.set_ylim(np.percentile(ratio, 0.5), np.percentile(ratio, 99.5))
    cb = plt.colorbar(hb, cax=cax, ticks=[-lim, 0, lim])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label(r"Mean $\Delta$" + NUTXT, fontsize=5.6, labelpad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2dir", default=str(B2DIR))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    d = Path(a.b2dir)
    pairs = pd.read_csv(d / "b2_pairs_300K_150mM.csv")
    feat = pd.read_csv(d / "b2_features_per_sequence.csv")
    reg = pd.read_csv(d / "b2_regression.csv")
    assert len(pairs) == 210 and len(feat) == 10282

    import importlib.util
    spec = importlib.util.spec_from_file_location("b2_mechanism", str(paths.ANALYSIS / "mpipi_recharged" / "b2_mechanism.py"))
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    classes = pd.read_csv(d / "b2_pair_classes.csv")

    fig = plt.figure(figsize=(7.0, 5.3))
    axa = fig.add_axes([0.035, 0.53, 0.30, 0.43])
    cax = fig.add_axes([0.345, 0.64, 0.009, 0.22])
    axb = fig.add_axes([0.505, 0.55, 0.19, 0.40])
    axc = fig.add_axes([0.79, 0.775, 0.195, 0.18])
    axd = fig.add_axes([0.79, 0.55, 0.195, 0.18])
    axe = fig.add_axes([0.075, 0.08, 0.215, 0.36])
    axf = fig.add_axes([0.40, 0.08, 0.215, 0.36])
    caxf = fig.add_axes([0.625, 0.16, 0.008, 0.2])
    axg = fig.add_axes([0.84, 0.08, 0.125, 0.36])
    lim = heatmap(axa, cax, pairs)
    dumbbells(axb, classes)
    potentials(axc, bm, "R", "Y", (-5.5, 3.0), show_x=False)
    potentials(axd, bm, "L", "L", (-1.6, 1.2))
    r = scatter(axe, feat)
    rho = contacts(axf, caxf, feat)
    rows = models(axg, reg)
    for x, y, s in ((0.005, 0.985, "a"), (0.39, 0.985, "b"), (0.735, 0.985, "c"), (0.735, 0.755, "d"),
                    (0.005, 0.475, "e"), (0.33, 0.475, "f"), (0.69, 0.475, "g")):
        fig.text(x, y, s, fontsize=9, fontweight="bold", va="top")
    out = Path(a.out).expanduser()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=400)
    print(f"wrote {out}.pdf/.png; colour limit +/-{lim} nm^3; r(dnu, mean dB2) {r:.3f}; "
          f"rho(Arg-aro term, cation-pi ratio) {rho:.3f}")
    for lab, v, terms, fam in rows:
        print(f"  {lab:40s} {terms:3d} terms  CV R2 {v:.3f}")
    top = pairs.reindex(pairs["dB2"].abs().sort_values(ascending=False).index).head(6)
    print("largest |dB2|:", ", ".join(f"{p} {v:+.2f}" for p, v in zip(top["pair"], top["dB2"])))


if __name__ == "__main__":
    main()
