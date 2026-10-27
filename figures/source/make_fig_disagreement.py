"""Figure 3 (JCTC): where CALVADOS-2 and MPIPI-Recharged disagree.

(a) nu from both force fields for all 10,282 sequences (density, identity line).
(b) Agreement by group: Pearson r of nu, and the mean difference with its 95%
    limits of agreement (Bland-Altman).
(c) dnu (MPIPI-Recharged - CALVADOS-2) against the aromatic fraction (F+W+Y),
    colored by lysine fraction, with running medians for Lys-poor and
    Lys-rich chains.
(d) dnu against the MPIPI-Recharged / CALVADOS-2 ratio of cation-pi contact
    frequencies (sequences with at least one Arg/Lys-aromatic pair).
(e) Per group, the share in the contested class (top decile of f_R + f_aro,
    f_K below the median) and the mean dnu inside it, raw and after the
    composition model (squared and pairwise terms, five-fold cross-validated).
(f) A contested protist, A0A7S2N1D9 (Alexandrium
    andersonii): eight conformers per force field on one scale, Rg over the
    production frames, and its Arg and aromatic positions.

Every number is computed here with the definitions of paper_numbers_recharged.py
(the composition model is that script's code) and printed for checking against
its output. Palette as Fig. 1 and the PAE figure (soft set): CALVADOS-2 blue,
MPIPI-Recharged plum, warm grays for density; sequence-derived quantities in
taupe and, for lysine fraction, a green ramp so that no force-field color is
reused for a sequence property.

Inputs:
  data/bender_bio/BENDER_BIO.csv
  data/mpipi_recharged/recharged_300K_nu.csv
  data/mpipi_recharged/recharged_300K_features.csv
  ./a0a7s2n1d9_conformers.json            (extract_contested_conformers.py)

Usage:  python make_fig_disagreement.py [--out output/crossff_disagreement]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
from scipy import stats
from scipy.stats import gaussian_kde

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import make_fig1_bender as fig1  # noqa: E402  (silhouettes; sets the soft palette)
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

BIO_CSV = paths.BIO
RDIR = paths.RECHARGED
EXAMPLE = "A0A7S2N1D9"
CONFORMERS = HERE / f"{EXAMPLE.lower()}_conformers.json"

import palette as P  # noqa: E402  (shared JCTC scheme after figures4papers, 2026-10-21)
CALVADOS, CALVADOS_DARK = P.C2_FILL, P.C2
RECHARGED, RECHARGED_DARK = P.MR_FILL, P.MR
INK, MUTED, RULE = P.INK, P.MUTED, P.RULE
SEQ, SEQ_DARK = P.SEQ, P.SEQ_DARK        # sequence-derived
ARO = P.ARO                              # aromatic residues
BAND = P.PALE
DENSITY = P.DENSITY
DIVERGING = P.DIVERGING   # red, MPIPI-Recharged more compact; blue, CALVADOS-2
LYS = P.LYS_MAP
ORDER = ["Mammals", "Fungi", "Plants", "Protists", "Bacteria", "Viruses"]
FEATS = ['FCR', 'NCPR', 'SCD', 'SHD', 'lambda_avg', 'hydropathy_avg', 'f_aro', 'f_R', 'f_K', 'f_E', 'f_D', 'f_G', 'f_P',
         'f_S', 'f_Q', 'f_N', 'f_H', 'f_A', 'f_V', 'f_I', 'f_L', 'f_M', 'f_T', 'f_C', 'f_F', 'f_W', 'f_Y']
NUTXT = r"$\mathsf{\nu}$"

plt.rcParams.update({
    "font.family": "Arial",
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
    "xtick.minor.width": 0.4,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.minor.size": 1.2,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "axes.labelsize": 6.6,
    "axes.labelcolor": INK,
    "axes.labelpad": 2.0,
})


def signed(x, digits):
    return f"{x:+.{digits}f}".replace("-", "−")


def minus(s):
    return s.replace("-", "−")


# ------------------------------------------------------------------ data --
def load():
    b = pd.read_csv(BIO_CSV, low_memory=False)
    b = b[b['kingdom'].isin(ORDER)].set_index('UniProt_ID')
    mr = pd.read_csv(RDIR / "recharged_300K_nu.csv").set_index('UniProt_ID').reindex(b.index)
    fm = pd.read_csv(RDIR / "recharged_300K_features.csv", low_memory=False).set_index('UniProt_ID').reindex(b.index)
    assert mr['nu'].notna().all() and (fm['status'] == 'ok').all()
    assert (mr['n_residues'].to_numpy() == b['n_residues'].to_numpy()).all()
    return b, mr, fm


def composition_model(b, d):
    """paper_numbers_recharged.py: standardized composition, charge and ln N with squared and pairwise terms of the
    main axes, ridge 1e-2, five folds (rng 0); returns the cross-validated prediction of dnu."""
    n = b['n_residues'].to_numpy(float)
    X = b[FEATS].copy()
    X['logN'], X['absNCPR'] = np.log(n), b['NCPR'].abs()
    X = ((X - X.mean()) / X.std()).fillna(0).to_numpy()
    key = [FEATS.index(f) for f in ('f_aro', 'f_R', 'f_K', 'hydropathy_avg', 'f_H', 'FCR')] + [X.shape[1] - 2]
    extra = []
    for i, c1 in enumerate(key):
        extra.append(X[:, c1] ** 2)
        extra += [X[:, c1] * X[:, c2] for c2 in key[i + 1:]]
    Xq = np.column_stack([X] + extra)
    y = d.to_numpy()
    folds = np.random.default_rng(0).integers(0, 5, len(y))
    pred = np.zeros(len(y))
    for f in range(5):
        tr, te = folds != f, folds == f
        A = np.column_stack([np.ones(tr.sum()), Xq[tr]])
        w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.column_stack([np.ones(te.sum()), Xq[te]]) @ w
    print(f"composition model, 5-fold CV R2 {1 - np.var(y - pred) / np.var(y):.3f}")
    return pred


def numbers(b, mr, fm):
    g = b['kingdom'].to_numpy()
    d = mr['nu'] - b['nu']
    ra = b['f_R'] + b['f_aro']
    cls = ((ra > ra.quantile(0.9)) & (b['f_K'] < b['f_K'].median())).to_numpy()
    res = d.to_numpy() - composition_model(b, d)
    rows = []
    for k in ORDER + ['All']:
        s = np.ones(len(b), bool) if k == 'All' else g == k
        dk = d[s]
        rows.append(dict(group=k, n=int(s.sum()), r=stats.pearsonr(b.loc[s, 'nu'], mr.loc[s, 'nu'])[0],
                         mean=dk.mean(), lo=dk.mean() - 1.96 * dk.std(), hi=dk.mean() + 1.96 * dk.std(),
                         share=cls[s].mean(), n_cls=int((cls & s).sum()), raw=d.to_numpy()[cls & s].mean(),
                         adj=res[cls & s].mean() if k != 'All' else np.nan,
                         big=(dk.abs() > 0.1).mean()))
    t = pd.DataFrame(rows).set_index('group')
    pairs = b['n_cation_pi_pairs'].fillna(0) > 0
    m = pairs & fm['cation_pi_contact_freq'].notna() & b['cation_pi_contact_freq'].notna()
    ratio = (fm['cation_pi_contact_freq'] + 1e-4) / (b['cation_pi_contact_freq'] + 1e-4)
    t.attrs.update(rho_aro=stats.spearmanr(b['f_aro'], d)[0], rho_cpi=stats.spearmanr(d[m], np.log(ratio[m]))[0],
                   n_cpi=int(m.sum()), cls_n=int(cls.sum()), rest=d[~cls].mean())
    return d, cls, ratio, m, t


# ---------------------------------------------------------------- helpers --
def label(fig, x, y, letter, title, W, H):
    fig.text(x / W, y / H, letter, fontsize=9.5, fontweight="bold", color=INK, va="bottom")
    fig.text((x + 0.17) / W, y / H, title, fontsize=6.8, fontweight="bold", color=INK, va="bottom")


def clean(ax, left=True):
    for side in ("top", "right") + (() if left else ("left",)):
        ax.spines[side].set_visible(False)


def ring(ax, x, y, text="f", dx=6, dy=6):
    ax.scatter([x], [y], s=26, facecolor="none", edgecolor=INK, lw=0.9, zorder=8)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.4,
                fontweight="bold", color=INK, ha="center", va="center", zorder=8)


def running_median(x, y, edges, min_n=25):
    xc, ym = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() >= min_n:
            xc.append(np.median(x[s]))
            ym.append(np.median(y[s]))
    return np.array(xc), np.array(ym)


def row_labels(fig, overlay, ax, W, H, x_text, x_icon, with_all=True):
    """Group names with their Fig. 1 silhouettes, aligned to the rows of ax."""
    for y, key in enumerate(ORDER + (['All'] if with_all else [])):
        _, yf = fig.transFigure.inverted().transform(ax.transData.transform((0, y)))
        yi = yf * H
        fig.text(x_text / W, yf, key, ha="right", va="center", fontsize=6.4, color=INK,
                 fontweight="bold" if key == 'All' else "normal")
        if key != 'All':
            fig1.lineage_icon(overlay, key, x_icon, yi, 0.15)


def conformer_panel(ax, conf, color, dark, seq, half):
    for c in conf[1:]:
        ax.plot(*c.T, color=color, lw=0.55, alpha=0.55, zorder=2, solid_capstyle="round")
    c0 = conf[0]
    ax.plot(*c0.T, color=dark, lw=0.9, zorder=4, solid_capstyle="round")
    for q, aa in zip(c0, seq):
        if aa == "R":
            ax.add_patch(Circle(q, 0.30, facecolor=INK, edgecolor="white", lw=0.3, zorder=6))
        elif aa in "FWY":
            ax.add_patch(Circle(q, 0.36, facecolor=ARO, edgecolor=INK, lw=0.35, zorder=7))
        else:
            ax.add_patch(Circle(q, 0.17, facecolor=dark, edgecolor="none", zorder=5))
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------------------ figure --
def main(out):
    b, mr, fm = load()
    d, cls, ratio, mcpi, t = numbers(b, mr, fm)
    ex = json.load(open(CONFORMERS))
    assert ex["protein"] == EXAMPLE and ex["sequence"] == b.loc[EXAMPLE, 'sequence']
    exc2, exmr = b.loc[EXAMPLE, 'nu'], mr.loc[EXAMPLE, 'nu']
    print(f"example {EXAMPLE}: {b.loc[EXAMPLE, 'organism']}, {int(b.loc[EXAMPLE, 'n_residues'])} aa, "
          f"nu {exc2:.3f} / {exmr:.3f}, fit R2 {b.loc[EXAMPLE, 'nu_fit_r2']:.3f} / {mr.loc[EXAMPLE, 'nu_r2']:.3f}, "
          f"Rg {b.loc[EXAMPLE, 'rg']:.2f} / {mr.loc[EXAMPLE, 'rg_mean']:.2f} nm, in class {cls[b.index.get_loc(EXAMPLE)]}, "
          f"cation-pi ratio {ratio[EXAMPLE]:.1f}")

    W, H = 7.0, 4.72
    fig = plt.figure(figsize=(W, H))
    overlay = fig.add_axes([0, 0, 1, 1], zorder=10)
    overlay.set_xlim(0, W)
    overlay.set_ylim(0, H)
    overlay.set_aspect("equal")
    overlay.axis("off")
    y1, h = 2.72, 1.66      # top row: bottom, height
    y2 = 0.40               # bottom row
    top1, top2 = y1 + h + 0.12, y2 + h + 0.20

    # (a) nu, both force fields ------------------------------------------
    ax = fig.add_axes([0.42 / W, y1 / H, h / W, h / H])
    lim = (0.15, 0.75)
    x_, y_ = b['nu'].to_numpy(), mr['nu'].to_numpy()
    out_a = int(((x_ < lim[0]) | (x_ > lim[1]) | (y_ < lim[0]) | (y_ > lim[1])).sum())
    hb = ax.hexbin(x_, y_, gridsize=42, extent=(*lim, *lim), cmap=DENSITY, norm=LogNorm(1, 300),
                   mincnt=1, linewidths=0.1, edgecolors="face", rasterized=True)
    ax.plot(lim, lim, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=3)
    ring(ax, exc2, exmr, text="")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks([0.2, 0.4, 0.6])
    ax.set_yticks([0.2, 0.4, 0.6])
    ax.set_xlabel(f"{NUTXT}, CALVADOS-2")
    ax.set_ylabel(f"{NUTXT}, MPIPI-Recharged", labelpad=1.5)
    ax.text(0.04, 0.96, f"$r$ = {t.loc['All', 'r']:.2f}\n{t.loc['All', 'n']:,} sequences", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.0, color=INK, linespacing=1.3)
    clean(ax)
    cax = fig.add_axes([(0.42 + h - 0.62) / W, (y1 + 0.17) / H, 0.52 / W, 0.05 / H])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_ticks([1, 10, 100])
    cb.set_ticklabels(["1", "10", "100"])
    cb.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=5.2, length=1.5, pad=1)
    cax.set_title("Sequences", fontsize=5.4, color=MUTED, pad=1.5)
    label(fig, 0.02, top1, "a", f"{NUTXT}, all sequences", W, H)

    # (b) agreement by group ---------------------------------------------
    rows = ORDER + ['All']
    ylim = (len(rows) - 0.4, -0.6)
    axr = fig.add_axes([3.05 / W, y1 / H, 0.56 / W, h / H])
    axl = fig.add_axes([3.78 / W, y1 / H, 0.95 / W, h / H])
    for i, k in enumerate(rows):
        axr.plot([0.6, t.loc[k, 'r']], [i, i], color=RULE, lw=0.8, zorder=1)
        axr.scatter([t.loc[k, 'r']], [i], s=13, color=INK if k == 'All' else MUTED, zorder=3, lw=0)
        axr.text(t.loc[k, 'r'] + 0.02, i, f"{t.loc[k, 'r']:.2f}", va="center", ha="left", fontsize=5.4, color=INK)
        axl.plot([t.loc[k, 'lo'], t.loc[k, 'hi']], [i, i], color=SEQ if k != 'All' else MUTED, lw=2.4,
                 solid_capstyle="butt", zorder=2)
        axl.scatter([t.loc[k, 'mean']], [i], s=10, facecolor="white", edgecolor=INK, lw=0.7, zorder=3)
    for a_ in (axr, axl):
        a_.set_ylim(*ylim)
        a_.set_yticks([])
        a_.axhline(len(ORDER) - 0.5, color=RULE, lw=0.5)
        clean(a_, left=False)
    axr.set_xlim(0.6, 0.95)
    axr.set_xticks([0.6, 0.8])
    axr.set_xlabel(f"$r$ ({NUTXT})")
    axl.axvline(0, color=INK, lw=0.5, zorder=1)
    axl.set_xlim(-0.24, 0.24)
    axl.set_xticks([-0.2, 0, 0.2])
    axl.set_xticklabels(["−0.2", "0", "0.2"])
    axl.set_xlabel(f"Δ{NUTXT}, mean and 95% limits")
    row_labels(fig, overlay, axr, W, H, 2.78, 2.92)
    label(fig, 2.30, top1, "b", "Agreement by group", W, H)

    # (c) dnu vs aromatic fraction, colored by lysine ---------------------
    ax = fig.add_axes([5.22 / W, y1 / H, 1.70 / W, h / H])
    fa, fk = b['f_aro'].to_numpy(), b['f_K'].to_numpy()
    yl = (-0.36, 0.22)
    order = np.argsort(-fk)                      # Lys-poor chains drawn last, on top
    sc = ax.scatter(fa[order], d.to_numpy()[order], c=fk[order], cmap=LYS, norm=Normalize(0, 0.15), s=1.4, lw=0,
                    alpha=0.85, rasterized=True, zorder=2)
    out_c = int(((fa > 0.25) | (d < yl[0]) | (d > yl[1])).sum())
    ax.axhline(0, color=INK, lw=0.5, zorder=1)
    kmed = np.median(fk)
    edges = np.arange(0, 0.2601, 0.02)
    for s, col, lab in ((fk < kmed, LYS(0.0), "Lys-poor"), (fk >= kmed, LYS(0.55), "Lys-rich")):
        xc, ym = running_median(fa[s], d.to_numpy()[s], edges)
        ax.plot(xc, ym, color="white", lw=2.6, zorder=4, solid_capstyle="round")
        ax.plot(xc, ym, color=col, lw=1.3, zorder=5, solid_capstyle="round")
        ax.plot([], [], color=col, lw=1.3, label=lab)
    ring(ax, b.loc[EXAMPLE, 'f_aro'], exmr - exc2, text="")
    ax.set_xlim(0, 0.25)
    ax.set_ylim(*yl)
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_xlabel("Aromatic fraction (F + W + Y)")
    ax.set_ylabel(f"Δ{NUTXT} (MPIPI-Recharged − CALVADOS-2)", labelpad=1.5)
    ax.text(0.97, 0.80, minus(f"ρ = {t.attrs['rho_aro']:.2f}"), transform=ax.transAxes, ha="right", va="top",
            fontsize=6.0, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=5.6, handlelength=1.4, borderaxespad=0.2,
              labelspacing=0.3, title="Running median", title_fontsize=5.4)
    clean(ax)
    cax = fig.add_axes([(5.22 + 1.70 - 0.62) / W, (y1 + h - 0.12) / H, 0.52 / W, 0.05 / H])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.05, 0.10, 0.15])
    cb.set_ticklabels(["0", ".05", ".10", "≥.15"])
    cb.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cax.set_title("Lys fraction", fontsize=5.4, color=MUTED, pad=1.5)
    label(fig, 4.80, top1, "c", "Aromatics and Lysine", W, H)

    # (d) dnu vs cation-pi contact ratio ----------------------------------
    ax = fig.add_axes([0.42 / W, y2 / H, h / W, h / H])
    xr = np.log2(ratio[mcpi].to_numpy())          # binned on log2 of the ratio, labeled as folds
    yd = d[mcpi].to_numpy()
    xl = (-1, 4)
    out_d = int(((xr < xl[0]) | (xr > xl[1]) | (yd < yl[0]) | (yd > yl[1])).sum())
    ax.hexbin(xr, yd, gridsize=(30, 30), extent=(*xl, *yl), cmap=DENSITY, norm=LogNorm(1, 300), mincnt=1,
              linewidths=0.1, edgecolors="face", rasterized=True)
    xc, ym = running_median(xr, yd, np.arange(-1, 4.01, 0.4), min_n=40)
    ax.plot(xc, ym, color="white", lw=2.6, zorder=4)
    ax.plot(xc, ym, color=INK, lw=1.3, zorder=5)
    ax.axhline(0, color=INK, lw=0.5, zorder=3)
    ax.axvline(0, color=INK, lw=0.5, ls=(0, (1.5, 1.5)), zorder=3)
    ring(ax, np.log2(ratio[EXAMPLE]), exmr - exc2, text="")
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xticks([-1, 0, 1, 2, 3, 4])
    ax.set_xticklabels(["½×", "1×", "2×", "4×", "8×", "16×"])
    ax.set_xlabel("Cation–π contacts, MPIPI-Recharged / CALVADOS-2")
    ax.set_ylabel(f"Δ{NUTXT}", labelpad=1.5)
    ax.text(0.97, 0.96, minus(f"ρ = {t.attrs['rho_cpi']:.2f}\n{t.attrs['n_cpi']:,} sequences"),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color=INK, linespacing=1.3)
    clean(ax)
    label(fig, 0.02, top2, "d", "Cation–π contacts", W, H)

    # (e) composition map: mean dnu over Arg + aromatic and Lys fractions -------------------
    ax = fig.add_axes([3.02 / W, y2 / H, 1.62 / W, h / H])
    ra = (b['f_R'] + b['f_aro']).to_numpy()
    xe, ye = (0, 0.42), (0, 0.32)
    lim_e = 0.06
    hb = ax.hexbin(ra, fk, C=d.to_numpy(), reduce_C_function=np.mean, gridsize=(26, 22), extent=(*xe, *ye),
                   cmap=DIVERGING, norm=TwoSlopeNorm(0.0, -lim_e, lim_e), mincnt=6, linewidths=0.1,
                   edgecolors="face", rasterized=True)
    x90, kmed = np.quantile(ra, 0.9), np.median(fk)
    ax.add_patch(plt.Rectangle((x90, 0), xe[1] - x90, kmed, facecolor="none", edgecolor=INK, lw=0.8,
                               ls=(0, (2.5, 1.5)), zorder=5))
    ax.text(x90 + 0.008, kmed + 0.008, "Contested", fontsize=5.6, color=INK, va="bottom", ha="left")
    ring(ax, ra[b.index.get_loc(EXAMPLE)], fk[b.index.get_loc(EXAMPLE)], text="")
    ax.set_xlim(*xe)
    ax.set_ylim(*ye)
    ax.set_xticks([0, 0.2, 0.4])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Arg + Aromatic fraction")
    ax.set_ylabel("Lys fraction", labelpad=1.5)
    clean(ax)
    cax = fig.add_axes([(3.02 + 1.62 - 0.62) / W, (y2 + h - 0.12) / H, 0.52 / W, 0.05 / H])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_ticks([-lim_e, 0, lim_e])
    cb.set_ticklabels([minus(f"{-lim_e:.2f}"), "0", f"{lim_e:.2f}"])
    cb.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cax.set_title(f"Mean Δ{NUTXT}", fontsize=5.4, color=MUTED, pad=1.5)
    label(fig, 2.30, top2, "e", "Where the chemistry sits", W, H)

    # (f) dnu vs hydropathy: the opposite direction -------------------------------------------
    ax = fig.add_axes([5.22 / W, y2 / H, 1.70 / W, h / H])
    hy = b['hydropathy_avg'].to_numpy()
    xf = (-2.6, 0.2)
    out_f = int(((hy < xf[0]) | (hy > xf[1]) | (d < yl[0]) | (d > yl[1])).sum())
    ax.hexbin(hy, d.to_numpy(), gridsize=(30, 30), extent=(*xf, *yl), cmap=DENSITY, norm=LogNorm(1, 300), mincnt=1,
              linewidths=0.1, edgecolors="face", rasterized=True)
    xc, ym = running_median(hy, d.to_numpy(), np.arange(-2.6, 0.21, 0.2), min_n=40)
    ax.plot(xc, ym, color="white", lw=2.6, zorder=4)
    ax.plot(xc, ym, color=INK, lw=1.3, zorder=5)
    ax.axhline(0, color=INK, lw=0.5, zorder=3)
    ring(ax, b.loc[EXAMPLE, 'hydropathy_avg'], exmr - exc2, text="")
    rho_h = stats.spearmanr(hy, d)[0]
    ax.text(0.03, 0.96, minus(f"ρ = {rho_h:+.2f}"), transform=ax.transAxes, ha="left", va="top", fontsize=6.0,
            color=INK)
    ax.text(0.97, 0.96, "MPIPI-Recharged\nmore expanded", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.4, color=MUTED, linespacing=1.2)
    ax.text(0.97, 0.04, "MPIPI-Recharged\nmore compact", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.4, color=MUTED, linespacing=1.2)
    ax.set_xlim(*xf)
    ax.set_ylim(*yl)
    ax.set_xlabel("Mean hydropathy (Kyte–Doolittle)")
    ax.set_ylabel(f"Δ{NUTXT}", labelpad=1.5)
    clean(ax)
    label(fig, 4.80, top2, "f", "Hydrophobic chains", W, H)

    out = Path(out).expanduser()
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), dpi=600, **kw)
    fig.savefig(out.with_suffix(".png"), dpi=400, **kw)
    print("wrote", out.with_suffix(".pdf"))
    print(f"points outside the axes: (a) {out_a}, (c) {out_c}, (d) {out_d}, (f) {out_f}")
    print(f"Spearman dnu vs hydropathy {rho_h:+.3f}")
    print(f"contested class n {t.attrs['cls_n']}, mean dnu {t.loc['All', 'raw']:+.4f}, rest {t.attrs['rest']:+.4f}")
    pd.set_option("display.width", 160)
    print(t.round(4).to_string())
    print(f"Spearman dnu vs f_aro {t.attrs['rho_aro']:+.2f}; vs ln cation-pi ratio {t.attrs['rho_cpi']:+.2f} "
          f"(n {t.attrs['n_cpi']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.out("crossff_disagreement")))
    main(ap.parse_args().out)
