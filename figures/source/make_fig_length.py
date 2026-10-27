"""Figure 5 (JCTC, 2026-10-21): chain length explains the differences between groups in both force fields.

(a) One panel per group: nu against chain length N (log axis), running median and interquartile band for
    CALVADOS-2 (blue) and MPIPI-Recharged (plum); gray dashed line, the running median of all 10,282 sequences
    under CALVADOS-2.
(b) Variance explained by chain length (ln N and (ln N)^2), and the further variance explained by group (six
    indicators), for nu, Rg, Re and degree assortativity in both force fields (paper_numbers_recharged.py).
(c) Group medians of nu within length bins (30-50, 50-70, 70-100, 100-150 residues) in each force field; bar, range
    across groups; the protists of 100-150 residues under MPIPI-Recharged are marked.
Data and styling as make_fig_disagreement.py.

Usage:  python make_fig_length.py [--out output/length]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fig_disagreement as fd  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)
fig1 = fd.fig1
NUTXT, INK, MUTED, RULE, SEQ = fd.NUTXT, fd.INK, fd.MUTED, fd.RULE, fd.SEQ
BINS = ((30, 50), (50, 70), (70, 100), (100, 151))


def r2(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - np.var(y - X @ b) / np.var(y)


def length_group(y, n, g):
    L = np.log(n)
    base = r2(y, [L, L ** 2])
    return base, r2(y, [L, L ** 2] + [(g == k).astype(float) for k in fd.ORDER[1:]]) - base


def running(x, y, edges, min_n=40):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() >= min_n:
            q = np.percentile(y[s], [25, 50, 75])
            out.append((np.sqrt(lo * hi), *q))
    return np.array(out)


def main(out):
    b, mr, fm = fd.load()
    n = b['n_residues'].to_numpy(float)
    g = b['kingdom'].to_numpy()
    nu_c, nu_m = b['nu'].to_numpy(), mr['nu'].to_numpy()

    W, H = 7.0, 4.35
    fig = plt.figure(figsize=(W, H))
    overlay = fig.add_axes([0, 0, 1, 1], zorder=10)
    overlay.set_xlim(0, W)
    overlay.set_ylim(0, H)
    overlay.set_aspect("equal")
    overlay.axis("off")

    # (a) nu against N, one panel per group --------------------------------------------------
    ya, sa = 2.55, 1.02
    edges = np.geomspace(30, 500, 13)
    ref = running(n, nu_c, edges)
    for i, k in enumerate(fd.ORDER):
        x0 = 0.42 + i * 1.10
        ax = fig.add_axes([x0 / W, ya / H, sa / W, sa / H])
        s = g == k
        ax.plot(ref[:, 0], ref[:, 2], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
        for v, col, dark in ((nu_c, fd.CALVADOS, fd.CALVADOS_DARK), (nu_m, fd.RECHARGED, fd.RECHARGED_DARK)):
            rr = running(n[s], v[s], edges)
            ax.fill_between(rr[:, 0], rr[:, 1], rr[:, 3], color=col, alpha=0.28, lw=0, zorder=1)
            ax.plot(rr[:, 0], rr[:, 2], color=dark, lw=1.1, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(30, 500)
        ax.set_ylim(0.42, 0.66)
        ax.set_xticks([30, 100, 300], ["30", "100", "300"])
        ax.minorticks_off()
        ax.set_yticks([0.45, 0.5, 0.55, 0.6, 0.65])
        ax.tick_params(labelsize=5.2)
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(NUTXT, fontsize=6.5)
        ax.set_xlabel("$N$", fontsize=6, labelpad=1)
        fd.clean(ax)
        fig.text((x0 + sa / 2 + 0.10) / W, (ya + sa + 0.05) / H, f"{k} ({int(s.sum()):,})", ha="center",
                 va="bottom", fontsize=6.2, color=INK)
        fig1.lineage_icon(overlay, k, x0 + 0.08, ya + sa + 0.12, 0.15)
    fd.label(fig, 0.02, ya + sa + 0.30, "a", f"{NUTXT} against chain length, by group", W, H)
    xk = 3.30
    for txt, col in (("CALVADOS-2", fd.CALVADOS_DARK), ("MPIPI-Recharged", fd.RECHARGED_DARK),
                     ("Dashed line shows all sequences (CALVADOS-2)", MUTED)):
        fig.text(xk / W, (ya + sa + 0.30) / H, txt, ha="left", va="bottom", fontsize=5.6, color=col)
        xk += 0.07 * len(txt) + 0.18

    # (b) variance explained by length and by group -----------------------------------------
    y2, h = 0.40, 1.55
    top2 = y2 + h + 0.22
    props = [(NUTXT, b['nu'], mr['nu']), ("$R_g$", b['rg'], mr['rg_mean']), ("$R_e$", b['ree'], fm['ree']),
             ("Assortativity", b['degree_assortativity'], fm['degree_assortativity'])]
    axb = fig.add_axes([1.10 / W, y2 / H, 1.75 / W, h / H])
    yy, labels = [], []
    for i, (lab, c2, mrv) in enumerate(props):
        for j, (v, col, ff) in enumerate(((c2, fd.CALVADOS, "C2"), (mrv, fd.RECHARGED, "MR"))):
            y = i * 2.6 + j
            base, extra = length_group(v.to_numpy(float), n, g)
            axb.barh(y, 100 * base, height=0.8, color=col, lw=0, zorder=2)
            axb.barh(y, 100 * extra, left=100 * base, height=0.8, color=INK, lw=0, zorder=3)
            axb.text(100 * (base + extra) + 1.5, y, f"{100 * base:.0f}% + {100 * extra:.1f}%", va="center",
                     fontsize=5.3, color=INK)
            print(f"  {lab:14s} {ff}: length {100 * base:.1f}%  + group {100 * extra:.2f}%")
        yy.append(i * 2.6 + 0.5)
        labels.append(lab)
    axb.set_yticks(yy, labels, fontsize=6.2)
    axb.set_ylim(len(props) * 2.6 - 1.2, -0.8)
    axb.set_xlim(0, 100)
    axb.set_xticks([0, 50, 100])
    axb.set_xlabel("Variance explained (%)")
    axb.tick_params(axis="y", length=0)
    fd.clean(axb, left=False)
    fig.text(1.10 / W, (y2 + h + 0.03) / H, "Length", fontsize=5.6, color=MUTED, va="bottom")
    fig.text(1.52 / W, (y2 + h + 0.03) / H, "+ group (dark)", fontsize=5.6, color=INK, va="bottom")
    fd.label(fig, 0.02, top2, "b", "Length, then group", W, H)

    # (c) group medians at matched length ----------------------------------------------------
    for j, (v, col, dark, name) in enumerate(((nu_c, fd.CALVADOS, fd.CALVADOS_DARK, "CALVADOS-2"),
                                              (nu_m, fd.RECHARGED, fd.RECHARGED_DARK, "MPIPI-Recharged"))):
        x0 = 4.05 + j * 1.50
        ax = fig.add_axes([x0 / W, y2 / H, 1.30 / W, h / H])
        for bi, (lo, hi) in enumerate(BINS):
            s = (n >= lo) & (n < hi)
            meds = {k: np.median(v[s & (g == k)]) for k in fd.ORDER}
            vals = np.array(list(meds.values()))
            ax.plot([bi, bi], [vals.min(), vals.max()], color=col, lw=3.0, alpha=0.45, solid_capstyle="round",
                    zorder=1)
            ax.scatter(np.full(len(vals), bi), vals, s=9, color=dark, lw=0, zorder=3)
            ax.text(bi + 0.16, vals.max() + 0.004, f"{vals.max() - vals.min():.3f}", fontsize=5.0, color=INK,
                    va="bottom", ha="left")
            if j == 1 and bi == 3:
                ax.annotate("Protists", (bi, meds["Protists"]), xytext=(-24, -4), textcoords="offset points",
                            fontsize=5.4, color=dark, ha="right", va="center",
                            arrowprops=dict(arrowstyle="-", color=dark, lw=0.5))
        ax.set_xticks(range(len(BINS)), [f"{lo}–{hi - 1 if hi == 151 else hi}" for lo, hi in BINS], fontsize=5.2)
        ax.set_xlim(-0.5, len(BINS) - 0.3)
        ax.set_ylim(0.48, 0.60)
        ax.set_yticks([0.48, 0.52, 0.56, 0.60])
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(f"Group median {NUTXT}", fontsize=6)
        ax.set_xlabel("$N$ (residues)", fontsize=6)
        ax.set_title(name, fontsize=6.2, color=dark, pad=2)
        fd.clean(ax)
    fd.label(fig, 3.62, top2, "c", "Groups at matched length", W, H)

    out = Path(out).expanduser()
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), dpi=600, **kw)
    fig.savefig(out.with_suffix(".png"), dpi=400, **kw)
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.out("length")))
    main(ap.parse_args().out)
