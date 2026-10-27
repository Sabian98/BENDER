"""Figure (2026-10-23): the Gly/Ser/Pro swap of MPIPI-Recharged on the 77-construct SAXS benchmark.

On constructs independent of CALVADOS-2 training the two force fields are about equally accurate; MPIPI-Recharged
loses only on the Gly/Pro-rich third, which it makes too expanded. Two B2-matched swaps (simulations/controls, set saxs_swap):
Gly/Ser/Pro pairs only (swap_gsp, 6 pairs) and all pairs among G S P A I L M V (swap_gspaliph, 36 pairs).
(a) Mean signed Rg error against SAXS by third of Gly + Pro fraction, for CALVADOS-2, MPIPI-Recharged and both swaps;
    bars, 95% bootstrap intervals over constructs.
(b) Per construct, the change in Rg from each swap against the Gly + Pro fraction.
(c) The six most Gly/Pro-rich constructs: Rg error against SAXS in each model.
Data: simulations/inputs/saxs_benchmark_300K.tsv (SAXS and unswapped Rg) and data/controls/
recharged_saxs_swap_{gsp,gspaliph}.csv (analyze_jctc_runs.py). Styling as make_fig_tests.py.

Usage:  python make_fig_saxs_swap.py [--out output/saxs_swap]
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
# (column, legend label, marker, face, edge, interval line); swaps in lighter steps of the MPIPI-Recharged tan
MODELS = [("calvados2_rg_nm", "CALVADOS-2", "o", P.C2, "white", P.C2),
          ("recharged_rg_nm", "MPIPI-Recharged", "s", P.MR, "white", P.MR),
          ("gsp_rg_nm", "Gly/Ser/Pro swap", "D", P.MR_FILL, P.MR, P.MR_FILL),
          ("gspaliph_rg_nm", "Gly/Ser/Pro + aliphatic swap", "^", P.MR_LIGHT, P.MR, P.MR_LIGHT)]
THIRDS = [("low", "Gly + Pro poor"), ("mid", "Middle"), ("high", "Gly + Pro rich")]
SHORT = {"SASDUS9": "ELP", "SASDUT9": "ELP, labeled", "SASDNV3": "Elastin",       # ELP, elastin-like polypeptide
         "SASDC62": "TRF2 basic domain", "SASDTN2": "hnRNPA1 LCD V5", "P04637": "p53 N-terminus"}


def boot_ci(x, B=4000, seed=0):
    x = np.asarray(x, float)
    bs = np.random.default_rng(seed).choice(x, (B, len(x))).mean(1)
    return np.percentile(bs, [2.5, 97.5])


def load():
    t = pd.read_csv(INPUTS / "saxs_benchmark_300K.tsv", sep="\t").set_index("traj_id")
    for k in ("gsp", "gspaliph"):
        s = pd.read_csv(DATA / f"recharged_saxs_swap_{k}.csv").drop_duplicates("id", keep="last").set_index("id")
        t[f"{k}_rg_nm"] = s.rg.where(s.status == "ok")
    assert t[["gsp_rg_nm", "gspaliph_rg_nm"]].notna().all().all(), "swap runs missing"
    return t


def main(out):
    t = load()
    err = {c: 100 * (t[c] / t.saxs_rg_nm - 1) for c, *_ in MODELS}
    W, H = 7.0, 2.5
    fig = plt.figure(figsize=(W, H))
    y0, h = 0.42, 1.55
    top = y0 + h + 0.38

    # (a) signed error by third ------------------------------------------------------------------
    ax = fig.add_axes([0.95 / W, y0 / H, 1.65 / W, h / H])
    for i, (key, lab) in enumerate(THIRDS):
        idx = t.index[t.gp_third == key]
        ax.axhspan(i - 0.42, i + 0.42, color=P.PALE, lw=0, zorder=0)
        for j, (c, _, mk, fc, ec, lc) in enumerate(MODELS):
            e = err[c][idx]
            lo, hi = boot_ci(e, seed=j)
            yy = i - 0.27 + 0.18 * j
            ax.plot([lo, hi], [yy, yy], color=lc, lw=0.9, zorder=2)
            ax.scatter(e.mean(), yy, s=16, marker=mk, facecolor=fc, edgecolor=ec, lw=0.5, zorder=4)
    ax.axvline(0, color=P.RULE, lw=0.7, zorder=0)
    ax.set_yticks(range(len(THIRDS)), [f"{lab} ({int((t.gp_third == k).sum())})" for k, lab in THIRDS], fontsize=6.0)
    ax.set_ylim(len(THIRDS) - 0.5, -0.55)
    ax.set_xlabel(r"Mean $R_g$ error vs SAXS (%)", fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    fd.clean(ax, left=False)
    for k, (c, lab, mk, fc, ec, _) in enumerate(MODELS):
        xk, yk = 0.95 + (k % 2) * 1.55, top - 0.17 - 0.11 * (k // 2)
        fig.add_artist(plt.Line2D([(xk + 0.02) / W], [yk / H], marker=mk, color=fc, markeredgecolor=ec,
                                  markeredgewidth=0.6, markersize=3.6, transform=fig.transFigure, ls=""))
        fig.text((xk + 0.08) / W, yk / H, lab, fontsize=5.2, color=P.DARK, va="center")
    fd.label(fig, 0.02, top, "a", "Error by Gly + Pro content", W, H)

    # (b) change from each swap against Gly + Pro fraction ------------------------------------------------
    ax = fig.add_axes([3.30 / W, y0 / H, 1.35 / W, h / H])
    for c, lab, mk, fc, ec, _ in MODELS[2:]:
        d = 100 * (t[c] / t.recharged_rg_nm - 1)
        ax.scatter(t.f_GP, d, s=9, marker=mk, facecolor=fc, edgecolor=ec, lw=0.5, zorder=3, alpha=0.9)
    ax.axhline(0, color=P.RULE, lw=0.7, zorder=0)
    ax.set_xlabel("Gly + Pro fraction", fontsize=6.2)
    ax.set_ylabel(r"$R_g$ change from swap (%)", fontsize=6.2)
    fd.clean(ax)
    fd.label(fig, 2.80, top, "b", "Effect of each swap", W, H)

    # (c) the most Gly/Pro-rich constructs -----------------------------------------------------------------
    ax = fig.add_axes([5.75 / W, y0 / H, 1.15 / W, h / H])
    rows = [r for r in t.sort_values("f_GP", ascending=False).index if r in SHORT][:6]
    for i, r in enumerate(rows):
        ax.plot([err["recharged_rg_nm"][r], err["gspaliph_rg_nm"][r]], [i, i], color=P.MR_PALE, lw=1.6, zorder=1,
                solid_capstyle="round")
        for c, _, mk, fc, ec, _ in MODELS:
            ax.scatter(err[c][r], i, s=16, marker=mk, facecolor=fc, edgecolor=ec, lw=0.5, zorder=3)
    ax.axvline(0, color=P.RULE, lw=0.7, zorder=0)
    ax.set_yticks(range(len(rows)), [f"{SHORT[r]} ({int(t.n_res[r])})" for r in rows], fontsize=5.8)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlabel(r"$R_g$ error vs SAXS (%)", fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    fd.clean(ax, left=False)
    fd.label(fig, 4.55, top, "c", "Most Gly/Pro-rich constructs", W, H)

    out = Path(out).expanduser()
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), dpi=600, **kw)
    fig.savefig(out.with_suffix(".png"), dpi=300, **kw)
    print("wrote", out.with_suffix(".pdf"))
    for key, lab in THIRDS:
        s = t.gp_third == key
        print(f"  {lab:15s} n {int(s.sum()):2d} | " + " | ".join(
            f"{m[1]} MARE {err[m[0]][s].abs().mean():.1f}% bias {err[m[0]][s].mean():+.1f}%" for m in MODELS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.out("saxs_swap")))
    main(ap.parse_args().out)
