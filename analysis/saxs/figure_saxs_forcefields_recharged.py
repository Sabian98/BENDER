#!/usr/bin/env python3
"""SI force-field comparison against SAXS with MPIPI-Recharged added (2026-10-21).

Same panels as figure_saxs_validation.fig_forcefields (whose styles, rows and helpers are imported), now for
three force fields on the same 77 constructs: (a) Rg, (b) nu_eff, (c) per-construct dnu of each Mpipi model
against CALVADOS-2, (d) dnu by taxonomic group. MPIPI-Recharged values come from recharged_vs_saxs.csv
(validate_saxs_recharged.py). MPIPI-Recharged is plum (#9C6D9E), as in the BENDER main figures.

Usage:  python -B figure_saxs_forcefields_recharged.py --out <dir>   (writes fig_saxs_forcefields.pdf/png)
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # analysis/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_saxs_validation as F  # noqa: E402  (rcParams, colors, load_rows, short_name, protein_key)
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

C_MR, C_MR_DARK = "#9C6D9E", "#6B4570"
FFS = [("CALVADOS-2", F.C_SIM, "#3F6E96", "traj_rg_nm", "nu_traj", "delta_nu"),
       ("MPIPI-GG", F.C_MP, "#A9573F", "mpipi_rg_nm", "nu_mpipi", "delta_nu_mpipi"),
       ("MPIPI-Recharged", C_MR, C_MR_DARK, "recharged_rg_nm", "nu_recharged", "delta_nu_recharged")]


def load(recharged_csv):
    rows = F.load_rows()
    mr = {r["traj_id"]: r for r in csv.DictReader(open(recharged_csv))}
    assert set(mr) == {r["traj_id"] for r in rows} and len(rows) == 77, "constructs differ"
    for r in rows:
        m = mr[r["traj_id"]]
        assert abs(float(m["saxs_rg_nm"]) - r["saxs_rg_nm"]) < 1e-9
        r.update({k: float(m[k]) for k in ("recharged_rg_nm", "nu_recharged", "delta_nu_recharged")})
    return rows


def main(out, recharged_csv):
    pr = load(recharged_csv)
    rx = np.array([r["saxs_rg_nm"] for r in pr]); nx = np.array([r["nu_saxs"] for r in pr])
    col = lambda k: np.array([r[k] for r in pr])  # noqa: E731
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.08, wspace=0.34, hspace=0.42)
    dot = lambda c, s=14: dict(s=s, facecolors=c, edgecolors=F.INK, linewidths=0.3, zorder=3, alpha=0.95)  # noqa: E731
    handle = lambda c: Line2D([0], [0], marker="o", ls="", markerfacecolor=c, markeredgecolor=F.INK,  # noqa: E731
                              markeredgewidth=0.3, markersize=4.6)

    ax = axes[0, 0]
    lim = (0.8, 10.0)
    xx = np.linspace(*lim, 50)
    ax.fill_between(xx, 0.8 * xx, 1.2 * xx, color=F.BAND, zorder=0, lw=0)
    ax.plot(lim, lim, color=F.INK3, lw=0.9, ls="--", zorder=1)
    labels = []
    for name, c, _, rk, _, _ in FFS:
        rs = col(rk)
        ax.scatter(rx, rs, **dot(c))
        labels.append(f"{name}  MARE {100 * np.mean(np.abs(rs - rx) / rx):.1f}%, $r$ = {np.corrcoef(rs, rx)[0, 1]:.3f}")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SAXS $R_g$ (nm)"); ax.set_ylabel("Simulated $R_g$ (nm)")
    ax.legend([handle(f[1]) for f in FFS], labels, loc="lower right", handletextpad=0.2, borderaxespad=0.2, fontsize=6.6)
    F.panel_label(ax, "a")

    ax = axes[0, 1]
    lim = (0.36, 0.72)
    ax.fill_between(lim, [lim[0] - 0.05, lim[1] - 0.05], [lim[0] + 0.05, lim[1] + 0.05], color=F.BAND, zorder=0, lw=0)
    ax.plot(lim, lim, color=F.INK3, lw=0.9, ls="--", zorder=1)
    labels = []
    for name, c, _, _, nk, dk in FFS:
        d = col(dk)
        ax.scatter(nx, col(nk), **dot(c))
        labels.append(f"{name}  $\\langle\\Delta\\mathsf{{\\nu}}\\rangle$ = {d.mean():+.3f} $\\pm$ {d.std(ddof=1):.3f}; "
                      f"{100 * np.mean(np.abs(d) < 0.05):.0f}% < 0.05")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SAXS $\\mathsf{\\nu}_{\\mathrm{eff}}$"); ax.set_ylabel("Simulated $\\mathsf{\\nu}_{\\mathrm{eff}}$")
    ax.legend([handle(f[1]) for f in FFS], labels, loc="lower right", handletextpad=0.2, borderaxespad=0.2, fontsize=6.2)
    F.panel_label(ax, "b")

    ax = axes[1, 0]
    dc = col("delta_nu")
    lim = (-0.11, 0.11)
    ax.add_patch(plt.Rectangle((-0.05, -0.05), 0.10, 0.10, color=F.BAND, zorder=0, lw=0))
    ax.plot(lim, lim, color=F.INK3, lw=0.9, ls="--", zorder=1)
    ax.axhline(0, color=F.RULE_C, lw=0.6, zorder=1); ax.axvline(0, color=F.RULE_C, lw=0.6, zorder=1)
    for name, c, _, _, _, dk in FFS[1:]:
        ax.scatter(dc, col(dk), **dot(c, 15))
    # one label per construct outside |dnu| < 0.05 in any force field, at the point furthest from the origin;
    # hand-placed offsets (points) for the crowded upper-right group
    place = {"SASDQS8": ((5, 0), "left", "center"), "SASDNV3": ((0, 6), "center", "bottom"),
             "SASDC62": ((-5, 0), "right", "center"), "SASDEE2": ((5, -3), "left", "center"),
             "SASDQJ7": ((0, -6), "center", "top")}
    for r in pr:
        ys = {dk: r[dk] for *_, dk in FFS[1:]}
        if max(abs(r["delta_nu"]), *map(abs, ys.values())) < 0.05:
            continue
        y = max(ys.values(), key=abs)
        right = r["delta_nu"] >= 0
        off, ha, va = place.get(r["traj_id"], ((5 if right else -5, 0), "left" if right else "right", "center"))
        ax.annotate(F.short_name(r["protein_name"], 18), (r["delta_nu"], y), xytext=off,
                    textcoords="offset points", fontsize=5.8, color=F.INK2, ha=ha, va=va, zorder=4)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks([-0.10, -0.05, 0, 0.05, 0.10]); ax.set_yticks([-0.10, -0.05, 0, 0.05, 0.10])
    ax.set_xlabel("$\\Delta\\mathsf{\\nu}$, CALVADOS-2"); ax.set_ylabel("$\\Delta\\mathsf{\\nu}$, Mpipi model")
    txt = []
    for name, c, _, _, _, dk in FFS[1:]:
        d = col(dk)
        txt.append(f"{name}: $r$ = {np.corrcoef(dc, d)[0, 1]:.2f}, closer for {100 * np.mean(np.abs(d) < np.abs(dc)):.0f}%")
    ax.legend([handle(f[1]) for f in FFS[1:]], txt, loc="upper left", fontsize=6.3, handletextpad=0.2,
              borderaxespad=0.2)
    F.panel_label(ax, "c")

    ax = axes[1, 1]
    present = [t for t in F.TAXON_ORDER if any(r["taxon"] == t for r in pr)]
    ax.axvspan(-0.05, 0.05, color=F.BAND, zorder=0, lw=0)
    ax.axvline(0, color=F.INK3, lw=0.9, ls="--", zorder=1)
    rng = np.random.default_rng(7)
    for yi, t in enumerate(present):
        sub = [r for r in pr if r["taxon"] == t]
        for off, (name, c, dark, _, _, dk) in zip((-0.26, 0.0, 0.26), FFS):
            v = np.array([r[dk] for r in sub])
            ax.scatter(v, yi + off + rng.uniform(-0.07, 0.07, len(v)), s=9, facecolors=c, edgecolors=F.INK,
                       linewidths=0.25, zorder=3)
            ax.plot([v.mean()] * 2, [yi + off - 0.11, yi + off + 0.11], color=dark, lw=1.5, zorder=4,
                    solid_capstyle="round")
        ax.text(0.125, yi, f"{len(sub)} ({len({F.protein_key(r) for r in sub})})", va="center", ha="left",
                fontsize=6.8, color=F.INK2)
    ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=7.5)
    ax.set_xlim(-0.12, 0.12); ax.set_ylim(len(present) - 0.5, -0.5)
    ax.set_xlabel("$\\Delta\\mathsf{\\nu} = \\mathsf{\\nu}_{\\mathrm{sim}} - \\mathsf{\\nu}_{\\mathrm{SAXS}}$")
    ax.text(0.125, -0.95, "constructs\n(proteins)", fontsize=6.3, color=F.INK3, ha="left", va="bottom", linespacing=1.1)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend([handle(f[1]) for f in FFS], [f[0] for f in FFS], loc="lower left", fontsize=6.6, handletextpad=0.2,
              borderaxespad=0.2, bbox_to_anchor=(-0.02, 1.0), ncol=3, columnspacing=0.8)
    F.panel_label(ax, "d")

    out = Path(out)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_saxs_forcefields.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "fig_saxs_forcefields.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--recharged", default=str(F.BENCH / "recharged_vs_saxs.csv"))
    a = ap.parse_args()
    main(a.out, a.recharged)
