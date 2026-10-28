"""Figure 1 (JCTC, 2026-10-21): BENDER at a glance, from one protein to the whole dataset (IDRome-style layout).

(a) The dataset on a schematic tree of life: sequences per group (one bar; every sequence has both force fields),
    split into the three most frequent organisms of the group (named above the bar) and the rest; organisms; and
    the MPIPI-GG subset as a thin bar below.
(b) Worked example, N0AZA6 (Bacteria, 74 residues): its sequence colored by the chemistry the paper turns on
    (arginine, aromatic, lysine, acidic), and eight production-half conformers from each force field on one scale.
(c) Its contact maps: Calpha-Calpha contact frequency (< 0.8 nm), CALVADOS-2 above the diagonal and MPIPI-Recharged
    below.
(d) Its Rg distributions in both force fields.
(e) nu over all 10,282 sequences in both force fields (log counts).
(f) nu by group in both force fields (split violins: CALVADOS-2 above, MPIPI-Recharged below each row).
(g) SAXS validation: simulated against measured Rg for the 77 constructs in the three force fields.

The worked-example data are extracted once from the trajectories (production half, as in every BENDER analysis)
and cached in ./n0aza6_overview.npz; delete it to re-extract. Palette, silhouettes and the tree come from
make_fig1_bender.py (soft palette: CALVADOS-2 blue, MPIPI-Recharged plum, MPIPI-GG peach).

Usage:  python make_fig1_overview.py [--out output/BENDER]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fig1_bender as fig1  # noqa: E402  (silhouettes, tree, counts)
import palette as P  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

BIO_CSV = paths.BIO
MR_NU = paths.RECHARGED / "recharged_300K_nu.csv"
SAXS_DIR = paths.SAXS
EXAMPLE, EXAMPLE_GROUP = "N0AZA6", "Bacteria"
CACHE = HERE / "n0aza6_overview.npz"
OUT = paths.out("BENDER")
ORDER = fig1.ORDER

C2, C2D, MR, MRD, GG = P.C2_FILL, P.C2, P.MR_FILL, P.MR, P.GG
INK, MUTED, RULE = P.INK, P.MUTED, P.RULE
BAR = P.SEQ                                      # sequences per group (both force fields)
TOP_ORGS = 3
ORG_SHADES = ("#8E8D8D", "#A7A6A6", "#BDBCBC")     # the three most frequent organisms, darkest first
VIRUS_ABBR = {"Human immunodeficiency virus type 1": "HIV-1", "Hepatitis delta virus": "HDV",
              "Epstein-Barr virus": "EBV"}
RES_COL = {"R": P.ARG, "K": P.LYS_C, "D": P.ACID, "E": P.ACID}
for a in "FWY":
    RES_COL[a] = P.ARO
OTHER_RES = P.OTHER_RES
C2_MAP, MR_MAP = P.C2_MAP, P.MR_MAP
NUTXT = r"$\mathsf{\nu}$"

plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "xtick.major.size": 2, "ytick.major.size": 2,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5})


# ---------------------------------------------------------------- data
def extract_example():
    """Conformers, Rg per frame and contact frequencies of N0AZA6 in both force fields (cached)."""
    if CACHE.exists():
        return dict(np.load(CACHE, allow_pickle=True))
    import mdtraj as md
    import extract_contested_conformers as ecc
    out = {}
    runs = {"c2": (ecc.CALVADOS_RUNS / EXAMPLE_GROUP / EXAMPLE, None),
            "mr": (ecc.RECHARGED_RUNS / EXAMPLE, None)}
    for key, (run, _) in runs.items():
        unwrap = ecc.box_nm(74) if key == "mr" else None
        c, seq = ecc.conformers(run, EXAMPLE, unwrap_box=unwrap)
        traj = md.load(str(run / f"{EXAMPLE}.dcd"), top=str(run / "top.pdb"))
        xyz = traj.xyz[traj.n_frames // 2:].astype(float)
        if unwrap is not None and ecc.bc.max_bond(xyz) > 1.0:
            xyz = ecc.bc.unwrap_chain(xyz, unwrap)
        sub = xyz[np.linspace(0, len(xyz) - 1, 2000).astype(int)]
        d = np.linalg.norm(sub[:, :, None, :] - sub[:, None, :, :], axis=-1)
        out[f"{key}_contacts"] = (d < 0.8).mean(0)
        out[f"{key}_rg"] = np.array(c["rg_frames_nm"])
        out[f"{key}_conf"] = np.array(c["conformers_nm"])
        out["sequence"] = np.array(seq)
    np.savez_compressed(CACHE, **out)
    return out


def load_nu():
    b = pd.read_csv(BIO_CSV, usecols=["UniProt_ID", "kingdom", "nu"])
    b = b[b["kingdom"].isin(ORDER)].set_index("UniProt_ID")
    m = pd.read_csv(MR_NU, usecols=["UniProt_ID", "nu"]).set_index("UniProt_ID").reindex(b.index)
    b["nu_mr"] = m["nu"]
    assert b["nu_mr"].notna().all() and len(b) == 10282
    return b


def load_saxs():
    d = pd.read_csv(SAXS_DIR / "traj_vs_saxs_rg.csv")
    r = pd.read_csv(SAXS_DIR / "recharged_vs_saxs.csv", usecols=["traj_id", "recharged_rg_nm"])
    d = d.merge(r, on="traj_id", how="left")
    assert len(d) == 77 and d[["traj_rg_nm", "mpipi_rg_nm", "recharged_rg_nm"]].notna().all().all()
    return d


# ---------------------------------------------------------------- panels
def label(fig, x, y, s):
    fig.text(x, y, s, fontsize=9, fontweight="bold", va="top", ha="left")


def tidy(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def load_top_organisms(n=TOP_ORGS):
    """The n most frequent organisms (NCBI taxonomy identifiers) of each group: short name and sequence count."""
    bio = pd.read_csv(BIO_CSV, usecols=["UniProt_ID", "kingdom", "organism", "taxonomy_id"], low_memory=False)
    top = {}
    for g, s in bio[bio["kingdom"].isin(ORDER)].groupby("kingdom"):
        t = (s.groupby("taxonomy_id").agg(n=("UniProt_ID", "size"), name=("organism", "first"))
             .sort_values("n", ascending=False).head(n))
        top[g] = [(short_name(r.name, g == "Viruses"), int(r.n)) for r in t.itertuples()]
    return top


def short_name(name, virus):
    """'Oryza sativa subsp. japonica' -> 'O. sativa'; viruses by their usual abbreviation."""
    name = re.sub(r"\s*\(.*$", "", str(name)).strip()
    if virus:
        return VIRUS_ABBR.get(name, name)
    words = name.split()
    return f"{words[0][0]}. {words[1]}" if len(words) > 1 else name


def panel_dataset(ax, seqs, orgs, mpipi, tot, top_orgs, size_in):
    W, H = size_in                                     # axes size in inches: data units = inches, aspect equal
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    top, step = H - 0.52, 0.31
    rows = {k: top - i * step for i, k in enumerate(ORDER)}
    fig1.tree(ax, rows, leaf_x=0.50)
    xb0, xb1 = 1.50, 2.45
    vmax = max(seqs.values())
    scale = (xb1 - xb0) / vmax
    y0, hb = -0.075, 0.14                              # bar bottom and height, relative to the row
    for k, y in rows.items():
        fig1.lineage_icon(ax, k, 0.66, y, 0.25)
        ax.text(0.86, y, k, va="center", ha="left", fontsize=6.6)
        # the bar is split into the group's most frequent organisms (darkest first) and all others
        x = xb0
        for (name, n), colour in zip(top_orgs[k], ORG_SHADES):
            ax.add_patch(Rectangle((x, y + y0), n * scale, hb, facecolor=colour, edgecolor="white", lw=0.4,
                                   zorder=3))
            x += n * scale
        w = seqs[k] * scale
        ax.add_patch(Rectangle((x, y + y0), xb0 + w - x, hb, facecolor=BAR, edgecolor="white", lw=0.4, zorder=3))
        ax.text(xb0, y + y0 + hb + 0.045, ", ".join(name for name, _ in top_orgs[k]), va="center", ha="left",
                fontsize=4.8, color=P.DARK, fontstyle="normal" if k == "Viruses" else "italic")
        ax.add_patch(Rectangle((xb0, y + y0 - 0.047), mpipi[k] * scale, 0.032, facecolor=GG, edgecolor="none",
                               zorder=3))
        ax.text(xb0 + w + 0.04, y + 0.02, f"{seqs[k]:,}", va="center", fontsize=6.2)
        ax.text(xb0 + w + 0.04, y - 0.085, f"{orgs[k]} organisms", va="center", fontsize=4.9, color=MUTED)
    ax.text(xb0, rows["Viruses"] - 0.30, f"{tot['seqs']:,} IDPs, {tot['orgs']:,} organisms", fontsize=6.6,
            fontweight="bold", va="center")
    ly = H - 0.10
    for i, colour in enumerate((*ORG_SHADES, BAR)):
        ax.add_patch(Rectangle((xb0 + i * 0.06, ly - 0.045), 0.06, 0.09, facecolor=colour, edgecolor="white",
                               lw=0.4))
    ax.text(xb0 + 0.27, ly, "Top three organisms, then the rest", fontsize=5.4, va="center", color=MUTED)
    ax.add_patch(Rectangle((xb0, ly - 0.156), 0.24, 0.032, facecolor=GG, edgecolor="none"))
    ax.text(xb0 + 0.27, ly - 0.14, f"Also MPIPI-GG ({tot['mpipi']})", fontsize=5.4, va="center", color=MUTED)
    ax.axis("off")


def panel_example(ax_seq, ax_c2, ax_mr, ex):
    seq = str(ex["sequence"])
    n = len(seq)
    per = 37
    for i, aa in enumerate(seq):
        r, c = divmod(i, per)
        ax_seq.add_patch(Rectangle((c, -r), 0.92, 0.82, facecolor=RES_COL.get(aa, OTHER_RES), edgecolor="none"))
        ax_seq.text(c + 0.46, -r + 0.41, aa, fontsize=3.6, ha="center", va="center",
                    color="white" if aa == "R" else INK, family="DejaVu Sans Mono")
    ax_seq.set_xlim(-0.2, per + 0.2)
    ax_seq.set_ylim(-(n // per) - 0.3, 1.0)
    ax_seq.axis("off")
    keys = [("R", "Arg", 0.0), ("F", "Aromatic", 4.6), ("K", "Lys", 12.2), ("D", "Acidic", 16.4)]
    for aa, name, x in keys:
        ax_seq.add_patch(Rectangle((x, 1.25), 0.8, 0.55, facecolor=RES_COL[aa], edgecolor="none", clip_on=False))
        ax_seq.text(x + 1.1, 1.52, name, fontsize=5.2, va="center", color=MUTED, clip_on=False)
    ax_seq.text(0.0, 2.55, f"{EXAMPLE}, Bacteria, {n} residues", fontsize=6.0, ha="left", va="center",
                clip_on=False)
    lim = max(np.abs(ex["c2_conf"]).max(), np.abs(ex["mr_conf"]).max()) * 1.05
    for ax, key, col, dark, name in ((ax_c2, "c2", C2, C2D, "CALVADOS-2"), (ax_mr, "mr", MR, MRD, "MPIPI-Recharged")):
        conf = ex[f"{key}_conf"]
        for k, c in enumerate(conf[1:], 1):
            ax.plot(c[:, 0], c[:, 1], color=col, lw=0.5, alpha=0.55, zorder=1)
        ref = conf[0]
        ax.plot(ref[:, 0], ref[:, 1], color=dark, lw=0.9, zorder=3)
        ax.scatter(ref[:, 0], ref[:, 1], s=2.2, color=dark, zorder=4, linewidths=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.5, -0.02, name, transform=ax.transAxes, ha="center", va="top", fontsize=6, color=dark)
    ax_c2.plot([-lim * 0.9, -lim * 0.9 + 2.0], [-lim * 0.92] * 2, color=INK, lw=0.8)
    ax_c2.text(-lim * 0.9 + 1.0, -lim * 0.86, "2 nm", fontsize=5.2, ha="center", va="bottom")


def panel_contacts(ax, cax1, cax2, ex):
    a, b = ex["c2_contacts"], ex["mr_contacts"]
    n = len(a)
    iu = np.triu_indices(n, 3)
    il = np.tril_indices(n, -3)
    up = np.full((n, n), np.nan)
    lo = np.full((n, n), np.nan)
    up[iu], lo[il] = a[iu], b[il]
    vmin = 1e-3
    vmax = max(np.nanmax(up), np.nanmax(lo))
    norm = LogNorm(vmin, vmax)
    i1 = ax.imshow(np.clip(up, vmin, None), cmap=C2_MAP, norm=norm, origin="lower", interpolation="nearest")
    i2 = ax.imshow(np.clip(lo, vmin, None), cmap=MR_MAP, norm=norm, origin="lower", interpolation="nearest")
    ax.plot([0, n - 1], [0, n - 1], color=RULE, lw=0.6)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Residue")
    ax.set_xticks([0, 35, 70], ["1", "36", "71"])
    ax.set_yticks([0, 35, 70], ["1", "36", "71"])
    ax.text(0.04, 0.96, "MPIPI-Recharged", transform=ax.transAxes, fontsize=5.4, color=MRD, va="top")
    ax.text(0.96, 0.04, "CALVADOS-2", transform=ax.transAxes, fontsize=5.4, color=C2D, ha="right")
    for cax, im in ((cax1, i1), (cax2, i2)):
        cb = plt.colorbar(im, cax=cax, ticks=[1e-3, 1e-2, 1e-1])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=4.8, length=1.5, pad=1)
    cax1.figure.text(cax1.get_position().x1 + 0.036, (cax2.get_position().y0 + cax1.get_position().y1) / 2,
                     "Contact frequency", rotation=90, fontsize=5.6, va="center", ha="center")
    tidy(ax, ("left", "bottom", "top", "right"))


def panel_rg(ax, ex):
    bins = np.linspace(1.0, 4.5, 50)
    for key, col, name in (("c2", C2D, "CALVADOS-2"), ("mr", MRD, "MPIPI-Recharged")):
        h, e = np.histogram(ex[f"{key}_rg"], bins=bins, density=True)
        x = 0.5 * (e[1:] + e[:-1])
        ax.fill_between(x, h, color=col, alpha=0.18, lw=0)
        ax.plot(x, h, color=col, lw=1.0, label=name)
        ax.axvline(np.median(ex[f"{key}_rg"]), color=col, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel(r"$R_g$ (nm)")
    ax.set_ylabel(r"$P(R_g)$")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(frameon=False, fontsize=5.4, loc="upper right", handlelength=1.2, borderaxespad=0.1)
    ax.text(0.98, 0.74, f"{EXAMPLE}", transform=ax.transAxes, fontsize=5.4, color=MUTED, ha="right")
    ax.set_yticks([])
    tidy(ax, ("bottom",))


def panel_nu_hist(ax, b):
    bins = np.linspace(0.20, 0.74, 37)
    for col, dark, name in (("nu", C2D, "CALVADOS-2"), ("nu_mr", MRD, "MPIPI-Recharged")):
        h, e = np.histogram(b[col], bins=bins)
        ax.stairs(np.maximum(h, 0.8), e, color=dark, lw=1.0, label=name, baseline=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3000)
    for v in (0.5, 0.588):
        ax.axvline(v, color=RULE, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.set_xlabel(NUTXT)
    ax.set_ylabel("Sequences")
    ax.text(0.03, 0.97, f"All {len(b):,}", transform=ax.transAxes, fontsize=5.6, va="top", color=MUTED)
    tidy(ax)


def panel_groups(ax, b):
    rows = {k: len(ORDER) - 1 - i for i, k in enumerate(ORDER)}
    grid = np.linspace(0.3, 0.75, 200)
    from scipy.stats import gaussian_kde
    for k, y in rows.items():
        s = b[b["kingdom"] == k]
        for col, sign, fill, dark in (("nu", 1, C2, C2D), ("nu_mr", -1, MR, MRD)):
            v = s[col].clip(0.3, 0.75).to_numpy()
            dens = gaussian_kde(v, bw_method=0.25)(grid)
            dens = 0.42 * dens / dens.max()
            ax.fill_between(grid, y, y + sign * dens, color=fill, alpha=0.85, lw=0)
            med = np.median(s[col])
            ax.plot([med, med], [y, y + sign * 0.36], color=dark, lw=0.8)
        ax.text(0.302, y + 0.18, k, fontsize=5.6, va="center", ha="left", color=INK)
    for v in (0.5, 0.588):
        ax.axvline(v, color=RULE, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.set_yticks([])
    ax.set_xlim(0.3, 0.72)
    ax.set_ylim(-0.6, len(ORDER) - 0.45)
    ax.set_xlabel(NUTXT)
    tidy(ax, ("bottom",))


def panel_saxs(ax, d):
    exp = d["saxs_rg_nm"].to_numpy()
    lim = (1.0, 9.6)
    ax.plot(lim, lim, color=RULE, lw=0.7, ls=(0, (3, 2)), zorder=0)
    for col, c, m, name in (("mpipi_rg_nm", GG, "^", "MPIPI-GG"), ("recharged_rg_nm", MRD, "s", "MPIPI-Recharged"),
                            ("traj_rg_nm", C2D, "o", "CALVADOS-2")):
        sim = d[col].to_numpy()
        mare = 100 * np.mean(np.abs(sim - exp) / exp)
        ax.scatter(exp, sim, s=9, marker=m, facecolor=c, edgecolor="white", linewidth=0.3, zorder=3,
                   label=f"{name}, {mare:.1f}%")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R_g$, SAXS (nm)")
    ax.set_ylabel(r"$R_g$, simulation (nm)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, fontsize=4.9, loc="lower right", handletextpad=0.05,
              borderaxespad=0.05)
    ax.text(0.03, 0.97, "77 SAXS constructs\nMean $R_g$ error in legend", transform=ax.transAxes, fontsize=5.2,
            color=MUTED, va="top", linespacing=1.3)
    tidy(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    seqs, orgs, mpipi, tot = fig1.load_counts()
    top_orgs = load_top_organisms()
    ex = extract_example()
    b = load_nu()
    d = load_saxs()

    fig = plt.figure(figsize=(7.0, 5.4))
    pos_a = [0.0, 0.53, 0.43, 0.44]
    ax_a = fig.add_axes(pos_a)
    ax_seq = fig.add_axes([0.45, 0.835, 0.29, 0.085])
    ax_c2 = fig.add_axes([0.445, 0.545, 0.15, 0.29])
    ax_mr = fig.add_axes([0.595, 0.545, 0.15, 0.29])
    ax_c = fig.add_axes([0.79, 0.60, 0.145, 0.30])
    cax1 = fig.add_axes([0.942, 0.755, 0.008, 0.145])
    cax2 = fig.add_axes([0.942, 0.60, 0.008, 0.145])
    ax_d = fig.add_axes([0.055, 0.09, 0.17, 0.34])
    ax_e = fig.add_axes([0.29, 0.09, 0.19, 0.34])
    ax_f = fig.add_axes([0.53, 0.09, 0.19, 0.38])
    ax_g = fig.add_axes([0.79, 0.09, 0.19, 0.34])

    panel_dataset(ax_a, seqs, orgs, mpipi, tot, top_orgs, (pos_a[2] * 7.0, pos_a[3] * 5.4))
    panel_example(ax_seq, ax_c2, ax_mr, ex)
    panel_contacts(ax_c, cax1, cax2, ex)
    panel_rg(ax_d, ex)
    panel_nu_hist(ax_e, b)
    panel_groups(ax_f, b)
    panel_saxs(ax_g, d)
    for x, y, s in ((0.005, 0.985, "a"), (0.44, 0.985, "b"), (0.765, 0.985, "c"), (0.005, 0.475, "d"),
                    (0.245, 0.475, "e"), (0.505, 0.475, "f"), (0.745, 0.475, "g")):
        label(fig, x, y, s)
    out = Path(a.out).expanduser()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=400)
    print(f"wrote {out}.pdf/.png")
    print(f"N0AZA6 median Rg: CALVADOS-2 {np.median(ex['c2_rg']):.2f} nm, MPIPI-Recharged {np.median(ex['mr_rg']):.2f} nm")
    for col in ("traj_rg_nm", "mpipi_rg_nm", "recharged_rg_nm"):
        print(col, f"MARE {100 * np.mean(np.abs(d[col] - d['saxs_rg_nm']) / d['saxs_rg_nm']):.1f}%")


if __name__ == "__main__":
    main()
