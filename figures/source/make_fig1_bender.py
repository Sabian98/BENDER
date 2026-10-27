"""Figure 1: BENDER overview.

(a) Composition by lineage. Counts are bar lengths on one shared baseline, so
    the 12x range between lineages is visible. Each bar is split lengthwise
    into a CALVADOS-2 strip over a MPIPI-Recharged strip (every sequence has
    both); the MPIPI-GG subset is a nested bar on the same scale.
(b) How the dataset is built, read left to right.
(c) SAXS validation as headline numbers (a 2x2 grid of big numbers, each
    with a label; no plot): benchmark size, R_g error, R_g correlation and
    the share of constructs within |dnu| < 0.05. Big numbers are CALVADOS-2
    (the dataset's force field); the MPIPI-GG and MPIPI-Recharged values for
    the same 77 constructs sit beneath each metric in small type. Until the
    MPIPI-Recharged SAXS runs exist, their slot shows a red "[pending]", like
    the red placeholders in the text.

Design follows Cleveland & McGill (length on a common baseline over area),
Rougier et al. 2014 (Ten simple rules for better figures), Krzywinski 2013
(consistent icon style; color reserved for meaning) and headline-number
practice (few numbers, each labeled with its context, number large and bold
in the body sans, label small and muted; Schwabish, PolicyViz 2023).

Palette: the "soft" set chosen by the user (PALETTES below): CALVADOS-2
powder blue, MPIPI-GG peach, MPIPI-Recharged plum, SAXS box cream with an
amber edge, warm grays for text and silhouettes. The plum is darker than the
two pastels so that it stays apart from both under simulated protan, deutan
and tritan vision (OKLab dE >= 10; pastel greens and lavenders at their
lightness fall below 6). The supplementary SAXS figures use the same blue
and peach, with a dark warm neutral for experiment. Taxa are identified by
label and a uniform gray silhouette.

Numbers are read from the data so the figure cannot drift:
  data/bender_bio/BENDER_BIO.csv       (10,282 sequences, six groups; kingdom = group)
  data/mpipi_recharged/recharged_300K_nu.csv
                                                           (MPIPI-Recharged, every sequence)
  data/mpipi_gg/mpipi_results_v2.csv    (in_paper_set, restricted to the six groups)
  data/saxs/traj_vs_saxs_rg.csv
                                                           (77 SAXS constructs)
MPIPI-Recharged SAXS values are drawn once traj_vs_saxs_rg.csv has a
recharged_rg_nm column, or a CSV with traj_id and recharged_rg_nm is passed
with --saxs-recharged, for all 77 constructs; until then panel (c) shows the
placeholder.
Silhouettes are the alpha masks of the icons in ./icons (taken from the
previous Fig. 1); the ciliate is drawn here.

Usage:  python make_fig1_bender.py [--out ../BENDER] [--saxs-recharged CSV]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon)
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

HERE = Path(__file__).resolve().parent
ICONS = HERE / "icons"
BIO_CSV = paths.BIO
MPIPI_CSV = paths.GG_RESULTS
RECHARGED_CSV = paths.RECHARGED / "recharged_300K_nu.csv"
SAXS_CSV = paths.SAXS / "traj_vs_saxs_rg.csv"
# Construct and protein counts of the benchmark (bender_benchmark_supplement.md,
# "Agreement by subset", all constructs); constructs are re-checked below.
SAXS_CONSTRUCTS, SAXS_PROTEINS = 77, 53

# Palettes differ only in these values; color always encodes the same things
# (CALVADOS-2, MPIPI-GG, MPIPI-Recharged, the SAXS validation box, neutrals).
# Only "soft" has been checked for color-vision separation.
PALETTES = {
    # github.com/ChenLiu-1996/figures4papers: saturated blue / red / green
    "figures4papers": dict(
        CALVADOS="#3775BA", CALVADOS_DARK="#0F4D92", MPIPI="#B64342",
        RECHARGED="#6B4C9A",
        BAR_EDGE="#0F4D92", INK="#272727", MUTED="#767676",
        SILHOUETTE="#4D4D4D", RULE="#CFCECE", BOX_EDGE="#CFCECE",
        BOX_FILL="#F7F7F7", VALID_FILL="#EEF8EE", VALID_EDGE="#8BCF8B",
        CONTACT="#9FBCE0"),
    # the cream-leaning palette approved for the ICLR 2027 figures
    "soft": dict(
        CALVADOS="#7FA7CB", CALVADOS_DARK="#3F6E96", MPIPI="#E0937A",
        RECHARGED="#9C6D9E",
        BAR_EDGE=None, INK="#33302C", MUTED="#7A746C",
        SILHOUETTE="#6B645B", RULE="#DDD6CB", BOX_EDGE="#E3DBCF",
        BOX_FILL="#FBF8F3", VALID_FILL="#FBF4E6", VALID_EDGE="#DCAA68",
        CONTACT="#B7CCE0"),
    # desaturated slate blue and terracotta on cool grays
    "muted": dict(
        CALVADOS="#5B7FA6", CALVADOS_DARK="#34506F", MPIPI="#C8745A",
        RECHARGED="#7E6A9A",
        BAR_EDGE=None, INK="#2B2B2B", MUTED="#707070",
        SILHOUETTE="#555555", RULE="#D6D6D6", BOX_EDGE="#D9D9D9",
        BOX_FILL="#F8F8F7", VALID_FILL="#F1F5EE", VALID_EDGE="#A9C49A",
        CONTACT="#A9BCD2"),
    # deep teal and ochre
    "teal": dict(
        CALVADOS="#2F7F80", CALVADOS_DARK="#1D5657", MPIPI="#E0A458",
        RECHARGED="#8C5E8F",
        BAR_EDGE=None, INK="#26302F", MUTED="#6E7877",
        SILHOUETTE="#4F5857", RULE="#D3D9D8", BOX_EDGE="#D6DCDB",
        BOX_FILL="#F7F9F9", VALID_FILL="#EEF6F5", VALID_EDGE="#93C3BF",
        CONTACT="#9CC5C4"),
}
import palette as _P  # noqa: E402  (shared JCTC scheme, 2026-10-21)
PALETTES["f4p"] = dict(
    CALVADOS=_P.C2_FILL, CALVADOS_DARK=_P.C2, MPIPI=_P.GG, RECHARGED=_P.MR_FILL,
    BAR_EDGE=None, INK=_P.INK, MUTED=_P.MUTED, SILHOUETTE=_P.DARK, RULE=_P.RULE, BOX_EDGE=_P.RULE,
    BOX_FILL="#F7F7F7", VALID_FILL="#EEF8EE", VALID_EDGE="#8BCF8B", CONTACT=_P.C2_PALE)
DEFAULT_PALETTE = "f4p"    # figures4papers scheme, user request 2026-10-21 (was "soft", chosen 2026-09-26)
PENDING = "#D62728"        # red, like the placeholders in the text


def use_palette(name):
    globals().update(PALETTES[name])
    globals()["CHAIN"] = PALETTES[name]["CALVADOS_DARK"]   # glyphs = CALVADOS-2


use_palette(DEFAULT_PALETTE)

# First residues of BENDER entry N0AZA6 (Bacteria), shown in panel b.
EXAMPLE_SEQ = ("MTAEIVNLNKVRKAR", "ERANRERDAQENRLK")

# Rows follow the tree drawn beside them: the eukaryote groups together,
# then bacteria (the two cellular branches meet at the root), then the
# non-cellular viruses.
ORDER = ["Mammals", "Fungi", "Plants", "Protists", "Bacteria", "Viruses"]
EUKARYOTES = ["Mammals", "Fungi", "Plants", "Protists"]
CONFORMERS = HERE / "n0aza6_conformers.json"   # extract_n0aza6_conformers.py

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
})


def load_counts():
    bio = pd.read_csv(BIO_CSV, usecols=["UniProt_ID", "kingdom", "taxonomy_id",
                                        "n_residues"])
    bio = bio[bio["kingdom"].isin(ORDER)]             # the six BENDER groups
    mp = pd.read_csv(MPIPI_CSV, usecols=["UniProt_ID", "status", "in_paper_set"])
    mp = mp[mp["in_paper_set"] & (mp["status"] == "ok")].merge(
        bio[["UniProt_ID", "kingdom"]], on="UniProt_ID", how="inner")
    mr = pd.read_csv(RECHARGED_CSV, usecols=["UniProt_ID", "n_residues", "nu"])
    mr = mr[mr["nu"].notna()].merge(bio, on="UniProt_ID", how="inner",
                                    suffixes=("", "_bio"))
    assert (mr["n_residues"] == mr["n_residues_bio"]).all(), \
        "MPIPI-Recharged chain length differs from BENDER_BIO"
    seqs = bio["kingdom"].value_counts().to_dict()
    recharged = mr["kingdom"].value_counts().to_dict()
    # bar lengths and the caption say every sequence has both force fields
    assert recharged == seqs, f"MPIPI-Recharged {recharged} != CALVADOS-2 {seqs}"
    orgs = bio.groupby("kingdom")["taxonomy_id"].nunique().to_dict()
    mpipi = mp["kingdom"].value_counts().to_dict()
    totals = dict(seqs=len(bio), orgs=bio["taxonomy_id"].nunique(), mpipi=len(mp),
                  recharged=len(mr))
    return seqs, orgs, mpipi, totals


def load_saxs(recharged_csv=None):
    """(MARE %, Pearson r, % within |dnu| < 0.05) for CALVADOS-2, MPIPI-GG and
    MPIPI-Recharged; MPIPI-Recharged is None until all constructs have it."""
    d = pd.read_csv(SAXS_CSV)
    assert len(d) == SAXS_CONSTRUCTS, f"expected {SAXS_CONSTRUCTS} constructs"
    assert d["mpipi_rg_nm"].notna().all(), "MPIPI-GG missing for some constructs"
    if recharged_csv is not None:
        r = pd.read_csv(recharged_csv, usecols=["traj_id", "recharged_rg_nm"])
        d = d.drop(columns="recharged_rg_nm", errors="ignore").merge(
            r, on="traj_id", how="left")
    have = int(d["recharged_rg_nm"].notna().sum()) if "recharged_rg_nm" in d else 0
    print(f"MPIPI-Recharged SAXS: {have} of {len(d)} constructs")
    exp = d["saxs_rg_nm"].to_numpy()
    out = {"MPIPI-Recharged": None}
    for ff, col in (("CALVADOS-2", "traj_rg_nm"), ("MPIPI-GG", "mpipi_rg_nm"),
                    ("MPIPI-Recharged", "recharged_rg_nm")):
        if col == "recharged_rg_nm" and have < len(d):
            continue
        sim = d[col].to_numpy()
        # nu_eff = ln(Rg/R0)/ln N, so the difference needs no R0
        dnu = np.log(sim / exp) / np.log(d["n_res"].to_numpy())
        out[ff] = (100 * np.mean(np.abs(sim - exp) / exp), np.corrcoef(sim, exp)[0, 1],
                   100 * np.mean(np.abs(dnu) < 0.05))
    return out


# ---------------------------------------------------------------- glyphs --
def silhouette(name):
    """Icon alpha mask filled with one flat color (uniform icon style)."""
    rgba = np.asarray(Image.open(ICONS / f"{name}.png").convert("RGBA")).copy()
    rgba[..., :3] = (np.array(matplotlib.colors.to_rgb(SILHOUETTE)) * 255
                     ).astype(np.uint8)
    return rgba


def place(ax, name, cx, cy, height=None, width=None):
    img = silhouette(name)
    h, w = img.shape[:2]
    if height is None:
        height = width * h / w
    if width is None:
        width = height * w / h
    ax.imshow(img, extent=(cx - width / 2, cx + width / 2,
                           cy - height / 2, cy + height / 2),
              interpolation="lanczos", zorder=6)


def ciliate(ax, cx, cy, R):
    """Slipper-shaped ciliate with a fringe of cilia."""
    t = np.linspace(0, 2 * np.pi, 200)
    x, y = 1.25 * R * np.cos(t), 0.62 * R * np.sin(t) * (1 - 0.18 * np.cos(t))
    y -= 0.12 * R * np.exp(-((t - 1.2) / 0.35) ** 2)          # oral groove
    rot = np.deg2rad(28)
    X, Y = x * np.cos(rot) - y * np.sin(rot), x * np.sin(rot) + y * np.cos(rot)
    ax.add_patch(Polygon(np.c_[cx + X, cy + Y], closed=True,
                         facecolor=SILHOUETTE, lw=0, zorder=6))
    for k in range(0, 200, 8):
        nx, ny = np.gradient(Y)[k], -np.gradient(X)[k]
        norm = np.hypot(nx, ny)
        ax.plot([cx + X[k], cx + X[k] + 0.22 * R * nx / norm],
                [cy + Y[k], cy + Y[k] + 0.22 * R * ny / norm],
                color=SILHOUETTE, lw=0.35, zorder=6)


def lineage_icon(ax, key, cx, cy, size):
    if key == "Mammals":
        place(ax, "human", cx, cy, height=size)
    elif key == "Plants":
        place(ax, "plant", cx, cy, height=size)
    elif key == "Fungi":
        place(ax, "fungi", cx, cy, height=size * 0.85)
    elif key == "Protists":
        ciliate(ax, cx, cy, size * 0.42)
    elif key == "Bacteria":
        place(ax, "bacteria", cx, cy, width=size * 1.5)
    elif key == "Viruses":
        place(ax, "virus", cx, cy, height=size * 0.9)


def chain(seed, width, height, n=60, noise=0.9, smooth=(2.0, 1.5)):
    """Smoothed 2D random walk: a coarse-grained disordered chain. Larger
    noise and less smoothing give a more compact coil."""
    rng = np.random.default_rng(seed)
    ang = gaussian_filter1d(np.cumsum(rng.normal(0, noise, n)), smooth[0])
    pts = gaussian_filter1d(np.c_[np.cos(ang), np.sin(ang)].cumsum(0),
                            smooth[1], axis=0)
    pts -= (pts.max(0) + pts.min(0)) / 2
    span = np.ptp(pts, 0)
    return pts / max(span[0] / width, span[1] / height)


def ensemble_glyph(ax, cx, cy, width=0.66, height=0.37, n_show=5):
    """Eight real CALVADOS-2 conformers of N0AZA6 (the protein whose sequence
    opens panel b), superposed and projected onto their principal plane; the
    median-Rg conformer is drawn as beads."""
    conf = np.array(json.load(open(CONFORMERS))["conformers_nm"])[:n_show]
    pts = conf.reshape(-1, 2)
    centre = (pts.max(0) + pts.min(0)) / 2
    scale = min(width / np.ptp(pts[:, 0]), height / np.ptp(pts[:, 1]))
    conf = (conf - centre) * scale + (cx, cy)
    for c in conf[1:]:
        ax.plot(*c.T, color=CALVADOS, lw=0.7, alpha=0.4, zorder=5,
                solid_capstyle="round", solid_joinstyle="round")
    ax.plot(*conf[0].T, color=CHAIN, lw=1.0, zorder=6,
            solid_joinstyle="round")
    for q in conf[0][::4]:
        ax.add_patch(Circle(q, 0.012, facecolor=CHAIN, edgecolor=BOX_FILL,
                            lw=0.3, zorder=7))


GEOMETRY_SEED = 27


def geometry_glyph(ax, cx, cy, seed=None):
    """Coiled chain with its radius of gyration (dashed circle) and
    end-to-end vector."""
    seed = GEOMETRY_SEED if seed is None else seed
    p = chain(seed, 0.31, 0.22, noise=1.3, smooth=(1.6, 1.2)) + (cx, cy)
    c = p.mean(0)
    rg = np.sqrt(((p - c) ** 2).sum(1).mean())
    ax.add_patch(Circle(c, rg, facecolor="none", edgecolor=MUTED, lw=0.6,
                        ls=(0, (2, 1.5)), zorder=4))
    ax.plot(*p.T, color=CHAIN, lw=0.8, zorder=5)
    for q in p[::4]:
        ax.add_patch(Circle(q, 0.012, facecolor=CHAIN, edgecolor=BOX_FILL,
                            lw=0.3, zorder=6))
    ax.add_patch(FancyArrowPatch(p[0], p[-1],
                                 arrowstyle="<|-|>,head_length=1.6,head_width=0.9",
                                 color=INK, lw=0.55, shrinkA=1.5, shrinkB=1.5,
                                 zorder=7))
    return c, rg, p[0], p[-1]


def network_glyph(ax, cx, cy):
    """Contact network: backbone (dark) plus long-range contacts (light);
    the most connected residue (hub) is drawn larger."""
    nodes = np.array([(-0.17, -0.06), (-0.12, 0.07), (-0.03, 0.11),
                      (0.00, -0.01), (0.07, -0.10), (0.15, -0.04),
                      (0.12, 0.08), (-0.08, -0.11)]) + (cx, cy)
    contacts = [(0, 3), (3, 6), (3, 7), (2, 6), (4, 7)]
    for i, j in contacts:
        ax.plot(*nodes[[i, j]].T, color=CONTACT, lw=0.6, zorder=4)
    ax.plot(*nodes[:7].T, color=CHAIN, lw=0.9, zorder=5)
    ax.plot(*nodes[[7, 0]].T, color=CHAIN, lw=0.9, zorder=5)
    deg = np.zeros(len(nodes), int)
    for i, j in contacts + [(k, k + 1) for k in range(6)] + [(7, 0)]:
        deg[i] += 1
        deg[j] += 1
    for q, k in zip(nodes, deg):
        hub = k == deg.max()
        ax.add_patch(Circle(q, 0.026 if hub else 0.016,
                            facecolor=INK if hub else CHAIN,
                            edgecolor=BOX_FILL, lw=0.4, zorder=6))


def text_width(ax, t):
    """Width of a text artist in data units (1 unit = 1 inch here)."""
    return t.get_window_extent(ax.figure.canvas.get_renderer()).width / ax.figure.dpi


def swatch_text(ax, x, y, color, text, fs, bold=False, bar_h=0.06,
                text_color=INK):
    """Swatch then text, from x; returns the x where the text ends."""
    ax.add_patch(FancyBboxPatch((x, y - bar_h / 2), 0.11, bar_h,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=color, lw=0, zorder=6))
    t = ax.text(x + 0.15, y, text, ha="left", va="center", fontsize=fs,
                color=text_color, fontweight="bold" if bold else "normal",
                zorder=6)
    return x + 0.15 + text_width(ax, t)


def marked_values(ax, x, y, marks, lh, fs=5.9):
    """Stacked small swatch + value lines (the per-force-field values under
    each SAXS headline number), swatches in one column, the block centered
    on x. marks: [(swatch color, text, text color)], first line at y."""
    sw, gap = 0.075, 0.03
    texts = [ax.text(x, y - j * lh, text, ha="left", va="center", fontsize=fs,
                     color=tcol, zorder=6) for j, (_, text, tcol) in enumerate(marks)]
    x0 = x - (sw + gap + max(map(lambda t: text_width(ax, t), texts))) / 2
    for j, ((color, _, _), t) in enumerate(zip(marks, texts)):
        t.set_x(x0 + sw + gap)
        ax.add_patch(FancyBboxPatch((x0, y - j * lh - 0.018), sw, 0.036,
                                    boxstyle="round,pad=0,rounding_size=0.01",
                                    facecolor=color, lw=0, zorder=6))


def panel_label(ax, x, y, letter):
    ax.text(x, y, letter, fontsize=9.5, fontweight="bold", va="top",
            color=INK)


# ---------------------------------------------------------------- panels --
def tree(ax, rows, leaf_x):
    """Schematic tree of life. Eukaryote groups join at one node, which meets
    bacteria at the root; viruses are non-cellular and hang from a dashed stub."""
    kw = dict(color=MUTED, lw=0.7, solid_capstyle="butt", zorder=2)
    root_x, euk_x = 0.10, 0.24
    euk_ys = [rows[k] for k in EUKARYOTES]
    for yk in euk_ys:
        ax.plot([euk_x, leaf_x], [yk, yk], **kw)
    ax.plot([euk_x, euk_x], [min(euk_ys), max(euk_ys)], **kw)
    euk_mid = (min(euk_ys) + max(euk_ys)) / 2
    for yk in (euk_mid, rows["Bacteria"]):
        ax.plot([root_x, euk_x if yk == euk_mid else leaf_x], [yk, yk], **kw)
    ax.plot([root_x, root_x], [euk_mid, rows["Bacteria"]], **kw)
    ax.plot([root_x + 0.07, leaf_x], [rows["Viruses"]] * 2, color=MUTED,
            lw=0.7, ls=(0, (2, 1.5)), zorder=2)
    # beside the upper eukaryote branches, clear of the root connector
    ax.text((root_x + euk_x) / 2 + 0.005, rows["Mammals"] - 0.07, "Eukaryotes",
            rotation=90, ha="center", va="center", fontsize=5.4,
            color=MUTED)
    ax.text(root_x + 0.07, rows["Viruses"] - 0.075, "Non-cellular",
            ha="left", va="center", fontsize=5.0,
            color=MUTED)


def panel_composition(ax, top, seqs, orgs, mpipi, tot):
    icon_x, label_x, bar_x0, bar_max = 0.60, 0.82, 1.48, 2.30
    org_x, mp_x = 4.52, 5.38
    row_h, bar_h, gap = 0.215, 0.155, 0.012
    strip = (bar_h - gap) / 2
    scale = bar_max / max(seqs.values())

    panel_label(ax, 0.02, top + 0.02, "a")
    y = top - 0.05
    x = swatch_text(ax, bar_x0, y, CALVADOS, "CALVADOS-2", 6.4)
    swatch_text(ax, x + 0.13, y, RECHARGED, "MPIPI-Recharged", 6.4)
    ax.text(org_x, y, "organisms", ha="right", va="center", fontsize=6.4,
            color=INK)
    swatch_text(ax, mp_x - 0.60, y, MPIPI, "MPIPI-GG", 6.4)

    y -= 0.04
    rows = {}
    for key in ORDER:
        y -= row_h
        rows[key] = y
        n = seqs[key]
        lineage_icon(ax, key, icon_x, y, 0.16)
        ax.text(label_x, y, key, ha="left", va="center", fontsize=7.2,
                color=INK)
        # every sequence has both force fields: CALVADOS-2 strip over a
        # MPIPI-Recharged strip of the same length (load_counts checks this)
        for y0, color in ((y + gap / 2, CALVADOS), (y - bar_h / 2, RECHARGED)):
            ax.add_patch(FancyBboxPatch(
                (bar_x0, y0), n * scale, strip,
                boxstyle="round,pad=0,rounding_size=0.018",
                facecolor=color, edgecolor=BAR_EDGE or color,
                lw=0.5 if BAR_EDGE else 0, zorder=3))
        if key in mpipi:
            h = bar_h * 0.46
            ax.add_patch(FancyBboxPatch(
                (bar_x0, y - h / 2), mpipi[key] * scale, h,
                boxstyle="round,pad=0,rounding_size=0.012",
                facecolor=MPIPI, edgecolor="white", lw=0.6, zorder=4))
        ax.text(bar_x0 + n * scale + 0.05, y, f"{n:,}", ha="left",
                va="center", fontsize=6.8, color=INK)
        ax.text(org_x, y, f"{orgs[key]:,}", ha="right", va="center",
                fontsize=6.8, color=INK)
        ax.text(mp_x, y, f"{mpipi[key]:,}" if key in mpipi else "–",
                ha="right", va="center", fontsize=6.8,
                color=INK if key in mpipi else MUTED)

    tree(ax, rows, leaf_x=icon_x - 0.12)

    y -= row_h + 0.02
    ax.plot([0.10, 5.40], [y + row_h / 2 + 0.005] * 2, color=RULE, lw=0.6)
    ax.text(label_x, y, "Total", ha="left", va="center", fontsize=7.2,
            fontweight="bold", color=INK)
    ax.text(bar_x0, y, f"{tot['seqs']:,} IDPs", ha="left", va="center",
            fontsize=6.8, fontweight="bold", color=INK)
    ax.text(org_x, y, f"{tot['orgs']:,}", ha="right", va="center",
            fontsize=6.8, fontweight="bold", color=INK)
    ax.text(mp_x, y, f"{tot['mpipi']:,}", ha="right", va="center",
            fontsize=6.8, fontweight="bold", color=INK)
    return y - row_h / 2


def panel_pipeline(ax, top, x0, x1, box_h, tot):
    gap = 0.19
    box_w = (x1 - x0 - 2 * gap) / 3
    panel_label(ax, 0.02, top + 0.19, "b")
    glyph_y, text_y = top - 0.43, top - 0.75
    fs, lh = 5.9, 0.115

    centres = []
    for i in range(3):
        x = x0 + i * (box_w + gap)
        centres.append(x + box_w / 2)
        ax.add_patch(FancyBboxPatch((x, top - box_h), box_w, box_h,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=BOX_FILL, edgecolor=BOX_EDGE,
                                    lw=0.7, zorder=2))
        if i:
            ax.add_patch(FancyArrowPatch(
                (x - gap + 0.035, top - box_h / 2), (x - 0.035, top - box_h / 2),
                arrowstyle="-|>,head_length=2.6,head_width=1.6", color=MUTED,
                lw=0.8, zorder=3))

    def title(i, text):
        ax.text(centres[i], top - 0.11, text, ha="center", va="center",
                fontsize=7.0, fontweight="bold", color=INK)

    def lines(i, texts):
        for k, t in enumerate(texts):
            ax.text(centres[i], text_y - k * lh, t, ha="center", va="center",
                    fontsize=fs, color=INK)

    title(0, "Curation")
    for k, s in enumerate(EXAMPLE_SEQ):
        ax.text(centres[0], glyph_y + 0.055 - k * 0.11,
                s + ("…" if k else ""), ha="center", va="center",
                fontsize=6.0, family="Menlo", color=CHAIN)
    lines(0, ["MobiDB, fully disordered", "CD-HIT 40%, all lineages",
              f"{tot['seqs']:,} IDPs"])

    title(1, "Simulation")
    ensemble_glyph(ax, centres[1], glyph_y)
    items = [(CALVADOS, "CALVADOS-2: all"),
             (RECHARGED, "MPIPI-Recharged: all"),
             (MPIPI, f"MPIPI-GG: {tot['mpipi']:,}")]
    probes = [ax.text(0, 0, t, fontsize=fs) for _, t in items]
    block = 0.15 + max(text_width(ax, p) for p in probes)   # swatch + longest
    for p in probes:
        p.remove()
    assert block < box_w - 0.06, f"simulation legend {block:.2f} in > box {box_w:.2f} in"
    for k, (col, txt) in enumerate(items):
        swatch_text(ax, centres[1] - block / 2, text_y - k * lh, col, txt, fs,
                    bar_h=0.05)

    title(2, "Ensemble properties")
    gx, nx = centres[2] - 0.27, centres[2] + 0.28
    c, rg, e0, e1 = geometry_glyph(ax, gx + 0.03, glyph_y + 0.005)
    # R_g to the left of its circle; R_e just past the arrow's lower end,
    # both in empty space clear of the chain and the box title
    ax.text(c[0] - rg - 0.012, c[1], r"$R_g$", fontsize=5.6, color=MUTED,
            ha="right", va="center", zorder=8)
    low, high = (e0, e1) if e0[1] < e1[1] else (e1, e0)
    ax.text(low[0] + 0.03, low[1] - 0.03, r"$R_e$", fontsize=5.6,
            color=MUTED, ha="left", va="center", zorder=8)
    network_glyph(ax, nx, glyph_y + 0.02)
    for x, head, sub in [(gx, "5 geometric", r"$R_g$ $R_e$ $\mathsf{\nu}$ $\Delta$ $A_0$"),
                         (nx, "5 network", "topology")]:
        ax.text(x, text_y + 0.02, head, ha="center", va="center", fontsize=fs,
                color=INK)
        ax.text(x, text_y + 0.02 - lh * 0.95, sub, ha="center", va="center",
                fontsize=5.5, color=MUTED)
    ax.text(centres[2], text_y - 2 * lh, "+ π–π, cation–π contacts",
            ha="center", va="center", fontsize=fs, color=INK)


def panel_saxs(ax, top, x0, x1, box_h, recharged_csv=None):
    """SAXS validation as four labeled headline numbers (CALVADOS-2), with
    the MPIPI-GG and MPIPI-Recharged values for the same constructs beneath
    each metric (MPIPI-Recharged as a red placeholder until it exists)."""
    stats = load_saxs(recharged_csv)
    fmts = ("{:.1f}%", "{:.2f}", "{:.0f}%")         # MARE, r, within
    small = []                                      # per metric: [(color, text, text color)]
    for m, f in enumerate(fmts):
        small.append([(MPIPI, f.format(stats["MPIPI-GG"][m]), MUTED)])
        mr = stats["MPIPI-Recharged"]
        small[-1].append((RECHARGED, f.format(mr[m]), MUTED) if mr else
                         (RECHARGED, "[pending]", PENDING))
    mare, r, within = stats["CALVADOS-2"]
    panel_label(ax, x0 - 0.17, top + 0.19, "c")
    ax.add_patch(FancyBboxPatch((x0, top - box_h), x1 - x0, box_h,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=VALID_FILL, edgecolor=VALID_EDGE,
                                lw=0.7, zorder=2))
    cx = (x0 + x1) / 2
    ax.text(cx, top - 0.11, "SAXS validation", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=INK)
    # the big numbers are CALVADOS-2; the small marked values take the
    # colors defined in panels a and b
    t = ax.text(0, 0, "CALVADOS-2", fontsize=5.9)
    w = 0.15 + text_width(ax, t)
    t.remove()
    swatch_text(ax, cx - w / 2, top - 0.215, CALVADOS, "CALVADOS-2", 5.9,
                bar_h=0.05, text_color=MUTED)
    tiles = [
        (f"{SAXS_CONSTRUCTS}", ["IDP constructs", f"({SAXS_PROTEINS} proteins)"], []),
        (f"{mare:.1f}%", [r"mean $R_g$ error"], small[0]),
        (f"{r:.2f}", [r"Pearson $r$, $R_g$"], small[1]),
        (f"{within:.0f}%", [r"within $|\Delta\mathsf{\nu}| < 0.05$"], small[2]),
    ]
    col_x = [x0 + (x1 - x0) * 0.27, x0 + (x1 - x0) * 0.73]
    row_y = [top - 0.37, top - 0.79]
    lh = 0.075
    for k, (value, label, marks) in enumerate(tiles):
        x, y = col_x[k % 2], row_y[k // 2]
        ax.text(x, y, value, ha="center", va="center", fontsize=12,
                fontweight="bold", color=INK)
        for j, line in enumerate(label):
            ax.text(x, y - 0.115 - j * lh, line, ha="center",
                    va="center", fontsize=5.7, color=MUTED)
        if marks:
            marked_values(ax, x, y - 0.115 - len(label) * lh, marks, lh)
    return stats


def main(out, palette=DEFAULT_PALETTE, saxs_recharged=None):
    use_palette(palette)
    seqs, orgs, mpipi, tot = load_counts()
    W, H = 5.5, 3.23   # six rows in panel a
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    bottom_a = panel_composition(ax, H - 0.04, seqs, orgs, mpipi, tot)
    top_b, box_h = bottom_a - 0.30, 1.15
    panel_pipeline(ax, top_b, 0.07, 3.78, box_h, tot)
    saxs = panel_saxs(ax, top_b, 4.08, 5.43, box_h, saxs_recharged)

    out = Path(out)
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), **kw)
    fig.savefig(out.with_suffix(".png"), dpi=400, **kw)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    print("sequences", seqs, "\norganisms", orgs, "\nmpipi", mpipi, "\n", tot)
    for ff, v in saxs.items():
        print(f"SAXS {ff}: " + ("pending" if v is None else
                                "MARE %.2f%%  r %.3f  |dnu|<0.05 %.1f%%" % v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # no default: JCTC Fig. 1 (figures/BENDER.pdf) is now drawn by make_fig1_overview.py, which imports this module
    ap.add_argument("--out", required=True)
    ap.add_argument("--palette", default=DEFAULT_PALETTE, choices=PALETTES)
    ap.add_argument("--saxs-recharged", default=None,
                    help="CSV with traj_id and recharged_rg_nm for the 77 constructs")
    args = ap.parse_args()
    main(args.out, args.palette, args.saxs_recharged)
