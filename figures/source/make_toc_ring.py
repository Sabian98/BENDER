"""TOC graphic, ring version (2026-10-26), after the BENDER overview graphic in ~/Downloads/BENDER.pdf.

ACS table-of-contents size, 3.25 x 1.75 in. BENDER holds three force fields (CALVADOS-2 and MPIPI-Recharged on all 10,282
sequences, MPIPI-GG on a subset). The six groups of life sit on a soft elliptical band around the dataset
name. In the center, one conformer of the contested protist A0A7S2N1D9 from each of the three force fields, drawn to the
same scale, shows the paper's point. The chain is extended under CALVADOS-2 (blue), intermediate under MPIPI-GG
(mauve) and compact under MPIPI-Recharged (tan), with
Arginine beads in black and aromatic beads in gold. Group icons come from the source graphic (toc_icons/), and rim
colors follow it. Per ACS guidance there are no counts, and the pills name the force fields.

Usage:  python make_toc_ring.py [--out output/toc_ring]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import palette as P  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                            "pdf.fonttype": 42})
W, H = 3.25, 1.75
CX, CY, RX, RY, R = W / 2, 0.875, 1.20, 0.585, 0.245          # ring center and semi-axes, circle radius (inches)
ICONS = HERE / "toc_icons"
# (group, angle on the ring in degrees, rim color from the source graphic, icon, icon height as a fraction of R)
GROUPS = [("Bacteria", 90, "#C49BC9", "bacteria.png", 0.42),
          ("Viruses", 30, "#E07A7E", "virus.png", 0.55),
          ("Protists", -30, "#E99A5F", "protist.png", 0.26),
          ("Fungi", -90, "#73BFB8", "fungi.png", 0.62),
          ("Plants", 210, "#93AED8", "plant.png", 0.80),
          ("Mammals", 150, "#8584BA", "mammal.png", 0.85)]



def tint(c, f):
    c = np.array(to_rgb(c))
    return tuple(c + (1 - c) * f)


# (conformer key, x offset from the center, line, faint ensemble, glow and label fill, beads, label)
FFS = [("calvados2", -0.55, P.C2, P.C2_LIGHT, P.C2_PALE, P.C2_DARK, "CALVADOS-2"),
       ("mpipi_gg", 0.0, P.GG, tint(P.GG, 0.45), tint(P.GG, 0.75), P.GG_DARK, "MPIPI-GG"),
       ("recharged", 0.55, P.MR, P.MR_LIGHT, P.MR_PALE, P.MR_DARK, "MPIPI-Recharged")]

def place_icon(ax, path, x, y, height):
    img = plt.imread(path)
    h, w = img.shape[:2]
    width = height * w / h
    ax.imshow(img, extent=(x - width / 2, x + width / 2, y - height / 2, y + height / 2), zorder=6,
              interpolation="lanczos")


def ring_point(deg, rx=RX, ry=RY):
    t = np.radians(deg)
    return CX + rx * np.cos(t), CY + ry * np.sin(t)


def halo(ax):
    """The band behind the circles, its color drifting between the rims of neighboring groups."""
    angs = [g[1] % 360 for g in GROUPS]
    cols = [np.array(to_rgb(g[2])) for g in GROUPS]
    order = np.argsort(angs)
    angs, cols = [angs[i] for i in order], [cols[i] for i in order]
    t = np.linspace(0, 360, 721)
    pts = np.array([ring_point(a) for a in t])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    colours = []
    for a in t[:-1]:
        k = np.searchsorted(angs, a) % len(angs)
        a0, a1 = angs[k - 1], angs[k] + (360 if angs[k] < angs[k - 1] else 0)
        aa = a if a >= a0 else a + 360
        f = (aa - a0) / (a1 - a0)
        c = (1 - f) * cols[k - 1] + f * cols[k]
        colours.append((*(0.45 * c + 0.55), 1.0))                 # pale, opaque, so overlaps do not darken
    ax.add_collection(LineCollection(segs, colors=colours, linewidths=9, capstyle="round", zorder=1))


def conformer(ax, xy, seq, centre, scale, colour, dark):
    xy = (np.asarray(xy) - np.asarray(xy).mean(0)) * scale + centre
    ax.plot(*xy.T, color=colour, lw=0.9, zorder=8, solid_capstyle="round", solid_joinstyle="round")
    for q, aa in zip(xy, seq):
        if aa == "R":
            ax.add_patch(Circle(q, 0.012, facecolor=P.INK, edgecolor="white", lw=0.2, zorder=10))
        elif aa in "FWY":
            ax.add_patch(Circle(q, 0.014, facecolor=P.ARO, edgecolor=P.INK, lw=0.25, zorder=11))
        else:
            ax.add_patch(Circle(q, 0.006, facecolor=dark, edgecolor="none", zorder=9))


def main(out):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # one soft glow behind each force field's chain
    yy, xx = np.mgrid[0:H:350j, 0:W:650j]
    for (ff, gx, colour, light, pale, dark, label) in FFS:
        d = np.sqrt(((xx - (CX + gx)) / 0.32) ** 2 + ((yy - (CY - 0.15)) / 0.22) ** 2)
        glow = np.zeros((*d.shape, 4))
        glow[..., :3] = to_rgb(pale)
        glow[..., 3] = np.clip(1 - d, 0, 1) ** 1.4
        ax.imshow(glow, extent=(0, W, 0, H), origin="lower", zorder=0, interpolation="bilinear")

    halo(ax)
    yy, xx = np.mgrid[-1:1:200j, -1:1:200j]
    rim_alpha = np.clip((np.hypot(xx, yy) - 0.55) / 0.45, 0, 1) ** 1.5 * 0.16
    for name, deg, rim, icon, ih in GROUPS:
        x, y = ring_point(deg)
        for k in range(7):                                             # soft, blurred drop shadow
            ax.add_patch(Circle((x + 0.010, y - 0.014), R + 0.0045 * k, facecolor="#000000", alpha=0.022, lw=0,
                                zorder=3))
        face = Circle((x, y), R, facecolor="white", edgecolor="none", zorder=4)
        ax.add_patch(face)
        tint = np.zeros((*rim_alpha.shape, 4))                       # a faint wash of the rim color at the edge
        tint[..., :3] = to_rgb(rim)
        tint[..., 3] = rim_alpha
        im = ax.imshow(tint, extent=(x - R, x + R, y - R, y + R), origin="lower", zorder=4.5, interpolation="bilinear")
        im.set_clip_path(face)
        ax.add_patch(Circle((x, y), R, facecolor="none", edgecolor=rim, lw=1.5, zorder=5))
        place_icon(ax, ICONS / icon, x, y + 0.07, ih * R)
        if icon == "virus.png":                                      # a small cluster, as in the source graphic
            for dx, dy, s in ((-0.075, 0.03, 0.55), (0.08, 0.045, 0.45), (0.065, -0.005, 0.35)):
                place_icon(ax, ICONS / icon, x + dx, y + 0.07 + dy, s * ih * R)
        ax.text(x, y - 0.115, name, ha="center", va="center", fontsize=5.4, fontweight="bold", color=P.INK,
                zorder=7)

    ex = json.load(open(HERE / "a0a7s2n1d9_conformers.json"))
    seq = ex["sequence"]
    scale = 0.034                                                    # inches per nm, the same for all three chains
    for (ff, gx, colour, light, pale, dark, label) in FFS:
        x0, y0 = CX + gx, CY - 0.15
        confs = ex[ff]["conformers_nm"]
        oval = Ellipse((x0, y0), 0.50, 0.27, facecolor="none", edgecolor="none")
        ax.add_patch(oval)
        for c in confs[1:5]:                                          # a faint ensemble behind the chosen conformer
            c = (np.asarray(c) - np.asarray(c).mean(0)) * scale + (x0, y0)
            line, = ax.plot(*c.T, color=light, lw=0.45, alpha=0.65, zorder=7, solid_capstyle="round")
            line.set_clip_path(oval)
        conformer(ax, confs[0], seq, (x0, y0), scale, colour, dark)
        ax.text(x0, CY + 0.07, label, ha="center", va="center", fontsize=4.5, color=P.INK, fontweight="bold",
                zorder=12, bbox=dict(boxstyle="round,pad=0.30,rounding_size=0.5", facecolor=pale, edgecolor=colour,
                                     linewidth=0.6))
    ax.text(CX, CY + 0.245, "BENDER", ha="center", va="center", fontsize=11.5, fontweight="bold", color=P.INK,
            zorder=12)
    bar = FancyBboxPatch((CX - 0.30, CY + 0.165), 0.60, 0.014, boxstyle="round,pad=0,rounding_size=0.007",
                         facecolor="none", edgecolor="none", zorder=12)
    ax.add_patch(bar)
    f = np.linspace(0, 1, 256)
    stops = [np.array(to_rgb(c)) for c in (P.C2, P.GG, P.MR)]       # the three force fields, left to right
    grad = np.where(f[:, None] < 0.5, (1 - 2 * f[:, None]) * stops[0] + 2 * f[:, None] * stops[1],
                    (2 - 2 * f[:, None]) * stops[1] + (2 * f[:, None] - 1) * stops[2])[None, :, :]
    im = ax.imshow(grad, extent=(CX - 0.30, CX + 0.30, CY + 0.165, CY + 0.179), zorder=12, aspect="auto")
    im.set_clip_path(bar)

    out = Path(out).expanduser()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    print("wrote", out.with_suffix(".pdf"), f"({W} x {H} in)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.out("toc_ring")))
    main(ap.parse_args().out)
