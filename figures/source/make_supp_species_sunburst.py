"""SI figure (2026-10-26): the organisms behind BENDER as a taxonomic sunburst, in the style of Krona.

Three rings, every arc proportional to its number of sequences: the six groups (inner), one major clade per organism
(middle) and the organisms themselves, as unique NCBI taxonomy identifiers (outer). Clades are read from the NCBI
lineages in ncbi_taxonomy_cache.json at one rank per group: phylum for bacteria, class for fungi, order for plants and
mammals, family for viruses and the eukaryotic supergroup for protists, many of which NCBI places in no phylum. Clades
holding under 3% of their group, or too narrow to name, are pooled as "Other". Organisms with fewer than ORG_MIN sequences are pooled into one
pale arc per clade, and those with at least LABEL_MIN are named around the circle with their sequence counts. Group
colors follow the TOC graphic (make_toc_ring.py), and groups run clockwise in the order of Figure 1a.

Usage:  python make_supp_species_sunburst.py [--out output/bender_species_sunburst]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Wedge
from matplotlib.textpath import TextPath

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import palette as P  # noqa: E402
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

BIO = paths.BIO
TAX = paths.TAXONOMY
GROUPS = [("Mammals", "#8584BA"), ("Fungi", "#73BFB8"), ("Plants", "#93AED8"), ("Protists", "#E99A5F"),
          ("Bacteria", "#C49BC9"), ("Viruses", "#E07A7E")]
RANK = {"Bacteria": "phylum", "Fungi": "class", "Plants": "order", "Mammals": "order", "Viruses": "family"}
SUPERGROUPS = ["Alveolata", "Stramenopiles", "Rhizaria", "Discoba", "Amoebozoa", "Metamonada", "Haptista",
               "Cryptophyceae", "Rhodophyta", "Glaucocystophyceae", "Opisthokonta", "Apusozoa", "CRuMs",
               "Ancyromonadida", "Provora", "Hemimastigophora", "Malawimonadida", "Telonemida"]
ABBR = {"Human immunodeficiency virus type 1": "HIV-1", "Simian immunodeficiency virus": "SIV",
        "Human respiratory syncytial virus": "Human RSV"}
MIN_SHARE, ORG_MIN, LABEL_MIN, CLADE_LABEL_DEG = 0.03, 12, 45, 3.0
GAP = 2.0                                                  # degrees between groups
R_GROUP, R_CLADE, R_ORG, R_LABEL = (0.62, 1.00), (1.03, 1.92), (1.95, 2.22), 2.40
FS_ORG, FS_CLADE = 6.0, 5.6

matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                            "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                            "mathtext.it": "Arial:italic"})


def tint(c, f):
    """Mix a color with white by fraction f."""
    c = np.array(to_rgb(c))
    return tuple(c + (1 - c) * f)


def clean(name):
    name = re.sub(r"\s*\(.*$", "", str(name)).strip()
    name = re.sub(r"\s+(subsp\.|f\. sp\.|DSM|ATCC|strain)\s.*$", "", name)
    words = name.split()
    if len(words) == 3 and words[1] == words[2]:              # Plasmodium yoelii yoelii
        name = " ".join(words[:2])
    name = ABBR.get(name, name)
    return name[:1].upper() + name[1:]


def text_width(s, size):
    """Width of a label in inches, so that its count can follow it along the same radius."""
    return TextPath((0, 0), s, size=size, prop=FontProperties(family="Arial")).get_extents().width / 72


def clade_of(group, lineage):
    if group == "Protists":
        names = {n for _, n in lineage}
        return next((s for s in SUPERGROUPS if s in names), "Other")
    return dict((r, n) for r, n in lineage).get(RANK[group], "Unclassified")


def spread(angles, sep):
    """Push sorted label angles apart until neighbors are at least sep degrees apart, staying close to the targets."""
    a = np.array(angles, float)
    for _ in range(500):
        moved = False
        for i in range(len(a) - 1):
            d = a[i + 1] - a[i]
            if d < sep:
                a[i] -= (sep - d) / 2
                a[i + 1] += (sep - d) / 2
                moved = True
        if not moved:
            break
    return a


def radial_text(ax, r, deg, s, colour, size, ha_out="left", **kw):
    """Text along a radius, reading outwards on the right half and inwards (flipped) on the left."""
    flip = 90 < (deg % 360) < 270
    t = np.radians(deg)
    ax.text(r * np.cos(t), r * np.sin(t), s, rotation=deg + (180 if flip else 0), rotation_mode="anchor",
            ha=("right" if ha_out == "left" else "left") if flip else ha_out, va="center", fontsize=size,
            color=colour, **kw)


def main(out):
    cache = json.load(open(TAX))
    b = pd.read_csv(BIO, usecols=["UniProt_ID", "kingdom", "organism", "taxonomy_id"], low_memory=False)
    b = b[b.kingdom.isin([g for g, _ in GROUPS])]
    total = len(b)
    span = 360 - GAP * len(GROUPS)

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-3.6, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")

    labels = []                                                # (angle, name, count, group color)
    start = 90 - GAP / 2                                       # clockwise from the top: angles decrease
    for g, colour in GROUPS:
        s = b[b.kingdom == g]
        orgs = (s.groupby("taxonomy_id").agg(n=("UniProt_ID", "size"), name=("organism", "first")).reset_index())
        orgs["clade"] = [clade_of(g, cache[str(int(t))]["lineage"]) for t in orgs.taxonomy_id]
        width = span * len(s) / total
        share = orgs.groupby("clade").n.sum() / len(s)
        # a clade is drawn on its own only if it holds MIN_SHARE of the group and its arc is wide enough to name
        orgs.loc[orgs.clade.map(share) < max(MIN_SHARE, CLADE_LABEL_DEG / width), "clade"] = "Other"
        size = orgs.groupby("clade").n.sum()
        order = [c for c in size.sort_values(ascending=False).index if c not in ("Other", "Unclassified")]
        order += [c for c in ("Unclassified", "Other") if c in size.index]

        a0, a1 = start - width, start
        ax.add_patch(Wedge((0, 0), R_GROUP[1], a0, a1, width=R_GROUP[1] - R_GROUP[0], facecolor=colour,
                           edgecolor="white", lw=0.8))
        mid = (a0 + a1) / 2
        upper = 0 <= (mid % 360) <= 180
        mr, rr = np.radians(mid), sum(R_GROUP) / 2
        rot = mid - 90 if upper else mid + 90
        for dr, text, kw in ((0.045, g, dict(fontsize=7.2, fontweight="bold", color=P.INK)),
                             (-0.085, f"by {RANK.get(g, 'supergroup')}", dict(fontsize=5.2, color=P.DARK))):
            r = rr + (dr if upper else -dr)                       # the group name stays on the outer side
            ax.text(r * np.cos(mr), r * np.sin(mr), text, rotation=rot, rotation_mode="anchor", ha="center",
                    va="center", **kw)

        pos = a1
        for k, c in enumerate(order):
            o = orgs[orgs.clade == c].sort_values("n", ascending=False)
            cw = width * o.n.sum() / len(s)
            pooled = c in ("Other", "Unclassified")
            ax.add_patch(Wedge((0, 0), R_CLADE[1], pos - cw, pos, width=R_CLADE[1] - R_CLADE[0],
                               facecolor=tint(colour, 0.62 if pooled else (0.22 if k % 2 == 0 else 0.40)),
                               edgecolor="white", lw=0.6))
            if cw >= CLADE_LABEL_DEG:
                radial_text(ax, R_CLADE[0] + 0.07, pos - cw / 2, c, P.INK, FS_CLADE)
            # organisms, largest first, then the clade's small organisms pooled into one pale arc
            opos = pos
            big = o[o.n >= ORG_MIN]
            for j, r in enumerate(big.itertuples()):
                w = width * r.n / len(s)
                ax.add_patch(Wedge((0, 0), R_ORG[1], opos - w, opos, width=R_ORG[1] - R_ORG[0],
                                   facecolor=tint(colour, 0.38 if j % 2 == 0 else 0.55), edgecolor="white", lw=0.3))
                if r.n >= LABEL_MIN:
                    labels.append((opos - w / 2, clean(r.name), int(r.n), colour, g == "Viruses"))
                opos -= w
            rest = o.n.sum() - big.n.sum()
            if rest:
                w = width * rest / len(s)
                ax.add_patch(Wedge((0, 0), R_ORG[1], opos - w, opos, width=R_ORG[1] - R_ORG[0],
                                   facecolor=tint(colour, 0.80), edgecolor="white", lw=0.3))
            pos -= cw
        start = a0 - GAP

    # organism names around the circle, spread apart where they crowd, with leader lines to their arcs
    labels.sort(key=lambda x: x[0])
    sep = np.degrees(FS_ORG * 1.25 / 72 / R_LABEL)
    targets = spread([x[0] for x in labels], sep)
    for (m, name, n, colour, virus), t in zip(labels, targets):
        mt, tt = np.radians(m), np.radians(t)
        ax.plot([R_ORG[1] * np.cos(mt), (R_ORG[1] + 0.05) * np.cos(mt), (R_LABEL - 0.04) * np.cos(tt)],
                [R_ORG[1] * np.sin(mt), (R_ORG[1] + 0.05) * np.sin(mt), (R_LABEL - 0.04) * np.sin(tt)],
                color=tint(colour, 0.25), lw=0.55, solid_capstyle="round", zorder=1)
        label = name if virus else r"$\mathit{" + name.replace(" ", r"\ ") + r"}$"
        count = f"{n:,}"
        flip = 90 < (t % 360) < 270
        if flip:                                               # left half reads inwards, so the count goes first
            rn, rc = R_LABEL + text_width(count, FS_ORG) + 0.06, R_LABEL
        else:
            rn, rc = R_LABEL, R_LABEL + text_width(label, FS_ORG) + 0.06
        radial_text(ax, rn, t, label, P.INK, FS_ORG)
        radial_text(ax, rc, t, count, P.MUTED, FS_ORG)

    ax.text(0, 0.09, "BENDER", ha="center", va="center", fontsize=11, fontweight="bold", color=P.INK)
    ax.text(0, -0.11, f"{total:,} sequences\n{b.taxonomy_id.nunique():,} organisms", ha="center", va="center",
            fontsize=6.2, color=P.DARK, linespacing=1.3)

    out = Path(out).expanduser()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print("wrote", out.with_suffix(".pdf"), f"| {len(labels)} organisms named")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.out("bender_species_sunburst")))
    main(ap.parse_args().out)
