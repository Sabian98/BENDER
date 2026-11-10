"""
Abstract TOC graphic for the JPC Letter (3.25 in x 1.75 in, sans-serif, thematic rather than exact).
Top: one shared rule along nu (compact charge-balanced chain -> coil -> extended polyelectrolyte).
Bottom: six taxon icons from the BENDER overview figure (taxon_icons/, cropped from Figure 1 of the BENDER paper), all linked to the shared rule, each with a little expanded chain in its own style
(basic blue for Bacteria, Protists and Viruses, acidic red for Plants, mixed for Mammals, mostly neutral for Fungi).
Usage: python make_toc_icons.py <outdir> <bio.csv>
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge, Rectangle, Ellipse, Circle, Polygon, RegularPolygon
from PIL import Image

plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'cm', 'pdf.fonttype': 42, 'savefig.dpi': 600})
INK, MUTED, ICON = '#333333', '#7A7A7A', '#5B6670'
POS, NEG, NEU = '#537A9E', '#CA534E', '#C9C9C9'


def chain(n, kind, rng):
    xy = np.zeros((n, 2)); ang = rng.uniform(0, 2 * np.pi)
    for i in range(1, n):
        ang += rng.normal(0, 0.35 if kind == 'extended' else 1.1)
        step = np.array([np.cos(ang), np.sin(ang)])
        if kind == 'compact':
            step -= 0.9 * xy[i - 1] / (0.5 + np.linalg.norm(xy[i - 1])); step /= np.linalg.norm(step)
        xy[i] = xy[i - 1] + step
    return xy - xy.mean(0)


def draw_chain(ax, xy, q, cx, cy, size, bead=5, lw=0.6, flat=False, shadow=False):
    if flat:                                              # rotate the long axis of the chain to the horizontal
        u, sv, vt = np.linalg.svd(xy - xy.mean(0)); xy = (xy - xy.mean(0)) @ vt.T
        xy[:, 1] *= 0.6
    p = xy / np.ptp(xy, axis=0).max() * size + np.array([cx, cy])
    if shadow:
        for k, a in [(0.022, 0.06), (0.014, 0.08), (0.007, 0.10)]:
            ax.plot(p[:, 0] + k, p[:, 1] - k, color='#5A6470', lw=lw + 2.2, alpha=a, zorder=2, solid_capstyle='round')
    ax.plot(p[:, 0], p[:, 1], color='#A0A0A0', lw=lw, zorder=3, solid_capstyle='round')
    ax.scatter(p[:, 0], p[:, 1], s=bead, c=[{1: POS, -1: NEG, 0: NEU}[v] for v in q], edgecolors='white', linewidths=0.25, zorder=4)


# ---------------------------------------------------------------- icons (centred at cx, cy, about s wide)
def bacterium(ax, cx, cy, s):
    ax.add_patch(FancyBboxPatch((cx - 0.42 * s, cy - 0.17 * s), 0.84 * s, 0.34 * s,
                                boxstyle=f'round,pad=0,rounding_size={0.17 * s}', fc=ICON, ec='none', zorder=5))
    t = np.linspace(0, 1, 40)
    ax.plot(cx + 0.42 * s + 0.35 * s * t, cy + 0.06 * s * np.sin(t * 4 * np.pi), color=ICON, lw=0.8, zorder=5)


def fungus(ax, cx, cy, s):
    ax.add_patch(Rectangle((cx - 0.08 * s, cy - 0.38 * s), 0.16 * s, 0.38 * s, fc=ICON, ec='none', zorder=5))
    ax.add_patch(Wedge((cx, cy - 0.02 * s), 0.42 * s, 0, 180, fc=ICON, ec='none', zorder=6))
    for dx, dy, r in [(-0.18, 0.15, 0.05), (0.1, 0.24, 0.045), (0.22, 0.08, 0.04)]:
        ax.add_patch(Circle((cx + dx * s, cy + dy * s), r * s, fc='white', ec='none', zorder=7))


def mammal(ax, cx, cy, s):
    ax.add_patch(Ellipse((cx - 0.05 * s, cy - 0.08 * s), 0.62 * s, 0.36 * s, fc=ICON, ec='none', zorder=5))
    ax.add_patch(Circle((cx + 0.28 * s, cy + 0.02 * s), 0.16 * s, fc=ICON, ec='none', zorder=5))
    ax.add_patch(Circle((cx + 0.26 * s, cy + 0.2 * s), 0.08 * s, fc=ICON, ec='none', zorder=5))
    ax.add_patch(Circle((cx + 0.33 * s, cy + 0.05 * s), 0.025 * s, fc='white', ec='none', zorder=6))
    t = np.linspace(0, 1, 30)
    ax.plot(cx - 0.36 * s - 0.25 * s * t, cy - 0.1 * s + 0.12 * s * np.sin(t * np.pi * 1.4), color=ICON, lw=0.8, zorder=5)


def plant(ax, cx, cy, s):
    ax.plot([cx, cx], [cy - 0.4 * s, cy + 0.15 * s], color=ICON, lw=1.0, zorder=5)
    ax.add_patch(Ellipse((cx - 0.17 * s, cy + 0.12 * s), 0.36 * s, 0.16 * s, angle=30, fc=ICON, ec='none', zorder=5))
    ax.add_patch(Ellipse((cx + 0.17 * s, cy + 0.02 * s), 0.36 * s, 0.16 * s, angle=-30, fc=ICON, ec='none', zorder=5))
    ax.add_patch(Ellipse((cx, cy + 0.27 * s), 0.14 * s, 0.24 * s, fc=ICON, ec='none', zorder=5))
    ax.plot([cx - 0.25 * s, cx + 0.25 * s], [cy - 0.4 * s, cy - 0.4 * s], color=ICON, lw=0.9, zorder=5)


def protist(ax, cx, cy, s, rng):
    th = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    r = 0.33 + 0.07 * np.sin(3 * th + 0.5) + 0.05 * np.sin(5 * th + 1.3)
    ax.add_patch(Polygon(np.c_[cx + s * r * np.cos(th), cy + s * 0.85 * r * np.sin(th)], fc=ICON, ec='none', zorder=5))
    ax.add_patch(Circle((cx + 0.05 * s, cy + 0.02 * s), 0.09 * s, fc='white', ec='none', zorder=6))
    ax.add_patch(Circle((cx + 0.05 * s, cy + 0.02 * s), 0.045 * s, fc=ICON, ec='none', zorder=7))


def virus(ax, cx, cy, s):
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=0.3 * s, orientation=np.pi / 6, fc=ICON, ec='none', zorder=5))
    for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        x0, y0 = cx + 0.3 * s * np.cos(a), cy + 0.3 * s * np.sin(a)
        x1, y1 = cx + 0.44 * s * np.cos(a), cy + 0.44 * s * np.sin(a)
        ax.plot([x0, x1], [y0, y1], color=ICON, lw=0.8, zorder=5)
        ax.add_patch(Circle((x1, y1), 0.04 * s, fc=ICON, ec='none', zorder=5))


def place_icon(ax, path, cx, cy, maxw, maxh):
    """Draw a transparent PNG icon centred at (cx, cy) inside a maxw x maxh box, keeping its aspect ratio."""
    im = Image.open(path); w, h = im.size; sc = min(maxw / w, maxh / h)
    ax.imshow(np.asarray(im), extent=(cx - w * sc / 2, cx + w * sc / 2, cy - h * sc / 2, cy + h * sc / 2), zorder=5,
              interpolation='lanczos')


def main(out, bio=None):
    """Two-branch axis with taxon occupancy. The curve is a thematic U of charged fraction along nu, whose left
    branch is labelled 'added charge compacts' and right branch 'added charge expands', with the shared window
    shaded. Taxon badges link to each taxon's mean nu (CALVADOS-2) and their rings show the share of strong
    polycations (blue) and polyanions (red), so taxa sit on the same axis in different ways."""
    import pandas as pd
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Arc
    from matplotlib.collections import LineCollection
    RING = {'Bacteria': '#B997C6', 'Fungi': '#7DBDB3', 'Mammals': '#7F7FB6', 'Plants': '#9AACD7',
            'Protists': '#DD9A63', 'Viruses': '#D47C80'}
    d = pd.read_csv(bio); d = d[d.nu_ok_cal.astype(bool) & d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig', 'nu'])
    aN = d.NCPR.abs(); strong = d.FCR > 0.35
    d['pc'] = strong & (aN > 0.35) & (d.NCPR > 0); d['pa'] = strong & (aN > 0.35) & (d.NCPR < 0)
    st = d.groupby('kingdom').agg(nu=('nu_orig', 'mean'), pc=('pc', 'mean'), pa=('pa', 'mean'))
    order = st.sort_values('nu').index.tolist()

    fig = plt.figure(figsize=(3.25, 1.75))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 3.25); ax.set_ylim(0, 1.75); ax.axis('off'); ax.set_aspect('equal')
    x0, x1 = 0.33, 3.02
    def X(nu):
        return x0 + (nu - 0.44) / (0.68 - 0.44) * (x1 - x0)
    yb = 0.95                                    # nu axis
    # shared window
    ax.add_patch(FancyBboxPatch((X(0.49), yb), X(0.60) - X(0.49), 0.72, boxstyle='round,pad=0,rounding_size=0.04',
                                fc='#EEF1F5', ec='none', zorder=0))
    ax.plot([X(0.588)] * 2, [yb, yb + 0.72], color=MUTED, lw=0.45, ls=':', zorder=1)
    ax.text((X(0.49) + X(0.60)) / 2, yb + 0.69, 'Shared window', ha='center', fontsize=6.0, color=MUTED)
    # thematic U of charged fraction along nu, scaled to fit the panel
    def F(nu):
        return np.where(nu < 0.535, (0.535 - nu) / 0.09, ((nu - 0.535) / 0.14) ** 1.6 * 1.9)
    lo, hi = yb + 0.09, yb + 0.60
    def Y(nu):
        return lo + F(nu) / F(np.array(0.675)) * (hi - lo)
    nus = np.linspace(0.445, 0.675, 200)
    pts = np.c_[X(nus), Y(nus)]; segs = np.stack([pts[:-1], pts[1:]], axis=1)
    cols = [('#6E8FB3' if v < 0.54 else '#B5566A') for v in nus[:-1]]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.0, zorder=3, capstyle='round'))
    def along(n0, n1, col, off=0.06):                 # arrow parallel to the curve, offset above it
        p0 = np.array([X(n0), Y(n0)]); p1 = np.array([X(n1), Y(n1)]); v = p1 - p0; nrm = np.array([-v[1], v[0]]) / np.linalg.norm(v)
        if nrm[1] < 0: nrm = -nrm
        ax.annotate('', xy=p1 + off * nrm, xytext=p0 + off * nrm,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=0.9, mutation_scale=6), zorder=4)
        return p1 + off * nrm
    e1 = along(0.505, 0.455, '#6E8FB3')
    ax.text(e1[0] + 0.02, e1[1] + 0.05, 'More charge,\nmore compact', ha='left', va='bottom', fontsize=6.4, color='#4F6E92', linespacing=1.05)
    e2 = along(0.615, 0.66, '#B5566A')
    ax.text(e2[0] - 0.12, e2[1] - 0.02, 'More charge,\nmore expanded', ha='right', va='bottom', fontsize=6.4, color='#94404F', linespacing=1.05)
    ax.text(X(0.445) - 0.05, yb + 0.12, 'Charge', ha='right', va='center', fontsize=6.0, color=MUTED, rotation=90)
    # nu axis
    ax.annotate('', xy=(x1 + 0.05, yb), xytext=(x0 - 0.02, yb), arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.7, mutation_scale=6))
    ax.text(x1 + 0.09, yb, r'$\nu$', ha='left', va='center', fontsize=10.5, color=INK)
    # taxa: badges ordered by mean nu, linked to their mean nu on the axis
    xs = np.linspace(0.33, 2.92, 6); yi, R = 0.50, 0.195
    for xb, name in zip(xs, order):
        c = RING[name]; xm = X(st.loc[name, 'nu'])
        ax.add_patch(Polygon([(xm, yb - 0.005), (xm - 0.03, yb - 0.055), (xm + 0.03, yb - 0.055)], fc=c, ec='none', zorder=4))
        verts = [(xm, yb - 0.06), (xm, yb - 0.17), (xb, yi + R + 0.12), (xb, yi + R + 0.02)]
        ax.add_patch(PathPatch(Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                               fc='none', ec=c, lw=0.8, alpha=0.8, zorder=2))
        ax.add_patch(Circle((xb, yi), R + 0.03, fc=c, ec='none', alpha=0.12, zorder=3))
        ax.add_patch(Circle((xb, yi), R, fc='white', ec='none', zorder=4))
        # ring split by occupancy: polycation share blue, polyanion share red, rest in taxon colour
        pc, pa = st.loc[name, 'pc'], st.loc[name, 'pa']
        a0 = 90
        for frac, col in [(pc, POS), (pa, NEG), (1 - pc - pa, c)]:
            if frac <= 0:
                continue
            ax.add_patch(Arc((xb, yi), 2 * R, 2 * R, theta1=a0 - 360 * frac, theta2=a0, color=col,
                             lw=2.2 if col != c else 1.3, zorder=5))
            a0 -= 360 * frac
        place_icon(ax, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'taxon_icons', f'{name}.png'), xb, yi, 0.31, 0.26)
        ax.text(xb, yi - R - 0.035, name, ha='center', va='top', fontsize=6.6, color=INK, fontweight='medium')
    ax.text(X(0.522), yb - 0.035, r'Mean $\nu$', ha='right', va='center', fontsize=5.8, color=MUTED)
    ax.text(0.04, 0.035, 'Ring', ha='left', va='bottom', fontsize=5.8, color=MUTED)
    ax.add_patch(Arc((0.31, 0.065), 0.08, 0.08, theta1=30, theta2=150, color=POS, lw=1.8))
    ax.text(0.37, 0.035, 'Polycations', ha='left', va='bottom', fontsize=5.8, color=MUTED)
    ax.add_patch(Arc((0.98, 0.065), 0.08, 0.08, theta1=30, theta2=150, color=NEG, lw=1.8))
    ax.text(1.04, 0.035, 'Polyanions', ha='left', va='bottom', fontsize=5.8, color=MUTED)
    ax.text(3.22, 0.035, 'One axis, taxa differ in where they sit', ha='right', va='bottom', fontsize=6.0, color=MUTED, style='italic')
    fig.savefig(f'{out}/toc_map.pdf'); fig.savefig(f'{out}/toc_map.png', dpi=300)
    fig.savefig(f'{out}/toc.pdf'); fig.savefig(f'{out}/toc.png', dpi=300); plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])   # usage: make_toc_icons.py <outdir> <bio.csv>
