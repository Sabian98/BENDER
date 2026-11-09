"""
Figure 1 of the JPC Letter: representative ensembles along the nu axis in two force fields, from the ChimeraX renders in
ensembles/ (render_ensembles.cxc, 0.3 A per pixel). All eight images are cropped to one shared box so that sizes stay
comparable, and a 3 nm scale bar is drawn once.
Usage: python -I make_fig_ensembles.py <bio.csv> <outdir>
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'cm', 'font.size': 7.5, 'pdf.fonttype': 42,
                     'savefig.dpi': 300})
INK, MUTED = '#333333', '#777777'
CAL, RCH = '#6FA3D2', '#E3A35A'
REPS = [('compact_A0A540LLY7', 'A0A540LLY7', 'Compact, charge-balanced'),
        ('minimum_A0A0C3P091', 'A0A0C3P091', 'FCR minimum'),
        ('basic_Q95UN5', 'Q95UN5', 'Expanded, basic'),
        ('acidic_A0A091CXV9', 'A0A091CXV9', 'Expanded, acidic')]
PX_A = 0.3                     # Angstrom per pixel in the renders


def main(bio, out):
    d = pd.read_csv(bio).set_index('UniProt_ID')
    ims = {(n, ff): np.asarray(Image.open(f'{out}/ensembles/{n}_{ff}.png').convert('RGB')) for n, _, _ in REPS for ff in ('cal', 'rch')}
    mask = np.zeros(next(iter(ims.values())).shape[:2], bool)
    for im in ims.values():
        mask |= im.min(axis=2) < 245
    ys, xs = np.where(mask); pad = 15
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
    side = max(y1 - y0, x1 - x0); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, y1, x0, x1 = cy - side // 2, cy + side // 2, cx - side // 2, cx + side // 2

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.15), gridspec_kw={'hspace': 0.05, 'wspace': 0.02})
    for j, (n, uid, title) in enumerate(REPS):
        r = d.loc[uid]
        for i, (ff, col, lab) in enumerate([('cal', 'nu_orig', 'CALVADOS-2'), ('rch', 'nu', 'Mpipi-Recharged')]):
            ax = axes[i, j]; ax.imshow(ims[(n, ff)][y0:y1, x0:x1]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.text(0.5, 0.04, rf'$\nu$ = {r[col]:.2f}', transform=ax.transAxes, ha='center', fontsize=7, color=INK)
            if j == 0:
                ax.text(-0.04, 0.5, lab, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=7.5,
                        color=CAL if ff == 'cal' else RCH, fontweight='bold')
        ncpr = f'{r.NCPR:+.2f}'.replace('-', '\u2212')
        axes[0, j].set_title(f'{title}\n{uid} ({r.kingdom})\nFCR {r.FCR:.2f}, NCPR {ncpr}',
                             fontsize=6.5, color=INK, pad=3, linespacing=1.3)
    bar = 30 / PX_A
    axes[1, 3].plot([side - bar - 20, side - 20], [side - 60] * 2, color=INK, lw=1.5)
    axes[1, 3].text(side - bar / 2 - 20, side - 75, '3 nm', ha='center', fontsize=6.5, color=INK)
    fig.text(0.13, 0.035, 'Compact', fontsize=6.5, color=MUTED, ha='left')
    fig.text(0.805, 0.035, r'Expanded along $\nu$', fontsize=6.5, color=MUTED, ha='left')
    fig.patches.append(matplotlib.patches.FancyArrow(0.2, 0.045, 0.59, 0, width=0.002, head_width=0.012, head_length=0.012,
                                                     color=INK, transform=fig.transFigure, length_includes_head=True))
    fig.text(0.13, 0.965, '● Lys, Arg', color='#537A9E', fontsize=6.5)
    fig.text(0.22, 0.965, '● Asp, Glu', color='#CA534E', fontsize=6.5)
    fig.text(0.315, 0.965, '● Other residues in one frame, with nine more frames as faint lines', color='#8C8C8C', fontsize=6.5)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.84, bottom=0.07)
    fig.savefig(f'{out}/fig0_ensembles.pdf'); fig.savefig(f'{out}/fig0_ensembles.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
