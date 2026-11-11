"""
Figures for the JPC Letter "The Flory Exponent as a Common Axis" (main_jpcl.tex).

Usage:  python -I make_figures.py <bio.csv> <mpipi_results_v2.csv> <saxs63_with_map.csv>
        <go_enrichment_expanded.tsv> <outdir> [figure numbers]

House look follows BENDER-DYN/methods_paper/figures/figstyle.py (Arial, 7.5 pt, bold lowercase panel letters,
BENDER-MT 'state-light' colors, user 2026-11-08). Color encodes force field as in BENDER-MT: CALVADOS-2 light
steel blue, Mpipi-Recharged light amber (plus solid/dashed lines and circle/square markers as redundant cues). Taxa that share a panel are always labeled in text.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold

plt.rcParams.update({
    'font.family': 'Arial', 'font.sans-serif': ['Arial'],
    'mathtext.fontset': 'cm',
    'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
})

TAXA = ['Bacteria', 'Fungi', 'Mammals', 'Plants', 'Protists', 'Viruses']
LC_OLD = {'Bacteria': '#C0682F', 'Fungi': '#4F9BD0', 'Mammals': '#B5473A', 'Plants': '#2F7DB5',
      'Protists': '#B8841F', 'Viruses': '#6A5DAF'}
INK, MUTED, GRID = '#333333', '#777777', '#E5E5E5'
# BENDER-MT 'state-light' palette (code/figfont.py): color encodes force field, taxa are separated by panel or text label
CAL, RCH, GG_FILL, GG_LINE, TERRA, SLATE, RED = '#6FA3D2', '#E3A35A', '#C5E0C4', '#3A6E38', '#AC7D5C', '#537A9E', '#CA534E'
FFC = {'nu_orig': CAL, 'nu': RCH}
WINDOW = (0.49, 0.60)
GOOD_SOLVENT = 0.588
FF = [('nu_orig', 'CALVADOS-2', '-', 'o'), ('nu', 'Mpipi-Recharged', '--', 'o')]
BINS = np.arange(0.38, 0.76, 0.02)
DESC = ['FCR', 'NCPR', 'SCD', 'SHD', 'f_aro', 'f_P', 'f_G', 'hydropathy_avg', 'f_K', 'f_R', 'logN']


def panel(ax, letter, title=None, x=-0.22, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left', color=INK)
    if title:
        ax.text(x + 0.12, y, title, transform=ax.transAxes, fontsize=7.5, va='bottom', ha='left', color=INK)


def check_layout(fig, name):
    """Report text that overlaps other text, sits on data points or lines, or runs off the figure."""
    from matplotlib.text import Text
    from matplotlib.collections import PathCollection
    fig.canvas.draw(); r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height
    hidden = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for loc, lab in zip(axis.get_majorticklocs(), axis.get_majorticklabels()):
                if not (lo - 1e-9 <= loc <= hi + 1e-9):
                    hidden.add(id(lab))
    texts = [x for x in fig.findobj(Text) if x.get_visible() and x.get_text().strip() and id(x) not in hidden
             and x.get_window_extent(r).width > 0]
    boxes = [(x, x.get_window_extent(r).expanded(1.02, 1.05)) for x in texts]
    issues = []
    for (a, ba), (b, bb) in __import__('itertools').combinations(boxes, 2):
        if ba.overlaps(bb) and a.axes is not None and a.axes is b.axes and a.axes.images:
            continue  # heatmap cell values sit side by side by design
        if ba.overlaps(bb):
            issues.append(f'text/text  "{a.get_text()[:28]}"  x  "{b.get_text()[:28]}"')
    for x, bx in boxes:
        if bx.x0 < -1 or bx.y0 < -1 or bx.x1 > W + 1 or bx.y1 > H + 1:
            issues.append(f'off figure "{x.get_text()[:30]}"')
    for ax in fig.axes:
        pts = []
        for c in ax.collections:
            if isinstance(c, PathCollection) and len(c.get_offsets()):
                pts.append(c.get_offset_transform().transform(c.get_offsets()))
        for ln in ax.lines:
            xy = ln.get_xydata()
            if len(xy) and ln.get_visible() and ln.get_linestyle() not in ('None', ' ', '') and ln.get_label().startswith('_child'):
                xs = np.linspace(0, 1, 30)
                for i in range(len(xy) - 1):
                    seg = xy[i] + np.outer(xs, xy[i + 1] - xy[i]); pts.append(ln.get_transform().transform(seg))
            elif len(xy) and ln.get_marker() not in (None, 'None', ''):
                pts.append(ln.get_transform().transform(xy))
        if not pts:
            continue
        P = np.vstack(pts)
        for x, bx in boxes:
            if x.axes is ax and not ax.images and x not in ax.get_xticklabels() + ax.get_yticklabels():
                inside = (P[:, 0] > bx.x0) & (P[:, 0] < bx.x1) & (P[:, 1] > bx.y0) & (P[:, 1] < bx.y1)
                if inside.any():
                    issues.append(f'text/data  "{x.get_text()[:30]}"  ({inside.sum()} pts)')
        leg = ax.get_legend()
        if leg is not None:
            lb = leg.get_window_extent(r)
            inside = (P[:, 0] > lb.x0) & (P[:, 0] < lb.x1) & (P[:, 1] > lb.y0) & (P[:, 1] < lb.y1)
            if inside.any():
                issues.append(f'legend/data in axis {fig.axes.index(ax)} ({inside.sum()} pts)')
    print(f'[{name}] {len(issues)} layout issues'); [print('   ', i) for i in issues]


def window(ax, ymark=True):
    ax.axvspan(*WINDOW, color='#EFEFEF', zorder=0, lw=0)
    ax.axvline(GOOD_SOLVENT, color=MUTED, lw=0.6, ls=':', zorder=1)


def binned(g, col, var, minn=10):
    b = pd.cut(g[col], BINS)
    m = g.groupby(b, observed=True)[var].agg(['mean', 'sem', 'size'])
    m = m[m['size'] >= minn]
    return np.array([i.mid for i in m.index]), m['mean'].values, m['sem'].values


def hinge(x, y):
    b = pd.cut(x, BINS)
    m = pd.DataFrame({'y': y, 'b': b}).groupby('b', observed=True).y.agg(['mean', 'size'])
    m = m[m['size'] >= 10]
    xs = np.array([i.mid for i in m.index]); ys = m['mean'].values; best = (None, -1)
    for c in np.arange(0.46, 0.66, 0.005):
        X = np.c_[np.ones_like(xs), np.maximum(xs - c, 0)]
        beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        r2 = 1 - np.sum((X @ beta - ys) ** 2) / np.sum((ys - ys.mean()) ** 2)
        if r2 > best[1]:
            best = (c, r2)
    return best[0]


def boot_hinge(g, col, var, n=200, seed=0):
    rng = np.random.default_rng(seed)
    v = [hinge(s[col].values, s[var].values)
         for s in (g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _ in range(n))]
    return np.percentile(v, [50, 2.5, 97.5])


KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
      'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}


def ch_hydropathy(s):
    """Mean Kyte-Doolittle hydropathy normalized to 0..1 over windows of five residues (Uversky et al. 2000)."""
    h = np.array([(KD.get(c, 0) + 4.5) / 9 for c in s])
    return h.mean() if len(h) < 5 else np.convolve(h, np.ones(5) / 5, mode='valid').mean()


def fcr_min_and_cross(g, col):
    b = pd.cut(g[col], BINS)
    m = g.groupby(b, observed=True)[['FCR', 'f_K', 'f_R']].mean(); n = g.groupby(b, observed=True).size(); m = m[n >= 20]
    mid = np.array([i.mid for i in m.index])
    fmin, fval = m.FCR.idxmin().mid, m.FCR.min()
    diff = (m.f_K - m.f_R).values; idx = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0]
    cross = mid[idx[0]] + 0.01 if len(idx) else np.nan
    return fmin, fval, cross


def load(bio_path):
    d = pd.read_csv(bio_path)
    d = d[d.nu_ok_cal.astype(bool) & d.nu_ok_rech.astype(bool)].dropna(subset=['nu_orig', 'nu'])
    d['logN'] = np.log(d.length); d['aN'] = d.NCPR.abs()
    return d.dropna(subset=DESC)


# ---------------------------------------------------------------- Figure 1
def fig1(d, out):
    rows = [('FCR', 'FCR'), ('aN', '|NCPR|'), ('NCPR', 'NCPR'), ('SCD', 'SCD'), ('f_aro', 'Aromatic fraction')]
    fig, axes = plt.subplots(len(rows), 6, figsize=(7.0, 6.0), sharex=True, sharey='row')
    for j, k in enumerate(TAXA):
        g = d[d.kingdom == k]
        for i, (var, lab) in enumerate(rows):
            ax = axes[i, j]; window(ax)
            if var == 'NCPR':
                ax.axhline(0, color=MUTED, lw=0.5)
            for col, ffn, ls, _ in FF:
                x, y, e = binned(g, col, var)
                ax.fill_between(x, y - e, y + e, color=FFC[col], alpha=0.22, lw=0)
                ax.plot(x, y, ls=ls, color=FFC[col], lw=1.3)
            if i == 0:
                ax.set_title(k, color=INK, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(lab)
            ax.set_xlim(0.40, 0.72); ax.set_xticks([0.45, 0.55, 0.65])
    handles = [Line2D([], [], color=CAL, lw=1.3, ls='-', label='CALVADOS-2'),
               Line2D([], [], color=RCH, lw=1.3, ls='--', label='Mpipi-Recharged'),
               Patch(color='#EFEFEF', label=r'Common window ($\nu$ 0.49 to 0.60)'),
               Line2D([], [], color=MUTED, lw=0.6, ls=':', label=r'Good solvent ($\nu$ = 0.588)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.supxlabel(r'Flory exponent $\nu$', y=0.04, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.3)
    check_layout(fig, 'fig1_axis')
    fig.savefig(f'{out}/fig1_axis.pdf'); fig.savefig(f'{out}/fig1_axis.png', dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig2(d, out):
    fig = plt.figure(figsize=(7.0, 4.9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15], hspace=0.62, wspace=1.6)
    gs2 = gs[1, :].subgridspec(1, 3, wspace=0.62)
    strips = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]
    titles = ['FCR minimum', '|NCPR| begins to rise', 'SCD begins to rise']
    rows = []
    for k in TAXA:
        g = d[d.kingdom == k]
        for col, ffn, ls, _ in FF:
            fmin, fval, cross = fcr_min_and_cross(g, col)
            n_m, n_lo, n_hi = boot_hinge(g, col, 'aN', seed=1)
            s_m, s_lo, s_hi = boot_hinge(g, col, 'SCD', seed=2)
            rows.append(dict(taxon=k, ff=ffn, fcr_min=fmin, fcr_min_value=fval, ncpr_onset=n_m, ncpr_lo=n_lo, ncpr_hi=n_hi,
                             scd_onset=s_m, scd_lo=s_lo, scd_hi=s_hi, lys_over_arg=cross))
    R = pd.DataFrame(rows); R.to_csv(f'{out}/fig2a_stage_markers.csv', index=False)
    keys = [('fcr_min', None, None), ('ncpr_onset', 'ncpr_lo', 'ncpr_hi'), ('scd_onset', 'scd_lo', 'scd_hi')]
    for si, (ax, title, (m, lo, hi)) in enumerate(zip(strips, titles, keys)):
        window(ax)
        for yi, k in enumerate(TAXA):
            for fi, (col, ffn, ls, _) in enumerate(FF):
                r = R[(R.taxon == k) & (R.ff == ffn)].iloc[0]; y = yi + (-0.16 if fi == 0 else 0.16)
                if lo:
                    ax.plot([r[lo], r[hi]], [y, y], color=FFC[col], lw=0.9, alpha=0.8, solid_capstyle='butt')
                ax.plot(r[m], y, 'o' if fi == 0 else 's', ms=4.5, color=FFC[col], mec='white', mew=0.5)
        ax.set_yticks(range(6)); ax.set_yticklabels(TAXA if si == 0 else [])
        ax.set_ylim(5.6, -0.6); ax.set_xlim(0.45, 0.65); ax.set_xticks([0.45, 0.55, 0.65])
        ax.set_xlabel(r'$\nu$')
        ax.set_title(title, fontsize=7, color=INK, pad=4)
    fig.text(0.015, 0.975, 'a', fontsize=9, fontweight='bold', color=INK, va='top')
    fig.text(0.045, 0.975, 'Charge transitions of every taxon along the axis', fontsize=7.5, color=INK, va='top')
    fig.legend(handles=[Line2D([], [], marker='o', ls='none', mfc=CAL, mec='white', ms=5, label='CALVADOS-2'),
                        Line2D([], [], marker='s', ls='none', mfc=RCH, mec='white', ms=5, label='Mpipi-Recharged')],
               frameon=False, fontsize=6, loc='upper right', bbox_to_anchor=(0.9, 0.99), ncol=2, handletextpad=0.1)

    b = fig.add_subplot(gs2[0, 0])
    edges = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 1.0)]
    labels = ['0 to\n0.05', '0.05 to\n0.1', '0.1 to\n0.2', '0.2 to\n0.35', 'above\n0.35']
    b.axhline(0, color=MUTED, lw=0.6)
    for fi, (col, ffn, ls, _) in enumerate(FF):
        rs = []
        for lo_, hi_ in edges:
            s = d[(d.aN > lo_) & (d.aN <= hi_)] if lo_ > 0 else d[d.aN <= hi_]
            rs.append(spearmanr(s.FCR, s[col]).correlation)
        b.plot(range(5), rs, ls=ls, color=FFC[col], lw=1.3, marker='o' if fi == 0 else 's', ms=4.5, mec='white', mew=0.5, label=ffn)
    b.set_xticks(range(5)); b.set_xticklabels(labels, fontsize=5.5)
    b.set_xlabel('|NCPR| bin'); b.set_ylabel(r'Spearman $\rho$ (FCR, $\nu$)'); b.set_ylim(-0.34, 0.5)
    b.text(0.02, 0.97, 'Added charge expands chains', transform=b.transAxes, color=MUTED, fontsize=6, va='top')
    b.text(0.02, 0.03, 'Added charge compacts chains', transform=b.transAxes, color=MUTED, fontsize=6, va='bottom')
    b.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.9))
    panel(b, 'b', 'The effect of charge changes sign', x=-0.3, y=1.04)

    c = fig.add_subplot(gs2[0, 1])
    fx = np.arange(0.0, 0.85, 0.05); ny = np.arange(0.0, 0.70, 0.05)
    H = np.full((len(ny) - 1, len(fx) - 1), np.nan)
    for i in range(len(ny) - 1):
        for j in range(len(fx) - 1):
            s = d[(d.FCR >= fx[j]) & (d.FCR < fx[j + 1]) & (d.aN >= ny[i]) & (d.aN < ny[i + 1])]
            if len(s) >= 10:
                H[i, j] = s.nu_orig.median()
    im = c.imshow(H, origin='lower', extent=(fx[0], fx[-1], ny[0], ny[-1]), aspect='auto', cmap='Blues', vmin=0.50, vmax=0.66)
    c.plot([0, 0.68], [0, 0.68], color=MUTED, lw=0.6, ls=':')
    c.set_xlabel('FCR'); c.set_ylabel('|NCPR|'); c.set_xlim(0, 0.8); c.set_ylim(0, 0.68)
    c.text(0.40, 0.012, 'Polyampholytes', color=INK, fontsize=5.5, ha='center', va='bottom')
    c.text(0.17, 0.50, 'Polyelectrolytes\n(|NCPR| = FCR)', color=INK, fontsize=5.5, ha='center')
    panel(c, 'c', 'Diagram of states', x=-0.3, y=1.04)

    # d, the charge-hydropathy plane of Uversky et al. 2000, on the same color scale as c
    e = fig.add_subplot(gs2[0, 2])
    if 'H_ch' not in d:
        d = d.assign(H_ch=d.sequence.map(ch_hydropathy))
    hx = np.arange(0.15, 0.575, 0.025)
    H2 = np.full((len(ny) - 1, len(hx) - 1), np.nan)
    for i in range(len(ny) - 1):
        for j in range(len(hx) - 1):
            s = d[(d.H_ch >= hx[j]) & (d.H_ch < hx[j + 1]) & (d.aN >= ny[i]) & (d.aN < ny[i + 1])]
            if len(s) >= 10:
                H2[i, j] = s.nu_orig.median()
    e.imshow(H2, origin='lower', extent=(hx[0], hx[-1], ny[0], ny[-1]), aspect='auto', cmap='Blues', vmin=0.50, vmax=0.66)
    rr = np.array([0, 0.68]); e.plot((rr + 1.151) / 2.785, rr, color=INK, lw=0.8, ls='--')
    cp = d[(d.FCR > 0.35) & (d.aN <= 0.05)]
    e.plot(cp.H_ch.median(), 0.025, marker='v', ms=5, color='#C0504D', mec='white', mew=0.5)
    e.annotate('Charge-rich,\nnearly neutral', (cp.H_ch.median(), 0.04), xytext=(0.20, 0.19), fontsize=5.5, color=INK,
               arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5))
    e.text(0.535, 0.62, 'Folded\nside', fontsize=5.5, color=INK, ha='center', va='top')
    e.text(0.22, 0.62, 'Disordered side', fontsize=5.5, color=INK, ha='left', va='top')
    e.set_xlabel('Mean hydropathy'); e.set_ylabel('|NCPR|'); e.set_xlim(0.15, 0.575); e.set_ylim(0, 0.68)
    cax = e.inset_axes([1.05, 0.0, 0.05, 1.0]); cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'Median $\nu$ (CALVADOS-2)', fontsize=6.5); cb.outline.set_visible(False)
    panel(e, 'd', 'Charge-hydropathy plot', x=-0.3, y=1.04)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88)
    check_layout(fig, 'fig2_stages')
    fig.savefig(f'{out}/fig2_stages.pdf'); fig.savefig(f'{out}/fig2_stages.png', dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig3(d, gg_path, saxs_path, out):
    D = ['FCR', 'aN', 'SCD', 'f_K', 'f_R', 'f_aro', 'hydropathy_avg', 'NCPR']
    DL = ['FCR', '|NCPR|', 'SCD', 'Lys', 'Arg', 'Aro', 'Hyd', 'NCPR']

    def curve(x, col, q=10):
        b = pd.qcut(x[col].rank(method='first'), q, labels=False)
        return x.groupby(b)[D].mean()
    M, rl = [], []
    for k in TAXA:
        x = d[d.kingdom == k]; a_, b_ = curve(x, 'nu_orig'), curve(x, 'nu')
        M.append([pearsonr(a_[v], b_[v])[0] for v in D]); rl.append(k)
    a_, b_ = curve(d, 'nu_orig'), curve(d, 'nu'); M.append([pearsonr(a_[v], b_[v])[0] for v in D]); rl.append('Pooled')
    g = pd.read_csv(gg_path); g = g[g.nu_r2_mpipi >= 0.99][['UniProt_ID', 'nu_mpipi']]
    G = d.merge(g, on='UniProt_ID'); a_, c_ = curve(G, 'nu_orig', 8), curve(G, 'nu_mpipi', 8)
    M.append([pearsonr(a_[v], c_[v])[0] for v in D]); rl.append('Mpipi-GG*')
    M = np.array(M)

    fig = plt.figure(figsize=(7.0, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], hspace=0.55, wspace=0.55)
    a = fig.add_subplot(gs[0, 0])
    im = a.imshow(M, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            a.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.5, color='white' if v > 0.7 else INK)
    a.set_xticks(range(len(DL))); a.set_xticklabels([l if i % 2 == 0 else '\n' + l for i, l in enumerate(DL)], fontsize=6)
    a.set_yticks(range(len(rl))); a.set_yticklabels(rl)
    a.axhline(5.5, color='white', lw=1.5); a.axhline(6.5, color='white', lw=1.5)
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=0)
    cax = a.inset_axes([1.03, 0.0, 0.035, 1.0]); cb = fig.colorbar(im, cax=cax); cb.set_label('Pearson r', fontsize=6.5, labelpad=1)
    cb.outline.set_visible(False)
    panel(a, 'a', 'Same shape in every force field', x=-0.2)

    bg = fig.add_subplot(gs[0, 1])
    edges = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 1.0)]
    G['aN'] = G.NCPR.abs(); bgrows = []
    bg.axhline(0, color=MUTED, lw=0.6)
    for col, ffn, colr, ls, mk in [('nu_orig', 'CALVADOS-2', CAL, '-', 'o'), ('nu_mpipi', 'Mpipi-GG', GG_LINE, '-.', 'D')]:
        rs, ns = [], []
        for lo_, hi_ in edges:
            s = G[(G.aN > lo_) & (G.aN <= hi_)] if lo_ > 0 else G[G.aN <= hi_]
            rs.append(spearmanr(s.FCR, s[col]).correlation); ns.append(len(s))
        bgrows.append(dict(ff=ffn, **{f'bin{i}': r for i, r in enumerate(rs)}))
        bg.plot(range(5), rs, ls=ls, color=colr, lw=1.3, marker=mk, ms=4.5, mec='white', mew=0.5, label=ffn)
    bg.set_xticks(range(5)); bg.set_xticklabels([f'{a_}\n(n={n})' for a_, n in zip(['0-0.05', '0.05-0.1', '0.1-0.2', '0.2-0.35', '>0.35'], ns)], fontsize=5.5)
    bg.set_xlabel('|NCPR| bin'); bg.set_ylabel(r'Spearman $\rho$ (FCR, $\nu$)'); bg.set_ylim(-0.35, 0.6)
    bg.legend(frameon=False, fontsize=5.5, loc='upper left')
    panel(bg, 'b', f'Mpipi-GG subset (n = {len(G)})', x=-0.3)
    pd.DataFrame(bgrows).to_csv(f'{out}/fig3b_gg_branch.csv', index=False)

    x = pd.read_csv(saxs_path)
    b = fig.add_subplot(gs[1, 0])
    lo = x[x.aN <= 0.1]; hi = x[x.aN > 0.1]
    rh = spearmanr(hi.FCR, hi.nueff_saxs); rl_ = spearmanr(lo.FCR, lo.nueff_saxs)
    b.scatter(lo.FCR, lo.nueff_saxs, s=13, facecolor='#D0D0D0', edgecolor='white', lw=0.4,
              label=f'|NCPR| up to 0.1, $\\rho$ = {rl_.correlation:+.2f} (n = {len(lo)})')
    b.scatter(hi.FCR, hi.nueff_saxs, s=13, facecolor=TERRA, edgecolor='white', lw=0.4,
              label=f'|NCPR| above 0.1, $\\rho$ = {rh.correlation:+.2f} (n = {len(hi)})')
    b.set_xlabel('FCR'); b.set_ylabel(r'Measured $\nu_{\mathrm{eff}}$ (SAXS)')
    b.legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.02, 1.03), fontsize=5.5, handletextpad=0.2)
    b.set_ylim(0.505, 0.72)
    panel(b, 'c', 'SAXS', x=-0.22)

    c = fig.add_subplot(gs[1, 1])
    rowsc = [('CALVADOS-2', 'pred_nueff_cal', 'nueff_cal_sim', CAL), ('Mpipi-Recharged', 'pred_nueff_rech', 'nueff_rech_sim', RCH)]
    for gi, (ffn, mcol, scol, fc) in enumerate(rowsc):
        rm = pearsonr(x[mcol], x.nueff_saxs)[0]; rs = pearsonr(x[scol], x.nueff_saxs)[0]
        mm = np.mean(np.abs(x[mcol] - x.nueff_saxs)); ms = np.mean(np.abs(x[scol] - x.nueff_saxs))
        c.bar(gi - 0.19, rm, width=0.36, color='white', edgecolor=fc, lw=1.4, hatch='////')
        c.bar(gi + 0.19, rs, width=0.36, color=fc, edgecolor=fc, lw=1.4)
        c.text(gi - 0.19, rm + 0.015, f'{rm:.2f}', ha='center', fontsize=5.5, color=INK)
        c.text(gi + 0.19, rs + 0.015, f'{rs:.2f}', ha='center', fontsize=5.5, color=INK)
        c.text(gi, -0.2, f'MAE {mm:.3f} / {ms:.3f}', ha='center', fontsize=5.5, color=MUTED, transform=c.get_xaxis_transform())
    c.set_xticks([0, 1]); c.set_xticklabels([r[0] for r in rowsc]); c.set_ylim(0, 0.9)
    c.set_ylabel('Pearson r with SAXS')
    c.legend(handles=[Patch(facecolor='white', edgecolor=MUTED, hatch='////', label='Nonlinear model'),
                      Patch(facecolor=MUTED, edgecolor=MUTED, label='Direct simulation')],
             frameon=False, fontsize=5.5, loc='upper right', handlelength=1.2)
    panel(c, 'd', 'Model vs simulation', x=-0.22)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.09, top=0.93)
    check_layout(fig, 'fig3_robust')
    fig.savefig(f'{out}/fig3_robust.pdf'); fig.savefig(f'{out}/fig3_robust.png', dpi=200); plt.close(fig)
    pd.DataFrame(M, index=rl, columns=DL).to_csv(f'{out}/fig3a_curve_agreement.csv')


# ---------------------------------------------------------------- Figure 4
def fig4(d, out, go_path):
    def reg(r):
        if r.FCR < 0.25: return 'Weak polyampholyte'
        if r.FCR <= 0.35: return 'Janus'
        if r.aN <= 0.35: return 'Strong polyampholyte'
        return 'Strong polyanion' if r.NCPR < 0 else 'Strong polycation'
    d = d.copy(); d['reg'] = d.apply(reg, axis=1)
    order = ['Weak polyampholyte', 'Janus', 'Strong polyampholyte', 'Strong polyanion', 'Strong polycation']
    rc = {'Weak polyampholyte': '#E5E5E5', 'Janus': '#C9C9C9', 'Strong polyampholyte': '#A6A6A6',
          'Strong polyanion': RED, 'Strong polycation': SLATE}
    occ = pd.crosstab(d.kingdom, d.reg, normalize='index').reindex(index=TAXA, columns=order) * 100

    fig, axs = plt.subplots(3, 2, figsize=(7.0, 7.4))
    a, b, e4, c, e5, e6 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]
    left = np.zeros(6)
    for r in order:
        a.barh(range(6), occ[r].values, left=left, color=rc[r], edgecolor='white', lw=0.8, height=0.72, label=r)
        left += occ[r].values
    for i, k in enumerate(TAXA):
        a.text(101, i, f"{occ.loc[k, 'Strong polycation']:.1f}% / {occ.loc[k, 'Strong polyanion']:.1f}%",
               va='center', fontsize=5.5, color=INK)
    a.set_yticks(range(6)); a.set_yticklabels(TAXA); a.invert_yaxis()
    a.set_xlim(0, 100); a.set_xlabel('Share of IDPs (%)')
    a.text(101, -0.9, 'Polycation / polyanion', fontsize=5.5, color=MUTED)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=5.5, handlelength=1)
    panel(a, 'a', 'Share in each region', x=-0.32)

    b.axvline(0, color=MUTED, lw=0.6)
    for fi, (col, ffn, ls, _) in enumerate(FF):
        e = d[d[col] > 0.63].groupby('kingdom').NCPR.mean().reindex(TAXA)
        b.barh(np.arange(6) + (-0.19 if fi == 0 else 0.19), e.values, height=0.36, color=FFC[col],
               edgecolor='white', lw=0.6)
    b.set_yticks(range(6)); b.set_yticklabels(TAXA); b.invert_yaxis(); b.set_ylim(5.6, -0.6)
    b.set_xlabel(r'Mean NCPR, expanded IDPs ($\nu$ > 0.63)'); b.set_xlim(-0.42, 0.55)
    b.legend(handles=[Patch(color=CAL, label='CALVADOS-2'), Patch(color=RCH, label='Mpipi-Recharged')],
             frameon=False, fontsize=5.5, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=1)
    b.text(0.02, 0.5, 'Acidic', transform=b.transAxes, fontsize=5.5, color=MUTED, rotation=90, va='center')
    b.text(0.98, 0.5, 'Basic', transform=b.transAxes, fontsize=5.5, color=MUTED, rotation=90, va='center', ha='right')
    panel(b, 'b', 'Charge sign at the expanded end', x=-0.3)

    # rows of taxa as in panels b and e: filled = observed mean nu, open = mean predicted without a taxon label
    for fi, (col, ffn, ls, _) in enumerate(FF):
        p = cross_val_predict(HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0),
                              d[DESC], d[col], cv=KFold(5, shuffle=True, random_state=0))
        dd = d.assign(p=p).groupby('kingdom').agg(act=(col, 'mean'), pred=('p', 'mean')).reindex(TAXA)
        dev_a = dd.act - dd.act.mean(); dev_r = dd.act - dd.pred
        r2 = 1 - np.sum(dev_r ** 2) / np.sum(dev_a ** 2)
        mk = 'o' if fi == 0 else 's'
        for yi, k in enumerate(TAXA):
            y = yi + (-0.19 if fi == 0 else 0.19)
            c.scatter([dd.loc[k, 'act']], [y], s=30, marker=mk, color=FFC[col], edgecolors='white', lw=0.4, zorder=2)
            c.plot([dd.loc[k, 'act'], dd.loc[k, 'pred']], [y, y], color=FFC[col], lw=1.3, solid_capstyle='butt', zorder=1)
            c.scatter([dd.loc[k, 'pred']], [y], s=44, marker=mk, facecolors='none', edgecolors=FFC[col], lw=1.1, zorder=3)
        c.text(0.03, 0.10 - 0.07 * fi, f'{"CALVADOS-2" if fi == 0 else "Recharged"} {100 * r2:.0f}%',
               transform=c.transAxes, fontsize=5.5, va='bottom', ha='left', color=INK)
    c.text(0.03, 0.17, 'Variance explained', transform=c.transAxes, fontsize=5.5, ha='left', va='bottom', color=MUTED)
    c.text(0.03, 0.98, 'Filled, observed\nOpen, predicted', transform=c.transAxes, fontsize=5.5, ha='left', va='top', color=MUTED)
    c.set_yticks(range(6)); c.set_yticklabels(TAXA); c.set_ylim(5.6, -0.6)
    c.set_xlim(0.522, 0.578); c.set_xlabel(r'Mean $\nu$ of each taxon')
    c.legend(handles=[Patch(color=CAL, label='CALVADOS-2'), Patch(color=RCH, label='Mpipi-Recharged')],
             frameon=False, fontsize=5.5, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=1)
    panel(c, 'd', 'Sequences set mean size', x=-0.3)
    G = pd.read_csv(go_path, sep='\t')
    terms = ['CC: nucleosome', 'MF: DNA binding', 'MF: structural constituent of chromatin', 'KW: DNA condensation',
             'CC: ribosome', 'BP: translation', 'KW: Ribonucleoprotein']
    nice = ['Nucleosome component', 'DNA binding', 'Chromatin structure', 'DNA condensation', 'Ribosome component', 'Translation',
            'Ribonucleoprotein']
    for fi, ffn in enumerate(['CALVADOS-2', 'Mpipi-Recharged']):
        x = G[(G.ff == ffn) & (G.group == 'expanded_basic')].set_index('term').reindex(terms)
        e4.barh(np.arange(len(terms)) + (-0.19 if fi == 0 else 0.19), x.odds.values - 1, left=1, height=0.36,
                color=CAL if fi == 0 else RCH, edgecolor='white', lw=0.6, label=ffn)
    e4.set_xscale('log'); e4.set_xlim(1, 200)
    e4.set_yticks(range(len(terms))); e4.set_yticklabels(nice); e4.set_ylim(len(terms) - 0.4, -0.6)
    e4.set_xlabel('Odds ratio against all annotated IDPs')
    e4.legend(frameon=False, fontsize=5.5, loc='lower right', handlelength=1)
    panel(e4, 'c', 'Expanded basic IDPs bind nucleic acids', x=-0.42)

    # e, linear vs flexible taxon offsets (mean out-of-fold residual per taxon)
    from sklearn.linear_model import LinearRegression
    erows = []
    for fi, (col, ffn, ls, _) in enumerate(FF):
        for mname, M in [('Linear', lambda: LinearRegression()),
                         ('Flexible', lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0))]:
            p = cross_val_predict(M(), d[DESC], d[col], cv=KFold(5, shuffle=True, random_state=0))
            res = (d[col] - p).groupby(d.kingdom).mean().reindex(TAXA)
            erows.append(dict(ff=ffn, model=mname, **res.to_dict()))
    E = pd.DataFrame(erows); E.to_csv(f'{out}/fig4e_taxon_offsets.csv', index=False)
    e5.axvline(0, color=MUTED, lw=0.6)
    e5.axvspan(-2.4, 2.4, color='#F2F2F2', lw=0, zorder=0)
    for fi, (col, ffn, ls, _) in enumerate(FF):
        rl_ = E[(E.ff == ffn) & (E.model == 'Linear')].iloc[0]; rf_ = E[(E.ff == ffn) & (E.model == 'Flexible')].iloc[0]
        for yi, k in enumerate(TAXA):
            y = yi + (-0.19 if fi == 0 else 0.19)
            e5.annotate('', xy=(rf_[k] * 1000, y), xytext=(rl_[k] * 1000, y),
                        arrowprops=dict(arrowstyle='-|>', color=FFC[col], lw=1.3, mutation_scale=7, shrinkA=0, shrinkB=0))
            e5.plot(rl_[k] * 1000, y, '|', color=FFC[col], ms=6, mew=1.3)
    e5.set_yticks(range(6)); e5.set_yticklabels(TAXA); e5.set_ylim(5.6, -0.6)
    e5.set_xlabel(r'Mean prediction error ($\nu \times 10^{3}$)')
    e5.set_xlim(-7.5, 12)
    e5.text(0.98, 0.98, 'Tick, linear\nArrowhead, nonlinear\nShaded, $\\pm$0.0024', transform=e5.transAxes,
            fontsize=5.5, color=MUTED, ha='right', va='top')
    e5.legend(handles=[Patch(color=CAL, label='CALVADOS-2'), Patch(color=RCH, label='Mpipi-Recharged')],
              frameon=False, fontsize=5.5, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=1)
    panel(e5, 'e', 'Linear fits leave false taxon offsets', x=-0.3)

    # f, leave one taxon out
    frows = []
    for fi, (col, ffn, ls, _) in enumerate(FF):
        for k in TAXA:
            g = d[d.kingdom == k]; tr = d[d.kingdom != k]
            pw = cross_val_predict(HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0),
                                   g[DESC], g[col], cv=KFold(5, shuffle=True, random_state=0))
            pl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(tr[DESC], tr[col]).predict(g[DESC])
            frows.append(dict(ff=ffn, taxon=k, within=pearsonr(pw, g[col])[0] ** 2, held_out=pearsonr(pl, g[col])[0] ** 2))
    Fd = pd.DataFrame(frows); Fd.to_csv(f'{out}/fig4f_leave_one_taxon_out.csv', index=False)
    # dumbbells in the layout of panel e: open marker = within-taxon cross-validation, filled = taxon held out
    for fi, (col, ffn, ls, _) in enumerate(FF):
        s_ = Fd[Fd.ff == ffn].set_index('taxon')
        mk = 'o' if fi == 0 else 's'
        for yi, k in enumerate(TAXA):
            y = yi + (-0.19 if fi == 0 else 0.19)
            w, h = s_.loc[k, 'within'], s_.loc[k, 'held_out']
            e6.plot([w, h], [y, y], color=FFC[col], lw=1.3, solid_capstyle='butt', zorder=1)
            e6.scatter([w], [y], s=26, marker=mk, facecolors='white', edgecolors=FFC[col], lw=1.1, zorder=2)
            e6.scatter([h], [y], s=26, marker=mk, color=FFC[col], edgecolors='white', lw=0.4, zorder=3)
    e6.set_yticks(range(6)); e6.set_yticklabels(TAXA); e6.set_ylim(5.6, -0.6)
    e6.set_xlim(0.55, 0.92)
    e6.set_xlabel(r'$R^2$ of the nonlinear model')
    e6.text(0.02, 0.98, 'Open, within taxon\nFilled, taxon held out', transform=e6.transAxes,
            fontsize=5.5, color=MUTED, ha='left', va='top')
    e6.legend(handles=[Patch(color=CAL, label='CALVADOS-2'), Patch(color=RCH, label='Mpipi-Recharged')],
              frameon=False, fontsize=5.5, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=1)
    panel(e6, 'f', 'The model transfers between taxa', x=-0.3)
    fig.tight_layout(w_pad=2.0, h_pad=2.4, rect=(0, 0, 0.985, 1))
    check_layout(fig, 'fig4_occupancy')
    fig.savefig(f'{out}/fig4_occupancy.pdf'); fig.savefig(f'{out}/fig4_occupancy.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    bio, gg, saxs, go, out = sys.argv[1:6]
    d = load(bio)
    which = sys.argv[6:] or ['1', '2', '3', '4']
    if '1' in which: fig1(d, out)
    if '2' in which: fig2(d, out)
    if '3' in which: fig3(d, gg, saxs, out)
    if '4' in which: fig4(d, out, go)
    print('done', which)
