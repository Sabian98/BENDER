#!/usr/bin/env python3
"""SAXS validation figures. Stages: `compute` caches CALVADOS-2 and MPIPI-GG I(q); `plot` draws Fig. 1, Fig. 2,
the force-field comparison and the supplementary Kratky pages."""
import csv, json, math, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/
import bender_paths as paths  # noqa: E402  (data and trajectory locations: paths.py at the repository root)
import bender_common as bc  # noqa: E402  (frame selection, MPIPI-GG unwrapping)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

EXP = paths.SAXS                      # processed/ holds the SASBDB entry and condition tables
BENCH = paths.SAXS
FIG = BENCH / "figures"
RAW = paths.SASBDB
TRAJ_DIR = paths.TRAJ_SAXS / "calvados2"
MPIPI_DIR = paths.TRAJ_SAXS / "mpipi_gg"
CACHE = FIG / "calvados_iq_cache.json"
CACHE_MP = FIG / "mpipi_iq_cache.json"
OUT = paths.outdir("saxs")

# soft palette of the BENDER preprint: simulation (CALVADOS-2) powder blue at
# line weight, experiment (SAXS) warm dark neutral; MPIPI-GG peach (fill; deeper at line weight).
C_EXP, C_SIM = "#33302C", "#5B8DB8"
C_MP, C_MP_LINE = "#E0937A", "#CD7B5F"
INK, INK2, INK3, BAND = "#1f1f1f", "#52514e", "#8a8984", "#ecebe7"
POINT_C, RULE_C = "#9A938A", "#DDD6CB"
Q_MAX_SIM = 2.5
Q_GRID = np.linspace(0.05, Q_MAX_SIM, 120)
TARGET_FRAMES = 200
R_BINS = 600
TAXON_ORDER = ["Human", "Virus", "Other vertebrate", "Plant", "Bacteria", "Fungi", "Protist", "Synthetic"]
IDP_TIERS = ("disordered", "disordered_by_nu")
SHORT_NAMES = [
    ("Isoform Tau-F of Microtubule-associated protein tau", "Tau K18"), ("Microtubule-associated protein tau", "Tau"),
    ("Microtubule-associated protein 2", "MAP2c"), ("Mothers against decapentaplegic homolog 2", "SMAD2 linker"),
    ("Mothers against decapentaplegic homolog 4", "SMAD4 linker"), ("Peroxisome proliferator-activated receptor gamma", "PPARγ NTD"),
    ("Peroxisome proliferator-activated receptor alpha", "PPARα NTD"), ("Transient receptor potential cation channel", "TRPV N-terminus"),
    ("Segment S(", "NF-L segment S("), ("Neurofilament light polypeptide", "NF-L tail"),
    ("Retinoic acid-induced protein 2", "RAI2"), ("Eukaryotic initiation factor 4F subunit p150", "eIF4F p150"),
    ("Translocated intimin receptor", "Tir"), ("Nuclear receptor coactivator 2", "NCOA2"),
    ("Chromatin assembly factor 1 subunit A", "CHAF1A"), ("Polyglutamine-binding protein 1", "PQBP1"),
    ("Cellular tumor antigen p53", "p53 TAD"), ("N-myc proto-oncogene protein", "N-Myc"), ("Non-structural protein V", "V protein"),
    ("Aryl hydrocarbon receptor nuclear translocator-like", "BMAL1"), ("Nuclear receptor CoRepressor 1", "N-CoR NID"),
    ("Partner and localizer of BRCA2", "PALB2"), ("Isoform 2 of B-cell CLL/lymphoma 7", "BCL7B"),
    ("Basic domain of telomeric repeat-binding factor 2", "TRF2 basic domain"), ("Myelin basic protein", "MBP"),
    ("La-related protein 6", "LARP6"), ("DNA repair protein XRCC4", "XRCC4"), ("Tag + Intracellular region", "PCDH15 cytoplasmic"),
    ("Elastin-like polypeptide", "Elastin-like polypeptide"), ("V5 variant of the low complexity domain of hnRNPA1", "hnRNPA1 LCD V5"),
    ("Human alpha-synuclein", "α-Synuclein"), ("Alpha-synuclein V1 K102R", "α-Synuclein K102R"),
    ("Colicin N Translocation domain", "Colicin N T-domain"), ("NURS complex subunit red1", "Red1"), ("Early E1A protein", "E1A"),
    ("Histatin-3, His3-(20-43)", "Histatin-3 (20–43)"), ("Estrogen receptor", "ERα NTD"),
    ("Huntingtin", "Huntingtin"), ("Beclin-1", "Beclin-1 NTD"), ("Nucleoprotein", "N protein linker"),
    ("Circumsporozoite protein", "CSP"), ("Protein W", "W protein"), ("Phosphoprotein", "Phosphoprotein"),
    ("Filaggrin-like protein", "Filaggrin-like protein"), ("NAC domain-containing protein", "NAC"),
    ("Δ91 construct of FATZ-1", "FATZ-1 Δ91"), ("N-ter construct of FATZ-1", "FATZ-1 N-term"),
    ("Elastin", "Tropoelastin"), ("Nuclear receptor coactivator", "NCOA")]

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    # upright maths in Arial; Greek via \\mathsf (DejaVu Sans) so nu differs from v
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "DejaVu Sans",
    "axes.linewidth": 0.8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8.5, "legend.fontsize": 7,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
})


def load_rows():
    rg = {r["traj_id"]: r for r in csv.DictReader(open(BENCH / "traj_vs_saxs_rg.csv"))}
    conds = {r["condition_id"]: r for r in csv.DictReader(open(EXP / "processed/sasbdb_conditions.csv"))}
    rows = []
    for r in csv.DictReader(open(BENCH / "traj_vs_saxs_nu.csv")):
        if r["flag"]:
            continue
        rows.append({**r, "taxon": "Human" if r["taxon"] == "Unassigned" else r["taxon"],
                     "n_res": int(r["n_res"]), "traj_rg_nm": float(r["traj_rg_nm"]), "saxs_rg_nm": float(r["saxs_rg_nm"]),
                     "nu_traj": float(r["nu_traj"]), "nu_saxs": float(r["nu_saxs"]), "delta_nu": float(r["delta_nu"]),
                     "independent": r["calvados2_overlap"] == "none", "strict": r["set"] == "strict",
                     "label": r["uniprot_id"] or r["sasbdb_id"], "cond": conds[r["saxs_condition_id"]],
                     "saxs_rg_err_nm": rg[r["traj_id"]]["saxs_rg_err_nm"],
                     "mpipi_rg_nm": float(r["mpipi_rg_nm"]) if r.get("mpipi_rg_nm") else None,
                     "nu_mpipi": float(r["nu_mpipi"]) if r.get("nu_mpipi") else None,
                     "delta_nu_mpipi": float(r["delta_nu_mpipi"]) if r.get("delta_nu_mpipi") else None})
    return rows


def debye_iq(xyz):
    """I(q)/I(0) from C-alpha coordinates (frames, N, 3) via the frame-averaged pair-distance histogram."""
    from scipy.spatial.distance import pdist
    dists = [pdist(f) for f in xyz]
    edges = np.linspace(0, max(x.max() for x in dists) * 1.05, R_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pr = sum(np.histogram(x, bins=edges)[0] for x in dists) / len(dists)
    n = xyz.shape[1]
    qr = np.outer(Q_GRID, centers)
    sinc = np.where(qr > 1e-8, np.sin(qr) / np.where(qr > 1e-8, qr, 1), 1.0)
    return ((n + 2.0 * sinc @ pr) / n ** 2).tolist()


def sim_iq(traj_id):
    """CALVADOS-2: production half, 200 evenly spaced frames."""
    d = TRAJ_DIR / traj_id
    xyz, _, _ = bc.load_calvados(str(d / f"{traj_id}.dcd"), str(d / "top.pdb"), k=TARGET_FRAMES)
    return debye_iq(xyz)


def mpipi_iq(traj_id, n_res):
    """MPIPI-GG: every frame is production; unwrapped, 200 evenly spaced frames."""
    xyz, _ = bc.load_mpipi(str(MPIPI_DIR / traj_id / "traj.xtc"), n_res=n_res, k=TARGET_FRAMES)
    return debye_iq(xyz)


def compute():
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    for path, fn in ((CACHE, lambda r: sim_iq(r["traj_id"])), (CACHE_MP, lambda r: mpipi_iq(r["traj_id"], r["n_res"]))):
        cache = json.load(open(path)) if path.exists() else {}
        todo = [r for r in rows if r["traj_id"] not in cache and (path == CACHE or r["mpipi_rg_nm"] is not None)]
        for i, r in enumerate(todo, 1):
            cache[r["traj_id"]] = fn(r)
            json.dump({"q": Q_GRID.tolist(), **{k: v for k, v in cache.items() if k != "q"}}, open(path, "w"))
            print(f"  {path.name} {i}/{len(todo)} {r['traj_id']}", flush=True)


def representative_entry(r, entries):
    cands = [entries[e] for e in r["saxs_entries"].split(";") if entries[e].get("tier") in IDP_TIERS]
    target = r["saxs_rg_nm"]
    return min(cands, key=lambda e: (abs(float(e["rg_nm"]) - target), e["sasbdb_id"]))["sasbdb_id"]


def exp_kratky(code):
    d = json.load(open(RAW / "summaries" / f"{code}.json"))
    rows = []
    for line in (RAW / f"{code}.dat").read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) >= 2:
            try:
                rows.append((float(p[0]), float(p[1])))
            except ValueError:
                pass
    q, I = np.array(rows).T
    unit = str(d.get("angular_unit") or "")
    if unit == "1/A" or (unit != "1/nm" and q.max() < 1.5):
        q = q * 10.0
    rg = float(d["guinier_rg"])
    windows = []
    pf, pl = d.get("guinier_point_first"), d.get("guinier_point_last")
    if pf and pl and 1 <= pf < pl <= len(q):
        windows.append(np.arange(pf - 1, pl))
    windows.append(np.where((q * rg >= 0.2) & (q * rg <= 1.3))[0])
    i0 = None
    for w in windows:
        ok = w[I[w] > 0]
        if len(ok) >= 5:
            slope, icpt = np.polyfit(q[ok] ** 2, np.log(I[ok]), 1)
            if slope < 0 and abs(np.sqrt(-3 * slope) - rg) / rg < 0.15:
                i0 = float(np.exp(icpt))
                break
    x = q * rg
    k = x ** 2 * I / i0
    keep = (x > 0) & (x <= 8)
    x, k = x[keep], k[keep]
    width = max(0.1, 3 * 8 / max(len(x), 1))
    edges = np.arange(0, 8 + width, width)
    idx = np.digitize(x, edges)
    xb, kb = [], []
    for b in range(1, len(edges)):
        sel = idx == b
        if sel.sum() >= 2:
            xb.append(float(np.mean(x[sel])))
            kb.append(float(np.median(k[sel])))
    return np.array(xb), np.array(kb), rg


def marker_style(r, color, size=26):
    shape = "o" if r["strict"] else "D"
    s = size if shape == "o" else size * 0.75
    if r["independent"]:
        return dict(marker=shape, s=s, facecolors=color, edgecolors=INK, linewidths=0.4)
    return dict(marker=shape, s=s, facecolors="white", edgecolors=color, linewidths=1.1)


def encoding_legend(ax, color, loc, **kw):
    h = [Line2D([0], [0], marker="o", ls="", markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.4, markersize=5),
         Line2D([0], [0], marker="o", ls="", markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1, markersize=5),
         Line2D([0], [0], marker="D", ls="", markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.4, markersize=4.3),
         Line2D([0], [0], marker="D", ls="", markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1, markersize=4.3)]
    ax.legend(h, ["Primary, independent of CALVADOS-2", "Primary, in CALVADOS-2 training/test",
                  "No UniProt mapping", "No UniProt mapping, in CALVADOS-2 training"], loc=loc,
              handletextpad=0.3, labelspacing=0.35, borderaxespad=0.2, **kw)


def panel_label(ax, s):
    ax.text(-0.17, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK)


def fig_summary(rows):
    rs = np.array([r["traj_rg_nm"] for r in rows]); rx = np.array([r["saxs_rg_nm"] for r in rows])
    ns = np.array([r["nu_traj"] for r in rows]); nx = np.array([r["nu_saxs"] for r in rows])
    n = np.array([r["n_res"] for r in rows]); dnu = ns - nx
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.08, wspace=0.34, hspace=0.42)

    ax = axes[0, 0]
    lim = (0.8, 8.0)
    xx = np.linspace(*lim, 50)
    ax.fill_between(xx, 0.8 * xx, 1.2 * xx, color=BAND, zorder=0, lw=0)
    ax.plot(lim, lim, color=INK3, lw=0.9, ls="--", zorder=1)
    for r in rows:
        ax.scatter(r["saxs_rg_nm"], r["traj_rg_nm"], zorder=3, **marker_style(r, C_SIM))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SAXS $R_g$ (nm)"); ax.set_ylabel("CALVADOS-2 $R_g$ (nm)")
    ax.text(0.97, 0.04, f"$n$ = {len(rows)}\nMARE = {100 * np.mean(np.abs(rs - rx) / rx):.1f}%\n$r$ = {np.corrcoef(rs, rx)[0, 1]:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2, linespacing=1.4)
    encoding_legend(fig, C_SIM, "upper center", bbox_to_anchor=(0.54, 1.03), ncol=2, fontsize=7, columnspacing=1.6)
    panel_label(ax, "a")

    ax = axes[0, 1]
    lim = (0.36, 0.72)
    ax.fill_between(lim, [lim[0] - 0.05, lim[1] - 0.05], [lim[0] + 0.05, lim[1] + 0.05], color=BAND, zorder=0, lw=0)
    ax.plot(lim, lim, color=INK3, lw=0.9, ls="--", zorder=1)
    ax.axhline(0.588, color=INK3, lw=0.7, ls=":", zorder=1)
    ax.text(lim[0] + 0.005, 0.592, "good solvent ($\\mathsf{\\nu}$ = 0.588)", fontsize=6.3, color=INK3, va="bottom")
    for r in rows:
        ax.scatter(r["nu_saxs"], r["nu_traj"], zorder=3, **marker_style(r, C_SIM))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SAXS $\\mathsf{\\nu}_{\\mathrm{eff}}$"); ax.set_ylabel("CALVADOS-2 $\\mathsf{\\nu}_{\\mathrm{eff}}$")
    ax.text(0.97, 0.04, f"$\\langle\\Delta\\mathsf{{\\nu}}\\rangle$ = {dnu.mean():+.3f} $\\pm$ {dnu.std(ddof=1):.3f}\n"
            f"{100 * np.mean(np.abs(dnu) < 0.05):.0f}% within $|\\Delta\\mathsf{{\\nu}}|$ < 0.05",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2, linespacing=1.4)
    panel_label(ax, "b")

    ax = axes[1, 0]
    ax.scatter(n, rx, marker="o", s=20, facecolors="white", edgecolors=C_EXP, linewidths=1.0, zorder=3)
    ax.scatter(n, rs, marker="o", s=16, facecolors=C_SIM, edgecolors=INK, linewidths=0.35, zorder=4)
    grid = np.linspace(n.min() * 0.85, n.max() * 1.15, 200)
    fits = {}
    for key, y, color, ls in (("SAXS", rx, C_EXP, "--"), ("CALVADOS-2", rs, C_SIM, "-")):
        nu_fit, logr0 = np.polyfit(np.log(n), np.log(y), 1)
        r2 = np.corrcoef(np.log(n), np.log(y))[0, 1] ** 2
        fits[key] = (nu_fit, r2)
        ax.plot(grid, np.exp(logr0) * grid ** nu_fit, color=color, lw=1.6, ls=ls, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([20, 50, 100, 200, 500]); ax.set_yticks([1, 2, 3, 5, 8])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.ScalarFormatter()); axis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Chain length $N$ (residues)"); ax.set_ylabel("$R_g$ (nm)")
    ax.legend([Line2D([0], [0], marker="o", color=C_SIM, markerfacecolor=C_SIM, markeredgecolor=INK, markeredgewidth=0.35, markersize=4.5, lw=1.6, ls="-"),
               Line2D([0], [0], marker="o", color=C_EXP, markerfacecolor="white", markeredgecolor=C_EXP, markersize=4.5, lw=1.6, ls="--")],
              [f"CALVADOS-2  $\\mathsf{{\\nu}}$ = {fits['CALVADOS-2'][0]:.3f} ($R^2$ = {fits['CALVADOS-2'][1]:.2f})",
               f"SAXS  $\\mathsf{{\\nu}}$ = {fits['SAXS'][0]:.3f} ($R^2$ = {fits['SAXS'][1]:.2f})"], loc="upper left", handlelength=2.2)
    panel_label(ax, "c")

    ax = axes[1, 1]
    present = [t for t in TAXON_ORDER if any(r["taxon"] == t for r in rows)]
    ax.axvspan(-0.05, 0.05, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=INK3, lw=0.9, ls="--", zorder=1)
    rng = np.random.default_rng(7)
    for yi, t in enumerate(present):
        sub = [r for r in rows if r["taxon"] == t]
        jit = rng.uniform(-0.22, 0.22, len(sub))
        for r, j in zip(sub, jit):
            ax.scatter(r["delta_nu"], yi + j, zorder=3, **marker_style(r, C_SIM, size=18))
        m = np.mean([r["delta_nu"] for r in sub])
        ax.plot([m, m], [yi - 0.34, yi + 0.34], color=INK, lw=1.6, zorder=4, solid_capstyle="round")
        prots = len({protein_key(r) for r in sub})
        ax.text(0.125, yi, f"{len(sub)} ({prots})", va="center", ha="left", fontsize=6.8, color=INK2)
    ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.12, 0.12); ax.set_ylim(len(present) - 0.5, -0.5)
    ax.set_xlabel("$\\Delta\\mathsf{\\nu} = \\mathsf{\\nu}_{\\mathrm{sim}} - \\mathsf{\\nu}_{\\mathrm{SAXS}}$")
    ax.text(0.125, -0.95, "constructs\n(proteins)", fontsize=6.3, color=INK3, ha="left", va="bottom", linespacing=1.1)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    panel_label(ax, "d")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_saxs_summary.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_forcefields(rows):
    """CALVADOS-2 and MPIPI-GG against SAXS on the same constructs."""
    pr = [r for r in rows if r["mpipi_rg_nm"] is not None]
    rx = np.array([r["saxs_rg_nm"] for r in pr]); nx = np.array([r["nu_saxs"] for r in pr])
    ffs = [("CALVADOS-2", C_SIM, "#3F6E96", "traj_rg_nm", "nu_traj", "delta_nu"),
           ("MPIPI-GG", C_MP, "#A9573F", "mpipi_rg_nm", "nu_mpipi", "delta_nu_mpipi")]
    col = lambda k: np.array([r[k] for r in pr])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.08, wspace=0.34, hspace=0.42)
    dot = lambda c: dict(s=17, facecolors=c, edgecolors=INK, linewidths=0.3, zorder=3)
    handle = lambda c: Line2D([0], [0], marker="o", ls="", markerfacecolor=c, markeredgecolor=INK, markeredgewidth=0.3, markersize=4.6)

    ax = axes[0, 0]
    lim = (0.8, 10.0)
    xx = np.linspace(*lim, 50)
    ax.fill_between(xx, 0.8 * xx, 1.2 * xx, color=BAND, zorder=0, lw=0)
    ax.plot(lim, lim, color=INK3, lw=0.9, ls="--", zorder=1)
    labels = []
    for name, c, _, rk, _, _ in ffs:
        rs = col(rk)
        ax.scatter(rx, rs, **dot(c))
        labels.append(f"{name}  MARE {100 * np.mean(np.abs(rs - rx) / rx):.1f}%, $r$ = {np.corrcoef(rs, rx)[0, 1]:.3f}")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SAXS $R_g$ (nm)"); ax.set_ylabel("Simulated $R_g$ (nm)")
    ax.legend([handle(f[1]) for f in ffs], labels, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    panel_label(ax, "a")

    ax = axes[0, 1]
    lim = (0.36, 0.72)
    ax.fill_between(lim, [lim[0] - 0.05, lim[1] - 0.05], [lim[0] + 0.05, lim[1] + 0.05], color=BAND, zorder=0, lw=0)
    ax.plot(lim, lim, color=INK3, lw=0.9, ls="--", zorder=1)
    labels = []
    for name, c, _, _, nk, dk in ffs:
        d = col(dk)
        ax.scatter(nx, col(nk), **dot(c))
        labels.append(f"{name}  $\\langle\\Delta\\mathsf{{\\nu}}\\rangle$ = {d.mean():+.3f} $\\pm$ {d.std(ddof=1):.3f}; "
                      f"{100 * np.mean(np.abs(d) < 0.05):.0f}% < 0.05")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SAXS $\\mathsf{\\nu}_{\\mathrm{eff}}$"); ax.set_ylabel("Simulated $\\mathsf{\\nu}_{\\mathrm{eff}}$")
    ax.legend([handle(f[1]) for f in ffs], labels, loc="upper left", handletextpad=0.2, borderaxespad=0.2, fontsize=6.6)
    panel_label(ax, "b")

    ax = axes[1, 0]
    dc, dm = col("delta_nu"), col("delta_nu_mpipi")
    lim = (-0.10, 0.10)
    ax.add_patch(plt.Rectangle((-0.05, -0.05), 0.10, 0.10, color=BAND, zorder=0, lw=0))
    ax.plot(lim, lim, color=INK3, lw=0.9, ls="--", zorder=1)
    ax.axhline(0, color=RULE_C, lw=0.6, zorder=1); ax.axvline(0, color=RULE_C, lw=0.6, zorder=1)
    ax.scatter(dc, dm, s=17, facecolors=POINT_C, edgecolors=INK, linewidths=0.3, zorder=3)
    # hand-placed labels for the constructs outside |dnu| < 0.05 (offsets in points)
    place = {"SASDQS8": ((5, 0), "left", "center"), "SASDNV3": ((-5, 3), "right", "center"),
             "SASDEE2": ((6, 2), "left", "center"), "P04637": ((0, 6), "center", "bottom"),
             "SASDQJ7": ((0, -6), "center", "top"), "SASDSZ9": ((5, 0), "left", "center")}
    for r, x, y in zip(pr, dc, dm):
        if max(abs(x), abs(y)) >= 0.05:
            off, ha, va = place.get(r["traj_id"], ((5, 0), "left", "center"))
            ax.annotate(short_name(r["protein_name"], 18), (x, y), xytext=off, textcoords="offset points",
                        fontsize=6.0, color=INK2, ha=ha, va=va, zorder=4)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks([-0.10, -0.05, 0, 0.05, 0.10]); ax.set_yticks([-0.10, -0.05, 0, 0.05, 0.10])
    ax.set_xlabel("$\\Delta\\mathsf{\\nu}$, CALVADOS-2"); ax.set_ylabel("$\\Delta\\mathsf{\\nu}$, MPIPI-GG")
    ax.text(0.03, 0.97, f"$r$ = {np.corrcoef(dc, dm)[0, 1]:.2f}\nMPIPI-GG closer to SAXS\nfor {100 * np.mean(np.abs(dm) < np.abs(dc)):.0f}% of constructs",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0, color=INK2, linespacing=1.35)
    panel_label(ax, "c")

    ax = axes[1, 1]
    present = [t for t in TAXON_ORDER if any(r["taxon"] == t for r in pr)]
    ax.axvspan(-0.05, 0.05, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=INK3, lw=0.9, ls="--", zorder=1)
    rng = np.random.default_rng(7)
    for yi, t in enumerate(present):
        sub = [r for r in pr if r["taxon"] == t]
        for off, (name, c, dark, _, _, dk) in zip((-0.19, 0.19), ffs):
            v = np.array([r[dk] for r in sub])
            ax.scatter(v, yi + off + rng.uniform(-0.1, 0.1, len(v)), s=12, facecolors=c, edgecolors=INK, linewidths=0.3, zorder=3)
            ax.plot([v.mean()] * 2, [yi + off - 0.15, yi + off + 0.15], color=dark, lw=1.6, zorder=4, solid_capstyle="round")
        ax.text(0.125, yi, f"{len(sub)} ({len({protein_key(r) for r in sub})})", va="center", ha="left", fontsize=6.8, color=INK2)
    ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=7.5)
    ax.set_xlim(-0.12, 0.12); ax.set_ylim(len(present) - 0.5, -0.5)
    ax.set_xlabel("$\\Delta\\mathsf{\\nu} = \\mathsf{\\nu}_{\\mathrm{sim}} - \\mathsf{\\nu}_{\\mathrm{SAXS}}$")
    ax.text(0.125, -0.95, "constructs\n(proteins)", fontsize=6.3, color=INK3, ha="left", va="bottom", linespacing=1.1)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend([handle(f[1]) for f in ffs], [f[0] for f in ffs], loc="lower left", fontsize=6.8, handletextpad=0.2,
              borderaxespad=0.2, bbox_to_anchor=(0.0, 1.0), ncol=2)
    panel_label(ax, "d")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_saxs_forcefields.{ext}", bbox_inches="tight")
    plt.close(fig)


def pick_representatives(rows):
    reps = []
    for t in TAXON_ORDER:
        pool = [r for r in rows if r["taxon"] == t and r["strict"] and r["independent"]]
        if not pool:
            continue
        med = float(np.median([abs(r["delta_nu"]) for r in pool]))
        reps.append(min(pool, key=lambda r: (abs(abs(r["delta_nu"]) - med), -int(r["saxs_n_entries"]), -r["n_res"], r["label"])))
    return reps


def short_name(name, width=34):
    variant = ""
    m = re.search(r"\(([^)]*)\)", name)
    muts = re.findall(r"[A-Z]\d+[A-Z]", m.group(1)) if m else []
    if muts:
        variant = " " + ", ".join(muts[:2]) + (f" +{len(muts) - 2}" if len(muts) > 2 else "")
    for prefix, short in SHORT_NAMES:
        if name.startswith(prefix):
            if prefix == "Segment S(":
                return "NF-L " + name.split(" of ")[0].replace("Segment ", "segment ")
            if prefix == "NAC domain-containing protein":
                return "NAC" + name.split("protein")[-1].strip()
            rest = name[len(prefix):].split("(")[0].strip(" ,;")
            if name[len(prefix):].startswith(",") and rest:
                return rest[:24] if rest.startswith(short) else f"{short} {rest[:20]}"
            return short + variant
    base = name.split(",")[0].split(" (")[0]
    base = base if len(base) <= width else base[: width - 1] + "…"
    return base + variant


def fmt_dnu(v):
    return "0.000" if abs(v) < 0.0005 else f"{v:+.3f}"


def protein_key(r):
    if r["uniprot_id"]:
        return r["uniprot_id"]
    n = r["protein_name"].lower()
    for k in ("synuclein", "neurofilament", "elastin-like", "low complexity domain", "huntingtin", "protein w", "protocadherin"):
        if k in n:
            return k
    return n


def kratky_panel(ax, r, cache, entries, title_size=7.6, mark="", cache_mp=None):
    code = representative_entry(r, entries)
    xb, kb, rg = exp_kratky(code)
    q = np.array(cache["q"]); iq = np.array(cache[r["traj_id"]])
    xs = q * rg
    ax.axvline(math.sqrt(3), color=BAND, lw=1.0, zorder=0)
    ax.plot(xb, kb, color=C_EXP, lw=1.5, zorder=3)
    ksim = xs ** 2 * iq
    ax.plot(xs, ksim, color=C_SIM, lw=1.4, ls=(0, (4, 1.6)), zorder=4)
    tops = [np.nanmax(ksim[xs <= 7])]
    if cache_mp is not None and r["traj_id"] in cache_mp:
        kmp = xs ** 2 * np.array(cache_mp[r["traj_id"]])
        ax.plot(xs, kmp, color=C_MP_LINE, lw=1.8, ls=(0, (0.1, 2.0)), dash_capstyle="round", zorder=5)
        tops.append(np.nanmax(kmp[xs <= 7]))
    top = max(np.nanmax(kb[xb <= 7]) if len(kb) else 0, *tops)
    ax.set_xlim(0, 7); ax.set_ylim(0, max(2.2, top * 1.12))
    ax.set_title(f"{short_name(r['protein_name'])}{mark}", fontsize=title_size, color=INK, pad=3, loc="left")
    ident = f"{r['label']} ({r['traj_id']})" if r["traj_id"] != r["label"] else r["label"]
    dnu = fmt_dnu(r["delta_nu"]) + (f" / {fmt_dnu(r['delta_nu_mpipi'])}" if r["delta_nu_mpipi"] is not None else "")
    ax.text(0.97, 0.04, f"{ident} · {r['taxon']}\n$N$ = {r['n_res']}, $\\Delta\\mathsf{{\\nu}}$ = {dnu}",
            transform=ax.transAxes, fontsize=6.2, color=INK2, va="bottom", ha="right", linespacing=1.3)
    return code


def fig_representative(rows, cache, entries, cache_mp=None):
    reps = pick_representatives(rows)
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.6))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.12, wspace=0.28, hspace=0.55)
    used = []
    for ax, r in zip(axes.flat, reps):
        used.append((r["traj_id"], kratky_panel(ax, r, cache, entries, cache_mp=cache_mp)))
    for i, ax in enumerate(axes.flat):
        if i >= len(reps):
            ax.axis("off")
            continue
        if i % 4 == 0:
            ax.set_ylabel("$(qR_g)^2\\, I(q)/I(0)$")
        if i >= 4 or i + 4 >= len(reps):
            ax.set_xlabel("$qR_g$")
    fig.legend([Line2D([0], [0], color=C_EXP, lw=1.5), Line2D([0], [0], color=C_SIM, lw=1.4, ls=(0, (4, 1.6))),
                Line2D([0], [0], color=C_MP_LINE, lw=1.8, ls=(0, (0.1, 2.0)), dash_capstyle="round")],
               ["SAXS (SASBDB)", "CALVADOS-2", "MPIPI-GG"], loc="lower right", bbox_to_anchor=(0.97, 0.13), fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_saxs_kratky_representative.{ext}", bbox_inches="tight")
    plt.close(fig)
    return reps, used


def fig_supplementary(rows, cache, entries, reps, cache_mp=None):
    rep_ids = {r["traj_id"] for r in reps}
    rest = sorted([r for r in rows if r["traj_id"] not in rep_ids],
                  key=lambda r: (TAXON_ORDER.index(r["taxon"]), not r["strict"], r["label"], r["traj_id"]))
    per = 8
    pages = [rest[i:i + per] for i in range(0, len(rest), per)]
    manifest = []
    for p, chunk in enumerate(pages, 1):
        fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.6))
        fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.12, wspace=0.28, hspace=0.55)
        for i, ax in enumerate(axes.flat):
            if i >= len(chunk):
                ax.axis("off")
                continue
            r = chunk[i]
            mark = ("†" if not r["independent"] else "") + ("§" if not r["strict"] else "")
            code = kratky_panel(ax, r, cache, entries, mark=mark, cache_mp=cache_mp)
            manifest.append({"figure": f"S{p}", "traj_id": r["traj_id"], "label": r["label"], "protein": r["protein_name"],
                             "taxon": r["taxon"], "sasbdb_entry_plotted": code})
            if i % 4 == 0:
                ax.set_ylabel("$(qR_g)^2\\, I(q)/I(0)$")
            if i >= 4 or i + 4 >= len(chunk):
                ax.set_xlabel("$qR_g$")
        fig.legend([Line2D([0], [0], color=C_EXP, lw=1.5), Line2D([0], [0], color=C_SIM, lw=1.4, ls=(0, (4, 1.6))),
                Line2D([0], [0], color=C_MP_LINE, lw=1.8, ls=(0, (0.1, 2.0)), dash_capstyle="round")],
                   ["SAXS (SASBDB)", "CALVADOS-2", "MPIPI-GG"], loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=3, fontsize=7.5)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"figS_saxs_kratky_{p:02d}.{ext}", bbox_inches="tight")
        plt.close(fig)
    with open(FIG / "kratky_panel_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    return pages


def plot():
    rows = load_rows()
    cache = json.load(open(CACHE))
    cache_mp = json.load(open(CACHE_MP)) if CACHE_MP.exists() else None
    entries = {r["sasbdb_id"]: r for r in csv.DictReader(open(EXP / "processed/sasbdb_entries_screened.csv"))}
    fig_summary(rows)
    fig_forcefields(rows)
    reps, used = fig_representative(rows, cache, entries, cache_mp)
    with open(FIG / "representative_panels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["taxon", "traj_id", "label", "protein", "n_res", "delta_nu", "sasbdb_entry_plotted"])
        for r, (_, code) in zip(reps, used):
            w.writerow([r["taxon"], r["traj_id"], r["label"], r["protein_name"], r["n_res"], r["delta_nu"], code])
    pages = fig_supplementary(rows, cache, entries, reps, cache_mp)
    print("representatives:", [(r["taxon"], r["traj_id"], r["label"], round(r["delta_nu"], 3)) for r in reps])
    print("supplementary pages:", len(pages), [len(p) for p in pages])


if __name__ == "__main__":
    {"compute": compute, "plot": plot}[sys.argv[1]]()
