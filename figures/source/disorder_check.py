"""IDRome-style disorder check of the 10,282 BENDER chains (2026-10-21).

Reproduces the IDR selection of Tesei et al. (Nature 2024; KULL-Center/_2023_Tesei_IDRome, idr_selection/code/
disordered_regions.r) on every BENDER sequence, to ask whether each chain, selected from MobiDB consensus disorder,
is also fully disordered by AlphaFold2 confidence:

  1. per-residue AlphaFold2 pLDDT (0-1) averaged over a 20-residue window (IDRome used the 20-residue windowed
     pLDDT of Akdel et al.); 10- and 40-residue windows as sensitivity checks
  2. <pLDDT> < 0.7 disordered (D), > 0.8 folded (F), otherwise gap (g)
  3. D and F runs shorter than 10 residues become gaps
  4. a gap becomes D if flanked by D on both sides, or if terminal and next to D; every other gap becomes F
  5. IDRs are D runs of 30-1,500 residues
  A chain is fully disordered when the whole sequence is one IDR.
  f_domain: fraction of residues inside UniProt 'Domain', 'Zinc finger', 'Coiled coil' or 'Transmembrane'
  features that overlap the chain by at least 10 residues (IDRome's rule).

Sources: per-residue pLDDT from MobiDB ('prediction-plddt-alphafold', which mirrors the AlphaFold Database; the AFDB
file server refuses direct downloads from here), UniProt features from the UniProt REST API, and, for a random 500
chains, the whole-model pLDDT summary of the latest AFDB model from the AFDB API as a cross-check of the MobiDB copy.

Stages:
  python disorder_check.py fetch      (cached in data/disorder_check/, safe to rerun)
  python disorder_check.py analyze    (per-sequence CSV, summary, SI figure)
"""
import gzip
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import bender_paths as paths  # noqa: E402  (data and output locations: paths.py at the repository root)

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DATA = paths.DISORDER
BIO = paths.BIO
ORDER = ["Mammals", "Fungi", "Plants", "Protists", "Bacteria", "Viruses"]
FEATURES = ("Domain", "Zinc finger", "Coiled coil", "Transmembrane")
UA = {"User-Agent": "BENDER-disorder-check (academic; taseefr@usf.edu)"}


# ---------------------------------------------------------------------------------------------- fetch --
MIN_INTERVAL = {"alphafold.ebi.ac.uk": 0.6, "rest.uniprot.org": 0.2}   # seconds between requests per host
_last, _lock = {}, threading.Lock()


def _throttle(url):
    host = urllib.parse.urlparse(url).netloc
    gap = MIN_INTERVAL.get(host, 0.0)
    with _lock:
        wait = _last.get(host, 0.0) + gap - time.monotonic()
        _last[host] = max(time.monotonic(), _last.get(host, 0.0) + gap)
    if wait > 0:
        time.sleep(wait)


def get(url, tries=4, timeout=60, max_429=12):
    k, n429, err = 0, 0, None
    while k < tries:
        _throttle(url)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                raw = r.read()
                if raw[:2] == b"\x1f\x8b":                  # gzip body without a decoded Content-Encoding
                    raw = gzip.decompress(raw)
                return r.status, raw
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, b""
            if e.code == 429 and n429 < max_429:           # rate limited: honour Retry-After, do not count as a try
                n429 += 1
                ra = e.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.replace(".", "").isdigit() else 30 * n429)
                continue
            err = e
        except Exception as e:  # noqa: BLE001  (network hiccups: retry)
            err = e
        k += 1
        time.sleep(2 * k)
    raise RuntimeError(f"{url}: {err}")


def cached(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if "error" in rec and rec["acc"] in done and "error" not in done[rec["acc"]]:
                    continue
                done[rec["acc"]] = rec
    return done


def fetch_mobidb(accs, path, batch=100):
    done = cached(path)
    todo = [a for a in accs if a not in done]
    with open(path, "a") as fh:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            url = ("https://mobidb.org/api/download?acc=" + ",".join(chunk)
                   + "&projection=acc,sequence,prediction-plddt-alphafold&format=json")
            status, raw = get(url)
            text = raw.decode()
            try:
                recs = json.loads(text)
                recs = recs if isinstance(recs, list) else [recs]
            except json.JSONDecodeError:
                recs = [json.loads(l) for l in text.splitlines() if l.strip()]
            got = set()
            for r in recs:
                pl = r.get("prediction-plddt-alphafold") or {}
                fh.write(json.dumps({"acc": r["acc"], "sequence": r.get("sequence"),
                                     "plddt": pl.get("scores")}) + "\n")
                got.add(r["acc"])
            for a in set(chunk) - got:                       # not in MobiDB any more
                fh.write(json.dumps({"acc": a, "sequence": None, "plddt": None}) + "\n")
            fh.flush()
            print(f"  mobidb {min(i + batch, len(todo))}/{len(todo)}", flush=True)
            time.sleep(0.5)


def fetch_uniprot(accs, path, batch=80):
    done = cached(path)
    todo = [a for a in accs if a not in done]
    with open(path, "a") as fh:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            q = " OR ".join(f"accession:{a}" for a in chunk)
            url = ("https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
                {"query": q, "fields": "accession,length,ft_domain,ft_zn_fing,ft_coiled,ft_transmem",
                 "format": "json", "size": 500}))
            status, raw = get(url)
            res = json.loads(raw)["results"]
            got = set()
            for r in res:
                feats = [(f["type"], f["location"]["start"]["value"], f["location"]["end"]["value"])
                         for f in r.get("features", []) if f["type"] in FEATURES]
                fh.write(json.dumps({"acc": r["primaryAccession"], "length": r.get("sequence", {}).get("length"),
                                     "features": feats}) + "\n")
                got.add(r["primaryAccession"])
            for a in set(chunk) - got:                       # merged, demerged or obsolete entries
                fh.write(json.dumps({"acc": a, "length": None, "features": None}) + "\n")
            fh.flush()
            print(f"  uniprot {min(i + batch, len(todo))}/{len(todo)}", flush=True)
            time.sleep(0.3)


def fetch_afdb(accs, path, n=500, seed=0):
    done = cached(path)
    pick = sorted(np.random.default_rng(seed).choice(sorted(accs), n, replace=False))
    todo = [a for a in pick if a not in done]

    def one(a):
        status, raw = get(f"https://alphafold.ebi.ac.uk/api/prediction/{a}")
        if status == 404 or not raw:
            return {"acc": a, "model": None}
        m = json.loads(raw)[0]
        keep = ("latestVersion", "globalMetricValue", "fractionPlddtVeryLow", "fractionPlddtLow",
                "fractionPlddtConfident", "fractionPlddtVeryHigh", "uniprotStart", "uniprotEnd", "sequence")
        return {"acc": a, "model": {k: m.get(k) for k in keep}}

    with open(path, "a") as fh, ThreadPoolExecutor(3) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            fh.write(json.dumps(rec) + "\n")
            if i % 50 == 0:
                fh.flush()
                print(f"  afdb {i}/{len(todo)}", flush=True)


def fetch_afdb_plddt(accs, path):
    """Per-residue pLDDT (0-100) of the latest AFDB model; v6 file first, else the API's plddtDocUrl."""
    done = {a: r for a, r in cached(path).items() if "error" not in r}
    todo = [a for a in accs if a not in done]
    todo = [todo[i] for i in np.random.default_rng(1).permutation(len(todo))]   # partial results stay unbiased

    def one(a):
        try:
            return _one(a)
        except Exception as e:  # noqa: BLE001
            return {"acc": a, "version": None, "plddt": None, "error": str(e)[:200]}

    def _one(a):
        st, raw = get(f"https://alphafold.ebi.ac.uk/files/AF-{a}-F1-confidence_v6.json")
        version = 6
        if st == 404 or not raw:
            st, meta = get(f"https://alphafold.ebi.ac.uk/api/prediction/{a}")
            if st == 404 or not meta:
                return {"acc": a, "version": None, "plddt": None}
            m = json.loads(meta)[0]
            version = m.get("latestVersion")
            st, raw = get(m["plddtDocUrl"])
            if st == 404 or not raw:
                return {"acc": a, "version": version, "plddt": None}
        doc = json.loads(raw)
        doc = doc[0] if isinstance(doc, list) else doc
        return {"acc": a, "version": version, "plddt": doc["confidenceScore"]}

    with open(path, "a") as fh:
        for i, a in enumerate(todo, 1):
            fh.write(json.dumps(one(a)) + "\n")
            fh.flush()
            if i % 200 == 0:
                print(f"  afdb plddt {i}/{len(todo)}", flush=True)


FT_KEYS = {"DOMAIN": "Domain", "ZN_FING": "Zinc finger", "COILED": "Coiled coil", "TRANSMEM": "Transmembrane"}


def fetch_unisave(accs, path):
    """Last archived release of obsolete UniProt entries: sequence length and the four feature types."""
    done = {a: r for a, r in cached(path).items() if "error" not in r}
    todo = [a for a in accs if a not in done]

    def one(a):
        try:
            return _one(a)
        except Exception as e:  # noqa: BLE001
            return {"acc": a, "length": None, "features": None, "entry_version": None, "error": str(e)[:200]}

    def _one(a):
        st, raw = get(f"https://rest.uniprot.org/unisave/{a}?format=json")
        res = json.loads(raw).get("results", []) if raw else []
        if not res:
            return {"acc": a, "length": None, "features": None, "entry_version": None}
        last = max(r["entryVersion"] for r in res)
        st, raw = get(f"https://rest.uniprot.org/unisave/{a}?format=txt&versions={last}")
        length, feats = None, []
        for line in raw.decode().splitlines():
            if line.startswith("SQ   SEQUENCE"):
                length = int(line.split()[2])
            elif line.startswith("FT   ") and line[5:21].strip() in FT_KEYS:
                loc = line[21:].strip().replace("<", "").replace(">", "")
                if ".." in loc:
                    a0, a1 = loc.split("..")
                    if a0.isdigit() and a1.isdigit():
                        feats.append((FT_KEYS[line[5:21].strip()], int(a0), int(a1)))
        return {"acc": a, "length": length, "features": feats, "entry_version": last}

    with open(path, "a") as fh, ThreadPoolExecutor(2) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            fh.write(json.dumps(rec) + "\n")
            if i % 200 == 0:
                fh.flush()
                print(f"  unisave {i}/{len(todo)}", flush=True)


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(BIO, usecols=["UniProt_ID", "kingdom"])
    accs = b.loc[b.kingdom.isin(ORDER), "UniProt_ID"].tolist()
    assert len(accs) == 10282, len(accs)
    fetch_mobidb(accs, DATA / "mobidb_plddt.jsonl")
    fetch_uniprot(accs, DATA / "uniprot_features.jsonl")
    fetch_afdb(accs, DATA / "afdb_subset.jsonl")
    fetch_afdb_plddt(accs, DATA / "afdb_plddt.jsonl")
    uni = cached(DATA / "uniprot_features.jsonl")
    fetch_unisave([a for a in accs if uni.get(a, {}).get("length") is None], DATA / "unisave_features.jsonl")
    print("fetched")


# -------------------------------------------------------------------------------------------- analyze --
def windowed(p, w):
    """Centered moving average over w residues, truncated at the termini."""
    c = np.concatenate([[0.0], np.cumsum(p)])
    i = np.arange(len(p))
    lo, hi = np.maximum(0, i - w // 2), np.minimum(len(p), i - w // 2 + w)
    return (c[hi] - c[lo]) / (hi - lo)


def runs(lab):
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append([lab[s], s, i])
            s = i
    return out


def idrome_labels(wp, min_run=10):
    lab = np.where(wp < 0.7, "D", np.where(wp > 0.8, "F", "g"))
    for t, s, e in runs(lab):                                # short D/F runs become gaps
        if t in "DF" and e - s < min_run:
            lab[s:e] = "g"
    rr = runs(lab)
    for k, (t, s, e) in enumerate(rr):
        if t != "g":
            continue
        left = rr[k - 1][0] if k > 0 else None
        right = rr[k + 1][0] if k + 1 < len(rr) else None
        flanked = left == "D" and right == "D"
        terminal = (left is None and right == "D") or (right is None and left == "D")
        lab[s:e] = "D" if (flanked or terminal) else "F"
    return lab


def per_sequence(seq, pl, w):
    wp = windowed(np.asarray(pl, float), w)
    lab = idrome_labels(wp)
    rr = runs(lab)
    idr = [(s, e) for t, s, e in rr if t == "D" and 30 <= e - s <= 1500]
    in_idr = sum(e - s for s, e in idr)
    folded = [e - s for t, s, e in rr if t == "F"]
    return {"full": len(idr) == 1 and idr[0] == (0, len(seq)), "frac_idr": in_idr / len(seq),
            "n_idr": len(idr), "longest_folded": max(folded, default=0)}


def f_domain(n, feats, min_overlap=10):
    cover = np.zeros(n, bool)
    for _, s, e in feats or []:
        s, e = max(1, int(s)), min(n, int(e))
        if e - s + 1 >= min_overlap:
            cover[s - 1:e] = True
    return cover.mean()


def analyse():
    sys.path.insert(0, str(HERE))
    import make_fig_disagreement as fd                         # same loading and contested-class definition as Fig. 2
    b, mr, fm = fd.load()
    seqs = b["sequence"]
    afp, mob = cached(DATA / "afdb_plddt.jsonl"), cached(DATA / "mobidb_plddt.jsonl")
    uni, uns = cached(DATA / "uniprot_features.jsonl"), cached(DATA / "unisave_features.jsonl")
    rows = []
    for acc, seq in seqs.items():
        rec = {"acc": acc, "group": b.at[acc, "kingdom"], "n": len(seq)}
        a, m = afp.get(acc, {}), mob.get(acc, {})
        if a.get("plddt") and len(a["plddt"]) == len(seq):
            p, src = np.asarray(a["plddt"], float) / 100, f"AFDB v{a.get('version')}"
        elif m.get("plddt") and m.get("sequence") == seq:
            p, src = np.asarray(m["plddt"], float), "MobiDB"
        else:
            p, src = None, ("no model" if not a.get("plddt") else "length differs")
        rec["plddt_source"] = src
        if m.get("plddt") and m.get("sequence") == seq and a.get("plddt") and len(a["plddt"]) == len(seq):
            rec["mobidb_mean_plddt"] = float(np.mean(m["plddt"]))
        if p is not None:
            rec.update(mean_plddt=p.mean(), frac_plddt_lt70=(p < 0.7).mean(), frac_plddt_gt80=(p > 0.8).mean())
            for w in (10, 20, 40):
                rec.update({f"{k}_w{w}": v for k, v in per_sequence(seq, p, w).items()})
        u = uni.get(acc, {})
        if u.get("length") == len(seq):
            feats, fsrc = u.get("features") or [], "UniProt"
        elif uns.get(acc, {}).get("length") == len(seq):
            feats, fsrc = uns[acc].get("features") or [], "UniSave"
        else:
            feats, fsrc = None, "missing"
        rec["feature_source"] = fsrc
        if feats is not None:
            rec["f_domain"] = f_domain(len(seq), feats)
            rec["domain_types"] = ";".join(sorted({t for t, s0, e0 in feats if int(e0) - int(s0) + 1 >= 10}))
        rows.append(rec)
    d = pd.DataFrame(rows).set_index("acc")
    d.to_csv(DATA / "bender_disorder_check.csv")

    lines = []
    P = lambda s="": lines.append(s)  # noqa: E731
    have = d[d.full_w20.notna()].copy()
    have["full"] = have.full_w20.astype(bool)
    P(f"pLDDT: {len(have):,} of {len(d):,} chains ({d.plddt_source.value_counts().to_dict()})")
    for w in (20, 10, 40):
        full = have[f"full_w{w}"].astype(bool)
        P(f"window {w}: fully disordered {full.sum():,} ({100 * full.mean():.1f}%); "
          f">= 90% of residues in IDRs {100 * (have[f'frac_idr_w{w}'] >= 0.9).mean():.1f}%; "
          f">= 50% {100 * (have[f'frac_idr_w{w}'] >= 0.5).mean():.1f}%; no IDR {100 * (have[f'n_idr_w{w}'] == 0).mean():.1f}%")
    resid = np.average(have.frac_idr_w20, weights=have.n)
    P(f"residues inside IDRs (w20): {100 * resid:.1f}%; residues with raw pLDDT < 0.7: "
      f"{100 * np.average(have.frac_plddt_lt70, weights=have.n):.1f}%, > 0.8: "
      f"{100 * np.average(have.frac_plddt_gt80, weights=have.n):.1f}%")
    P(f"median per-chain mean pLDDT {have.mean_plddt.median():.3f}; chains with a folded run >= 30 residues: "
      f"{100 * (have.longest_folded_w20 >= 30).mean():.1f}%")
    P()
    P("by group (w20): n | fully disordered | >= 90% in IDRs | residues in IDRs | median mean pLDDT")
    for g in fd.ORDER:
        sg = have[have.group == g]
        P(f"  {g:9s} {len(sg):5d} | {100 * sg.full.mean():5.1f}% | {100 * (sg.frac_idr_w20 >= 0.9).mean():5.1f}% | "
          f"{100 * np.average(sg.frac_idr_w20, weights=sg.n):5.1f}% | {sg.mean_plddt.median():.3f}")
    cmp_ = d.dropna(subset=["mobidb_mean_plddt", "mean_plddt"])
    if len(cmp_):
        P(f"MobiDB vs AFDB latest, per-chain mean pLDDT ({len(cmp_)} chains): r = "
          f"{np.corrcoef(cmp_.mobidb_mean_plddt, cmp_.mean_plddt)[0, 1]:.3f}, median |diff| "
          f"{np.median(np.abs(cmp_.mobidb_mean_plddt - cmp_.mean_plddt)):.3f}")

    fdd = d.dropna(subset=["f_domain"])
    P()
    P(f"f_domain: {len(fdd):,} chains ({d.feature_source.value_counts().to_dict()})")
    P(f"  > 0: {100 * (fdd.f_domain > 0).mean():.1f}%; > 0.5: {100 * (fdd.f_domain > 0.5).mean():.1f}%; "
      f"> 0.9: {100 * (fdd.f_domain > 0.9).mean():.1f}%   (Human-IDRome IDRs: 22%, 5.1%, 3.4%)")
    types = fdd.domain_types.dropna().str.split(";").explode()
    P(f"  feature types: {types[types != ''].value_counts().to_dict()}")
    for g in fd.ORDER:
        sg = fdd[fdd.group == g]
        P(f"  {g:9s} f_domain > 0: {100 * (sg.f_domain > 0).mean():5.1f}%  > 0.5: {100 * (sg.f_domain > 0.5).mean():4.1f}%")
    both = have.join(fdd[["f_domain"]].rename(columns={"f_domain": "fdom"}), how="inner")
    clean = both.full & (both.fdom == 0)
    P(f"  fully disordered by pLDDT and f_domain = 0: {clean.sum():,} of {len(both):,} ({100 * clean.mean():.1f}%)")

    # does any conclusion depend on the chains that fail the check? --------------------------------------
    dnu = mr["nu"] - b["nu"]
    ra = b["f_R"] + b["f_aro"]
    cls = (ra > ra.quantile(0.9)) & (b["f_K"] < b["f_K"].median())       # defined on all 10,282, as in the paper
    P()
    P("robustness (w20): subset | n | r(nu) | mean dnu | SD | contested n, mean dnu | |dnu|>0.1 | rho(dnu, f_aro)")
    from scipy import stats
    for name, idx in (("all", b.index), ("fully disordered", have.index[have.full]),
                      ("not fully disordered", have.index[~have.full])):
        x, y, dd = b.loc[idx, "nu"], mr.loc[idx, "nu"], dnu.loc[idx]
        c = cls.loc[idx]
        P(f"  {name:21s} {len(idx):6,d} | {np.corrcoef(x, y)[0, 1]:.3f} | {dd.mean():+.4f} | {dd.std():.3f} | "
          f"{int(c.sum()):4d}, {dd[c].mean():+.3f} | {100 * (dd.abs() > 0.1).mean():4.1f}% | "
          f"{stats.spearmanr(b.loc[idx, 'f_aro'], dd)[0]:+.2f}")
    r_pl = stats.spearmanr(have.mean_plddt, dnu.loc[have.index])[0]
    P(f"  Spearman(mean pLDDT, dnu) = {r_pl:+.2f}; Spearman(mean pLDDT, nu C2) = "
      f"{stats.spearmanr(have.mean_plddt, b.loc[have.index, 'nu'])[0]:+.2f}, (nu MR) = "
      f"{stats.spearmanr(have.mean_plddt, mr.loc[have.index, 'nu'])[0]:+.2f}")
    bad = b.loc[have.index, "nu_fit_r2"] < 0.99
    P(f"  Flory fits R2 < 0.99 (CALVADOS-2): {100 * bad[have.full].mean():.1f}% of fully disordered, "
      f"{100 * bad[~have.full].mean():.1f}% of the rest")
    P()
    P("r(nu) by group: all chains | AF2 fully disordered (n) | AF2 not fully (n) | not-fully median f_K vs fully")
    for g in fd.ORDER:
        ia = b.index[b.kingdom == g]
        i1 = have.index[have.full & (have.group == g)]
        i0 = have.index[~have.full & (have.group == g)]
        rr = lambda idx: np.corrcoef(b.loc[idx, "nu"], mr.loc[idx, "nu"])[0, 1]  # noqa: E731
        P(f"  {g:9s} {rr(ia):.3f} | {rr(i1):.3f} ({len(i1):5,d}) | {rr(i0):.3f} ({len(i0):5,d}) | "
          f"{b.loc[i0, 'f_K'].median():.3f} vs {b.loc[i1, 'f_K'].median():.3f}")

    mp = DATA / "metapredict.csv"
    if mp.exists():
        m = pd.read_csv(mp).set_index("acc")
        d2 = d.join(m[["full_V2", "frac_idr_V2", "full_V3"]])
        d2.to_csv(DATA / "bender_disorder_check.csv")
        P()
        for v in ("V2", "V3"):
            P(f"metapredict {v}: fully disordered {int(m[f'full_{v}'].sum()):,} ({100 * m[f'full_{v}'].mean():.1f}%)")
        x = d2.dropna(subset=["full_w20"])
        P(f"  AF2 fully disordered among metapredict-V2 fully disordered: "
          f"{100 * x.loc[x.full_V2.astype(bool), 'full_w20'].astype(bool).mean():.1f}%")
    (DATA / "bender_disorder_check_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _accs():
    b = pd.read_csv(BIO, usecols=["UniProt_ID", "kingdom"])
    return b.loc[b.kingdom.isin(ORDER), "UniProt_ID"].tolist()


def fetch_plddt():
    fetch_afdb_plddt(_accs(), DATA / "afdb_plddt.jsonl")
    print("fetched afdb")


def fetch_obsolete():
    uni = cached(DATA / "uniprot_features.jsonl")
    fetch_unisave([a for a in _accs() if uni.get(a, {}).get("length") is None], DATA / "unisave_features.jsonl")
    print("fetched unisave")


if __name__ == "__main__":
    {"fetch": fetch, "analyse": analyse, "fetch-plddt": fetch_plddt, "fetch-unisave": fetch_obsolete}[sys.argv[1]]()
