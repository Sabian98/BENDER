"""
graph_features.py — per-frame contact-graph features (v1 definitions, unchanged), vendored from
Scripts/traj_analysis_calvados.py before that script was retired. The v1 bug was in the adjacency
(|i−j| ≥ 2); v2 builds graphs with bender_common.adjacency (|i−j| ≥ 3) and passes them in here.
"""

import numpy as np
import networkx as nx

BETWEENNESS_K = 30     # pivot approximation for betweenness (as v1)


def graph_features_from_adj(adj: np.ndarray,
                             sequence: str = "") -> dict:
    """
    adj      : (n_frames, n_res, n_res) boolean
    sequence : amino acid sequence string (used for CV computation)
    Compute sequence- and residue-level graph features averaged over frames.
    CV computed correctly — only on residues where mean betweenness > 0.
    Uses approximate betweenness for speed.
    """
    n_frames, n_res, _ = adj.shape

    # accumulators
    seq_acc = {
        "fragmentation_index": [],
        "global_efficiency":   [],
        "avg_clustering":      [],
        "transitivity":        [],
        "degree_assortativity":[],
    }
    res_acc = {
        "degree_centrality":      [],
        "betweenness_centrality": [],
        "harmonic_centrality":    [],
        "pagerank":               [],
        "core_number":            [],
        "local_clustering":       [],
        "in_lcc":                 [],
    }

    nodes = list(range(n_res))

    for f in range(n_frames):
        G = nx.from_numpy_array(adj[f].astype(int))

        # ── sequence-level ───────────────────────────────────────
        comps    = list(nx.connected_components(G))
        lcc      = max(comps, key=len)
        frag_idx = len(lcc) / n_res

        seq_acc["fragmentation_index"].append(frag_idx)
        seq_acc["global_efficiency"].append(nx.global_efficiency(G))
        seq_acc["avg_clustering"].append(nx.average_clustering(G))
        seq_acc["transitivity"].append(nx.transitivity(G))

        try:
            seq_acc["degree_assortativity"].append(
                nx.degree_assortativity_coefficient(G))
        except Exception:
            seq_acc["degree_assortativity"].append(np.nan)

        # ── residue-level ────────────────────────────────────────
        # degree centrality — O(N), instant
        dc = nx.degree_centrality(G)
        res_acc["degree_centrality"].append(
            [dc.get(i, 0.0) for i in nodes])

        # betweenness — approximated with k pivots, O(k*N*E)
        try:
            bc = nx.betweenness_centrality(G, k=BETWEENNESS_K,
                                           normalized=True)
        except Exception:
            bc = {i: 0.0 for i in nodes}
        res_acc["betweenness_centrality"].append(
            [bc.get(i, 0.0) for i in nodes])

        # harmonic centrality — O(N^2) but fast in practice
        try:
            hc = nx.harmonic_centrality(G)
        except Exception:
            hc = {i: 0.0 for i in nodes}
        res_acc["harmonic_centrality"].append(
            [hc.get(i, 0.0) for i in nodes])

        # pagerank — fast iterative
        try:
            pr = nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception:
            pr = {i: 0.0 for i in nodes}
        res_acc["pagerank"].append(
            [pr.get(i, 0.0) for i in nodes])

        # core number
        try:
            cn = nx.core_number(G)
        except Exception:
            cn = {i: 0 for i in nodes}
        res_acc["core_number"].append(
            [float(cn.get(i, 0)) for i in nodes])

        # local clustering
        lc = nx.clustering(G)
        res_acc["local_clustering"].append(
            [lc.get(i, 0.0) for i in nodes])

        # in LCC
        lcc_set = set(lcc)
        res_acc["in_lcc"].append(
            [1.0 if i in lcc_set else 0.0 for i in nodes])

    # ── average over frames ──────────────────────────────────────
    seq_out = {}
    for k, vals in seq_acc.items():
        arr = np.array(vals, dtype=float)
        seq_out[k]          = float(np.nanmean(arr))
        seq_out[k + "_std"] = float(np.nanstd(arr))

    res_out = {}
    for k, vals in res_acc.items():
        arr = np.array(vals, dtype=float)            # (n_frames, n_res)
        res_out[k + "_mean"] = np.nanmean(arr, axis=0)
        res_out[k + "_std"]  = np.nanstd(arr,  axis=0)

    # ── CV computation — CORRECT METHOD ──────────────────────────────────────
    # CV = std/mean computed ONLY where mean > 0
    # Residues with mean=0 get CV=NaN (they never participate in network)
    # This avoids zero-inflation artifact for rare amino acids (W, C, etc.)
    #
    # Key distinction:
    #   WRONG: CV across all sequences including those where AA is absent
    #          → mean pulled toward 0, CV inflated/unstable
    #   RIGHT: CV per residue position where mean betweenness > 0
    #          → reflects genuine temporal variability of hub behavior

    for k in list(res_acc.keys()):
        mean_arr = res_out[k + "_mean"]   # (n_res,)
        std_arr  = res_out[k + "_std"]    # (n_res,)
        cv_arr   = np.where(
            mean_arr > 1e-10,             # residue actually participates
            std_arr / (mean_arr + 1e-12), # genuine CV
            np.nan                        # absent residue → NaN, not 0
        )
        res_out[k + "_cv"] = cv_arr       # (n_res,) CV per residue position

    # ── per-AA CV arrays (for kingdom-level hub CV analysis) ─────────────────
    # For each amino acid, store the CV values of all residues of that AA
    # in this protein — as a variable-length array indexed by AA identity
    # This enables correct cross-kingdom CV comparison:
    #   cv_by_aa["W"] = CV values of all W residues in this protein
    #   Aggregated across proteins in post-processing
    if sequence:
        n = min(len(sequence), len(res_out["betweenness_centrality_mean"]))
        btw_mean = res_out["betweenness_centrality_mean"][:n]
        btw_std  = res_out["betweenness_centrality_std"][:n]
        btw_cv   = res_out["betweenness_centrality_cv"][:n]

        AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
        for aa in AA_VOCAB:
            positions = [i for i, r in enumerate(sequence[:n]) if r == aa]
            if positions:
                aa_means = btw_mean[positions]
                aa_cvs   = btw_cv[positions]
                # only keep residues that genuinely participate (mean > 0)
                valid    = aa_means > 1e-10
                if valid.any():
                    res_out[f"btw_cv_{aa}"] = aa_cvs[valid]
                    res_out[f"btw_mean_{aa}"] = aa_means[valid]
                    res_out[f"n_{aa}"] = np.array([valid.sum()])
                else:
                    res_out[f"btw_cv_{aa}"]   = np.array([np.nan])
                    res_out[f"btw_mean_{aa}"]  = np.array([np.nan])
                    res_out[f"n_{aa}"]         = np.array([0])
            else:
                # AA not present in this sequence
                res_out[f"btw_cv_{aa}"]   = np.array([np.nan])
                res_out[f"btw_mean_{aa}"]  = np.array([np.nan])
                res_out[f"n_{aa}"]         = np.array([0])

        # save sequence for reference
        res_out["sequence"] = np.array(list(sequence[:n]))

    return seq_out, res_out
