"""Transparent analysis recipes — the statistics the course teaches, unrolled.

These mirror the Lipid Brain Atlas / EUCLID logic but as short, readable functions:
per-lipid 0-1 normalization, Moran's I (spatial autocorrelation), the Wilcoxon +
Benjamini-Hochberg differential test (the course's case-control method), marker
lipids per cluster, and the composite scores (membrane remodeling, myelination).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests


def min01_per_lipid(X, lo=0.005, hi=0.995):
    """Per-lipid: clip to [lo, hi] quantiles, rescale to [0, 1].

    MALDI intensities are only comparable *within* a lipid (ionization efficiency
    differs per molecule), so each lipid is scaled to its own 0-1 range.
    """
    X = np.asarray(X, dtype=float)
    ql = np.quantile(X, lo, axis=0)
    qh = np.quantile(X, hi, axis=0)
    rng = np.where(qh > ql, qh - ql, 1.0)
    return np.clip((X - ql) / rng, 0, 1)


def morans_i(values, coords, k=6):
    """Moran's I spatial autocorrelation of one feature over pixels.

    Builds a symmetric k-nearest-neighbour graph on the pixel coordinates, then
    I = (N/W) * sum_ij w_ij (x_i-xbar)(x_j-xbar) / sum_i (x_i-xbar)^2.
    High I = spatially structured (real anatomy); near 0 = salt-and-pepper noise.
    """
    x = np.asarray(values, float)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    if denom == 0:
        return 0.0
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # first neighbour is self
    idx = idx[:, 1:]
    num = (xc[:, None] * xc[idx]).sum()  # each edge counted once per direction
    W = idx.size
    n = len(x)
    return (n / W) * (num / denom)


def differential_lipids(adata, group_col, group1, group2, lipids=None,
                        min_fc=0.2, pthr=0.05, eps=1e-11):
    """Wilcoxon rank-sum (Mann-Whitney U) per lipid + Benjamini-Hochberg FDR.

    Returns a tidy table with log2 fold change (group2 / group1), raw p, BH-corrected
    q, and a significance flag (|log2FC| > min_fc and q < pthr). This is exactly the
    course's case-control method (and EUCLID's `differential_lipids`).
    """
    obs = adata.obs
    var = list(adata.var_names)
    cols = lipids if lipids is not None else var
    jdx = [var.index(c) for c in cols]
    X = np.asarray(adata.X)[:, jdx]
    a = X[(obs[group_col] == group1).to_numpy()]
    b = X[(obs[group_col] == group2).to_numpy()]
    rows = []
    for j, name in enumerate(cols):
        ma, mb = a[:, j].mean() + eps, b[:, j].mean() + eps
        log2fc = np.log2(mb / ma)
        try:
            _, p = mannwhitneyu(a[:, j], b[:, j], alternative="two-sided")
        except ValueError:
            p = 1.0
        rows.append((name, ma, mb, log2fc, p))
    df = pd.DataFrame(rows, columns=["lipid", f"mean_{group1}", f"mean_{group2}", "log2fc", "pval"])
    df["qval"] = multipletests(df["pval"].values, alpha=pthr, method="fdr_bh")[1]
    df["sig"] = (df["qval"] < pthr) & (df["log2fc"].abs() > min_fc)
    return df.sort_values("log2fc").reset_index(drop=True)


def marker_lipids(adata, group_key, top_n=5):
    """Per-cluster marker lipids: one-vs-rest Wilcoxon, ranked by log2 fold change.

    Used to interpret clusters in the control brain through the lipids that define them.
    """
    obs = adata.obs
    X = np.asarray(adata.X)
    names = np.array(adata.var_names)
    out = {}
    groups = pd.unique(obs[group_key].dropna())
    for g in groups:
        m = (obs[group_key] == g).to_numpy()
        if m.sum() < 5 or (~m).sum() < 5:
            continue
        inside, outside = X[m], X[~m]
        lfc = np.log2((inside.mean(0) + 1e-11) / (outside.mean(0) + 1e-11))
        order = np.argsort(lfc)[::-1][:top_n]
        out[g] = list(zip(names[order], np.round(lfc[order], 3)))
    return out


SPHINGO_PREFIXES = ("HexCer", "Cer", "SM")


def myelination_score(adata):
    """Mean z-scored sphingolipid (HexCer/Cer/SM) intensity per pixel — a myelination proxy."""
    names = list(adata.var_names)
    cols = [i for i, n in enumerate(names) if n.startswith(SPHINGO_PREFIXES)]
    X = np.asarray(adata.X)[:, cols]
    z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return z.mean(1)


def membrane_remodeling_score(diff_by_group):
    """Membrane remodeling per cluster = sum of lipid log2FCs.

    `diff_by_group`: dict {cluster -> differential_lipids table}. Negative sum = net
    turnover, positive = net biosynthesis (LBA convention).
    """
    return {g: float(d["log2fc"].sum()) for g, d in diff_by_group.items()}
