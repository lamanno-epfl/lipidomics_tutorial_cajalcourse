"""Ready-made, beautiful plotting functions for the course.

Recycled/adapted from the Lipid Brain Atlas so students get publication quality for
free and spend their effort on interpretation. The spatial-scatter style and the
cosine optimal-leaf-ordering heatmap sorting are copied faithfully from EUCLID's
`plotting.py` (the sorting is an art; we do not reinvent it).

Conventions for our substrate (`data/sections_pair.h5ad`):
  adata.X         pixels x lipids (uMAIA-normalized), var_names = lipid names
  adata.obs       SectionID, Condition, zccf, yccf, lipizone_names, lipizone_color,
                  acronym, allencolor, division, boundary, ...
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

from .style import FS, lightweight_colorbar, spatial_axes


def distinct_colors(n):
    """n visually distinct hex colours via golden-angle hue spacing with varied lightness."""
    import colorsys
    out = []
    for i in range(n):
        h = (i * 0.6180339887) % 1.0
        light = 0.45 + 0.18 * (i % 3) / 2.0
        sat = 0.65 + 0.25 * ((i // 3) % 2)
        out.append("#%02x%02x%02x" % tuple(int(255 * c) for c in colorsys.hls_to_rgb(h, light, sat)))
    return out


def lipizone_colors(adata, key="lipizone", rep="X_nmf"):
    """Assign each cluster a colour so molecularly similar clusters get adjacent colours.

    Orders clusters by hierarchical leaf-ordering on their centroid cosine distance (the
    Lipid Brain Atlas idea), then maps that order to a distinct palette, so the spatial map
    reads as a smooth, coherent anatomy rather than random confetti. Returns {label: hex}.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity
    labels = adata.obs[key].astype(str)
    cats = sorted(labels.unique())
    Z = adata.obsm[rep] if rep in adata.obsm else np.asarray(adata.X)
    cent = np.vstack([Z[(labels == c).to_numpy()].mean(0) for c in cats])
    if len(cats) > 2:
        d = squareform(1 - cosine_similarity(cent), checks=False)
        order = leaves_list(linkage(d, method="average", optimal_ordering=True))
    else:
        order = np.arange(len(cats))
    pal = distinct_colors(len(cats))
    return {cats[order[i]]: pal[i] for i in range(len(cats))}


def _lipid_vector(adata, lipid: str) -> np.ndarray:
    j = list(adata.var_names).index(lipid)
    x = adata.X[:, j]
    return np.asarray(x).ravel()


def spectrum(mz, intensity, ax=None, title="mean MALDI spectrum", lw=0.7, color="k"):
    """A mass spectrum is two aligned arrays: m/z and intensity. Draw it as sticks."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.6))
    ax.vlines(np.asarray(mz), 0, np.asarray(intensity), lw=lw, color=color)
    ax.set_xlabel("m/z")
    ax.set_ylabel("intensity")
    ax.set_title(title, fontsize=FS["m"])
    ax.set_ylim(bottom=0)
    return ax


def spatial_lipid(adata, lipid, section_key="SectionID", title_key="Condition",
                  point_size=4.0, background=True, show_contours=True, axes=None):
    """Per-section scatter of one lipid on CCF coordinates (zccf, -yccf), plasma cmap.

    vmin/vmax = median across sections of the per-section 2nd/98th percentiles, so the
    same colour means the same intensity across panels. A faint grayscale lipizone map
    sits underneath for anatomical context. Faithful to EUCLID `plot_lipid_distribution`.
    """
    obs = adata.obs
    vals = _lipid_vector(adata, lipid)
    secs = sorted(obs[section_key].unique())
    # per-section 2/98 percentiles -> medians (robust shared colour scale)
    p2 = [np.quantile(vals[(obs[section_key] == s).values], 0.02) for s in secs]
    p98 = [np.quantile(vals[(obs[section_key] == s).values], 0.98) for s in secs]
    vmin, vmax = float(np.median(p2)), float(np.median(p98))

    if axes is None:
        fig, axes = plt.subplots(1, len(secs), figsize=(4.2 * len(secs), 3.6), squeeze=False)
        axes = axes.ravel()
    z_all, y_all = obs["zccf"].to_numpy(), -obs["yccf"].to_numpy()
    sc = None
    for ax, s in zip(axes, secs):
        m = (obs[section_key] == s).to_numpy()
        if background and "lipizone_names" in obs:
            uniq = pd.unique(obs.loc[m, "lipizone_names"].dropna())
            if len(uniq):
                g = dict(zip(uniq, np.linspace(0.2, 0.8, len(uniq))))
                ax.scatter(obs.loc[m, "zccf"], -obs.loc[m, "yccf"],
                           c=obs.loc[m, "lipizone_names"].map(g), cmap="gray",
                           s=point_size * 0.5, alpha=0.3, rasterized=True)
        sc = ax.scatter(obs.loc[m, "zccf"], -obs.loc[m, "yccf"], c=vals[m], cmap="plasma",
                        s=point_size, alpha=0.85, rasterized=True, vmin=vmin, vmax=vmax)
        if show_contours and "boundary" in obs:
            b = m & (obs["boundary"].astype(float) == 1).to_numpy()
            ax.scatter(obs.loc[b, "zccf"], -obs.loc[b, "yccf"], c="k",
                       s=point_size / 2, alpha=0.5, rasterized=True)
        spatial_axes(ax)
        ax.set_xlim(z_all.min(), z_all.max())
        ax.set_ylim(y_all.min(), y_all.max())
        lab = obs.loc[m, title_key].iloc[0] if title_key in obs else s
        ax.set_title(f"{lab}", fontsize=FS["m"])
    if sc is not None:
        lightweight_colorbar(sc, list(axes), label=lipid)
    return axes


def spatial_categorical(adata, color_key="lipizone_color", section_key="SectionID",
                        title_key="Condition", point_size=4.0, axes=None):
    """Scatter pixels coloured by a stored hex colour (lipizone_color or allencolor)."""
    obs = adata.obs
    secs = sorted(obs[section_key].unique())
    if axes is None:
        fig, axes = plt.subplots(1, len(secs), figsize=(4.2 * len(secs), 3.6), squeeze=False)
        axes = axes.ravel()
    z_all, y_all = obs["zccf"].to_numpy(), -obs["yccf"].to_numpy()
    for ax, s in zip(axes, secs):
        m = (obs[section_key] == s).to_numpy()
        ax.scatter(obs.loc[m, "zccf"], -obs.loc[m, "yccf"], c=obs.loc[m, color_key],
                   s=point_size, alpha=0.9, rasterized=True)
        spatial_axes(ax)
        ax.set_xlim(z_all.min(), z_all.max())
        ax.set_ylim(y_all.min(), y_all.max())
        lab = obs.loc[m, title_key].iloc[0] if title_key in obs else s
        ax.set_title(f"{lab}", fontsize=FS["m"])
    return axes


def sorted_lipid_heatmap(adata, group_key, mask=None, cmap="Reds", figsize=(18, 10),
                         max_labels=60, title=None, ax=None):
    """Group-mean lipid heatmap with rows AND columns cosine optimal-leaf-ordered.

    Copied faithfully from EUCLID `plot_olosorted_lipid_lipizone`: per-lipid 0-1
    normalize, average by `group_key` (e.g. 'acronym' for anatomy x lipid, or
    'lipizone_names' for lipizone x lipid), then order both axes by hierarchical
    clustering on cosine distance with optimal leaf ordering. The sorting is the art.
    """
    obs = adata.obs if mask is None else adata.obs[mask]
    X = adata.X if mask is None else adata.X[np.asarray(mask)]
    X = np.asarray(X)
    # per-lipid 0-1 normalization (column-wise)
    cmin = X.min(0)
    rng = X.max(0) - cmin
    rng[rng == 0] = 1.0
    norm = (X - cmin) / rng
    df = pd.DataFrame(norm, columns=list(adata.var_names))
    df[group_key] = obs[group_key].to_numpy()
    df = df.dropna(subset=[group_key])
    avg = df.groupby(group_key, observed=True).mean()

    # cosine distance + optimal leaf ordering on both axes (verbatim recipe)
    row_d = squareform(1 - cosine_similarity(avg.values), checks=False)
    col_d = squareform(1 - cosine_similarity(avg.values.T), checks=False)
    row_order = leaves_list(linkage(row_d, method="average", optimal_ordering=True))
    col_order = leaves_list(linkage(col_d, method="average", optimal_ordering=True))
    sorted_df = avg.iloc[row_order, col_order]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sorted_df.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    # show a readable subset of labels when there are many rows
    rows = sorted_df.index.to_list()
    if len(rows) > max_labels:
        step = max(1, len(rows) // max_labels)
        ax.set_yticks(range(0, len(rows), step))
        ax.set_yticklabels(rows[::step], fontsize=FS["xs"])
    else:
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=FS["xs"])
    cols = sorted_df.columns.to_list()
    if len(cols) > max_labels:
        step = max(1, len(cols) // max_labels)
        ax.set_xticks(range(0, len(cols), step))
        ax.set_xticklabels(cols[::step], rotation=90, fontsize=FS["xs"])
    else:
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=FS["xs"])
    ax.set_xlabel("lipids")
    ax.set_ylabel(group_key)
    if title:
        ax.set_title(title, fontsize=FS["m"])
    lightweight_colorbar(im, ax, label="0-1 intensity")
    return ax, sorted_df


def volcano(diff_df, fc_col="log2fc", q_col="qval", fc_thresh=0.2, q_thresh=0.05,
            label_col=None, top_n=15, ax=None, title=None):
    """Volcano plot for a differential-lipid table (log2FC vs -log10 corrected p)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    d = diff_df.copy()
    y = -np.log10(d[q_col].clip(lower=1e-300))
    sig = (d[q_col] < q_thresh) & (d[fc_col].abs() > fc_thresh)
    ax.scatter(d.loc[~sig, fc_col], y[~sig], s=10, c="0.7", rasterized=True)
    ax.scatter(d.loc[sig, fc_col], y[sig], s=14, c="crimson", rasterized=True)
    ax.axhline(-np.log10(q_thresh), ls="--", lw=0.6, c="0.4")
    for x in (-fc_thresh, fc_thresh):
        ax.axvline(x, ls="--", lw=0.6, c="0.4")
    ax.set_xlabel("log2 fold change (pregnant / control)")
    ax.set_ylabel("-log10 adjusted p")
    if title:
        ax.set_title(title, fontsize=FS["m"])
    if label_col is not None:
        top = d[sig].reindex(d[sig][fc_col].abs().sort_values(ascending=False).index).head(top_n)
        for _, r in top.iterrows():
            ax.annotate(str(r[label_col]), (r[fc_col], -np.log10(max(r[q_col], 1e-300))),
                        fontsize=FS["xs"], ha="center")
    return ax
