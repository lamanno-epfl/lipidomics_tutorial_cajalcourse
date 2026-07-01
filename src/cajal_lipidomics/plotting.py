"""Ready-made, beautiful plotting functions for the course.

Adapted from the Lipid Brain Atlas / EUCLID so students get publication quality for free and
spend their effort on interpretation. What is taken VERBATIM from his code: the reciprocal-enrichment
recipe (001-IDCARDS), the optimal-leaf-ordering heatmap sorting, the per-section 2/98 shared colour
scale (plot_lipid_distribution), the 8 lipizone colormaps (assign_cluster_colors). What is ADAPTED:
the 8-colormap assignment to flat Leiden clusters; the group x lipid heatmap normalised to relative
abundance (his regional-enrichment style, as he requested, not the 0-1 stretch of plot_olosorted).

Works on the student-built data/derived chain: adata.X = pixels x ions, var['lipid'] = names,
adata.obs has SectionID, Condition, x, y, xccf/yccf/zccf, acronym, allencolor, lipizone, lipizone_color.
"""
from __future__ import annotations

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


LIPIZONE_COLORMAPS = ["RdYlBu", "terrain", "PiYG", "cividis", "plasma", "PuRd", "inferno", "PuOr"]


def lipizone_colors(adata, key="lipizone", rep="X_nmf", n_families=8):
    """Colour lipizones with the Lipid Brain Atlas 8-colormap scheme (EUCLID `assign_cluster_colors`).

    Lipizones are grouped into up to 8 families by the similarity of their molecular centroids; each
    family is given one of 8 colormaps (RdYlBu, terrain, PiYG, cividis, plasma, PuRd, inferno, PuOr),
    and members within a family are shaded along that colormap in similarity order. So sibling
    lipizones share a colour family and the section reads as coherent anatomy. Returns {str(label): hex}.
    """
    import matplotlib as mpl
    from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
    from scipy.spatial.distance import pdist
    labels = adata.obs[key].astype(str).to_numpy()
    cats = sorted(pd.unique(labels))
    if len(cats) <= 1:
        return {c: "#777777" for c in cats}
    Z = adata.obsm[rep] if rep in adata.obsm else np.asarray(adata.X)
    cent = np.vstack([Z[labels == c].mean(0) for c in cats])
    link = linkage(pdist(cent), method="average", optimal_ordering=True)
    nfam = int(min(n_families, len(cats)))
    fam = fcluster(link, t=nfam, criterion="maxclust")          # family id per lipizone
    order = list(leaves_list(link))                             # global similarity order
    out = {}
    for fi, f in enumerate(sorted(np.unique(fam))):
        members = [cats[i] for i in order if fam[i] == f]        # this family, in similarity order
        cmap = mpl.colormaps[LIPIZONE_COLORMAPS[fi % len(LIPIZONE_COLORMAPS)]]  # mpl>=3.9: cm.get_cmap removed
        for j, m in enumerate(members):
            frac = 0.15 + 0.70 * (j / max(len(members) - 1, 1))  # stay off the colormap's extremes
            r, g, b, _ = cmap(frac)
            out[m] = "#%02x%02x%02x" % (int(255 * r), int(255 * g), int(255 * b))
    return out


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


def sorted_lipid_heatmap(adata, group_key, mask=None, cmap="Reds", figsize=(20, 9),
                         name_key="lipid", title=None, ax=None):
    """Group x lipid heatmap of RELATIVE abundance, rows and columns cosine optimal-leaf-ordered.

    The value shown is each lipid's group-mean divided by its grand mean, so 1 = the lipid's
    average level, 2 = twice it, 0.5 = half. This is the meaningful, eyeballable scaling the Lipid
    Brain Atlas uses (a plain 0-1 stretch wastes the colour range on noise). Columns are LABELLED
    with lipid names (from `adata.var[name_key]`, falling back to var_names). Average by `group_key`
    (e.g. 'acronym' for anatomy x lipid, 'lipizone' for lipizone x lipid); both axes ordered by
    optimal leaf ordering on cosine distance. Colour limits are the 2nd/98th percentiles.
    """
    obs = adata.obs if mask is None else adata.obs[np.asarray(mask)]
    X = np.asarray(adata.X if mask is None else adata.X[np.asarray(mask)])
    names = list(adata.var[name_key]) if (name_key in adata.var) else list(adata.var_names)
    df = pd.DataFrame(X, columns=names)
    df[group_key] = obs[group_key].to_numpy()
    df = df.dropna(subset=[group_key])
    avg = df.groupby(group_key, observed=True).mean()
    rel = avg / (avg.mean(0) + 1e-12)                           # relative abundance per lipid

    row_d = squareform(np.clip(1 - cosine_similarity(rel.values), 0, None), checks=False)
    col_d = squareform(np.clip(1 - cosine_similarity(rel.values.T), 0, None), checks=False)
    row_order = leaves_list(linkage(row_d, method="average", optimal_ordering=True))
    col_order = leaves_list(linkage(col_d, method="average", optimal_ordering=True))
    sorted_df = rel.iloc[row_order, col_order]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = np.percentile(sorted_df.values, [2, 98])
    im = ax.imshow(sorted_df.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    rows, cols = sorted_df.index.to_list(), sorted_df.columns.to_list()
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=FS["xs"])
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=max(4, FS["xs"] - 2))
    ax.set_xlabel("lipid"); ax.set_ylabel(group_key)
    if title:
        ax.set_title(title, fontsize=FS["m"])
    lightweight_colorbar(im, ax, label="relative abundance (group mean / grand mean)")
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


# ---- v3 teaching plots (faithful to the Lipid Brain Atlas building blocks) ----

def allen_contours(obs, ax, region_key="acronym", x="x", y="y", color="k", lw=0.4, s=0.6):
    """Draw Allen region contours: a pixel sits on a contour if a 4-neighbour has a different
    region. Rasterise the region ids on the (x,y) grid, mark the edges, scatter them."""
    xs = obs[x].to_numpy().astype(int); ys = obs[y].to_numpy().astype(int)
    codes = pd.Series(obs[region_key].astype(str)).astype("category").cat.codes.to_numpy()
    H, W = ys.max() + 2, xs.max() + 2
    grid = np.full((H, W), -1); grid[ys, xs] = codes
    edge = np.zeros_like(grid, bool)
    for dy, dx in ((0, 1), (1, 0)):
        nb = np.full_like(grid, -1)
        nb[:-dy or None, :-dx or None] = grid[dy:, dx:]
        edge |= (grid >= 0) & (nb >= 0) & (grid != nb)
    ey, ex = np.nonzero(edge)
    ax.scatter(ex, -ey, c=color, s=s, lw=lw, rasterized=True)


def rgb_overlay(adata, lipids, section_key="SectionID", layer=None, x="x", y="y", axes=None):
    """Overlay up to three lipids as the R, G, B channels of one image per section (each channel
    per-section 2/98 clipped). Shows where lipids co-localise (white) or separate (pure colours)."""
    X = adata.layers[layer] if layer else np.asarray(adata.X)
    cols = [list(adata.var_names).index(l) for l in lipids[:3]]
    secs = sorted(adata.obs[section_key].unique())
    if axes is None:
        fig, axes = plt.subplots(1, len(secs), figsize=(4.2 * len(secs), 3.6), squeeze=False); axes = axes.ravel()
    for ax, sname in zip(axes, secs):
        m = (adata.obs[section_key] == sname).to_numpy()
        xs = adata.obs.loc[m, x].to_numpy().astype(int); ys = adata.obs.loc[m, y].to_numpy().astype(int)
        H, W = ys.max() + 1, xs.max() + 1
        img = np.zeros((H, W, 3))
        for ch, c in enumerate(cols):
            v = X[m, c]; lo, hi = np.percentile(v, [2, 98]); img[ys, xs, ch] = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
        ax.imshow(img); spatial_axes(ax)
        ax.set_title(adata.obs.loc[m, "Condition"].iloc[0] if "Condition" in adata.obs else str(sname), fontsize=FS["m"])
    handles = " / ".join(f"{c}={l}" for c, l in zip("RGB", lipids[:3]))
    axes[0].set_ylabel(handles, fontsize=FS["xs"])
    return axes


def lipid_lipid_corr(adata, layer=None, ax=None, cmap="RdBu_r", name_key="lipid", figsize=(16, 15)):
    """Lipid-lipid correlation heatmap, rows/cols cosine optimal-leaf-ordered so co-varying lipids
    sit together, LABELLED with lipid names on both axes (a tiny unlabelled heatmap is useless).
    Reveals the molecular modules NMF will later summarise."""
    X = adata.layers[layer] if layer else np.asarray(adata.X)
    names = list(adata.var[name_key]) if (name_key in adata.var) else list(adata.var_names)
    C = np.nan_to_num(np.corrcoef(X.T))
    d = squareform(1 - np.abs(C), checks=False)
    order = leaves_list(linkage(d, method="average", optimal_ordering=True))
    Cs = C[np.ix_(order, order)]
    labels = [names[i] for i in order]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(Cs, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=max(4, FS["xs"] - 2))
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=max(4, FS["xs"] - 2))
    ax.set_title("lipid-lipid correlation (clustered, labelled)", fontsize=FS["m"])
    lightweight_colorbar(im, ax, label="Pearson r")
    return ax, labels


def cluster_size_hist(labels, ax=None):
    """Histogram of how many pixels each cluster holds (a quick clustering sanity check)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3))
    sizes = pd.Series(np.asarray(labels)).value_counts().values
    ax.hist(sizes, bins=20, color="0.4")
    ax.set_xlabel("pixels per cluster"); ax.set_ylabel("number of clusters")
    ax.set_title("cluster-size distribution", fontsize=FS["m"]); return ax


def tsne_colored(adata, by, kind="lipid", tsne_key="X_tsne", cmap="plasma", point_size=2.0, ax=None, title=None):
    """t-SNE scatter coloured by a lipid (continuous, 2/98 clipped), an obs column of hex colours
    (kind='color'), or a continuous obs column (kind='value'). Faithful to EUCLID plot_tsne."""
    xy = adata.obsm[tsne_key]; order = np.random.RandomState(0).permutation(adata.n_obs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    if kind == "lipid":
        v = _lipid_vector(adata, by); lo, hi = np.percentile(v, [2, 98])
        sc = ax.scatter(xy[order, 0], xy[order, 1], c=v[order], cmap=cmap, s=point_size, vmin=lo, vmax=hi, rasterized=True)
        lightweight_colorbar(sc, ax, label=by)
    elif kind == "color":
        ax.scatter(xy[order, 0], xy[order, 1], c=adata.obs[by].to_numpy()[order], s=point_size, rasterized=True)
    else:
        v = adata.obs[by].to_numpy().astype(float)[order]; lo, hi = np.nanpercentile(v, [2, 98])
        sc = ax.scatter(xy[order, 0], xy[order, 1], c=v, cmap="RdBu_r", s=point_size, vmin=lo, vmax=hi, rasterized=True)
        lightweight_colorbar(sc, ax, label=by)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title(title or f"t-SNE coloured by {by}", fontsize=FS["m"]); return ax


def lipizones_on_gray(adata, color_key="lipizone_color", section_key="SectionID", x="x", y="y",
                      point_size=3.0, axes=None):
    """Lipizones in colour on top of a faint grey rendering of all tissue (the LBA background style),
    so a few highlighted territories read against the whole section."""
    obs = adata.obs; secs = sorted(obs[section_key].unique())
    if axes is None:
        fig, axes = plt.subplots(1, len(secs), figsize=(4.2 * len(secs), 3.6), squeeze=False); axes = axes.ravel()
    for ax, sname in zip(axes, secs):
        m = (obs[section_key] == sname).to_numpy()
        ax.scatter(obs.loc[m, x], -obs.loc[m, y], c="0.85", s=point_size, rasterized=True)
        ax.scatter(obs.loc[m, x], -obs.loc[m, y], c=obs.loc[m, color_key], s=point_size, alpha=0.9, rasterized=True)
        spatial_axes(ax)
        ax.set_title(obs.loc[m, "Condition"].iloc[0] if "Condition" in obs else str(sname), fontsize=FS["m"])
    return axes


def marker_barplot(adata, mask, layer=None, lipid_props=None, top=20, ax=None, title="enriched lipids"):
    """Sorted barplot of the lipids most enriched in a pixel subset (e.g. white matter) vs the rest,
    coloured by lipid class. Use to see, for example, HexCer/SM enrichment in white matter."""
    X = adata.layers[layer] if layer else np.asarray(adata.X)
    m = np.asarray(mask); lfc = np.log2((X[m].mean(0) + 1e-9) / (X[~m].mean(0) + 1e-9))
    s = pd.Series(lfc, index=list(adata.var_names)).sort_values(ascending=False).head(top)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, max(3, top * 0.22)))
    colors = None
    if lipid_props is not None:
        colors = [lipid_props["color"].get(n, "#999999") if hasattr(lipid_props["color"], "get") else "#777" for n in s.index]
    ax.barh(range(len(s))[::-1], s.values, color=colors or "#4477aa", height=0.7)
    ax.set_yticks(range(len(s))[::-1]); ax.set_yticklabels(s.index, fontsize=FS["xs"])
    ax.set_xlabel("log2 fold change vs rest"); ax.set_title(title, fontsize=FS["m"]); return ax


def mosaic(adata, lipizone_color_key="lipizone_color", section_key="SectionID", section=None,
           xlim=None, ylim=None, x="x", y="y", point_size=12, ax=None, title=None):
    """Zoom into a region of interest and read the lipizone mosaic there (EUCLID plot_mosaic idea).

    Crops one section to a bounding box (xlim, ylim in pixel coords) and paints each pixel by its
    lipizone colour, with large points, so students can inspect how lipizones tile a chosen region.
    If no box is given it shows the whole section. Black background, like the atlas mosaic.
    """
    obs = adata.obs
    sid = section if section is not None else sorted(obs[section_key].unique())[0]
    m = (obs[section_key] == sid).to_numpy()
    sub = obs[m]
    if xlim is not None:
        sub = sub[(sub[x] >= xlim[0]) & (sub[x] <= xlim[1])]
    if ylim is not None:
        sub = sub[(sub[y] >= ylim[0]) & (sub[y] <= ylim[1])]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7)); fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.scatter(sub[x], -sub[y], c=sub[lipizone_color_key], s=point_size, marker="s", rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or f"lipizone mosaic (section {sid})", color="white", fontsize=FS["m"])
    return ax


def highlight_lipizone(adata, lipizone, key="lipizone", section_key="SectionID",
                       x="x", y="y", point_size=2, axes=None):
    """Highlight ONE lipizone in red over a faint grayscale cast of ALL lipizones (each one its own
    grey shade, the Lipid Brain Atlas style), per section. The way to reason about a single object in
    spatial omics: where, exactly, does this one territory sit?"""
    obs = adata.obs
    secs = sorted(obs[section_key].unique())
    if axes is None:
        fig, axes = plt.subplots(1, len(secs), figsize=(4.2 * len(secs), 3.6), squeeze=False); axes = axes.ravel()
    target = obs[key].astype(str) == str(lipizone)
    for ax, sname in zip(axes, secs):
        m = (obs[section_key] == sname).to_numpy()
        # faint grayscale rendering of every lipizone underneath: each lipizone gets its own grey shade
        uniq = pd.unique(obs.loc[m, key].astype(str))
        g = dict(zip(uniq, np.linspace(0.2, 0.8, len(uniq))))
        ax.scatter(obs.loc[m, x], -obs.loc[m, y], c=obs.loc[m, key].astype(str).map(g),
                   cmap="gray", s=point_size, alpha=0.3, rasterized=True)
        hit = m & target.to_numpy()
        ax.scatter(obs.loc[hit, x], -obs.loc[hit, y], c="red", s=point_size, rasterized=True)
        spatial_axes(ax)
        ax.set_title(obs.loc[m, "Condition"].iloc[0] if "Condition" in obs else str(sname), fontsize=FS["m"])
    axes[0].set_ylabel(f"lipizone {lipizone}", fontsize=FS["s"])
    return axes


def lipizone_hierarchy_levels(adata, key="lipizone", rep="X_nmf", max_level=4):
    """Build a binary hierarchy over lipizones from their centroids and return, for each pixel, its
    ancestor group at levels 1..max_level (a divisive tree cut at 2, 4, 8, ... groups). Lets you show
    the split progressively instead of jumping to the terminal leaves."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    labels = adata.obs[key].astype(str).to_numpy()
    cats = sorted(pd.unique(labels))
    Z = adata.obsm[rep] if rep in adata.obsm else np.asarray(adata.X)
    cent = np.vstack([Z[labels == c].mean(0) for c in cats])
    link = linkage(pdist(cent), method="average", optimal_ordering=True)
    out = {}
    for L in range(1, max_level + 1):
        grp = fcluster(link, t=min(2 ** L, len(cats)), criterion="maxclust")  # lipizone -> group id
        lut = dict(zip(cats, grp))
        out[f"level_{L}"] = pd.Series(labels, index=adata.obs_names).map(lut).astype(int)
    return pd.DataFrame(out, index=adata.obs_names)


def plot_hierarchy_levels(adata, levels_df, color_key="lipizone_color", section_key="SectionID",
                          x="x", y="y", section=None, point_size=4):
    """Show the lipizone hierarchy split by split: at each level a pixel takes the MODAL lipizone
    colour of its group, so you see 2 colours, then 4, then 8 ... (EUCLID's progressive-splitting view)."""
    obs = adata.obs
    sid = section if section is not None else sorted(obs[section_key].unique())[0]
    m = (obs[section_key] == sid).to_numpy()
    cols = list(levels_df.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(3.8 * len(cols), 3.8), squeeze=False); axes = axes.ravel()
    base = obs[color_key].astype(str)
    for ax, lv in zip(axes, cols):
        modal = base.groupby(levels_df[lv]).agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "#999999")
        pix_color = levels_df[lv].map(modal)
        ax.scatter(obs.loc[m, x], -obs.loc[m, y], c=pix_color[m].to_numpy(), s=point_size, rasterized=True)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{lv}: {levels_df[lv][m].nunique()} groups", fontsize=FS["s"])
    return axes
