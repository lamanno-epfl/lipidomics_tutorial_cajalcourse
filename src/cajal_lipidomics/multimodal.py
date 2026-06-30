"""Multimodal interpretation: which gene programs predict the pregnancy lipid changes.

Region-level integration (the clean path): both modalities share the Allen atlas. We
build, per Allen region, the pregnancy lipid change (control vs pregnant) and join the
region's average MERFISH-imputed gene expression, then ask XGBoost which gene programs
predict each lipid's change, read out the drivers with SHAP, and name the programs with
gene ontology. Mirrors the LBA genes->lipids analysis (digests 07, 08).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import xgboost as xgb


def region_change_matrix(adata, region_key="acronym", cond_key="Condition",
                         control="naive", case="pregnant", min_pixels=50, eps=1e-11):
    """Per-region log2 fold change (case vs control) for each lipid.

    Returns a DataFrame indexed by region (acronym), columns = lipids, restricted to
    regions with >= min_pixels in BOTH conditions.
    """
    obs = adata.obs
    X = np.asarray(adata.X)
    lipids = list(adata.var_names)
    rows = {}
    for reg, idx in obs.groupby(region_key, observed=True).indices.items():
        sub = obs.iloc[idx]
        m_c = (sub[cond_key] == control).to_numpy()
        m_k = (sub[cond_key] == case).to_numpy()
        if m_c.sum() < min_pixels or m_k.sum() < min_pixels:
            continue
        mean_c = X[idx][m_c].mean(0) + eps
        mean_k = X[idx][m_k].mean(0) + eps
        rows[str(reg)] = np.log2(mean_k / mean_c)
    return pd.DataFrame.from_dict(rows, orient="index", columns=lipids)


def join_genes(change_df, genes_df):
    """Align the region x lipid-change matrix with the region x gene matrix on region."""
    genes_df = genes_df.loc[:, [c for c in genes_df.columns if c != "__index_level_0__"]]
    common = change_df.index.intersection(genes_df.index)
    return change_df.loc[common], genes_df.loc[common]


def gene_programs(genes_df, n_programs=20, random_state=42):
    """NMF gene programs from the region x gene matrix. Returns W (regions x programs),
    H (programs x genes), the fitted model. MinMax to [0,1] first (NMF needs >=0)."""
    Xg = MinMaxScaler().fit_transform(genes_df.values)
    model = NMF(n_components=n_programs, init="nndsvda", random_state=random_state, max_iter=500)
    W = model.fit_transform(Xg)
    return W, model.components_, model


def predict_changes(W_programs, change_df, region_index, test_size=0.25, random_state=42):
    """One XGBoost regressor per lipid: predict its per-region change from gene programs.

    Returns a per-lipid table (test Pearson r) and the mean |SHAP| program-importance
    matrix (programs x lipids), computed with XGBoost's exact TreeSHAP (pred_contribs).
    """
    Wz = StandardScaler().fit_transform(W_programs)
    idx = np.arange(Wz.shape[0])
    tr, te = train_test_split(idx, test_size=test_size, random_state=random_state)
    n_prog = Wz.shape[1]
    rows, shap_cols = [], {}
    for lipid in change_df.columns:
        y = change_df[lipid].to_numpy()
        m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=random_state, objective="reg:squarederror")
        m.fit(Wz[tr], y[tr])
        pred = m.predict(Wz[te])
        r = pearsonr(pred, y[te])[0] if np.std(pred) > 0 and np.std(y[te]) > 0 else 0.0
        rows.append((lipid, float(r)))
        contribs = m.get_booster().predict(xgb.DMatrix(Wz), pred_contribs=True)[:, :-1]
        shap_cols[lipid] = np.abs(contribs).mean(0)  # mean |SHAP| per program
    scores = pd.DataFrame(rows, columns=["lipid", "test_r"]).sort_values("test_r", ascending=False)
    shap_mat = pd.DataFrame(shap_cols, index=[f"program{j+1}" for j in range(n_prog)])
    return scores, shap_mat


# ---- per-pixel integration via shared CCF coordinates (per-cell MERFISH) ----

def merfish_gene_columns(all_columns):
    """The 500 measured-gene columns of cell_filtered_w500genes.parquet (Ensembl transcript IDs)."""
    return [c for c in all_columns if c.startswith("ENSMUS")]


def load_merfish_cells(parquet_path, ap_range, genes=None, drop_divisions=("6 Vascular", "7 Immune")):
    """Load per-cell MERFISH within an AP (x_ccf) window, dropping vascular/immune cells.

    Returns a DataFrame with x_ccf/y_ccf/z_ccf, the cell-type columns (division/class/subclass/
    supertype), and the 500 gene columns. `ap_range` is (lo, hi) in CCF units (the MSI sections'
    xccf min/max with a small buffer).
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    names = [n for n in pq.read_schema(parquet_path).names]
    gene_cols = genes if genes is not None else merfish_gene_columns(names)
    keep = ["x_ccf", "y_ccf", "z_ccf", "division", "class", "subclass", "supertype"] + list(gene_cols)
    keep = [c for c in keep if c in names]
    flt = (pc.field("x_ccf") >= ap_range[0]) & (pc.field("x_ccf") <= ap_range[1])
    df = pq.read_table(parquet_path, columns=keep, filters=flt).to_pandas()
    df = df[~df["division"].astype(str).isin(drop_divisions)]
    return df, [c for c in gene_cols if c in df.columns]


def match_pixels_to_cells(adata_obs, cells, gene_cols, radius=0.075, celltype_key="subclass"):
    """For each MSI pixel, average the genes (and majority-vote the cell type) of the MERFISH
    cells within `radius` CCF units in 3D. Pixels with no cell in the ball are left NaN.

    Returns (gene_df indexed like adata_obs with gene columns; celltype Series). This is the
    transparent 'integration by shared coordinates' core (KD-tree ball query + aggregate).
    """
    from scipy.spatial import cKDTree
    cell_xyz = cells[["x_ccf", "y_ccf", "z_ccf"]].to_numpy()
    tree = cKDTree(cell_xyz)
    G = cells[gene_cols].to_numpy(np.float32)
    ct = cells[celltype_key].astype(str).to_numpy()
    q = adata_obs[["xccf", "yccf", "zccf"]].to_numpy()
    nbrs = tree.query_ball_point(q, r=radius)
    gene_out = np.full((len(q), len(gene_cols)), np.nan, np.float32)
    ct_out = np.array([None] * len(q), dtype=object)
    for i, nb in enumerate(nbrs):
        if nb:
            gene_out[i] = G[nb].mean(0)
            vals, counts = np.unique(ct[nb], return_counts=True)
            ct_out[i] = vals[counts.argmax()]
    gdf = pd.DataFrame(gene_out, index=adata_obs.index, columns=gene_cols)
    return gdf, pd.Series(ct_out, index=adata_obs.index, name=celltype_key)


def reciprocal_enrichment(row_labels, col_labels):
    """Reciprocal enrichment between two categorical pixel labelings, exactly as the Lipid Brain
    Atlas does it (001-IDCARDS): the element-wise product of two directional enrichments. NOT a
    plain crosstab, which would just count co-occurrence and be dominated by big categories.

    enrichment of a column-category within a row-category = (its fraction in that row) divided by
    its mean fraction across rows. We compute it both ways (cols-in-rows and rows-in-cols) and
    multiply, so a pair scores high only if EACH is enriched in the other. Returns a (rows x cols)
    DataFrame of enrichment scores. Display it raw with vmin/vmax at the 2nd/98th percentiles (the LBA
    convention), never on a log scale.
    """
    r, c = pd.Series(np.asarray(row_labels)), pd.Series(np.asarray(col_labels))
    c1 = pd.crosstab(r, c)                       # (rows, cols)
    n1 = c1 / c1.sum()                           # each column sums to 1
    n1 = (n1.T / n1.T.mean()).T                  # -> enrichment of each col within each row
    c2 = pd.crosstab(c, r)                       # (cols, rows)
    n2 = c2 / c2.sum()
    n2 = (n2.T / n2.T.mean()).T                  # -> enrichment of each row within each col
    return n1 * n2.T                             # reciprocal enrichment, (rows, cols)


def top_genes_for_program(H, genes_df, program_idx, top=50):
    """Leading genes of a gene program (largest NMF loadings)."""
    loadings = pd.Series(H[program_idx], index=genes_df.columns)
    return loadings.sort_values(ascending=False).head(top)
