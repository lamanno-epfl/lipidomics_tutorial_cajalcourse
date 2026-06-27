"""Embedding + clustering recipes, transparent versions of the EUCLID pipeline.

Learned on the control brain, applied to both: seeded NMF (parts-based lipid programs),
Harmony batch integration (for clustering/transfer only), Leiden clustering on the
neighbour graph, and kNN label transfer control -> pregnant. Mirrors EUCLID
`learn_seeded_nmf_embeddings` / `harmonize_nmf_batches` but short and readable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


def nmf_embeddings(X, n_factors=15, random_state=42, max_iter=400):
    """Factor X ~ W H with all entries non-negative (NMF), giving `n_factors` parts-based
    lipid programs. `init='nndsvda'` is deterministic and standard. W (pixels x factors)
    is the per-pixel program activity; H (factors x lipids) is each program's lipid loadings.

    Returns W, H, the fitted model.
    """
    X = np.asarray(X, float)
    Xpos = X - X.min() + 1e-7  # NMF requires non-negative input
    model = NMF(n_components=n_factors, init="nndsvda", random_state=random_state, max_iter=max_iter)
    W = model.fit_transform(Xpos)
    return W, model.components_, model


def seeded_nmf(X, n_factors=15, random_state=42, max_iter=400):
    """NMF with a correlation-informed initialization (the LBA idea, made robust): group
    the lipids into `n_factors` clusters by their correlation, take the most central lipid
    of each cluster as a seed program, and initialize W/H from those. Same output as
    `nmf_embeddings` but with an interpretable, seeded start.
    """
    from sklearn.cluster import AgglomerativeClustering
    X = np.asarray(X, float)
    Xpos = X - X.min() + 1e-7
    corr = np.nan_to_num(np.corrcoef(Xpos.T))
    D = 1 - np.abs(corr)
    lab = AgglomerativeClustering(n_clusters=n_factors, metric="precomputed",
                                  linkage="average").fit_predict(D)
    reps = []
    for c in range(n_factors):
        members = np.where(lab == c)[0]
        if len(members):
            reps.append(members[np.argmin(D[np.ix_(members, members)].mean(1))])
    reps = np.array(reps)
    W_init = Xpos[:, reps].astype(np.float64)
    H_init = np.clip(corr[reps, :], 0, None).astype(np.float64)
    model = NMF(n_components=len(reps), init="custom", random_state=random_state, max_iter=max_iter)
    W = model.fit_transform(Xpos, W=W_init.copy(), H=H_init.copy())
    return W, model.components_, model


def apply_nmf(model, X):
    """Project new pixels into the learned NMF factor space (W for new data)."""
    X = np.asarray(X, float)
    Xpos = X - X.min() + 1e-7
    return model.transform(Xpos)


def harmonize(W, batch_labels, random_state=0):
    """Harmony batch integration on an embedding (clustering/transfer ONLY).

    Removes section/batch offsets in the factor space so clusters reflect biology, not
    acquisition. Never used for the differential test.
    """
    import harmonypy
    meta = pd.DataFrame({"batch": np.asarray(batch_labels).astype(str)})
    Wm = np.asarray(W, float)
    ho = harmonypy.run_harmony(Wm, meta, ["batch"], random_state=random_state)
    Z = ho.Z_corr
    Z = Z.detach().cpu().numpy() if hasattr(Z, "detach") else np.asarray(Z)
    # orient to (pixels x factors) regardless of harmonypy version's convention
    n = Wm.shape[0]
    if Z.shape[0] != n and Z.shape[1] == n:
        Z = Z.T
    return Z


def leiden_clusters(emb, n_neighbors=40, n_iterations=5, resolution=None, seed=230598):
    """Fast Leiden exactly as EUCLID `_leidenalg_clustering`: a k-nearest-neighbour graph
    (k=40) on the embedding, then leidenalg with the ModularityVertexPartition. Modularity gives
    clean, contiguous lipid territories; the newer scanpy igraph flavour over-fragments into
    speckle and the full sc.tl.leiden is far too slow on ~200k pixels. Pass `resolution` to switch
    to the resolution-tunable RBConfigurationVertexPartition instead of plain modularity."""
    import leidenalg, igraph as ig
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(emb, float)
    knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(X).kneighbors_graph(X)
    src, dst = knn.nonzero()
    g = ig.Graph(n=X.shape[0], edges=list(zip(src.tolist(), dst.tolist())))
    g.simplify()
    if resolution is None:
        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                        n_iterations=n_iterations, seed=seed)
    else:
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                        resolution_parameter=resolution, n_iterations=n_iterations, seed=seed)
    return np.array(part.membership)


def knn_transfer(emb_ref, labels_ref, emb_query, k=15):
    """Copy labels from a reference onto query pixels by majority vote of k nearest
    neighbours in the shared (Harmonized) embedding. Returns labels + confidence."""
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    clf.fit(np.asarray(emb_ref, float), np.asarray(labels_ref))
    pred = clf.predict(np.asarray(emb_query, float))
    conf = clf.predict_proba(np.asarray(emb_query, float)).max(1)
    return pred, conf
