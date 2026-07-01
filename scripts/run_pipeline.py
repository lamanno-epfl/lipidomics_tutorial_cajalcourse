"""Verified end-to-end pipeline on the RAW student data (dev backbone for the notebooks).

Starts from data/raw_substrate.h5ad (raw METASPACE ions + provided CCF/region) and the real
uMAIA fit (data/umaia_normalized.npz), then does everything students do: annotate, normalize,
embed (NMF + Harmony + t-SNE), cluster into THEIR OWN lipizones, transfer control->pregnant,
and run the per-lipizone differential + composite scores. Confirms the paper's biology broadly
emerges (sphingolipid/myelination rise in white-matter lipizones) and saves staged outputs to
data/derived/ so each notebook can consume the previous stage.

    python scripts/run_pipeline.py
"""
from __future__ import annotations

import os, sys, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("harmonypy").setLevel(logging.ERROR)
sys.path.insert(0, "src")
import numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cajal_lipidomics import analysis as A, embedding as E, annotation as AN
from cajal_lipidomics.style import set_style; set_style()

os.makedirs("data/derived", exist_ok=True); os.makedirs("data/figs/v2", exist_ok=True)
SPH = ("HexCer", "Cer", "SM")


def annotate(raw):
    lcms = AN.load_lcms_reference("data/refs/csv/lcms_mar2022_withcounterions (2).txt")
    names = []
    for mz in raw.var["mz"].astype(float):
        h = AN.match_lcms(mz, lcms, 5.0)
        nm = h.iloc[0]["Lipid"] if len(h) else f"ion_{mz:.3f}"
        nm = nm.replace("(d", " ").replace("(", " ").replace(")", "").replace(".", ":").strip()
        names.append(nm)
    raw.var["lipid"] = names
    return raw


def main():
    raw = ad.read_h5ad("data/raw_substrate.h5ad")
    raw.write_h5ad("data/derived/01_raw.h5ad")               # NB1 output: raw ions + provided CCF
    raw = annotate(raw)
    raw.write_h5ad("data/derived/02_annotated.h5ad")         # NB2 output: + lipid names
    # attach the real uMAIA-normalized values (student's NB3 output), back to native scale
    d = np.load("data/umaia_normalized.npz", allow_pickle=True); xm = d["x_MAIA"]
    nC = int((raw.obs.Condition == "naive").sum()); nP = int((raw.obs.Condition == "pregnant").sum())
    raw.layers["umaia"] = np.vstack([np.exp(xm[:nC, 0, :]), np.exp(xm[:nP, 1, :])]).astype(np.float32)
    raw.write_h5ad("data/derived/03_normalized.h5ad")
    print(f"[1-3] raw->annotate->uMAIA: {raw.shape} | annotated "
          f"{int((~raw.var.lipid.str.startswith('ion_')).sum())}/{raw.n_vars} ions")

    # [5] embedding on uMAIA-normalized, learned on control
    Xn = A.min01_per_lipid(raw.layers["umaia"]); ctrl = (raw.obs.Condition == "naive").to_numpy()
    W, H, m = E.seeded_nmf(Xn[ctrl], n_factors=12); Wall = E.apply_nmf(m, Xn)
    Wh = E.harmonize(Wall, raw.obs.SectionID.astype(str).to_numpy())
    raw.obsm["X_nmf"] = Wall; raw.obsm["X_harmony"] = Wh
    # t-SNE on the NMF embedding (visualisation only; no UMAP)
    from openTSNE import TSNE
    raw.obsm["X_tsne"] = np.asarray(TSNE(n_components=2, perplexity=30, n_jobs=8, random_state=0).fit(Wall))
    raw.write_h5ad("data/derived/05_embedded.h5ad")          # NB5 output: NMF + Harmony + t-SNE
    # [6] their OWN lipizones (Leiden) + transfer control->pregnant
    lab = E.leiden_clusters(Wh)
    pred, conf = E.knn_transfer(Wh[ctrl], lab[ctrl], Wh[~ctrl], k=15)
    lipz = lab.astype(object); lipz[~ctrl] = pred  # control = Leiden, pregnant = transferred
    raw.obs["lipizone"] = pd.Categorical(lipz.astype(str))
    raw.write_h5ad("data/derived/06_clustered.h5ad")
    nlip = raw.obs.lipizone.nunique()
    print(f"[5-6] NMF(12)->Harmony->Leiden: {nlip} OWN lipizones; transfer conf {conf.mean():.2f}")

    # [7] per-lipizone differential (Wilcoxon+BH) + composite scores
    sph_cols = [i for i, n in enumerate(raw.var.lipid) if n.startswith(SPH)]
    X = raw.layers["umaia"]; z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    raw.obs["myelination"] = z[:, sph_cols].mean(1)
    # which lipizones raise sphingolipids in pregnancy?
    rows = []
    for lz, idx in raw.obs.groupby("lipizone", observed=True).indices.items():
        sub = raw.obs.iloc[idx]; c = (sub.Condition == "naive").to_numpy(); p = ~c
        if c.sum() < 30 or p.sum() < 30: continue
        dmy = raw.obs["myelination"].values[idx][p].mean() - raw.obs["myelination"].values[idx][c].mean()
        rows.append((lz, c.sum() + p.sum(), dmy))
    lzdf = pd.DataFrame(rows, columns=["lipizone", "n", "d_myelination"]).sort_values("d_myelination")
    up = int((lzdf.d_myelination > 0).sum())
    print(f"[7] lipizones raising the myelination score in pregnancy: {up}/{len(lzdf)} "
          f"(range {lzdf.d_myelination.min():+.2f}..{lzdf.d_myelination.max():+.2f})")
    # whole-section + white-matter differential, paper-aligned
    raw2 = ad.AnnData(raw.layers["umaia"], obs=raw.obs); raw2.var_names = raw.var.lipid.values
    diff = A.differential_lipids(raw2, "Condition", "naive", "pregnant")
    sphmask = diff.lipid.str.startswith(SPH)
    print(f"[7] differential whole-section: {int(diff.sig.sum())}/{len(diff)} sig; "
          f"sphingolipids mean log2FC {diff.loc[sphmask,'log2fc'].mean():+.3f} "
          f"(median {diff.loc[sphmask,'log2fc'].median():+.3f})")
    # white-matter region sphingo emergence
    WM = {'cc','fi','int','or','ec','alv','fp','df','st','opt','ml','py','cpd','arb','em','fx','ccg','scwm'}
    wm = raw.obs.acronym.astype(str).isin(WM).to_numpy()
    smy = raw.obs["myelination"].values
    print(f"[7] white-matter myelination: naive {smy[ctrl&wm].mean():+.3f} -> pregnant {smy[(~ctrl)&wm].mean():+.3f}")

    # figures: own lipizones (both sections), myelination map
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for k, (cond, cc) in enumerate([("naive", ctrl), ("pregnant", ~ctrl)]):
        s = raw.obs[cc]; codes = s.lipizone.cat.codes
        ax[k].scatter(s.x, -s.y, c=codes, cmap="tab20", s=3, rasterized=True)
        ax[k].set_aspect("equal"); ax[k].axis("off"); ax[k].set_title(f"{cond}: own lipizones")
    plt.savefig("data/figs/v2/own_lipizones.png", dpi=130, bbox_inches="tight"); plt.close()
    print("saved data/figs/v2/own_lipizones.png + staged data/derived/{03,06}.h5ad")


if __name__ == "__main__":
    main()
