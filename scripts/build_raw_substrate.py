"""Build the RAW student starting point from METASPACE (no pre-processed science leaked).

Students must produce the science end to end. So the only things we hand them are:
  - raw MALDI ion intensities pulled from the public METASPACE project (pixels x ions),
  - per-pixel image coordinates (x, y) and a tissue mask,
  - per-pixel Allen CCF coordinates + region (the REGISTRATION output, the one infra piece we
    provide because ABBA/STalign registration is taught as a concept). We obtain it by
    registering the raw tissue silhouette to maindata_2's silhouette (similarity transform,
    IoU ~0.97) and transferring CCF/acronym from the nearest maindata pixel. No lipid names,
    no normalization, no lipizones: students annotate, uMAIA-normalize, embed and cluster
    themselves, getting their OWN lipizones (which will not match the paper, the realistic case).

Output: data/raw_substrate.h5ad  (X = raw ion intensities; obs has x,y + provided xccf/yccf/zccf/
acronym/allencolor + Condition/SectionID; var = formula/adduct/mz). Gitignored.

    python scripts/build_raw_substrate.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.spatial import cKDTree
from skimage.transform import SimilarityTransform, warp
from metaspace import SMInstance

DB = ("CoreMetabolome", "v3"); FDR = 0.1
SECTIONS = {
    "control_217D":       dict(ds_id="2025-04-27_08h23m47s", sid=75.0,  cond="naive",
        mask="data/masks/BrainAtlas/Control_Brains/female/20220416_MouseBrain_female_217D_447x332_Att30_25um/mask.npy"),
    "pregnant_Brain1_C2": dict(ds_id="2024-07-14_14h24m11s", sid=110.0, cond="pregnant",
        mask="data/masks/PREGNANT/20240712_MouseBrain_LipidAtlas_Pregnant_Brain1_C2_459x352_25um_Att30/mask.npy"),
}


def pull_section(ds_id):
    sm = SMInstance(); ds = sm.dataset(id=ds_id)
    res = ds.results(database=DB, fdr=FDR)
    imgs = ds.all_annotation_images(fdr=FDR, database=DB, only_first_isotope=True, scale_intensity=False)
    keys = [f"{f}__{a}" for f, a in res.index]
    arrs = {k: np.asarray(s[0], np.float32) for k, s in zip(keys, imgs)}
    var = pd.DataFrame({"formula": [i[0] for i in res.index], "adduct": [i[1] for i in res.index],
                        "mz": res["mz"].values}, index=keys)
    return arrs, var


def register_ccf(raw_mask, sid):
    """Register raw tissue silhouette -> maindata silhouette (similarity via image moments),
    then transfer CCF + Allen acronym/color from the nearest maindata pixel to each raw pixel."""
    md = pq.read_table("maindata_2.parquet",
                       columns=["SectionID", "x", "y", "xccf", "yccf", "zccf", "acronym", "allencolor"],
                       filters=(pc.field("SectionID") == sid)).to_pandas()
    A = np.zeros((int(md.y.max()) + 2, int(md.x.max()) + 2), bool); A[md.y.astype(int), md.x.astype(int)] = True

    def moments(m):
        ys, xs = np.nonzero(m); c = np.array([xs.mean(), ys.mean()])
        cov = np.cov(np.stack([xs - c[0], ys - c[1]])); w, v = np.linalg.eigh(cov)
        return c, np.arctan2(v[1, -1], v[0, -1]), m.sum()
    cA, aA, sA = moments(A); cB, aB, sB = moments(raw_mask); scale = np.sqrt(sA / sB)

    def iou(a, b):
        h = max(a.shape[0], b.shape[0]); w = max(a.shape[1], b.shape[1])
        aa = np.zeros((h, w), bool); aa[:a.shape[0], :a.shape[1]] = a
        bb = np.zeros((h, w), bool); bb[:b.shape[0], :b.shape[1]] = b
        return (aa & bb).sum() / max((aa | bb).sum(), 1)
    best = None
    for dang in (aA - aB, aA - aB + np.pi):
        T = (SimilarityTransform(translation=-cB) + SimilarityTransform(scale=scale)
             + SimilarityTransform(rotation=dang) + SimilarityTransform(translation=cA))
        Bw = warp(raw_mask.astype(float), T.inverse, output_shape=A.shape, order=0) > 0.5
        sc = iou(A, Bw)
        if best is None or sc > best[0]:
            best = (sc, T)
    score, T = best
    # raw tissue pixels -> maindata coords -> nearest maindata pixel -> CCF/region
    ys, xs = np.nonzero(raw_mask)
    pts = T(np.column_stack([xs, ys]))  # (x,y) in maindata frame
    tree = cKDTree(md[["x", "y"]].to_numpy())
    _, nn = tree.query(pts, k=1)
    ccf = md.iloc[nn][["xccf", "yccf", "zccf", "acronym", "allencolor"]].reset_index(drop=True)
    return score, ys, xs, ccf


def main():
    var_sets = {}; per = {}
    for name, cfg in SECTIONS.items():
        arrs, var = pull_section(cfg["ds_id"]); var_sets[name] = set(var.index)
        per[name] = dict(arrs=arrs, var=var, mask=np.load(cfg["mask"]), cfg=cfg)
        print(f"{name}: {len(arrs)} ions, image {next(iter(arrs.values())).shape}")
    common = sorted(set.intersection(*var_sets.values()))
    print(f"common ions across sections: {len(common)}")

    blocks = []
    for name, P in per.items():
        mask = P["mask"]; score, ys, xs, ccf = register_ccf(mask, P["cfg"]["sid"])
        print(f"{name}: silhouette registration IoU {score:.3f}, {len(xs)} tissue pixels")
        X = np.stack([P["arrs"][k][ys, xs] for k in common], axis=1).astype(np.float32)
        obs = pd.DataFrame({"SectionID": P["cfg"]["sid"], "Condition": P["cfg"]["cond"],
                            "x": xs, "y": ys, "xccf": ccf["xccf"].values, "yccf": ccf["yccf"].values,
                            "zccf": ccf["zccf"].values, "acronym": ccf["acronym"].values,
                            "allencolor": ccf["allencolor"].values})
        obs.index = [f"{name}_{i}" for i in range(len(obs))]
        a = ad.AnnData(X=X, obs=obs); a.var_names = common
        blocks.append(a)
    adata = ad.concat(blocks)
    v0 = per[list(per)[0]]["var"].loc[common]
    adata.var = v0
    adata.obs["Condition"] = adata.obs["Condition"].astype(str)
    adata.write_h5ad("data/raw_substrate.h5ad")
    print(f"\nwrote data/raw_substrate.h5ad: {adata.shape[0]} pixels x {adata.shape[1]} raw ions")
    print(adata.obs.groupby(["Condition"], observed=True).size())
    print("regions per condition:", adata.obs.groupby("Condition", observed=True)["acronym"].nunique().to_dict())


if __name__ == "__main__":
    main()
