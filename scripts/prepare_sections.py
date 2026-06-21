"""Build the analysis substrate for the course's two chosen sections.

From maindata_2.parquet (the full Lipid Brain Atlas table) we extract just the two
matched sections and write a small AnnData that every notebook can load instantly:

    control  : 217D            (SectionID 75,  naive female,  xccf ~6.521)
    pregnant : Brain1_C2       (SectionID 110, pregnant,      xccf ~6.514)

These are an essentially identical coronal plane (AP gap ~0.007 in CCF units), both
on the public METASPACE project `mlba-2025`, both feature-rich. maindata_2 already
carries, per pixel: the 172 uMAIA-normalized lipids, CCF coordinates, Allen region
(acronym / colour / division), and the lipizone hierarchy. We use it as ground truth
for building the notebooks; the METASPACE pull and uMAIA steps are taught separately.

X        = pixels x 172 lipids (uMAIA-normalized, native scale)
obs      = SectionID, Sample, Sex, Condition, Path, CCF (xccf/yccf/zccf + indices),
           Allen (acronym/name/allencolor/division), lipizone hierarchy, tsne.

Run:  python scripts/prepare_sections.py
Out:  data/sections_pair.h5ad   (gitignored)
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import anndata as ad

MAINDATA = "maindata_2.parquet"
OUT = "data/sections_pair.h5ad"
CONTROL_SECTION = 75.0   # 217D, naive female
PREGNANT_SECTION = 110.0  # Brain1_C2, pregnant


def is_lipid(col: str) -> bool:
    # lipid shorthand always carries a "carbons:doublebonds" token, e.g. "PC 38:6",
    # "HexCer 42:2;O2"; metadata columns never do.
    return bool(re.search(r"\d+:\d+", col))


def main() -> None:
    schema = pq.read_schema(MAINDATA)
    cols = list(schema.names)
    cols = [c for c in cols if c != "__index_level_0__"]
    lipids = [c for c in cols if is_lipid(c)]
    meta = [c for c in cols if not is_lipid(c)]
    print(f"columns: {len(cols)} | lipids: {len(lipids)} | metadata: {len(meta)}")

    filt = pc.is_in(pc.field("SectionID"),
                    value_set=__import__("pyarrow").array([CONTROL_SECTION, PREGNANT_SECTION]))
    tbl = pq.read_table(MAINDATA, columns=lipids + meta, filters=filt)
    df = tbl.to_pandas()
    print(f"rows for the two sections: {len(df)}")
    print(df.groupby(["SectionID", "Condition", "Sample"]).size())

    X = df[lipids].to_numpy(dtype=np.float32)
    obs = df[meta].copy()
    obs.index = obs.index.astype(str)
    # h5ad can't store mixed/list-valued object columns; stringify them (lists -> joined).
    for c in obs.columns:
        if obs[c].dtype == object:
            obs[c] = obs[c].apply(
                lambda v: ",".join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray)) else
                ("" if v is None or (np.isscalar(v) and pd.isna(v)) else str(v))
            )
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = pd.Index(lipids)
    adata.var["is_lipid"] = True

    # quick sanity: both conditions, finite values, CCF + regions present
    assert set(adata.obs["Condition"].unique()) == {"naive", "pregnant"}, adata.obs["Condition"].unique()
    assert np.isfinite(adata.X).all(), "non-finite lipid values"
    for c in ["xccf", "yccf", "zccf", "acronym", "allencolor", "lipizone_names"]:
        assert c in adata.obs, f"missing {c}"

    import os
    os.makedirs("data", exist_ok=True)
    adata.write_h5ad(OUT)
    print(f"\nwrote {OUT}: {adata.shape[0]} pixels x {adata.shape[1]} lipids")
    print("regions per section (top 5):")
    for sid, sub in adata.obs.groupby("SectionID"):
        print(f"  section {sid} ({sub['Condition'].iloc[0]}): "
              f"{sub['acronym'].nunique()} Allen regions, "
              f"{sub['lipizone_names'].nunique()} lipizones")


if __name__ == "__main__":
    main()
