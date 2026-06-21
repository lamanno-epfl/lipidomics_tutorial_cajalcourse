"""Assemble the student data bundle: everything the notebooks load, in one zip.

Includes a MERFISH plane-subset (cells near the course AP level) so students do not need
the full 5.7 GB per-cell file. Run after the substrate, refs, and uMAIA input exist.

    python scripts/build_bundle.py            # build merfish subset + zip
    python scripts/build_bundle.py --no-sdf   # skip the 254 MB LIPID MAPS sdf
"""
from __future__ import annotations

import os
import sys
import zipfile
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

BUNDLE = "data/course_data_bundle.zip"
MERFISH_FULL = "cell_filtered_w500genes.parquet"
MERFISH_SUB = "data/merfish_plane.parquet"
AP = (5.8, 7.2)  # CCF x window around the course coronal plane (~6.5)

# (path_on_disk, archive_name) — what students get
def manifest(include_sdf=True):
    items = [
        ("data/sections_pair.h5ad", "sections_pair.h5ad"),           # the analysis substrate
        ("data/umaia_input.npz", "umaia_input.npz"),                 # NB3 uMAIA input tensor
        ("avemerfish_imputed_named.parquet", "avemerfish_imputed_named.parquet"),  # NB8 region x gene
        (MERFISH_SUB, "merfish_plane.parquet"),                      # NB8 per-cell (plane subset)
        ("data/refs/HMDB_complete.csv", "refs/HMDB_complete.csv"),
        ("data/refs/lipidclasscolors.h5ad", "refs/lipidclasscolors.h5ad"),
    ]
    for f in ["cleanedANNOTATIONS_20250215.csv", "ALLANNOTATIONSCORES_20250215.csv",
              "QuantitativeLCMS.csv", "lcms_mar2022_withcounterions (2).txt", "qLCMS_regions_fitzner.csv",
              "acquisitions_metadata.csv"]:
        items.append((f"data/refs/csv/{f}", f"refs/csv/{f}"))
    for m in ["BrainAtlas/Control_Brains/female/20220416_MouseBrain_female_217D_447x332_Att30_25um/mask.npy",
              "PREGNANT/20240712_MouseBrain_LipidAtlas_Pregnant_Brain1_C2_459x352_25um_Att30/mask.npy"]:
        items.append((f"data/masks/{m}", f"masks/{m}"))
    if include_sdf:
        items.append(("data/refs/structures.sdf", "refs/structures.sdf"))
    return items


def build_merfish_subset():
    if os.path.exists(MERFISH_SUB):
        print(f"  {MERFISH_SUB} exists, skipping")
        return
    print(f"  building MERFISH plane subset (x_ccf in {AP}) ...")
    flt = (pc.field("x_ccf") >= AP[0]) & (pc.field("x_ccf") <= AP[1])
    t = pq.read_table(MERFISH_FULL, filters=flt)
    pq.write_table(t, MERFISH_SUB, compression="zstd")
    print(f"  {MERFISH_SUB}: {t.num_rows} cells, {os.path.getsize(MERFISH_SUB)/1e6:.0f} MB")


def main():
    include_sdf = "--no-sdf" not in sys.argv
    build_merfish_subset()
    items = manifest(include_sdf)
    missing = [p for p, _ in items if not os.path.exists(p)]
    if missing:
        print("WARNING missing (will skip):")
        for m in missing:
            print("   ", m)
    total = 0
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in items:
            if os.path.exists(p):
                z.write(p, arcname=f"data/{arc}")
                total += os.path.getsize(p)
        # a README inside the bundle
        z.writestr("data/README.txt",
                   "CAJAL spatial-metabolomics course data bundle.\n"
                   "Unzip into the repo root so files land under data/.\n"
                   "See docs/SETUP.md and the notebooks for how each file is used.\n")
    print(f"\nwrote {BUNDLE}: {os.path.getsize(BUNDLE)/1e6:.0f} MB "
          f"({total/1e6:.0f} MB uncompressed, {len([p for p,_ in items if os.path.exists(p)])} files)")


if __name__ == "__main__":
    main()
