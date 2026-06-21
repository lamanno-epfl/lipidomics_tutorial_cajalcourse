"""Fetch the annotation reference data into data/refs/ (run once, before notebook 2).

All from public Zenodo records. The LC-MS / annotation CSVs live inside the LBA `csv.zip`
(3.7 GB); we extract just the few small files we need via HTTP range requests (remotezip),
so nothing large is downloaded. Verified against the live records 2026-06-21.

    pip install remotezip
    python scripts/fetch_references.py
"""
from __future__ import annotations

import os
import urllib.request

OUT = "data/refs"
CSV_ZIP = "https://zenodo.org/records/15379565/files/csv.zip?download=1"
# small files to pull out of csv.zip (Zenodo 15379565):
CSV_MEMBERS = [
    "csv/cleanedANNOTATIONS_20250215.csv",      # m/z -> Annotation + Score + per-source matches
    "csv/ALLANNOTATIONSCORES_20250215.csv",
    "csv/lcms_mar2022_withcounterions (2).txt",  # in-house LC-MS: Lipid, Adduct, m/z (the ppm-plot reference)
    "csv/QuantitativeLCMS.csv",                  # bulk LC-MS abundances (Female/Male) for tie-breaking
    "csv/qLCMS_regions_fitzner.csv",
    "csv/acquisitions_metadata.csv",
]
# LIPID MAPS + HMDB reference databases (Zenodo 15650014):
DB_FILES = {
    "HMDB_complete.csv": "https://zenodo.org/records/15650014/files/HMDB_complete.csv?download=1",
    "lipidclasscolors.h5ad": "https://zenodo.org/records/15650014/files/lipidclasscolors.h5ad?download=1",
    "structures.sdf": "https://zenodo.org/records/15650014/files/structures.sdf?download=1",  # 254 MB
}


def main(get_sdf=True):
    os.makedirs(OUT, exist_ok=True)
    from remotezip import RemoteZip
    print("Extracting LC-MS / annotation CSVs from csv.zip via range requests ...")
    with RemoteZip(CSV_ZIP) as z:
        for m in CSV_MEMBERS:
            try:
                z.extract(m, OUT)
                print("  +", m)
            except Exception as e:
                print("  skip", m, repr(e)[:80])
    print("Downloading LIPID MAPS / HMDB databases ...")
    for fn, url in DB_FILES.items():
        if fn == "structures.sdf" and not get_sdf:
            continue
        dst = os.path.join(OUT, fn)
        if os.path.exists(dst):
            print("  =", fn, "(present)")
            continue
        urllib.request.urlretrieve(url, dst)
        print("  +", fn, f"({os.path.getsize(dst)/1e6:.1f} MB)")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
