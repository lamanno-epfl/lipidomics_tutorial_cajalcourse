"""Fetch the annotation reference data (run once, before building notebook 2).

Pulls, into ../data/:
  - LIPID MAPS + HMDB databases from Zenodo 15650014 (structures.sdf, HMDB_complete.csv,
    lipidclasscolors.h5ad), ~265 MB.
  - tissue masks from Zenodo 16524009 (~50 KB).
  - the two LC-MS CSVs (lipids_processed.csv, lcms_females_tutorial.csv) from the LBA
    csv.zip in Zenodo 15379565, by HTTP range request so we never download the full 3.7 GB.

Implemented in M-data (after the disk move frees space). The range-extraction of the two
CSVs uses the zip central directory + per-file byte ranges (e.g. the `remotezip` package),
so only a few MB are transferred. Verified against the real records before use; no guessing.
"""
from __future__ import annotations

# TODO(M-data): implement; download DBs + masks, range-extract the two LC-MS CSVs.
