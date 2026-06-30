"""Download the course data bundle from Zenodo and unzip it into data/.

    python scripts/fetch_data_bundle.py

The bundle holds the provided inputs (registration/CCF, reference databases, tissue masks, the
MERFISH plane subset + region-averaged expression, and the Gene Ontology files). The raw MALDI-MSI
you pull yourself from METASPACE in notebook 1; data/derived/ you build by running the notebooks.
"""
from __future__ import annotations
import os, sys, zipfile, urllib.request

URL = "https://zenodo.org/records/21058014/files/course_data_bundle.zip?download=1"
DEST = "data/course_data_bundle.zip"

def main():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DEST):
        print(f"downloading {URL}\n(this is ~1 GB) ...")
        urllib.request.urlretrieve(URL, DEST)
    print("unzipping into data/ ...")
    with zipfile.ZipFile(DEST) as z:
        z.extractall(".")          # archive paths are already data/...
    print("done: provided inputs are now under data/")

if __name__ == "__main__":
    main()
