"""Download the course data bundle from Zenodo and unzip it into data/.

    python scripts/fetch_data_bundle.py

Resumable (safe to re-run on a flaky connection: it continues, it does not restart). If the Zenodo
record is still a draft, set ZENODO_TOKEN in your environment and the script fetches the draft via
the API; once the record is published the same command works for everyone with no token.

The bundle holds the provided inputs (registration/CCF, reference databases, tissue masks, the
MERFISH plane subset + region-averaged expression, Gene Ontology files). You pull the raw MALDI-MSI
yourself from METASPACE in notebook 1; data/derived/ you build by running notebooks 1-6 in order.
"""
from __future__ import annotations
import os, subprocess, zipfile

DEST = "data/course_data_bundle.zip"
PUBLIC = "https://zenodo.org/records/21058014/files/course_data_bundle.zip?download=1"
BUCKET = "https://zenodo.org/api/files/bf78b5ec-d682-47fb-82e6-58c4dc7f0f93/course_data_bundle.zip"
MIN_BYTES = 1_000_000_000  # the bundle is ~1.01 GB; anything smaller is a partial download

# The full LIPID MAPS structure database (NB2 matches peaks against all of it). It is 254 MB and
# ships separately on Zenodo rather than inside the bundle zip; fetch it into data/refs/ if absent.
SDF_DEST = "data/refs/structures.sdf"
SDF_URL = "https://zenodo.org/records/15650014/files/structures.sdf?download=1"
SDF_MIN_BYTES = 200_000_000  # ~254 MB; anything smaller is a partial download

def main():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(DEST) and os.path.getsize(DEST) >= MIN_BYTES:
        print(f"{DEST} already present.")
    else:
        tok = os.environ.get("ZENODO_TOKEN")
        url = f"{BUCKET}?access_token={tok}" if tok else PUBLIC  # token -> draft access; else public
        print("downloading the data bundle (~1 GB, resumable) ...")
        subprocess.run(["curl", "-L", "-C", "-", "--fail", url, "-o", DEST], check=True)
    print("unzipping into data/ ...")
    with zipfile.ZipFile(DEST) as z:
        z.extractall(".")  # archive paths are already data/...

    if os.path.exists(SDF_DEST) and os.path.getsize(SDF_DEST) >= SDF_MIN_BYTES:
        print(f"{SDF_DEST} already present.")
    else:
        os.makedirs("data/refs", exist_ok=True)
        print("downloading the LIPID MAPS structure database (~254 MB, resumable) ...")
        subprocess.run(["curl", "-L", "-C", "-", "--fail", SDF_URL, "-o", SDF_DEST], check=True)
    print("done: provided inputs are under data/")

if __name__ == "__main__":
    main()
