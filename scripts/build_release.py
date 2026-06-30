"""Assemble the student-facing release tree (the second, public repo).

This repo is the private dev repo (it holds the solutions, dev notes, and source material).
The student release is a clean subset: the intro notebooks, the STUDENT notebooks (no
solutions), the helper package, setup + environment, and a data manifest. Run this, then push
release/ to the student GitHub repo (or zip it).

    python scripts/build_release.py        # -> release/  (gitignored)
"""
from __future__ import annotations

import os
import shutil
import glob

REL = "release"


def main():
    if os.path.exists(REL):
        shutil.rmtree(REL)
    os.makedirs(REL)

    # top-level files students need
    for f in ["README.md", "environment.yml", "environment-umaia.yml",
              "requirements-extra.txt", "pyproject.toml", ".gitignore"]:
        if os.path.exists(f):
            shutil.copy(f, REL)
    shutil.copytree("docs", f"{REL}/docs")
    shutil.copytree("src", f"{REL}/src")
    shutil.copytree("assets", f"{REL}/assets")

    # intro notebooks (runnable) + STUDENT notebooks only (no solutions)
    os.makedirs(f"{REL}/notebooks/00_intro", exist_ok=True)
    for p in glob.glob("notebooks/00_intro/*.ipynb"):
        shutil.copy(p, f"{REL}/notebooks/00_intro/")
    for lvl in ["level1", "level2", "level3"]:
        os.makedirs(f"{REL}/notebooks/{lvl}", exist_ok=True)
        for p in glob.glob(f"notebooks/{lvl}/*_student.ipynb"):
            shutil.copy(p, f"{REL}/notebooks/{lvl}/")

    # the scripts students run (fetch data, build uMAIA input); NOT make_student/build_release/destyle
    os.makedirs(f"{REL}/scripts", exist_ok=True)
    for s in ["fetch_data_bundle.py", "fetch_references.py", "build_umaia_input.py", "run_umaia.py", "move_to_mnt.py"]:
        if os.path.exists(f"scripts/{s}"):
            shutil.copy(f"scripts/{s}", f"{REL}/scripts/")

    # data manifest: a STUDENT-facing one (no teacher-only files like maindata_2)
    os.makedirs(f"{REL}/data", exist_ok=True)
    with open(f"{REL}/data/README.md", "w") as fh:
        fh.write(
            "# Data\n\n"
            "The raw MALDI-MSI you pull yourself from METASPACE (project `mlba-2025`) inside notebook 1.\n"
            "Everything else you build by running the notebooks in order (they save into `data/derived/`).\n\n"
            "The only things provided for you live in the course data bundle "
            "(`course_data_bundle.zip`, ~1 GB on Zenodo). Fetch + unzip it in one step:\n\n"
            "```\npython scripts/fetch_data_bundle.py      # downloads from Zenodo, unzips into data/\n```\n\n"
            "| file | what it is | used in |\n|---|---|---|\n"
            "| `provided/registration_ccf.parquet` | Allen CCF coordinate + region per pixel (the registration output; ABBA is taught as a concept) | NB1, NB4 |\n"
            "| `refs/` | LIPID MAPS / HMDB / LC-MS reference tables for annotation | NB2 |\n"
            "| `masks/` | tissue masks per section | NB1, NB3 |\n"
            "| `merfish_plane.parquet` | per-cell MERFISH near the course plane | NB8 |\n"
            "| `avemerfish_imputed_named.parquet` | region-averaged MERFISH gene expression | NB8 |\n\n"
            "`data/derived/` (01_raw -> 06_clustered) is produced BY YOU as you run notebooks 1-6, in order.\n")

    n_nb = len(glob.glob(f"{REL}/notebooks/**/*.ipynb", recursive=True))
    print(f"release tree at {REL}/ : {n_nb} notebooks (intros + student), helpers, setup, manifest.")
    print("Solutions, dev notes, and source material are intentionally excluded.")
    print("Next: cd release && git init && git add -A && git commit && push to the student repo,")
    print("or zip it for distribution.")


if __name__ == "__main__":
    main()
