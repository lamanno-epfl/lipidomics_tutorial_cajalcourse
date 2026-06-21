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
    for s in ["fetch_references.py", "build_umaia_input.py", "run_umaia.py", "move_to_mnt.py"]:
        if os.path.exists(f"scripts/{s}"):
            shutil.copy(f"scripts/{s}", f"{REL}/scripts/")

    # data manifest (the bundle is downloaded separately, not committed)
    os.makedirs(f"{REL}/data", exist_ok=True)
    if os.path.exists("data/README.md"):
        shutil.copy("data/README.md", f"{REL}/data/")

    n_nb = len(glob.glob(f"{REL}/notebooks/**/*.ipynb", recursive=True))
    print(f"release tree at {REL}/ : {n_nb} notebooks (intros + student), helpers, setup, manifest.")
    print("Solutions, dev notes, and source material are intentionally excluded.")
    print("Next: cd release && git init && git add -A && git commit && push to the student repo,")
    print("or zip it for distribution.")


if __name__ == "__main__":
    main()
