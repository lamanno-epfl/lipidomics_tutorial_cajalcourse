# Setup guide — get your laptop ready before the course

Do this at home, with time to spare. None of it needs prior programming. Each step ends with a
check: if you see the expected output, move on; if not, write to **luca.fusarbassini@epfl.ch** with
the error, or ask Claude Code once you reach that step.

You need about an hour, plus download time. Works on macOS, Windows, and Linux.

---

## 1. A terminal

The terminal is a text window where you type commands. You already have one.

- **macOS**: open Spotlight (`Cmd`+`Space`), type `Terminal`, press Enter.
- **Windows**: install [Git for Windows](https://git-scm.com/download/win) (accept the defaults).
  This gives you **Git Bash**, a terminal that understands the same commands as Mac and Linux. Use
  Git Bash throughout this course, not PowerShell.
- **Linux**: open your Terminal app.

**Check**: type `pwd` and press Enter. It prints the folder you are in.

## 2. Miniforge (Python + the conda/mamba package manager)

We use Miniforge: it is free, minimal, and installs scientific packages reliably. Do not install
the full Anaconda.

- Download the installer for your system from <https://github.com/conda-forge/miniforge>.
  - macOS / Linux: download the `.sh`, then run `bash Miniforge3-*.sh` and accept the defaults.
  - Windows: download the `.exe`, double-click, accept the defaults. Afterwards use the
    **Miniforge Prompt** (or Git Bash) as your terminal.
- Close and reopen the terminal.

**Check**: `mamba --version` prints a version number.

## 3. Get the course and create the environment

```bash
git clone <this repository>
cd lipidomics_tutorial_cajalcourse

# 1) the main analysis environment
mamba env create -f environment.yml
mamba activate cajal-lipidomics
pip install -e .                      # the cajal_lipidomics helper package the notebooks import
pip install -r requirements-extra.txt
python -m ipykernel install --user --name cajal-lipidomics --display-name "cajal-lipidomics"
mamba deactivate

# 2) the normalization environment (notebook 3 only: uMAIA needs numpy<2 + jax)
mamba env create -f environment-umaia.yml
mamba activate cajal-umaia
python -m ipykernel install --user --name cajal-umaia --display-name "cajal-umaia"
mamba deactivate
```

Most notebooks use the **cajal-lipidomics** kernel; **notebook 3** (uMAIA normalization) uses the
**cajal-umaia** kernel. EUCLID is not installed here: notebook 6 clones and runs it at that point.

**Check**: with `cajal-lipidomics` active, `python -c "import scanpy, anndata, xgboost, cajal_lipidomics; print('ok')"` prints `ok`.

(Later, during the course, you will refresh the notebooks with `git pull`.)

## 3b. Get the data bundle

The provided inputs (registration, references, masks, MERFISH, Gene Ontology) live in one
~1 GB bundle on Zenodo. With `cajal-lipidomics` active, from the repo root:

```bash
python scripts/fetch_data_bundle.py    # downloads + unzips into data/
```

You pull the raw MALDI-MSI yourself from METASPACE in notebook 1, and build `data/derived/`
by running notebooks 1-6 in order.


## 4. VS Code and Jupyter

- Install [VS Code](https://code.visualstudio.com/).
- In VS Code, open the Extensions panel and install **Python** and **Jupyter** (both by Microsoft).
- Open the course folder: `File > Open Folder...` and choose `lipidomics_tutorial_cajalcourse`.
- Open `notebooks/00_intro/00_tooling_student.ipynb`.
- **Select the kernel**: top right of the notebook, click the kernel picker and choose
  **cajal-lipidomics**. This is the single most common thing beginners miss.

**Check**: run the first cell with `Shift`+`Enter`. It runs without an error.

## 5. Claude Code (your AI pair-programmer)

We use Claude Code from the second half of the course. Quickstart:
<https://code.claude.com/docs/en/quickstart>.

- It needs a Claude login (we will sort access for the course).
- You run `claude` in a terminal inside the course folder, or use the VS Code integration.
- It can read your files, explain code, and propose edits, and it asks before changing anything.
- Use it well: read what it writes. If you do not understand a line, stop and look it up. The point
  is to learn, not to autocomplete past your understanding.

## 6. A GitHub account

Create a free account at <https://github.com>. You will use it to pull updates and, if you like, to
save your own work.

---

You are ready when steps 3 and 4 both print `ok` and a notebook cell runs. Next, work through the
three notebooks in `notebooks/00_intro/`.
