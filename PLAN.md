# CAJAL NEUROMICS 2026 — "A spatial metabolomics primer" — Course Build Plan

Author: Luca Fusar Bassini (EPFL). Block 2 (computational week), Bordeaux, July 2026.
This is the living design doc. It is informed by 14 deep-reading digests under `knowledge/digests/`.
Status: **DRAFT for Luca's feedback** — nothing in `course/` built yet.

---

## 1. What we are building (one paragraph)

A transparent, didactic, notebook-based course that walks absolute beginners through the full
spatial-metabolomics path on **two MALDI-MSI sections** — one control female, one pregnant female
mouse brain — from raw spectra to a publication-quality multi-panel figure. The science mirrors the
Lipid Brain Atlas pregnancy story but **simplified, extended, and made fully transparent**: students
read/write explicit Python line by line, calling the public EUCLID + uMAIA **minimally**; we unroll
EUCLID's internals into visible teaching code, borrowing cleaner implementations from the unreleased
`euclid_portal` where useful. Theory is dense, in Luca's LBA writing voice, with plots at every step.

Non-negotiables (from PHILOSOPHY.md + LEGISLATION.md + the brief): no fabrication; never guess —
verify or ask; transparency over API; reproducibility (pinned deps, seeds); test small then scale;
commit aggressively; clean/minimal code; figures per the LEGISLATION figure rules; every number paired
with a biological reading.

---

## 2. Hard constraints (from context.md + digests)

- **9 main sessions × 4.5 h (Jul 2–9), ONE notebook per session, difficulty graded evenly.** Plus a
  **self-guided ~5 h intro on Jul 1** (students do alone to arrive prepared) + SETUP done at home before.
  "Build slowly and steady, digest every concept." Generous time → we cover the full arc thoroughly.
- **No GPU** anywhere (IFB cluster: 280 CPU / 2.2 TB RAM / 1 TB; school Windows PCs + weaker Linux).
  Everything must be CPU-feasible. This rules out the lipiMap VAE and Lipidome2Location MLP as core.
- **Exactly 2 sections** → uMAIA normalization is "a touch dirty"; several atlas steps degenerate
  (AP-continuity, dropout/variance feature filters, per-section ML). Teach the corner case honestly.
- **Absolute beginners** (Janine: zero programming). Home-prep + maximal hand-holding at Level 1.
- **METASPACE project `mlba-2025` is fully public — no login needed** to list/download (corrected from
  my earlier assumption). The MERFISH data and the LC-MS annotation references are NOT yet in hand.

---

## 3. Repository architecture (two repos, one source of truth)

**(A) Private dev repo** = THIS repo `/home/fusar/lipidomics_tutorial_cajalcourse`. Holds full solutions,
all development, the digests, data-prep, and the generator. `bordeaux-material/` stays gitignored.

```
lipidomics_tutorial_cajalcourse/            (PRIVATE)
  bordeaux-material/        gitignored: source material + _extracted/ (EUCLID, uMAIA, repos, papers)
  knowledge/                digests/ + nb_text/  (research notes — gitignored or kept; decide)
  PLAN.md                   this file
  course/
    env/                    environment-intro.yml, environment-project.yml  (pinned, CPU)
    SETUP_GUIDE.md          cross-platform zero-to-running checklist
    notebooks/
      00_intro/             00_tooling, 01_python_for_data, 02_concepts        (SOLUTION = runnable)
      level1/               the transparent MSI core (SOLUTIONS, fully runnable)
      level2/               MERFISH multimodal + reproduce-a-panel (SOLUTIONS)
      level3/               XGBoost+SHAP+GO capstone + open analysis (SOLUTION + blank)
    src/cajal_lipidomics/   small importable helper+plotting module (the "ready plotting functions")
    data/                   gitignored data bundle + build/download scripts (§10)
    scripts/                pull_metaspace.py, run_umaia.py, build_ccf.py, make_student.py, ...
  student-release/          generated staging for the PUBLIC student repo (or push to separate GH repo)
```

**(B) Student release repo** (separate GitHub repo students clone). Contains: SETUP_GUIDE, the 3 intro
notebooks, the **student** notebooks (TODO-blanked), the helper module, `environment.yml`, the data
bundle (or download scripts), and a `README` learning guide. Updated via `git pull` (boss's mechanism).

**Generator `scripts/make_student.py`**: ONE source of truth = the solution notebooks. The student
notebooks are **auto-generated** by stripping code cells to `# TODO` blanks while keeping all prose,
the 🔬 TASK / 💡 HINT / ❓ QUESTION / ⚠️ CHECKPOINT markers, and any cells tagged `keep` (provided
plotting/scaffold). Tag cells with notebook metadata (`solution`, `keep`, `task`) to control stripping.
This honors LEGISLATION ("generated files not hand-edited; one command produces all outputs").

---

## 4. Environments (conda/mamba, CPU, pinned)

- **`cajal-intro`** (home prep, light, instant): python, numpy, pandas, matplotlib, seaborn, scipy,
  statsmodels, scikit-learn, scanpy+anndata, jupyter, ipykernel. No uMAIA/EUCLID. Used by the 3 intro
  notebooks (synthetic toy data only).
- **`cajal-lipidomics`** (project): the above + `euclid_msi` (pip install -e the cloned EUCLID),
  `uMAIA` (+ jax/jaxlib/numpyro/optax **CPU** pins from uMAIA requirements: jax==0.4.14 etc.),
  `metaspace2020`, `xgboost`, `shap`, `squidpy`, `goatools`, `mygene`, `kneed` (optional),
  `allensdk`/`brainglobe-atlasapi` (heavy — pre-cache atlas), `STalign` (optional, registration),
  `adjustText`. Provide `environment-project.yml` + a one-line `mamba env create` + an `ipykernel`
  registration step (the #1 beginner stumble — covered in SETUP_GUIDE).

Test the full env install on a clean machine before the course (LEGISLATION: reproducibility).

---

## 5. Home prep — SETUP_GUIDE + 3 intro notebooks (Digest 14)

Done at home / around the Wed Jul 1 Exercise-0 slot. Synthetic/toy data only (no pregnancy data).

- **`SETUP_GUIDE.md`** (markdown, not a notebook): per-OS (Mac / Windows-via-GitBash / Linux) checklist
  with copy-paste blocks and a "you're done when you see X" check per step: terminal basics → Miniforge
  → `mamba env create` → `ipykernel install` → git clone/pull → VSCode + Python + Jupyter + kernel
  selection → Claude Code basics. Adapt Carpentries setup pages + Miniforge README (links in Digest 14).
- **`00_tooling`** — "Your computer as a lab bench": shell (pwd/ls/cd), git clone/pull, conda/mamba
  envs, VSCode+Jupyter kernel, Claude Code (ask-to-explain, permission-to-edit, "verify, don't trust").
- **`01_python_for_data`** — numpy / pandas / matplotlib (+ a taste of sklearn `fit/transform`) on a
  synthetic **pixels×lipids** toy table; the AnnData "smart container" mental model (`.X/.obs/.var/.obsm`),
  reframed cells→pixels, genes→lipid m/z. Point to VanderPlas PDSH for depth.
- **`02_concepts`** — "the ideas behind the buttons", just-in-time order mirroring the pipeline:
  PCA → NMF → (t-SNE/UMAP on top of PCA/NMF) → Harmony → kNN graph + Leiden → kNN label transfer →
  Wilcoxon + Benjamini-Hochberg → MALDI-MSI & lipidomics primer. Each = 1 short explainer + 1 video
  link (StatQuest etc.) + (where cheap) 1 toy demo. Explicitly teach: t-SNE/UMAP distances aren't
  quantitative; Harmony only for clustering/transfer, Wilcoxon on uMAIA non-Harmonized data.

---

## 6. The science arc and how it maps to levels (Digests 01,02,05,06,09,10)

Full transparent path (control vs pregnant), with UNROLL vs minimal-API decisions baked in:

| # | Step | Decision | Key recycle |
|---|---|---|---|
| a | Mass spectra & MS-behind-MSI intro | UNROLL (5-line `plt.vlines` spectrum) | uMAIA `Spectrum` two-array view |
| b | Pull 2 sections from METASPACE → AnnData | UNROLL `download_metaspace_dataset` | euclid `io/metaspace.py` (Digest 13 §3A) |
| c | uMAIA normalization (2-section) | API call for the SVI fit; UNROLL the CDF histogram-matching transform + before/after hist | uMAIA `_transform.py` |
| d | Peak annotation + **side-by-side LC-MS↔MSI ppm plot** | UNROLL the ppm matcher + adducts; NEW ppm plot (Digest 06 §7.3) | EUCLID `annotate_molecules`, portal `compress_lipid_name_robust`, `lipid_properties` |
| e | Feature selection | UNROLL Moran's I + combined score + dropout; skip manual KMeans curation | EUCLID `feature_selection(modality="moran"/"manual")` |
| f | NMF embedding (learned on control only) | UNROLL `compute_seeded_NMF` | `embedding.learn_seeded_nmf_embeddings` |
| g | Harmony on NMF (clustering/transfer ONLY) | minimal API + explain covariates | `harmonize_nmf_batches` |
| h | Registration → Allen CCF coords | **pre-ship coords** + concept; STalign optional (§9) | `003-DataPreparation` CCF ingest |
| i | Clustering (lipizones) + label transfer control→pregnant | Hybrid: transparent Leiden-on-Harmonized-NMF as main + euclid divisive splitter as clone-and-call concept (§ fork) | `differential_lipids` gate is the unifying thread |
| j | Wilcoxon + BH differential per cluster/region | UNROLL — this IS the course's test | `differential_lipids` (recycle verbatim) + volcano plot (boss repo) |
| k | Composite scores: **membrane remodeling** (Σ log2FC) + **myelination/sphingolipid** (z-scored HexCer/Cer/SM) | UNROLL — one-line arithmetic → publishable map | LBA 009b §7c/7d |
| l | MERFISH integration by shared CCF coords | UNROLL the cKDTree-per-region matching loop | `004-GenesVSLipids-MERFISH` cells 5-7; `assets/data_handler.py` |
| m | XGBoost→lipid-change + SHAP + GO | UNROLL feature-eng + SHAP read-out + GO; fixed XGBoost; permutation null | predictors notebooks (cleaned, Digest 08) |
| n | Publication-quality multi-panel figure | students do "good plotting" here | EUCLID plotting + LEGISLATION rules |

### 6b. The 9 sessions (one notebook each, Jul 2–9) + the Jul 1 intro

**Jul 1 — self-guided intro (~5 h, alone, no instructor):** the 3 intro notebooks (§5) + SETUP done at
home before. Goal: arrive able to run a notebook, read code, and hold the concepts. This is where we
"make sure they get super prepared to then do wonders from Thursday 2nd."

Difficulty ramps evenly across the 9; agents/Claude Code introduced from N6 onward. Each notebook is one
self-contained 4.5 h session with its own ⚠️ CHECKPOINT and a one-line biological payoff.

| NB | Session theme | Core content (UNROLL unless noted) | Level |
|----|---|---|---|
| **N1** | Mass spectra & the data | MS-behind-MSI; a spectrum is two arrays (`plt.vlines`); MALDI/lipidomics primer; pull the 2 sections from METASPACE → AnnData; explore the data object; first per-lipid spatial maps | easy |
| **N2** | From m/z peaks to lipid names | ppm + adducts + de-ionization; LC-MS/MS coupling; **the side-by-side LC-MS↔MSI ppm plot**; LIPID MAPS + user-CSV matching; lipid nomenclature + `lipid_properties`; rename features | easy |
| **N3** | Normalization with uMAIA | batch effects; the bimodal fg/bg model (conceptual); run uMAIA (minimal API) on the 2 sections; UNROLL the CDF histogram-matching transform; before/after histograms; the **2-section corner case taught honestly**; per-lipid 0–1 scaling | easy→med |
| **N4** | Feature selection & NMF | Moran's I + combined score + dropout; seeded NMF (`compute_seeded_NMF`) learned on **control only**, applied to both; what factors mean; t-SNE/UMAP for viz | med |
| **N5** | Harmony, registration & flat clustering | Harmony on NMF (concept + API, clustering/transfer ONLY); load **pre-shipped CCF coords** + ABBA concept/demo; transparent **flat Leiden** on Harmonized NMF (kNN graph + Leiden, unrolled) → lipizones as territories | med |
| **N6** | Divisive clustering & label transfer | reason through + implement a **simplified from-scratch divisive top-down splitter** (the Wilcoxon+log2FC split gate); then **clone EUCLID and call `learn_euclid_clustering`** as the production method (the git/clone hurdle); transfer labels control→pregnant. *(Claude Code introduced here.)* | med→hard |
| **N7** | Differential lipids: control vs pregnant | UNROLL `differential_lipids` (Wilcoxon + BH) per cluster/region; volcano; **composite scores**: membrane remodeling (Σ log2FC) + myelination/sphingolipid (z-scored HexCer/Cer/SM); spatial maps — the biological payoff | hard |
| **N8** | Multimodal: MERFISH ↔ lipids | integration by shared CCF coords (UNROLL the cKDTree-per-region matching loop); gene-expr-per-pixel; lipizone↔cell-type colocalization (reciprocal enrichment); intro to predicting lipids from genes | hard |
| **N9** | Interpretation + the figure | XGBoost gene→lipid-change + SHAP + GO to name the programs (fixed model, permutation null); assemble the **publication-quality multi-panel figure** (students do "good plotting"); + the open/free own-analysis template (Claude Code) | hard |

This is **3 easy / 3 medium / 3 hard**, matching "divide equally for difficulty and time." N9 doubles as
the capstone + the springboard to their own analysis. If N8+N9 prove too heavy for one session each, the
MERFISH XGBoost/SHAP/GO can split across N8/N9 with the figure pulled into a short N9 finale.

Per-cluster-vs-per-region note: with 2 sections, statistical units for differential/XGBoost are pixels
pooled per cluster or per Allen region. Teach this explicitly (no biological replicates; Wilcoxon is
across pixels within a region/cluster, p-values get tiny → effect-size gate |log2FC|>0.2 matters more).

---

## 7. Simplifications vs the paper (Digests 01,02,10) — what we deliberately drop/replace

- **Bayesian hierarchical case-control model → Wilcoxon + BH** (the paper itself uses Mann-Whitney+BH in
  places; recycle `differential_lipids`). Optionally SHOW the Bayesian model read-only as "what the
  paper did", with the bayes-vs-centroid scatter to motivate why we still run a test.
- **uMAIA**: keep `covariate_vector=None` (don't hand it the condition labels), full-res sections,
  seed=42, ~2000–5000 SVI steps on CPU (single-digit minutes for 2 sections).
- Skip (degenerate with 2 sections): xgboost feature restoration (needs ≥3 sections), ICC, 3D
  interpolation/movies, AP-continuity in the splitter.
- Cut research-grade cruft: 7-source additive annotation scoring (use EUCLID `Score`); SHAP heatmaps and
  GO-term permutation overlap (author-rejected); hyperparameter grid search (use fixed XGBoost);
  `kneed` (use top-N by |loading|).

---

## 8. The "data bundle" we must assemble and ship (CRITICAL — several files are missing)

Flagged across Digests 06,07,08,09,13. Build/ship these (gitignored in dev; included or download-scripted
in student release; pre-cache the heavy ones to avoid live downloads in class):

1. **The 2 chosen METASPACE sections** (control + pregnant) → pulled AnnData (§ fork on which pair).
2. **Annotation references.** Split into two layers, with different availability (verified 2026-06-21):
   - *Database layer* — `structures.sdf` (LIPID MAPS, 254 MB) + `HMDB_complete.csv` (10 MB) +
     `lipidclasscolors.h5ad` (1 MB): standalone in **Zenodo 15650014** (265 MB total). Pull at N2 build.
   - *LC-MS-confirmed layer* — `lipids_processed.csv` (m/z, Lipids, Score) + `lcms_females_tutorial.csv`
     (Lipid, nmol_fraction_LCMS): **NOT standalone anywhere.** Not in the repos, not in the masks tar
     (16524009), not in record 15650014. They are either user-provided (tutorial says "ideally a paired
     LCMS dataset") or bundled at the END of the 13.4 GB uMAIA-subset tarball (16521812), which leads with
     zarr — so streaming just the CSVs would cost ~all 13.4 GB on a 97%-full disk. **→ Ask Luca to send
     these two small CSVs directly** rather than pulling 13.4 GB. They are his paired LC-MS reference.
   - Tissue `mask.npy` files for control females + pregnant Brain1/2/4 are in the 50 KB masks tar
     (16524009), cached at `/tmp/masks.tar.gz` — useful for the uMAIA step.
3. **Pre-registered CCF coordinates** for the 2 sections (`xccf/yccf/zccf` per pixel) — depends on the
   ABBA/registration decision (§9). Or ship the Allen reference slices + run STalign live.
4. **MERFISH data** (Allen 500-gene parquet; optionally the 8460-gene imputed h5ad) — **Luca to send.**
5. **Allen CCF atlas artifacts** (annotation volume via allensdk, `eroded_annot.npy`,
   `allen_name_to_annots.pkl`) — heavy; pre-cache/ship.
6. **GO files** (`go-basic.obo`, `gene2go` taxid 10090) — ship locally to avoid live downloads.

---

## 9. Registration / ABBA — DECIDED: coords + ABBA concept only (no STalign hands-on)

ABBA's core (elastix/BigWarp) is Java/C++ and cannot be unrolled into transparent Python; with 2 sections
registration is a one-time setup, not the science. Plan (N5):
- **Pre-register both sections offline** (or reuse LBA `_Coords.tif`); ship per-pixel CCF coords as a
  provided file students just **load** (exactly as `003-DataPreparation` consumes it: `image.reshape(-1,3)`
  → `xccf/yccf/zccf`, `(xccf*40).astype(int)` voxel indices, then Allen region lookup).
- Teach the **concept** of CCF registration with a short (5–10 min) recorded or live ABBA demo
  ("the community-standard GUI you'd use in your own lab; here we hand you its output").
- **No STalign hands-on, no live ABBA.** Mention STalign in one line as "the scriptable Python
  alternative" for the curious. This frees session time for the lipidomics/MERFISH/ML story.
- Producing the coords is an instructor/dev task (M-data); the heavy AllenSDK/BrainGlobe atlas downloads
  are pre-cached and the region-annotated pixel table is pre-shipped.

---

## 10. METASPACE section selection (Digest 13) — FORK + dependency

Public, no login. Concrete verified-downloadable starter pair: **pregnant `Pregnant_Brain4_A1`** (id
`2024-07-23_14h41m04s`, 236 features @ FDR 0.1) + a **control `Brain2_2_*`** at matched AP (e.g. `_H1`
186 feat). BUT: there is **no bregma field** in METASPACE — AP match must be done visually (e.g. by the
myelin lipid HexCer 42:2 morphology), and **one section should sit at the MERFISH plane** (ideally the
control, per the brief), which we can't fix until we have the MERFISH section's plane. Plan: build a
**section-picker notebook** that enumerates candidates, ranks by feature-richness, and shows ion images
for visual AP matching — then lock the pair with Luca. We must also verify the 2 chosen sections have
enough tissue pixels for uMAIA subsampling.

---

## 11. Clustering — DECIDED: teach all three (the "1+3" answer), across N5–N6

Given the generous time, we teach the full spectrum so students truly understand clustering of spatial-
omics objects:
1. **N5 — transparent flat Leiden** on the Harmonized NMF (build kNN graph → Leiden, unrolled with the
   scanpy primitives explained). The accessible entry point; robust with 2 sections (AP-continuity is
   meaningless here, so we don't lean on it).
2. **N6 — from-scratch simplified divisive top-down splitter**: students reason through and implement a
   minimal recursive binary split — re-embed the subset, split, and **accept a split only if it yields
   differential lipids (the Wilcoxon + |log2FC| gate)**. This is the "reason about how spatial-omics
   objects work" goal; the same differential gate ties N6 to N7.
3. **N6 — clone-and-call the production method**: students `git clone` EUCLID and call
   `learn_euclid_clustering(...)` (and `apply_euclid_clustering` for transfer). The git/clone hurdle +
   "this is what the package does for you, now that you've built a baby version yourself."
Unifying thread across N5→N6→N7: the Wilcoxon+log2FC differential gate appears in the splitter AND as the
standalone case-control test.

---

## 12. Writing voice & plotting policy (Digests 10, LEGISLATION §8–9)

- Notebook prose in Luca's LBA voice: dense, active, every sentence triggers the next, numbers paired
  with a biological reading, no AI-isms (no em-dashes-as-pauses, no Title Case, no scare quotes except
  coined terms like *lipizone*). Template sentences captured in Digest 10 §Part 2.
- Most plotting functions provided **ready** in `src/cajal_lipidomics/plotting.py` (recycle EUCLID's
  spatial-scatter style: `ax.scatter(zccf, -yccf, c=lipid, cmap, vmin/vmax=per-section 2/98 pct)` +
  grayscale lipizone background + black boundary contour). Students do "good scientific plotting" only at
  chosen moments: the **ppm annotation plot**, the **volcano**, and the **final multi-panel figure**.
  Follow LEGISLATION figure rules (compact, no overlaps, consistent fonts, divergent maps for ±, rasterize
  scatter + vector text rcparams 42, export PDF).

---

## 13. Development methodology (PHILOSOPHY + LEGISLATION)

- **Iterate on the 2 real sections end-to-end before assigning** (Luca: "run the tutorial step by step,
  iteratively, until we're convinced we like what we see"). Especially the dirty uMAIA 2-section step.
- Test small first; set seeds; pin deps; commit aggressively with meaningful messages; verbose outputs;
  show data-object structure; absolute paths; never overwrite outputs. Backup before editing key files.
- Build a **solution notebook**, run it green, THEN auto-generate the student version, THEN dry-run the
  student version to confirm the TODO blanks are solvable in the time budget.

---

## 14. Proposed build order (milestones)

1. **M0 Scaffolding**: repo layout, both `environment-*.yml`, `make_student.py`, `src/cajal_lipidomics`
   skeleton, SETUP_GUIDE. Verify env installs clean on a fresh machine.
2. **M1 Section picker + data pull**: enumerate `mlba-2025`, propose the pair, pull both sections to
   AnnData; assemble the data bundle stubs. Lock the pair with Luca.
3. **M2 uMAIA 2-section run**: get normalization working + before/after plots on the real pair; iterate
   until it looks right. (De-risks the dirtiest step early.)
4. **M3 Intro notebooks** (00/01/02) + their toy data.
5. **M4 Level 1 solutions** (a–k), run green end-to-end on the real pair.
6. **M5 Level 2** (MERFISH — needs Luca's data + the chosen panels).
7. **M6 Level 3** (XGBoost+SHAP+GO + open template) + the final-figure assembly.
8. **M7 Generate student notebooks, dry-run, write student-repo README**, package the data bundle.

We checkpoint with Luca after each milestone.

---

## 15. Forks — RESOLVED (2026-06-21)

1. **Registration/ABBA** → ship coords + ABBA concept/demo only; no STalign hands-on, no live ABBA (§9).
2. **Clustering** → teach all three: flat Leiden + from-scratch divisive + euclid clone-and-call (§11).
3. **Section pair** → I propose a candidate pair now via a picker notebook, refine the MERFISH-adjacent
   member once Luca sends MERFISH (§10).
4. **Sessions** → 9 × 4.5 h (Jul 2–9), 1 notebook each, graded 3 easy/3 med/3 hard; + Jul 1 self-guided
   ~5 h intro (§6b).

## 17. Refinements from Luca (session 2) — fold into every notebook

- **Plots are the teaching.** The ppm plot was just one example; build many meaningful, fully
  transparent plots throughout — e.g. uMAIA before/after histograms, good-vs-bad Moran features as
  side-by-side spatial maps, etc. Don't rush; this must be highly pedagogic and beautiful.
- **Teach good scientific plotting** per the LEGISLATION figure rules (compact, no overlaps, fixed
  font set, divergent maps for signed data, rasterized scatter + vector text). `src/cajal_lipidomics/
  style.py` encodes the defaults; selected moments ask students to make publication-grade panels.
- **Use scanpy** somewhere visibly (natural fit: kNN graph + Leiden + UMAP in NB5/NB6, and AnnData QC).
- **NB1 teaches data formats**: what you get from METASPACE and how it looks, zarr, pandas DataFrames,
  parquet, AnnData. We have the time; go slowly and concretely.
- **Cluster interpretation through marker lipids** (must not be missed): in the NON-pregnant (control)
  brain, find each cluster's marker lipids with Wilcoxon, and teach the **beautiful sorted
  anatomy×lipid and anatomy×lipizone heatmaps** from the Lipid Brain Atlas — **copy the sorting code
  verbatim** (cosine optimal-leaf-ordering; `plot_olosorted_lipid_lipizone` / 001-IDCARDS). Sorting is
  an art; do not reinvent it. This lands in NB6/NB7.
- **At-home preamble**: solid `environment.yml` + `requirements-extra.txt`, the `docs/SETUP.md` guide,
  and the 3 self-guided intro notebooks. Luca's email **luca.fusarbassini@epfl.ch** is in the README.
- **Beautiful repo** with the LBA images (`assets/mosaic.png`, `lipizones.png`, `lipids.png`,
  `diagram.svg`) and all the project intro + learning objectives shared in chat (now in README.md).

### Decisions locked (session 2)
- **One repo**: `git@github.com:lamanno-epfl/lipidomics_tutorial_cajalcourse.git`, work on `main`,
  frequent commits + push. Heavy files gitignored (verified). Solutions + generated student notebooks
  both live here; split a release-only copy later if wanted.
- **Data realities**: `maindata_2.parquet` (8.6 GB, rosetta stone via `Path`) and
  `avemerfish_imputed_named.parquet` (55 MB, region×gene) landed locally; `C57BL6J...h5ad` (47 GB,
  per-cell, no coords) moved to `/mnt/data/cajal_lipidomics/` + symlinked back (not needed for the
  region-level path). MERFISH **coords** incoming from Luca. LC-MS CSVs to be range-extracted from the
  LBA `csv.zip` (Zenodo 15379565).
- **Env conflict resolved by isolation**: student/main env = EUCLID stack (jax 0.4.35), **no uMAIA**;
  uMAIA (jax 0.4.14 / numpyro 0.12.1) runs once in a dev `cajal-umaia` env. Students receive the
  uMAIA-normalized data + fitted params and unroll the histogram-matching transform in numpy/scipy.

**Still needed from Luca (data, currently missing from the material):**
- the **LC-MS annotation reference CSV(s)** + LIPID MAPS subset (for N2 annotation + ppm plot), and
- the **MERFISH data** (for N8–N9). Both block specific notebooks but not the scaffolding/N1/N3–N7 build.

## 16. No-regret next steps (ready to start on go-ahead)

These depend on nothing outstanding and de-risk the hardest bits early:
- **M0** repo scaffolding + both pinned CPU envs + `make_student.py` + `src/cajal_lipidomics` skeleton.
- **M1** section-picker notebook → propose the control+pregnant pair; pull both to AnnData.
- **M2** get the uMAIA 2-section normalization working + before/after plots on the real pair (the dirtiest
  step — prove it early).
Then N1 → N9 in order, checkpointing with Luca after each, generating + dry-running the student version.
