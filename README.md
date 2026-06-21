<h1 align="center">A spatial metabolomics primer</h1>
<p align="center"><b>CAJAL course — Neuromics: Single-cell and Spatial omics in the Nervous System</b><br>
Bordeaux, July 2026 · Instructor: Luca Fusar Bassini (EPFL)</p>

<p align="center"><img src="assets/mosaic.png" width="720" alt="Lipid Brain Atlas mosaic"></p>

Spatial metabolomics reads the chemistry of tissue one pixel at a time. In this project you
work with MALDI mass spectrometry imaging (MALDI-MSI) of the mouse brain and compare a coronal
section from a control female with one from a pregnant female, learning to turn a grid of mass
spectra into biological insight. You will walk the whole path of a spatial-omics analysis, and
you will write most of it yourself, line by line, so that nothing stays a black box.

The science follows the Lipid Brain Atlas, simplified and slowed down for learning. Every step
is transparent: where a published package gives you a one-liner, we first unroll what it does in
plain Python, then show you the shortcut. You will see the data at every step, through plots that
are built to be read.

## What you will be able to do

- Read MALDI-MSI data and the formats it travels in (METASPACE ion images, zarr, parquet, AnnData).
- Annotate mass peaks to putative lipids using ppm matching and a paired LC-MS reference.
- Normalize across sections with uMAIA and understand exactly what the correction does.
- Register sections to the Allen Brain Atlas and transfer anatomical regions to every pixel.
- Reduce dimensionality with NMF, integrate with Harmony, and cluster pixels into lipid territories.
- Transfer those territories from the control brain onto the pregnant brain.
- Find the lipids that change in pregnancy with a Wilcoxon test and Benjamini-Hochberg correction,
  and design composite scores that capture myelination and membrane remodeling.
- Ask which gene programs predict the lipid changes, using MERFISH transcriptomics with XGBoost,
  SHAP, and gene ontology.
- Assemble a publication-quality, multi-panel summary figure.

## The path (one notebook per session)

Nine hands-on sessions, graded from gentle to demanding, plus a self-guided intro you do before
we start. The full breakdown lives in [`PLAN.md`](PLAN.md).

| | Notebook | Theme |
|---|---|---|
| intro | `00_intro/` | tooling, scientific Python, and the ideas behind the methods (self-guided) |
| 1 | mass spectra & the data | what MALDI-MSI is, the data formats, first spatial maps |
| 2 | peaks to lipid names | ppm, adducts, LC-MS coupling, annotation |
| 3 | normalization with uMAIA | batch effects and what the correction does |
| 4 | anatomy: registration & the Allen atlas | CCF coordinates, region transfer |
| 5 | molecular variability | feature selection, NMF, t-SNE/UMAP, Harmony |
| 6 | clustering & label transfer | Leiden, lipid territories, control → pregnant |
| 7 | clustering from scratch & the pregnancy changes | a divisive splitter, Wilcoxon+BH, composite scores |
| 8 | which genes explain the changes | MERFISH, XGBoost, SHAP, gene ontology, the figure |
| 9 | your own analysis | open, creative, AI-assisted |

## Before the course

1. Follow [`docs/SETUP.md`](docs/SETUP.md) to install everything (Mac, Windows, or Linux). Do this
   at home, in good time, not on the first day.
2. Work through the three notebooks in [`notebooks/00_intro/`](notebooks/00_intro). They take a few
   hours and assume no prior programming. They are how you arrive ready to do real work with me.

## Data and tools

- Sections are pulled from the public Lipid Brain Atlas project on
  [METASPACE](https://metaspace2020.org/project/mlba-2025?tab=datasets).
- Normalization uses [uMAIA](https://github.com/lamanno-epfl/uMAIA).
- The analysis uses [EUCLID](https://github.com/lamanno-epfl/EUCLID), the package behind the atlas;
  you clone it and call it sparingly, after building the same ideas by hand.
- Anatomy comes from the Allen Mouse Brain Common Coordinate Framework (CCFv3).

## References

- Fusar Bassini et al., *The lipidomic architecture of the mouse brain*, bioRxiv 2025.10.13.682018 —
  <https://www.biorxiv.org/content/10.1101/2025.10.13.682018v1>
- uMAIA, *Nature Methods* (2025), s41592-025-02771-7.
- Explore the atlas: <https://lbae-v2.epfl.ch/>

## Getting help

Questions before or during the course: **luca.fusarbassini@epfl.ch**. During sessions you also have
Claude Code as a pair-programmer, used the right way: read what it writes, and if you do not
understand a line, stop and look it up.

## Repository layout

```
notebooks/00_intro/   self-guided preparation (tooling, python, concepts)
notebooks/level1..3/  the nine session notebooks (solution + student versions)
src/cajal_lipidomics/ ready-made style, plotting, IO and data helpers
scripts/              data fetching, section selection, student-notebook generation
docs/SETUP.md         cross-platform setup guide
assets/               figures
```

<p align="center"><img src="assets/lipizones.png" width="640" alt="Lipizones"></p>
