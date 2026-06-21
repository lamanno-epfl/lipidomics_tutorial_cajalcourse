# Data manifest

Heavy data is never committed (see `.gitignore`). This file records what each input is, where it
comes from, and how to obtain it. Scripts that fetch/build these live in `../scripts/`.

## Provided by the instructor (local, large)

| file | size | what it is | used in |
|---|---|---|---|
| `maindata_2.parquet` | 8.6 GB | full LBA lipidome: 7.6 M pixels × (172 lipids + metadata). Has `Path` (matches METASPACE section names), `Condition`, `Sample`, CCF coords (`xccf/yccf/zccf`, `*_index`), Allen `acronym`/`allencolor`/`division`, and the lipizone hierarchy (`level_1..11`, `lipizone_names`, `subclass`). The rosetta stone: picks sections, lifts CCF coords, drives region transfer. | section selection, NB4 |
| `avemerfish_imputed_named.parquet` | 55 MB | region-averaged imputed gene expression, 670 Allen regions × 8460 genes (index = region acronym, columns = gene symbol). Predictor matrix for the genes → lipid-change models. | NB8 |
| `C57BL6J-638850-imputed-log2.h5ad` | 47 GB | per-cell Allen MERFISH imputed transcriptome (4.3 M cells × 8460 genes), no coordinates. Moved to `/mnt/data/cajal_lipidomics/` and symlinked back. Not needed for the region-level path; kept for an optional per-cell demo. | optional |

## Fetched by scripts (see `../scripts/`)

| file | source | fetch |
|---|---|---|
| `structures.sdf`, `HMDB_complete.csv`, `lipidclasscolors.h5ad` | Zenodo 15650014 (265 MB) — LIPID MAPS + HMDB reference databases | `scripts/fetch_references.py` |
| `lipids_processed.csv` (m/z, Lipids, Score), `lcms_females_tutorial.csv` (Lipid, nmol_fraction_LCMS) | LBA `csv.zip` (Zenodo 15379565) — extracted by HTTP range request, no full download | `scripts/fetch_references.py` |
| tissue masks (`mask.npy` per section) | Zenodo 16524009 (50 KB) | `scripts/fetch_references.py` |
| Allen CCFv3 annotation volume, `eroded_annot.npy`, region lookup | allensdk / brainglobe (cached on first use) | used in NB4 |
| `go-basic.obo`, `gene2go` (mouse, taxid 10090) | goatools downloads | used in NB8 |

## Built during the course / dev

- The two chosen MALDI sections (one control female, one pregnant), pulled from METASPACE
  (`scripts/pull_metaspace.py`), normalized with uMAIA, with CCF coords lifted from `maindata_2`.
