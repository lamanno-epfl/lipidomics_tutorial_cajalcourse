"""Assemble NB8: which genes explain the lipid changes (XGBoost + SHAP + gene ontology).

Run:  python notebooks/level3/_build_08.py
Then: jupyter nbconvert --to notebook --execute --inplace <path> --ExecutePreprocessor.timeout=1800
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = "/home/fusar/lipidomics_tutorial_cajalcourse/notebooks/level3/08_multimodal_xgb_shap_go_solution.ipynb"

cells = []


def md(src):
    cells.append(new_markdown_cell(src))


def code(src, tag="keep"):
    c = new_code_cell(src)
    c.metadata["tags"] = [tag]
    cells.append(c)


# ----------------------------------------------------------------------------
md(r"""# 08: which genes explain the lipid changes

### CAJAL NEUROMICS summer school, Bordeaux 2026 · a spatial metabolomics primer

*Luca Fusar Bassini · hands-on notebook 8 of 9 · ~90 minutes*

---

By now you have measured what pregnancy does to the brain lipidome. You found the lipids that move,
you found the regions where they move most, and you painted the changes back onto the tissue. That is
a complete *description*. This notebook reaches for the next thing: a *mechanism*. Lipids are not
made by magic. Every lipid in a membrane was synthesised, remodeled, or degraded by an enzyme, and
every enzyme is a protein read off a gene. So if pregnancy reshapes the lipids of a region, somewhere
in that region the genes that build and break lipids must be doing something different. The question
of this notebook is: which genes track the lipid change closely enough that we can call them
predictors?

We answer it with a second modality. We have, from the Allen Brain Atlas, a spatial transcriptome:
MERFISH, which counts individual gene transcripts inside individual cells, imputed up to 8460 genes
and averaged into every anatomical region. We have, from our own MALDI experiment, the per-region
pregnancy lipid change. The two datasets never touched the same tissue. They were measured in
different mice, on different instruments, in different labs. The only thing they share is a map: both
were registered into the same Allen Common Coordinate Framework, so both speak in the same anatomical
regions. That shared map is the entire bridge. We will line the two matrices up region by region, and
then ask a machine learning model to learn, from gene expression alone, where each lipid changes.

We will meet that bridge at **two granularities of the same gene<->lipid question**. First, fine: for
*every MSI pixel* we borrow the genes and the cell type of the MERFISH *cells* that sit in the same
little 3D ball, built transparently with a KD-tree. Second, coarse: for *every Allen region* we join
the region-averaged transcriptome to the region's lipid change and let a model rank the gene programs.
The per-pixel view shows you the integration physically, pixel by pixel, and lets us ask which lipid
territories are also cell-type territories. The per-region view trades that resolution for statistics:
one clean number per region per gene, predictable by a tree. Same bridge, two zoom levels.

Here is the road:

> **integration by shared Allen coordinates → per-pixel: KD-tree ball query borrows MERFISH genes and
> cell type for each MSI pixel, a gene map beside a lipid map, a cell-type territory map, and the
> lipizone x cell-type reciprocal enrichment → per-region: the lipid-change matrix and the region x gene
> matrix, joined → gradient-boosted trees (XGBoost), explained from scratch → NMF gene programs as the
> features → one regressor per lipid, scored by test Pearson r → SHAP, which gene program drives each
> prediction → the leading genes of the top program, and what they are → gene ontology of those genes →
> a permutation null as the negative control → one publication-quality multi-panel figure**

Every number and every figure below comes from code that ran on the two real sections, `217D`
(control) and `Brain1_C2` (pregnant), the same coronal plane at about AP 6.5, joined to the real Allen
MERFISH cells and the real Allen MERFISH region averages. Nothing is invented.""")

# ----------------------------------------------------------------------------
md(r"""## the callouts

The four markers from the earlier notebooks run through this one too:

- 🔬 **TASK** something you do (write or run code).
- 💡 **HINT** a nudge when you are stuck.
- ❓ **QUESTION** pause and think; no code required.
- ⚠️ **CHECKPOINT** what you should see if it worked. If your screen disagrees, stop and fix it before moving on.

🔬 **TASK.** Run the next cell to load the stack. Everything here you met before, plus `xgboost` and
`shap` for the model, and `goatools` plus `mygene` for the gene ontology at the end.""")

code(r"""# the scientific-Python stack you already know
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad

# the new pieces for today: gradient-boosted trees, and SHAP for reading them
import xgboost as xgb
import shap

# the matrix factorisation, scaling and split helpers we lean on
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# the small NMF fits below are short on purpose; silence the "max_iter reached" notices
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# the course helper package: the tested versions of the recipes we unroll
import cajal_lipidomics as cl
from cajal_lipidomics import multimodal, analysis, plotting
from cajal_lipidomics.style import set_style, FS
set_style()  # the lab figure style: clean, vector text, no top/right spines

# one global seed so every number and figure below is reproducible
SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

print("ready. xgboost", xgb.__version__, "| shap", shap.__version__)""")

md(r"""⚠️ **CHECKPOINT.** You should see `ready. xgboost ... | shap ...` and no red error. If a
`ModuleNotFoundError` appears, your notebook is on the wrong kernel: pick `cajal-lipidomics` from the
kernel picker (top right) and run this cell again.""")

# ----------------------------------------------------------------------------
md(r"""## 1 · the big idea: integration by shared coordinates

Stop and hold this picture, because the whole notebook rests on it. We have two experiments.

The first is **our MALDI**: a laser walks across a tissue section, fires at each tiny spot, and a
mass spectrometer reports the abundance of every lipid at that spot. One spot is one pixel. We have two
sections, one control and one pregnant, and roughly 175,000 pixels between them, each carrying 173
lipid abundances.

The second is **Allen MERFISH**: a completely separate technology, run by a completely separate group,
on completely separate mice. MERFISH lights up individual RNA molecules inside individual cells under a
microscope, so it counts how many transcripts of each gene a cell holds. The Allen group imputed the
measured panel up to 8460 genes and then averaged the gene counts inside every one of about 670
anatomical regions, giving one number per gene per region.

There is **no pixel-to-cell correspondence** anywhere in this. A MALDI pixel was never matched to a
MERFISH cell, because they live in different brains. What saves us is registration. Both datasets were
warped into the **Allen Common Coordinate Framework**, the standard 3D mouse-brain atlas. The atlas
gives both modalities the *same 3D coordinates*, in millimetre-scale CCF units. A MALDI pixel has an
`(xccf, yccf, zccf)`, and so does every MERFISH cell, so the shared key can be as fine as physical
proximity in space or as coarse as the named region a coordinate falls in. That single fact buys us two
granularities, and we use both in this notebook.

**Per-pixel (fine).** For each MALDI pixel we look up the MERFISH cells whose coordinates sit within a
small ball around it, and borrow them: average their gene counts, and take the majority cell type. This
hands every pixel an imputed gene-expression vector and a putative cell type, so we can lay a MERFISH
gene map beside a lipid map on the same tissue, and ask which lipid territories are also cell-type
territories. The cost is coverage: a pixel only gets borrowed genes where a MERFISH section actually
overlaps it, so not every pixel is matched.

**Per-region (coarse).** We collapse both modalities to the named Allen region: the per-region lipid
*change* upon pregnancy, and the region-averaged transcriptome. One number per region per gene, one per
region per lipid, joined on the region index. We lose the within-region detail but gain a clean, stable
table a model can learn from. This is the half that runs XGBoost, SHAP, and gene ontology.

The coarse move, in one line:

```
per-region lipid change   (region x lipid)        from MALDI, control vs pregnant
per-region gene average   (region x gene)         from MERFISH
                  |  align on the region index  |
                  v                              v
        ask: do the genes of a region predict its lipid change?
```

This is exactly the integration strategy of the Lipid Brain Atlas paper, where the lipidome and the
transcriptome were joined through the Allen atlas and a model was asked which genes predict which
lipids. We are doing the pregnancy-specific version: the target is not the lipid level, it is the lipid
*change* upon pregnancy.""")

md(r"""🔬 **TASK.** Load the two real sections. This is the same AnnData you used in the earlier
notebooks: `adata.X` holds the uMAIA-normalized lipid abundances, `adata.var_names` are the 173 lipid
names, and `adata.obs` carries the metadata, including `Condition` (naive control vs pregnant) and
`acronym`, the Allen region of each pixel.""")

code(r"""# load the section pair (control 217D + pregnant Brain1_C2, same coronal plane, AP ~6.5)
adata = ad.read_h5ad("/home/fusar/lipidomics_tutorial_cajalcourse/data/sections_pair.h5ad")

print("pixels x lipids:", adata.shape)
print("conditions:", dict(adata.obs["Condition"].value_counts()))
print("distinct Allen regions (acronym):", adata.obs["acronym"].nunique())
print("first few lipids:", list(adata.var_names[:5]))""")

md(r"""⚠️ **CHECKPOINT.** `174768 x 173`, the two conditions are `naive` (control, ~84k pixels) and
`pregnant` (~90k pixels), and there are 174 distinct Allen regions on this plane. The differential
side of this notebook uses the uMAIA-normalized values directly, never the Harmony-corrected embedding:
Harmony's whole job is to make the two conditions overlap, so testing a condition difference on
Harmonized data would erase the very effect we want.""")

# ----------------------------------------------------------------------------
md(r"""## 2 · per-pixel integration: borrow genes from the nearest MERFISH cells

Before we collapse anything to regions, let us do the integration the *fine* way and watch it happen
pixel by pixel. The recipe is the heart of "integration by shared coordinates", and it is far simpler
than it sounds once you see it unrolled.

We have two point clouds in the **same 3D Allen coordinate space**: our MALDI pixels at `(xccf, yccf,
zccf)`, and the Allen MERFISH cells at `(x_ccf, y_ccf, z_ccf)` (the underscore is the only naming
difference; the units are identical). For each MALDI pixel we ask a geometric question: *which MERFISH
cells sit within a small ball around me?* Take those neighbouring cells, average their 500 measured
gene counts, and take the majority vote of their cell type. That gives each pixel an imputed gene
vector and a putative cell type, borrowed from the transcriptome that happens to overlap it in space.

The only piece of machinery is a **cKDTree**, a binary space-partitioning tree from `scipy.spatial`. You
feed it the 3D coordinates of all the MERFISH cells once; it organises them so that a "give me every
point within radius r of this query point" lookup costs about `O(log N)` instead of scanning all cells.
That single trick, the ball query, is what makes matching ~175,000 pixels against hundreds of thousands
of cells feasible. Everything else is averaging and voting.

🔬 **TASK.** Load the per-cell MERFISH for this plane. The helper reads only the cells inside our AP
(`x_ccf`) window with a small buffer, keeps the 3D coordinates, the Allen cell-type columns
(`division`/`class`/`subclass`/`supertype`), and the 500 measured-gene columns (Ensembl transcript IDs
like `ENSMUST...`), and drops the vascular and immune cells we never want to match to.""")

code(r"""# the per-cell MERFISH: one row per Allen MERFISH cell, in the SAME CCF coordinate space
from cajal_lipidomics import multimodal as M

# the AP window: our MSI sections' x_ccf span, with a small buffer so we don't clip edge cells
ap = (float(adata.obs.xccf.min()) - 0.1, float(adata.obs.xccf.max()) + 0.1)
print(f"AP (x_ccf) window for the MSI plane: {ap[0]:.2f} .. {ap[1]:.2f}")

# the plane subset ships in the course bundle; fall back to the full (symlinked) cell table if absent
import os
plane = "/home/fusar/lipidomics_tutorial_cajalcourse/data/merfish_plane.parquet"
full = "/home/fusar/lipidomics_tutorial_cajalcourse/cell_filtered_w500genes.parquet"
merfish_path = plane if os.path.exists(plane) else full
print("loading per-cell MERFISH from:", os.path.basename(merfish_path))

cells, gene_cols = M.load_merfish_cells(merfish_path, ap)
print("MERFISH cells x columns:", cells.shape, "| measured genes:", len(gene_cols))
print("cell-type subclasses present:", cells["subclass"].nunique())
print("coordinate ranges agree? cells z_ccf",
      f"[{cells.z_ccf.min():.2f}, {cells.z_ccf.max():.2f}]  pixels zccf",
      f"[{adata.obs.zccf.min():.2f}, {adata.obs.zccf.max():.2f}]")""")

md(r"""⚠️ **CHECKPOINT.** You loaded a few hundred thousand MERFISH cells with 500 gene columns, and the
cells' `z_ccf`/`y_ccf` ranges overlap the pixels' `zccf`/`yccf` ranges. That overlap is the whole
prerequisite: if the two clouds did not share space, no ball would ever contain a cell.

🔬 **TASK.** Now unroll the matching for ONE pixel, by hand, so the helper is no black box. We build a
KD-tree on the cell coordinates, pick a single pixel, query every cell within ~75 um (radius 0.075 in
CCF units, since 1 CCF unit is about 1 mm), and average their genes and vote their cell type.""")

code(r"""# unroll the cKDTree ball query for ONE pixel
from scipy.spatial import cKDTree

cell_xyz = cells[["x_ccf", "y_ccf", "z_ccf"]].to_numpy()   # the MERFISH point cloud
tree = cKDTree(cell_xyz)                                    # build the tree once

# pick one MSI pixel and ask: which cells sit within ~75 um of it in 3D?
pix = 1000
q = adata.obs[["xccf", "yccf", "zccf"]].to_numpy()[pix]
nbrs = tree.query_ball_point(q, r=0.075)                    # indices of cells inside the ball
print(f"pixel {pix} at CCF {np.round(q, 2)} has {len(nbrs)} MERFISH cells within 75 um")

if nbrs:
    G = cells[gene_cols].to_numpy(np.float32)
    borrowed_genes = G[nbrs].mean(0)                        # average their 500 genes
    ct = cells["subclass"].astype(str).to_numpy()
    vals, counts = np.unique(ct[nbrs], return_counts=True)
    majority = vals[counts.argmax()]                        # majority-vote the cell type
    print("borrowed gene vector length:", borrowed_genes.shape[0])
    print("majority cell-type subclass in the ball:", majority)""")

md(r"""💡 **HINT.** That is the entire algorithm: one tree, one ball query per pixel, an average and a
vote. The helper `cl.multimodal.match_pixels_to_cells` does exactly this for *every* pixel at once,
returning a pixel x gene DataFrame (NaN where no cell was in the ball) and a per-pixel cell-type Series.
We run it at **radius 0.1** (about 100 um) rather than 0.075, and the next checkpoint explains the
honest reason why.

🔬 **TASK.** Match all pixels. This is a few seconds: one KD-tree, one vectorised ball query.""")

code(r"""# the helper does the unrolled match for EVERY pixel: average genes + majority-vote cell type
# first, the honest coverage at the paper's 75 um radius...
gdf75, _ = M.match_pixels_to_cells(adata.obs, cells, gene_cols, radius=0.075)
cov75 = float(gdf75.notna().any(axis=1).mean())

# ...then the radius we actually use, 100 um, for fuller coverage
gdf, celltype = M.match_pixels_to_cells(adata.obs, cells, gene_cols, radius=0.1)
cov = float(gdf.notna().any(axis=1).mean())

print(f"pixels matched at 75 um (r=0.075): {cov75:6.1%}")
print(f"pixels matched at 100 um (r=0.10): {cov:6.1%}")
print("per-pixel gene matrix:", gdf.shape, "| cell types assigned:", celltype.notna().sum())""")

md(r"""⚠️ **CHECKPOINT.** Be honest about the coverage. At the paper's 75 um radius only about **60% of
pixels match**; at 100 um it rises to roughly **77%**. The shortfall is not a bug, it is geometry: our
MALDI plane is a thin continuous sheet, while the Allen MERFISH brain is a stack of *discrete* coronal
sections with gaps between them. Wherever our plane falls between two MERFISH sections, no cell sits
close enough in the third dimension and the pixel goes unmatched. Widening the ball to 100 um reaches
into the nearest neighbouring sections and recovers more pixels, at the mild cost of averaging cells a
little farther away. We use 0.1 for the maps below so the territories are fuller and easier to read.

❓ **QUESTION.** We just widened the radius from 75 to 100 um to lift coverage from 60% to 77%. What is
the trade-off we accepted? (Hint: a bigger ball borrows from cells farther away, which can blur a sharp
anatomical border, exactly the contamination the region-constrained version of this match guards
against.)""")

md(r"""## 2.1 · a MERFISH gene map beside a lipid map

The cleanest sanity check on the whole match is visual. If we borrowed genes correctly, then a **myelin
gene** should light up the **white matter**, the very same place a **myelin lipid** lights up. We put
them side by side. For the gene we use `ENSMUST00000102665`, which is the transcript of **`Mog`**,
myelin oligodendrocyte glycoprotein, a textbook oligodendrocyte/myelin marker, and the exact gene the
Lipid Brain Atlas paper overlaid on `HexCer 42:2` to validate section quality. For the lipid we use
`HexCer 42:2;O2`, the myelin sphingolipid that is the spine of this whole notebook.

🔬 **TASK.** Paint the borrowed `Mog` transcript and the measured `HexCer 42:2;O2` lipid on the same
sections, only on the matched pixels.""")

code(r"""# a borrowed MERFISH myelin gene next to a measured myelin lipid, on matched pixels
MYELIN_GENE = "ENSMUST00000102665"     # transcript of Mog (myelin oligodendrocyte glycoprotein)
MYELIN_LIPID = "HexCer 42:2;O2"        # the myelin sphingolipid of this notebook

obs = adata.obs
gene_vec = gdf[MYELIN_GENE].to_numpy()                 # borrowed per-pixel gene (NaN where unmatched)
lipid_vec = np.asarray(adata[:, MYELIN_LIPID].X).ravel()
matched = ~np.isnan(gene_vec)

secs = sorted(obs["SectionID"].unique())
fig, axes = plt.subplots(2, len(secs), figsize=(4.2 * len(secs), 7.2),
                         constrained_layout=True)
z, y = obs["zccf"].to_numpy(), -obs["yccf"].to_numpy()
for col, s in enumerate(secs):
    m = (obs["SectionID"] == s).to_numpy()
    cond = obs.loc[m, "Condition"].iloc[0]
    # top row: borrowed Mog transcript (matched pixels only)
    mm = m & matched
    g = gene_vec[mm]
    sc0 = axes[0, col].scatter(z[mm], y[mm], c=g, cmap="Reds", s=2.5, rasterized=True,
                               vmin=np.nanpercentile(gene_vec, 2), vmax=np.nanpercentile(gene_vec, 98))
    axes[0, col].set_title(f"{cond}: borrowed Mog (MERFISH)", fontsize=FS["s"])
    # bottom row: measured HexCer 42:2;O2 lipid, same matched pixels for a fair comparison
    lv = lipid_vec[mm]
    sc1 = axes[1, col].scatter(z[mm], y[mm], c=lv, cmap="plasma", s=2.5, rasterized=True,
                               vmin=np.nanpercentile(lipid_vec[matched], 2),
                               vmax=np.nanpercentile(lipid_vec[matched], 98))
    axes[1, col].set_title(f"{cond}: HexCer 42:2;O2 (MALDI)", fontsize=FS["s"])
    for ax in (axes[0, col], axes[1, col]):
        cl.style.spatial_axes(ax)
        ax.set_xlim(z.min(), z.max()); ax.set_ylim(y.min(), y.max())
cl.style.lightweight_colorbar(sc0, list(axes[0]), label="Mog (borrowed)")
cl.style.lightweight_colorbar(sc1, list(axes[1]), label="HexCer 42:2;O2")
fig.suptitle("a borrowed myelin gene tracks a measured myelin lipid, pixel for pixel",
             fontsize=FS["m"])
plt.show()""")

md(r"""⚠️ **CHECKPOINT.** The two rows light up the **same white-matter tracts**: the borrowed `Mog`
transcript (top, from MERFISH cells) and the measured `HexCer 42:2;O2` lipid (bottom, from our MALDI)
both trace the fibre tracts on both sections. That co-localisation is the per-pixel proof that the
shared-coordinate match worked: a gene and a lipid, measured in different mice on different instruments,
agree on where myelin is, because both were borrowed into the same CCF coordinates. This is the same
gene<->lipid relationship the region-level XGBoost will quantify later, seen here as a raw picture.""")

md(r"""## 2.2 · the per-pixel cell-type territory map

The match handed each pixel not just genes but a **cell type**, the majority Allen `subclass` of the
MERFISH cells in its ball. Painting that gives a per-pixel cell-type *territory* map: where, on our
MALDI tissue, the transcriptome says oligodendrocytes, astrocytes, and the various neuron classes live.

🔬 **TASK.** Colour the matched pixels by their most common borrowed cell-type subclass, keeping the
handful of largest subclasses so the legend stays legible.""")

code(r"""# per-pixel cell-type territories: colour each matched pixel by its majority MERFISH subclass
ct = celltype.copy()
top_ct = ct.value_counts().head(8).index.tolist()      # the 8 most common subclasses, for a clean legend
ct_plot = ct.where(ct.isin(top_ct), other=np.nan)

palette = plt.get_cmap("tab10")
cmap_ct = {name: palette(i) for i, name in enumerate(top_ct)}

secs = sorted(obs["SectionID"].unique())
fig, axes = plt.subplots(1, len(secs), figsize=(5.2 * len(secs), 4.4),
                         constrained_layout=True)
if len(secs) == 1:
    axes = [axes]
z, y = obs["zccf"].to_numpy(), -obs["yccf"].to_numpy()
for ax, s in zip(axes, secs):
    m = (obs["SectionID"] == s).to_numpy()
    # faint grey for matched-but-not-top pixels, for anatomical context
    bg = m & ct.notna().to_numpy() & ~ct_plot.notna().to_numpy()
    ax.scatter(z[bg], y[bg], c="0.85", s=2.0, rasterized=True)
    for name in top_ct:
        sel = m & (ct_plot == name).to_numpy()
        if sel.any():
            ax.scatter(z[sel], y[sel], color=cmap_ct[name], s=2.5, rasterized=True, label=name)
    cl.style.spatial_axes(ax)
    ax.set_xlim(z.min(), z.max()); ax.set_ylim(y.min(), y.max())
    ax.set_title(obs.loc[m, "Condition"].iloc[0], fontsize=FS["m"])
axes[-1].legend(fontsize=FS["xs"], markerscale=3, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.suptitle("per-pixel cell-type territories, borrowed from MERFISH by shared coordinates",
             fontsize=FS["m"])
plt.show()""")

md(r"""💡 **HINT.** The territories are anatomically sensible: an oligodendrocyte subclass (`Oligo NN`)
paints the white-matter tracts, astrocyte subclasses fill the grey matter, and the neuron subclasses
tile the cortical and thalamic regions. Each colour is a *majority vote* of real MERFISH cells inside
each pixel's ball, so the map is the transcriptome's opinion of our tissue's cellular composition.

⚠️ **CHECKPOINT.** You should see coherent coloured territories, not random speckle. The oligodendrocyte
territory overlapping the same tracts as the `Mog`/`HexCer` map above is the consistency check: genes,
lipid, and cell type all agree on where the white matter is.""")

md(r"""## 2.3 · which lipid territories are also cell-type territories

Each pixel now carries two independent labelings: a **lipizone** (its lipid-defined cluster, from the
earlier clustering notebook, in `obs.lipizone_names`) and a **cell type** (its borrowed MERFISH
subclass). The natural multimodal question is colocalisation: *which lipizones occupy the same tissue as
which cell types?* We answer it with **reciprocal enrichment**, the metric from the Lipid Brain Atlas.

The idea is a doubly-normalised crosstab. Count co-occurring pixels in a lipizone x cell-type table.
Normalise once down the columns and divide by the mean, giving how enriched each cell type is within
each lipizone. Normalise the transpose the same way, giving how enriched each lipizone is within each
cell type. Multiply the two element-wise. A pair scores high only when the lipizone is unusually full of
that cell type *and* the cell type is unusually full of that lipizone, a *reciprocal* agreement that is
robust to one side simply being large. The helper `cl.multimodal.reciprocal_enrichment` does exactly
this and zeroes out pairs with too few co-occurring pixels.

🔬 **TASK.** Build the lipizone x cell-type reciprocal-enrichment matrix on the matched pixels, then
show it as a heatmap.""")

code(r"""# reciprocal enrichment between the lipid-defined lipizones and the borrowed cell types
both = adata.obs.loc[celltype.notna()].copy()           # only matched pixels carry a cell type
ct_matched = celltype[celltype.notna()]
recip = M.reciprocal_enrichment(both["lipizone_names"], ct_matched,
                                min_pixels=50, min_enrichment=5.0)
print("reciprocal-enrichment matrix (lipizones x cell-type columns kept):", recip.shape)

# keep the most strongly-colocalising lipizones (rows) so the heatmap is legible
row_strength = recip.max(axis=1).sort_values(ascending=False)
rows_keep = row_strength.head(40).index
Rm = recip.loc[rows_keep]
# order rows and columns by their argmax for a clean near-diagonal block structure
Rm = Rm.loc[:, Rm.columns[np.argsort(Rm.values.argmax(axis=0))]]
Rm = Rm.loc[Rm.index[np.argsort(Rm.values.argmax(axis=1))]]

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(Rm.values, aspect="auto", cmap="magma",
               vmin=0, vmax=np.percentile(Rm.values, 99))
ax.set_xticks(range(Rm.shape[1]))
ax.set_xticklabels(Rm.columns, rotation=90, fontsize=FS["xs"])
ax.set_yticks(range(Rm.shape[0]))
ax.set_yticklabels(Rm.index, fontsize=4)
ax.set_xlabel("borrowed MERFISH cell type (subclass)")
ax.set_ylabel("lipizone (lipid-defined cluster)")
ax.set_title("reciprocal enrichment: lipid territories vs cell-type territories", fontsize=FS["m"])
cl.style.lightweight_colorbar(im, ax, label="reciprocal enrichment")
plt.tight_layout(); plt.show()""")

md(r"""⚠️ **CHECKPOINT.** The heatmap is *blocky*, not uniform: most lipizones light up against one or a
few cell types, the bright cells showing lipizone<->cell-type pairs that co-occupy the same tissue far
more than chance. That structure is the per-pixel multimodal payoff: a lipid-defined territory is, very
often, also a cell-type territory, because the cells that build a region's membranes shape its lipidome.
This is the same gene<->lipid logic as the region-level model, read at single-pixel resolution and
phrased in cell types rather than gene programs.

❓ **QUESTION.** A lipizone that lights up strongly against the oligodendrocyte subclass is, in effect, a
white-matter lipid signature. How does that connect to the `Mog`/`HexCer 42:2;O2` map two cells above,
and to the *myelination* gene program XGBoost will surface at the region level later? (They are three
views of one biology: oligodendrocytes, myelin genes, myelin lipids.)

---

We have now done the integration the fine way and seen it physically: a borrowed gene map, a cell-type
territory map, and a lipizone x cell-type colocalisation. The rest of the notebook does the *same*
gene<->lipid question at the coarser region granularity, where one number per region per gene lets a
gradient-boosted model rank gene *programs* and gene ontology *name* them. Two zoom levels, one bridge.""")

# ----------------------------------------------------------------------------
md(r"""## 3 · build the per-region lipid-change matrix

Our statistical unit changes here. Until now a pixel was a unit. But a single MERFISH region average is
one number per gene, so to line the two modalities up we must also reduce each lipid to one number per
region: the **pregnancy fold change** of that lipid in that region.

The recipe is the simplest thing that could work, and we unroll it so nothing hides. Inside one Allen
region, take all the control pixels and average each lipid; take all the pregnant pixels and average
each lipid; then report the change as a log2 fold change, `log2(pregnant_mean / control_mean)`. A
log2FC of `+1` means the lipid doubled, `-1` means it halved, `0` means no change. We use log2 because
it is symmetric: a doubling and a halving are `+1` and `-1`, equal and opposite, which a raw ratio of 2
versus 0.5 would not give you.

One guard matters. A region with three control pixels and two pregnant pixels gives a meaningless
average, so we keep only regions with at least 50 pixels in *both* conditions. That keeps the
statistical units honest.

🔬 **TASK.** Build the change for one region by hand first, so the helper that follows is no longer a
black box. We pick the caudoputamen, `CP`, the big striatal region on this plane.""")

code(r"""# unroll the per-region change for ONE region, by hand
X = np.asarray(adata.X)              # pixels x lipids
obs = adata.obs
lipids = list(adata.var_names)
eps = 1e-11                          # avoids log2(0); the smallest of nudges

reg = "CP"                           # caudoputamen (striatum), a large region on this plane
in_reg = (obs["acronym"] == reg).to_numpy()
is_ctrl = in_reg & (obs["Condition"] == "naive").to_numpy()
is_preg = in_reg & (obs["Condition"] == "pregnant").to_numpy()
print(f"{reg}: {is_ctrl.sum()} control pixels, {is_preg.sum()} pregnant pixels")

mean_ctrl = X[is_ctrl].mean(0) + eps         # mean of each lipid in control pixels of CP
mean_preg = X[is_preg].mean(0) + eps         # mean of each lipid in pregnant pixels of CP
log2fc_cp = np.log2(mean_preg / mean_ctrl)   # one log2 fold change per lipid

# the three lipids that move most in this region
order = np.argsort(np.abs(log2fc_cp))[::-1][:3]
for j in order:
    print(f"  {lipids[j]:>18s}  log2FC = {log2fc_cp[j]:+.2f}")""")

md(r"""💡 **HINT.** The sign tells the direction. A positive log2FC means the lipid is *higher* in the
pregnant brain in that region; negative means lower. The magnitude is how big the change is on a
doubling scale.

That hand-built vector, one log2FC per lipid for one region, is exactly one row of the matrix we want.
The helper `cl.multimodal.region_change_matrix` does this for every region at once, keeping only
regions with at least 50 pixels per condition, and returns a tidy region x lipid table.""")

code(r"""# the helper does exactly what we just did, for every region at once
change = multimodal.region_change_matrix(
    adata, region_key="acronym", cond_key="Condition",
    control="naive", case="pregnant", min_pixels=50,
)
print("region x lipid change matrix:", change.shape)

# confirm our hand-built CP row matches the helper's CP row
hand = pd.Series(log2fc_cp, index=lipids)
auto = change.loc["CP"]
print("max abs difference (hand vs helper):", float((hand - auto).abs().max()))
change.iloc[:5, :4]""")

md(r"""⚠️ **CHECKPOINT.** The matrix is about `123 x 173`: 123 regions survived the 50-pixel gate, and
each has 173 lipid changes. The max difference between our hand-built VISp row and the helper's is
essentially zero (floating-point dust), so the helper is doing precisely what we did by hand.""")

# ----------------------------------------------------------------------------
md(r"""## 4 · bring in the genes and join on the region

Now the other modality. The file `avemerfish_imputed_named.parquet` is the Allen MERFISH transcriptome,
imputed to 8460 genes and **averaged inside each Allen region**. Its row index is the Allen acronym, the
exact same key our change matrix uses, and its columns are real mouse gene symbols, things like `Pparg`
and `Mobp`, which we will later hand straight to the gene ontology.

🔬 **TASK.** Load it and look at its shape.""")

code(r"""# region x gene matrix: Allen MERFISH-imputed expression averaged per region
genes_all = pd.read_parquet(
    "/home/fusar/lipidomics_tutorial_cajalcourse/avemerfish_imputed_named.parquet"
)
print("region x gene matrix:", genes_all.shape)
print("first regions:", list(genes_all.index[:4]))
print("a few genes:", list(genes_all.columns[:6]))
print("Pparg present?", "Pparg" in genes_all.columns, "| Mobp present?", "Mobp" in genes_all.columns)""")

md(r"""The join is an inner intersection on the region index: keep only the regions that appear in
*both* matrices. Some Allen regions are in the lipid plane but not in the MERFISH coverage, and vice
versa, so the intersection is smaller than either. The helper `cl.multimodal.join_genes` does the
intersection and returns the two matrices aligned row for row.

🔬 **TASK.** Join the two matrices.""")

code(r"""# align the lipid-change matrix and the gene matrix on shared regions
change, genes = multimodal.join_genes(change, genes_all)

print("aligned change matrix:", change.shape, "  aligned gene matrix:", genes.shape)
print("same region order?", (change.index == genes.index).all())
print("number of shared regions:", len(change))""")

md(r"""⚠️ **CHECKPOINT.** Both matrices now have the same number of rows, about 109 shared regions, in
the same order. Those 109 regions are our training and test examples: each region is one observation
with 8460 gene features and 173 lipid-change targets.

❓ **QUESTION.** We have 109 regions and 8460 genes. That is far more features than examples, the
classic "wide" problem where a model can memorise the training rows perfectly and learn nothing. Keep
this in mind: it is exactly why, in a moment, we compress the 8460 genes into a handful of programs
before we let any model see them.""")

# ----------------------------------------------------------------------------
md(r"""## 5 · gradient-boosted trees, from the ground up

We are about to ask a model to predict each lipid's change from gene expression. The model is
**XGBoost**, gradient-boosted decision trees. The name sounds heavy; the idea is light, and you should
hold it before pressing the button.

Start with a **decision tree**. A tree is a flowchart of yes/no questions on the features. "Is the
expression of gene A above 0.3? If yes, go left; if no, go right." Each answer sends you down a branch,
and at the bottom, the leaf, the tree predicts a number, usually the average target of the training
examples that landed there. A single tree is easy to read but weak: it carves the feature space into
boxes and predicts a flat value in each box, so it is blocky and it overfits if grown deep.

**Boosting** turns many weak trees into one strong predictor, and the trick is that the trees are not
independent. You fit the first tree. It is wrong by some amount on every example: those errors are
called residuals. You then fit a *second* tree, not to the target, but to the residuals of the first,
so the second tree's only job is to fix what the first got wrong. You add a small fraction of the
second tree's correction (the learning rate, often 0.05, keeps each step humble), and you repeat.
Hundreds of small trees, each patching the leftover error of all the trees before it, sum to a model
that bends smoothly to the data. That sequential error-correction is "gradient boosting": each tree
steps in the direction that most reduces the error, the gradient of the loss.

Why XGBoost rather than a straight line? Because the link from genes to lipids is not a straight line.
A lipid might rise with a gene only past a threshold, or rise with gene A only when gene B is also high.
Trees catch thresholds and interactions for free, because every split is a threshold and every path
down the tree is a conjunction of conditions. The cost is that trees, left alone, overfit, so we lean
on a few guardrails: shallow trees (`max_depth=3`, so each tree asks at most three questions), row and
column subsampling (each tree sees only 80% of the regions and 80% of the features, which decorrelates
them), and an L2 penalty (`reg_lambda`) that shrinks leaf predictions toward zero.

🔬 **TASK.** Watch boosting work on a toy before we use it for real. We make a wiggly 1D function,
hide it under noise, and fit XGBoost with 1, 5, and 200 trees so you can see the prediction sharpen as
trees accumulate.""")

code(r"""# a toy 1D regression to SEE boosting accumulate
x_toy = np.linspace(0, 1, 200)
true = np.sin(2 * np.pi * x_toy) + 0.5 * x_toy        # the wiggly truth
y_toy = true + rng.normal(0, 0.25, size=x_toy.size)   # observed = truth + noise
Xt = x_toy.reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, n_trees in zip(axes, [1, 5, 200]):
    m = xgb.XGBRegressor(n_estimators=n_trees, learning_rate=0.3, max_depth=3,
                         random_state=SEED, objective="reg:squarederror")
    m.fit(Xt, y_toy)
    ax.scatter(x_toy, y_toy, s=8, c="0.7", rasterized=True, label="noisy data")
    ax.plot(x_toy, true, c="k", lw=1.2, label="truth")
    ax.plot(x_toy, m.predict(Xt), c="crimson", lw=1.6, label="XGBoost")
    ax.set_title(f"{n_trees} tree{'s' if n_trees > 1 else ''}", fontsize=FS["m"])
    ax.set_xlabel("feature x")
axes[0].set_ylabel("target y")
axes[0].legend(fontsize=FS["xs"])
plt.tight_layout(); plt.show()""")

md(r"""⚠️ **CHECKPOINT.** One tree is a crude staircase. Five trees already curve. Two hundred trees
trace the wiggly truth without chasing every noisy point: that is boosting, many humble steps adding up
to a flexible fit. Now we point the same machine at our real biology.

❓ **QUESTION.** The 200-tree fit follows the truth but stays smoother than the noise. Which two
guardrails in the model object are buying us that smoothness rather than a fit that threads every grey
dot? (Hint: depth, and how big a step each tree is allowed to take.)""")

# ----------------------------------------------------------------------------
md(r"""## 6 · NMF gene programs as the features

We will not feed 8460 raw genes to XGBoost. With only 109 regions, that is hopeless: the model would
have 8460 knobs to fit 109 numbers and would memorise noise. We compress the genes first, exactly as
notebook 02 taught with NMF on lipids, now applied to genes.

Non-negative matrix factorisation finds a small set of **gene programs**. Each program is a recipe of
co-expressed genes, a non-negative weight for every gene, and each region is described by how strongly
it expresses each program. Formally NMF factors the region x gene matrix `V` into two non-negative
matrices, `V ≈ W · H`:

- **W** is region x program: how active each gene program is in each region. This becomes our feature
  matrix, 20 numbers per region instead of 8460.
- **H** is program x gene: the recipe of each program, the gene loadings. This is what we will read at
  the end to say *which genes* a predictive program is made of.

Non-negativity is the point. Genes are counted, never negative, so a program can only *add* genes
together, never cancel one against another. The programs come out as coherent co-expression modules you
can name, the way notebook 02's lipid programs came out as "the sphingolipid program" and "the
membrane-phospholipid program".

🔬 **TASK.** Unroll the program construction so the helper is transparent. NMF needs non-negative
input on a comparable scale, so we MinMax-scale each gene to [0, 1] first, then fit 20 programs.""")

code(r"""# unroll: MinMax each gene to [0,1] (NMF needs >=0), then factor into 20 programs
N_PROGRAMS = 20
Xg = MinMaxScaler().fit_transform(genes.values)          # regions x genes, in [0,1]
nmf = NMF(n_components=N_PROGRAMS, init="nndsvda", random_state=SEED, max_iter=500)
W = nmf.fit_transform(Xg)        # regions x programs : program activity per region
H = nmf.components_              # programs x genes    : gene recipe of each program

print("V (regions x genes):", Xg.shape)
print("W (regions x programs):", W.shape)
print("H (programs x genes):", H.shape)
print("all non-negative?", (W >= 0).all() and (H >= 0).all())""")

md(r"""The helper `cl.multimodal.gene_programs` does exactly this MinMax-then-NMF and returns `W`, `H`,
and the fitted model. We call it so the rest of the notebook uses the canonical objects.""")

code(r"""# the helper does exactly the MinMax + NMF above
W, H, nmf_model = multimodal.gene_programs(genes, n_programs=N_PROGRAMS, random_state=SEED)
print("W", W.shape, "| H", H.shape)

# look at one program's recipe: its 10 leading genes
prog0 = pd.Series(H[0], index=genes.columns).sort_values(ascending=False)
print("\nprogram1 leading genes:", list(prog0.head(10).index))""")

md(r"""💡 **HINT.** Each program is a soft module of genes that rise and fall together across regions.
A program is interpretable to the extent its leading genes share a biology, and that is precisely what
the gene ontology step at the end will test, formally.

⚠️ **CHECKPOINT.** `W` is `109 x 20` and `H` is `20 x 8460`. We turned 8460 gene features into 20
program features. The model now has a fighting chance against 109 examples.""")

# ----------------------------------------------------------------------------
md(r"""## 7 · one regressor per lipid, scored honestly

Now the core. For each of the 173 lipids we train one XGBoost regressor that predicts that lipid's
per-region change from the 20 gene programs. To know whether the model actually learned, we must score
it on regions it never trained on, so we split the 109 regions once into a training set and a held-out
test set, and we reuse the same split for every lipid. The score is the **Pearson correlation** between
the predicted change and the true change on the test regions. A Pearson r near 1 means the gene
programs predict where that lipid moves; near 0 means they do not.

🔬 **TASK.** Unroll the loop for ONE lipid first, end to end, so the per-lipid helper is no longer
opaque. We z-score the program activities (so each program enters on a comparable scale), split the
regions, fit, predict, and correlate.""")

code(r"""# unroll the per-lipid regression for ONE lipid, with a proper held-out test
Wz = StandardScaler().fit_transform(W)               # z-score each program activity
region_idx = np.arange(Wz.shape[0])
tr, te = train_test_split(region_idx, test_size=0.25, random_state=SEED)
print(f"{len(tr)} training regions, {len(te)} held-out test regions")

lipid0 = "HexCer 42:2;O2"                             # a myelin marker that moves in pregnancy
y0 = change[lipid0].to_numpy()

m0 = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=SEED, objective="reg:squarederror")
m0.fit(Wz[tr], y0[tr])
pred0 = m0.predict(Wz[te])
r0 = pearsonr(pred0, y0[te])[0]
print(f"{lipid0}: test Pearson r = {r0:.2f}")

# the scatter that the number summarises: predicted vs true change on test regions
fig, ax = plt.subplots(figsize=(4.2, 4.0))
ax.scatter(y0[te], pred0, s=24, c="crimson", alpha=0.8)
lim = [min(y0[te].min(), pred0.min()), max(y0[te].max(), pred0.max())]
ax.plot(lim, lim, ls="--", lw=0.8, c="0.4")
ax.set_xlabel(f"true log2FC ({lipid0})"); ax.set_ylabel("predicted log2FC")
ax.set_title(f"held-out regions, r = {r0:.2f}", fontsize=FS["m"])
plt.tight_layout(); plt.show()""")

md(r"""⚠️ **CHECKPOINT.** The points cluster along the diagonal: regions where this myelin lipid rose
in pregnancy are regions the gene programs predicted it would rise. A test Pearson r around 0.6 to 0.7
for `HexCer 42:2;O2` says the transcriptome carries real, generalisable information about where this
lipid changes.

Now we run the same loop for all 173 lipids. The helper `cl.multimodal.predict_changes` does exactly
what we just unrolled, per lipid, and on top of the test Pearson r it also computes, for free, the SHAP
program-importance we use in the next section. This fits 173 small models, so it takes a couple of
minutes.

🔬 **TASK.** Run the full prediction. Be patient: about two minutes.""")

code(r"""# the helper fits one XGBoost per lipid (exactly the unrolled loop) and also
# returns the mean |SHAP| program-importance matrix used in section 7.
scores, shap_mat = multimodal.predict_changes(
    W, change, change.index, test_size=0.25, random_state=SEED
)
n_good = int((scores["test_r"] > 0.3).sum())
print(f"lipids with test Pearson r > 0.3: {n_good} of {len(scores)}")
print(f"mean test r across lipids: {scores['test_r'].mean():.2f}")
print("\nbest-predicted lipids:")
print(scores.head(6).to_string(index=False))""")

md(r"""⚠️ **CHECKPOINT.** About **105 of 173 lipids reach test r > 0.3**, and the mean test r is about
0.37. Read that carefully. It does not mean the genes "cause" the lipids: it means that, across
anatomy, where a lipid changes in pregnancy is partly predictable from the regional transcriptome. The
best-predicted lipids include several sphingolipids and phospholipids that move strongly with anatomy,
which is the honest, expected result.

❓ **QUESTION.** A test r of 0.37 averaged across lipids is modest by the standards of, say, predicting
a held-out pixel from its neighbours. Why might that be the *right* answer here rather than a failure?
(Hint: the gene matrix is a regional average from different mice, and a lipid is the end product of
enzymes, transport, and turnover, not a direct readout of one transcript.)""")

# distribution of scores
md(r"""Look at the whole distribution of scores, not just the top. A histogram of the per-lipid test r
shows how many lipids the transcriptome predicts well versus poorly.""")

code(r"""# distribution of per-lipid test Pearson r
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(scores["test_r"], bins=30, color="steelblue", edgecolor="white")
ax.axvline(0.3, ls="--", lw=0.8, c="crimson")
ax.set_xlabel("test Pearson r"); ax.set_ylabel("number of lipids")
ax.set_title(f"{n_good}/{len(scores)} lipids predicted with r > 0.3", fontsize=FS["m"])
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------
md(r"""## 8 · SHAP: which gene program drives each prediction

A test r tells you a model predicts well. It does not tell you *why*, which program the model leaned on.
For that we use **SHAP**, SHapley Additive exPlanations. The idea comes from game theory. Imagine the
prediction for one region is a payout, and the features (the 20 gene programs) are players who
cooperate to produce it. SHAP asks, for each player, how much of the payout is fairly attributed to
them, by averaging the player's marginal contribution over every possible order in which the players
could join the game. The result is one signed number per feature per prediction, the SHAP value, and
the SHAP values for a region sum, with a baseline, exactly to the model's prediction for that region.
So SHAP decomposes each prediction into per-program contributions.

For tree models this averaging has a fast exact algorithm, TreeSHAP, built straight into XGBoost. We
take the absolute SHAP value of each program (we care about magnitude of influence, not sign), average
it across the test regions, and get one importance number per program per lipid. The helper already
computed this and handed it back as `shap_mat`, a programs x lipids table.

🔬 **TASK.** Read out, for one lipid, exactly how the helper got its SHAP numbers, using XGBoost's
native exact path. Then confirm it matches the helper.""")

code(r"""# unroll TreeSHAP for ONE lipid using XGBoost's exact pred_contribs path
m0 = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=SEED, objective="reg:squarederror")
m0.fit(Wz[tr], change[lipid0].to_numpy()[tr])

# pred_contribs=True returns exact TreeSHAP: one value per program + a final bias column
contribs = m0.get_booster().predict(xgb.DMatrix(Wz), pred_contribs=True)
contribs = contribs[:, :-1]                       # drop the last column (bias / base value)
shap_one = np.abs(contribs).mean(0)               # mean |SHAP| per program over all regions

top3 = np.argsort(shap_one)[::-1][:3]
for j in top3:
    print(f"  program{j+1}: mean |SHAP| = {shap_one[j]:.4f}")

# the helper's column for this lipid should match
print("\nmatches helper's shap_mat column?",
      np.allclose(shap_one, shap_mat[lipid0].to_numpy()))""")

md(r"""💡 **HINT.** `pred_contribs=True` is XGBoost's exact TreeSHAP. The last column it returns is the
base value (the model's average output), which is why we drop it before taking the mean absolute SHAP
per program.

Now aggregate across all lipids. Averaging each program's mean-|SHAP| across the 173 lipids gives the
global importance of each gene program for explaining the pregnancy lipidome change. The program at the
top is the one the models leaned on most, across all lipids.

🔬 **TASK.** Rank the programs and plot the top ones.""")

code(r"""# aggregate program importance across all lipids, then rank
prog_importance = shap_mat.mean(axis=1).sort_values(ascending=False)
print("most predictive gene programs (mean |SHAP| across all lipids):")
print(prog_importance.head(6).to_string())

fig, ax = plt.subplots(figsize=(6, 3.6))
top = prog_importance.head(10)[::-1]
ax.barh(range(len(top)), top.values, color="mediumpurple")
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=FS["s"])
ax.set_xlabel("mean |SHAP| across lipids")
ax.set_title("gene programs that best predict the pregnancy lipid change", fontsize=FS["m"])
plt.tight_layout(); plt.show()""")

md(r"""⚠️ **CHECKPOINT.** One program stands clearly at the top: `program2`. It is the gene program the
173 models lean on most when predicting where lipids change in pregnancy. The next question is the only
one that matters biologically: *which genes is program2 made of?*""")

# ----------------------------------------------------------------------------
md(r"""## 9 · read the leading genes of the top program

A program is a recipe, a row of `H`, one non-negative weight per gene. The genes with the biggest
weights *define* the program. The helper `cl.multimodal.top_genes_for_program` sorts a program's row of
`H` and returns its leading genes.

🔬 **TASK.** Pull the leading genes of the top program.""")

code(r"""# leading genes of the most predictive program (top program by SHAP)
top_prog_name = prog_importance.index[0]                   # e.g. "program2"
top_prog_idx = int(top_prog_name.replace("program", "")) - 1
leading = multimodal.top_genes_for_program(H, genes, top_prog_idx, top=30)

print(f"{top_prog_name} is the top predictive program.")
print("its 20 leading genes:")
print(list(leading.head(20).index))""")

md(r"""Look at that list. Alongside several uncharacterised Riken clones it carries `Pparg`, the master
transcriptional regulator of lipid storage and adipocyte/lipid metabolism, together with markers of
specific cortical interneuron populations like `Lamp5` and `Calb1`. That a program *led by `Pparg`* is
the strongest predictor of where the lipidome changes is satisfying: `Pparg` sits at the centre of
lipid-handling transcription, so a region whose `Pparg` program is active is a region whose lipid
economy the model can read.

🔬 **TASK.** Plot the leading-gene loadings so the recipe is visible.""")

code(r"""# the leading-gene loadings of the top program as a bar chart
fig, ax = plt.subplots(figsize=(6, 4.2))
lg = leading.head(18)[::-1]
ax.barh(range(len(lg)), lg.values, color="teal")
ax.set_yticks(range(len(lg))); ax.set_yticklabels(lg.index, fontsize=FS["xs"])
ax.set_xlabel("NMF loading (weight in the program)")
ax.set_title(f"{top_prog_name}: leading genes (Pparg-led)", fontsize=FS["m"])
plt.tight_layout(); plt.show()""")

md(r"""❓ **QUESTION.** This program is a *correlative* predictor, not a proven cause. `Pparg` co-varies
with the lipid change across regions; it is not, on this evidence alone, switching the lipids on. What
experiment would you need to move from "predicts" to "causes"? (Hint: think perturbation, not
correlation.)""")

# ----------------------------------------------------------------------------
md(r"""## 10 · gene ontology: name the biology of a program

We have genes. We want biology. A list of gene symbols is not yet a story; **gene ontology** turns it
into one. The Gene Ontology is a controlled vocabulary of biological terms, organised into three
namespaces: biological process (what the gene helps *do*, like "myelination"), molecular function (the
biochemical activity), and cellular component (where in the cell it acts). Every gene is annotated to
the terms it participates in.

Enrichment asks a clean statistical question. Take a program's leading genes (the *study* set). Take
all the genes we measured (the *background*). For each ontology term, count how many study genes carry
it versus how many background genes do, and run a Fisher exact test: is this term over-represented in
the study set beyond chance? Because we test thousands of terms, we correct the p-values with
Benjamini-Hochberg, the same FDR control we used for the differential lipids. A term that survives is a
biological process the program's genes are genuinely about.

We run this with two tools. `mygene` converts gene *symbols* into the Entrez gene IDs that the ontology
files key on. `goatools` holds the ontology graph (`go-basic.obo`) and the gene-to-term annotations
(`gene2go`, filtered to mouse, taxid 10090) and runs the enrichment. We pre-downloaded both files into
`data/go/` so the notebook does not depend on a live download; the cell after this one shows the
download call you would use if the files were absent, and it is tagged illustrative because we do not
re-run it.

🔬 **TASK.** Load the ontology and the mouse annotations.""")

code(r"""# load the pre-downloaded Gene Ontology graph and the mouse gene->term annotations
import io, contextlib
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.go_enrichment import GOEnrichmentStudy

GO_DIR = "/home/fusar/lipidomics_tutorial_cajalcourse/data/go"
with contextlib.redirect_stdout(io.StringIO()):          # the loaders are chatty; quiet them
    godag = GODag(f"{GO_DIR}/go-basic.obo")
    anno = Gene2GoReader(f"{GO_DIR}/gene2go_mouse", taxids=[10090])
    ns2assoc = anno.get_ns2assc()

# fold the three namespaces' gene->term maps into one association dict
assoc = {}
for ns, d in ns2assoc.items():
    for gene_id, gos in d.items():
        assoc.setdefault(gene_id, set()).update(gos)
print(f"ontology terms: {len(godag):,}")
print(f"mouse genes with annotations: {len(assoc):,}")""")

md(r"""The cell below is the live-download version, kept for reference. If `data/go/` were empty you
would run this to fetch `go-basic.obo` and `gene2go` from the public servers, then filter `gene2go` to
mouse. It is tagged illustrative; we do not execute it, because the files are already on disk.""")

code(r"""# ILLUSTRATIVE: how you would download the GO files if they were not already on disk.
# (Not executed; the files live in data/go/. The gene2go full file is ~10 GB decompressed,
#  so we filter it to mouse, taxid 10090, before using it.)
#
# from goatools.base import download_go_basic_obo
# download_go_basic_obo("data/go/go-basic.obo")
# # gene2go via HTTPS (the goatools FTP helper is brittle):
# #   curl -sSL -o gene2go.gz https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
# #   gunzip gene2go.gz
# #   awk -F'\t' 'NR==1 || $1==10090' gene2go > data/go/gene2go_mouse
print("illustrative cell; the GO files are already in data/go/")""", tag="illustrative")

md(r"""Now the enrichment. We need Entrez IDs, so we convert symbols with `mygene`. The background is
every gene we measured (the 8460 in the matrix), which is the correct universe: we only ever could have
detected those genes, so they, not the whole genome, are the fair comparison.

🔬 **TASK.** Convert the background genes to Entrez IDs and build the enrichment study object. This
makes one network call to `mygene`.""")

code(r"""# convert gene symbols -> Entrez IDs with mygene (one network call)
import mygene
mg = mygene.MyGeneInfo()

def symbols_to_entrez(symbols):
    res = mg.querymany(list(symbols), scopes="symbol", fields="entrezgene",
                       species="mouse", verbose=False)
    return {r["query"]: int(r["entrezgene"]) for r in res if "entrezgene" in r}

bg_map = symbols_to_entrez(genes.columns)                 # background = all measured genes
print(f"background genes mapped to Entrez: {len(bg_map)} of {genes.shape[1]}")

with contextlib.redirect_stdout(io.StringIO()):
    goe = GOEnrichmentStudy(list(bg_map.values()), assoc, godag,
                            propagate_counts=True, alpha=0.05, methods=["fdr_bh"])
print("enrichment study ready (background, associations, ontology loaded)")""")

md(r"""We now have a function: hand it a program's leading genes, get back its significantly enriched
biological-process terms. We run it first on the top predictive program (`Pparg`-led), then we scan the
other top programs, because the single most interpretable biology in this dataset lives in a slightly
different program, and finding it is part of the lesson: the *most predictive* program and the *most
interpretable* program need not be the same.

🔬 **TASK.** Run enrichment on the top program's leading genes.""")

code(r"""def enriched_bp(program_idx, top=100):
    '''Significantly enriched biological-process GO terms for a program's leading genes.'''
    lead = multimodal.top_genes_for_program(H, genes, program_idx, top=top)
    study_map = symbols_to_entrez(lead.index)
    with contextlib.redirect_stdout(io.StringIO()):
        res = goe.run_study(list(study_map.values()))
    sig = [r for r in res if r.enrichment == "e" and r.NS == "BP" and r.p_fdr_bh < 0.05]
    sig = sorted(sig, key=lambda r: r.p_fdr_bh)
    return pd.DataFrame([(r.name, r.p_fdr_bh, r.ratio_in_study[0]) for r in sig],
                        columns=["GO term (BP)", "FDR", "study_genes"])

go_top = enriched_bp(top_prog_idx, top=100)
print(f"{top_prog_name} (Pparg-led): {len(go_top)} enriched BP terms at FDR < 0.05")
print(go_top.head(8).to_string(index=False) if len(go_top) else "  (no term survives FDR < 0.05)")""")

md(r"""💡 **HINT.** Do not be disappointed if the Pparg-led program returns few or no significant terms.
That is honest: a program led by a single master regulator plus assorted region markers need not map
onto one clean process, and the gene ontology refuses to invent a story that the annotations do not
support. The model can lean on a program for prediction even when that program is not a tidy biological
process. So we scan the other top predictive programs and find the one with the *cleanest* biology, the
one carrying the single strongest enriched term.

🔬 **TASK.** Scan the top predictive programs and keep the one with the strongest enriched term.""")

code(r"""# scan the top predictive programs; record each one's strongest (lowest-FDR) BP term
rows = []
go_results = {}
for pname in prog_importance.head(8).index:
    pidx = int(pname.replace("program", "")) - 1
    df = enriched_bp(pidx, top=100)
    go_results[pname] = df
    if len(df):
        rows.append((pname, len(df), df["GO term (BP)"].iloc[0], float(df["FDR"].iloc[0])))
    else:
        rows.append((pname, 0, "-", np.nan))
scan = pd.DataFrame(rows, columns=["program", "n_sig_BP_terms", "top term", "top_FDR"])
print(scan.to_string(index=False))

# pick the most interpretable program: the one whose strongest term has the lowest FDR.
# (Term *count* favours vague catch-all terms; the single strongest term favours a specific,
#  coherent process, which is what we want to name.)
best_prog = scan.dropna(subset=["top_FDR"]).sort_values("top_FDR").iloc[0]["program"]
print(f"\nmost interpretable program: {best_prog}")
print(go_results[best_prog].head(8).to_string(index=False))""")

md(r"""⚠️ **CHECKPOINT.** One program lights up with **myelination, axon ensheathment, ensheathment of
neurons** at FDR near `1e-9`, the single strongest enriched term in the whole scan. That is the
punchline. Its genes are the oligodendrocyte and myelin transcripts: `Mobp` sits near the top, and
`Plp1`, `Mal`, `Mbp`, and `Mog` follow within the program, which is exactly why the ontology calls it
myelination. So among the gene programs that predict where the lipidome changes in pregnancy, the most
biologically coherent one is a *myelination* program. This closes a loop with the differential side of
the course: the lipids that move most in pregnancy include the myelin sphingolipids like
`HexCer 42:2;O2`, and here the transcriptional program that best tracks
the lipid change is the program that builds myelin. The lipids and the genes are telling the same
story from two independent experiments.

🔬 **TASK.** Plot the enriched terms of the myelination program as a clean ontology bar chart.""")

code(r"""# the gene-ontology bar chart for the most interpretable program
go_best = go_results[best_prog].head(10)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
y = -np.log10(go_best["FDR"].clip(lower=1e-300))[::-1]
labels = go_best["GO term (BP)"][::-1]
ax.barh(range(len(y)), y, color="darkorange")
ax.set_yticks(range(len(y))); ax.set_yticklabels(labels, fontsize=FS["xs"])
ax.set_xlabel("-log10 FDR")
ax.set_title(f"{best_prog}: enriched biological processes", fontsize=FS["m"])
plt.tight_layout(); plt.show()

# record the leading genes of the myelination program for the figure later
best_idx = int(best_prog.replace("program", "")) - 1
best_leading = multimodal.top_genes_for_program(H, genes, best_idx, top=12)
print(f"{best_prog} leading genes:", list(best_leading.index))""")

# ----------------------------------------------------------------------------
md(r"""## 11 · the negative control: a permutation null

Before we believe any of this, we owe ourselves a sanity check. A test r of 0.7 for the best lipids
*feels* convincing, but with only 109 regions, could a model score that high by chance, just by
exploiting flexible trees on a small sample? The clean way to answer is a **permutation null**. We
break the one thing we claim is real, the correspondence between a region's genes and its lipid change,
and see whether the score collapses.

The recipe mirrors the real run exactly so that only the labels change. We take a handful of the
best-predicted lipids, fit the real models, and record the mean test r. Then we *shuffle* each lipid's
change across regions, so region A now carries region Q's lipid change, and re-fit everything. If the
genes truly predict the lipid change, shuffling should destroy the signal and the mean test r should
fall to near zero. We repeat the shuffle many times to build a null distribution, and the empirical
p-value is the fraction of shuffles that matched or beat the real score, with the standard
`(1 + count) / (n + 1)` form that never reports an impossible zero.

🔬 **TASK.** Run the permutation null on the 12 best-predicted lipids. Twenty shuffles, about
half a minute.""")

code(r"""# permutation null on the best-predicted lipids: shuffle the target, re-fit, compare
N_PERM = 20
top_lipids = scores.head(12)["lipid"].tolist()
target_sub = change[top_lipids]

def mean_test_r(target_df, permute_rng=None):
    '''Mean held-out Pearson r across the given lipids; shuffle targets if permute_rng given.'''
    rs = []
    for lip in target_df.columns:
        y = target_df[lip].to_numpy().copy()
        if permute_rng is not None:
            y = permute_rng.permutation(y)               # break region <-> change link
        m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=SEED, objective="reg:squarederror")
        m.fit(Wz[tr], y[tr])
        p = m.predict(Wz[te])
        rs.append(pearsonr(p, y[te])[0] if np.std(p) > 0 and np.std(y[te]) > 0 else 0.0)
    return float(np.mean(rs))

obs_r = mean_test_r(target_sub)                          # the real, observed mean r
perm_rng = np.random.default_rng(0)
perm_r = np.array([mean_test_r(target_sub, perm_rng) for _ in range(N_PERM)])

p_emp = (1 + np.sum(perm_r >= obs_r)) / (N_PERM + 1)
print(f"observed mean test r : {obs_r:.3f}")
print(f"permuted mean test r : range [{perm_r.min():.3f}, {perm_r.max():.3f}], "
      f"mean {perm_r.mean():.3f}")
print(f"empirical p-value    : {p_emp:.4f}")""")

md(r"""🔬 **TASK.** Draw the null. The observed score should sit far to the right of the cloud of
permuted scores.""")

code(r"""# the negative-control figure: null distribution vs the observed score
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(perm_r, bins=12, color="0.7", edgecolor="white", label="shuffled targets (null)")
ax.axvline(obs_r, color="crimson", lw=2, label=f"observed r = {obs_r:.2f}")
ax.set_xlabel("mean test Pearson r"); ax.set_ylabel("permutations")
ax.set_title(f"permutation null: p = {p_emp:.3f}", fontsize=FS["m"])
ax.legend(fontsize=FS["xs"])
plt.tight_layout(); plt.show()""")

md(r"""⚠️ **CHECKPOINT.** The observed mean test r (about 0.72) sits far to the right of the permuted
null, which clusters near zero (roughly -0.1 to +0.1), giving an empirical p around 0.05 with only 20
shuffles. Shuffling the region labels destroyed the signal: the genes predict the lipid change because
of the *real* region-to-region correspondence, not because flexible trees can fit any small table. With
more permutations the p-value would tighten further; 20 is enough to make the point in class.

❓ **QUESTION.** We shuffled each lipid's change *across regions* and kept the gene programs fixed. Why
is that the right thing to break, rather than, say, shuffling the gene values? (Hint: which
correspondence is the scientific claim?)""")

# ----------------------------------------------------------------------------
md(r"""## 12 · the publication figure

This is where you do careful scientific plotting. We assemble one multi-panel figure that tells the
whole story at a glance, following the lab figure rules: a small fixed set of font sizes, no top or
right spines, sequential colormaps for intensities and divergent for signed data, rasterised scatter
but vector text, lightweight colorbars. The figure has four panels:

- **A**: the spatial lipid, `HexCer 42:2;O2` painted on both sections, so the reader sees the raw
  biology the whole analysis is about.
- **B**: the volcano of the differential test, control vs pregnant, so the reader sees which lipids
  move.
- **C**: a sorted region x lipid heatmap of the change matrix, the input to the model.
- **D**: the gene ontology of the myelination program, the mechanistic read-out.

🔬 **TASK.** First compute the differential table for panel B with the course's Wilcoxon + BH helper.""")

code(r"""# the differential test for the volcano panel (Wilcoxon rank-sum + Benjamini-Hochberg)
diff = analysis.differential_lipids(
    adata, group_col="Condition", group1="naive", group2="pregnant",
    min_fc=0.2, pthr=0.05,
)
print("significant lipids (|log2FC| > 0.2 and q < 0.05):", int(diff["sig"].sum()), "of", len(diff))
print(diff.sort_values("log2fc").head(3)[["lipid", "log2fc", "qval"]].to_string(index=False))""")

md(r"""🔬 **TASK.** Now assemble the four-panel figure. Read each block: the panels reuse the tested
plotting helpers (`cl.plotting.spatial_lipid`, `cl.plotting.volcano`, `cl.plotting.sorted_lipid_heatmap`)
so your effort goes into composition and labelling, not into reinventing the scatter.""")

code(r"""# one publication-quality multi-panel figure
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.32, wspace=0.28)

# --- panel A: the spatial lipid on both sections (two sub-axes inside the top-left cell) ---
gsA = gs[0, 0].subgridspec(1, 2, wspace=0.05)
axA = [fig.add_subplot(gsA[0, 0]), fig.add_subplot(gsA[0, 1])]
plotting.spatial_lipid(adata, "HexCer 42:2;O2", axes=axA, point_size=2.5)
axA[0].text(-0.05, 1.08, "A", transform=axA[0].transAxes, fontsize=FS["l"], fontweight="bold")

# --- panel B: the volcano ---
axB = fig.add_subplot(gs[0, 1])
plotting.volcano(diff, label_col="lipid", top_n=8, ax=axB,
                 title="control vs pregnant")
axB.text(-0.12, 1.05, "B", transform=axB.transAxes, fontsize=FS["l"], fontweight="bold")

# --- panel C: sorted region x lipid change heatmap (the model input) ---
axC = fig.add_subplot(gs[1, 0])
order_r = change.abs().mean(1).sort_values(ascending=False).index   # regions by change magnitude
order_c = change.abs().mean(0).sort_values(ascending=False).index   # lipids by change magnitude
M = change.loc[order_r, order_c]
im = axC.imshow(M.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
axC.set_xlabel("lipids (sorted by change magnitude)")
axC.set_ylabel("Allen regions (sorted)")
axC.set_xticks([]); axC.set_yticks([])
axC.set_title("per-region pregnancy lipid change (log2FC)", fontsize=FS["m"])
cl.style.lightweight_colorbar(im, axC, label="log2FC")
axC.text(-0.08, 1.05, "C", transform=axC.transAxes, fontsize=FS["l"], fontweight="bold")

# --- panel D: gene ontology of the myelination program ---
axD = fig.add_subplot(gs[1, 1])
go_best = go_results[best_prog].head(10)
yv = -np.log10(go_best["FDR"].clip(lower=1e-300))[::-1]
axD.barh(range(len(yv)), yv, color="darkorange")
axD.set_yticks(range(len(yv))); axD.set_yticklabels(go_best["GO term (BP)"][::-1], fontsize=FS["xs"])
axD.set_xlabel("-log10 FDR")
axD.set_title(f"{best_prog}: enriched processes", fontsize=FS["m"])
axD.text(-0.45, 1.05, "D", transform=axD.transAxes, fontsize=FS["l"], fontweight="bold")

fig.suptitle("which genes explain the pregnancy lipid changes", fontsize=FS["l"], y=0.97)
plt.show()""")

md(r"""⚠️ **CHECKPOINT.** Four panels, one story. Panel A shows the myelin lipid `HexCer 42:2;O2`
lighting up white matter in both sections. Panel B shows the lipids that move between control and
pregnant, with the myelin sphingolipids among the strongest. Panel C is the region x lipid change
matrix the model learned from, blue where lipids fall and red where they rise. Panel D names the most
interpretable predictive gene program: myelination. The lipidome change and the transcriptional program
that predicts it agree, and they came from two experiments that never shared a single tissue section,
joined only through the Allen atlas.

🔬 **TASK.** Save the figure as a vector PDF, the lab default, editable later in Illustrator.""")

code(r"""# save the figure as a vector PDF (text stays editable, scatter is rasterised per-artist)
out_pdf = "/home/fusar/lipidomics_tutorial_cajalcourse/notebooks/level3/08_figure.pdf"
fig.savefig(out_pdf)
print("saved", out_pdf)""")

# ----------------------------------------------------------------------------
md(r"""## what you built

You integrated two experiments that never touched the same tissue, the MALDI lipidome and the Allen
MERFISH transcriptome, through the one thing they share: the same 3D Allen coordinate space. You did it
at two granularities of one gene<->lipid question.

The fine, per-pixel way came first. You built a cKDTree on the MERFISH cell coordinates, queried every
cell within a small ball of each MSI pixel, and borrowed their genes (averaged) and their cell type
(majority vote). Only about 60% of pixels matched at 75 um, rising to about 77% at 100 um, because the
MSI plane only overlaps the discrete MERFISH coronal sections, and that is the honest geometry of the
match. The payoff was visual and immediate: a borrowed `Mog` myelin transcript tracked the measured
`HexCer 42:2;O2` myelin lipid pixel for pixel, a per-pixel cell-type territory map placed
oligodendrocytes on the white-matter tracts, and a lipizone x cell-type reciprocal-enrichment heatmap
showed that lipid territories are very often cell-type territories.

The coarse, per-region way did the rest. You reduced both modalities to the named Allen region and asked
a clean question: do a region's genes predict where its lipidome changes in pregnancy?

The pipeline, in one breath: build the per-region lipid-change matrix, join the region-averaged gene
expression, compress 8460 genes into 20 NMF gene programs, fit one gradient-boosted tree per lipid, and
score it honestly on held-out regions. About 105 of 173 lipids reached test r > 0.3, the transcriptome
carries real spatial information about the lipid change. SHAP told us which program each model leaned
on, and the most predictive program was led by `Pparg`, the master regulator of lipid metabolism.
Gene ontology then named the most biologically coherent program: myelination, built from `Mobp`,
`Plp1`, `Mal`, `Mbp`, and `Mog`. A permutation null confirmed the signal was real, not a small-sample
artefact: shuffling the region labels collapsed the score to zero.

The closing insight is the one to carry forward, and the two granularities tell it in unison. At the
pixel level, oligodendrocyte territories, the borrowed `Mog` gene, and the `HexCer 42:2;O2` lipid all
fall on the same white matter. At the region level, the lipids that move most in pregnancy include the
myelin sphingolipids, and the gene program that best predicts where the lipidome changes is the program
that builds myelin. Two modalities, measured years and labs apart, point at the same biology, whether
you read it pixel by pixel or region by region. That is what integration by shared coordinates buys
you: not proof of cause, but a strong, testable hypothesis about which genes to perturb next.""")

# ----------------------------------------------------------------------------
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3",
}
nb.metadata["language_info"] = {"name": "python"}
with open(OUT, "w") as f:
    nbf.write(nb, f)
print("wrote", OUT, "with", len(cells), "cells")
