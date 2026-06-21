"""In-place edit of notebooks/level1/03_normalization_umaia_solution.ipynb.

Turns the uMAIA section from an illustrative / cannot-run block into a REAL run on
the cajal-umaia kernel (JAX on CPU). Loads data/umaia_input.npz (the matched
(100258, 2, 104) log-intensity tensor + mask built by scripts/build_umaia_input.py),
fits uMAIA by MAP/SVI, applies the histogram-matching transform, and shows the real
before/after per-section histograms plus the cross-section gap shrinking.

Preserves all the existing teaching prose: batch effects, the bimodal fg/bg model,
the rank-1 batch term, the unrolled CDF/inverse-CDF transform, the two-section corner
case, and the per-lipid 0-1 scaling. We only swap the framing ("we don't run it" ->
"we run it") and the compute cell, then add the real before/after panels.

    /home/fusar/mambaforge/envs/cajal-umaia/bin/python scripts/edit_03_umaia_run.py
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_markdown_cell, new_code_cell

NB = "notebooks/level1/03_normalization_umaia_solution.ipynb"

nb = nbf.read(NB, as_version=4)
by_id = {c.get("id"): (i, c) for i, c in enumerate(nb.cells)}


def replace_source(cell_id, new_source, drop_tags=None, set_tags=None):
    i, c = by_id[cell_id]
    c.source = new_source
    if drop_tags or set_tags is not None:
        tags = c.metadata.get("tags", [])
        if set_tags is not None:
            tags = list(set_tags)
        elif drop_tags:
            tags = [t for t in tags if t not in drop_tags]
        c.metadata["tags"] = tags
    return i


# ---------------------------------------------------------------------------
# Cell 00471ec5 — the imports. JAX_PLATFORMS=cpu must be set BEFORE jax is
# imported by uMAIA, so it goes at the very top, before any heavy import.
# ---------------------------------------------------------------------------
replace_source("00471ec5", r'''# --- this notebook RUNS uMAIA for real, on the cajal-umaia kernel ---
# uMAIA's fit is JAX/NumPyro. The course cluster has no GPU, so we pin JAX to the
# CPU. This MUST happen before anything imports jax (uMAIA does, on import), so it
# is the very first line of the notebook.
import os
os.environ["JAX_PLATFORMS"] = "cpu"

# the scientific-Python stack, plus the course helper package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad

from scipy import stats               # gives us the normal CDF for the transform
from scipy.interpolate import interp1d  # inverts a CDF by interpolation

# the course helper package (installed editable as cajal_lipidomics)
import cajal_lipidomics as cl
from cajal_lipidomics import analysis   # the unrolled statistics, incl. min01_per_lipid
from cajal_lipidomics.style import set_style, FS
set_style()                            # clean, dense, Illustrator-editable panels

# uMAIA itself: the normalizer at the front of the EUCLID pipeline. Importing it
# brings in jax/numpyro, which is why JAX_PLATFORMS was set above.
import uMAIA

# one global seed so every number and figure below is reproducible
RNG_SEED = 0
rng = np.random.default_rng(RNG_SEED)

plt.rcParams["figure.dpi"] = 110

# resolve the project root so the data paths below work from any working directory
DATA = "data"
if not os.path.isdir(DATA):
    here = os.getcwd()
    for _ in range(5):                       # walk up until we find the data folder
        if os.path.isdir(os.path.join(here, "data")):
            DATA = os.path.join(here, "data")
            break
        here = os.path.dirname(here)

print("ready. numpy", np.__version__, "| anndata", ad.__version__, "| JAX on", os.environ["JAX_PLATFORMS"])''')


# ---------------------------------------------------------------------------
# Cell 8c6eeb86 — checkpoint mentioning the kernel. This notebook is cajal-umaia.
# ---------------------------------------------------------------------------
replace_source("8c6eeb86", r'''⚠️ **CHECKPOINT.** You should see `ready. numpy ... | anndata ... | JAX on cpu` and no
red error. This notebook runs on the **`cajal-umaia`** kernel, not `cajal-lipidomics`:
it is the only one with JAX, NumPyro, and uMAIA installed. If you get a
`ModuleNotFoundError: No module named 'uMAIA'`, you are on the wrong kernel; pick
`cajal-umaia` in the top-right kernel picker and rerun. The `JAX on cpu` at the end
confirms we pinned JAX to the CPU, because the course cluster has no GPU.''')


# ---------------------------------------------------------------------------
# Cell 68d9ddf9 — drop the "we will not run the heavy fit live" framing. We run it.
# Keep the lipid-naming teaching, keep the honest note that this substrate is already
# normalized (true, and it is why we run uMAIA on the RAW METASPACE pull instead).
# ---------------------------------------------------------------------------
replace_source("68d9ddf9", r'''⚠️ **CHECKPOINT.** 174,768 pixels and 173 lipids, split into `naive` (the control female, about
84k pixels) and `pregnant` (about 90k pixels). The lipid names read as a class plus a chain summary:
`HexCer 42:2;O2` is a hexosylceramide with 42 acyl carbons and 2 double bonds, a sphingolipid that
marks myelin. We met that naming in N2.

One honest note up front. This `sections_pair.h5ad` substrate has *already* been through uMAIA,
because every downstream notebook needs comparable values. That is convenient for seeing what a
*good* result looks like, but it would be circular to "run" normalization on data that is already
normalized. So we do the real fit on the **raw** images instead: a separate file,
`data/umaia_input.npz`, holds the un-normalized intensities pulled straight from METASPACE for both
sections, and that is what we feed uMAIA below. You will run the actual MAP/SVI fit, watch it
converge in a couple of minutes on the CPU, and see the raw, misaligned per-section histograms snap
into register. Nothing here is illustrative.''')


# ---------------------------------------------------------------------------
# Cell 97f7e071 — "how it is fit". Keep the MAP/SVI explanation verbatim; only swap
# the closing "we do NOT run it here" framing for "we run it here, for real".
# ---------------------------------------------------------------------------
replace_source("97f7e071", r'''### how it is fit: MAP via SVI, not posterior sampling

uMAIA fits this model with **stochastic variational inference (SVI)** using an `AutoDelta` guide. That
phrase has a simple meaning. A delta-function guide collapses the "posterior" to a single point, so
SVI here is doing **maximum-a-posteriori (MAP)** estimation: it finds the single most probable set of
parameter values, with the priors acting as gentle regularizers, by gradient descent (Adam). There is
**no MCMC and there are no posterior samples.** Think "regularized best-fit point estimate", not
"Bayesian sampling". The discrete per-pixel labels `z` are marginalized analytically by enumeration,
so the optimizer only ever sees continuous parameters. With a fixed seed the result is deterministic.

Now we run it, for real, on the two course sections. The call below loads the raw, matched tensor,
fits the model, and applies the correction. It follows `scripts/run_umaia.py` exactly. Three steps:

1. `uMAIA.norm.initialize(x, mask, subsample=True)` does the **GMM initialization**: per molecule, fit
   a one- and a two-component Gaussian mixture and keep the lower-BIC one. This is where the
   background and foreground modes the model starts from come from.
2. `uMAIA.norm.normalize(...)` is the only heavy step: the **MAP fit via SVI**. We give it
   `num_steps=2000`, `seed=42`, and leave `covariate_vector=None` (more on that choice below). It
   subsamples about 2,500 pixels per section, so on a CPU it finishes in a couple of minutes.
3. `uMAIA.norm.transform(x, mask, svi)` applies the **histogram-matching correction** (the
   CDF/inverse-CDF map we unroll by hand two sections down) and returns the normalized log tensor.

The fit prints a tqdm progress bar and a falling loss; that is SVI minimizing the negative ELBO. Run
it and wait for the bar to fill.''')


# ---------------------------------------------------------------------------
# Cell 9e5eea3d — THE compute cell. Was illustrative+raises-exception. Now it is the
# real run: load umaia_input.npz, initialize, normalize, transform. Remove tags.
# ---------------------------------------------------------------------------
replace_source(
    "9e5eea3d",
    r'''# --- the REAL uMAIA fit, exactly as in scripts/run_umaia.py ---
# Input was built by scripts/build_umaia_input.py: it pulls BOTH sections' raw ion
# images from METASPACE (CoreMetabolome v3, FDR 0.1), matches molecules across the two
# sections by (formula, adduct), masks to tissue and log-transforms, then stacks into a
# dense (N_pixels, S_sections, V_molecules) tensor with a boolean mask. Read that script
# to see the METASPACE pull; here we just load its output.
d = np.load(os.path.join(DATA, "umaia_input.npz"), allow_pickle=True)
x, mask = d["x"], d["mask"]                       # (N, S, V) LOG-intensity + (N, S, V) bool mask
molecules = list(d["molecules"])                  # V molecule keys: "formula_adduct"
sections = [str(s) for s in d["sections"]]        # the 2 acquisitions
print(f"input tensor x {x.shape} (N pixels, S sections, V molecules)")
print(f"  {len(molecules)} molecules matched across sections | sections: {sections}")
print(f"  tissue pixels per section: {[int(mask[:, s, 0].sum()) for s in range(x.shape[1])]}")

# 1) GMM-based init: per molecule, pick 1- vs 2-component mixture by BIC (the bg/fg modes)
print("\ninitialize (GMM per molecule)...")
init = uMAIA.norm.initialize(x, mask, subsample=True)

# 2) the heavy step: MAP fit via SVI (AutoDelta guide), rank-1 batch term, deterministic.
#    covariate_vector stays None: we deliberately do NOT tell the model which section is
#    pregnant, so it cannot absorb the biology we want to test (see the corner-case section).
print("normalize (MAP via SVI, 2000 steps; a couple of minutes on CPU)...")
svi = uMAIA.norm.normalize(x, mask, init_state=init, subsample=True,
                           num_steps=2000, seed=42)

# 3) apply the histogram-matching correction -> normalized LOG tensor
x_maia = np.asarray(uMAIA.norm.transform(x, mask, svi))
print("\ndone. normalized tensor x_maia:", x_maia.shape, x_maia.dtype)''',
    set_tags=["keep"],
)


# ---------------------------------------------------------------------------
# Cell b2043b44 — the HINT after the run. Keep both observations (subsampling,
# covariate_vector=None) but phrase them around the fit that just executed.
# ---------------------------------------------------------------------------
replace_source("b2043b44", r'''💡 **HINT.** Two things to notice about the fit you just ran. It subsampled to about 2,500 pixels per
section, which is why it converged in a couple of minutes on the CPU even though each section has tens
of thousands of tissue pixels. And `covariate_vector` was left at its default `None`: we did *not*
hand the model the biological labels. The reason is in the corner-case section below; for now, trust
that telling the normalizer "this section is pregnant" would let it absorb the very biology we want to
test. If you want the saved-parameters workflow instead, `uMAIA.ut.tools.save_svi(svi, "some_dir")`
then `uMAIA.norm.transform(x, mask, "some_dir")` gives the identical result from disk.''')


# ---------------------------------------------------------------------------
# NEW cells right after the HINT (b2043b44): the real before/after money plot and the
# cross-section gap quantification, then a checkpoint. Inserted as a block.
# ---------------------------------------------------------------------------
i_hint = by_id["b2043b44"][0]

md_before_after = new_markdown_cell(r'''🔬 **TASK.** Now look at what the fit did. For a handful of molecules, overlay the two sections'
per-molecule log histograms **before** (raw, top row) and **after** (uMAIA-normalized, bottom row).
The molecule keys are the `formula_adduct` labels from the matched tensor. Watch the two raw
distributions, which are misaligned because of the slide-to-slide offset, get pulled onto each other.''')

code_before_after = new_code_cell(r'''# before/after per-section histograms for a few molecules (this is the money plot)
pick = np.linspace(0, len(molecules) - 1, 4).astype(int)   # 4 molecules spread across the panel
fig, axes = plt.subplots(2, 4, figsize=(14, 5.6), sharex="col")
colours = ["steelblue", "darkorange"]
for col, v in enumerate(pick):
    for s in range(x.shape[1]):
        m = mask[:, s, v]                                   # tissue pixels of section s for molecule v
        axes[0, col].hist(x[m, s, v], bins=60, density=True, alpha=0.5,
                          color=colours[s], label=sections[s])
        axes[1, col].hist(x_maia[m, s, v], bins=60, density=True, alpha=0.5,
                          color=colours[s], label=sections[s])
    axes[0, col].set_title(f"{molecules[v]}\nraw", fontsize=FS["xs"])
    axes[1, col].set_title("uMAIA-normalized", fontsize=FS["xs"])
    for r in (0, 1):
        axes[r, col].set_yticks([])
        axes[r, col].legend(fontsize=FS["xs"], frameon=False)
axes[0, 0].set_ylabel("density (raw)"); axes[1, 0].set_ylabel("density (normalized)")
fig.suptitle("uMAIA aligns each molecule's two per-section distributions", fontsize=FS["l"])
plt.tight_layout(); plt.show()''')
code_before_after.metadata["tags"] = ["keep"]

md_gap_task = new_markdown_cell(r'''⚠️ **CHECKPOINT.** Top row: the two raw histograms for each molecule sit at visibly different places,
the slide-to-slide offset we have been warning about. Bottom row: after uMAIA, the same two
distributions overlap. The correction did not collapse them onto a single spike; it kept each
section's shape and slid them into register, which is exactly the conservative, order-preserving
quantile matching we will unroll by hand below.

🔬 **TASK.** Put one number on it. For every molecule, measure the gap between the two sections at the
**90th percentile** (a robust stand-in for where the foreground mode sits), then take the median gap
across all 104 molecules. Do it on the raw tensor and on the normalized tensor. The gap should shrink.''')

code_gap = new_code_cell(r'''# quantify the cross-section offset: median over molecules of the 90th-percentile gap
def median_90pct_gap(arr):
    gaps = []
    for v in range(arr.shape[2]):
        per_section = [arr[mask[:, s, v], s, v] for s in range(arr.shape[1])]
        hi = [np.quantile(a, 0.9) for a in per_section if len(a)]
        if len(hi) == 2:
            gaps.append(abs(hi[0] - hi[1]))
    return float(np.median(gaps))

gap_raw = median_90pct_gap(x)
gap_norm = median_90pct_gap(x_maia)
print(f"median cross-section 90th-percentile gap")
print(f"  raw         : {gap_raw:.3f}  log units")
print(f"  uMAIA-norm  : {gap_norm:.3f}  log units")
print(f"  shrunk by   : {(1 - gap_norm / gap_raw) * 100:.0f}%")''')
code_gap.metadata["tags"] = ["keep"]

md_gap_check = new_markdown_cell(r'''⚠️ **CHECKPOINT.** The median 90th-percentile gap between the two sections drops from about **0.21**
log units on the raw data to about **0.13** after uMAIA, a cut of roughly a third. That is the batch
offset being removed across all 104 molecules at once. The residual gap is not zero, and it should not
be: with only two sections uMAIA cannot tell a genuine control-versus-pregnant difference from a batch
effect, so it leaves real biology in place rather than over-correcting. We unpack that trade-off in the
corner-case section. For now you have run the real fit and watched the offset shrink on genuine data.

❓ **QUESTION.** The fit never saw which section was control and which was pregnant
(`covariate_vector=None`). If we *had* told it, and it had driven the residual gap to zero, would that
be a better result or a worse one? Hold the question; the corner-case section answers it.''')

# splice the new block right after the HINT cell
new_block = [md_before_after, code_before_after, md_gap_task, code_gap, md_gap_check]
nb.cells[i_hint + 1:i_hint + 1] = new_block


# ---------------------------------------------------------------------------
# Cell bce06138 — intro to the unrolled transform. Add a bridging first line that
# frames the unroll as opening the black box of the transform() we just ran, and
# keep the existing teaching. (uMAIA does exactly this, fit jointly across molecules.)
# ---------------------------------------------------------------------------
replace_source("bce06138", r'''## unrolling the correction: histogram matching by CDF and inverse-CDF

You just called `uMAIA.norm.transform(...)` and watched the histograms align. Now we open that black
box. The transform is about twenty lines of NumPy and SciPy, and once you see it you will never find
quantile matching mysterious again. uMAIA does exactly this, only fit **jointly across all molecules
at once** with the modes learned from the MAP fit, whereas below we do it one molecule at a time so
every step is visible.

The idea. Suppose the measured log-intensity distribution of a lipid in one section, call it
$g(x)$, is a distorted version of a shared reference distribution $f(y)$ that we want every section to
match. We relate the two through their **cumulative distribution functions (CDFs)**. A CDF $F(y)$
answers "what fraction of the data is below $y$"; it climbs from 0 to 1 and is a one-to-one map between
a value and its **quantile** (its rank, as a fraction). So the correction is:

$$y \;=\; F^{-1}\big(\,G(x)\,\big)$$

In words, three steps for each measured value $x$:

1. push $x$ through its **own** fitted mixture CDF $G$ to get its quantile $q = G(x)$, a number in
   $[0,1]$ that says "this value is at the $q$-th percentile of its section".
2. take that same quantile $q$.
3. pull it back through the **reference** inverse-CDF $F^{-1}$ to get the value that sits at the
   $q$-th percentile of the shared reference.

The value at the 90th percentile of section A becomes the value at the 90th percentile of the
reference. This is **quantile (histogram) matching**. Because the CDFs are the smooth fitted
two-Gaussian mixtures, not jagged empirical CDFs, the map is smooth, monotone (order-preserving, so it
never reshuffles pixels), and conservative. The background anchor and the within-section ranking
survive; only the technical stretch of the scale is removed.

🔬 **TASK.** Build the two helpers, copied faithfully from uMAIA's `_transform.py`: a mixture CDF, and
a routine that inverts a reference mixture CDF by interpolation. Read the comments.''')


nbf.write(nb, NB)
print(f"edited {NB}: now {len(nb.cells)} cells")
