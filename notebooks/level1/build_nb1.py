"""Assemble NB1 'mass spectra and the data' for the CAJAL spatial-metabolomics course.

Run me, then execute the produced notebook with nbconvert.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(src, tag="keep"):
    c = new_code_cell(src)
    c.metadata["tags"] = [tag]
    cells.append(c)


# ----------------------------------------------------------------------------
# TITLE
# ----------------------------------------------------------------------------
md(r"""# NB1: mass spectra and the data

Welcome to the first hands-on notebook of the spatial metabolomics primer. By the end of this session you will know what a MALDI mass spectrometry imaging experiment physically measures, what a brain lipid is and how it is named, how a measured mass becomes a lipid identity, and what the data looks like once it reaches your laptop. We will finish by drawing the very first maps of two lipids across a control and a pregnant mouse brain, on the same coronal plane, and you will read them yourself.

This notebook assumes zero programming background. Every step is unrolled in a few transparent lines so you see the mechanics, and only then do we point at the matching helper in our small course package `cajal_lipidomics`. Read the code, run it, change a number, run it again. The plots are the teaching.

What we will build, in order:

1. the physics: a laser, a tiny spot of tissue, a mass spectrometer, and one spectrum per pixel.
2. brain lipids: what a lipid is, why the brain is mostly lipid, the classes, and the naming `PC 38:6`.
3. a real mean MALDI spectrum, pulled live from the public data archive.
4. from a mass to a name: parts-per-million matching and adducts.
5. data formats, slowly: the METASPACE pull, zarr, the pandas dataframe and parquet, and AnnData as the smart container.
6. the course substrate: two sections, 174768 pixels, 173 lipids, fully annotated with anatomy.
7. first spatial maps: a myelin sphingolipid and a phospholipid, control versus pregnant.
""")

md(r"""## setup

We import the scientific Python stack and our course package. `numpy` handles arrays of numbers, `pandas` handles tables, `matplotlib` draws, `anndata` is the container that holds the imaging data, and `cajal_lipidomics` (imported as `cl`) is the small set of helpers we wrote for this course. We call `set_style()` once so every figure comes out clean and consistent. We also fix a random seed so anything random is reproducible.""")

code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad

import cajal_lipidomics as cl
from cajal_lipidomics import plotting, annotation, analysis
from cajal_lipidomics.style import set_style, FS
set_style()

# fix randomness so the notebook is reproducible
RNG = np.random.default_rng(0)
np.random.seed(0)

print("numpy", np.__version__)
print("course package cajal_lipidomics", cl.__version__)
""")

# ----------------------------------------------------------------------------
# SECTION 1 — WHAT IS MALDI-MSI
# ----------------------------------------------------------------------------
md(r"""## 1. what MALDI mass spectrometry imaging actually does

Imagine a thin slice of frozen mouse brain, about twelve micrometers thick, sitting flat on a glass slide. We spray it with a fine layer of a small organic molecule called a matrix, here a compound named DHB. The matrix soaks into the tissue and co-crystallizes with the molecules already there. Now we fire a laser at one tiny spot of that slide. The matrix absorbs the laser energy, heats up in a flash, and explodes off the surface, dragging the tissue molecules with it into the gas phase. This gentle blast is called desorption, and because it is helped by the matrix and a laser, the whole technique is named MALDI: Matrix-Assisted Laser Desorption/Ionization.

As the molecules leave the surface they pick up or lose a charge, so they become ions. An electric field then accelerates these ions into a mass spectrometer. The spectrometer does one thing supremely well: it sorts ions by their mass-to-charge ratio, written m/z. Heavy ions and light ions separate, and a detector counts how many ions arrive at each m/z. The output for that one laser spot is therefore a list: for every m/z value, how much signal we saw. That list is a mass spectrum.

Now the imaging part. Instead of firing once, the instrument rasters the laser across the whole slice on a regular grid, here every 25 micrometers, and records a full spectrum at each grid point. Each grid point is called a pixel. Stack the spectra back onto the grid and you have an image: at every pixel you know the intensity of every measured molecule. The pixel size, 25 micrometers here, is the spatial resolution: it is the finest spatial detail the experiment can resolve.""")

md(r"""### a pixel is not a cell

This is the single most important caveat to carry through the whole course. A 25 micrometer pixel is far larger than a single neuron's soma and contains a mixture of cell bodies, axons, dendrites, extracellular matrix, and glial projections. So a pixel reports the average lipid composition of a small piece of tissue, not the lipid composition of one cell. When we later see a lipid painting a brain region, we are seeing the bulk chemistry of that territory, not a single cell type. Keep this in mind every time you read a map.

⚠️ CHECKPOINT: in one sentence, say what physical thing the number stored at one pixel, one m/z, represents. (It is roughly: how many ions of that mass-to-charge were desorbed from that 25 by 25 micrometer patch of tissue.)""")

md(r"""### the acquisition we will use

The two brain sections in this course were both acquired on an Orbitrap mass spectrometer in positive ion mode, with a mass resolving power of R = 240000 at m/z 200, a pixel size of 25 by 25 micrometers, over a mass range of m/z 400 to 1600, with DHB matrix. That resolving power is what lets the instrument separate two ions whose masses differ by only a few thousandths of a Dalton, which is exactly what we will need when we try to tell similar lipids apart. Because both sections share one acquisition protocol, they are directly comparable, which is the whole point: one control brain and one pregnant brain, measured the same way, on the same coronal plane.""")

# ----------------------------------------------------------------------------
# SECTION 2 — BRAIN LIPIDS
# ----------------------------------------------------------------------------
md(r"""## 2. brain lipids: what they are and how they are named

A lipid is a small biological molecule that does not dissolve well in water but dissolves in oily solvents. The most important lipids for us are the ones that build membranes. A membrane lipid has two parts: one or two long fatty chains made of carbon and hydrogen, which hate water, and a small head group, which likes water. Pack millions of these molecules tail-to-tail and the water-hating tails huddle inside while the water-loving heads face the watery world on both sides. That is a lipid bilayer, the two-molecule-thick sheet that wraps every cell and every organelle.

The brain cares about lipids more than almost any other organ. After you remove the water, about half of the brain's dry weight is lipid. Lipids build the membranes of neurons and glia, they form the synaptic vesicles that store neurotransmitters, and above all they build myelin, the fatty insulation that oligodendrocytes wrap around axons so electrical signals travel fast. White matter is white precisely because it is dense with myelin lipid. So when we map lipids across the brain, we are mapping a chemistry that the brain spends enormous resources to build and maintain.""")

md(r"""### the classes we will meet

Membrane lipids come in families called classes, defined by their head group. The ones that dominate our data are:

- glycerophospholipids, built on a glycerol backbone with a phosphate and a head group:
  - PC, phosphatidylcholine: the most abundant membrane lipid, neutral head, the workhorse of the outer membrane.
  - PE, phosphatidylethanolamine: smaller head, abundant, helps membranes curve.
  - PS, phosphatidylserine: negatively charged, enriched on the inner membrane leaflet, a signal during cell death.
  - PI, phosphatidylinositol: negatively charged, a hub for cell signaling.
  - plus PA, PG, and the lyso forms LPC and LPE that carry a single chain.
- sphingolipids, built on a sphingosine backbone instead of glycerol, central to myelin:
  - SM, sphingomyelin: a phosphocholine head on a ceramide, abundant in myelin.
  - Cer, ceramide: the bare backbone, a signaling lipid and a building block.
  - HexCer, hexosylceramide: a ceramide with a sugar attached, a hallmark of myelin and a marker we will return to again and again.

These few classes already account for most of the brain's membrane lipid mass, and they are most of the 173 lipids in our dataset.""")

md(r"""### reading a lipid name like `PC 38:6`

Lipid shorthand packs a lot into a few characters. Read `PC 38:6` as three pieces:

- `PC` is the class, here phosphatidylcholine.
- `38` is the total number of carbon atoms summed across both fatty chains.
- `6` is the total number of carbon-carbon double bonds, also called unsaturations, summed across both chains.

So `PC 38:6` is a phosphatidylcholine whose two tails together carry 38 carbons and 6 double bonds. A sphingolipid name like `HexCer 42:2;O2` adds a `;O2` suffix that records two oxygens on the backbone, a detail of the ceramide hydroxylation. One subtlety: MALDI measures a mass, and a mass cannot tell apart a `18:0 / 20:6` arrangement from a `16:0 / 22:6` arrangement, because both sum to `38:6`. So a single name is really a sum composition that may collapse several true molecules, called isomers, into one entry. We will see the consequence of this ambiguity in the annotation step.

🔬 TASK: below we write a tiny parser that turns a lipid name into its class, carbon count, and double-bond count using regular expressions. This is exactly the recipe the Lipid Brain Atlas uses to color and group lipids. Read it, then run it on the dataset.""")

code(r"""import re

def parse_lipid_name(name):
    '''Split a lipid shorthand into (class, carbons, double_bonds).

    'PC 38:6'        -> ('PC', 38, 6)
    'HexCer 42:2;O2' -> ('HexCer', 42, 2)
    The class is the text before the first space or parenthesis; the carbons are the
    number before the colon; the double bonds are the number right after the colon.
    '''
    lipid_class = re.split(r"[ (]", name)[0]          # text before first space/paren
    carbons = re.search(r"(\d+):", name)              # number before the colon
    doublebonds = re.search(r":(\d+)", name)          # number after the colon
    carbons = int(carbons.group(1)) if carbons else np.nan
    doublebonds = int(doublebonds.group(1)) if doublebonds else np.nan
    return lipid_class, carbons, doublebonds

# try it on a few names
for nm in ["PC 38:6", "HexCer 42:2;O2", "SM 36:1;O2", "PE 40:6"]:
    print(f"{nm:18s} -> {parse_lipid_name(nm)}")
""")

md(r"""💡 HINT: the helper `cl.data.lipid_properties` does exactly this parsing over a whole list of names at once and returns a tidy table indexed by lipid name, with columns for the class, total carbons, double bonds, oxygens, an ether flag, and a per-class color ready for plotting. We unroll the regex here so you can see there is no magic: a lipid name is just a class string and a few integers, and a regular expression pulls them out.""")

# ----------------------------------------------------------------------------
# SECTION 3 — A REAL MEAN MALDI SPECTRUM
# ----------------------------------------------------------------------------
md(r"""## 3. a spectrum is two aligned arrays

We said a spectrum is a list of m/z values and their intensities. Concretely, it is two arrays of the same length, lined up index by index: `mz[0]` has intensity `intensity[0]`, `mz[1]` has intensity `intensity[1]`, and so on. To draw it we plant a vertical stick at each m/z whose height is its intensity. That stick picture is the canonical mass spectrum.

We will not invent fake numbers. Instead we pull a real section straight from METASPACE, the public archive that hosts the Lipid Brain Atlas data, and compute the mean intensity of every annotated peak across the tissue. That mean-per-peak is itself a spectrum: the average mass spectrum of the whole brain section. METASPACE is a web service, so this cell talks to the internet; the small retry loop below simply tries again if a download hiccups, which occasionally happens with image transfers.""")

code(r"""from metaspace import SMInstance
import time

# the public, login-free project behind the Lipid Brain Atlas
PROJECT_ID = "9ab478cc-013a-11ed-89bf-07e42afac14e"   # mlba-2025 / MouseBrainAtlas
DB = ("CoreMetabolome", "v3")                          # the annotation database we trust
FDR = 0.1                                              # keep annotations at <=10% false discovery
PREGNANT_ID = "2024-07-14_14h24m11s"                   # Brain1_C2, our pregnant section
CONTROL_ID  = "2025-04-27_08h23m47s"                   # 217D, our control section


def with_retry(fn, tries=4, pause=3.0):
    '''Call fn(); on a transient network/PNG error wait and try again.'''
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as err:          # noqa: BLE001 (we genuinely want any transient error)
            last = err
            print(f"  attempt {attempt + 1} failed ({type(err).__name__}); retrying...")
            time.sleep(pause)
    raise last


sm = SMInstance()                                      # anonymous: this project is public
ds = with_retry(lambda: sm.dataset(id=PREGNANT_ID))
print("dataset name:", ds.name)
""")

md(r"""The dataset name encodes everything: the date, that it is the LipidAtlas, that it is `Pregnant`, the brain and section `Brain1_C2`, and the image size `459x352` in pixels. Now we pull the annotation table. Each row is one detected ion, indexed by its chemical formula and its adduct, with the measured m/z, a quality score `msm`, and the false discovery rate `fdr`.""")

code(r"""# the annotation table: one row per detected ion
res = with_retry(lambda: ds.results(database=DB, fdr=FDR))
print("annotated ions:", len(res))
print("columns:", list(res.columns))
res[["mz", "msm", "fdr"]].head(6)
""")

md(r"""Now the images. `all_annotation_images` returns, for each annotated ion, a 2D intensity image over the tissue grid. We ask for the monoisotopic peak only (`only_first_isotope=True`, matching the published quality control) and raw intensities (`scale_intensity=False`, so values stay comparable). We flatten each image to a 1D vector of pixels and stack them, giving a pixels-by-ions matrix. Then we average over pixels to get the mean intensity of each ion: that vector, paired with the m/z column, is the mean spectrum.""")

code(r"""# one 2D ion image per annotation, then stack flattened images -> pixels x ions
img_sets = with_retry(lambda: ds.all_annotation_images(
    fdr=FDR, database=DB, only_first_isotope=True,
    scale_intensity=False, hotspot_clipping=False))

images = [np.asarray(s[0]) for s in img_sets]          # each is a (height, width) image
H, W = images[0].shape
X_metaspace = np.stack([np.nan_to_num(im.flatten()) for im in images], axis=1).astype("float32")
print("one ion image is", (H, W), "pixels =", H * W, "flattened")
print("stacked matrix (pixels x ions):", X_metaspace.shape)

# the mean spectrum: average each ion over all pixels
mz_axis = res["mz"].to_numpy()
mean_spectrum = X_metaspace.mean(axis=0)
print("mean spectrum has", len(mz_axis), "sticks (one per annotated ion)")
""")

md(r"""Now we draw it with `cl.plotting.spectrum`, which simply plants a vertical stick at each m/z with height equal to its mean intensity. This is the average MALDI mass spectrum of a whole pregnant brain section, built from real measured numbers.""")

code(r"""ax = plotting.spectrum(mz_axis, mean_spectrum,
                       title="mean MALDI spectrum, pregnant section Brain1_C2")
plt.show()
""")

md(r"""Read this figure. The horizontal axis is mass-to-charge, running from about 400 to 1060. Each stick is one lipid ion that METASPACE confidently annotated, and its height is how much of that ion the section contains on average. The tallest sticks crowd between roughly m/z 700 and 900, which is exactly where the abundant glycerophospholipids and sphingolipids of brain membranes sit. Notice how many sticks there are and how close together some of them stand. That crowding is the heart of the annotation problem: when two real lipids have nearly the same mass, their sticks nearly overlap, and we need very precise mass measurement to tell them apart.

❓ QUESTION: the brief says METASPACE annotated this section with 233 ions, yet our course substrate keeps 173 lipids. Why fewer? Because the published atlas applies stricter quality control and removes redundant adducts and noisy peaks. We will explain adducts next, and the full feature selection appears in a later notebook.""")

md(r"""### zoom into a crowded window

Let us zoom the spectrum into a narrow mass window to see the crowding directly. We keep only the sticks between m/z 798 and 812 and replot. Several lipids live in this small band, and their masses differ by less than one Dalton.""")

code(r"""window = (mz_axis >= 798) & (mz_axis <= 812)
ax = plotting.spectrum(mz_axis[window], mean_spectrum[window],
                       title="zoom on m/z 798-812: several lipids, nearly the same mass", lw=2.0)
for m, v in zip(mz_axis[window], mean_spectrum[window]):
    ax.annotate(f"{m:.3f}", (m, v), ha="center", va="bottom", fontsize=FS["xs"], rotation=90)
plt.show()
""")

md(r"""Each labeled stick is a distinct ion within fourteen mass units of its neighbors. To turn any one of these masses into a lipid name, we need a rule for matching a measured mass to a reference mass, and we need to handle the fact that one lipid can show up at several masses at once. That is the next section.""")

# ----------------------------------------------------------------------------
# SECTION 4 — FROM MASS TO NAME: ppm AND ADDUCTS
# ----------------------------------------------------------------------------
md(r"""## 4. from a mass to a name: parts-per-million matching and adducts

MALDI hands us a mass, never an identity. To guess the lipid behind an m/z we compare it against a reference list of known lipids and their exact masses, and we accept a match only when the two masses agree closely enough. Closely enough is defined in parts per million, ppm, a relative error:

$$ \text{ppm error} = 10^6 \times \frac{|\text{observed} - \text{reference}|}{\text{reference}} $$

We use a 5 ppm tolerance, the community standard for this kind of high-resolution data. At a mass of 800, 5 ppm is a window of only $5/10^6 \times 800 = 0.004$ Dalton, four thousandths of one mass unit. That is how precise the Orbitrap is, and that precision is what lets us discriminate most, though not all, lipids of similar mass.""")

code(r"""# the ppm formula, exactly as in cl.annotation.ppm_error
def ppm_error(observed, reference):
    return 1e6 * abs(observed - reference) / reference

# what does 5 ppm mean as an absolute window at a few masses?
for m in [400, 800, 1000]:
    window_da = 5 / 1e6 * m
    print(f"at m/z {m:5d}: a 5 ppm window is +/- {window_da:.4f} Da")
""")

md(r"""### adducts: one lipid, several masses

Here is the twist. A reference database lists the neutral mass of a lipid, the mass of the bare molecule. But the mass spectrometer only ever sees charged ions, and a neutral lipid becomes charged by grabbing a small ion. In positive mode it can grab a proton (`H+`), a sodium (`Na+`), a potassium (`K+`), or an ammonium (`NH4+`). Each of these adds its own little mass. So the same neutral lipid appears in the spectrum at four different m/z values, one per adduct, each shifted by the adduct's mass. To match an observed peak, we therefore subtract each adduct's mass to recover a candidate neutral mass, and check whether any of those candidates lands within 5 ppm of a reference lipid.""")

code(r"""# positive-mode adduct masses in Daltons (this is cl.annotation.ADDUCTS)
ADDUCTS = {"H+": 1.007276, "Na+": 22.989769, "K+": 38.963707, "NH4+": 18.033823}
for name, mass in ADDUCTS.items():
    print(f"adduct {name:4s} adds {mass:9.5f} Da")
""")

md(r"""### a real match, made visible

Our course ships an in-house LC-MS reference list, a separate liquid-chromatography mass spectrometry experiment that confidently identified lipids and recorded the exact m/z of each lipid with each adduct already attached. We load it, then ask: for a chosen observed MSI peak, which reference lipids fall inside the 5 ppm window? The helper `cl.annotation.plot_ppm_match` draws this as two stacked stick spectra sharing the mass axis. The top panel shows the reference peaks near our observed mass, the bottom panel shows the single MSI peak, the shaded gold band is the 5 ppm acceptance window, and each reference is labeled with its name and its ppm distance. References inside the band are accepted matches, drawn green; those outside are rejected, drawn gray.""")

code(r"""# load the in-house LC-MS reference: columns Lipid, Adduct, m/z (already ionized)
lcms = annotation.load_lcms_reference("../../data/refs/csv/lcms_mar2022_withcounterions (2).txt")
print("LC-MS reference rows:", len(lcms))
lcms.head(4)
""")

code(r"""# pick a real MSI peak and show which reference lipids match it within 5 ppm.
# 806.5723 is a measured peak: PC 38:6 protonated lands 3.6 ppm away, PE 41:6 protonated 4.6 ppm.
observed = 806.5723
axL, axM = annotation.plot_ppm_match(observed, lcms, ppm_tol=5.0, span_ppm=40)
plt.show()
""")

md(r"""This one figure teaches the whole annotation idea. The MSI peak at m/z 806.5723 sits inside the gold band of `PC 38:6` protonated, only 3.6 ppm away, so it is accepted and donates its name to the peak. But look again: `PE 41:6` protonated is right next door at 4.6 ppm, also inside the 5 ppm band. Both candidates qualify, so we have an isobaric tie: two real lipids competing for one mass, both passing the same window. This is precisely why a single name is a best guess and why the atlas uses extra evidence, such as which candidate is more abundant by bulk LC-MS, to break ties. We dedicate a whole later notebook to annotation; here the point is simply that a number on the mass axis becomes a lipid name through a tiny, transparent window, and that the window is sometimes wide enough to admit more than one answer.""")

code(r"""# the matcher behind the plot: which references fall inside the window, sorted by closeness
hits = annotation.match_lcms(observed, lcms, ppm_tol=5.0)
print("reference lipids within 5 ppm of", observed, ":")
hits[["Lipid", "Adduct", "m/z", "ppm"]]
""")

md(r"""🔬 TASK: change `observed` above to `828.5522`, the sodiated form of the same lipid (`PC 38:6` plus `Na+` instead of `H+`), and rerun the plot and the matcher. You should see the same lipid recovered through a different adduct, which is exactly the point of adducts: one lipid, several masses.""")

# ----------------------------------------------------------------------------
# SECTION 5 — DATA FORMATS
# ----------------------------------------------------------------------------
md(r"""## 5. data formats, slowly

We have now met a spectrum, a peak, and an annotation. Next we need to understand the containers that hold all of this on disk and in memory, because spatial metabolomics data is large and you will meet three formats again and again: zarr, the pandas dataframe stored as parquet, and AnnData. We build up to AnnData, the smart container we use for everything.""")

md(r"""### the METASPACE pull, unrolled

We already did the essential work above. Let us name the three steps explicitly, because this is the recipe for turning any imaging dataset into a single tidy object:

1. `ds.results(...)` gives the annotation table: this defines the features, one row per detected ion, with its mass and identity.
2. `ds.all_annotation_images(...)` gives one 2D image per feature: this defines the pixels and their intensities.
3. stack the flattened images into a pixels-by-features matrix, attach the feature table and the pixel coordinates, and wrap it all in one object.

We package those three steps into a small function, with the same retry logic, and run it on the pregnant section we already downloaded so you can see the resulting object. We also pull the control section the same way, so later we have both as freshly built objects.""")

code(r"""def pull_section(dataset_id, database=DB, fdr=FDR, scale_intensity=False):
    '''Pull one METASPACE section into an AnnData: pixels x ions, with feature metadata.

    This is the transparent version of cl.io.pull_metaspace_section: results -> images ->
    stack -> AnnData. Returns the smart container described below.
    '''
    ds = with_retry(lambda: sm.dataset(id=dataset_id))

    # 1) features = annotated ions
    res = with_retry(lambda: ds.results(database=database, fdr=fdr))

    # 2) one 2D image per feature (monoisotopic peak, raw intensity)
    img_sets = with_retry(lambda: ds.all_annotation_images(
        fdr=fdr, database=database, only_first_isotope=True,
        scale_intensity=scale_intensity, hotspot_clipping=False))
    images = [np.asarray(s[0]) for s in img_sets]
    H, W = images[0].shape

    # 3) stack flattened images -> X is (n_pixels, n_features)
    X = np.stack([np.nan_to_num(im.flatten()) for im in images], axis=1).astype("float32")

    # feature (var) table from the (formula, adduct) index
    formulas = [ix[0] for ix in res.index]
    adducts = [ix[1] for ix in res.index]
    var = pd.DataFrame({
        "formula": formulas, "adduct": adducts,
        "mz": res["mz"].values, "msm": res["msm"].values, "fdr": res["fdr"].values,
        "moleculeNames": [",".join(map(str, m)) for m in res["moleculeNames"]],
    })
    var.index = pd.Index([f"{f}_{a}" for f, a in zip(formulas, adducts)])

    # pixel coordinates in row-major order (y outer, x inner)
    ys, xs = np.divmod(np.arange(H * W), W)
    obs_names = pd.Index([f"x{int(x)}_y{int(y)}" for x, y in zip(xs, ys)])

    out = ad.AnnData(X=X)
    out.obs_names = obs_names
    out.var = var
    out.var_names = var.index
    out.obsm["spatial"] = np.column_stack([xs, ys]).astype(np.int32)
    out.uns["img_shape"] = [int(W), int(H)]
    out.uns["dataset_id"] = dataset_id
    return out


adata_pregnant_raw = pull_section(PREGNANT_ID)
print(adata_pregnant_raw)
""")

md(r"""Look at what we got. The object prints its shape, here pixels by ions, and lists the per-pixel metadata it carries (`obs`), the per-feature metadata (`var`), the spatial coordinates (`obsm['spatial']`), and the dataset notes (`uns`). This is an AnnData, the smart container. Before we explore it fully, let us understand why the data even needs special formats.""")

md(r"""### zarr: how big imaging data is stored on disk

A single brain section here is hundreds of pixels wide and tall, times hundreds to thousands of measured masses, and the full atlas spans millions of pixels across 109 sections. If you tried to load all of that into memory at once your laptop would choke. The atlas therefore stores its processed imaging data as zarr.

Zarr is a format for very large array data that splits an array into many small compressed chunks on disk, organized in a folder tree. You can open the store without loading anything, then read just the chunks you need, for example one lipid in one section, and only those chunks are decompressed into memory. In the Lipid Brain Atlas the zarr store is keyed by peak, then by section, so `store["800.553832"][section_index]` returns just the 2D image of that one peak in that one section. We will not parse zarr live in this course, because the data reaches you already processed, but the mental model matters: zarr is the on-disk warehouse for imaging arrays too big to hold all at once.""")

md(r"""### pandas dataframe and parquet: tables of pixels

Once we extract pixels and their values, the natural shape is a table: one row per pixel, columns for each lipid plus columns for metadata like the section, the spatial coordinates, and the brain region. A pandas DataFrame is exactly that, a labeled table you can filter, group, and join. The atlas keeps the full per-pixel table, with every lipid and every coordinate, as a file called `maindata_2`, stored in the parquet format.

Parquet is a columnar file format: instead of writing the table row by row, it writes each column together and compresses it. That makes it small on disk and fast to read when you only want a few columns, for example just the spatial coordinates and one lipid. We will read columns from the big atlas table later; for now, picture the per-pixel data as one enormous spreadsheet saved efficiently as parquet.""")

md(r"""### AnnData: the smart container we actually use

A bare DataFrame mixes measurements and metadata in one flat table, which gets awkward fast. AnnData (annotated data) is the container built for exactly this kind of data, and it keeps four things cleanly separated around one central matrix:

- `.X`: the core matrix, pixels by features. Rows are pixels, columns are lipids, each entry is an intensity.
- `.obs`: a table of per-pixel metadata, one row per pixel, aligned to the rows of `.X`. This holds the condition, the section, the brain region, the coordinates.
- `.var`: a table of per-feature metadata, one row per lipid, aligned to the columns of `.X`. This holds the lipid name, mass, and class.
- `.obsm`: matrices of per-pixel vectors, for example the 2D spatial coordinates or a low-dimensional embedding.
- `.uns`: a free-form bag for everything else, such as notes about the dataset.

The power is that `.X`, `.obs`, and `.var` stay aligned automatically: if you keep only the pregnant pixels, AnnData subsets `.X` and `.obs` together so they never fall out of step. Let us inspect the three faces of the object we just built.""")

code(r"""print("X, the core matrix (pixels x ions):", adata_pregnant_raw.X.shape, adata_pregnant_raw.X.dtype)
print()
print("var, per-feature metadata (first 5 ions):")
print(adata_pregnant_raw.var.head(5).to_string())
print()
print("obsm['spatial'], per-pixel coordinates (first 5 pixels):")
print(adata_pregnant_raw.obsm["spatial"][:5])
""")

# ----------------------------------------------------------------------------
# SECTION 6 — LOAD THE COURSE SUBSTRATE
# ----------------------------------------------------------------------------
md(r"""## 6. the course substrate: two matched sections, fully annotated

The object we pulled from METASPACE has raw ion intensities and only the chemical formula per feature. For the rest of the course we use a prepared substrate, `sections_pair.h5ad`, that is already cleaned and enriched: the intensities are normalized so they are comparable across sections, the features are named lipids, and every pixel is tagged with its brain region from the Allen atlas and its data-driven lipid territory. It holds the two matched sections, one control and one pregnant, on the same coronal plane. We load it with one line.""")

code(r"""adata = ad.read_h5ad("../../data/sections_pair.h5ad")
print(adata)
""")

md(r"""The shape tells the story: 174768 pixels and 173 lipids. The `.X` matrix is the uMAIA-normalized intensity of each lipid at each pixel, and the `.var_names` are real lipid names rather than masses. uMAIA is the normalization tool the atlas uses to make intensities comparable across sections; we treat its output as our starting data here and explain its internals in a later notebook. Let us confirm the two conditions and look at how the pixels split.""")

code(r"""print("conditions:")
print(adata.obs["Condition"].value_counts())
print()
print("sections (SectionID) and their sample names:")
print(adata.obs.groupby("SectionID", observed=True)["Sample"].agg(["first", "count"]))
""")

md(r"""So `naive` is the control brain (sample `Female1`, section 217D) with 84321 pixels, and `pregnant` is the pregnant brain (sample `Pregnant1`, section Brain1_C2) with 90447 pixels. Both are female, and crucially both sit at the same coronal plane, roughly anterior-posterior 6.5 in atlas coordinates, so any difference we find later is a difference between the conditions and not between different parts of the brain.""")

md(r"""### exploring the per-pixel metadata

The `.obs` table is where the richness lives. Let us list its columns, then look at the few that we will use constantly: the common-coordinate-framework coordinates `xccf`, `yccf`, `zccf` that register every pixel into the standard Allen brain, the `acronym` that names the Allen region each pixel falls in, the `division` of gray versus white matter, and the data-driven `lipizone_names`, the lipid territories discovered by clustering. We do not need to understand lipizones yet; we just note they are there and that there are many of them.""")

code(r"""# the metadata columns available per pixel
key_cols = ["Condition", "SectionID", "Sample", "xccf", "yccf", "zccf",
            "acronym", "division", "lipizone_names", "subclass_name"]
print("a selection of the per-pixel metadata columns:")
adata.obs[key_cols].head(5)
""")

code(r"""print("CCF coordinate ranges (atlas-registered, in mm-like units):")
spans = {}
for c in ["xccf", "yccf", "zccf"]:
    lo, hi = adata.obs[c].min(), adata.obs[c].max()
    spans[c] = hi - lo
    print(f"  {c}: {lo:.2f} to {hi:.2f}  (span {hi - lo:.2f})")
print()
print(f"xccf (anterior-posterior) spans only {spans['xccf']:.2f} units, against "
      f"{spans['yccf']:.2f} for yccf and {spans['zccf']:.2f} for zccf.")
print("So all pixels sit in essentially one coronal plane; the small ~0.9 unit AP spread")
print("is registration jitter and a slight tilt of the slice, not a second plane.")
print()
print("number of distinct Allen regions present:", adata.obs["acronym"].nunique())
print("most common regions:")
print(adata.obs["acronym"].value_counts().head(6))
print()
print("number of distinct lipid territories (lipizones):", adata.obs["lipizone_names"].nunique())
""")

md(r"""The control and pregnant sections together touch 174 Allen brain regions, the most populous being the caudoputamen (`CP`) and the piriform cortex (`PIR`). The clustering split the lipid signal into hundreds of fine territories, the lipizones, which later notebooks will build from scratch. For now this confirms the substrate is fully wired to anatomy: every pixel knows where it is in the standard brain and which lipid territory it belongs to.

⚠️ CHECKPOINT: explain in your own words what each of `.X`, `.obs`, and `.var` holds for this object, and why subsetting to `pregnant` pixels must change `.X` and `.obs` together but leave `.var` alone. (`.var` is per-lipid, and we keep all 173 lipids when we drop pixels.)""")

# ----------------------------------------------------------------------------
# SECTION 7 — FIRST SPATIAL MAPS
# ----------------------------------------------------------------------------
md(r"""## 7. first spatial maps: read the brain in lipids

Now the payoff. Each pixel carries its registered coordinates, so we can place every pixel at its true position and color it by the intensity of a chosen lipid. That is a spatial map, and it is the most direct way to see lipid chemistry painting anatomy. We plot with `cl.plotting.spatial_lipid`, which draws one panel per section, places each pixel at its atlas coordinates, colors it by the lipid's intensity on a shared scale so the two panels are comparable, and lays a faint grayscale anatomy underneath for context. We start with a myelin marker.""")

md(r"""### a myelin sphingolipid: HexCer 42:2;O2

`HexCer 42:2;O2` is a hexosylceramide, a sphingolipid with a sugar head, and it is a textbook marker of myelin. Oligodendrocytes pack it into the myelin sheaths they wrap around axons, so it should light up the white matter, the fiber tracts that carry signals between regions. If our maps are honest, this lipid should trace the brain's wiring.""")

code(r"""myelin_lipid = "HexCer 42:2;O2"
plotting.spatial_lipid(adata, myelin_lipid)
plt.suptitle(f"{myelin_lipid}: a myelin sphingolipid", y=1.02, fontsize=FS["l"])
plt.show()
""")

md(r"""Read the two panels side by side. The left is the control brain, the right is the pregnant brain, both the same coronal plane. The bright yellow streaks are the white matter fiber tracts, for example the corpus callosum arcing over the top and the internal capsule cutting through the middle, exactly where myelin is dense. The darker purple interior is gray matter, where cell bodies dominate and myelin is sparse. This is the single most important division in brain lipidomics: gray matter versus white matter, written in the sphingolipids of myelin. The control and pregnant patterns look broadly alike here, which is reassuring, because the gross wiring of the brain does not reorganize in pregnancy; the subtle shifts come later and need statistics to see.""")

md(r"""### a phospholipid: PC 38:6

Now a glycerophospholipid, `PC 38:6`, a phosphatidylcholine. PCs are the workhorse membrane lipid of every cell, so unlike the myelin marker we expect this one to favor the gray matter, the cell-body-rich interior, rather than the white tracts. Plot it the same way and compare the spatial pattern to the myelin map.""")

code(r"""phospholipid = "PC 38:6"
plotting.spatial_lipid(adata, phospholipid)
plt.suptitle(f"{phospholipid}: a gray-matter phospholipid", y=1.02, fontsize=FS["l"])
plt.show()
""")

md(r"""Notice the pattern has flipped. Where `HexCer 42:2;O2` lit the white tracts, `PC 38:6` brightens the gray matter interior and the cortical sheet, and leaves the fiber tracts comparatively dark. Two lipids, two opposite maps, because they belong to two different compartments of the brain: one builds myelin, the other builds the membranes of cell-body-rich regions. This contrast is the whole promise of spatial metabolomics, that the chemical identity of a lipid predicts where in the brain you find it.

🔬 TASK: pick any other lipid from `adata.var_names` and map it. Try a sphingomyelin like `SM 36:1;O2` (myelin-leaning) and a phosphatidylserine like one of the `PS` entries, and decide for each whether it favors gray or white matter. Use the cell below.""")

code(r"""# list a few lipids of each class to choose from
names = list(adata.var_names)
for cls in ["PC", "PE", "PS", "SM", "HexCer", "Cer"]:
    members = [n for n in names if n.split(" ")[0] == cls][:4]
    print(f"{cls:7s}: {members}")
""", tag="keep")

code(r"""# your turn: replace the lipid name and read the map
my_lipid = "SM 36:1;O2"
plotting.spatial_lipid(adata, my_lipid)
plt.suptitle(my_lipid, y=1.02, fontsize=FS["l"])
plt.show()
""", tag="task")

md(r"""### quantifying gray versus white at a glance

We can put a number on the impression. The helper `cl.analysis.myelination_score` averages the z-scored sphingolipid intensities per pixel, a simple myelination proxy. White-matter pixels should score high, gray-matter pixels low. We compute it and color the pixels by it, which should reproduce the white-matter skeleton we saw in the HexCer map, now built from all sphingolipids at once.""")

code(r"""# myelination proxy: mean z-scored sphingolipid (HexCer/Cer/SM) intensity per pixel
adata.obs["myelination"] = analysis.myelination_score(adata)

secs = sorted(adata.obs["SectionID"].unique())
fig, axes = plt.subplots(1, len(secs), figsize=(4.2 * len(secs), 3.6))
for ax, s in zip(np.atleast_1d(axes), secs):
    m = (adata.obs["SectionID"] == s).to_numpy()
    sc = ax.scatter(adata.obs.loc[m, "zccf"], -adata.obs.loc[m, "yccf"],
                    c=adata.obs.loc[m, "myelination"], cmap="RdBu_r",
                    s=4, vmin=-1.5, vmax=1.5, rasterized=True)
    cl.style.spatial_axes(ax)
    ax.set_title(str(adata.obs.loc[m, "Condition"].iloc[0]), fontsize=FS["m"])
cl.style.lightweight_colorbar(sc, list(np.atleast_1d(axes)), label="myelination score")
plt.suptitle("sphingolipid myelination score: white matter stands out in red", y=1.02, fontsize=FS["l"])
plt.show()
""")

md(r"""Red marks high sphingolipid, blue marks low, and the red lattice is the white matter, just as the single HexCer map suggested. We have gone from a physical laser shot, to a spectrum, to a named lipid, to a map, to a quantitative score of brain chemistry, all on real measured data.""")

# ----------------------------------------------------------------------------
# CHECKPOINT + WHAT'S NEXT
# ----------------------------------------------------------------------------
md(r"""## checkpoint and what comes next

You now hold the foundations of the whole course:

- MALDI mass spectrometry imaging fires a laser at a 25 micrometer tissue spot, the spectrometer sorts the desorbed ions by mass-to-charge, and each spot becomes a pixel holding one spectrum. A pixel is a patch of tissue, not a cell.
- a spectrum is two aligned arrays, m/z and intensity, drawn as sticks. We plotted a real mean MALDI spectrum of a pregnant brain section.
- brain lipids are mostly membrane lipids, the brain is about half lipid by dry weight, and the name `PC 38:6` reads as class, total carbons, total double bonds, with isomers collapsed.
- a mass becomes a name by matching against a reference within a 5 ppm window, after accounting for adducts that put one lipid at several masses. Isobaric ties are real and need extra evidence to resolve.
- the data lives as zarr on disk for huge arrays, as parquet for per-pixel tables, and as AnnData in memory, the smart container with aligned `.X`, `.obs`, and `.var`.
- our substrate is two matched sections, control and pregnant, on the same coronal plane, 174768 pixels by 173 lipids, fully registered to the Allen atlas.
- the first maps already separate gray matter from white matter: a myelin sphingolipid traces the fiber tracts, a phosphatidylcholine fills the gray matter, and a sphingolipid score makes the white-matter skeleton quantitative.

⚠️ CHECKPOINT before you move on: can you (1) state what one number in `.X` means physically, (2) parse `HexCer 42:2;O2` into class, carbons, double bonds, and oxygens, (3) explain why `PC 38:6` appears at four different m/z values, and (4) predict whether a sphingolipid or a phosphatidylcholine will be brighter in the corpus callosum?

What comes next. In the following notebooks we open up the steps we treated as given here: how uMAIA normalizes the intensities so sections are comparable, how feature selection keeps only the spatially meaningful lipids, how we compress 173 lipids into a handful of interpretable factors, how clustering discovers the lipid territories called lipizones, and finally how we test, with Wilcoxon statistics, which lipids truly change between the control and the pregnant brain. Every one of those steps will be unrolled in transparent code first, exactly as we did today.""")

# ----------------------------------------------------------------------------
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb["metadata"]["language_info"] = {"name": "python"}

out_path = "/home/fusar/lipidomics_tutorial_cajalcourse/notebooks/level1/01_mass_spectra_and_data_solution.ipynb"
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    nbf.write(nb, f)
print("wrote", out_path, "with", len(cells), "cells")
