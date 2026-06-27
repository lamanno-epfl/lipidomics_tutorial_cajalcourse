"""Rebuild NB4 'anatomy and the Allen atlas' from the real 03_normalized stage."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(s): cells.append(new_markdown_cell(s))
def co(s): cells.append(new_code_cell(s))

md("""# 04 · anatomy: registration and the Allen atlas

So far every pixel has been a little spectrum floating in space. We know its lipids, we know
which section it came from, we even know where it sits on the slide. What we still do not know
is the thing a neuroscientist asks first: where in the brain is it? Is this pixel in the
caudoputamen or the hippocampus, in grey matter or in a white-matter tract, in cortical layer 1
or layer 6? Without that label a lipid map is a pretty picture. With it, the map becomes
anatomy, and anatomy is what lets us compare control against pregnant region by region, and
later join our lipids to gene expression in the exact same regions.

This notebook is about how every pixel gets an anatomical address. The procedure is called
registration to a reference atlas, and the reference is the Allen Mouse Brain Common Coordinate
Framework, version 3 (CCFv3). We will:

1. understand what a reference atlas is, and why we need a per-pixel anatomical coordinate,
2. understand registration as an affine transform plus a nonlinear warp, built from a little
   linear algebra you can hold in your head,
3. meet ABBA, the community-standard tool, as a concept: it is the one piece of infrastructure
   we ship pre-computed rather than run live,
4. load the per-pixel CCF coordinates and region labels that already live on our two sections,
5. make the coordinate-to-region lookup completely explicit, and read the slicing angle off the data,
6. draw region maps for control and pregnant, split grey from white matter, and check that a
   myelin lipid traces the white-matter tracts on its own,
7. build region-level lipid views, the mean lipid profile of every Allen region, which is the
   table that powers the region-level differential test and the gene join in later notebooks.

The whole notebook runs on the two sections you already normalized in notebook 3: a control
female and a pregnant female, cut at the same coronal plane near AP 6.5.""")

md("""## the callouts

The same four markers run through every notebook:

- 🔬 **TASK**: something you do (write or run code).
- 💡 **HINT**: a nudge when you are stuck.
- ❓ **QUESTION**: pause and think; no code required.
- **check:** what you should see if it worked. If your screen disagrees, stop and fix it.

🔬 **TASK.** Run the next cell to load the libraries and our helper package. `cajal_lipidomics`
is the small course package whose plotting and analysis functions we lean on, after we have
unrolled the idea behind each one by hand.""")

co("""# the stack you know, plus our course helper package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad

import cajal_lipidomics as cl
from cajal_lipidomics import plotting, analysis
from cajal_lipidomics.style import set_style, FS
set_style()  # the course-wide clean figure style

# one global seed so every number and figure below is reproducible
RNG_SEED = 0
np.random.seed(RNG_SEED)

print("ready. cajal_lipidomics", cl.__version__)""")

md("""check: you should see `ready. cajal_lipidomics 0.0.1` and no red error. If you get a
`ModuleNotFoundError`, your notebook is on the wrong kernel; pick the `cajal-lipidomics` kernel
from the kernel picker.""")

md("""## 1. what a reference atlas is, and why we need one

A pixel is one MALDI laser spot, about 25 micrometers across. The mass spectrometer hands us,
for each pixel, an intensity for every lipid. It does not hand us a brain region. The scanner
only knows that this spot was at row `x`, column `y` of the slide. That slide coordinate is
meaningless across sections: the tissue was placed at a slightly different angle, cut at a
slightly different depth, the brain was a little bigger or smaller. Two pixels with the same
`(x, y)` on two slides are almost never the same piece of brain.

We need a shared coordinate system that every section can be mapped into, so that "this location"
means the same thing in control and in pregnant, in your brain and in mine. That shared system is
a reference atlas.

The Allen Mouse Brain Common Coordinate Framework v3 (CCFv3) is the standard one. Think of it as
two things bolted together:

- a 3D coordinate space. The whole mouse brain is described as a box of cubic voxels, each 25
  micrometers on a side. A point in the brain is just three numbers, `(x, y, z)`, in millimeters,
  measuring how far along the three anatomical axes you are: the anterior-posterior (AP) axis
  (front to back), the dorsal-ventral axis (top to bottom), and the medial-lateral axis (midline
  to side). This is the same idea as latitude, longitude, and altitude on Earth: any place gets
  one fixed triple of numbers.
- an annotation volume. For every one of those voxels, the Allen team filled in which brain region
  it belongs to. The brain was carved, by experts, into a tree of about a thousand structures: big
  divisions like cortex, striatum, thalamus, then finer ones like caudoputamen, then finer still
  like cortical layers. Each region has a short acronym (`CP` for caudoputamen) and a color chosen
  so that related regions look alike.

So once a pixel has a CCF coordinate `(x, y, z)`, getting its region is a pure lookup: go to that
voxel in the annotation volume, read off the region. Coordinate first, anatomy second. The hard
part, and the subject of this notebook, is getting that coordinate. That is registration: bending
each real, distorted section until it lines up with the atlas, then reading each pixel's atlas
coordinate.

❓ **QUESTION.** Why not just label regions by eye on each section? You can, for big obvious
structures. But our analysis compares more than a hundred fine regions across two sections, and
later against a third dataset. Doing that consistently, pixel by pixel, by eye, across hundreds of
thousands of pixels, is hopeless. A shared coordinate frame does it once and exactly.""")

md("""## 2. registration as an affine transform plus a nonlinear warp

Registration is the act of finding the geometric transformation that takes your section onto the
atlas. It comes in two layers, and the layers matter, so we build the intuition from the linear
algebra up.

### the affine layer: rotate, scale, shear, translate

The first layer handles the gross misalignment: your section is rotated a few degrees, slightly
bigger or smaller than the atlas, maybe squashed along one axis because the tissue shrank, and
shifted off-center. Every one of those is an affine transformation: a linear map (a 2x2 matrix `A`
in 2D) followed by a translation (a vector `t`). A point `p` moves to

$$ p' = A\\,p + t. $$

The matrix `A` packs four operations into four numbers: rotation, isotropic scaling (same in both
directions), anisotropic scaling (different per axis, which is how you undo a 20% shrinkage along
one direction), and shear. The translation `t` slides the whole thing. Affine maps keep straight
lines straight and parallel lines parallel: a grid drawn on the section stays a grid, just rotated,
stretched, and sheared. That is the key limitation, and why we need a second layer.

🔬 **TASK.** Build the intuition with a tiny demo. We take a small square grid of points, apply one
affine transform (a rotation, an anisotropic scale, and a shift), and look at what happened. No
atlas yet, just the geometry.""")

co("""# a small grid of points: this stands in for "landmarks on a section"
gx, gy = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6))
pts = np.column_stack([gx.ravel(), gy.ravel()])  # (36 points, 2 coords)

# build one affine transform A p + t
theta = np.deg2rad(20)                            # rotate 20 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.diag([1.3, 0.7])                            # stretch x by 1.3, squash y to 0.7
A = R @ S                                          # compose: first scale, then rotate
t = np.array([0.4, -0.2])                          # then translate

pts_affine = pts @ A.T + t                         # apply to every point at once

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(pts[:, 0], pts[:, 1], s=18, c="0.6", label="original grid (section)")
ax.scatter(pts_affine[:, 0], pts_affine[:, 1], s=18, c="crimson",
           label="after affine A·p + t")
ax.set_aspect("equal"); ax.legend(fontsize=FS["s"])
ax.set_title("affine transform: rotate, scale, shear, translate")
plt.tight_layout(); plt.show()
print("A =\\n", np.round(A, 2), "\\nt =", t)""")

md("""check: the red grid is a rotated, stretched, shifted copy of the gray grid, but it is still a
perfect grid: straight lines stayed straight, parallel stayed parallel. That is exactly what an
affine map can and cannot do.

❓ **QUESTION.** Real tissue does not just rotate and stretch uniformly. A section gets torn, folded,
locally swollen near a ventricle, compressed where the blade dragged. Can a single matrix `A` undo a
local fold in one corner while leaving the rest untouched? No: one matrix acts on the whole plane
identically. That is the job of the second layer.""")

md("""### the nonlinear layer: a smooth warp that bends locally

After the affine has done the gross alignment, regions are roughly in place but the fine boundaries
still do not match: the hippocampus is a touch too curved here, the cortex a bit too thick there.
The second layer is a nonlinear warp, a smooth deformation field that can push each point a different
little amount, bending locally without tearing.

Formally it is a displacement field: at every location it stores a small arrow saying "move this bit
of tissue by this much, in this direction". The good algorithms constrain the field to be smooth and
invertible (a diffeomorphism), so tissue never folds over itself or rips. Methods like elastix splines
(used by ABBA) or LDDMM (used by STalign) learn this field by minimizing the mismatch between the
warped section and the atlas, while penalizing roughness.

🔬 **TASK.** Add a gentle nonlinear warp on top of the affine grid, so you can see the difference. We
push each point by a smooth sine-shaped displacement: now the lines bend.""")

co("""# take the affine-transformed grid and add a smooth, local, nonlinear displacement
x, y = pts_affine[:, 0], pts_affine[:, 1]
amp = 0.12  # how strong the warp is
dx = amp * np.sin(2.0 * np.pi * y)   # horizontal push depends smoothly on y
dy = amp * np.sin(2.0 * np.pi * x)   # vertical push depends smoothly on x
pts_warped = np.column_stack([x + dx, y + dy])

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(pts_affine[:, 0], pts_affine[:, 1], s=18, c="crimson",
           label="affine only (lines stay straight)")
ax.scatter(pts_warped[:, 0], pts_warped[:, 1], s=22, c="navy",
           label="affine + nonlinear warp (lines bend)")
# draw the displacement arrows to make the warp visible
ax.quiver(pts_affine[:, 0], pts_affine[:, 1], dx, dy,
          angles="xy", scale_units="xy", scale=1, width=0.004, color="0.5")
ax.set_aspect("equal"); ax.legend(fontsize=FS["s"])
ax.set_title("nonlinear warp: a smooth displacement field on top of the affine")
plt.tight_layout(); plt.show()""")

md("""check: the blue points no longer form straight rows. The gray arrows show each point being
nudged a little, differently in different places, but smoothly. That bending is what lets the
section's curved hippocampus snap onto the atlas's curved hippocampus.

So the full registration is: affine first to get the gross fit, nonlinear warp second to bend the
fine boundaries into place. Run it, and you can read off, for every pixel, the atlas coordinate it
landed on. That coordinate is the CCF triple `(xccf, yccf, zccf)` we will load shortly.""")

md("""## 3. ABBA: the one piece we ship pre-computed

In a real lab you do not write the warp by hand. You use a tool. The community standard for
registering serial brain sections to the Allen CCFv3 is ABBA, which stands for Aligning Big Brains
and Atlases, from Nicolas Chiaruttini's group at the EPFL bioimaging platform.

ABBA is a Fiji (ImageJ) plugin, usually driven together with QuPath, through a mouse-driven
graphical interface. You feed it your section images, place them roughly along the atlas, and it runs
exactly the two layers we just built:

- an optional machine-learning pre-alignment (DeepSlice) guesses which atlas slice you are looking at
  and the cutting angle,
- an affine elastix step does the gross rotate/scale/shear/translate fit,
- a spline elastix step does the nonlinear warp,
- and for stubborn slices you hand-place landmark pairs in a tool called BigWarp.

A crucial detail for our data: ABBA also corrects the slicing angle. A coronal block is almost never
cut perfectly perpendicular to the AP axis, so a single physical section actually grazes a small
range of AP levels across its width. ABBA estimates that tilt and accounts for it. You will see the
fingerprint of this in our pregnant section in a moment.

When ABBA is done, you export a per-pixel CCF coordinate map: an image whose channels are the `x`,
`y`, `z` atlas coordinates of every pixel. Run that coordinate through the Allen annotation volume and
each pixel also gets a region acronym and color. That pair of products, the coordinate and the region
label, is exactly what we hand you on the two sections.

### why we ship this one ready-made

Everywhere else in this course you build the analysis yourself, from raw METASPACE ions through
normalization, embedding, clustering, and differential testing, with the code open in front of you.
Registration is the single exception, and it is worth understanding why:

- ABBA's core (elastix, BigWarp, DeepSlice) is Java and C++, so it cannot be unrolled into transparent
  Python the way every other step can. It would be an opaque black box in a course built on reading the
  code.
- It is a desktop GUI. Live-driving a mouse-heavy interface for two sections is high effort and low
  teaching payoff, and its headless mode on Linux is unreliable.
- Registration is a one-time setup, not a method you need to master to do the science. The differential
  lipidomics and the gene join downstream are where the value is.

So here ABBA is a concept with a described demo: in your own lab you would create a QuPath project, run
DeepSlice then affine then spline elastix, fix slices with BigWarp, and export the coordinate map. The
per-pixel CCF coordinate and region label that you load next are that export. They are the only thing in
this entire course that arrives pre-computed.

💡 **HINT.** The original Lipid Brain Atlas refined ABBA's alignment in-plane with STalign, a fully
scriptable Python LDDMM tool, aligning a lipid image against an Allen cell-density image. That STalign
step is the transparent, hands-on flavour of registration, and the Allen CCFv3 plus the region lookup
are reachable programmatically through the `allensdk` and `brainglobe` packages. We keep those calls out
of the live notebook because they download large atlas files on first use; the result is already baked
into the data you load now.""")

md("""## 4. loading the per-pixel CCF coordinates

Time to look at the real output. We pick up exactly where notebook 3 left off, loading the normalized
two-section pair `03_normalized.h5ad`. Everything registration produced is already sitting in
`adata.obs`: the CCF coordinates and, from the Allen lookup, the region acronym and color for every
pixel.

One housekeeping step. This file keeps two versions of the data: `adata.X` holds the raw per-pixel ion
images, and `adata.layers["umaia"]` holds the uMAIA-normalized values you computed in notebook 3. For
the lipid views in this notebook we want the normalized values, so we promote that layer to `adata.X`
once, up front, and everything downstream uses it.

🔬 **TASK.** Load the substrate, switch to the normalized values, and confirm its shape.""")

co('''# pick up notebook 3's output: control + pregnant, same coronal plane (~AP 6.5)
adata = ad.read_h5ad("../../data/derived/03_normalized.h5ad")
adata.X = adata.layers["umaia"]      # use the uMAIA-normalized values from notebook 3

print("pixels x lipids:", adata.shape)
print("\\nconditions:")
print(adata.obs["Condition"].value_counts())
print("\\nsections (SectionID):")
print(adata.obs["SectionID"].value_counts())''')

md("""The anatomy lives in a handful of `obs` columns. The provided registration gives us, per pixel,
the CCF coordinate and the region label that the annotation-volume lookup returned. Let us look at
exactly those columns for a few pixels.

🔬 **TASK.** Print the CCF coordinates and the Allen region acronym and color for the first few pixels.""")

co('''# the anatomy columns produced by registration + Allen lookup
anat_cols = ["Condition",
             "xccf", "yccf", "zccf",   # CCF coordinate, in millimeters, along the 3 anatomical axes
             "acronym", "allencolor"]  # the region's short code and its Allen palette color
adata.obs[anat_cols].head(6)''')

md("""Read one row. `xccf, yccf, zccf` are the pixel's position in the Allen brain, in millimeters
along the three anatomical axes. `acronym` is the region's short code, and `allencolor` is the Allen
palette color for that region as a hex string. The coordinate is what registration computed directly;
the acronym and color are what fell out of looking that coordinate up in the annotation volume.

❓ **QUESTION.** There are three CCF numbers but each pixel only sits on a 2D section. Where do all
three come from? The two in-plane coordinates come from where the pixel landed after the warp. The
out-of-plane one (here `xccf`, the AP axis) comes from which atlas level the section was placed at,
plus the slicing-angle correction. Together they pin the pixel in 3D.""")

md("""## 5. how a coordinate indexes the annotation volume

The acronym and color came from a lookup, and it is worth making that lookup completely explicit,
because once you have seen it there is nothing mysterious left. The CCF is a grid of 25-micrometer
voxels. A millimeter is 1000 micrometers, so one millimeter spans 1000 / 25 = 40 voxels. To turn a
coordinate in millimeters into a voxel index you multiply by 40 and take the floor (drop the
fractional part), because a coordinate anywhere inside a voxel belongs to that voxel:

$$ \\text{index} = \\big\\lfloor \\text{ccf}_\\text{mm} \\times 40 \\big\\rfloor. $$

That integer triple is the address you read in the Allen annotation volume to get the region. We do
not ship the multi-gigabyte annotation volume here (that is one reason the lookup was done for you up
front), but we can still reconstruct the voxel indices from the coordinates, to see the first half of
the chain with our own eyes.

🔬 **TASK.** Convert the CCF coordinates into integer voxel indices.""")

co('''# index = floor(ccf_mm * 40), because 1 mm = 40 voxels at 25 um
for ax_name in ["xccf", "yccf", "zccf"]:
    adata.obs[ax_name.replace("ccf", "_index")] = np.floor(adata.obs[ax_name].to_numpy() * 40).astype(int)

print(adata.obs[["xccf", "x_index", "yccf", "y_index", "zccf", "z_index", "acronym"]].head(4))''')

md("""So the full chain is: registration gives `(xccf, yccf, zccf)` in mm, multiply by 40 and floor to
get an integer voxel index, look that voxel up in the Allen annotation volume, and out comes the region
`acronym` and its `allencolor`. The file you loaded already carries the end of that chain; the cell
above reconstructed its first step.

Now the slicing-angle point from section 3, made concrete. A perfectly coronal section sits at one
single AP level, so its `x_index` (the AP voxel) should take exactly one value. A section cut at a
slight tilt grazes a small range of AP levels across its width, so its `x_index` spreads over several
values. Let us check our two sections.

🔬 **TASK.** Count how many distinct AP levels each section spans, and the millimeter range of its AP
coordinate.""")

co('''# AP axis is x here: a flat coronal cut -> one x_index; a tilted cut -> a small range
for cond in ["naive", "pregnant"]:
    sub = adata.obs[adata.obs["Condition"] == cond]
    n_levels = sub["x_index"].nunique()
    lo, hi = sub["xccf"].min(), sub["xccf"].max()
    print(f"{cond:9s}: {n_levels:2d} distinct AP level(s), "
          f"xccf from {lo:.3f} to {hi:.3f} mm")''')

md("""check: the control section sits at a single AP level (one `x_index`, a flat coronal plane), while
the pregnant section spreads across a few dozen AP levels over roughly a millimeter. That spread is the
slicing-angle correction doing its job: ABBA recognized that the pregnant block was cut at a small tilt
and assigned each part of the section to the AP level it actually belongs to, rather than forcing the
whole thing onto one plane. Both sections still center on the same plane near AP 6.5, so they are
genuinely comparable; the tilt is just honestly accounted for.""")

md("""## 6. region maps: control vs pregnant

Now the payoff. Every pixel carries an Allen color in `allencolor`, so coloring pixels by it draws the
brain regions directly onto the section, in the standard Allen palette. We use the helper
`cl.plotting.spatial_categorical`, which scatters each pixel at its CCF position `(zccf, -yccf)` and
paints it with the stored hex color. (We plot `-yccf` so that dorsal is up, the way you would view a
coronal section.)

Before calling the helper, see what it does in one line: it is just a scatter where the color of each
point is read straight from a column of hex strings. No colormap, no normalization, because the colors
are already chosen by the Allen atlas.

🔬 **TASK.** Draw the Allen region map for both sections side by side.""")

co('''# region map: each pixel painted with its Allen region color (allencolor)
axes = plotting.spatial_categorical(adata, color_key="allencolor",
                                    section_key="SectionID", title_key="Condition",
                                    point_size=3.0)
axes[0].figure.suptitle("Allen regions on each section (control vs pregnant)",
                        y=1.02, fontsize=FS["l"])
plt.show()''')

md("""check: two coronal sections, each carved into colored territories. The big central blob is the
caudoputamen, the folded outer ribbon is cortex, the paired structures below are hippocampus and
thalamus. The two sections show the same anatomy in the same colors, because both were registered into
the same atlas. That shared frame is exactly what lets us compare them.

❓ **QUESTION.** The two sections are not pixel-for-pixel identical: tissue is torn here, missing there,
a little rotated. Yet the colored regions land in the same places. What made that possible? The
registration: each raw, distorted section was warped onto the common atlas, so anatomy, not slide
position, decides a pixel's color.

Let us confirm how many distinct Allen regions we are working with, and which are the largest.""")

co('''# how many Allen regions appear, and the most-sampled ones
print("distinct Allen regions on these two sections:", adata.obs["acronym"].nunique())
print("\\nlargest regions by pixel count:")
print(adata.obs["acronym"].value_counts().head(10))''')

md("""So our two sections together touch 174 distinct Allen regions. The caudoputamen (`CP`) is the
biggest, then piriform cortex (`PIR`), then medial amygdala and a hippocampal field, exactly the
structures you expect at this coronal level. Each of these is now a group we can average lipids over.""")

md("""## 7. the grey matter vs white matter split

The single biggest division in brain lipid composition is grey matter versus white matter. White matter
is dense with myelin, the lipid-rich insulating sheath that oligodendrocytes wrap around axons, and
myelin is built largely from sphingolipids like HexCer (hexosylceramides) and Cer (ceramides). Grey
matter, packed with cell bodies and synapses, is richer in glycerophospholipids. If a lipidomic method
cannot see the grey/white split, it cannot see anything; it is the first sanity check.

How do we get the split from what we have? The Allen ontology marks white matter as a top-level group
called fiber tracts, and the regions in that group have their own short acronyms: `cc` (corpus
callosum), `fi` (fimbria), `int` (internal capsule), `ec` (external capsule), and so on. Our provided
file gives us each pixel's `acronym`, so we can label a pixel as white matter exactly when its acronym
is one of those fiber-tract codes. (In a lab with the full ontology loaded you would instead test
whether the white-matter node is an ancestor in the region's structure path; with only the acronyms in
hand, the explicit set below is the honest equivalent.)

🔬 **TASK.** Build a grey/white label from the fiber-tract acronyms, and count how the pixels split.""")

co('''# Allen 'fiber tracts' acronyms that appear at this coronal level -> white matter
WHITE_MATTER_ACRONYMS = {"cc", "fi", "int", "or", "ec", "alv", "fp", "df", "st", "opt",
                         "ml", "py", "cpd", "arb", "em", "fx", "ccg", "scwm"}

adata.obs["matter"] = np.where(
    adata.obs["acronym"].astype(str).isin(WHITE_MATTER_ACRONYMS), "white", "grey")

print(adata.obs["matter"].value_counts())
print(f"\\nwhite matter is {100 * (adata.obs['matter'] == 'white').mean():.1f}% of all pixels")''')

md("""About one pixel in seventeen is white matter, the rest grey, the right ballpark for a coronal
section at this level (the internal capsule, fimbria, external capsule, and other tracts cut through
here). Now draw the split on the sections so you can see the tracts.

🔬 **TASK.** Map a simple two-color scheme: black for white matter, light gray for grey matter.""")

co('''# paint white matter black, grey matter light-gray, to expose the tracts
matter_color = {"white": "#111111", "grey": "#cfcfcf"}
adata.obs["matter_color"] = adata.obs["matter"].map(matter_color)

axes = plotting.spatial_categorical(adata, color_key="matter_color",
                                    section_key="SectionID", title_key="Condition",
                                    point_size=3.0)
axes[0].figure.suptitle("grey (light) vs white matter (black)", y=1.02, fontsize=FS["l"])
plt.show()''')

md("""check: the black pixels trace clear bands, the internal capsule cutting through the caudoputamen,
the fimbria near the hippocampus, the external capsule wrapping the cortex. These are the white-matter
tracts, recovered purely from the Allen ontology, with no lipid information used. Next we check whether
the lipids agree.""")

md("""### does myelin lipid track the white-matter anatomy?

Here is the test that ties anatomy to chemistry. If white matter is myelin and myelin is sphingolipid,
then a myelin marker lipid like HexCer 42:2 should be far brighter in the white-matter pixels than in
grey. `HexCer` is the class (hexosylceramide) and `42:2` is the sum composition (42 acyl carbons, 2
double bonds). We never told the lipids where the tracts are; if they light up the tracts anyway,
anatomy and lipidomics are telling the same story.

🔬 **TASK.** Compare the mean intensity of HexCer 42:2 in white vs grey matter, then plot the lipid
spatially so you can see it overlap the tracts.""")

co('''# myelin marker: HexCer 42:2. Compare its mean in white vs grey matter.
myelin_lipid = "HexCer 42:2"
j = list(adata.var["lipid"]).index(myelin_lipid)   # var_names are formulas; lipid names live in var['lipid']
vals = np.asarray(adata.X[:, j]).ravel()

for matter in ["white", "grey"]:
    m = (adata.obs["matter"] == matter).to_numpy()
    print(f"{myelin_lipid} mean in {matter} matter: {vals[m].mean():.4f}")

ratio = vals[(adata.obs['matter'] == 'white').to_numpy()].mean() / \\
        vals[(adata.obs['matter'] == 'grey').to_numpy()].mean()
print(f"\\nwhite / grey ratio: {ratio:.2f}x")''')

co('''# now SEE it: the helper draws one lipid on both sections with a shared color scale.
# var_names are chemical formulas, so address the lipid by its formula (the var index at j).
plotting.spatial_lipid(adata, adata.var_names[j], section_key="SectionID",
                       title_key="Condition", point_size=3.0,
                       background=False, show_contours=False)
plt.show()''')

md("""check: HexCer 42:2 is several times brighter in white matter than in grey, and the spatial map
lights up exactly the bands you saw in black a moment ago. The myelin lipid traces the white-matter
anatomy on its own. That agreement between an anatomy label (from the Allen atlas) and a lipid signal
(from the mass spectrometer), with neither one informed by the other, is the first proof that the
registration is sound and that lipids carry real anatomical structure.""")

md("""## 8. region-level lipid views: the mean lipid profile of every region

We have a region label on every pixel and a normalized intensity for every lipid. The natural summary
is the region-by-lipid matrix: for each Allen region, the mean intensity of each lipid across all its
pixels. One row per region, one column per lipid. This single table is the workhorse for everything
anatomical that follows: it is what a region-level differential test compares between conditions, and it
is the shape that matches a region-by-gene table when we join to gene expression later.

Before any helper, build it by hand in two lines, because the operation is just a `groupby` mean. Seeing
it built makes the helper obvious.

🔬 **TASK.** Construct the region-by-lipid mean matrix with a plain pandas groupby.""")

co('''# region x lipid: mean intensity of each lipid within each Allen region
X = np.asarray(adata.X)
lipid_df = pd.DataFrame(X, columns=list(adata.var["lipid"]))
lipid_df["acronym"] = adata.obs["acronym"].to_numpy()

region_by_lipid = lipid_df.groupby("acronym", observed=True).mean()
print("region x lipid matrix shape:", region_by_lipid.shape)
region_by_lipid.iloc[:5, :4]   # a corner of the table: 5 regions x 4 lipids''')

md("""That is the whole idea: 174 regions, 104 lipids, each entry a mean intensity. The helper
`cl.plotting.sorted_lipid_heatmap` does exactly this groupby mean and then draws it as a heatmap, with
one extra touch: it orders both the rows (regions) and the columns (lipids) by hierarchical clustering on
cosine similarity, so that regions with similar lipid profiles sit next to each other and the block
structure becomes visible. The sorting is the art; the table underneath is the two-line groupby you just
wrote.

🔬 **TASK.** Draw the region-by-lipid heatmap, clustered on both axes.""")

co('''# the helper: per-lipid 0-1 normalize, group-mean by acronym, then cosine-cluster both axes
ax, sorted_df = plotting.sorted_lipid_heatmap(
    adata, group_key="acronym", cmap="magma",
    figsize=(16, 9), title="region x lipid (mean intensity, both axes clustered)")
plt.show()
print("clustered matrix shape:", sorted_df.shape)''')

md("""check: the heatmap shows clear blocks, groups of regions (rows) that share a group of high lipids
(columns). Those blocks are the lipidomic signature of anatomy. A band of regions lighting up the
sphingolipid columns is the white matter; other blocks are cortical, striatal, thalamic. Regions that
landed next to each other did so because their lipid profiles are similar, which is the clustering
finding structure with no anatomy labels used in the sorting at all.

❓ **QUESTION.** Why average lipids within a region instead of testing pixel by pixel? Two reasons.
First, a single pixel is noisy and is a mixture of cell bodies, axons, and glia, not a clean readout;
averaging over a region cancels that noise. Second, gene-expression atlases are summarized per Allen
region, so to ask "do the lipids and the genes agree in the same region?" we need the lipids in the same
per-region shape. Anatomy is the shared key that lets two completely different measurement technologies
speak to each other.""")

md("""## 9. why this anatomy powers the next notebooks

Step back and see what we built, and why it matters downstream.

- We loaded the per-pixel CCF coordinates that registration (ABBA, refined in the original atlas by
  STalign) produced, the one piece of infrastructure shipped ready-made, and made the coordinate-to-region
  lookup explicit.
- We drew region maps for control and pregnant in the same atlas frame, so the two sections are directly
  comparable region by region.
- We split grey from white matter straight from the Allen fiber-tract acronyms, and confirmed the myelin
  lipid HexCer 42:2 traces the white-matter tracts on its own, the first proof that anatomy and lipidomics
  agree.
- We built the region-by-lipid matrix, the mean lipid profile of every region.

That region-by-lipid matrix is the hinge for what comes next:

- region-level differential lipidomics. With both sections in the same atlas, we can ask, for each region,
  which lipids changed between control and pregnant, using the Wilcoxon plus Benjamini-Hochberg test you
  have already met, region by region.
- the gene join. Gene-expression atlases give expression averaged over the same Allen regions, and the
  Allen acronym is the shared key: join our region-by-lipid table to a region-by-gene table on `acronym`,
  and we can finally ask whether a region's lipid profile and its gene-expression profile tell the same
  story.

You now know what a reference atlas is, how registration (affine plus nonlinear warp, done by ABBA in
practice) puts every pixel on it, how a coordinate indexes the annotation volume, and how to turn pixels
into region-level views. The next notebook puts this to work on the biology: which lipids change, region
by region, in pregnancy.""")

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "cajal-lipidomics", "language": "python", "name": "cajal-lipidomics"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "notebooks/level2/04_anatomy_allen_solution.ipynb")
print("wrote notebook with", len(cells), "cells")
