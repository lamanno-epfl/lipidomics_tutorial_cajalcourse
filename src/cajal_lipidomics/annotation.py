"""Peak annotation: matching MALDI m/z peaks to lipids, and the LC-MS <-> MSI ppm plot.

MALDI gives a mass, not an identity. One lipid appears at several m/z (adducts +H/+Na/
+K/+NH4), and several lipids can share a near-identical mass. Annotation matches each
observed m/z against a reference list within a parts-per-million window. The side-by-side
ppm plot makes that operation visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import FS

# positive-mode adduct masses (Da)
ADDUCTS = {"H+": 1.007276, "Na+": 22.989769, "K+": 38.963707, "NH4+": 18.033823}


def ppm_error(observed, reference):
    """Parts-per-million mass error. 5 ppm at m/z 800 is a 0.004 Da window."""
    return 1e6 * np.abs(observed - reference) / reference


def load_lcms_reference(path):
    """Load the in-house LC-MS reference (columns: Lipid, Adduct, m/z)."""
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    keep = [c for c in df.columns if c in ("Lipid", "Adduct", "m/z")]
    return df[keep].dropna(subset=["m/z"])


def match_lcms(observed_mz, lcms_df, ppm_tol=5.0):
    """LC-MS reference peaks within ppm_tol of an observed m/z, with their ppm distance."""
    mzs = lcms_df["m/z"].to_numpy(float)
    e = ppm_error(observed_mz, mzs)
    hit = e <= ppm_tol
    out = lcms_df.loc[hit].copy()
    out["ppm"] = e[hit]
    return out.sort_values("ppm")


def plot_ppm_match(observed_mz, lcms_df, ppm_tol=5.0, span_ppm=40, ax=None):
    """Two stacked stick spectra sharing the m/z axis: LC-MS/MS reference peaks (top) and
    the single MSI peak (bottom), with the +-ppm acceptance band shaded and the ppm
    distance annotated. Reference peaks inside the band are accepted matches (green).
    """
    lo = observed_mz * (1 - span_ppm * 1e-6)
    hi = observed_mz * (1 + span_ppm * 1e-6)
    mzs = lcms_df["m/z"].to_numpy(float)
    ref = lcms_df.loc[(mzs >= lo) & (mzs <= hi)].copy()
    tol = observed_mz * ppm_tol * 1e-6

    if ax is None:
        fig, (axL, axM) = plt.subplots(2, 1, sharex=True, figsize=(9, 4),
                                       gridspec_kw={"height_ratios": [2, 1]})
    else:
        axL, axM = ax
    for a in (axL, axM):
        a.axvspan(observed_mz - tol, observed_mz + tol, color="gold", alpha=0.25, zorder=0)
        a.axvline(observed_mz, color="k", ls="--", lw=0.8)
    axL.vlines(ref["m/z"], 0, 1, color="tab:blue", lw=2)
    # group references that sit at (nearly) the same m/z so their labels never overlap;
    # an isobaric tie (several lipids, one mass) gets one stacked, multi-line label.
    ref = ref.assign(_key=ref["m/z"].round(3))
    for key, grp in ref.groupby("_key"):
        e = ppm_error(observed_mz, key)
        inside = e <= ppm_tol
        names = [f"{r['Lipid']} [{r.get('Adduct','')}]" for _, r in grp.iterrows()]
        lab = "\n".join(names) + f"\nΔ={e:.1f} ppm"
        axL.annotate(lab, (key, 1.04), ha="center", va="bottom", fontsize=FS["xs"],
                     color=("tab:green" if inside else "0.6"))
    axL.set_ylabel("LC-MS reference"); axL.set_yticks([])
    axL.set_ylim(0, 1.05 + 0.18 * (1 + ref.groupby("_key").size().max()))
    axM.vlines(observed_mz, 0, 1, color="crimson", lw=3)
    axM.set_ylabel("MSI peak"); axM.set_yticks([]); axM.set_xlabel("m/z")
    axL.set_title(f"annotating MSI peak {observed_mz:.4f} (±{ppm_tol} ppm)", fontsize=FS["m"])
    return axL, axM
