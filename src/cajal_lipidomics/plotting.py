"""Ready-made, beautiful plotting functions for the course.

These are recycled/adapted from the Lipid Brain Atlas so students get publication
quality for free and spend their effort on interpretation. A few plots are left
for students to build themselves (the ppm-match exhibit, the volcano, the final
multi-panel figure) — those live in the notebooks, not here.

Planned API (filled in as the verified pipeline is built in M1/M2; signatures are
fixed now so notebooks can import against them):

    spectrum(mz, intensity, ...)                  # stick mass spectrum (vlines)
    spatial_lipid(adata, lipid, ...)              # per-section CCF scatter, plasma, 2/98 pct
    spatial_categorical(adata, key, ...)          # lipizones / regions colored scatter
    moran_panel(adata, good_mz, bad_mz, ...)      # good vs bad Moran features, side by side
    umaia_before_after(x_raw, x_norm, masks, v)   # per-section histograms + fitted modes
    sorted_heatmap(matrix, ...)                   # anatomy x lipid / anatomy x lipizone,
                                                  #   cosine optimal-leaf-ordered (copied verbatim
                                                  #   from LBA — the sorting is an art, not reinvented)

Implementations are added against real data so they are never fabricated.
"""
from __future__ import annotations

# NOTE: bodies are added in M1/M2 once the real 2-section AnnData exists, so every
# function is verified on real data before it reaches a student notebook.
