"""Figure style for the course.

Encodes the lab plotting rules so every panel is clean, dense, and
Illustrator-editable by default: text stays as vector (fonttype 42), scatter
is rasterised per-plot, a small fixed set of font sizes, no top/right spines,
divergent colormaps for signed data, lightweight colorbars.

Import once at the top of a notebook::

    from cajal_lipidomics.style import set_style, FS
    set_style()
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Four standardised font sizes (very small / small / medium / large).
FS = {"xs": 6, "s": 8, "m": 10, "l": 13}

# Sequential default for intensities.
CMAP_SEQUENTIAL = "plasma"


def set_style() -> None:
    """Apply the course-wide matplotlib defaults."""
    mpl.rcParams.update(
        {
            # vector text editable in Illustrator; rasterise heavy artists per-plot
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # typography: one family, the four FS sizes
            "font.size": FS["s"],
            "axes.titlesize": FS["m"],
            "axes.labelsize": FS["s"],
            "xtick.labelsize": FS["xs"],
            "ytick.labelsize": FS["xs"],
            "legend.fontsize": FS["xs"],
            "figure.titlesize": FS["l"],
            # clean, minimal axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.grid": False,
            "legend.frameon": False,
            "image.cmap": CMAP_SEQUENTIAL,
        }
    )


def lightweight_colorbar(mappable, ax, label: str | None = None, **kwargs):
    """A compact colorbar: thin, no black border, few ticks, vertical label."""
    cb = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02, **kwargs)
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.5, length=2, labelsize=FS["xs"])
    if label is not None:
        cb.set_label(label, fontsize=FS["xs"])
    return cb


def spatial_axes(ax) -> None:
    """Strip a spatial scatter to the essentials: equal aspect, no ticks/spines."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
