"""MSI dataframe <-> AnnData bridge and lipid-name parsing.

Adapted from the Lipid Brain Atlas `assets/data_handler.py`: cleanly separates the
lipid columns (the measured features) from the per-pixel metadata, and parses lipid
shorthand into class / carbons / unsaturations for coloring and grouping.

API:

    lipid_properties(names) -> DataFrame
        # regex over a list of names: class, total carbons, double bonds, oxygens,
        # ether flag, and a per-class color, indexed by the original name.

    msi_df_to_anndata(df, lipid_cols=None) -> AnnData
        # X = pixels x lipids; obs = metadata (x, y, SectionID, Condition, region, CCF...)
        # (built against the real 2-section table in a later milestone)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# A stable, colorblind-friendly color per lipid class. Classes not listed fall back
# to gray. These match the palette the notebook uses when grouping lipids by class.
CLASS_COLORS = {
    "PC": "#1f77b4",      # phosphatidylcholine
    "PE": "#ff7f0e",      # phosphatidylethanolamine
    "PS": "#2ca02c",      # phosphatidylserine
    "PI": "#d62728",      # phosphatidylinositol
    "PA": "#9467bd",      # phosphatidic acid
    "PG": "#8c564b",      # phosphatidylglycerol
    "LPC": "#e377c2",     # lyso-PC
    "LPE": "#7f7f7f",     # lyso-PE
    "SM": "#bcbd22",      # sphingomyelin
    "Cer": "#17becf",     # ceramide
    "HexCer": "#aec7e8",  # hexosylceramide (myelin)
}
_FALLBACK_COLOR = "#999999"


def parse_lipid_name(name):
    """Split one lipid shorthand into (class, carbons, double_bonds, oxygens, ether).

    'PC 38:6'        -> ('PC', 38, 6, 0, False)
    'HexCer 42:2;O2' -> ('HexCer', 42, 2, 2, False)
    'PC O-34:1'      -> ('PC', 34, 1, 0, True)   # the O- prefix marks an ether bond

    The class is the text before the first space or parenthesis; the carbons are the
    number before the colon; the double bonds are the number right after the colon;
    the oxygens are the number after a ';O' suffix if present.
    """
    name = str(name)
    lipid_class = re.split(r"[ (]", name)[0]          # text before first space/paren
    carbons = re.search(r"(\d+):", name)              # number before the colon
    doublebonds = re.search(r":(\d+)", name)          # number after the colon
    oxygens = re.search(r";O(\d*)", name)             # number after ';O' (default 1)
    ether = bool(re.search(r"\bO-\d", name))          # 'O-' before a chain = ether bond
    carbons = int(carbons.group(1)) if carbons else np.nan
    doublebonds = int(doublebonds.group(1)) if doublebonds else np.nan
    if oxygens:
        oxygens = int(oxygens.group(1)) if oxygens.group(1) else 1
    else:
        oxygens = 0
    return lipid_class, carbons, doublebonds, oxygens, ether


def lipid_properties(names):
    """Parse a list of lipid names into a tidy table, with a per-class color.

    Runs the same regex as `parse_lipid_name` over every name and returns a DataFrame
    indexed by the original name, with columns: lipid_class, carbons, double_bonds,
    oxygens, ether, color. This is the whole-list version of the parser the notebook
    unrolls by hand.
    """
    rows = [parse_lipid_name(n) for n in names]
    df = pd.DataFrame(
        rows, columns=["lipid_class", "carbons", "double_bonds", "oxygens", "ether"],
        index=pd.Index(list(names), name="name"),
    )
    df["color"] = df["lipid_class"].map(CLASS_COLORS).fillna(_FALLBACK_COLOR)
    return df
