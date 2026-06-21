"""MSI dataframe <-> AnnData bridge and lipid-name parsing.

Adapted from the Lipid Brain Atlas `assets/data_handler.py`: cleanly separates the
lipid columns (the measured features) from the per-pixel metadata, and parses lipid
shorthand into class / carbons / unsaturations for coloring and grouping.

Planned API (built against real data in M1):

    msi_df_to_anndata(df, lipid_cols=None) -> AnnData
        # X = pixels x lipids; obs = metadata (x, y, SectionID, Condition, region, CCF...)

    lipid_properties(names) -> DataFrame
        # regex: class, total carbons, double bonds, ether flag (recycled from EUCLID)
"""
from __future__ import annotations

# Implemented in M1 against the real 2-section table.
