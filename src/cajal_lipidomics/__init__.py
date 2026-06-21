"""cajal_lipidomics — small helper package for the CAJAL spatial-metabolomics course.

It holds the "ready" pieces we hand students so they spend their attention on the
science, not boilerplate: a consistent figure style (`style`), beautiful plotting
functions recycled from the Lipid Brain Atlas (`plotting`), data loaders and the
METASPACE pull (`io`), and the MSI dataframe <-> AnnData bridge (`data`).

Everything here is meant to be read: it is transparent, documented, and short.
Where a function wraps something students should understand, the notebook unrolls
the same logic first, then points at the wrapper.
"""
from __future__ import annotations

__version__ = "0.0.1"

from . import style  # noqa: F401
from . import data  # noqa: F401
