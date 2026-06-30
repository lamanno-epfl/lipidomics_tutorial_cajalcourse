"""Data loading for the course: the METASPACE pull and local artifact loaders.

The METASPACE pull is unrolled transparently in notebook 1 (students see
`results` -> `all_annotation_images` -> stack -> AnnData); this module holds the
tidy reusable version of that same logic, plus loaders for the provided artifacts
(the chosen sections, the reference
databases).

Planned API (built against real data in M1):

    pull_metaspace_section(dataset_id, database=("CoreMetabolome","v3"), fdr=0.1,
                           scale_intensity=False) -> AnnData
        # pixels x ions; var = formula/adduct/mz/moleculeNames; obsm['spatial']=(x,y)

    list_mlba_datasets() -> pandas.DataFrame
        # enumerate the public mlba-2025 project, split control vs pregnant by name

    attach_provided_ccf(section_adata, registration_path)
        # join per-pixel xccf/yccf/zccf + Allen acronym/color from the provided registration_ccf.parquet
"""
from __future__ import annotations

# Implemented in M1 once the section pair is locked and verified end to end.
