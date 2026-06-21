"""Generate student notebooks from solution notebooks (one source of truth).

For each solution notebook, write a sibling student notebook that keeps all markdown
(prose, the TASK / HINT / QUESTION / CHECKPOINT callouts) and any code cell tagged
`keep` (provided scaffolding/plotting), and replaces every code cell tagged `task`
(or untagged code, by default) with a `# TODO` blank plus its leading comment lines.

Cell tags live in notebook cell metadata (`{"tags": ["keep"|"task"|"solution"]}`).
This keeps solution and student versions in lockstep: edit the solution, regenerate.

Usage:
    python scripts/make_student.py            # regenerate all student notebooks
    python scripts/make_student.py NB.ipynb   # one notebook
"""
from __future__ import annotations

# TODO(M7): implement nbformat round-trip with the tag-driven stripping rules above.
