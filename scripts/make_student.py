"""Generate student notebooks from solution notebooks (one source of truth).

For each `*_solution.ipynb`, write a sibling `*_student.ipynb` that:
  - keeps every markdown cell verbatim (prose + the TASK/HINT/QUESTION/CHECKPOINT callouts),
  - keeps code cells tagged "keep" or "illustrative" (provided scaffolding / plotting),
  - replaces the body of code cells tagged "task" with a blank (their leading comment lines
    are preserved as guidance, then a "# Your code here" placeholder),
  - strips all outputs and execution counts (students start from a clean slate).

Untagged code cells default to "keep" (runnable scaffolding) so a student notebook never
breaks for lack of a tag. Cell tags live in cell.metadata["tags"].

Usage:
    python scripts/make_student.py                       # all notebooks/level*/*_solution.ipynb
    python scripts/make_student.py path/to/x_solution.ipynb
"""
from __future__ import annotations

import glob
import sys
import nbformat


def _leading_comments(src: str) -> list[str]:
    out = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#") or s == "":
            out.append(line)
        else:
            break
    return out


def to_student(sol_path: str) -> str:
    nb = nbformat.read(sol_path, as_version=4)
    out_cells = []
    for c in nb.cells:
        tags = set(c.get("metadata", {}).get("tags", []))
        if c.cell_type == "markdown":
            out_cells.append(c)
            continue
        # code cell
        c = nbformat.from_dict(c)
        c["outputs"] = []
        c["execution_count"] = None
        if "task" in tags:
            lead = _leading_comments(c.source)
            body = "\n".join(lead).rstrip()
            c["source"] = (body + "\n" if body else "") + "# 🔬 your code here\n"
        out_cells.append(c)
    nb.cells = out_cells
    student_path = sol_path.replace("_solution.ipynb", "_student.ipynb")
    nbformat.write(nb, student_path)
    n_task = sum(1 for c in nb.cells if c.cell_type == "code" and "task" in set(c.get("metadata", {}).get("tags", [])))
    print(f"  {sol_path} -> {student_path}  ({len(nb.cells)} cells, {n_task} task-blanks)")
    return student_path


def main(argv):
    paths = argv[1:] if len(argv) > 1 else sorted(glob.glob("notebooks/level*/*_solution.ipynb"))
    if not paths:
        print("no *_solution.ipynb found yet (solutions still being authored).")
        return
    print(f"generating {len(paths)} student notebooks:")
    for p in paths:
        to_student(p)


if __name__ == "__main__":
    main(sys.argv)
