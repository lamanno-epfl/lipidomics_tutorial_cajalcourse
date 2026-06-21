"""Strip AI-ism em-dash pauses from notebook markdown (Luca's style rule).

Replaces " — "/" – " pauses with the right punctuation by context:
  after a code span / bold / closing bracket -> ": "  (a definition dash)
  before a lower-case word                   -> ", "  (a continuation)
  before an upper-case word / new clause     -> ". "  (a clause break, post capitalised)

Only markdown cells are touched, so executed code outputs are preserved. Code cells are
left alone but their em-dash count is reported so display strings can be spot-fixed.

    python scripts/destyle.py notebooks/**/*.ipynb        # fix listed notebooks
"""
from __future__ import annotations

import glob
import re
import sys
import nbformat

_PAT = re.compile(r"(\S)[ \t]*[—–][ \t]*(\S)")


def destyle_text(text: str) -> str:
    def repl(m):
        pre, post = m.group(1), m.group(2)
        if pre in "`*)]\"'":
            return pre + ": " + post
        if post.islower():
            return pre + ", " + post
        return pre + ". " + post.upper()
    return _PAT.sub(repl, text)


def fix_notebook(path: str) -> tuple[int, int]:
    nb = nbformat.read(path, as_version=4)
    md_before = sum(c.source.count("—") + c.source.count("–") for c in nb.cells if c.cell_type == "markdown")
    for c in nb.cells:
        if c.cell_type == "markdown":
            c.source = destyle_text(c.source)
    md_after = sum(c.source.count("—") + c.source.count("–") for c in nb.cells if c.cell_type == "markdown")
    code_dash = sum(c.source.count("—") + c.source.count("–") for c in nb.cells if c.cell_type == "code")
    nbformat.write(nb, path)
    return md_before - md_after, code_dash


def main(argv):
    paths = []
    for a in argv[1:]:
        paths.extend(glob.glob(a, recursive=True))
    paths = sorted(set(paths))
    for p in paths:
        fixed, code_dash = fix_notebook(p)
        flag = f"  (code-cell dashes left: {code_dash})" if code_dash else ""
        print(f"{p}: removed {fixed} markdown em-dashes{flag}")


if __name__ == "__main__":
    main(sys.argv)
