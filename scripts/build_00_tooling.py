#!/usr/bin/env python
"""Assemble notebooks/00_intro/00_tooling.ipynb with nbformat.

Notebook 00 of the CAJAL Bordeaux 2026 self-guided intro: "your computer as a
lab bench". Mostly prose + tiny demonstrative shell cells (run with the ! prefix).
Code cells are tagged "keep" (provided, students just run) or "task" (students
write; become blanks in the student version).
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(src, tag="keep"):
    c = new_code_cell(src)
    c.metadata["tags"] = [tag]
    cells.append(c)


# ----------------------------------------------------------------------------
# Title + orientation
# ----------------------------------------------------------------------------
md(r"""# 00 — your computer as a lab bench

**A spatial metabolomics primer · CAJAL Neuromics Summer School, Bordeaux, July 2026**
Self-guided intro, notebook 0 of 3 · about 60 minutes · assumes no prior programming

A bench biologist learns the pipettes, the centrifuge, and the cold room before touching a
real sample. A computational biologist learns a handful of tools first too, and they are
fewer than you fear: a terminal, git, a package environment, and a notebook editor. This
notebook is the tour of that bench. By the end you can open a terminal without flinching,
find your way around folders, understand what an environment is and why it makes your work
reproducible, run a notebook cell on purpose, and use Claude Code without letting it think
for you.

You already did the installs in [`docs/SETUP.md`](../../docs/SETUP.md). If you have not,
stop and do that first, because the rest of this notebook assumes the `cajal-lipidomics`
environment exists and that you opened this file with that kernel selected. This notebook
re-teaches the *why* behind each setup step, slowly, so the commands stop being magic.""")

md(r"""## how to read these notebooks

Throughout all three intro notebooks you will see four little signposts. They mean the same
thing every time.

- 🔬 **TASK** — something for you to do or type. You learn by doing, not by reading.
- 💡 **HINT** — a nudge if you are stuck, or a pointer to where the answer lives.
- ❓ **QUESTION** — a moment to stop and think. There is no autograder. The thinking is the point.
- ⚠️ **CHECKPOINT** — what you should see on screen right now. If you do not see it, something
  is off, and it is worth fixing before moving on.

Two kinds of code cells appear. Most are **provided**: you press run and watch what happens.
A few are **for you to fill in**: in the student copy of this notebook those cells are blank
with a comment telling you what to write. This master copy already has them filled so you can
check yourself.""")

# ----------------------------------------------------------------------------
# Section 1: the terminal / shell
# ----------------------------------------------------------------------------
md(r"""## 1. the terminal, or talking to your computer in words

You usually drive your computer by pointing and clicking. The **terminal** (also called the
**shell**, or the **command line**) is the other door: a plain text window where you type a
command, press Enter, and the computer does exactly that one thing and prints the result. No
icons, no menus. It looks bare, and that is its strength. A typed command is precise,
repeatable, and easy to share. "Click the third icon, then the second menu" is none of those.

The shell matters for science because analysis pipelines are sequences of exact commands. If
you can write the commands down, you can re-run them next month, send them to a colleague, or
paste them into a methods section. That is reproducibility, and it starts here.

A small vocabulary note. The **shell** is the program that reads your commands (on Mac and
Linux it is usually `bash` or `zsh`; on Windows you installed **Git Bash** in setup so you get
the same commands as everyone else). The **terminal** is the window the shell runs inside. People
use the words interchangeably and so will we.""")

md(r"""### running shell commands from inside a notebook

You do not need to leave this notebook to try shell commands. In a Jupyter code cell, any line
that starts with an exclamation mark `!` is handed to the shell instead of to Python. So `!pwd`
in a cell runs the shell command `pwd`. We use that trick below to demonstrate commands live.

In your *real* terminal you would type the same thing **without** the `!`. So when a cell shows
`!ls`, the terminal equivalent you would type yourself is just `ls`.

🔬 **TASK** — run the next cell (click it, press `Shift`+`Enter`). It asks the shell: *which
folder am I standing in right now?*""")

code(r"""# pwd = "print working directory": the folder the shell is currently in.
# The ! sends this line to the shell instead of to Python.
!pwd""", tag="keep")

md(r"""⚠️ **CHECKPOINT** — you should see a path printed, ending in either
`lipidomics_tutorial_cajalcourse` or `notebooks/00_intro` (it depends on which folder your editor
launched the notebook from). Either is fine. That path is your **working directory**: the folder
every command treats as "here" unless you say otherwise. The shell always has exactly one working
directory, the way you are always standing in exactly one room.""")

md(r"""### listing what is around you: `ls`

`ls` ("list") prints the contents of the current folder: the files and the sub-folders. It is
how you look around. Commands often take **options** (also called flags), which are extra words
starting with a dash that change behaviour. `ls -l` gives a long, detailed listing; `ls -a`
also shows hidden files (the ones whose names start with a dot).""")

code(r"""# ls lists the current folder. -1 (the digit one) prints one entry per line, easy to read.
!ls -1""", tag="keep")

md(r"""❓ **QUESTION** — do you recognise the folders from the [README](../../README.md)? You
should see `notebooks`, `docs`, `src`, and a few more. The shell is showing you the same files
your file browser would, just as text.""")

md(r"""### moving around: `cd`, and the idea of a path

A **path** is an address for a file or folder. There are two flavours, and mixing them up is the
single most common beginner confusion, so we go slowly.

An **absolute path** starts from the very top of the filesystem and spells out the whole address.
On Mac and Linux it starts with a slash `/`; for example
`/home/you/lipidomics_tutorial_cajalcourse/notebooks`. It means the same thing no matter where
you are standing, the way a full postal address does.

A **relative path** starts from wherever you currently are. `notebooks` means "the `notebooks`
folder inside my current folder". Two special shorthands show up everywhere: `.` means "here, the
current folder", and `..` means "one folder up, my parent". So `../docs` means "go up one level,
then into `docs`".

`cd` ("change directory") is how you walk from folder to folder. `cd notebooks` steps in; `cd ..`
steps back out. Below we do not actually move the notebook's directory (each `!` cell starts fresh),
but we *show* both kinds of path so the difference is concrete.""")

code(r"""# Absolute path: the full address. pwd already printed ours; here we ask the shell to
# echo a relative path and an absolute one side by side so you can compare them.
!echo "relative example:  notebooks/00_intro"
!echo "the .. shorthand:  ..  means one folder up"
!echo -n "absolute (full):   " ; pwd""", tag="keep")

md(r"""💡 **HINT** — when something does not work because "the file is not found", 9 times out of
10 the path is the problem: you gave a relative path but were standing in the wrong folder. The
fix is almost always to run `pwd` first to see where you are, then adjust. Absolute paths never
have this ambiguity, which is exactly why this notebook tells you to use absolute paths when in
doubt.""")

md(r"""### making a folder: `mkdir`

`mkdir` ("make directory") creates a new, empty folder. You will do this to keep your outputs
tidy, for instance a `results/` folder for figures. Let us make a scratch folder, confirm it
appeared, then clean it up so we leave no mess.

🔬 **TASK** — the first cell below is **yours to write** in the student copy. The task: make a
folder called `scratch_demo` using `mkdir`. The master copy shows the answer.""")

code(r"""# TASK: make a new folder called scratch_demo
mkdir_target = "scratch_demo"   # the folder name we will create
!mkdir -p {mkdir_target}        # -p = "do not complain if it already exists"
!ls -d scratch_demo             # -d shows the folder itself, proving it now exists""", tag="task")

code(r"""# Tidy up: remove the demo folder so the repo stays clean.
# rmdir only removes EMPTY folders, which is a safe way to delete: it refuses if
# there is anything inside, so you cannot wipe out data by accident.
!rmdir scratch_demo
!echo "scratch_demo removed; folder is gone"
""", tag="keep")

md(r"""⚠️ **CHECKPOINT** — the first cell printed `scratch_demo`, and the second confirmed it was
removed. You just created and deleted a folder entirely from text. That is the whole game.""")

md(r"""### when you forget how a command works: `--help`

Nobody memorises every option. Almost every command tells you about itself if you add `--help`.
The output can be long and terse at first, but the shape is always the same: a one-line summary,
then a list of options. Learning to skim it is a real skill, and it beats guessing.""")

code(r"""# Ask ls to describe itself. We pipe through head so we only see the first lines.
# (A pipe | sends one command's output into the next; head -n 15 keeps 15 lines.)
!ls --help | head -n 15""", tag="keep")

md(r"""💡 **HINT** — if `ls --help` prints nothing useful on your Mac (the macOS version of `ls`
is older and uses `man ls` instead), do not worry. The lesson stands: every tool has a built-in
manual, and reading it is how professionals work. They look things up constantly. So should you.

The eight ideas above (`pwd`, `ls`, `cd`, `mkdir`, absolute vs relative paths, `.` and `..`, and
`--help`) cover almost everything you will type at a terminal during this course. To go deeper at
your own pace, the gentlest free course is the **Software Carpentry "Unix Shell"** lesson,
episodes 1 to 3: <https://swcarpentry.github.io/shell-novice/>.""")

# ----------------------------------------------------------------------------
# Section 2: git & GitHub
# ----------------------------------------------------------------------------
md(r"""## 2. git and GitHub, the lab notebook of code

A wet-lab notebook records every step so the work can be trusted and repeated. **git** is the
same thing for code and text files: it tracks the history of a folder. Every time you save a
meaningful state, git stores a snapshot called a **commit**, with a message saying what changed
and who changed it. You can always look back at any past commit, see what was different, and
restore it. Nothing is ever silently lost. That is why git is the lab notebook of the digital
world: an honest, time-stamped record of how your analysis came to be.

**GitHub** is a website that hosts git histories online so people can share them. Our course
lives there as a **repository** (a "repo"): one folder, tracked by git, published on GitHub at
`github.com/lamanno-epfl/lipidomics_tutorial_cajalcourse`. You already copied it to your laptop
in setup with one command:

```bash
git clone https://github.com/lamanno-epfl/lipidomics_tutorial_cajalcourse_students.git
```

`git clone` downloads the whole repo *and* its full history in one go. You only do it once.""")

md(r"""### the four git verbs you actually need

You will not branch or merge during this course. Five commands cover everything:

- `git clone <url>` — copy a repo from GitHub to your laptop (done once, in setup).
- `git status` — *what has changed since my last snapshot?* Your most-used command. It tells you
  which files you edited, which are new, and what git is about to record.
- `git pull` — *fetch the latest version from GitHub and merge it in.* **This is how you receive
  updates from me during the course**, including the solution notebooks. Run it at the start of
  each session.
- `git add <file>` then `git commit -m "message"` — take a snapshot. `add` chooses what goes in;
  `commit` records it with a message. Together they are how you save your own progress.

Let us look at the live history of *this* repo. The next cell asks git to show the working tree
status and the most recent commits. These are read-only: they change nothing.""")

code(r"""# git status: a summary of what changed since the last commit. Safe, read-only.
# -c color.ui=false turns OFF the colour codes so the saved output stays clean text.
!git -c color.ui=false status --short --branch""", tag="keep")

code(r"""# git log: the lab notebook itself. --oneline prints one commit per line;
# -n 3 keeps the three most recent. Each line is a snapshot someone saved.
# Again -c color.ui=false keeps the committed output free of colour escape codes.
!git -c color.ui=false log --oneline -n 3""", tag="keep")

md(r"""⚠️ **CHECKPOINT** — `git status` printed a branch line (probably `## main`) and `git log`
printed at least one commit with a short code and a message. You are reading the actual history
of the folder you are sitting in. That short code is the commit's fingerprint; git can return to
any of them.

❓ **QUESTION** — if you edit this notebook and run `git status` again, what do you expect to
change in its output? (Answer: the notebook file appears as "modified". Try it later and see.)""")

md(r"""💡 **HINT — the one git habit that saves you** — at the start of every session, before you
touch anything, run `git pull`. It pulls down whatever I pushed (new notebooks, fixes, the
solutions). If you have edited a file I also changed, git may report a *conflict*; if that happens
during the course, just ask me or Claude Code, and do not panic, because git never throws your
work away.

To understand git properly when you have an hour, the **Software Carpentry "Version Control with
Git"** lesson is written for scientists and is excellent: <https://swcarpentry.github.io/git-novice/>.
A shorter conceptual primer is Aalto Scientific Computing's git intro:
<https://scicomp.aalto.fi/scicomp/git/>.""")

# ----------------------------------------------------------------------------
# Section 3: conda / mamba environments
# ----------------------------------------------------------------------------
md(r"""## 3. environments, or why your code still runs next year

Here is a problem that bites every computational biologist eventually. Project A needs version 1
of some library. Project B needs version 2. If both projects share one global Python, installing
version 2 for B silently breaks A. Worse, a colleague runs your code, has slightly different
versions of fifty libraries, and gets different numbers. The analysis is no longer reproducible,
and you cannot even tell why.

An **environment** solves this. It is a self-contained, isolated box that holds one specific
Python plus one specific set of library versions, walled off from every other project. Project A
gets its box, Project B gets its box, and they never interfere. This course gets a box called
`cajal-lipidomics`, which you created in setup. The tool that builds and manages these boxes is
**conda**; **mamba** is a faster drop-in replacement that understands the exact same commands. You
installed both at once via **Miniforge**.

Two ideas make environments worth the trouble:

- **Isolation** — each project's versions cannot break another project's. The box keeps them apart.
- **Pinning** — we write down the exact versions in a file (`environment.yml` in this repo). Anyone,
  on any machine, can rebuild the *same* box from that file. Pinned versions are why the figures you
  make this week will reproduce next year and on my laptop too. It is the conda/mamba equivalent of
  recording lot numbers for your reagents.""")

md(r"""### looking inside your environment

You opened this notebook with the `cajal-lipidomics` kernel (section 4 explains what a kernel is),
so the Python running these cells *is* the one inside that box. Let us prove it and peek at a few of
the pinned versions. We do this from Python rather than the shell, because here Python is the cleaner
way to ask "which exact versions am I running?"

🔬 **TASK** — the next cell is **yours to write** in the student copy: import `numpy`, `pandas`, and
`scanpy`, then print each one's version. The master copy shows one clean way to do it.""")

code(r"""# TASK: import the core libraries and print their versions.
import sys
import importlib.metadata as md   # the modern, warning-free way to read a package version

print("Python executable:", sys.executable)        # the path proves WHICH python is running
print("Python version:   ", sys.version.split()[0])
for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "scanpy", "anndata"]:
    print(f"{pkg:>13}: {md.version(pkg)}")""", tag="task")

md(r"""⚠️ **CHECKPOINT** — six versions print without error, and the "Python executable" path points
into a Miniforge `envs` folder named `cajal-lipidomics`. If instead it points at a system Python or
`base`, you have the wrong kernel selected, which is exactly the trap section 4 warns about. Fix it
there before continuing.

❓ **QUESTION** — why print the versions at all? Because months from now, if a number looks off, the
first question is always "did a library version change?" Printing them at the top of an analysis is a
small habit that turns a baffling bug into a five-second check.""")

md(r"""💡 **HINT** — a few environment commands you will use, all to be typed in the **terminal**
(not in a cell), with the env active:

```bash
mamba activate cajal-lipidomics   # step into the box; do this every new terminal
mamba list                        # show every installed package and its version
mamba env export > my_env.yml     # write the exact recipe to a file, to share or archive
```

The friendliest references are the **Miniforge README** (<https://github.com/conda-forge/miniforge>)
and Harvard FAS Informatics' beginner guide to conda/mamba
(<https://informatics.fas.harvard.edu/resources/tutorials/installing-command-line-software-conda-mamba/>).""")

# ----------------------------------------------------------------------------
# Section 4: Jupyter + VSCode
# ----------------------------------------------------------------------------
md(r"""## 4. Jupyter and VS Code, where you actually work

This file is a **Jupyter notebook**. A notebook interleaves two kinds of cells: **markdown cells**
like this one, which hold formatted prose, and **code cells**, which hold runnable code and show
their output right underneath. That mix is why notebooks suit science: explanation, code, and
results live together in one readable document, which is also why your eventual analysis reads like
a story rather than a wall of script.

**VS Code** (Visual Studio Code) is the editor we use to open and run notebooks. You installed it in
setup, with the **Python** and **Jupyter** extensions. You could also use JupyterLab in a browser;
the concepts below are identical either way.""")

md(r"""### cells, and the thing that trips up everyone: run order

You run a code cell by clicking it and pressing `Shift`+`Enter`. It executes, prints any output
below itself, and moves the highlight to the next cell. The number in brackets to the left, like
`[3]`, is the **execution count**: it tells you the *order* in which cells were actually run, not the
order they appear on screen.

That gap between "order on screen" and "order you ran them" is the number-one beginner stumble. A
notebook keeps a single shared memory. If you define a variable in cell 5, then scroll up and run
cell 2 which uses it, cell 2 works, because the variable is still in memory from before, even though
it sits *above* the definition on the page. Run the notebook fresh from the top and cell 2 would
fail. Your notebook can therefore look correct while being silently broken for anyone who runs it
top to bottom.

The cure is one habit: when in doubt, **Restart and Run All** (in VS Code, the kernel menu, or the
double-arrow ⏩ button). It wipes memory and runs every cell from the top in order, which is exactly
how a reader or a grader will run it. A notebook that survives Restart-and-Run-All is a notebook that
actually works.

Let us watch shared memory in action. Run the next two cells in order.""")

code(r"""# Cell A: define a value. Note the execution count [n] that appears to the left after running.
n_pixels = 1200
print("defined n_pixels =", n_pixels)""", tag="keep")

code(r"""# Cell B: use the value defined in Cell A. It works because the notebook shares one memory.
# The lipid count here is just an illustration; nothing scientific yet.
n_lipids = 173
print(f"a toy MALDI section: {n_pixels} pixels x {n_lipids} lipids "
      f"= {n_pixels * n_lipids:,} numbers")""", tag="keep")

md(r"""⚠️ **CHECKPOINT** — the second cell printed `a toy MALDI section: 1200 pixels x 173 lipids =
207,600 numbers`. Look at the `[ ]` counts on the left: Cell A has a smaller number than Cell B,
proving A ran first. That ordering is what Restart-and-Run-All guarantees for the whole notebook.

🔬 **TASK** — try the failure mode safely. Scroll up, re-run Cell A alone, then re-run Cell B. Watch
both execution counts jump. Now imagine you had *changed* `n_pixels` in between. The output would
reflect whichever value was in memory last, not what the page seems to say. That is the whole hazard,
and Restart-and-Run-All is the whole cure.""")

md(r"""### the kernel, and choosing the right one

Behind every open notebook runs a **kernel**: the actual Python process that executes your cells and
holds the shared memory. When you select a kernel, you are choosing *which Python* runs your code, and
therefore which environment's libraries are available.

This is the single most common thing beginners get wrong, so it is worth saying plainly. You can
have a perfect `cajal-lipidomics` environment and still get `ModuleNotFoundError: No module named
'scanpy'`, purely because the notebook is pointed at the wrong kernel (often the base or system
Python, where scanpy is not installed). The code is fine. The kernel is wrong.

In VS Code the kernel picker sits at the **top right** of the notebook. Click it and choose
**cajal-lipidomics**. The cell below confirms which kernel you are actually on. If the path does not
mention `cajal-lipidomics`, switch kernels now, before notebook 01.""")

code(r"""# Confirm the live kernel. This is the same idea as in section 3, repeated on purpose:
# the kernel IS the environment, and beginners conflate "I created the env" with "I selected it".
import sys
from pathlib import Path

# Where does THIS kernel's python live? The path is the honest source of truth: it does not
# depend on which env your terminal happened to have active when the kernel started. An env
# python sits in .../envs/<env_name>/bin/python, so the parent of "bin" is the env folder.
exe = Path(sys.executable)
env_name = exe.parent.parent.name   # .../envs/cajal-lipidomics/bin/python -> "cajal-lipidomics"
print("kernel python:   ", exe)
print("environment name:", env_name)

# The honest test of "right setup" is whether the course libraries import, not the literal
# folder name (your Miniforge path may differ). If scanpy imports, the box is the right box.
try:
    import scanpy  # the heaviest course dependency; if this works, the env is built and selected
    print("scanpy imports cleanly -> this kernel can run the course. Good.")
except ModuleNotFoundError:
    print("scanpy is MISSING -> wrong kernel. Use the picker (top-right in VS Code)")
    print("and choose cajal-lipidomics, then run this cell again.")""", tag="keep")

md(r"""⚠️ **CHECKPOINT** — the cell printed `scanpy imports cleanly -> this kernel can run the
course`, and the environment name read `cajal-lipidomics`, derived straight from the kernel's own
python path. If it shows some other name but scanpy still imports, you simply named your env
differently, which is fine. If instead it said scanpy is missing, you are on the wrong kernel:
switch with the picker and rerun.

💡 **HINT** — the official, screenshot-rich guides are Microsoft's **"Jupyter Notebooks in VS
Code"** (<https://code.visualstudio.com/docs/datascience/jupyter-notebooks>) and the **"Data Science
in VS Code"** tutorial (<https://code.visualstudio.com/docs/datascience/data-science-tutorial>). If
`cajal-lipidomics` does not appear in the kernel list at all, you likely skipped the
`python -m ipykernel install` line in [`docs/SETUP.md`](../../docs/SETUP.md) step 3; run it and reopen
the notebook.""")

# ----------------------------------------------------------------------------
# Section 5: Claude Code
# ----------------------------------------------------------------------------
md(r"""## 5. Claude Code, a pair-programmer you keep on a leash

From the second half of the course we use **Claude Code**, an AI assistant built by Anthropic (the
makers of the Claude models) that works *inside* your project. You run it by typing `claude` in a
terminal that is sitting in the course folder, or through its VS Code integration. Unlike a chat
window in a browser, Claude Code can actually read the files in your repo, so it can explain the
specific code in front of you and propose edits to it.

Use it for three things, in roughly this order of value:

- **Ask it to explain.** Paste a line you do not understand, or point it at a cell, and ask "what
  does this do, and why?" This is the best use by far. It turns a wall of unfamiliar code into a
  tutor that never tires.
- **Ask it to write small things.** Plotting code is the classic example: "I have a numpy array
  `intensities` and a pandas Series `region`; make a violin plot split by region with matplotlib."
  It will produce working code in seconds.
- **Let it edit, with permission.** Claude Code **asks before it changes any file.** You see exactly
  what it wants to modify and you approve or decline. Nothing is edited behind your back. Keep it
  that way: read the diff before you accept it.""")

md(r"""### the rule that makes AI help instead of harm

Here is the lighthouse for this whole course, and it is not optional:

> **If you do not understand a line the AI produced, stop and look it up.**

AI can generate code faster than you can read it. The temptation is to accept, run, get a plot, and
move on. Resist it. The goal of these notebooks is for *you* to understand the analysis, because in
the project you will defend your figures and your conclusions, and "the AI wrote it" is not a result.
Treat Claude Code like a brilliant, fast labmate whose work you always check: wonderful for
explaining and drafting, never a substitute for your own understanding. Verify, do not trust.

There is nothing to run in this section. To learn the tool, the official **Claude Code Quickstart** is
short and clear: <https://code.claude.com/docs/en/quickstart>. We will use it together when the
analysis gets real.""")

# Tiny non-AI, fully-deterministic demonstrative cell so the section still has a runnable artefact
# that ties back to "verify, don't trust": a self-check the student can read line by line.
code(r"""# A "verify, don't trust" habit you can apply to any code, AI-written or not:
# read it, predict the output, THEN run and compare. Try predicting before you run this.
def lipid_label(head_group, n_carbons, n_double_bonds):
    # Lipid shorthand: head group, then carbons:double-bonds. e.g. PC 34:1
    return f"{head_group} {n_carbons}:{n_double_bonds}"

print(lipid_label("PC", 34, 1))   # predict this first, then check
print(lipid_label("PE", 36, 2))""", tag="keep")

md(r"""⚠️ **CHECKPOINT** — it printed `PC 34:1` and `PE 36:2`. Did your prediction match? That tiny
loop, predict then run then compare, is the entire discipline of working with code you did not write
yourself. (And you just met lipid shorthand: head group, carbons, colon, double bonds. Notebook 02
returns to it.)""")

# ----------------------------------------------------------------------------
# Wrap-up
# ----------------------------------------------------------------------------
md(r"""## you made it: the bench is yours

You can now do the things every later notebook quietly assumes:

- **terminal** — `pwd` to see where you are, `ls` to look around, `cd` to move, `mkdir` to make a
  folder, and `--help` when memory fails. You know an absolute path from a relative one.
- **git** — the lab notebook of code. `git status` to see changes, `git pull` to receive my updates
  and the solutions, `git add`/`git commit` to save your own snapshots.
- **environments** — the isolated, version-pinned box `cajal-lipidomics` that makes your work
  reproducible on any machine and next year.
- **Jupyter + VS Code** — code and prose in one document; run with `Shift`+`Enter`; respect run order
  and trust **Restart-and-Run-All**; and above all, **select the right kernel**.
- **Claude Code** — ask-to-explain first, let-it-edit with permission, and the rule that outranks all
  others: if you do not understand a line, stop and look it up.

For the full reference on installs and the per-step success checks, your home base is always
[`docs/SETUP.md`](../../docs/SETUP.md).""")

md(r"""## a final self-check, then onward

Run the cell below. It is a friendly summary that confirms the two things that must be true before
notebook 01: you are on the course environment, and the core scientific libraries import. If it prints
all green, you are genuinely ready.""")

code(r"""# Final readiness check: do the course libraries import? Nothing scientific, just confidence.
# If they all import, this kernel can run every later notebook, whatever your env is named.
import sys
from pathlib import Path

# Same trick as section 4: read the env name off the kernel's own python path, which is honest
# regardless of what your terminal had active when the kernel launched.
env_name = Path(sys.executable).parent.parent.name
print(f"environment: {env_name}\n")

checks = []
for pkg in ["numpy", "pandas", "matplotlib", "scipy", "sklearn", "scanpy", "anndata"]:
    try:
        __import__(pkg)
        checks.append((pkg, True))
    except Exception:
        checks.append((pkg, False))

for name, ok in checks:
    print(f"  {'OK ' if ok else 'XX '} import {name}")
all_ok = all(ok for _, ok in checks)
print("\nReady for notebook 01." if all_ok
      else "\nSome imports failed (XX) -> almost always the wrong kernel. See section 4.")""", tag="keep")

md(r"""⚠️ **CHECKPOINT** — every line reads `OK import ...`, and the last line says **Ready for
notebook 01**. The environment name should read `cajal-lipidomics`. If a library shows `XX`, you are
almost certainly on the wrong kernel; revisit section 4. If `numpy` and friends genuinely will not
import even on the right kernel, the environment did not build; see
[`docs/SETUP.md`](../../docs/SETUP.md) step 3 or email **luca.fusarbassini@epfl.ch**.

### what comes next

**Notebook 01 — speaking Python for data.** Now that the bench is set up, you learn the language you
will actually compute in: variables and lists, then numpy arrays, pandas tables, and matplotlib
figures, all on a tiny synthetic pixels-by-lipids table that previews the real MALDI data. Open
`01_python_for_data.ipynb` in this same folder, and remember to select the `cajal-lipidomics` kernel.

See you there.""")

# ----------------------------------------------------------------------------
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "cajal-lipidomics",
        "language": "python",
        "name": "cajal-lipidomics",
    },
    "language_info": {"name": "python"},
}

out = "/home/fusar/lipidomics_tutorial_cajalcourse/notebooks/00_intro/00_tooling.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"wrote {out} with {len(cells)} cells")
