"""Run the uMAIA normalization on the two course sections (cajal-umaia env).

Loads data/umaia_input.npz (the matched (N,S,V) log-intensity tensor + mask built by
build_umaia_input.py), fits the uMAIA model by MAP/SVI, applies the histogram-matching
transform, and saves the normalized tensor + fitted parameters. This is the real fit
students run in notebook 3.

    JAX_PLATFORMS=cpu python scripts/run_umaia.py
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uMAIA

NUM_STEPS = 2000
SEED = 42
PARAM_DIR = "data/umaia_params"


def main():
    d = np.load("data/umaia_input.npz", allow_pickle=True)
    x, mask = d["x"], d["mask"]
    molecules = list(d["molecules"]); sections = list(d["sections"])
    print(f"input: x {x.shape} (N,S,V) | {len(molecules)} molecules | sections {sections}")

    print("initialize (GMM per molecule)...")
    init = uMAIA.norm.initialize(x, mask, subsample=True)
    print("normalize (MAP via SVI, %d steps)..." % NUM_STEPS)
    svi = uMAIA.norm.normalize(x, mask, init_state=init, subsample=True, num_steps=NUM_STEPS, seed=SEED)
    os.makedirs(PARAM_DIR, exist_ok=True)
    uMAIA.ut.tools.save_svi(svi, PARAM_DIR)
    print("transform (histogram matching)...")
    x_maia = np.asarray(uMAIA.norm.transform(x, mask, PARAM_DIR))
    np.savez_compressed("data/umaia_normalized.npz", x_MAIA=x_maia,
                        molecules=np.array(molecules), sections=np.array(sections))
    print("saved data/umaia_normalized.npz:", x_maia.shape)

    # before/after: per-section histograms for a few molecules (the money plot)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    pick = np.linspace(0, len(molecules) - 1, 4).astype(int)
    for col, v in enumerate(pick):
        for s in range(x.shape[1]):
            m = mask[:, s, v]
            axes[0, col].hist(x[m, s, v], bins=60, density=True, alpha=0.5, label=sections[s])
            axes[1, col].hist(x_maia[m, s, v], bins=60, density=True, alpha=0.5, label=sections[s])
        axes[0, col].set_title(f"{molecules[v]}\nraw", fontsize=8)
        axes[1, col].set_title("uMAIA-normalized", fontsize=8)
        for r in (0, 1):
            axes[r, col].set_yticks([]); axes[r, col].legend(fontsize=6, frameon=False)
    fig.suptitle("uMAIA aligns each molecule's per-section distributions", fontsize=12)
    plt.tight_layout()
    plt.savefig("data/figs/08_umaia_before_after.png", dpi=130, bbox_inches="tight")
    print("saved data/figs/08_umaia_before_after.png")

    # quantify: cross-section foreground-mode gap before vs after (should shrink)
    def fg_gap(arr):
        gaps = []
        for v in range(arr.shape[2]):
            ms = [arr[mask[:, s, v], s, v] for s in range(arr.shape[1])]
            hi = [np.quantile(a, 0.9) for a in ms if len(a)]
            if len(hi) == 2:
                gaps.append(abs(hi[0] - hi[1]))
        return float(np.median(gaps))
    print(f"median cross-section 90th-pct gap: raw {fg_gap(x):.3f} -> normalized {fg_gap(x_maia):.3f}")


if __name__ == "__main__":
    main()
