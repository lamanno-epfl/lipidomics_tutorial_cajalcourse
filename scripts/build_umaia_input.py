"""Build the uMAIA normalization input for the two course sections (run in base env).

uMAIA normalizes the SAME molecule across sections, so we:
  1. pull both sections' raw ion images from METASPACE (CoreMetabolome v3, FDR 0.1),
  2. match molecules across sections by (formula, adduct),
  3. mask to tissue (the provided mask.npy), log-transform,
  4. stack into a dense (N_pixels, S_sections, V_molecules) tensor + boolean mask,
and save to data/umaia_input.npz. Notebook 3 (cajal-umaia env) loads this and runs the
actual uMAIA fit + transform. This pulls from METASPACE so it runs in the base/cajal-lipidomics
env (uMAIA itself is not needed here).

    python scripts/build_umaia_input.py
"""
from __future__ import annotations

import time
import numpy as np
from metaspace import SMInstance

DB = ("CoreMetabolome", "v3")
FDR = 0.1
EPS = 2e-4
SECTIONS = {
    "control_217D": dict(ds_id="2025-04-27_08h23m47s",
                         mask="data/masks/BrainAtlas/Control_Brains/female/20220416_MouseBrain_female_217D_447x332_Att30_25um/mask.npy"),
    "pregnant_Brain1_C2": dict(ds_id="2024-07-14_14h24m11s",
                              mask="data/masks/PREGNANT/20240712_MouseBrain_LipidAtlas_Pregnant_Brain1_C2_459x352_25um_Att30/mask.npy"),
}


def pull_images(ds, tries=4):
    """all_annotation_images with retry (the client's parallel PNG fetch is transiently flaky)."""
    last = None
    for k in range(tries):
        try:
            return ds.all_annotation_images(fdr=FDR, database=DB, only_first_isotope=True,
                                            scale_intensity=False, hotspot_clipping=False)
        except Exception as e:  # transient "not a PNG file"
            last = e; time.sleep(2 * (k + 1))
    raise last


def main():
    sm = SMInstance()
    per = {}
    for name, cfg in SECTIONS.items():
        ds = sm.dataset(id=cfg["ds_id"])
        res = ds.results(database=DB, fdr=FDR)
        imgs = pull_images(ds)
        # key each annotation by (formula, adduct) from the results MultiIndex
        keys = [f"{f}_{a}" for f, a in res.index]
        arrs = {k: np.asarray(s[0], dtype=np.float32) for k, s in zip(keys, imgs)}
        mask2d = np.load(cfg["mask"])
        per[name] = dict(arrs=arrs, mask=mask2d)
        print(f"{name}: {len(arrs)} molecules, image {next(iter(arrs.values())).shape}, "
              f"mask {mask2d.shape}, tissue px {int(mask2d.sum())}")

    names = list(SECTIONS)
    common = sorted(set(per[names[0]]["arrs"]).intersection(per[names[1]]["arrs"]))
    print(f"common molecules across both sections: {len(common)}")

    # per section: tissue-pixel x molecule, log-transformed
    cols = {}
    for name in names:
        mask = per[name]["mask"].ravel().astype(bool)
        cols[name] = np.stack([np.log(per[name]["arrs"][k].ravel()[mask] + EPS) for k in common], axis=1)
        print(f"{name}: tensor {cols[name].shape}")

    N = max(c.shape[0] for c in cols.values())
    S, V = len(names), len(common)
    x = np.zeros((N, S, V), np.float32)
    mask = np.zeros((N, S, V), bool)
    for s, name in enumerate(names):
        n = cols[name].shape[0]
        x[:n, s, :] = cols[name]
        mask[:n, s, :] = True

    np.savez_compressed("data/umaia_input.npz", x=x, mask=mask,
                        molecules=np.array(common), sections=np.array(names), epsilon=EPS)
    print(f"saved data/umaia_input.npz: x {x.shape}, mask {mask.shape}, {V} molecules, {S} sections")


if __name__ == "__main__":
    main()
