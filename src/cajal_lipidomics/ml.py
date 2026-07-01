"""A small neural network that predicts a pixel's CCF position from its lipid profile.

This mirrors the Lipid Brain Atlas Lipid2Position model (assets/lipid2position.py, a torch MLP
input -> 1024 -> 512 -> 256 -> 128 -> 64 -> 3). Here we use scikit-learn's MLPRegressor so it runs
on any CPU with no extra dependencies; the idea is identical. It is a worked example of NONLINEAR
regression and a first taste of machine learning: the lipidome carries enough information to place
a pixel in the brain, and a neural net learns that nonlinear map.

We work on one section and one hemisphere, holding out a random fraction of those pixels the model
never sees while training, then score on those. Using a single hemisphere avoids the brain's
bilateral symmetry (a pixel and its mirror twin carry near-identical lipids but opposite zccf). The
target is the in-plane CCF position (yccf, zccf): xccf is essentially constant within one coronal
section, so it is not a meaningful target.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr


def predict_position(adata, layer="umaia", coord_keys=("yccf", "zccf"),
                     cond_key="Condition", section="naive", hemisphere="right", test_size=0.25,
                     hidden=(256, 128, 64), max_iter=300, random_state=0):
    """Train an MLP on one section's chosen hemisphere to predict in-plane CCF position from the
    lipidome, holding out a random `test_size` fraction of those pixels the model never sees, then
    score on them.

    `section` picks the section (a value of `cond_key`, e.g. the control 'naive'); `hemisphere` keeps
    one side only (split on zccf about its within-section median), so bilateral symmetry cannot make
    the map ambiguous. Returns the held-out true/predicted positions, per-axis Pearson r, and sizes.
    """
    obs = adata.obs
    X = np.asarray(adata.layers[layer] if (layer and layer in adata.layers) else adata.X, float)
    Y = np.column_stack([obs[c].to_numpy(float) for c in coord_keys])
    sec = (obs[cond_key].to_numpy() == section)                   # one section only
    z = obs["zccf"].to_numpy(float)
    zmid = float(np.median(z[sec]))                               # midline within this section
    side = (z >= zmid) if hemisphere == "right" else (z < zmid)   # keep one hemisphere only
    keep = sec & side
    Xs, Ys = X[keep], Y[keep]
    itr, ite = train_test_split(np.arange(len(Xs)), test_size=test_size, random_state=random_state)
    sc = StandardScaler().fit(Xs[itr])
    net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", random_state=random_state,
                       max_iter=max_iter, early_stopping=True)
    net.fit(sc.transform(Xs[itr]), Ys[itr])
    pred = net.predict(sc.transform(Xs[ite]))
    r = {c: float(pearsonr(pred[:, i], Ys[ite][:, i])[0]) for i, c in enumerate(coord_keys)}
    return {"true": Ys[ite], "pred": pred, "r": r, "coord_keys": list(coord_keys),
            "n_train": int(len(itr)), "n_test": int(len(ite)), "model": net}
