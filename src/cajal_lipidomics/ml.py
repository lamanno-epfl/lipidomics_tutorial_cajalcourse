"""A small neural network that predicts a pixel's CCF position from its lipid profile.

This mirrors the Lipid Brain Atlas Lipid2Position model (assets/lipid2position.py, a torch MLP
input -> 1024 -> 512 -> 256 -> 128 -> 64 -> 3). Here we use scikit-learn's MLPRegressor so it runs
on any CPU with no extra dependencies; the idea is identical. It is a worked example of NONLINEAR
regression and a first taste of machine learning: the lipidome carries enough information to place
a pixel in the brain, and a neural net learns that nonlinear map.

Crucially this is a real generalization test, exactly as in the atlas: we TRAIN on one section and
DEPLOY on the OTHER, and we use a single hemisphere so the brain's bilateral symmetry cannot let the
model cheat by predicting a mirrored position. The target is the in-plane CCF position (yccf, zccf):
xccf is essentially constant within one coronal section, so it is not a meaningful target.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


def predict_position(adata, layer="umaia", coord_keys=("yccf", "zccf"),
                     cond_key="Condition", train_cond="naive", test_cond="pregnant",
                     hemisphere="left", hidden=(256, 128, 64), max_iter=300, random_state=0):
    """Train an MLP on the TRAIN section's chosen hemisphere to predict CCF position from the
    lipidome, then deploy it on the TEST section's same hemisphere.

    train_cond/test_cond pick the two sections (control vs pregnant). `hemisphere` keeps one side
    only (split on zccf, the medio-lateral axis) so bilateral symmetry cannot make the map ambiguous.
    Returns the held-out true/predicted positions, per-axis Pearson r, and the masks' sizes.
    """
    obs = adata.obs
    X = np.asarray(adata.layers[layer] if (layer and layer in adata.layers) else adata.X, float)
    Y = np.column_stack([obs[c].to_numpy(float) for c in coord_keys])
    zmid = float(np.median(obs["zccf"].to_numpy(float)))
    side = (obs["zccf"].to_numpy(float) < zmid) if hemisphere == "left" else (obs["zccf"].to_numpy(float) >= zmid)
    tr = (obs[cond_key].to_numpy() == train_cond) & side          # train: section 1, one hemisphere
    te = (obs[cond_key].to_numpy() == test_cond) & side           # deploy: section 2, same hemisphere
    sc = StandardScaler().fit(X[tr])
    net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", random_state=random_state,
                       max_iter=max_iter, early_stopping=True)
    net.fit(sc.transform(X[tr]), Y[tr])
    pred = net.predict(sc.transform(X[te]))
    r = {c: float(pearsonr(pred[:, i], Y[te][:, i])[0]) for i, c in enumerate(coord_keys)}
    return {"true": Y[te], "pred": pred, "r": r, "coord_keys": list(coord_keys),
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "model": net}
