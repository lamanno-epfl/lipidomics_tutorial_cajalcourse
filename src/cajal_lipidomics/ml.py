"""A small neural network that predicts a pixel's CCF position from its lipid profile.

This mirrors the Lipid Brain Atlas Lipid2Position model (assets/lipid2position.py, a torch MLP
input -> 1024 -> 512 -> 256 -> 128 -> 64 -> 3). Here we use scikit-learn's MLPRegressor so it runs
on any CPU with no extra dependencies; the idea is identical. It is a worked example of NONLINEAR
regression and a first taste of machine learning: the lipidome carries enough information to place
a pixel in the brain, and a neural net learns that nonlinear map.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr


def predict_position(adata, layer=None, coord_keys=("xccf", "yccf", "zccf"),
                     hidden=(256, 128, 64), max_iter=200, random_state=0, test_size=0.25):
    """Train an MLP to predict CCF coordinates from the lipid profile. Returns a dict with the
    held-out true/predicted coordinates and the per-axis Pearson r (the ML analogue of R)."""
    X = np.asarray(adata.layers[layer] if layer else adata.X, float)
    Y = np.column_stack([adata.obs[c].to_numpy(float) for c in coord_keys])
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    sc = StandardScaler().fit(Xtr)
    net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", random_state=random_state,
                       max_iter=max_iter, early_stopping=True)
    net.fit(sc.transform(Xtr), Ytr)
    pred = net.predict(sc.transform(Xte))
    r = {c: float(pearsonr(pred[:, i], Yte[:, i])[0]) for i, c in enumerate(coord_keys)}
    return {"true": Yte, "pred": pred, "r": r, "coord_keys": list(coord_keys), "model": net}
