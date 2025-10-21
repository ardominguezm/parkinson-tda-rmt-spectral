"""
redundancy_analysis.py
----------------------
Performs correlation, PCA, and weighted fusion between TDA, RMT, and spectral feature spaces.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from itertools import product
import matplotlib.pyplot as plt

def pca_block_summary(X_blocks, names):
    rep = {}
    for n, X in zip(names, X_blocks):
        pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X)).ravel()
        rep[n + "_PC1"] = pc1
    df = pd.DataFrame(rep)
    print("Correlations between PC1s:\n", df.corr())
    return df


def global_pca(X_blocks):
    X_all = np.hstack(X_blocks)
    Xs = StandardScaler().fit_transform(X_all)
    pca = PCA(n_components=0.95)
    Xp = pca.fit_transform(Xs)
    print(f"\nExplained variance (95%): {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Components retained: {Xp.shape[1]} / {X_all.shape[1]}")
    return pca, Xp


def weighted_fusion(X_blocks, y_train, y_test, grid=(0, 0.25, 0.5, 0.75, 1.0)):
    best = (-1, None)
    for w in product(grid, repeat=3):
        if np.isclose(sum(w), 0.0): continue
        X_train_f = np.hstack([wi * StandardScaler().fit_transform(X) for wi, X in zip(w, X_blocks)])
        X_test_f = np.hstack([wi * StandardScaler().fit_transform(X) for wi, X in zip(w, X_blocks)])
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_f, y_train)
        acc = (model.predict(X_test_f) == y_test).mean()
        if acc > best[0]:
            best = (acc, w)
    print(f"Best weighted fusion: ACC={best[0]:.3f} with weights (TDA,RMT,SPEC)={best[1]}")


def plot_3d_pca(X_blocks, labels=["TDA","RMT","Spectral"]):
    X = np.hstack(X_blocks)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3).fit(Xs)
    load = pca.components_.T
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    n1, n2 = X_blocks[0].shape[1], X_blocks[1].shape[1]
    splits = [range(0,n1), range(n1,n1+n2), range(n1+n2, X.shape[1])]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    for i, group in enumerate(labels):
        ax.scatter(load[list(splits[i]),0], load[list(splits[i]),1], load[list(splits[i]),2],
                   color=colors[i], alpha=0.7, label=group)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("./figures/pca_3d_feature_overlap.png", dpi=300)
    plt.show()
