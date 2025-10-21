"""
visualization_tools.py
----------------------
Visualization utilities for analyzing feature space geometry and redundancy.
Includes PCA 2D/3D, t-SNE, and correlation heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def tsne_plot(X_syn, X_real, title="t-SNE: Synthetic vs Real"):
    """2D t-SNE plot to visualize domain shift."""
    X = np.vstack([X_syn, X_real])
    y = np.array([0]*len(X_syn) + [1]*len(X_real))
    X_embedded = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)

    plt.figure(figsize=(6,6))
    plt.scatter(X_embedded[y==0,0], X_embedded[y==0,1], 
                c="#1f77b4", alpha=0.6, label="Synthetic", s=30)
    plt.scatter(X_embedded[y==1,0], X_embedded[y==1,1], 
                c="#d62728", alpha=0.6, label="Real", s=30)
    plt.xlabel("t-SNE Dim 1"); plt.ylabel("t-SNE Dim 2")
    plt.legend(frameon=False)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig("./figures/tsne_synthetic_vs_real.png", dpi=300)
    plt.show()


def pca_3d_plot(X_blocks, labels=["TDA","RMT","Spectral"]):
    """3D PCA plot of combined feature space highlighting block contributions."""
    X = np.hstack(X_blocks)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3).fit(Xs)
    loadings = pca.components_.T

    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    sizes = [X_blocks[0].shape[1], X_blocks[1].shape[1], X_blocks[2].shape[1]]
    idx_ranges = np.cumsum([0] + sizes)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")

    for i, (label, color) in enumerate(zip(labels, colors)):
        start, end = idx_ranges[i], idx_ranges[i+1]
        ax.scatter(loadings[start:end,0], loadings[start:end,1], loadings[start:end,2],
                   s=40, alpha=0.7, color=color, label=label)

    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_zlabel("PC3", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("./figures/pca_3d_blocks.png", dpi=300)
    plt.show()


def correlation_heatmap(matrix, labels=None, title="Correlation Heatmap"):
    """Display correlation matrix as heatmap."""
    plt.figure(figsize=(6,5))
    sns.heatmap(matrix, cmap="coolwarm", annot=True, fmt=".2f",
                xticklabels=labels, yticklabels=labels, cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("./figures/correlation_heatmap.png", dpi=300)
    plt.show()
