"""
feature_extraction.py
---------------------
Implements feature extraction functions for TDA, RMT, and Spectral descriptors.
Used in Parkinson voice classification.
"""

import numpy as np
import pandas as pd
from ripser import ripser
from persim import PersistenceImager
from skimage.transform import resize
import librosa

# =============== TDA FUNCTIONS ===============

def takens_embedding(signal, dimension, delay):
    N = len(signal)
    if N - (dimension-1)*delay <= 0:
        raise ValueError("Signal too short for embedding.")
    return np.array([signal[i:N-(dimension-1)*delay+i:delay] for i in range(dimension)]).T


def compute_persistence(signal, dimension, delay):
    embedded = takens_embedding(signal, dimension, delay)
    diagrams = ripser(embedded)['dgms']
    return diagrams


def get_PI(signal, dimension, delay, pixel_size=0.1, resolution=(30,30)):
    """Persistence Image."""
    dgm = compute_persistence(signal, dimension, delay)[1]
    if len(dgm)==0:
        return np.zeros(resolution[0]*resolution[1])
    pimg = PersistenceImager(pixel_size=pixel_size)
    pimg.fit(dgm)
    img = pimg.transform(dgm)
    return resize(img, resolution, anti_aliasing=True).flatten()


def get_PL(signal, dimension, delay, num_points=50, levels=3):
    """Persistence Landscape (vectorized)."""
    dgm = compute_persistence(signal, dimension, delay)[1]
    if len(dgm)==0:
        return np.zeros(num_points*levels)
    x_max = np.max(dgm[:,1])
    grid = np.linspace(0, x_max, num_points)
    landscapes = np.zeros((levels, num_points))
    for i, x in enumerate(grid):
        vals = [max(0, min(x - b, d - x)) for (b, d) in dgm]
        vals = np.sort(vals)[::-1]
        for lvl in range(levels):
            landscapes[lvl, i] = vals[lvl] if lvl < len(vals) else 0
    return landscapes.flatten()


def get_PE(signal, dimension, delay):
    """Persistent Entropy."""
    dgm = compute_persistence(signal, dimension, delay)[1]
    if len(dgm)==0:
        return 0.0
    lifetimes = dgm[:,1]-dgm[:,0]
    p = lifetimes / (np.sum(lifetimes)+1e-10)
    return -np.sum(p * np.log(p + 1e-10))


# =============== RMT FUNCTIONS ===============

def compute_rmt_features(signal, dimension, delay):
    """Compute six eigenvalue-based RMT descriptors."""
    emb = takens_embedding(signal, dimension, delay)
    cov = np.cov(emb, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    eigvals_sorted = np.sort(eigvals)[::-1]
    p = eigvals_sorted / (np.sum(eigvals_sorted)+1e-10)
    ipr = np.mean(np.sum(eigvecs**4, axis=0))

    return np.array([
        eigvals_sorted[0],
        np.mean(eigvals_sorted),
        np.var(eigvals_sorted),
        eigvals_sorted[0] / (np.sum(eigvals_sorted)+1e-10),
        -np.sum(p * np.log(p+1e-10)),  # spectral entropy
        ipr
    ])


# =============== SPECTRAL FUNCTIONS ===============

def get_spectral_features(signal, sr=24000, n_mfcc=13):
    """Compute MFCCs, deltas, and spectral statistics."""
    signal = np.asarray(signal, dtype=float)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfccs)
    sc = librosa.feature.spectral_centroid(y=signal, sr=sr)
    sb = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    sr_off = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=signal)

    return np.concatenate([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
        [np.mean(sc), np.std(sc), np.mean(sb), np.std(sb),
         np.mean(sr_off), np.std(sr_off), np.mean(zcr), np.std(zcr)]
    ])
