"""
utils.py
--------
Helper functions for data loading, scaling, and summary statistics.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_csv_signals(directory):
    """Load all CSV files from a directory (expects 'Normalized Amplitude' column)."""
    signals = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, file))
            signals.append(df["Normalized Amplitude"].values)
    return signals


def scale_features(X_train, X_test):
    """Standardize training and test feature sets."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def describe_features(X, name="Feature Set"):
    """Print shape and basic summary."""
    print(f"{name}: shape={X.shape}")
    print(f"Mean={np.mean(X):.4f}, Std={np.std(X):.4f}")

