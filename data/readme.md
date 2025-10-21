
---

## Data Organization

The data is provided as compressed `.zip` files located in the `data/` directory:

| Dataset Type | File | # Samples (Control) | # Samples (PD) | Description |
|---------------|-------|--------------------|----------------|--------------|
| **Synthetic** | `synthetic_signals.zip` | 1000 | 1000 | Generated using Big Vocoder Slicing Adversarial Network (BigVSAN) from PC-GITA recordings (https://doi.org/10.1109/OJCS.2024.3504864) |
| **Real** | `real_signals.zip` | 150 | 150 | Extracted from PC-GITA database (Colombian-Spanish speakers) |

Each `.zip` file contains audio waveforms (`.wav`) or preprocessed `.npy` arrays of normalized amplitude signals.
