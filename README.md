# Parkinson Voice Classification using TDA, RMT, and Spectral Features

This repository provides the implementation of a hybrid topological, spectral, and random matrix theory (RMT) framework 
for Parkinson’s disease detection from voice recordings.  It integrates **synthetic training data** and **real-world validation**, combining **Topological Data Analysis**, **Random Matrix Theory**, and **Spectral descriptors**.


Andy Domínguez-Monterroza, Alfonso Mateos-Caballero, Antonio Jiménez-Martín, Machine Learning for Parkinson’s Disease Detection: Analyzing Hybrid Voice Data with Spectral, Topological, and Random Matrix Methods, In review (2025), *IEEE Open Journal of Computer Society* 


---

## Overview  

This repository provides the full implementation of a hybrid **Topological Data Analysis(TDA)**, **Random Matrix Theory (RMT)**,  
and **Spectral feature** framework for *Parkinson’s disease voice classification*.   The model is trained on **synthetic voice data** and evaluated on **real recordings**,  
analyzing topological, spectral, and geometric structure in the voice dynamics of patients and controls.  

---

## Methodological Summary  

| Feature Family | Components | Description | Count |
|----------------|-------------|--------------|--------|
| **TDA** | Persistence Image (PI), Persistence Landscape (PL), Persistent Entropy (PE) | Captures homological patterns in reconstructed phase space | 1051 |
| **RMT** | Eigenvalue Statistics | Extracts spectral entropy, λₘₐₓ, and inverse participation ratio from covariance spectra | 6 |
| **Spectral** | MFCCs, Deltas, Spectral Moments | Conventional acoustic descriptors (MFCCs + roll-off, centroid, bandwidth, ZCR) | 60 |
| **Hybrid** | TDA + RMT + Spectral | Concatenated descriptor space | **1117** |

---

## Repository Structure  

```plaintext
parkinson-tda-rmt-spectral/
│
├── src/
│   ├── main_experiment.py          # Main classification pipeline (SVM & MLP)
│   ├── feature_extraction.py       # TDA, RMT, and spectral feature computation
│   ├── redundancy_analysis.py      # PCA, correlation, and weighted fusion analysis
│   ├── statistical_tests.py         # Performs statistical significance testing between classifiers (MLP vs SVM) and across feature sets (TDA, RMT, Spectral, and hybrids)
│   ├── visualization_tools.py      # t-SNE, PCA 3D, and correlation plots
│   └── utils.py                    # Data loading and preprocessing helpers
│
├── data/
│   ├── real_signals/               # Real recordings (control / pathological)
│   ├── synthetic_signals/          # Synthetic generated recordings
│   └── README.md                   # Notes on dataset organization
│

│
├── requirements.txt
├── LICENSE
└── README.md
