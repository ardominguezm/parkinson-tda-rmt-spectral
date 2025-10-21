# Parkinson Voice Classification using TDA, RMT, and Spectral Features

This repository provides the implementation of a hybrid topological, spectral, and random matrix theory (RMT) framework 
for Parkinson’s disease detection from voice recordings.  
It integrates **synthetic training data** and **real-world validation**, combining **Topological Data Analysis (TDA)**, 
**Random Matrix Theory (RMT)**, and **Spectral descriptors**.

---

## Repository Structure

parkinson-tda-rmt-spectral/
│
├── src/
│ ├── main_experiment.py
│ ├── feature_extraction.py
│ ├── redundancy_analysis.py
│ ├── visualization_tools.py
│ └── utils.py
│
├── data/
│ ├── real_signals/
│ ├── synthetic_signals/
│ └── README.md
│
├── figures/
├── results/
├── notebooks/
│ └── exploratory_analysis.ipynb
│
├── requirements.txt
└── README.md

---

## Feature Design

| Feature Family | Components | Description | Count |
|----------------|-------------|--------------|--------|
| **TDA** | PI, PL, PE | Persistence image, landscape, and entropy | 1051 |
| **RMT** | Eigenvalue statistics | Spectral entropy, λmax, variance, IPR | 6 |
| **Spectral** | MFCCs, deltas, moments | 13 MFCCs + derivatives + 4 spectral moments | 60 |
| **Hybrid** | TDA + RMT + Spectral | Concatenated full descriptor set | **1117** |

---

## Requirements
```bash
pip install -r requirements.txt
