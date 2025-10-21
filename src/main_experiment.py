"""
main_experiment.py
------------------
Main experiment pipeline: training with synthetic data and validating on real signals.
Models: SVM and MLP.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from feature_extraction import get_PI, get_PL, get_PE, compute_rmt_features, get_spectral_features


# =============== FEATURE EXTRACTION ===============

def extract_features(signals, d=5, tau=20):
    tda_PI, tda_PL, tda_PE, rmt, spec = [], [], [], [], []
    for s in signals:
        try:
            tda_PI.append(get_PI(s, d, tau))
            tda_PL.append(get_PL(s, d, tau))
            tda_PE.append(get_PE(s, d, tau))
        except Exception:
            tda_PI.append(np.zeros(900))
            tda_PL.append(np.zeros(150))
            tda_PE.append([0.0])
        try:
            rmt.append(compute_rmt_features(s, d, tau))
        except Exception:
            rmt.append(np.zeros(6))
        try:
            spec.append(get_spectral_features(s))
        except Exception:
            spec.append(np.zeros(60))

    tda = np.concatenate([np.array(tda_PI), np.array(tda_PL), np.array(tda_PE)], axis=1)
    return tda, np.array(rmt), np.array(spec)


# =============== MODEL EVALUATION ===============

def evaluate_model(X_train, y_train, X_real, y_real, model, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    acc, sens, spec, f1 = [], [], [], []
    for train_idx, test_idx in skf.split(X_real, y_real):
        X_test, y_test = X_real[test_idx], y_real[test_idx]
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        acc.append(accuracy_score(y_test, y_pred))
        sens.append(TP/(TP+FN))
        spec.append(TN/(TN+FP))
        f1.append(f1_score(y_test, y_pred))
    return np.mean(acc), np.mean(sens), np.mean(spec), np.mean(f1)


# =============== MAIN ===============

def main():
    # Carga de señales (ajustar paths según tu entorno)
    real_dir = "./data/real_signals/"
    synth_dir = "./data/synthetic_signals/"

    # Placeholder: señales deben cargarse como arrays de amplitud normalizada
    real_signals = []  # cargar tus señales reales
    synth_signals = []  # cargar tus señales sintéticas

    # Features
    tda_train, rmt_train, spec_train = extract_features(synth_signals)
    tda_test, rmt_test, spec_test = extract_features(real_signals)

    y_train = np.array([0]*1000 + [1]*1000)
    y_test = np.array([0]*150 + [1]*150)

    combined_train = np.hstack([tda_train, rmt_train, spec_train])
    combined_test = np.hstack([tda_test, rmt_test, spec_test])

    models = {
        "SVM": SVC(kernel="linear", random_state=42),
        "MLP": MLPClassifier(max_iter=1000, random_state=42)
    }

    results = []
    for name, model in models.items():
        acc, se, sp, f1 = evaluate_model(combined_train, y_train, combined_test, y_test, model)
        results.append([name, acc, se, sp, f1])
        print(f"{name}: ACC={acc:.3f}, SE={se:.3f}, SP={sp:.3f}, F1={f1:.3f}")

    df = pd.DataFrame(results, columns=["Model","Accuracy","Sensitivity","Specificity","F1"])
    df.to_csv("./results/metrics_summary.csv", index=False)
    print("\nSaved metrics to results/metrics_summary.csv")


if __name__ == "__main__":
    main()
