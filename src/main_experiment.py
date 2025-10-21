"""
main_experiment.py
------------------
Main experiment pipeline: training with synthetic data and validating on real signals.
Implements MLP and SVM models, performance evaluation, and statistical significance tests.
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

# Import statistical analysis functions
from statistical_tests import (
    evaluate_model_cv_on_real_perfold,
    paired_t_by_feature,
    anova_by_model,
    tukey_posthoc
)


# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_features(signals, d=5, tau=20):
    """Compute TDA, RMT, and Spectral features for each signal."""
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


# =========================================================
# MODEL EVALUATION (AVERAGE)
# =========================================================
def evaluate_model(X_train, y_train, X_real, y_real, model, folds=5):
    """Perform cross-validation on real data after training with synthetic."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    acc, sens, spec, f1 = [], [], [], []
    for _, test_idx in skf.split(X_real, y_real):
        X_test, y_test = X_real[test_idx], y_real[test_idx]
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        acc.append(accuracy_score(y_test, y_pred))
        sens.append(TP / (TP + FN))
        spec.append(TN / (TN + FP))
        f1.append(f1_score(y_test, y_pred))
    return np.mean(acc), np.mean(sens), np.mean(spec), np.mean(f1)


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    # 1 Load or define paths
    real_dir = "./data/real_signals/"
    synth_dir = "./data/synthetic_signals/"
    os.makedirs("./results", exist_ok=True)

    # 2 Load your preprocessed voice signals
    real_signals = []   # TODO: load your real signals here
    synth_signals = []  # TODO: load your synthetic signals here

    # 3 Feature extraction
    print("Extracting features...")
    tda_train, rmt_train, spec_train = extract_features(synth_signals)
    tda_test, rmt_test, spec_test = extract_features(real_signals)

    y_train_syn = np.array([0]*1000 + [1]*1000)
    y_real = np.array([0]*150 + [1]*150)

    groups_train = {
        "TDA": tda_train,
        "RMT": rmt_train,
        "Spectral": spec_train,
        "Hybrid": np.hstack([tda_train, rmt_train, spec_train])
    }
    groups_test = {
        "TDA": tda_test,
        "RMT": rmt_test,
        "Spectral": spec_test,
        "Hybrid": np.hstack([tda_test, rmt_test, spec_test])
    }

    models = {
        "SVM_linear": SVC(kernel="linear", random_state=42),
        "MLP": MLPClassifier(max_iter=1000, random_state=42)
    }

    # 4 Evaluate models (mean metrics)
    print("\nEvaluating models...")
    results = []
    for name, model in models.items():
        acc, se, sp, f1 = evaluate_model(groups_train["Hybrid"], y_train_syn, groups_test["Hybrid"], y_real, model)
        results.append([name, acc, se, sp, f1])
        print(f"{name}: ACC={acc:.3f}, SE={se:.3f}, SP={sp:.3f}, F1={f1:.3f}")

    df = pd.DataFrame(results, columns=["Model", "Accuracy", "Sensitivity", "Specificity", "F1"])
    df.to_csv("./results/metrics_summary.csv", index=False)
    print("\nSaved mean metrics to results/metrics_summary.csv")

    # =========================================================
    # 5 Statistical significance analysis
    # =========================================================
    print("\nRunning statistical tests (t-test, ANOVA, Tukey)...")

    perfold_results = []
    for feat_name in groups_train.keys():
        Xtr, Xte = groups_train[feat_name], groups_test[feat_name]
        for model_name, model in models.items():
            df_fold = evaluate_model_cv_on_real_perfold(Xtr, y_train_syn, Xte, y_real, model, n_splits=5)
            df_fold["model"] = model_name
            df_fold["feature_set"] = feat_name
            perfold_results.append(df_fold)
    perfold_df = pd.concat(perfold_results, ignore_index=True)

    # Paired t-test (MLP vs SVM)
    tt_acc = paired_t_by_feature(perfold_df, metric="accuracy")
    tt_f1  = paired_t_by_feature(perfold_df, metric="f1")
    print("\nPaired t-test (MLP vs SVM_linear) - Accuracy\n", tt_acc)
    print("\nPaired t-test (MLP vs SVM_linear) - F1\n", tt_f1)

    # ANOVA and Tukey post-hoc
    anova_acc = anova_by_model(perfold_df, metric="accuracy")
    anova_f1  = anova_by_model(perfold_df, metric="f1")
    print("\nANOVA - Accuracy\n", anova_acc)
    print("\nANOVA - F1\n", anova_f1)

    tukey_posthoc(perfold_df, model_name="MLP", metric="accuracy")
    tukey_posthoc(perfold_df, model_name="MLP", metric="f1")

    # Save results
    tt_acc.to_csv("./results/ttest_accuracy.csv", index=False)
    tt_f1.to_csv("./results/ttest_f1.csv", index=False)
    anova_acc.to_csv("./results/anova_accuracy.csv", index=False)
    anova_f1.to_csv("./results/anova_f1.csv", index=False)
    print("\nAll statistical results saved in ./results/")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()

