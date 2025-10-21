"""
statistical_tests.py
--------------------
Performs statistical significance testing between classifiers (MLP vs SVM)
and across feature sets (TDA, RMT, Spectral, and hybrids).

Implements:
  - Paired t-tests per feature set (MLP vs SVM)
  - ANOVA between feature sets for each model
  - Tukey post-hoc comparisons when ANOVA is significant
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# =========================================================
# 1. Metric collection per fold
# =========================================================
def evaluate_model_cv_on_real_perfold(X_train_syn, y_train_syn, X_real, y_real, clf, n_splits=5):
    """Return per-fold metrics (ACC, F1, Sensitivity, Specificity)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for fold_id, (_, test_index) in enumerate(skf.split(X_real, y_real), start=1):
        X_test_fold, y_test_fold = X_real[test_index], y_real[test_index]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_syn)
        X_test_s = scaler.transform(X_test_fold)
        clf.fit(X_train_s, y_train_syn)
        y_pred = clf.predict(X_test_s)

        cm = confusion_matrix(y_test_fold, y_pred)
        TN, FP, FN, TP = cm.ravel()
        rows.append({
            "fold": fold_id,
            "accuracy": (TP + TN) / np.sum(cm),
            "f1": f1_score(y_test_fold, y_pred),
            "sensitivity": TP / (TP + FN) if (TP + FN) > 0 else 0,
            "specificity": TN / (TN + FP) if (TN + FP) > 0 else 0,
        })
    return pd.DataFrame(rows)


# =========================================================
# 2. Paired t-test between MLP and SVM (within each feature set)
# =========================================================
def paired_t_by_feature(perfold_df, metric="accuracy", model_a="MLP", model_b="SVM_linear"):
    results = []
    for feat in perfold_df["feature_set"].unique():
        a = perfold_df.query("feature_set==@feat and model==@model_a")[metric].values
        b = perfold_df.query("feature_set==@feat and model==@model_b")[metric].values
        if len(a) == len(b) and len(a) > 1:
            t, p = ttest_rel(a, b)
            results.append({"feature_set": feat, "metric": metric, "t": t, "p": p})
    return pd.DataFrame(results)


# =========================================================
# 3. ANOVA between feature sets (within each model)
# =========================================================
def anova_by_model(perfold_df, metric="accuracy"):
    output = []
    for model_name in perfold_df["model"].unique():
        groups = [
            perfold_df.query("model==@model_name and feature_set==@fs")[metric].values
            for fs in perfold_df["feature_set"].unique()
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            F, p = f_oneway(*groups)
            output.append({"model": model_name, "metric": metric, "F": F, "p": p})
    return pd.DataFrame(output)


# =========================================================
# 4. Tukey post-hoc analysis
# =========================================================
def tukey_posthoc(perfold_df, model_name="MLP", metric="accuracy"):
    sub = perfold_df[perfold_df["model"] == model_name][["feature_set", metric]].dropna()
    if sub["feature_set"].nunique() >= 3:
        tk = pairwise_tukeyhsd(
            endog=sub[metric].values,
            groups=sub["feature_set"].values,
            alpha=0.05,
        )
        print(f"\nTukey HSD for {model_name} ({metric}):")
        print(tk.summary())
