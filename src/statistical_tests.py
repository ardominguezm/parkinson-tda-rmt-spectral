"""
statistical_tests.py
--------------------
Performs statistical significance and practical significance testing between classifiers (MLP vs SVM)
and across feature sets (TDA, RMT, Spectral, and hybrids).

Implements:
  - Paired t-tests per feature set (MLP vs SVM)
  - ANOVA between feature sets for each model
  - Tukey post-hoc comparisons when ANOVA is significant
  - Effect size measures (Cohen's d, eta-squared)
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
    """Compute paired t-test per feature set for the selected metric."""
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
    """Compute one-way ANOVA across feature sets for each model."""
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
    """Run Tukey HSD post-hoc test if ANOVA is significant."""
    sub = perfold_df[perfold_df["model"] == model_name][["feature_set", metric]].dropna()
    if sub["feature_set"].nunique() >= 3:
        tk = pairwise_tukeyhsd(
            endog=sub[metric].values,
            groups=sub["feature_set"].values,
            alpha=0.05,
        )
        print(f"\nTukey HSD for {model_name} ({metric}):")
        print(tk.summary())


# =========================================================
# 5. Effect size calculations
# =========================================================
def cohen_d(a, b):
    """Compute Cohen's d for paired samples."""
    diff = a - b
    return np.mean(diff) / np.std(diff, ddof=1)


def eta_squared_anova(F, df_between, df_within):
    """Compute eta-squared (η²) from ANOVA F-statistic."""
    return (F * df_between) / (F * df_between + df_within)


def compute_effect_sizes(perfold_df, ttest_df, anova_df, model_a="MLP", model_b="SVM_linear"):
    """Compute Cohen's d for t-tests and η² for ANOVA results."""
    # Cohen's d per feature set
    cohen_results = []
    for feat in perfold_df["feature_set"].unique():
        a = perfold_df.query("feature_set==@feat and model==@model_a")["accuracy"].values
        b = perfold_df.query("feature_set==@feat and model==@model_b")["accuracy"].values
        if len(a) == len(b) and len(a) > 1:
            d = cohen_d(a, b)
            cohen_results.append({"feature_set": feat, "Cohen_d": round(d, 3)})

    # η² per ANOVA
    eta_results = []
    for _, row in anova_df.iterrows():
        df_between = len(perfold_df["feature_set"].unique()) - 1
        df_within = len(perfold_df) - df_between - 1
        eta2 = eta_squared_anova(row["F"], df_between, df_within)
        eta_results.append({"model": row["model"], "metric": row["metric"], "eta_squared": round(eta2, 3)})

    return pd.DataFrame(cohen_results), pd.DataFrame(eta_results)

