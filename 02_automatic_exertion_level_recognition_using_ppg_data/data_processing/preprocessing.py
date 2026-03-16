"""Preprocessing helpers for windowed multimodal features."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut


def load_and_clean(csv_path, relabel=False):
    """
    Load merged windowed features and return feature matrix components.

    Returns:
        X: DataFrame of numeric features (subject excluded)
        y: Series of labels
        groups: Series of subject IDs
        df_clean: Cleaned DataFrame retained for diagnostics
    """
    df = pd.read_csv(csv_path)

    ppg_status_col = "status_ppg" if "status_ppg" in df.columns else "status"
    eda_status_col = "status_eda" if "status_eda" in df.columns else "status"

    ppg_failed = df[ppg_status_col].astype(str).str.contains("failed", case=False, na=True)
    eda_failed = df[eda_status_col].astype(str).str.contains("failed", case=False, na=True)

    # Keep rows where at least one modality succeeded.
    df = df[~(ppg_failed & eda_failed)].copy()

    if relabel:
        label_map = {
            "easy": "excitement",
            "hard": "frustration",
        }
        df["label"] = df["label"].map(lambda x: label_map.get(x, x))

    df["sub_id"] = df["sub_id"].astype(str)

    meta_cols = {
        "sub_id", "label", "window_index", "window_start_sec",
        "status", "status_ppg", "status_eda",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(str)
    groups = df["sub_id"].astype(str)

    return X, y, groups, df


def handle_missing(X_train, X_test):
    """Impute missing values using training-fold medians only."""
    medians = X_train.median(numeric_only=True)
    X_train_imputed = X_train.fillna(medians)
    X_test_imputed = X_test.fillna(medians)
    return X_train_imputed, X_test_imputed


def run_loso_cv(estimator, X, y, groups):
    """Run Leave-One-Subject-Out evaluation with train-fold-only imputation."""
    logo = LeaveOneGroupOut()

    y_true_all = []
    y_pred_all = []
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        X_train, X_test = handle_missing(X_train, X_test)

        model = clone(estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        fold_results.append({
            "fold": fold_idx,
            "test_subject": str(groups.iloc[test_idx].iloc[0]),
            "n_test_samples": int(len(test_idx)),
            "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        })

    labels = sorted(list(set(y_true_all) | set(y_pred_all)))
    return {
        "fold_results": fold_results,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=labels),
        "overall_report": classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0),
    }
