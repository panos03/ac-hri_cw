"""Orchestrate model training and LOSO-CV evaluation across modality configurations."""

import os
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels

from data_processing.preprocessing import (
    load_and_clean, run_loso_cv, build_fusion_preprocessor,
)
from models.config import (
    RF_PARAMS, SVM_PARAMS, SVM_GRID,
    PPG_FEATURES, EDA_FEATURES, ALL_FEATURES,
    RANDOM_SEED, PCA_VARIANCE_THRESHOLD,
)


def _available_cols(feature_list, X):
    """Return only feature names that exist in the DataFrame."""
    return [c for c in feature_list if c in X.columns]


class LateFusionClassifier(BaseEstimator, ClassifierMixin):
    """
    Late fusion: trains separate EDA and PPG models, combines via soft vote.

    Expects X to contain both eda_cols and ppg_cols. Splits internally so
    that run_loso_cv (and its handle_missing step) can be reused unchanged.

    Prediction: average predict_proba from both models, then argmax.
    Both base estimators must support predict_proba (for SVM use probability=True).
    """

    def __init__(self, eda_estimator, ppg_estimator, eda_cols, ppg_cols):
        self.eda_estimator = eda_estimator
        self.ppg_estimator = ppg_estimator
        self.eda_cols = eda_cols
        self.ppg_cols = ppg_cols

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.eda_model_ = clone(self.eda_estimator)
        self.ppg_model_ = clone(self.ppg_estimator)
        self.eda_model_.fit(X[self.eda_cols], y)
        self.ppg_model_.fit(X[self.ppg_cols], y)
        return self

    def predict_proba(self, X):
        proba_eda = self.eda_model_.predict_proba(X[self.eda_cols])
        proba_ppg = self.ppg_model_.predict_proba(X[self.ppg_cols])
        return (proba_eda + proba_ppg) / 2

    def predict(self, X):
        avg_proba = self.predict_proba(X)
        return self.classes_[np.argmax(avg_proba, axis=1)]


def build_models():
    """Return dict of named model instances."""
    return {
        "RF": RandomForestClassifier(**RF_PARAMS),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(**SVM_PARAMS),
        ),
    }


def run_all_experiments(csv_path, output_dir="results"):
    """
    Run experiments for all model, modality and fusion configurations, save predictions to CSV.

    Returns a dict of experiment results keyed by descriptive name.
    """
    os.makedirs(output_dir, exist_ok=True)
    X, y, groups, df_clean = load_and_clean(csv_path)

    eda_cols = _available_cols(EDA_FEATURES, X)
    ppg_cols = _available_cols(PPG_FEATURES, X)
    combined_cols = _available_cols(ALL_FEATURES, X)

    
    ppg_present = ~X[ppg_cols].isnull().all(axis=1)

    models = build_models()
    experiments = {}

   # --- Early (/ No) Fusion experiments ---
    configs = [
        ("No fusion - EDA-only", eda_cols, None),            # all 121 windows, zero NaN
        ("No fusion - PPG-only", ppg_cols, ppg_present),     # only windows with PPG data
        ("Early fusion", combined_cols, None),               # all 121 windows, PPG imputed
    ]

    for model_name, model in models.items():
        for config_name, cols, mask in configs:
            exp_name = f"{model_name}_{config_name}"
            print(f"Running: {exp_name} ...", end=" ")

            if mask is not None:
                X_sub = X.loc[mask, cols].copy()
                y_sub = y[mask].reset_index(drop=True)
                g_sub = groups[mask].reset_index(drop=True)
                X_sub = X_sub.reset_index(drop=True)
            else:
                X_sub = X[cols].copy()
                y_sub = y.copy()
                g_sub = groups.copy()

            result = run_loso_cv(model, X_sub, y_sub, g_sub)
            result["model_name"] = model_name
            result["modality"] = config_name
            result["feature_cols"] = cols
            result["n_samples"] = len(X_sub)
            experiments[exp_name] = result

            f1 = result["overall_report"]["macro avg"]["f1-score"]
            acc = result["overall_report"]["accuracy"]
            print(f"F1={f1:.3f}  Acc={acc:.3f}  (n={len(X_sub)})")

    # Shared subset for *_Complete variants: windows where PPG was actually recorded.
    X_ppg_avail = X.loc[ppg_present, combined_cols].copy().reset_index(drop=True)
    y_ppg_avail = y[ppg_present].reset_index(drop=True)
    g_ppg_avail = groups[ppg_present].reset_index(drop=True)

    # --- Intermediate fusion ---
    # Two variants: all rows (missing PPG imputed) and PPG-available rows only.
    # clone() is called on the preprocessor inside each fold, so one instance suffices.
    preprocessor = build_fusion_preprocessor(eda_cols, ppg_cols, PCA_VARIANCE_THRESHOLD)

    intermediate_configs = [
        ("IntermediateFusion",          X[combined_cols], y,          groups,      len(X)),
        ("IntermediateFusion_Complete",  X_ppg_avail,      y_ppg_avail, g_ppg_avail, len(X_ppg_avail)),
    ]
    print("\nIntermediate Fusion experiments:")
    for modality, X_if, y_if, g_if, n_if in intermediate_configs:
        for model_name, model in models.items():
            exp_name = f"{model_name}_{modality}"
            print(f"Running: {exp_name} ...", end=" ")

            fusion_pipeline = SkPipeline([
                ('preprocessor', clone(preprocessor)),
                ('classifier', clone(model)),
            ])

            result = run_loso_cv(fusion_pipeline, X_if, y_if, g_if)
            result["model_name"] = model_name
            result["modality"] = modality
            result["feature_cols"] = combined_cols
            result["n_samples"] = n_if
            experiments[exp_name] = result

            f1 = result["overall_report"]["macro avg"]["f1-score"]
            acc = result["overall_report"]["accuracy"]
            print(f"F1={f1:.3f}  Acc={acc:.3f}  (n={n_if})")

    # --- Late fusion ---
    # Two variants to match intermediate fusion above.
    # SVM requires probability=True for predict_proba used in soft-vote fusion.
    # Separate estimator instances are needed as each branch is fitted on different features.
    late_fusion_models = {
        "RF": (RandomForestClassifier(**RF_PARAMS), RandomForestClassifier(**RF_PARAMS)),
        "SVM": (
            make_pipeline(StandardScaler(), SVC(**SVM_PARAMS, probability=True)),
            make_pipeline(StandardScaler(), SVC(**SVM_PARAMS, probability=True)),
        ),
    }
    late_configs = [
        ("LateFusion",          X[combined_cols], y,          groups,      len(X)),
        ("LateFusion_Complete",  X_ppg_avail,      y_ppg_avail, g_ppg_avail, len(X_ppg_avail)),
    ]
    print("\nLate Fusion experiments:")
    for modality, X_lf, y_lf, g_lf, n_lf in late_configs:
        for model_name, (eda_model, ppg_model) in late_fusion_models.items():
            exp_name = f"{model_name}_{modality}"
            print(f"Running: {exp_name} ...", end=" ")

            clf = LateFusionClassifier(
                eda_estimator=eda_model,
                ppg_estimator=ppg_model,
                eda_cols=eda_cols,
                ppg_cols=ppg_cols,
            )

            result = run_loso_cv(clf, X_lf, y_lf, g_lf)
            result["model_name"] = model_name
            result["modality"] = modality
            result["feature_cols"] = combined_cols
            result["n_samples"] = n_lf
            experiments[exp_name] = result

            f1 = result["overall_report"]["macro avg"]["f1-score"]
            acc = result["overall_report"]["accuracy"]
            print(f"F1={f1:.3f}  Acc={acc:.3f}  (n={n_lf})")

    print("\nSVM hyperparameter sweep on EDA-only:")
    best_f1 = -1
    best_params = {}
    for C in SVM_GRID["C"]:
        for gamma in SVM_GRID["gamma"]:
            svm = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=C, gamma=gamma,
                    class_weight="balanced", random_state=RANDOM_SEED),
            )
            r = run_loso_cv(svm, X[eda_cols], y, groups)
            f1 = r["overall_report"]["macro avg"]["f1-score"]
            print(f"  C={C:<5} gamma={gamma:<6} => F1={f1:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                best_params = {"C": C, "gamma": gamma}

    print(f"  Best: C={best_params['C']}, gamma={best_params['gamma']}, F1={best_f1:.3f}")
    experiments["_svm_best_params"] = best_params

    
    for exp_name, result in experiments.items():
        if exp_name.startswith("_"):
            continue
        pred_df = pd.DataFrame({
            "y_true": result["y_true"],
            "y_pred": result["y_pred"],
        })
        pred_df.to_csv(
            os.path.join(output_dir, f"{exp_name}_predictions.csv"),
            index=False,
        )

    return experiments, X, y, groups, df_clean


def summary_table(experiments):
    """Build a comparison DataFrame from experiment results."""
    rows = []
    for name, result in experiments.items():
        if name.startswith("_"):
            continue
        report = result["overall_report"]
        rows.append({
            "Experiment": name,
            "Model": result["model_name"],
            "Modality": result["modality"],
            "N_samples": result["n_samples"],
            "Accuracy": report["accuracy"],
            "Macro_F1": report["macro avg"]["f1-score"],
            "Macro_Precision": report["macro avg"]["precision"],
            "Macro_Recall": report["macro avg"]["recall"],
        })
    return pd.DataFrame(rows).sort_values("Macro_F1", ascending=False).reset_index(drop=True)
