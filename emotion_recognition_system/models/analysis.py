"""Generate evaluation plots, confusion matrices, and analysis tables."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    classification_report, cohen_kappa_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from data_processing.preprocessing import handle_missing
from models.config import RANDOM_SEED



matplotlib.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def plot_confusion_matrices(experiments, keys=None, output_dir="results"):
    """Plot raw + normalised confusion matrices for selected experiments."""
    os.makedirs(output_dir, exist_ok=True)
    if keys is None:
        keys = [k for k in experiments if not k.startswith("_")]

    for exp_name in keys:
        result = experiments[exp_name]
        labels = result["labels"]
        cm = result["confusion_matrix"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Confusion Matrix — {exp_name}", fontsize=13)

        # Raw counts
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=axes[0], cmap="Blues", colorbar=False,
        )
        axes[0].set_title("Raw counts")

        # Normalised by true label
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(
            ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f",
        )
        axes[1].set_title("Normalised (by true label)")

        plt.tight_layout()
        path = os.path.join(output_dir, f"confusion_matrix_{exp_name}.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")

    return fig


def per_subject_table(experiments, keys=None):
    """Build a per-subject accuracy table across experiments."""
    if keys is None:
        keys = [k for k in experiments if not k.startswith("_")]

    rows = []
    for exp_name in keys:
        for fold in experiments[exp_name]["fold_results"]:
            rows.append({
                "Experiment": exp_name,
                "Subject": fold["test_subject"],
                "N_test": fold["n_test_samples"],
                "Accuracy": fold["report"]["accuracy"],
                "F1_macro": fold["report"].get("macro avg", {}).get("f1-score", np.nan),
            })
    return pd.DataFrame(rows)


def plot_per_subject_accuracy(experiments, keys=None, output_dir="results"):
    """Grouped bar chart of per-subject accuracy across experiments."""
    os.makedirs(output_dir, exist_ok=True)
    df = per_subject_table(experiments, keys)
    if df.empty:
        return

    pivot = df.pivot_table(index="Subject", columns="Experiment", values="Accuracy")
    ax = pivot.plot(kind="bar", figsize=(12, 5), rot=0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Subject Accuracy by Experiment")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, label="chance")

    path = os.path.join(output_dir, "per_subject_accuracy.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    return ax


def compute_feature_importance(X, y, groups, feature_cols, output_dir="results"):
    """
    Train RF on full data (with imputation) and extract both
    Gini importance and permutation importance.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_imp = X[feature_cols].copy()
    medians = X_imp.median(numeric_only=True)
    X_imp = X_imp.fillna(medians)

    rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED,
        class_weight="balanced", min_samples_leaf=3,
    )
    rf.fit(X_imp, y)

    gini = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()

    perm = permutation_importance(
        rf, X_imp, y, n_repeats=20, random_state=RANDOM_SEED, scoring="f1_macro",
    )
    perm_imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Importance (Random Forest)", fontsize=13)

    gini.plot(kind="barh", ax=axes[0], color="#4878CF")
    axes[0].set_title("Gini Importance")
    axes[0].set_xlabel("Importance")

    perm_imp.plot(kind="barh", ax=axes[1], color="#6ACC65")
    axes[1].set_title("Permutation Importance (F1 macro)")
    axes[1].set_xlabel("Mean decrease in F1")

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")

    return gini, perm_imp


def plot_modality_comparison(experiments, output_dir="results"):
    """Bar chart comparing F1 across modality configurations."""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for name, result in experiments.items():
        if name.startswith("_"):
            continue
        rows.append({
            "Experiment": name,
            "Model": result["model_name"],
            "Modality": result["modality"],
            "F1_macro": result["overall_report"]["macro avg"]["f1-score"],
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"EDA-only": "#4878CF", "Combined": "#D65F5F", "PPG-only": "#B47CC7", "IntermediateFusion": "#6ACC65", "IntermediateFusion_PPGonly": "#C5B0D5"}

    x = np.arange(len(df))
    bars = ax.bar(x, df["F1_macro"], color=[colors.get(m, "grey") for m in df["Modality"]])
    ax.set_xticks(x)
    ax.set_xticklabels(df["Experiment"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Model × Modality Comparison")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8)

    for bar, val in zip(bars, df["F1_macro"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "modality_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    return ax


def full_metrics_table(experiments, keys=None):
    """Detailed per-class + aggregate metrics table."""
    if keys is None:
        keys = [k for k in experiments if not k.startswith("_")]

    rows = []
    for exp_name in keys:
        result = experiments[exp_name]
        report = result["overall_report"]
        kappa = cohen_kappa_score(result["y_true"], result["y_pred"])

        fold_accs = [f["report"]["accuracy"] for f in result["fold_results"]]

        row = {
            "Experiment": exp_name,
            "N": result["n_samples"],
            "Accuracy": report["accuracy"],
            "Macro_F1": report["macro avg"]["f1-score"],
            "Macro_Prec": report["macro avg"]["precision"],
            "Macro_Rec": report["macro avg"]["recall"],
            "Kappa": kappa,
            "Fold_Acc_Mean": np.mean(fold_accs),
            "Fold_Acc_Std": np.std(fold_accs),
            "Fold_Acc_Min": np.min(fold_accs),
            "Fold_Acc_Max": np.max(fold_accs),
        }

        # Per-class metrics
        for label in result["labels"]:
            if label in report:
                row[f"{label}_precision"] = report[label]["precision"]
                row[f"{label}_recall"] = report[label]["recall"]
                row[f"{label}_f1"] = report[label]["f1-score"]

        rows.append(row)

    return pd.DataFrame(rows)


def save_all_results(experiments, X, y, groups, output_dir="results"):
    """Run all analysis functions and save everything."""
    os.makedirs(output_dir, exist_ok=True)

    # Key experiments to focus on (primary comparisons)
    primary_keys = [k for k in experiments if not k.startswith("_")]

    # 1. Confusion matrices
    plot_confusion_matrices(experiments, primary_keys, output_dir)

    # 2. Per-subject accuracy
    plot_per_subject_accuracy(experiments, primary_keys, output_dir)
    sub_df = per_subject_table(experiments, primary_keys)
    sub_df.to_csv(os.path.join(output_dir, "per_subject_breakdown.csv"), index=False)

    # 3. Feature importance (using EDA+PPG combined features)
    from models.config import ALL_FEATURES
    available = [c for c in ALL_FEATURES if c in X.columns]
    compute_feature_importance(X, y, groups, available, output_dir)

    # 4. Modality comparison bar chart
    plot_modality_comparison(experiments, output_dir)

    # 5. Full metrics table
    metrics_df = full_metrics_table(experiments, primary_keys)
    metrics_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    print(f"\nSaved metrics summary to {os.path.join(output_dir, 'metrics_summary.csv')}")

    plt.close("all")
    return metrics_df
