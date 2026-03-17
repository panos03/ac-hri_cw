"""Utilities to merge windowed PPG and EDA feature files."""

import os

import pandas as pd


def merge_windowed_features(ppg_csv_path, eda_csv_path, output_path):
    """Merge windowed modalities on subject, label, and window index."""
    ppg_df = pd.read_csv(ppg_csv_path)
    eda_df = pd.read_csv(eda_csv_path)

    key_cols = ["sub_id", "label", "window_index"]
    merged = ppg_df.merge(eda_df, on=key_cols, how="outer", suffixes=("_ppg", "_eda"))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(output_path, index=False)

    quality_report_path = os.path.join(out_dir or ".", "quality_report.txt")

    ppg_keys = set(tuple(v) for v in ppg_df[key_cols].to_records(index=False))
    eda_keys = set(tuple(v) for v in eda_df[key_cols].to_records(index=False))
    both_keys = ppg_keys & eda_keys

    per_subject = merged.groupby("sub_id").size().sort_index()

    report_lines = [
        "Windowed Feature Merge Quality Report",
        "",
        f"PPG windows: {len(ppg_df)}",
        f"EDA windows: {len(eda_df)}",
        f"Merged windows (outer join): {len(merged)}",
        f"Windows with both modalities: {len(both_keys)}",
        f"PPG-only windows: {len(ppg_keys - eda_keys)}",
        f"EDA-only windows: {len(eda_keys - ppg_keys)}",
        "",
        "Windows per subject:",
    ]
    for sub_id, count in per_subject.items():
        report_lines.append(f"- {sub_id}: {int(count)}")

    with open(quality_report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(report_lines))

    print("Wrote merged features to", output_path)
    print("Wrote quality report to", quality_report_path)
    return output_path, quality_report_path
