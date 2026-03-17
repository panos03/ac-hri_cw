"""
Map per-moment emotion annotations onto windowed feature rows.

Uses the ground-truth labels from self-report (80%) + observer (20%) annotations

"""

import os
import re
import glob

import numpy as np
import pandas as pd


def _normalize_text(text):
    """Clean whitespace artifacts from manually edited CSV files."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\xc2\xa0", " ").replace("\xa0", " ").replace("\ufeff", "")
    text = re.sub(r"[^\x20-\x7e:.\-/]", "", text)
    return text.strip()


def _parse_time_to_seconds(raw):
    """Parse video timestamp in various formats to seconds."""
    text = _normalize_text(raw)
    if "-" in text:
        text = text.split("-", 1)[0].strip()
    if ":" in text:
        parts = text.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    if re.fullmatch(r"\d+\.\d{1,2}", text):
        m, s = text.split(".", 1)
        return int(m) * 60 + int(s)
    return int(round(float(text) * 60))


def _normalize_label(label):
    """Normalise label variants to consistent names."""
    l = _normalize_text(label).lower().strip()
    if "excit" in l:
        return "excitement"
    if "frust" in l:
        return "frustration"
    if "neutral" in l:
        return "neutral"
    return l


def load_moment_labels(labels_folder, times_csv_path):
    """
    Load all per-moment emotion annotations and align to signal time.

    Returns DataFrame with columns: rec_id, signal_time_sec, label
    """
    times = pd.read_csv(times_csv_path)
    offsets = dict(zip(times["id"], times["offset_s"]))

    all_moments = []
    for f in sorted(glob.glob(os.path.join(labels_folder, "*labels*.csv"))):
        basename = os.path.basename(f)
        match = re.match(r"(.+?)(?:_labels|-labels)\.csv$", basename, re.IGNORECASE)
        if not match:
            continue
        rec_id = match.group(1)

        df = pd.read_csv(f, encoding="latin-1")
        cols = {
            re.sub(r"[^a-z0-9]+", "", _normalize_text(c).lower()): c
            for c in df.columns
        }
        time_col = cols.get("timeintovideo")
        label_col = cols.get("label")
        if not time_col or not label_col:
            continue

        offset = offsets.get(rec_id, 0)

        for _, row in df.iterrows():
            t_raw = _normalize_text(row[time_col])
            l_raw = _normalize_text(row[label_col])
            if not t_raw or not l_raw:
                continue
            all_moments.append({
                "rec_id": rec_id,
                "signal_time_sec": _parse_time_to_seconds(t_raw) - offset,
                "label": _normalize_label(l_raw),
            })

    return pd.DataFrame(all_moments)


def _resolve_window_label(labels_in_window):
    """
    Resolve the emotion label for a window containing multiple annotated moments.

    Returns the label string, or None if the window should be dropped.
    """
    if not labels_in_window:
        return None

    counts = {}
    for l in labels_in_window:
        counts[l] = counts.get(l, 0) + 1

    max_count = max(counts.values())
    winners = [l for l, c in counts.items() if c == max_count]

    if len(winners) == 1:
        return winners[0]

    # Tie between excitement and frustration — genuinely ambiguous
    if set(winners) == {"excitement", "frustration"}:
        return None

    # Tie involving neutral — prefer the high-arousal label
    if "excitement" in winners:
        return "excitement"
    if "frustration" in winners:
        return "frustration"

    return winners[0]


def relabel_features(
    features_csv_path,
    labels_folder,
    times_csv_path,
    output_csv_path,
    window_sec=30.0,
):
    """
    Relabel windowed features using ground-truth moment annotations.

    Reads the combined features CSV, maps emotion annotations onto each window,
    resolves conflicts, drops ambiguous/unlabelled windows, and writes the result.

    Returns:
        DataFrame with emotion labels, and a summary dict with statistics.
    """
    features = pd.read_csv(features_csv_path)
    moments = load_moment_labels(labels_folder, times_csv_path)

    # Build rec_id from sub_id + condition (easy/hard)
    features["rec_id"] = (
        features["sub_id"].astype(str).str.zfill(3) + "-" + features["label"]
    )

    emotion_labels = []
    for _, row in features.iterrows():
        rec_id = row["rec_id"]
        w_start = row["window_start_sec_eda"]
        w_end = w_start + window_sec

        rec_moments = moments[moments["rec_id"] == rec_id]
        overlapping = rec_moments[
            (rec_moments["signal_time_sec"] >= w_start)
            & (rec_moments["signal_time_sec"] < w_end)
        ]

        if len(overlapping) == 0:
            emotion_labels.append(None)
        else:
            resolved = _resolve_window_label(overlapping["label"].tolist())
            emotion_labels.append(resolved)

    features["emotion_label"] = emotion_labels

    # Drop windows with no label or ambiguous label
    n_before = len(features)
    features = features[features["emotion_label"].notna()].copy()
    n_after = len(features)

    # Replace the old condition-based 'label' column with the real emotion label
    features["label"] = features["emotion_label"]
    features = features.drop(columns=["rec_id", "emotion_label"])

    # Save
    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    features.to_csv(output_csv_path, index=False)

    summary = {
        "total_moments": len(moments),
        "windows_before": n_before,
        "windows_after": n_after,
        "windows_dropped": n_before - n_after,
        "label_distribution": features["label"].value_counts().to_dict(),
        "per_subject": features.groupby(["sub_id", "label"]).size().unstack(fill_value=0).to_dict(),
    }

    print(f"Relabelled: {n_after}/{n_before} windows kept ({n_before - n_after} dropped)")
    print(f"Label distribution: {summary['label_distribution']}")
    print(f"Saved to: {output_csv_path}")

    return features, summary
