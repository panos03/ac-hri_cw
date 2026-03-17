"""EDA feature extraction utilities for full recordings and windowed segments."""

import csv
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from utils.process_ppg import get_exertion_label_from_filename, get_overlapping_windows


def get_eda_signal_filepaths(datapath):
    """Return full_labelled CSV files with parsed subject IDs."""
    eda_files = []
    for path, dirs, files in os.walk(datapath):
        dirs.sort()
        for fn in sorted(files):
            if not fn.startswith("full_labelled_"):
                continue
            if os.path.splitext(fn)[-1].lower() != ".csv":
                continue

            name_body = os.path.splitext(fn)[0]
            sub_level = name_body.split("_")[-1]
            sub_id = sub_level.split("-")[0] if "-" in sub_level else os.path.basename(path)
            eda_files.append((sub_id, fn, os.path.join(path, fn)))
    return eda_files


def load_eda_signal(filepath):
    """Load EDA column from full_labelled CSV."""
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return np.array([], dtype=np.double)

    if "EDA" not in df.columns:
        return np.array([], dtype=np.double)

    eda = pd.to_numeric(df["EDA"], errors="coerce").dropna().to_numpy(dtype=np.double)
    return eda


def _safe_filter(data, btype, cutoff_hz, sample_rate=64.0, order=4):
    nyq = 0.5 * sample_rate
    cutoff = float(cutoff_hz) / nyq
    cutoff = min(max(cutoff, 1e-6), 0.999999)

    b, a = butter(order, cutoff, btype=btype)
    if len(data) <= (order * 3):
        return np.asarray(data, dtype=np.double)
    return filtfilt(b, a, np.asarray(data, dtype=np.double))


def get_filtered_eda(raw_signal, sample_rate=64.0):
    """
    Preprocess raw EDA and return tonic/phasic decomposition.

    Steps:
    - Discard first 5 seconds
    - Low-pass filter at 5 Hz (4th-order Butterworth)
    - High-pass at 0.05 Hz to estimate phasic component
    - Tonic is low-pass-cleaned minus phasic
    """
    raw_signal = np.asarray(raw_signal, dtype=np.double)
    raw_signal = raw_signal[np.isfinite(raw_signal)]
    if raw_signal.size == 0:
        return np.array([], dtype=np.double), np.array([], dtype=np.double), np.array([], dtype=np.double)

    discard = int(round(5.0 * sample_rate))
    if raw_signal.size > discard:
        raw_signal = raw_signal[discard:]

    if raw_signal.size < 8:
        return np.array([], dtype=np.double), np.array([], dtype=np.double), np.array([], dtype=np.double)

    cleaned = _safe_filter(raw_signal, btype="low", cutoff_hz=5.0, sample_rate=sample_rate, order=4)
    phasic = _safe_filter(cleaned, btype="high", cutoff_hz=0.05, sample_rate=sample_rate, order=4)
    tonic = cleaned - phasic
    return tonic, phasic, cleaned


def detect_scr_peaks(phasic, sample_rate=64.0):
    """Detect SCR peaks from the phasic EDA component."""
    phasic = np.asarray(phasic, dtype=np.double)
    phasic = phasic[np.isfinite(phasic)]
    if phasic.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.double)

    min_distance = max(1, int(round(sample_rate * 1.0)))
    peaks, props = find_peaks(phasic, height=0.01, distance=min_distance)
    amplitudes = props.get("peak_heights", np.array([], dtype=np.double))
    return peaks, amplitudes


def extract_eda_features_window(raw_eda, sample_rate=64.0):
    """Extract EDA features from one window of raw EDA samples."""
    tonic, phasic, cleaned = get_filtered_eda(raw_eda, sample_rate=sample_rate)
    if cleaned.size == 0:
        return {
            "eda_mean_scl": np.nan,
            "eda_slope_scl": np.nan,
            "eda_num_scr": np.nan,
            "eda_mean_scr_amp": np.nan,
            "eda_sum_scr_amp": np.nan,
            "eda_variance": np.nan,
            "eda_range": np.nan,
            "eda_deriv_mean": np.nan,
        }

    peaks, amplitudes = detect_scr_peaks(phasic, sample_rate=sample_rate)

    t = np.arange(tonic.size, dtype=np.double) / sample_rate
    tonic_slope = float(np.polyfit(t, tonic, 1)[0]) if tonic.size > 1 else 0.0

    deriv = np.diff(cleaned)
    deriv_mean = float(np.mean(deriv)) if deriv.size else 0.0

    mean_scr_amp = float(np.mean(amplitudes)) if amplitudes.size else np.nan
    sum_scr_amp = float(np.sum(amplitudes)) if amplitudes.size else 0.0

    return {
        "eda_mean_scl": float(np.mean(tonic)),
        "eda_slope_scl": tonic_slope,
        "eda_num_scr": int(peaks.size),
        "eda_mean_scr_amp": mean_scr_amp,
        "eda_sum_scr_amp": sum_scr_amp,
        "eda_variance": float(np.var(cleaned)),
        "eda_range": float(np.max(cleaned) - np.min(cleaned)),
        "eda_deriv_mean": deriv_mean,
    }


def get_eda_measures_windowed(
    datapath,
    window_sec=30,
    stride_sec=15,
    sample_rate=64.0,
    output_csv=None,
):
    """Extract EDA features from overlapping windows for each labelled recording."""
    window_len = int(round(window_sec * sample_rate))
    hop_len = int(round(stride_sec * sample_rate))
    if window_len <= 0 or hop_len <= 0:
        raise ValueError("window_sec and stride_sec must produce positive lengths")

    csv_fpath = output_csv or os.path.join(datapath, "EDA_features_windowed.csv")
    header = [
        "sub_id", "label", "window_index", "window_start_sec", "status",
        "eda_mean_scl", "eda_slope_scl", "eda_num_scr", "eda_mean_scr_amp",
        "eda_sum_scr_amp", "eda_variance", "eda_range", "eda_deriv_mean",
    ]

    with open(csv_fpath, "w", newline="") as csvfile:
        fp_writer = csv.writer(csvfile, delimiter=",")
        fp_writer.writerow(header)

    rows_written = 0
    eda_files = get_eda_signal_filepaths(datapath)
    for sub_id, fn, filepath in eda_files:
        label = get_exertion_label_from_filename(fn)
        raw_eda = load_eda_signal(filepath)
        if raw_eda.size == 0:
            continue

        tonic, phasic, cleaned = get_filtered_eda(raw_eda, sample_rate=sample_rate)
        del tonic, phasic
        if cleaned.size < window_len:
            continue

        windows = get_overlapping_windows(len(cleaned), window_len, hop_len)
        for win_idx, (start, end) in enumerate(windows):
            seg_raw = cleaned[start:end]
            feats = extract_eda_features_window(seg_raw, sample_rate=sample_rate)
            status = "ok" if np.isfinite(feats["eda_variance"]) else "failed_extract"
            vals = [sub_id, label, win_idx, start / sample_rate, status] + [feats[k] for k in header[5:]]

            with open(csv_fpath, "a+", newline="") as csvfile:
                fp_writer = csv.writer(csvfile, delimiter=",")
                fp_writer.writerow(vals)
            rows_written += 1

    print("Wrote", rows_written, "EDA window rows to", csv_fpath)
    return csv_fpath
