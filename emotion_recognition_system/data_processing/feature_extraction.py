"""
Entry point for all feature extraction.
Can be run directly or as a module:
    python emotion_recognition_system/data_processing/feature_extraction.py
    python -m data_processing.feature_extraction  (from emotion_recognition_system/)
"""

import os
import sys

# Ensure emotion_recognition_system/ is on the path when run directly
_workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _workspace not in sys.path:
    sys.path.insert(0, _workspace)

from utils.merge_features import merge_windowed_features
from utils.process_eda import get_eda_measures_windowed
from utils.process_ppg import get_ppg_measures_windowed

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(WORKSPACE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data_collection_and_label_synch", "data", "labelled_data")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "features")


def run_all(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, sample_rate=64.0, window_sec=30, stride_sec=15):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Windowed PPG
    ppg_window_csv = os.path.join(output_dir, "PPG_features_windowed.csv")
    get_ppg_measures_windowed(
        data_dir,
        window_sec=window_sec,
        stride_sec=stride_sec,
        sample_rate=sample_rate,
        output_csv=ppg_window_csv,
    )

    # 2. Windowed EDA
    eda_window_csv = os.path.join(output_dir, "EDA_features_windowed.csv")
    get_eda_measures_windowed(
        data_dir,
        window_sec=window_sec,
        stride_sec=stride_sec,
        sample_rate=sample_rate,
        output_csv=eda_window_csv,
    )

    # 3. Merge windowed modalities
    combined_csv = os.path.join(output_dir, "combined_features_windowed.csv")
    merge_windowed_features(ppg_window_csv, eda_window_csv, combined_csv)

    return {
        "ppg_window_csv": ppg_window_csv,
        "eda_window_csv": eda_window_csv,
        "combined_csv": combined_csv,
    }


if __name__ == "__main__":
    outputs = run_all()
    print("Extraction completed.")
    for k, v in outputs.items():
        print(k, "->", v)
