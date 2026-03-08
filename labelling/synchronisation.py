import glob
import os
import re
from datetime import datetime, timezone

import pandas as pd


def _parse_duration_seconds(duration_str):
    # Parse 'MM:SS' duration string to total seconds
    minutes, seconds = map(int, duration_str.split(":"))
    return minutes * 60 + seconds


def _hhmmss_to_seconds(hhmmss):
    # Convert HHMMSS integer (e.g. 144141 -> 14*3600 + 41*60 + 41) to seconds since midnight
    hh = hhmmss // 10000
    mm = (hhmmss % 10000) // 100
    ss = hhmmss % 100
    return hh * 3600 + mm * 60 + ss


def _extract_unix_timestamp_to_seconds(filepath):
    # Extract the Unix timestamp (seconds) embedded in a data filename
    basename = os.path.basename(filepath)
    match = re.search(r"_(\d{10})_", basename)
    if not match:
        raise ValueError(f"No Unix timestamp found in filename: {basename}")
    return int(match.group(1)) % 86400  # Modulo 86400 to get seconds since midnight


def estimate_sample_rates(video_timestamps_path, data_dir):
    # Estimate the sample rates of the data files by comparing the number of samples to the video durations
    videos = pd.read_csv(video_timestamps_path)
    rows = []

    for _, row in videos.iterrows():
        rec_id = row["id"]

        video_duration_s = _parse_duration_seconds(row["duration"])

        matches = glob.glob(os.path.join(data_dir, f"{rec_id}_*.csv"))
        if not matches:
            print(f"Warning: no data file found for {rec_id}")
            continue

        filepath = matches[0]

        num_samples = len(pd.read_csv(filepath))
        sample_rate_hz = num_samples / video_duration_s

        rows.append({
            "id": rec_id,
            "num_samples": num_samples,
            "video_duration_s": video_duration_s,
            "sample_rate_hz": sample_rate_hz,
        })

    # overall average sample rate across all videos
    if rows:
        avg_sample_rate = sum(row["sample_rate_hz"] for row in rows) / len(rows)
        print(f"Estimated average sample rate across all videos: {avg_sample_rate:.2f} Hz")
    else:
        print("No valid video-data pairs found to estimate sample rates.")
        avg_sample_rate = None

    return pd.DataFrame(rows), avg_sample_rate


def synchronisation_mapping(video_timestamps_path, data_dir):
    # Create a mapping of offset to video start time and data file start time, to enable synchronisation of data with video frames
    videos = pd.read_csv(video_timestamps_path)
    mapping = {}

    for _, row in videos.iterrows():
        rec_id = row["id"]

        video_start_time_s = _hhmmss_to_seconds(row["timestamp"])

        matches = glob.glob(os.path.join(data_dir, f"{rec_id}_*.csv"))
        if not matches:
            print(f"Warning: no data file found for {rec_id}")
            continue

        filepath = matches[0]
        data_start_time_s = _extract_unix_timestamp_to_seconds(filepath)

        offset_s = data_start_time_s - video_start_time_s

        mapping[rec_id] = offset_s

    return mapping


def create_labelled_data():
    # TODO
    pass


if __name__ == "__main__":
    _root = os.path.join(os.path.dirname(__file__), "..")
    video_timestamps_path = os.path.join(_root, "data", "video_timestamps.csv")
    data_dir = os.path.join(_root, "data")

    sample_rates_df, avg_sample_rate = estimate_sample_rates(video_timestamps_path, data_dir)
    print(sample_rates_df)

    mapping = synchronisation_mapping(video_timestamps_path, data_dir)
    print(mapping)