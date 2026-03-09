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
    # Estimate the sample rates using the actual data duration (video duration minus offset),
    # where offset accounts for data collection starting before/after the video
    videos = pd.read_csv(video_timestamps_path)
    rows = []

    for _, row in videos.iterrows():
        rec_id = row["id"]

        video_duration_s = _parse_duration_seconds(row["duration"])
        video_start_s = _hhmmss_to_seconds(row["timestamp"])

        matches = glob.glob(os.path.join(data_dir, f"{rec_id}_*.csv"))
        if not matches:
            print(f"Warning: no data file found for {rec_id}")
            continue

        filepath = matches[0]
        data_start_s = _extract_unix_timestamp_to_seconds(filepath)
        offset_s = data_start_s - video_start_s

        # Data duration = video duration minus offset (negative offset = data started early = more data)
        data_duration_s = video_duration_s - offset_s

        num_samples = len(pd.read_csv(filepath))
        sample_rate_hz = num_samples / data_duration_s

        rows.append({
            "id": rec_id,
            "num_samples": num_samples,
            "video_duration_s": video_duration_s,
            "offset_s": offset_s,
            "data_duration_s": data_duration_s,
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


def _seconds_to_hhmmss(total_seconds):
    # Convert seconds since midnight to HH:MM:SS string
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def create_times_csv(video_timestamps_path, data_dir, output_path):
    # Write a CSV with video start time, data start time, and offset for each recording
    videos = pd.read_csv(video_timestamps_path)
    rows = []

    for _, row in videos.iterrows():
        rec_id = row["id"]
        video_start_s = _hhmmss_to_seconds(row["timestamp"])

        matches = glob.glob(os.path.join(data_dir, f"{rec_id}_*.csv"))
        if not matches:
            print(f"Warning: no data file found for {rec_id}")
            continue

        data_start_s = _extract_unix_timestamp_to_seconds(matches[0])
        offset_s = data_start_s - video_start_s

        rows.append({
            "id": rec_id,
            "video_start": _seconds_to_hhmmss(video_start_s),
            "data_start": _seconds_to_hhmmss(data_start_s),
            "offset_s": offset_s,
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Times CSV written to {output_path}")


def create_labelled_data(labels_folder, video_timestamps_path, data_dir, labelled_data_output_path, window_s=1.0):
    # get rate estimates and synchronisation mapping
    _, sample_rate = estimate_sample_rates(video_timestamps_path, data_dir)
    mapping = synchronisation_mapping(video_timestamps_path, data_dir)

    all_labelled = []

    # go through labels folder
    for label_file in glob.glob(os.path.join(labels_folder, "*_labels.csv")):
        rec_id = os.path.basename(label_file).replace("_labels.csv", "")

        if rec_id not in mapping:
            print(f"Warning: no sync mapping for {rec_id}, skipping")
            continue

        offset_s = mapping[rec_id]

        # find corresponding data file
        matches = glob.glob(os.path.join(data_dir, f"{rec_id}_*.csv"))
        if not matches:
            print(f"Warning: no data file found for {rec_id}, skipping")
            continue

        data_df = pd.read_csv(matches[0])
        labels_df = pd.read_csv(label_file)
        window_samples = int(window_s * sample_rate)

        for _, label_row in labels_df.iterrows():
            time_into_video_s = _parse_duration_seconds(label_row["time_into_video"])
            label = label_row["label"]

            # align time_into_video to data: offset_s = data_start - video_start
            # so data_time = time_into_video - offset_s
            center_idx = int((time_into_video_s - offset_s) * sample_rate)

            start_idx = max(0, center_idx - window_samples)
            end_idx = min(len(data_df), center_idx + window_samples + 1)

            if center_idx < 0 or center_idx >= len(data_df):
                print(f"Warning: {rec_id} label '{label}' at {label_row['time_into_video']} maps outside data, skipping")
                continue

            window_df = data_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            window_df.insert(0, "sample_offset", range(start_idx - center_idx, end_idx - center_idx))
            window_df.insert(0, "time_into_video", label_row["time_into_video"])
            window_df.insert(0, "label", label)
            window_df.insert(0, "rec_id", rec_id)

            all_labelled.append(window_df)

    if not all_labelled:
        print("No labelled data generated.")
        return None

    result_df = pd.concat(all_labelled, ignore_index=True)
    csv_df = result_df[["rec_id", "time_into_video", "label", "EDA", "PPG1"]].rename(columns={"PPG1": "PPG"})
    csv_df.to_csv(labelled_data_output_path, index=False)
    print(f"Labelled data written to {labelled_data_output_path} ({len(result_df)} rows, {len(all_labelled)} windows)")
    return result_df


if __name__ == "__main__":
    _root = os.path.join(os.path.dirname(__file__), "..")
    video_timestamps_path = os.path.join(_root, "data", "video_timestamps.csv")
    data_dir = os.path.join(_root, "data", "raw_data")

    # sample_rates_df, avg_sample_rate = estimate_sample_rates(video_timestamps_path, data_dir)
    # print(sample_rates_df)

    # mapping = synchronisation_mapping(video_timestamps_path, data_dir)
    # print(mapping)

    times_csv_path = os.path.join(_root, "data", "times.csv")
    create_times_csv(video_timestamps_path, data_dir, times_csv_path)

    labels_folder = os.path.join(_root, "labelling", "labels")
    labelled_data_output_path = os.path.join(_root, "data", "labelled_data", "labelled_data.csv")
    df = create_labelled_data(labels_folder, video_timestamps_path, data_dir, labelled_data_output_path)
