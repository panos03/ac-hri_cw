import glob
import os
import re
from datetime import datetime, timezone

import pandas as pd


def _parse_duration_seconds(duration_str):
    # Parse 'MM:SS' duration string to total seconds
    minutes, seconds = map(int, duration_str.split(":"))
    return minutes * 60 + seconds


def _normalize_text(value):
    # Remove whitespace artifacts from manually edited CSV files.
    if pd.isna(value):
        return ""
    text = str(value)
    # Strip UTF-8 NBSP (\xc2\xa0) decoded through latin-1, bare NBSP, and BOM.
    text = text.replace("\xc2\xa0", " ").replace("\xa0", " ").replace("\ufeff", "")
    # Remove any remaining non-printable / non-ASCII artefact characters (e.g. stray Â).
    text = re.sub(r"[^\x20-\x7e:.\-/]", "", text)
    return text.strip()


def _parse_time_into_video_seconds(raw_value):
    # Accept either a single time (e.g. 0:10) or a range (e.g. 0:12-0:14).
    text = _normalize_text(raw_value)
    if "-" in text:
        text = text.split("-", 1)[0].strip()

    if ":" in text:
        return _parse_duration_seconds(text)

    # Some files use M.SS (e.g. 0.23, 1.04) instead of MM:SS.
    if re.fullmatch(r"\d+\.\d{1,2}", text):
        minutes_str, seconds_str = text.split(".", 1)
        minutes = int(minutes_str)
        seconds = int(seconds_str)
        if seconds < 60:
            return minutes * 60 + seconds

    # Fallback for true decimal-minute values.
    return int(round(float(text) * 60))


def _extract_rec_id_from_label_filename(label_file):
    basename = os.path.basename(label_file)
    match = re.match(r"(.+?)(?:_labels|-labels)\.csv$", basename, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def _get_label_columns(labels_df):
    normalized_to_actual = {}
    for col in labels_df.columns:
        clean_col = re.sub(r"[^a-z0-9]+", "", _normalize_text(col).lower())
        normalized_to_actual[clean_col] = col

    time_col = normalized_to_actual.get("timeintovideo")
    label_col = normalized_to_actual.get("label")
    return time_col, label_col


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
    for label_file in glob.glob(os.path.join(labels_folder, "*labels.csv")):
        rec_id = _extract_rec_id_from_label_filename(label_file)
        if not rec_id:
            print(f"Warning: could not parse recording id from {os.path.basename(label_file)}, skipping")
            continue

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
        labels_df = pd.read_csv(label_file, encoding="latin-1")
        time_col, label_col = _get_label_columns(labels_df)
        if not time_col or not label_col:
            print(f"Warning: {os.path.basename(label_file)} missing required columns, skipping")
            continue

        window_samples = int(window_s * sample_rate)

        for _, label_row in labels_df.iterrows():
            time_into_video_raw = _normalize_text(label_row[time_col])
            label = _normalize_text(label_row[label_col])
            if not time_into_video_raw or not label:
                continue

            time_into_video_s = _parse_time_into_video_seconds(time_into_video_raw)

            # align time_into_video to data: offset_s = data_start - video_start
            # so data_time = time_into_video - offset_s
            center_idx = int((time_into_video_s - offset_s) * sample_rate)

            start_idx = max(0, center_idx - window_samples)
            end_idx = min(len(data_df), center_idx + window_samples + 1)

            if center_idx < 0 or center_idx >= len(data_df):
                print(f"Warning: {rec_id} label '{label}' at {time_into_video_raw} maps outside data, skipping")
                continue

            window_df = data_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            window_df.insert(0, "sample_offset", range(start_idx - center_idx, end_idx - center_idx))
            window_df.insert(0, "time_into_video", time_into_video_raw)
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

    output_dir = os.path.dirname(labelled_data_output_path)
    for participant_id, participant_df in csv_df.groupby(csv_df["rec_id"].str.split("-").str[0]):
        participant_output_path = os.path.join(output_dir, f"labelled_data_{participant_id}.csv")
        participant_df.to_csv(participant_output_path, index=False)
        print(f"Participant file written: {participant_output_path} ({len(participant_df)} rows)")

    return result_df


def create_full_labelled_data(labels_folder, video_timestamps_path, data_dir, output_dir):
    # For each recording, compute freq = round(num_entries / video_duration_s).
    # Each row i maps to video_time = i / freq seconds.
    # Rows whose video_time falls within a labelled range receive that label; others are blank.
    videos = pd.read_csv(video_timestamps_path)
    video_info = {row["id"]: row for _, row in videos.iterrows()}

    all_dfs = []

    for label_file in sorted(glob.glob(os.path.join(labels_folder, "*labels.csv"))):
        rec_id = _extract_rec_id_from_label_filename(label_file)
        if not rec_id:
            continue

        if rec_id not in video_info:
            print(f"Warning: no video info for {rec_id}, skipping")
            continue

        video_duration_s = _parse_duration_seconds(video_info[rec_id]["duration"])

        matches = glob.glob(os.path.join(data_dir, f"{rec_id}_*.csv"))
        if not matches:
            print(f"Warning: no data file for {rec_id}, skipping")
            continue

        data_df = pd.read_csv(matches[0])
        n = len(data_df)
        freq = round(n / video_duration_s)

        labels_df = pd.read_csv(label_file, encoding="latin-1")
        time_col, label_col = _get_label_columns(labels_df)
        if not time_col or not label_col:
            print(f"Warning: {os.path.basename(label_file)} missing required columns, skipping")
            continue

        # Build label ranges: list of (start_s, end_s, label_str)
        label_ranges = []
        for _, label_row in labels_df.iterrows():
            time_raw = _normalize_text(label_row[time_col])
            label_str = _normalize_text(label_row[label_col])
            if not time_raw or not label_str:
                continue
            if "-" in time_raw:
                parts = time_raw.split("-", 1)
                start_s = _parse_time_into_video_seconds(parts[0])
                end_s = _parse_time_into_video_seconds(parts[1])
            else:
                start_s = _parse_time_into_video_seconds(time_raw)
                end_s = start_s + 1
            label_ranges.append((start_s, end_s, label_str))

        # Vectorised label assignment: compute video time for every row
        video_times = pd.Series(range(n), dtype=float) / freq
        row_labels = pd.Series([""] * n, dtype=object)
        for start_s, end_s, label_str in label_ranges:
            mask = (video_times >= start_s) & (video_times < end_s)
            row_labels[mask] = label_str

        result_df = data_df[["EDA", "PPG1"]].rename(columns={"PPG1": "PPG"}).copy()
        result_df.insert(0, "label", row_labels.values)
        result_df.insert(0, "rec_id", rec_id)

        out_path = os.path.join(output_dir, f"full_labelled_{rec_id}.csv")
        result_df.to_csv(out_path, index=False)
        print(f"Full labelled data written: {out_path} ({n} rows, freq={freq} Hz)")

        all_dfs.append(result_df)

    if not all_dfs:
        print("No full labelled data generated.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Per-participant files (combining easy + hard)
    for participant_id, p_df in combined_df.groupby(combined_df["rec_id"].str.split("-").str[0]):
        p_path = os.path.join(output_dir, f"full_labelled_{participant_id}.csv")
        p_df.to_csv(p_path, index=False)
        print(f"Participant full labelled file written: {p_path} ({len(p_df)} rows)")

    return combined_df


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

    full_output_dir = os.path.join(_root, "data", "labelled_data")
    create_full_labelled_data(labels_folder, video_timestamps_path, data_dir, full_output_dir)
