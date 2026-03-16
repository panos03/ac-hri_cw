# Python script tp process the raw PPG data acquired using correponding Android App
# Example Usage: python .\process_ppg.py <path to CSV file>
## Author(s): Jitesh Joshi (PhD Student, UCL Computer Science)

import numpy as np
import os, csv
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, periodogram, find_peaks
from scipy.interpolate import interp1d
import heartpy as hp
import pandas as pd


def get_ppg_signal_filepaths(datapath):
	ppg_files = []
	for path, dirs, files in os.walk(datapath):
		dirs.sort()
		for fn in sorted(files):
			if os.path.splitext(fn)[-1].lower() != '.csv':
				continue

			# Legacy app format under p00x/s00x folders.
			if fn.startswith('ppgSignal'):
				sub_id = os.path.basename(path)
				ppg_files.append((sub_id, fn, os.path.join(path, fn)))
				continue

			# New full_labelled export format under data/mydata.
			if fn.startswith('full_labelled_'):
				name_body = os.path.splitext(fn)[0]
				parts = name_body.split('_')
				sub_level = parts[-1] if parts else ''
				sub_id = sub_level.split('-')[0] if '-' in sub_level else os.path.basename(path)
				ppg_files.append((sub_id, fn, os.path.join(path, fn)))
	return ppg_files


def get_exertion_label_from_filename(filename):
	if filename.startswith('full_labelled_'):
		name_body = os.path.splitext(filename)[0]
		sub_level = name_body.split('_')[-1]
		if '-' in sub_level:
			return sub_level.split('-', 1)[1]
		return 'unknown'

	parts = filename.split('_')
	if len(parts) > 1:
		return parts[1].replace('.csv', '')
	return 'unknown'


def butter_bandpass(lowcut, highcut, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order=2):
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    y = lfilter(b, a, data)
    return y


def get_clean_segment(ppg_sig, std_n=1.5):
	filtered_clean = [[]]
	start = False
	mean_filtered = np.mean(ppg_sig)
	std_filtered = np.std(ppg_sig)
	min_val = mean_filtered - std_n*std_filtered
	max_val = mean_filtered + std_n*std_filtered
	# print('len(filtered):', len(ppg_sig))
	# print('mean_filtered:', mean_filtered)
	# print('std_filtered:', std_filtered)
	# print('min_val:', min_val)
	# print('max_val:', max_val)
	for dt in list(ppg_sig):
		if dt >= min_val and dt <= max_val:
			if not start:
				start = True
			filtered_clean[-1].append(dt)
		else:
			if start:
				start = False
				filtered_clean.append([])

	# print('Number of segments', len(filtered_clean))
	final_filtered = []
	max_length = 0
	for segs in filtered_clean:
		if len(segs) > max_length:
			final_filtered = segs
			max_length = len(segs)
	filtered_clean = np.array(final_filtered)
	# print('max length of clean signal', max_length)

	return filtered_clean


def get_filtered_ppg(raw_signal, sample_rate=30.0):
	raw_signal = np.asarray(raw_signal, dtype=np.double)
	if raw_signal.size == 0:
		return np.array([], dtype=np.double)

	tElapsed = np.array([i/sample_rate for i in range(len(raw_signal))])
	discard_len = 5.0 #seconds
	raw_signal = raw_signal[tElapsed>discard_len]
	tElapsed = tElapsed[tElapsed>discard_len]

	if raw_signal.size < 10:
		return np.array([], dtype=np.double)

	# raw_signal = get_clean_segment(raw_signal, std_n=3.0)

	f, Pxx_den = periodogram(raw_signal, sample_rate, 'flattop', scaling='spectrum')

	lowcut_ppg = 1.6
	highcut_ppg = 3.0
	lowcut_resp = 0.15
	highcut_resp = 0.3
	order = 2

	Pxx_den_PPG = Pxx_den[np.bitwise_and(f>=lowcut_ppg, f<=highcut_ppg)]
	f_PPG = f[np.bitwise_and(f>=lowcut_ppg, f<=highcut_ppg)]
	if f_PPG.size == 0 or Pxx_den_PPG.size == 0:
		max_power_freq_ppg = 2.2
	else:
		max_power_freq_ppg = f_PPG[np.argmax(Pxx_den_PPG)]
	# print("Freq with max power:", max_power_freq_ppg)

	# Adaptive bandpass filtering
	lowcut_ppg = max_power_freq_ppg - 0.8
	highcut_ppg = max_power_freq_ppg + 0.8

	filtered_PPG = butter_bandpass_filter(raw_signal, lowcut_ppg, highcut_ppg, sample_rate, order=order)
	filtered_resp = butter_bandpass_filter(raw_signal, lowcut_resp, highcut_resp, sample_rate, order=order)

	# discard_len = 5.0 #seconds
	# raw_signal = raw_signal[tElapsed>discard_len]
	# filtered_PPG = filtered_PPG[tElapsed>discard_len]
	# filtered_resp = filtered_resp[tElapsed>discard_len]
	# tElapsed = tElapsed[tElapsed>discard_len]

	filtered_PPG = get_clean_segment(filtered_PPG)

	# fig, ax = plt.subplots(3, 1, sharex=False)

	# ax[0].stem(f, np.sqrt(Pxx_den))
	# # ax[0].set_ylim([1e-4, 5e-1])
	# ax[0].set_xlim([1.6, 3.2])
	# ax[0].set_xlabel('frequency [Hz]')
	# ax[0].set_ylabel('PSD')

	# ax[1].plot(tElapsed, filtered_PPG)
	# ax[1].set_xlabel("Time (seconds)")
	# ax[1].set_ylabel("Filtered PPG Signal")

	# ax[2].plot(tElapsed, filtered_resp)
	# ax[2].set_xlabel("Time (seconds)")
	# ax[2].set_ylabel("Filtered Resp Signal")

	# plt.show()

	return filtered_PPG

def load_PPG_signal(filepath):
	# Try the new mydata tabular format first: rec_id,label,EDA,PPG
	try:
		df = pd.read_csv(filepath)
		if 'PPG' in df.columns:
			raw_signal = pd.to_numeric(df['PPG'], errors='coerce').dropna().to_numpy(dtype=np.double)
			line_index = np.arange(len(raw_signal), dtype=np.double)
			return raw_signal, line_index
	except Exception:
		pass

	# Fallback to legacy two-row format: timestamps(ms) on first row, PPG on second row.
	try:
		with open(filepath, newline='') as csvfile:
			txt_data = csv.reader(csvfile, delimiter=',')
			tElapsed_txt, raw_signal_txt = txt_data
	except Exception:
		if os.path.exists(filepath):
			print("Error reading the file", filepath)
			return np.array([], dtype=np.double), np.array([], dtype=np.double)
		else:
			print("Specified file not found", filepath)
			return np.array([], dtype=np.double), np.array([], dtype=np.double)

	raw_signal = np.array([], dtype=np.double)
	tElapsed = np.array([], dtype=np.double)
	for i in range(len(tElapsed_txt)):
		raw_signal = np.append(raw_signal, np.double(raw_signal_txt[i]))
		tElapsed = np.append(tElapsed, np.double(
			tElapsed_txt[i])/1000.0)  # milliseconds to seconds
	return raw_signal, tElapsed


def get_effective_sample_rate(filename, t_elapsed, default_sample_rate=64.0):
	# full_labelled files are WESAD-style wrist PPG exports sampled at 64 Hz.
	if filename.startswith('full_labelled_'):
		return 64.0, 'fixed_64hz'

	# For legacy files with timestamp vectors, infer sampling rate from median interval.
	t_elapsed = np.asarray(t_elapsed, dtype=np.double)
	if t_elapsed.size > 2:
		delta_t = np.diff(t_elapsed)
		delta_t = delta_t[np.isfinite(delta_t) & (delta_t > 0)]
		if delta_t.size > 0:
			sr = 1.0 / np.median(delta_t)
			if np.isfinite(sr) and 1.0 <= sr <= 500.0:
				return float(sr), 'inferred_from_timestamps'

	return float(default_sample_rate), 'default'

def get_ppg_measures_batch(datapath, sample_rate=64.0):

	csv_fpath = os.path.join(datapath, 'PPG_features.csv')
	csvfile = open(csv_fpath, 'w', newline='')
	fp_writer = csv.writer(csvfile, delimiter=',')
	header = [
		'sub_id', 'label', 'sample_rate_used', 'status',
		'bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50',
		'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate'
	]
	fp_writer.writerow(header)
	csvfile.close()
	
	pklfilepath = os.path.join(datapath, 'PPG_features.pkl')
	# pklfile = open(pklfilepath, 'wb')
	data_dict = {}
	ppg_files = get_ppg_signal_filepaths(datapath)
	print('Found', len(ppg_files), 'PPG signal files under', datapath)

	for sub_id, fn, filepath in ppg_files:
		print('Processing:', sub_id, ':', fn)
		data_dict[sub_id + '_' + fn] = {}
		raw_signal, tElapsed = load_PPG_signal(filepath)
		label = get_exertion_label_from_filename(fn)
		sample_rate_used, sample_rate_source = get_effective_sample_rate(
			fn, tElapsed, default_sample_rate=sample_rate)

		if len(raw_signal) == 0:
			status = 'failed_no_signal'
			vals = [sub_id, label, sample_rate_used, status] + [''] * (len(header) - 4)
			wd = {}
			m = {}
		else:
			filtered = get_filtered_ppg(raw_signal, sample_rate=sample_rate_used)

			if len(filtered) == 0:
				status = 'failed_filter_empty'
				vals = [sub_id, label, sample_rate_used, status] + [''] * (len(header) - 4)
				wd = {}
				m = {}
			else:
				window_len = int(30 * sample_rate_used)
				segments = []

				if len(filtered) > window_len:
					mid_start = int((len(filtered) // 2) - (window_len // 2))
					mid_end = mid_start + window_len
					segments.append(('middle', filtered[mid_start:mid_end]))
					segments.append(('start', filtered[:window_len]))
					segments.append(('end', filtered[-window_len:]))
				else:
					segments.append(('full', filtered))

				wd = {}
				m = {}
				hp_ok = False
				status = 'failed_hp_all_windows'

				for seg_name, seg_data in segments:
					if len(seg_data) < 10:
						continue
					try:
						wd, m = hp.process(seg_data, sample_rate=sample_rate_used)
						hp_ok = True
						if seg_name == 'middle':
							status = f'ok_{seg_name}_{sample_rate_source}'
						else:
							status = f'fallback_{seg_name}_{sample_rate_source}'
						break
					except Exception:
						continue

				if hp_ok:
					vals = [sub_id, label, sample_rate_used, status]
					for metric in header[4:]:
						vals.append(str(m.get(metric, '')))
				else:
					vals = [sub_id, label, sample_rate_used, status] + [''] * (len(header) - 4)

		m['exertion_level'] = label
		m['sub_id'] = sub_id
		data_dict[sub_id + '_' + fn]['wd'] = wd
		data_dict[sub_id + '_' + fn]['m'] = m

		csvfile = open(csv_fpath, 'a+', newline='')
		fp_writer = csv.writer(csvfile, delimiter=',')
		fp_writer.writerow(vals)
		csvfile.close()

	df = pd.DataFrame(data_dict)
	df.to_pickle(pklfilepath)


def get_ppg_measures_windowed(datapath, window_sec=30, stride_sec=15, sample_rate=64.0):
	"""Extract windowed PPG features from full_labelled CSV files."""
	window_len = int(round(window_sec * sample_rate))
	hop_len = int(round(stride_sec * sample_rate))
	if window_len <= 0 or hop_len <= 0:
		raise ValueError('window_sec and stride_sec must produce positive lengths')

	csv_fpath = os.path.join(datapath, 'PPG_features_windowed.csv')
	header = [
		'sub_id', 'label', 'window_index', 'window_start_sec', 'status',
		'bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50',
		'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate'
	]
	with open(csv_fpath, 'w', newline='') as csvfile:
		fp_writer = csv.writer(csvfile, delimiter=',')
		fp_writer.writerow(header)

	ppg_files = get_ppg_signal_filepaths(datapath)
	print('Found', len(ppg_files), 'PPG signal files under', datapath)

	rows_written = 0
	for sub_id, fn, filepath in ppg_files:
		print('Processing:', sub_id, ':', fn)
		label = get_exertion_label_from_filename(fn)
		raw_signal, tElapsed = load_PPG_signal(filepath)
		sample_rate_used, _ = get_effective_sample_rate(
			fn, tElapsed, default_sample_rate=sample_rate
		)

		if len(raw_signal) == 0:
			continue

		filtered = get_filtered_ppg(raw_signal, sample_rate=sample_rate_used)
		if len(filtered) < window_len:
			continue

		windows = get_overlapping_windows(len(filtered), window_len, hop_len)
		for win_idx, (start, end) in enumerate(windows):
			segment = filtered[start:end]
			status = 'ok'
			m = {}
			try:
				_, m = hp.process(segment, sample_rate=sample_rate_used)
			except Exception:
				status = 'failed_hp'

			vals = [sub_id, label, win_idx, start / sample_rate_used, status]
			for metric in header[5:]:
				if status == 'failed_hp':
					vals.append(np.nan)
				else:
					vals.append(m.get(metric, np.nan))

			with open(csv_fpath, 'a+', newline='') as csvfile:
				fp_writer = csv.writer(csvfile, delimiter=',')
				fp_writer.writerow(vals)
			rows_written += 1

	print('Wrote', rows_written, 'window rows to', csv_fpath)
	return csv_fpath


def load_dataframe(filepath):

	df = pd.read_pickle(filepath)
	params = ['sub_id', 'exertion_level', 'bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd',
			'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate']

	feature_dict = {}
	for pr in params:
		feature_dict[pr] = []


	params_dict = df.iloc[1, :]
	for fdict in params_dict:
		for key in fdict.keys():
			feature_dict[key].append(fdict[key])

	ppg_df = pd.DataFrame(feature_dict, columns=params)

	return df, ppg_df
# if __name__ == "__main__":
#    main(sys.argv[1:])


def load_multimodal_signals(filepath):
	"""Load PPG and EDA signals from a full_labelled CSV file."""
	try:
		df = pd.read_csv(filepath)
	except Exception:
		return np.array([], dtype=np.double), np.array([], dtype=np.double)

	ppg = np.array([], dtype=np.double)
	eda = np.array([], dtype=np.double)

	if 'PPG' in df.columns:
		ppg = pd.to_numeric(df['PPG'], errors='coerce').to_numpy(dtype=np.double)
	if 'EDA' in df.columns:
		eda = pd.to_numeric(df['EDA'], errors='coerce').to_numpy(dtype=np.double)

	if ppg.size == 0 or eda.size == 0:
		return np.array([], dtype=np.double), np.array([], dtype=np.double)

	# Keep only indices where both modalities are finite.
	valid = np.isfinite(ppg) & np.isfinite(eda)
	if np.count_nonzero(valid) == 0:
		return np.array([], dtype=np.double), np.array([], dtype=np.double)

	return ppg[valid], eda[valid]


def get_overlapping_windows(signal_len, window_len, hop_len):
	"""Return inclusive-exclusive (start, end) indices for overlapping windows."""
	if signal_len < window_len or window_len <= 0 or hop_len <= 0:
		return []

	windows = []
	start = 0
	while start + window_len <= signal_len:
		end = start + window_len
		windows.append((start, end))
		start += hop_len

	# Include one right-aligned final window when remainder exists.
	if windows and windows[-1][1] < signal_len:
		windows.append((signal_len - window_len, signal_len))

	# Deduplicate while preserving order.
	uniq = []
	seen = set()
	for win in windows:
		if win not in seen:
			seen.add(win)
			uniq.append(win)
	return uniq


def _safe_bandpass(data, low_hz, high_hz, sample_rate, order=2):
	nyq = 0.5 * sample_rate
	low = max(low_hz / nyq, 1e-6)
	high = min(high_hz / nyq, 0.999999)
	if low >= high:
		return np.asarray(data, dtype=np.double)
	b, a = butter(order, [low, high], btype='band')
	return filtfilt(b, a, np.asarray(data, dtype=np.double))


def _safe_lowpass(data, high_hz, sample_rate, order=2):
	nyq = 0.5 * sample_rate
	high = min(high_hz / nyq, 0.999999)
	if high <= 1e-6:
		return np.asarray(data, dtype=np.double)
	b, a = butter(order, high, btype='low')
	return filtfilt(b, a, np.asarray(data, dtype=np.double))


def extract_eda_features(eda_segment, sample_rate=64.0):
	"""Extract robust statistical and event-based EDA features from one segment."""
	eda_segment = np.asarray(eda_segment, dtype=np.double)
	eda_segment = eda_segment[np.isfinite(eda_segment)]
	if eda_segment.size < max(8, int(sample_rate * 3)):
		return {}

	# Band-limited decomposition into tonic and phasic proxies.
	tonic = _safe_lowpass(eda_segment, high_hz=0.05, sample_rate=sample_rate, order=2)
	phasic = eda_segment - tonic

	duration_sec = max(eda_segment.size / sample_rate, 1e-6)
	t = np.arange(eda_segment.size, dtype=np.double) / sample_rate
	slope = float(np.polyfit(t, eda_segment, 1)[0]) if eda_segment.size > 1 else 0.0

	grad = np.diff(eda_segment)
	abs_grad_mean = float(np.mean(np.abs(grad))) if grad.size else 0.0

	f, pxx = periodogram(eda_segment, fs=sample_rate, scaling='spectrum')
	total_power = float(np.sum(pxx)) if pxx.size else 0.0
	lf_mask = (f >= 0.0) & (f < 0.15)
	hf_mask = (f >= 0.15) & (f <= 0.5)
	lf_power = float(np.sum(pxx[lf_mask])) if np.any(lf_mask) else 0.0
	hf_power = float(np.sum(pxx[hf_mask])) if np.any(hf_mask) else 0.0

	# SCR-like peaks from phasic component with an adaptive prominence threshold.
	phasic_std = float(np.std(phasic))
	prom = max(0.05 * phasic_std, 1e-6)
	peaks, props = find_peaks(phasic, prominence=prom, distance=max(1, int(0.5 * sample_rate)))
	peak_prom = props.get('prominences', np.array([], dtype=np.double))

	peak_count = int(peaks.size)
	peak_rate_per_min = float((peak_count / duration_sec) * 60.0)
	peak_amp_mean = float(np.mean(peak_prom)) if peak_prom.size else 0.0
	peak_amp_max = float(np.max(peak_prom)) if peak_prom.size else 0.0

	return {
		'eda_mean': float(np.mean(eda_segment)),
		'eda_std': float(np.std(eda_segment)),
		'eda_min': float(np.min(eda_segment)),
		'eda_max': float(np.max(eda_segment)),
		'eda_range': float(np.max(eda_segment) - np.min(eda_segment)),
		'eda_slope': slope,
		'eda_auc': float(np.trapz(eda_segment, dx=1.0 / sample_rate)),
		'eda_abs_diff_mean': abs_grad_mean,
		'eda_tonic_mean': float(np.mean(tonic)),
		'eda_tonic_std': float(np.std(tonic)),
		'eda_phasic_mean': float(np.mean(phasic)),
		'eda_phasic_std': float(np.std(phasic)),
		'eda_power_total': total_power,
		'eda_power_lf_0_0.15': lf_power,
		'eda_power_hf_0.15_0.5': hf_power,
		'eda_scr_peak_count': peak_count,
		'eda_scr_peak_rate_per_min': peak_rate_per_min,
		'eda_scr_prom_mean': peak_amp_mean,
		'eda_scr_prom_max': peak_amp_max,
	}


def extract_ppg_features_from_segment(ppg_segment, sample_rate=64.0, hp_windowsize=1.2):
	"""Extract PPG-derived heart features from one segment using HeartPy."""
	ppg_segment = np.asarray(ppg_segment, dtype=np.double)
	if ppg_segment.size < max(8, int(sample_rate * 3)):
		return {}, 'failed_short_segment'

	try:
		filtered = get_filtered_ppg(ppg_segment, sample_rate=sample_rate)
		if filtered.size < max(8, int(sample_rate * 3)):
			return {}, 'failed_filter_empty'
		_, m = hp.process(filtered, sample_rate=sample_rate, windowsize=hp_windowsize)
	except Exception:
		return {}, 'failed_heartpy'

	ppg_keys = [
		'bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50',
		'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate'
	]
	features = {}
	for k in ppg_keys:
		features[f'ppg_{k.replace("/", "_")}'] = m.get(k, np.nan)
	return features, 'ok'


def get_windowed_multimodal_features_batch(
	datapath,
	sample_rate=64.0,
	window_sec=30.0,
	overlap=0.5,
	discard_initial_sec=5.0,
	hp_windowsize=1.2,
):
	"""Generate overlapping-window PPG+EDA features for all full_labelled files."""
	window_len = int(round(window_sec * sample_rate))
	hop_len = int(round(window_len * (1.0 - overlap)))
	if window_len <= 0:
		raise ValueError('window_sec must produce a positive window length')
	if hop_len <= 0:
		raise ValueError('overlap too high; hop length became non-positive')

	records = []
	all_status = []

	ppg_files = get_ppg_signal_filepaths(datapath)
	for sub_id, fn, filepath in ppg_files:
		label = get_exertion_label_from_filename(fn)
		ppg_raw, eda_raw = load_multimodal_signals(filepath)

		if ppg_raw.size == 0 or eda_raw.size == 0:
			all_status.append((fn, 'failed_missing_ppg_or_eda'))
			continue

		discard = int(round(discard_initial_sec * sample_rate))
		if discard > 0 and ppg_raw.size > discard and eda_raw.size > discard:
			ppg_raw = ppg_raw[discard:]
			eda_raw = eda_raw[discard:]

		sig_len = int(min(ppg_raw.size, eda_raw.size))
		windows = get_overlapping_windows(sig_len, window_len, hop_len)
		if not windows:
			all_status.append((fn, 'failed_too_short_for_window'))
			continue

		for win_idx, (start, end) in enumerate(windows):
			ppg_seg = ppg_raw[start:end]
			eda_seg = eda_raw[start:end]

			ppg_features, ppg_status = extract_ppg_features_from_segment(
				ppg_seg, sample_rate=sample_rate, hp_windowsize=hp_windowsize
			)
			eda_features = extract_eda_features(eda_seg, sample_rate=sample_rate)

			record = {
				'sub_id': sub_id,
				'label': label,
				'filename': fn,
				'window_id': win_idx,
				'window_start_sec': start / sample_rate,
				'window_end_sec': end / sample_rate,
				'sample_rate': sample_rate,
				'ppg_status': ppg_status,
				'eda_status': 'ok' if eda_features else 'failed_eda_short_segment',
			}
			record.update(ppg_features)
			record.update(eda_features)
			records.append(record)

		all_status.append((fn, f'ok_windows_{len(windows)}'))

	out_df = pd.DataFrame(records)
	csv_path = os.path.join(datapath, 'windowed_multimodal_features.csv')
	pkl_path = os.path.join(datapath, 'windowed_multimodal_features.pkl')
	status_path = os.path.join(datapath, 'windowed_multimodal_status.csv')

	out_df.to_csv(csv_path, index=False)
	out_df.to_pickle(pkl_path)
	pd.DataFrame(all_status, columns=['filename', 'status']).to_csv(status_path, index=False)

	print('Wrote:', csv_path)
	print('Wrote:', pkl_path)
	print('Wrote:', status_path)
	print('Total windows:', len(out_df))
	return out_df


def run_loso_cv(
	feature_df,
	label_col='label',
	group_col='sub_id',
	feature_cols=None,
	model=None,
):
	"""Run Leave-One-Subject-Out CV and return predictions + reports."""
	from sklearn.base import clone
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import classification_report, confusion_matrix
	from sklearn.model_selection import LeaveOneGroupOut
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import StandardScaler

	if label_col not in feature_df.columns:
		raise ValueError(f'Missing label column: {label_col}')
	if group_col not in feature_df.columns:
		raise ValueError(f'Missing group column: {group_col}')

	df = feature_df.copy()
	if feature_cols is None:
		exclude = {
			label_col, group_col, 'filename', 'window_id',
			'window_start_sec', 'window_end_sec', 'sample_rate',
			'ppg_status', 'eda_status'
		}
		feature_cols = [c for c in df.columns if c not in exclude]

	for col in feature_cols:
		df[col] = pd.to_numeric(df[col], errors='coerce')

	df = df.dropna(subset=[label_col, group_col])
	if df.empty:
		raise ValueError('No rows left after dropping missing labels/groups')

	# Median imputation for numeric feature columns.
	medians = df[feature_cols].median(numeric_only=True)
	df[feature_cols] = df[feature_cols].fillna(medians)

	X = df[feature_cols].to_numpy(dtype=np.double)
	y = df[label_col].astype(str).to_numpy()
	groups = df[group_col].astype(str).to_numpy()

	if model is None:
		model = make_pipeline(
			StandardScaler(),
			RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=2, class_weight='balanced')
		)

	logo = LeaveOneGroupOut()
	y_true_all = []
	y_pred_all = []
	fold_details = []

	for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
		clf = clone(model)
		clf.fit(X[train_idx], y[train_idx])
		y_pred = clf.predict(X[test_idx])

		y_true_fold = y[test_idx]
		y_true_all.extend(y_true_fold.tolist())
		y_pred_all.extend(y_pred.tolist())

		fold_details.append({
			'fold': fold_idx,
			'test_subject': str(groups[test_idx][0]),
			'n_test_samples': int(test_idx.size),
			'report': classification_report(y_true_fold, y_pred, output_dict=True, zero_division=0),
		})

	overall_report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
	labels = sorted(list(set(y_true_all) | set(y_pred_all)))
	cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

	return {
		'feature_cols': feature_cols,
		'labels': labels,
		'fold_details': fold_details,
		'y_true': y_true_all,
		'y_pred': y_pred_all,
		'overall_report': overall_report,
		'confusion_matrix': cm,
	}




