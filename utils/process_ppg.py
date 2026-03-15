# Python script tp process the raw PPG data acquired using correponding Android App
# Example Usage: python .\process_ppg.py <path to CSV file>
## Author(s): Jitesh Joshi (PhD Student, UCL Computer Science)

import numpy as np
import os, csv
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, periodogram
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




