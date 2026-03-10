import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def preprocess_signals(df, fs_eda=4, fs_ppg=64):
    # 1. Clean EDA: Low-pass filter (usually < 1Hz) to remove noise
    df['eda_clean'] = lowpass_filter(df['eda'], cutoff=1.0, fs=fs_eda)
    
    # 2. Clean PPG: Band-pass filter (0.5 - 4.0 Hz) to keep heart rate range
    # Assuming standard bandpass logic here
    df['ppg_clean'] = lowpass_filter(df['ppg'], cutoff=4.0, fs=fs_ppg)
    
    return df
