import os
import pandas as pd
import numpy as np

def extract_features(window_data):
    features = {}
    
    # --- EDA Features ---
    features['eda_mean'] = window_data['eda_clean'].mean()
    features['eda_std'] = window_data['eda_clean'].std()
    # Simple peak count (crude version)
    features['eda_peaks'] = (window_data['eda_clean'].diff() > 0).sum()

    # --- PPG Features ---
    features['ppg_mean'] = window_data['ppg_clean'].mean()
    # In a real scenario, use a library like 'heartpy' or 'neurokit2' for HRV
    features['ppg_max'] = window_data['ppg_clean'].max()
    
    return features

def process_coursework_data(raw_csv_path, label_df):
    raw_data = pd.read_csv(raw_csv_path)
    # Preprocess first
    # raw_data = preprocess_signals(raw_data) 
    
    feature_list = []
    
    for index, row in label_df.iterrows():
        # logic to slice raw_data based on 'time_into_video'
        # Example: take 5 seconds before the label timestamp
        timestamp = row['time_into_video'] 
        label = row['label']
        
        # Placeholder for slicing logic
        window = raw_data.iloc[index*100 : (index+1)*100] # Simplified indexing
        
        feat = extract_features(window)
        feat['target_label'] = label
        feature_list.append(feat)
    
    # Save to processed_data folder
    output_df = pd.DataFrame(feature_list)
    os.makedirs('processed_data', exist_ok=True)
    output_df.to_csv('processed_data/final_features.csv', index=False)
    print("Features saved to processed_data/final_features.csv")
