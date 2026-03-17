# Emotion Recognition from Physiological Signals

Automatic recognition of emotional states (excitement, frustration, neutral) from finger-worn EDA and PPG signals, collected  with 7 subjects across easy and hard videogame conditions.

---

## Pipeline

```
data_collection_and_label_synch/
  data/raw_data/                   Raw EDA + PPG @ 64 Hz (14 recordings)
  data/labelled_data/              Time-synchronised sensor data
  labelling/labels/                Manual emotion annotations (video-coded)
         │
         ▼
  labelled_data_generation.py      Aligns raw sensor data to video timestamps
         │
         ▼
emotion_recognition_system/
  utils/process_eda.py  ──────────► EDA features (8):
  utils/process_ppg.py  ──────────► PPG/HRV features (13)
         │                          30 s windows, 15 s stride
         ▼
  utils/merge_features.py ────────► combined_features_windowed.csv (121 windows)
         │
         ▼
  data_processing/relabel.py ─────► emotion_features.csv (88 labelled windows)
         │                          Maps video annotations → feature windows
         ▼
  models/train.py ────────────────► LOSO-CV (7 folds, one subject held out)
         │                          RF + SVM × EDA-only / PPG-only / Combined
         ▼
  results/                         Metrics, confusion matrices, feature importance
```

---

## Repository Structure

```
ac-hri_cw/
├── data_collection_and_label_synch/
│   ├── data/
│   │   ├── raw_data/              Raw sensor CSVs (EDA, PPG @ 64 Hz)
│   │   ├── labelled_data/         Time-indexed sensor data (input to feature extraction)
│   │   ├── times.csv              Recording start times
│   │   └── video_timestamps.csv   Video sync reference
│   └── labelling/
│       ├── labels/                Emotion annotations per recording (excitement/frustration/neutral)
│       └── labelled_data_generation.py
│
├── emotion_recognition_system/
│   ├── data_processing/
│   │   ├── feature_extraction.py  Entry point — runs full windowed extraction
│   │   ├── relabel.py             Maps emotion annotations to feature windows
│   │   └── preprocessing.py       Data loading, imputation, LOSO-CV splits
│   ├── utils/
│   │   ├── process_eda.py         EDA signal processing + feature extraction
│   │   ├── process_ppg.py         PPG signal processing + HRV feature extraction
│   │   └── merge_features.py      Merges EDA and PPG windowed features
│   ├── models/
│   │   ├── config.py              Feature lists, class names, model hyperparameters
│   │   ├── train.py               LOSO cross-validation experiments
│   │   ├── predict.py             EmotionPredictor — end-to-end inference API
│   │   └── analysis.py            Confusion matrices, feature importance, plots
│   ├── features/                  Extracted feature CSVs (generated)
│   ├── results/                   Experiment outputs — metrics, plots (generated)
│   └── 04_model_training_and_evaluation.ipynb   Main analysis notebook
│
└── requirements.txt
```

---

## Features

| Modality | Features |
|----------|----------|
| **EDA** (8) | Mean/slope of tonic (SCL), SCR count, mean/sum SCR amplitude, variance, range, mean derivative |
| **PPG** (13) | BPM, IBI, SDNN, SDSD, RMSSD, pNN20, pNN50, HR MAD, SD1, SD2, S, SD1/SD2, breathing rate |

---

## Models & Results

6 experiments: **Random Forest** and **SVM** × **EDA-only / PPG-only / Combined**, evaluated with Leave-One-Subject-Out cross-validation.

| Experiment | N | Accuracy | Macro F1 |
|------------|---|----------|----------|
| RF PPG-only | 39 | **56.4%** | **0.516** |
| RF EDA-only | 88 | 45.5% | 0.291 |
| RF Combined | 88 | 44.3% | 0.245 |
| SVM Combined | 88 | 30.7% | 0.275 |
| SVM PPG-only | 39 | 41.0% | 0.219 |
| SVM EDA-only | 88 | 25.0% | 0.212 |

Labels: **excitement** (51 windows), **frustration** (25), **neutral** (12). Class-balanced weights used to mitigate imbalance.

---

## Usage

**Re-run feature extraction** (from `emotion_recognition_system/`):
```bash
python -m data_processing.feature_extraction
```

**Re-run relabelling**:
```python
from data_processing.relabel import relabel_features
relabel_features(
    features_csv_path='features/combined_features_windowed.csv',
    labels_folder='../data_collection_and_label_synch/labelling/labels',
    times_csv_path='../data_collection_and_label_synch/data/times.csv',
    output_csv_path='features/emotion_features.csv',
)
```

**Train models** (from `emotion_recognition_system/`):
```bash
python -m models.train
```

**Inference**:
```python
from models.predict import EmotionPredictor
predictor = EmotionPredictor('models/trained/RF_EDA-only.pkl')
result = predictor.predict_from_raw(eda_window)   # 1920 samples @ 64 Hz
# result: {'prediction': 'excitement', 'confidence': 0.72, 'probabilities': {...}}
```

**Full walkthrough**: `04_model_training_and_evaluation.ipynb`

---

## Requirements

```
numpy scipy pandas matplotlib heartpy scikit-learn
```
```bash
pip install -r requirements.txt
```