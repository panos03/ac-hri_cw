"""Central configuration for emotion recognition experiments."""

RANDOM_SEED = 42

SAMPLE_RATE = 64.0
WINDOW_SEC = 30
STRIDE_SEC = 15

PPG_FEATURES = [
    "bpm", "ibi", "sdnn", "sdsd", "rmssd",
    "pnn20", "pnn50", "hr_mad",
    "sd1", "sd2", "s", "sd1/sd2", "breathingrate",
]

EDA_FEATURES = [
    "eda_mean_scl", "eda_slope_scl", "eda_num_scr",
    "eda_mean_scr_amp", "eda_sum_scr_amp",
    "eda_variance", "eda_range", "eda_deriv_mean",
]

ALL_FEATURES = EDA_FEATURES + PPG_FEATURES

CLASS_NAMES = ["excitement", "frustration", "neutral"]

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=RANDOM_SEED,
)

SVM_PARAMS = dict(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    random_state=RANDOM_SEED,
)

SVM_GRID = dict(
    C=[0.1, 1.0, 10.0],
    gamma=["scale", "auto"],
)
