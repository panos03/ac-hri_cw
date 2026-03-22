import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


df = pd.read_csv("../data_collection_and_label_synch/data/labelled_data/full_labelled_001-easy.csv")

df.columns = df.columns.str.strip()
df["label"] = df["label"].fillna("").astype(str).str.strip()

# Standardise label names
df["label"] = df["label"].replace({
    "Excited": "Excitement",
    "Frustrated": "Frustration"
})

# Convert signals to numeric
df["EDA"] = pd.to_numeric(df["EDA"], errors="coerce")
df["PPG"] = pd.to_numeric(df["PPG"], errors="coerce")

# Drop rows with missing values
df = df.dropna(subset=["EDA", "PPG"]).copy()


# Clip extreme values for visual clarity only
df["EDA"] = df["EDA"].clip(lower=0, upper=700)
df["PPG"] = df["PPG"].clip(lower=0, upper=500)

# Reset index after cleaning
df = df.reset_index(drop=True)

# Downsample from 64 Hz
# Window of 8 -> about 8 Hz view from 64 Hz original data
window = 8

def keep_label(x):
    non_empty = x[x != ""]
    return non_empty.iloc[0] if len(non_empty) > 0 else ""

df_down = df.groupby(df.index // window).agg({
    "EDA": "mean",
    "PPG": "mean",
    "label": keep_label
}).reset_index(drop=True)

df_down["time"] = range(len(df_down))

# Find continuous labelled regions
label_regions = []
start = None
current_label = None

for i in range(len(df_down)):
    label = df_down.loc[i, "label"]

    if label != "":
        if start is None:
            start = i
            current_label = label
        elif label != current_label:
            label_regions.append((start, i - 1, current_label))
            start = i
            current_label = label
    else:
        if start is not None:
            label_regions.append((start, i - 1, current_label))
            start = None
            current_label = None

if start is not None:
    label_regions.append((start, len(df_down) - 1, current_label))

# Define label colours + legend
label_colors = {
    "Neutral": "blue",
    "Excitement": "green",
    "Frustration": "orange"
}

emotion_patches = [
    Patch(facecolor="blue", alpha=0.15, label="Neutral"),
    Patch(facecolor="green", alpha=0.15, label="Excitement"),
    Patch(facecolor="orange", alpha=0.15, label="Frustration")
]

# Create stacked plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# EDA plot
ax1.plot(df_down["time"], df_down["EDA"], label="EDA", linewidth=1.5)

for start, end, label in label_regions:
    ax1.axvspan(start, end, color=label_colors.get(label, "gray"), alpha=0.15)

ax1.set_title("EDA over Time with Labelled Events", fontsize=14)
ax1.set_ylabel("EDA", fontsize=12)
ax1.set_ylim(200, 650)
ax1.grid(True, alpha=0.3)

ax1.legend(
    handles=[ax1.lines[0]] + emotion_patches,
    loc="upper right",
    frameon=True
)

# PPG plot
ax2.plot(df_down["time"], df_down["PPG"], label="PPG", linewidth=1.5, color="red")

for start, end, label in label_regions:
    ax2.axvspan(start, end, color=label_colors.get(label, "gray"), alpha=0.15)

ax2.set_title("PPG over Time with Labelled Events", fontsize=14)
ax2.set_xlabel("Time (downsampled index)", fontsize=12)
ax2.set_ylabel("PPG", fontsize=12)
ax2.set_ylim(150, 250)
ax2.grid(True, alpha=0.3)

ax2.legend(
    handles=[ax2.lines[0]] + emotion_patches,
    loc="upper right",
    frameon=True
)

plt.tight_layout()
plt.show()

# Summary output
print("Downsampled rows:", len(df_down))
print("Number of labelled regions:", len(label_regions))
print("\nFirst 10 labelled regions:")
for region in label_regions[:10]:
    print(region)