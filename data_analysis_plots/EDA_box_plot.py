import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data_collection_and_label_synch/data/labelled_data/labelled_data.csv")

# clean columns
df.columns = df.columns.str.strip()

# clean labels
df["label"] = df["label"].astype(str).str.strip()

# standardise labels
df["label"] = df["label"].replace({
    "Excited": "Excitement",
    "Frustrated": "Frustration"
})

# convert to numeric
df["EDA"] = pd.to_numeric(df["EDA"], errors="coerce")

# drop bad rows
df = df.dropna(subset=["rec_id", "time_into_video", "label", "EDA"])

# group into events
grouped = (
    df.groupby(["rec_id", "time_into_video", "label"], as_index=False)
      .agg({"EDA": "mean"})
)

grouped = grouped[grouped["EDA"] < 450]

# check result
print(grouped["label"].value_counts())

# plot
plt.figure(figsize=(8,6))
sns.boxplot(data=grouped, x="label", y="EDA")
plt.title("EDA Distribution by Emotion Label")
plt.xlabel("Emotion Label")
plt.ylabel("Mean EDA")
plt.show()