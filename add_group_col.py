# add_group_col.py
import pandas as pd
import numpy as np

df = pd.read_csv("features_csv/features.csv")

groups = pd.Series("Unknown", index=df.index)

has_info = df[["Control", "PD(-RBD)", "PD(+RBD)", "iRBD", "PLM"]].notna().any(axis=1)
groups[has_info] = "Control"

if "PD(-RBD)" in df.columns:
    groups[df["PD(-RBD)"] == 1] = "PD(-RBD)"
if "PD(+RBD)" in df.columns:
    groups[df["PD(+RBD)"] == 1] = "PD(+RBD)"
if "iRBD" in df.columns:
    groups[df["iRBD"] == 1] = "iRBD"
if "PLM" in df.columns:
    groups[df["PLM"] == 1] = "Control"  # PLM -> Control, same as prepare_data.py

df["group"] = groups
df.to_csv("features_csv/features_with_group.csv", index=False)
print(df["group"].value_counts())