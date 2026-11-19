"""Exploratory analysis: class balance, missing values, marital status, correlations."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_work import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/SchizophreniaSymptomnsData.csv")
parser.add_argument("--out", default="results/eda")
args = parser.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

df = load_data(args.data, drop_marital_status=False)
print("Shape:", df.shape)
print("Class counts:\n", df["Schizophrenia"].value_counts())
print("Missing values:\n", df.isna().sum())

df.isna().sum().to_csv(out / "missing_values.csv", header=["missing"])

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x="Schizophrenia", data=df, ax=ax)
ax.set_title("Class distribution (binary target)")
fig.tight_layout()
fig.savefig(out / "class_distribution.png", dpi=150)
plt.close(fig)

if "Marital_Status" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Marital_Status", y="Schizophrenia", data=df, estimator="mean", ax=ax)
    ax.set_ylabel("Share of class 1")
    ax.set_title("Target vs. marital status")
    fig.tight_layout()
    fig.savefig(out / "marital_status_vs_target.png", dpi=150)
    plt.close(fig)

encoded = pd.get_dummies(df, drop_first=False).astype(float)
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(encoded.corr(), annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, ax=ax)
ax.set_title("Correlation matrix")
fig.tight_layout()
fig.savefig(out / "correlation_matrix.png", dpi=150)
plt.close(fig)
print("EDA saved to", out)
