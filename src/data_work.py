from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Schizophrenia"

# Five proneness levels are merged into a binary target:
# 1 = elevated / high / very high proneness, 0 = low / moderate proneness.
CLASS_MAP = {
    "Low Proneness": 0,
    "Moderate Proneness": 0,
    "Elevated Proneness": 1,
    "High Proneness": 1,
    "Very High Proneness": 1,
}


def load_data(path: str | Path, drop_marital_status: bool = True) -> pd.DataFrame:
    """Load the raw CSV, drop identifiers and build the binary target."""
    df = pd.read_csv(path)

    # Patient name is an identifier and carries no signal
    df = df.drop(columns=["Name"], errors="ignore")

    df[TARGET] = df[TARGET].map(CLASS_MAP)
    unmapped = df[TARGET].isna().sum()
    if unmapped:
        raise ValueError(f"{unmapped} rows have an unknown value in '{TARGET}'")
    df[TARGET] = df[TARGET].astype(int)

    # Marital status showed near-zero correlation with the target in the EDA
    if drop_marital_status:
        df = df.drop(columns=["Marital_Status"], errors="ignore")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified split BEFORE any imputation, scaling or oversampling."""
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
