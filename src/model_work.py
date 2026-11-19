import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(x_train: pd.DataFrame, n_neighbors: int = 3, random_state: int = 42) -> Pipeline:
    """Imputation -> scaling/encoding -> SMOTE -> k-NN.

    Everything is fitted on the training data only. SMOTE is applied to the
    training data only (imblearn skips it at predict time and inside CV folds),
    so the test set stays untouched and keeps its real class distribution.
    """
    numeric = x_train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in x_train.columns if c not in numeric]

    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=random_state)),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )
