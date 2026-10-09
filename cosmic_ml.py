from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURES = [
    "Right_Ascension", "Declination",
    "U_Magnitude", "G_Magnitude", "R_Magnitude",
    "I_Magnitude", "Z_Magnitude", "Redshift"
]
CLASSES = ["STAR", "QSO", "GALAXY"]

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    if "Class" not in df.columns:
        raise ValueError("The dataset must contain a 'Class' column.")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    original_rows = len(df)
    original_cols = len(df.columns)
    missing_before = int(df[FEATURES + ["Class"]].isna().sum().sum())

    df["Class"] = df["Class"].astype("string").str.strip().str.upper()
    df = df[df["Class"].isin(CLASSES)].copy()
    valid_label_rows = len(df)
    before_dedup = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = before_dedup - len(df)

    # Use only measured numeric features. Exclude IDs, copied labels,
    # free text, and other columns that could leak the answer.
    X = df[FEATURES].copy()
    y = df["Class"].astype(str).copy()
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    X.loc[~X["Right_Ascension"].between(0, 360), "Right_Ascension"] = np.nan
    X.loc[~X["Declination"].between(-90, 90), "Declination"] = np.nan

    # Treat extreme negative magnitudes as common missing-value sentinels.
    # Ordinary negative magnitudes are retained.
    mag_cols = ["U_Magnitude", "G_Magnitude", "R_Magnitude",
                "I_Magnitude", "Z_Magnitude"]
    for col in mag_cols:
        X.loc[X[col] < -50, col] = np.nan
    X.loc[X["Redshift"] < 0, "Redshift"] = np.nan

    # Keep a feature if it has at least one usable value and is not constant.
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    y = y.loc[X.index]

    info = {
        "original_rows": original_rows,
        "original_columns": original_cols,
        "missing_values_before": missing_before,
        "invalid_class_rows_removed": original_rows - valid_label_rows,
        "duplicates_removed": duplicates_removed,
        "clean_rows": len(X),
        "features": list(X.columns),
        "missing_values_after": int(X.isna().sum().sum()),
        "class_counts": y.value_counts().to_dict(),
    }
    return X, y, info

def train_and_evaluate(csv_path):
    X, y, info = load_and_clean(csv_path)
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Need at least two classes and at least two records per class.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "report": classification_report(
            y_test, pred, labels=CLASSES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=CLASSES).tolist(),
        "labels": CLASSES,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "actual": y_test.reset_index(drop=True),
        "predicted": pd.Series(pred),
        "X_test": X_test.reset_index(drop=True),
    }
    return model, X, y, info, metrics
