"""
Data ingestion and imputation module.
"""
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from src.config import DEFAULT_DATA_PATHS


def find_dataset_file(explicit_path: Optional[str] = None) -> Path:
    """Find the dataset Excel file from explicit argument or default locations."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified dataset file not found: {explicit_path}")

    for p in DEFAULT_DATA_PATHS:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find 'CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx'. "
        "Please place it in 'data/' or pass --data-path."
    )


def load_datasets(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load Training_Data, Test_Data, and optional Sample_Submission sheets from the dataset workbook.
    """
    path = find_dataset_file(file_path)
    print(f"[DataLoader] Loading dataset from: {path.resolve()}")

    xl = pd.ExcelFile(path)
    train_df = pd.read_excel(xl, sheet_name="Training_Data")
    test_df = pd.read_excel(xl, sheet_name="Test_Data")

    sample_sub = None
    if "Sample_Submission" in xl.sheet_names:
        sample_sub = pd.read_excel(xl, sheet_name="Sample_Submission")

    print(f"[DataLoader] Loaded {len(train_df)} training records and {len(test_df)} test records.")
    return train_df, test_df, sample_sub


def impute_sensor_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Impute missing values in Sensor_S1, Sensor_S2, Sensor_S3 using the median of valid records from the training set.
    Sensor_S4 is intentionally not imputed since it is dropped as noise.
    Flags missing values in metadata for inspection.
    """
    train = train_df.copy()
    test = test_df.copy()

    # Calculate medians on valid training records
    valid_train = train[train["Validity_Label"] == "Valid"]
    impute_cols = ["Sensor_S1", "Sensor_S2", "Sensor_S3"]
    medians = {}

    for col in impute_cols:
        med = float(valid_train[col].median())
        medians[col] = med

        # Record missingness indicators prior to imputation
        train[f"{col}_was_missing"] = train[col].isnull()
        test[f"{col}_was_missing"] = test[col].isnull()

        # Perform median imputation
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)

    print("[Imputation] Sensor medians from valid records:", {k: round(v, 4) for k, v in medians.items()})
    return train, test, medians
