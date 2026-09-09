"""
Physics-informed feature engineering module.
"""
from typing import List, Tuple
import pandas as pd
from src.config import DROPPED_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate physics-informed and statistical features:
    - Joule heating dissipation: I^2 (current squared)
    - Electrical power dissipation proxy: I^2 * V
    - Terminal sensor averages: (S1 + S2) / 2
    - Global telemetry average: (S1 + S2 + S3) / 3
    """
    data = df.copy()

    # Joule Heating (I^2 * R thermal proxy)
    data["I_squared"] = data["Load_Current_A"] ** 2

    # Electrical Power Dissipation Proxy
    data["I_squared_V"] = data["I_squared"] * data["Applied_Voltage_kV"]

    # Terminal sensors mean
    data["S1_S2_mean"] = (data["Sensor_S1"] + data["Sensor_S2"]) / 2

    # Reliable sensor ensemble mean
    data["S_all_mean"] = (data["Sensor_S1"] + data["Sensor_S2"] + data["Sensor_S3"]) / 3

    return data


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Filter dataframe columns to return valid predictor features,
    explicitly dropping Sensor_S4, Test_ID, targets, and metadata.
    """
    return [
        col
        for col in df.columns
        if col not in DROPPED_FEATURES and not col.endswith("_was_missing")
    ]
