"""
Anomaly detection module: rule-based screening and ML validity classifier.
"""
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

from src.config import (
    SENTINEL_VALUES,
    SENSOR_S2_THRESHOLD,
    SENSOR_S3_SPIKE_THRESH,
    SENSOR_S3_LOAD_THRESH,
    SENSOR_S3_VOLTAGE_THRESH,
    SENSOR_S1_MIN_THRESH,
    FINGERPRINT_COLUMNS,
)


def apply_anomaly_rules(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Applies the 5 domain-specific invalidity screening rules discovered from EDA:
    1. Sentinel/negative values in S1, S2, S3 (0.0, 1.0, 25.0, or < 0).
    2. Duplicate 7-column sensor fingerprint (values rounded to 3 decimal places).
    3. Sensor_S2 exceeding physical threshold (> 21.6 °C).
    4. Sensor_S3 spike (> 30.0 °C) under low-load conditions (Load < 80 A and Voltage < 28 kV).
    5. Sensor_S1 dead sensor (< 1.0 °C and non-null).
    """
    # Rule 1: Sentinel & negative values
    r1 = pd.Series(False, index=df.index)
    for col in ["Sensor_S1", "Sensor_S2", "Sensor_S3"]:
        r1 |= df[col].isin(SENTINEL_VALUES) | (df[col] < 0)

    # Rule 2: Duplicate sensor fingerprinting
    rounded_fp = df[FINGERPRINT_COLUMNS].round(3)
    r2 = rounded_fp.duplicated(keep=False)

    # Rule 3: Sensor_S2 anomaly
    r3 = df["Sensor_S2"] > SENSOR_S2_THRESHOLD

    # Rule 4: Sensor_S3 extreme spike without high load
    r4 = (
        (df["Sensor_S3"] > SENSOR_S3_SPIKE_THRESH)
        & (df["Load_Current_A"] < SENSOR_S3_LOAD_THRESH)
        & (df["Applied_Voltage_kV"] < SENSOR_S3_VOLTAGE_THRESH)
    )

    # Rule 5: Near-zero Sensor_S1
    r5 = (df["Sensor_S1"] < SENSOR_S1_MIN_THRESH) & df["Sensor_S1"].notnull()

    rule_mask = r1 | r2 | r3 | r4 | r5

    breakdown = {
        "rule_1_sentinels_negative": int(r1.sum()),
        "rule_2_duplicate_fingerprint": int(r2.sum()),
        "rule_3_sensor_s2_anomaly": int(r3.sum()),
        "rule_4_sensor_s3_spike": int(r4.sum()),
        "rule_5_near_zero_s1": int(r5.sum()),
        "total_rule_flagged": int(rule_mask.sum()),
    }
    return rule_mask, breakdown


def train_validity_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    rule_mask: pd.Series,
    random_state: int = 42,
) -> Tuple[XGBClassifier, dict]:
    """
    Train an XGBoost classifier on all labeled training records to catch subtle edge cases.
    Performs 5-fold StratifiedKFold CV and ensembles rule-based flags with classifier probabilities.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    clf = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
    )

    # Cross-validation out-of-fold probability predictions
    oof_probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    # Combined prediction: Rule-based invalidity is hard positive, else probability > 0.5
    oof_pred = rule_mask | (oof_probs > 0.5)

    report_dict = classification_report(y, oof_pred, target_names=["Valid", "Invalid"], output_dict=True)
    report_text = classification_report(y, oof_pred, target_names=["Valid", "Invalid"])
    conf_mat = confusion_matrix(y, oof_pred)
    auc_score = roc_auc_score(y, oof_probs)

    metrics = {
        "confusion_matrix": conf_mat.tolist(),
        "classification_report": report_text,
        "classification_report_dict": report_dict,
        "roc_auc": float(auc_score),
    }

    # Fit final classifier on all training data
    clf.fit(X, y)
    return clf, metrics


def predict_test_validity(
    test_df: pd.DataFrame,
    feature_cols: list,
    clf: XGBClassifier,
    threshold: float = 0.5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Predict validity labels on test data.
    Final rule: if rule_invalid == True -> 'Invalid'.
    Else if classifier probability > threshold -> 'Invalid'.
    Otherwise -> 'Valid'.
    """
    rule_invalid, breakdown = apply_anomaly_rules(test_df)
    test_probs = clf.predict_proba(test_df[feature_cols])[:, 1]
    clf_invalid = test_probs > threshold

    final_invalid = rule_invalid | clf_invalid
    validity_labels = pd.Series(
        np.where(final_invalid, "Invalid", "Valid"),
        index=test_df.index,
        name="Validity_Label",
    )
    return validity_labels, pd.Series(test_probs, index=test_df.index, name="Invalidity_Probability")
