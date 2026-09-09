"""
Configuration module for CPRI PowerNext-AI pipeline.
"""
from pathlib import Path

# Team Configuration
DEFAULT_TEAM_NAME = "PowerNext_AI"

# File Paths
DEFAULT_DATA_PATHS = [
    Path("data/CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"),
    Path("CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"),
    Path("/Users/preethamhs/Downloads/CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"),
]

DELIVERABLES_DIR = Path("deliverables")

# Rule-Based Anomaly Constants
SENTINEL_VALUES = [0.0, 1.0, 25.0]
SENSOR_S2_THRESHOLD = 21.6
SENSOR_S3_SPIKE_THRESH = 30.0
SENSOR_S3_LOAD_THRESH = 80.0
SENSOR_S3_VOLTAGE_THRESH = 28.0
SENSOR_S1_MIN_THRESH = 1.0

FINGERPRINT_COLUMNS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
]

# Physical bounds for clipping Reference Parameter (°C above ambient)
PHYSICAL_BOUNDS = (10.0, 65.0)

# Ensemble Weights for Hot-Spot Temperature Regression
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.60,
    "rf": 0.30,
    "ridge": 0.10,
}

# Features to exclude from modeling
DROPPED_FEATURES = ["Sensor_S4", "Test_ID", "Validity_Label", "Reference_Parameter", "rule_invalid", "clf_invalid"]
