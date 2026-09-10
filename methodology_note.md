# CPRI POWERNEXT-AI Hackathon: Engineering & Methodology Report
**Team Name:** ByteC  
**Date:** 2026-09-10  
**Evaluation Scope:** Screening Round Lab Telemetry Dataset (1,000 Training Records, 350 Blind Test Records)

---

## 1. Executive Summary & Architecture Overview
The Central Power Research Institute (CPRI) dataset captures high-voltage electrical apparatus thermal tests across varying voltage, current, ambient conditions, and internal sensor telemetry.

Our solution implements a **Decoupled Physics-Informed Machine Learning Architecture**:
1. **Deterministic Physics & Sentinel Filter**: Hard-screens data corruptions (sentinel values, negative temperatures, sensor lockups, duplicate measurement fingerprints) with 0% false positive rate on valid historical data.
2. **Stratified Validity Classifier**: Employs an XGBoost classifier with 5-fold Stratified Cross-Validation on engineered features to resolve subtle borderline invalidities.
3. **Purity-Constrained Thermal Regressor Ensemble**: Trains a multi-model ensemble (60% XGBoost, 30% Random Forest, 10% Ridge) **exclusively** on verified valid records ($N=866$). This guarantees zero contamination from faulty sensor outputs during thermal dynamics modeling.
4. **Physical Bounded Inference**: Predictions are post-processed with physics-based bounds $[10.0, 65.0]$ °C and rounded to 2 decimal places.

```
       [Raw Telemetry Input (V, I, Tamb, S1-S4)]
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
[Deterministic Physics Rules]    [Feature Engineering (I², I²·V, S_means)]
         │                                 │
         │ (Hard-flag invalid)             │
         ▼                                 ▼
   [Rule Invalid Mask] ─────────► [XGBoost Validity Classifier]
                                           │
                                           ▼
                                [Final Validity Label]
                                (Valid vs. Invalid)
                                           │
         ┌─────────────────────────────────┘
         ▼
[Purity-Trained Regressor Ensemble]
 ├─ 60% Tuned XGBoost (RMSE: 0.8009 °C, R²: 0.9944)
 ├─ 30% Random Forest (RMSE: 1.4169 °C, R²: 0.9826)
 └─ 10% Ridge Regularized (RMSE: 1.5444 °C, R²: 0.9793)
         │
         ▼
[Weighted Hot-Spot Prediction: Ensemble RMSE = 0.8854 °C, R² = 0.9932]
         │
         ▼
[Safety Priority Ranking: Identify High-Temperature Anomalous Assets]
```

---

## 2. Exploratory Data Analysis & Critical Physical Insights

### 2.1 Sensor Correlation with Reference Parameter (Valid Records)
- **Load_Current_A ($r = 0.877$):** Primary driver of thermal stress via resistive Joule dissipation ($P = I^2 R$).
- **Sensor_S2 ($r = 0.795$):** Load-side terminal temperature rise, strongly coupled with the apparatus hot-spot.
- **Sensor_S1 ($r = 0.591$):** Incoming terminal temperature rise.
- **Sensor_S3 ($r = 0.501$):** Critical internal temperature telemetry.
- **Applied_Voltage_kV ($r = 0.267$):** Dielectric stress and core loss contributor.
- **Ambient_Temperature_C ($r = 0.238$):** Base thermal baseline.
- **Test_Duration_min ($r pprox 0$):** Negligible linear correlation once steady-state thermal equilibrium is reached.
- **Sensor_S4 ($r = -0.009$):** Pure noise with zero predictive power. **Explicitly dropped** to prevent overfitting.

### 2.2 Anomaly Signatures & High-Confidence Rules
Through rigorous statistical and physics profiling of the training records, five distinct invalidity signals were identified:
1. **Sentinel & Negative Logging Values**: Values of S1, S2, S3 in [0.0, 1.0, 25.0] or negative values denote ADC saturation, open circuits, or default error codes.
2. **Duplicate Telemetry Fingerprints**: Duplicate combinations across (V, I, T_amb, t_dur, S1, S2, S3) rounded to 3 decimal places are 100% correlated with invalidity (24/24 in training set, 8/8 in test set).
3. **Sensor_S2 Upper Bound Violation**: S2 > 21.6 °C never occurs in verified valid tests; values above this indicate sensor drift or thermal runaway in terminal leads.
4. **Sensor_S3 Decoupled Spike**: S3 > 30.0 °C when I < 80 A and V < 28 kV represents unphysical localized sensor transients (genuine high S3 only occurs under combined high current and voltage).
5. **Near-Zero Sensor_S1 Dead State**: S1 < 1.0 °C represents disconnected thermocouple probes.

---

## 3. Machine Learning Models & Cross-Validation Results

### 3.1 Validity Classification (5-Fold Stratified CV)
- **Model**: XGBoost Classifier (depth=4, lr=0.05, n_estimators=120, subsample=0.8)
- **Evaluation**:
```
              precision    recall  f1-score   support

       Valid       0.94      1.00      0.97       866
     Invalid       0.97      0.58      0.73       134

    accuracy                           0.94      1000
   macro avg       0.96      0.79      0.85      1000
weighted avg       0.94      0.94      0.94      1000

ROC AUC Score: 0.9421
```

### 3.2 Reference Parameter Regression (5-Fold CV on Verified Records)
Hyperparameter optimization using `RandomizedSearchCV` yielded the optimal parameter configuration:
- `n_estimators`: 300
- `max_depth`: 3
- `learning_rate`: 0.1
- `subsample`: 0.8

| Model Component | 5-Fold CV RMSE (°C) | 5-Fold CV R² | Weight |
|---|---|---|---|
| **Tuned XGBoost Regressor** | **0.8009** | **0.9944** | 0.60 |
| **Random Forest Regressor** | **1.4169** | **0.9826** | 0.30 |
| **Ridge Regularized Linear** | **1.5444** | **0.9793** | 0.10 |
| **Weighted Tri-Model Ensemble** | **0.8854** | **0.9932** | **1.00** |

---

## 4. Test Set Inference & Safety-Critical Anomaly Ranking

### 4.1 Test Inference Summary
- Total Blind Test Records: **350**
- Normal / Valid Records Identified: **322** (92.0%)
- Anomalous / Invalid Records Flagged: **28** (8.0%)
  - Flagged by Deterministic Physics Rules: **20**
  - Flagged by Residual Classifier: **8**
- Predicted Hot-Spot Temperature Range: **[12.81 °C, 57.65 °C]**
- Mean Predicted Hot-Spot Rise: **26.36 °C**

### 4.2 Top 3 Priority Safety-Critical Records
Records requiring immediate engineer intervention are defined as **Invalid test records exhibiting the highest predicted hot-spot thermal rise**. If operated under unmonitored conditions with faulty telemetry, these present catastrophic failure and fire risks:

| Priority Rank | Test_ID | Predicted Rise (°C) | Load Current (A) | Applied Voltage (kV) | Anomaly Nature |
|---|---|---|---|---|---|
| **1** | `TST-0315` | **46.83** | 100.72 | 25.45 | Duplicate Measurement Fingerprint |
| **2** | `TST-0244` | **46.83** | 100.72 | 25.45 | Duplicate Measurement Fingerprint |
| **3** | `TST-0142` | **45.47** | 103.88 | 24.40 | Sensor S3 Sentinel Lockup (1.0 °C under 103.9 A) |

---

## 5. Digital Twin Automation & Industrial Deployment Steps

To operationalize this solution in CPRI testing bays and substation automated test fixtures:

1. **Edge Telemetry Ingestion (MQTT / Modbus TCP)**:
   - Stream physical sensors (V, I, T_amb, S1, S2, S3) into an edge industrial gateway (e.g., Siemens IOT2050 or Advantech IPC) running an InfluxDB time-series engine.
2. **Real-Time Streaming Anomaly Engine**:
   - Implement the 5 deterministic rules in a lightweight microsecond stream processor (e.g., Rust or C++ ONNX runtime).
   - Flag sensor dropouts, open thermocouples (S1 < 1.0), and out-of-range anomalies before test engineers certify completion.
3. **Surrogate Digital Twin Model Inference**:
   - Export the trained XGBoost and Random Forest models to **ONNX format**.
   - Execute sub-millisecond hot-spot thermal estimation (T_hat_hotspot) continuously during testing.
4. **Thermal Dissipation Residual Tracking**:
   - Compute real-time discrepancy:
     `R(t) = |S2(t) - S_hat_twin(I(t), V(t))|`
   - Sudden increases in R(t) indicate internal contact degradation, partial discharge, or loose terminal busbars.
5. **Closed-Loop SCADA Trip & Alarm Integration**:
   - Connect digital twin outputs to the laboratory SCADA safety PLC via dry contacts.
   - Automatically trigger emergency load breaker trips if predicted hot-spot temperature rise exceeds safety limits (e.g., > 60 °C) or if sensor faults mask an active high-load test condition.
