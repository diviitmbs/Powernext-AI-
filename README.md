# PowerNext-AI: CPRI State-Level Hackathon Screening Solution

End-to-end Machine Learning pipeline for electrical apparatus thermal telemetry classification and hot-spot temperature rise prediction.

---

## 📌 Problem Overview
In electrical laboratory testing conducted by CPRI, apparatus undergoes high-voltage and high-current stress tests. Internal sensors and operating conditions are logged to determine apparatus safety, stability, and temperature rise.

This pipeline performs two simultaneous tasks on blind test telemetry:
1. **Validity Classification (`Validity_Label`)**: Discriminate between physically genuine tests (`Valid`) and corrupted/faulty telemetry (`Invalid`) using deterministic physics screening rules and an XGBoost residual classifier.
2. **Hot-Spot Thermal Rise Prediction (`Predicted_Reference_Parameter`)**: Accurately predict the critical verified reference hot-spot temperature rise (°C above ambient) using a weighted tri-model ensemble trained strictly on verified valid records.

---

## 📁 Project Directory Structure

```text
Powernext-AI-/
├── data/
│   └── CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx  # Dataset workbook
├── deliverables/
│   ├── PowerNext_AI.csv                                   # Submission CSV
│   ├── summary.json                                       # Metric summary JSON
│   └── methodology_note.md                                # Detailed methodology note
├── src/
│   ├── __init__.py
│   ├── anomaly_detection.py                               # 5 domain rules + XGBoost classifier
│   ├── config.py                                          # Constants, thresholds, and paths
│   ├── data_loader.py                                     # Dataset ingestion & median imputation
│   ├── feature_engineering.py                             # Joule heating & power dissipation features
│   ├── pipeline.py                                        # Orchestrator running end-to-end workflow
│   └── regressor.py                                       # Tuned XGBoost, RF, Ridge ensemble
├── main.py                                                # CLI execution entry point
├── requirements.txt                                       # Package dependencies
└── README.md                                              # Project documentation
```

---

## ⚡ Quickstart

### 1. Environment Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Run the complete end-to-end pipeline with default parameters:
```bash
python main.py
```

### 3. Custom Execution Options
You can configure team name, dataset path, and target output directory via CLI arguments:
```bash
python main.py \
  --team-name "PowerNext_AI" \
  --data-path "data/CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx" \
  --output-dir "deliverables"
```

To skip the RandomizedSearchCV hyperparameter search for faster execution:
```bash
python main.py --skip-tuning
```

---

## 🔬 Core Methodology & Architecture

### 1. Preprocessing & Imputation
- Missing values in `Sensor_S1`, `Sensor_S2`, and `Sensor_S3` are imputed using the **median of verified valid training records**.
- `Sensor_S4` has near-zero correlation ($r = -0.009$) and is explicitly **dropped** to prevent noise injection.

### 2. Deterministic Anomaly Screening Rules
Applied prior to ML modeling with 100% precision on historical data:
1. **Sentinel & Negative Values**: $S_1, S_2, S_3 \in \{0.0, 1.0, 25.0\}$ or $< 0$.
2. **Duplicate Telemetry Fingerprints**: Duplicate combinations of $(V, I, T_{amb}, t_{dur}, S_1, S_2, S_3)$ rounded to 3 decimal places.
3. **Sensor S2 Upper Threshold**: $S_2 > 21.6$ °C.
4. **Sensor S3 Decoupled Spike**: $S_3 > 30.0$ °C when $I < 80$ A and $V < 28$ kV.
5. **Near-Zero Sensor S1**: $S_1 < 1.0$ °C (dead sensor probe).

### 3. Validity Classification
- **Model**: `XGBoostClassifier` with 5-fold Stratified Cross-Validation.
- **Ensemble**: If any deterministic rule fires $\rightarrow$ `Invalid`. Otherwise, classifier probability threshold $> 0.5$.
- **CV Performance**: ROC AUC: `0.9421`, Valid F1: `0.97`, Invalid F1: `0.73`.

### 4. Hot-Spot Temperature Regression Ensemble
- **Training Subset**: Exclusively verified `Valid` records ($N=866$) to guarantee zero contamination from sensor faults.
- **Engineered Features**:
  - Joule heating dissipation proxy: $I^2$ (`Load_Current_A` squared)
  - Power dissipation proxy: $I^2 \cdot V$
  - Terminal sensor mean: $(S_1 + S_2) / 2$
  - Sensor ensemble mean: $(S_1 + S_2 + S_3) / 3$
- **Ensemble Composition**:
  - `60% Tuned XGBoost Regressor` (5-Fold CV RMSE: **0.8009 °C**, $R^2$: **0.9944**)
  - `30% Random Forest Regressor` (5-Fold CV RMSE: **1.4169 °C**, $R^2$: **0.9826**)
  - `10% Ridge Regularized Linear` (5-Fold CV RMSE: **1.5444 °C**, $R^2$: **0.9793**)
  - **Ensemble Result**: 5-Fold CV RMSE: **0.8854 °C**, $R^2$: **0.9932**
- **Physical Post-Processing**: Predictions clipped to $[10.0, 65.0]$ °C and rounded to 2 decimal places.

---

## 🏆 Deliverables Generated

1. **`deliverables/PowerNext_AI.csv`**: Contains exact test submission schema (`Test_ID`, `Predicted_Reference_Parameter`, `Validity_Label`).
2. **`deliverables/summary.json`**: Machine-readable JSON summarizing test record counts, abnormal record totals, min/max/average predicted temperatures, top 3 attention IDs, and validation metrics.
3. **`deliverables/methodology_note.md`**: Formal methodology report covering physics insights, validation benchmarks, safety priority ranking, and digital twin implementation steps.
