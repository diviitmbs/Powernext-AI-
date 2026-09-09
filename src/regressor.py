"""
Regression and ensembling module for Reference_Parameter prediction.
"""
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import ENSEMBLE_WEIGHTS, PHYSICAL_BOUNDS


def tune_xgboost_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 15,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[XGBRegressor, dict]:
    """
    Tune XGBoostRegressor hyperparameters using RandomizedSearchCV over specified ranges:
    n_estimators (100–500), max_depth (3–6), learning_rate (0.01–0.1), subsample (0.7–1.0).
    """
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }

    base_model = XGBRegressor(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    best_params = search.best_params_
    best_estimator = search.best_estimator_

    print(f"[Regressor] Best XGBoost Hyperparameters: {best_params}")
    return best_estimator, best_params


def train_regressor_ensemble(
    X_train_valid: pd.DataFrame,
    y_train_valid: pd.Series,
    tuned_xgb: XGBRegressor,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Trains XGBoost, Random Forest, and Ridge models on valid training records.
    Evaluates 5-fold CV RMSE and R² for each individual model and the weighted ensemble.
    Fits all final models on the entire valid training set.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Instantiate models
    xgb_model = tuned_xgb
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=random_state, n_jobs=-1)
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=random_state))
    ])

    # Out-of-fold Cross Validation Predictions
    print("[Regressor] Running 5-fold Cross Validation on valid records...")
    xgb_oof = cross_val_predict(xgb_model, X_train_valid, y_train_valid, cv=kf, n_jobs=-1)
    rf_oof = cross_val_predict(rf_model, X_train_valid, y_train_valid, cv=kf, n_jobs=-1)
    ridge_oof = cross_val_predict(ridge_pipeline, X_train_valid, y_train_valid, cv=kf, n_jobs=-1)

    w_xgb = ENSEMBLE_WEIGHTS["xgboost"]
    w_rf = ENSEMBLE_WEIGHTS["rf"]
    w_ridge = ENSEMBLE_WEIGHTS["ridge"]

    ensemble_oof = (w_xgb * xgb_oof) + (w_rf * rf_oof) + (w_ridge * ridge_oof)

    metrics = {
        "xgb_rmse": float(root_mean_squared_error(y_train_valid, xgb_oof)),
        "xgb_r2": float(r2_score(y_train_valid, xgb_oof)),
        "rf_rmse": float(root_mean_squared_error(y_train_valid, rf_oof)),
        "rf_r2": float(r2_score(y_train_valid, rf_oof)),
        "ridge_rmse": float(root_mean_squared_error(y_train_valid, ridge_oof)),
        "ridge_r2": float(r2_score(y_train_valid, ridge_oof)),
        "ensemble_rmse": float(root_mean_squared_error(y_train_valid, ensemble_oof)),
        "ensemble_r2": float(r2_score(y_train_valid, ensemble_oof)),
    }

    print(f"  [CV Evaluation] XGBoost:      RMSE = {metrics['xgb_rmse']:.4f} °C, R² = {metrics['xgb_r2']:.4f}")
    print(f"  [CV Evaluation] Random Forest: RMSE = {metrics['rf_rmse']:.4f} °C, R² = {metrics['rf_r2']:.4f}")
    print(f"  [CV Evaluation] Ridge:         RMSE = {metrics['ridge_rmse']:.4f} °C, R² = {metrics['ridge_r2']:.4f}")
    print(f"  [CV Evaluation] Ensemble:      RMSE = {metrics['ensemble_rmse']:.4f} °C, R² = {metrics['ensemble_r2']:.4f}")

    # Fit all models on complete valid training dataset
    xgb_model.fit(X_train_valid, y_train_valid)
    rf_model.fit(X_train_valid, y_train_valid)
    ridge_pipeline.fit(X_train_valid, y_train_valid)

    models = {
        "xgboost": xgb_model,
        "rf": rf_model,
        "ridge": ridge_pipeline,
    }
    return models, metrics


def predict_reference_parameter(
    models: Dict[str, Any],
    X: pd.DataFrame,
    clip_bounds: Tuple[float, float] = PHYSICAL_BOUNDS,
) -> pd.Series:
    """
    Generate weighted ensemble prediction: 0.6 * XGBoost + 0.3 * RF + 0.1 * Ridge.
    Clip predictions to physically valid temperature rise bounds [10.0, 65.0] °C.
    Round predictions to 2 decimal places.
    """
    pred_xgb = models["xgboost"].predict(X)
    pred_rf = models["rf"].predict(X)
    pred_ridge = models["ridge"].predict(X)

    w_xgb = ENSEMBLE_WEIGHTS["xgboost"]
    w_rf = ENSEMBLE_WEIGHTS["rf"]
    w_ridge = ENSEMBLE_WEIGHTS["ridge"]

    ensemble_pred = (w_xgb * pred_xgb) + (w_rf * pred_rf) + (w_ridge * pred_ridge)
    clipped_pred = np.clip(ensemble_pred, clip_bounds[0], clip_bounds[1])
    return pd.Series(clipped_pred.round(2), index=X.index, name="Predicted_Reference_Parameter")
