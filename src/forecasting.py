"""Random forest demand model, evaluation, and 7-day recursive forecast.

The model is trained only on historical non-stockout rows before the
chronological holdout. True_Demand is never a feature or training target.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.preprocessing import (
    FEATURE_COLS,
    HOLDOUT_DAYS,
    ensure_dataset,
    model_matrix,
    prepare_frame,
)
from src.stockout_analysis import apply_estimated_demand, artificial_stockout_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.joblib"


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_demand_model(df: pd.DataFrame, feature_cols: list[str]) -> RandomForestRegressor:
    train_end = df["Date"].max() - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    # Non-stockout observations strictly before the untouched holdout.
    train_mask = (
        (df["Date"] < train_end)
        & (df["Stockout_Flag"] == 0)
        & df[feature_cols].notna().all(axis=1)
    )
    train = df.loc[train_mask]
    if train.empty:
        raise RuntimeError("No valid non-stockout training rows.")

    x = model_matrix(train, feature_cols)
    y = train["Observed_Sales"].astype(float)
    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def one_step_metrics(df: pd.DataFrame, model, feature_cols: list[str]) -> dict:
    """Holdout MAE/RMSE vs observed sales and vs synthetic true demand."""
    cutoff = df["Date"].max() - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    test = df[(df["Date"] >= cutoff) & df[feature_cols].notna().all(axis=1)].copy()
    if test.empty:
        return {}
    pred = np.clip(model.predict(model_matrix(test, feature_cols)), 0, None)
    test = test.copy()
    test["Predicted_Demand"] = pred

    naive = test["Sales_Lag_7"].fillna(test["Sales_Lag_1"])
    rf_obs = (
        mean_absolute_error(test["Observed_Sales"], pred),
        _rmse(test["Observed_Sales"], pred),
    )
    naive_obs = (
        mean_absolute_error(test["Observed_Sales"], naive),
        _rmse(test["Observed_Sales"], naive),
    )
    rf_true = (
        mean_absolute_error(test["True_Demand"], pred),
        _rmse(test["True_Demand"], pred),
    )
    naive_true = (
        mean_absolute_error(test["True_Demand"], naive),
        _rmse(test["True_Demand"], naive),
    )
    stockout_test = test[test["Stockout_Flag"] == 1]
    recovery = {}
    if len(stockout_test) > 0:
        recovery = {
            "stockout_holdout_n": int(len(stockout_test)),
            "recovery_mae_vs_true_demand": float(
                mean_absolute_error(stockout_test["True_Demand"], stockout_test["Predicted_Demand"])
            ),
            "recovery_rmse_vs_true_demand": _rmse(
                stockout_test["True_Demand"], stockout_test["Predicted_Demand"]
            ),
        }
    return {
        "holdout_n": int(len(test)),
        "rf_mae_vs_observed": float(rf_obs[0]),
        "rf_rmse_vs_observed": float(rf_obs[1]),
        "naive_mae_vs_observed": float(naive_obs[0]),
        "naive_rmse_vs_observed": float(naive_obs[1]),
        "rf_mae_vs_true_demand": float(rf_true[0]),
        "rf_rmse_vs_true_demand": float(rf_true[1]),
        "naive_mae_vs_true_demand": float(naive_true[0]),
        "naive_rmse_vs_true_demand": float(naive_true[1]),
        **recovery,
        "note": (
            "True_Demand is available only in this synthetic dataset and is used "
            "for evaluation, never as a model input. In real stockouts, true demand "
            "is unobserved."
        ),
    }


def forecast_next_7(
    history: pd.DataFrame,
    model,
    feature_cols: list[str],
    store_id: str,
    product_id: str,
) -> pd.DataFrame:
    """Recursive 7-day forecast using predicted values as subsequent lags."""
    sku = history[(history["Store_ID"] == store_id) & (history["Product_ID"] == product_id)].copy()
    sku = sku.sort_values("Date")
    if sku.empty:
        return pd.DataFrame()

    last = sku.iloc[-1]
    sales_hist = sku["Estimated_Demand"].fillna(sku["Observed_Sales"]).tolist()
    future_rows = []
    last_date = pd.to_datetime(last["Date"])

    for step in range(1, 8):
        next_date = last_date + timedelta(days=step)
        dow = next_date.weekday()
        lag1 = sales_hist[-1]
        lag7 = sales_hist[-7] if len(sales_hist) >= 7 else sales_hist[0]
        roll = float(np.mean(sales_hist[-7:]))
        # Promotion is unknown in the future; default to no promo (conservative).
        row = {
            "Store_Code": int(last["Store_Code"]),
            "Product_Code": int(last["Product_Code"]),
            "Day_of_Week": int(dow),
            "Month": int(next_date.month),
            "Is_Weekend": int(dow >= 5),
            "Promotion": 0,
            "Sales_Lag_1": lag1,
            "Sales_Lag_7": lag7,
            "Sales_Roll_7": roll,
        }
        pred = float(np.clip(model.predict(pd.DataFrame([row])[feature_cols])[0], 0, None))
        pred = round(pred, 2)
        sales_hist.append(pred)
        future_rows.append(
            {
                "Date": next_date,
                "Store_ID": store_id,
                "Product_ID": product_id,
                "Forecast_Demand": pred,
                "Baseline_Naive": round(float(lag7), 2),
            }
        )
    return pd.DataFrame(future_rows)


def train_and_save(path: Path | None = None, model_path: Path | None = None) -> dict:
    ensure_dataset(path)
    framed, meta = prepare_frame(path)
    feature_cols = meta["feature_cols"]
    model = train_demand_model(framed, feature_cols)
    framed = apply_estimated_demand(framed, model, feature_cols)
    metrics = one_step_metrics(framed, model, feature_cols)
    experiment = artificial_stockout_experiment(framed, model, feature_cols)

    model_path = model_path or MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "store_map": meta["store_map"],
        "product_map": meta["product_map"],
        "metrics": metrics,
        "experiment": experiment,
        "train_end": meta["train_end"],
        "test_start": meta["test_start"],
    }
    joblib.dump(artifact, model_path)
    return {"artifact": artifact, "frame": framed, "path": str(model_path)}


def load_artifact(model_path: Path | None = None) -> dict:
    model_path = model_path or MODEL_PATH
    if not model_path.exists():
        result = train_and_save(model_path=model_path)
        return result["artifact"]
    return joblib.load(model_path)


if __name__ == "__main__":
    result = train_and_save()
    print(f"Saved model to {result['path']}")
    print("Holdout metrics:")
    for k, v in result["artifact"]["metrics"].items():
        print(f"  {k}: {v}")
    print("Artificial stockout experiment:")
    for k, v in result["artifact"]["experiment"].items():
        print(f"  {k}: {v}")
