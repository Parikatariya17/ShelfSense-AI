"""Stockout-aware demand estimation and controlled recovery experiment.

True_Demand is used only after prediction, to score the synthetic demo.
It is never passed into the RandomForest as a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.preprocessing import FEATURE_COLS, HOLDOUT_DAYS, model_matrix


def apply_estimated_demand(
    df: pd.DataFrame,
    model,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Predict demand wherever lag features exist; recover hidden demand on stockouts."""
    cols = feature_cols or FEATURE_COLS
    out = df.copy()
    valid = out[cols].notna().all(axis=1)
    preds = np.full(len(out), np.nan)
    if valid.any():
        preds[valid.to_numpy()] = np.clip(
            model.predict(model_matrix(out.loc[valid], cols)), 0, None
        )
    out["Predicted_Demand"] = preds

    # On in-stock days, recorded sales equal demand. On stockouts, use the model.
    estimated = np.where(
        out["Stockout_Flag"] == 1,
        out["Predicted_Demand"],
        out["Observed_Sales"].astype(float),
    )
    # If a stockout row lacks lags (early history), fall back to observed sales.
    estimated = np.where(np.isnan(estimated), out["Observed_Sales"].astype(float), estimated)
    out["Estimated_Demand"] = np.maximum(estimated, 0)
    out["Hidden_Demand"] = np.where(
        out["Stockout_Flag"] == 1,
        np.maximum(0, out["Estimated_Demand"] - out["Observed_Sales"]),
        0.0,
    )
    return out


def artificial_stockout_experiment(
    df: pd.DataFrame,
    model,
    feature_cols: list[str] | None = None,
    cap: float = 0.50,
) -> dict:
    """Cap in-stock holdout days, hide demand from the model, score vs known demand.

    Features already exclude same-day sales, so the model never sees the hidden
    quantity. We compare predictions with the original True_Demand / Observed_Sales
    (they match on these non-stockout rows).
    """
    cols = feature_cols or FEATURE_COLS
    cutoff = df["Date"].max() - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    holdout = df[
        (df["Date"] >= cutoff)
        & (df["Stockout_Flag"] == 0)
        & df[cols].notna().all(axis=1)
        & (df["True_Demand"] > 0)
    ].copy()
    if holdout.empty:
        return {"n": 0, "note": "No eligible holdout rows for the experiment."}

    known_demand = holdout["True_Demand"].astype(float)
    predicted = np.clip(model.predict(model_matrix(holdout, cols)), 0, None)
    simulated_stock = np.maximum(1, np.floor(known_demand * cap))
    simulated_sales = np.minimum(known_demand, simulated_stock)
    hidden = np.maximum(0, predicted - simulated_sales)

    return {
        "n": int(len(holdout)),
        "stock_cap_ratio": cap,
        "mae_predicted_vs_known_demand": float(mean_absolute_error(known_demand, predicted)),
        "rmse_predicted_vs_known_demand": float(
            np.sqrt(mean_squared_error(known_demand, predicted))
        ),
        "mean_known_demand": float(known_demand.mean()),
        "mean_simulated_sales": float(simulated_sales.mean()),
        "mean_predicted_demand": float(np.mean(predicted)),
        "mean_estimated_hidden_demand": float(np.mean(hidden)),
        "note": (
            "Known demand was hidden from the model. Features use only historical "
            "sales. True demand is scored after the fact because this is synthetic data."
        ),
    }


def stockout_frequency(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Store_ID", "Product_ID", "Product_Name"], as_index=False)
        .agg(
            days=("Date", "nunique"),
            stockout_days=("Stockout_Flag", "sum"),
            observed_sales=("Observed_Sales", "sum"),
            hidden_demand=("Hidden_Demand", "sum"),
            estimated_demand=("Estimated_Demand", "sum"),
        )
    )
    summary["stockout_rate"] = summary["stockout_days"] / summary["days"]
    return summary.sort_values("stockout_rate", ascending=False)
