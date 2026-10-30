"""Load, clean, and engineer leakage-safe features.

Forbidden as demand-model inputs:
    True_Demand, same-day Observed_Sales, Available_Stock, Stockout_Flag,
    Closing_Stock, Restock_Units, and any same-day future information.

Lag and rolling features use historical observations only:
    shift(1) is applied before any rolling average.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_generator import DATA_PATH, save_dataset

HOLDOUT_DAYS = 28
FORBIDDEN_MODEL_FEATURES = {
    "True_Demand",
    "Observed_Sales",
    "Available_Stock",
    "Stockout_Flag",
    "Closing_Stock",
    "Restock_Units",
}

FEATURE_COLS = [
    "Store_Code",
    "Product_Code",
    "Day_of_Week",
    "Month",
    "Is_Weekend",
    "Promotion",
    "Sales_Lag_1",
    "Sales_Lag_7",
    "Sales_Roll_7",
]


def ensure_dataset(path: Path | None = None) -> Path:
    path = path or DATA_PATH
    if not path.exists():
        save_dataset(path)
    return path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = ensure_dataset(path)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    numeric_cols = [
        "True_Demand",
        "Available_Stock",
        "Observed_Sales",
        "Closing_Stock",
        "Restock_Units",
        "Stockout_Flag",
        "Promotion",
        "Day_of_Week",
        "Demo_Unit_Price",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)
    df["Product_Name"] = df["Product_Name"].fillna("Unknown")
    df["Store_ID"] = df["Store_ID"].fillna("Unknown")
    return df.sort_values(["Store_ID", "Product_ID", "Date"]).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Month"] = out["Date"].dt.month.astype(int)
    out["Is_Weekend"] = (out["Day_of_Week"] >= 5).astype(int)
    out["Promotion"] = out["Promotion"].astype(int)
    return out


def add_identity_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Stable integer codes for store and product (identity, not a target leak)."""
    out = df.copy()
    store_ids = sorted(out["Store_ID"].unique())
    product_ids = sorted(out["Product_ID"].unique())
    store_map = {sid: i for i, sid in enumerate(store_ids)}
    product_map = {pid: i for i, pid in enumerate(product_ids)}
    out["Store_Code"] = out["Store_ID"].map(store_map).astype(int)
    out["Product_Code"] = out["Product_ID"].map(product_map).astype(int)
    return out, store_map, product_map


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only lags. Rolling mean is computed on already-shifted sales."""
    out = df.copy()
    grouped = out.groupby(["Store_ID", "Product_ID"], group_keys=False)

    out["Sales_Lag_1"] = grouped["Observed_Sales"].shift(1)
    out["Sales_Lag_7"] = grouped["Observed_Sales"].shift(7)
    # shift(1) first so the rolling window never includes same-day sales.
    out["Sales_Roll_7"] = grouped["Observed_Sales"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
    )
    return out


def chronological_split(df: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = df["Date"].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days - 1)
    train = df[df["Date"] < cutoff].copy()
    test = df[df["Date"] >= cutoff].copy()
    return train, test


def prepare_frame(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    raw = load_raw(path)
    framed = add_calendar_features(raw)
    framed, store_map, product_map = add_identity_codes(framed)
    framed = add_lag_features(framed)
    train, test = chronological_split(framed)
    meta = {
        "store_map": store_map,
        "product_map": product_map,
        "feature_cols": list(FEATURE_COLS),
        "train_end": train["Date"].max() if len(train) else None,
        "test_start": test["Date"].min() if len(test) else None,
        "forbidden": sorted(FORBIDDEN_MODEL_FEATURES),
    }
    return framed, meta


def model_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    cols = feature_cols or FEATURE_COLS
    overlap = set(cols) & FORBIDDEN_MODEL_FEATURES
    if overlap:
        raise ValueError(f"Forbidden columns in model features: {overlap}")
    return df[cols].copy()
