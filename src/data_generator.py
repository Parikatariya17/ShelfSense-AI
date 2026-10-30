"""Synthetic retail dataset generator for ShelfSense AI.

Inventory accounting (per store-product, chronological):
    Available_Stock      = opening inventory BEFORE sales
    Observed_Sales       = min(True_Demand, Available_Stock)
    Closing_Stock        = Available_Stock - Observed_Sales
    Next-day opening     = previous Closing_Stock + Restock_Units

Restock_Units is the overnight delivery booked after close; it is added
to the next day's opening inventory, not to the same-day shelf.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "demo_retail_data.csv"
SEED = 42
N_DAYS = 180
END_DATE = date(2026, 10, 30)

STORES = [
    {"Store_ID": "S01", "Store_Name": "Downtown", "scale": 1.30},
    {"Store_ID": "S02", "Store_Name": "Suburb", "scale": 1.00},
    {"Store_ID": "S03", "Store_Name": "Campus", "scale": 0.85},
]

# Distinct demand shapes: staple, weekend, seasonal, promo-driven, slow/lumpy.
PRODUCTS = [
    {
        "Product_ID": "P01",
        "Product_Name": "Fresh Milk 1L",
        "base": 42.0,
        "weekend_lift": 0.92,
        "promo_lift": 1.12,
        "noise": 0.08,
        "price": 2.40,
    },
    {
        "Product_ID": "P02",
        "Product_Name": "Whole Wheat Bread",
        "base": 30.0,
        "weekend_lift": 1.38,
        "promo_lift": 1.20,
        "noise": 0.12,
        "price": 3.10,
    },
    {
        "Product_ID": "P03",
        "Product_Name": "Ice Cream 500ml",
        "base": 20.0,
        "weekend_lift": 1.25,
        "promo_lift": 1.35,
        "noise": 0.18,
        "price": 4.80,
    },
    {
        "Product_ID": "P04",
        "Product_Name": "Instant Noodles 5pk",
        "base": 24.0,
        "weekend_lift": 1.10,
        "promo_lift": 1.65,
        "noise": 0.14,
        "price": 3.60,
    },
    {
        "Product_ID": "P05",
        "Product_Name": "Basmati Rice 5kg",
        "base": 9.0,
        "weekend_lift": 1.05,
        "promo_lift": 1.22,
        "noise": 0.28,
        "price": 12.50,
    },
]

DEMO_PRICES = {p["Product_ID"]: p["price"] for p in PRODUCTS}


def _month_seasonality(product_id: str, month: int) -> float:
    """Mild calendar seasonality; ice cream is strongly summer-peaked."""
    if product_id == "P03":
        summer = {5: 1.35, 6: 1.50, 7: 1.55, 8: 1.40, 9: 1.10, 10: 0.85}
        return summer.get(month, 1.0)
    if product_id == "P01":
        return 1.04 if month in (5, 6, 7) else 1.0
    if product_id == "P04":
        return 1.08 if month in (9, 10) else 1.0
    return 1.0


def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = END_DATE - timedelta(days=N_DAYS - 1)
    dates = [start + timedelta(days=i) for i in range(N_DAYS)]
    rows: list[dict] = []

    for store in STORES:
        restock_offset = {"S01": 1, "S02": 3, "S03": 5}[store["Store_ID"]]
        for product in PRODUCTS:
            opening = int(round(product["base"] * store["scale"] * rng.uniform(3.5, 6.5)))
            recent_demand: list[float] = []

            for i, d in enumerate(dates):
                dow = d.weekday()  # Monday=0
                is_weekend = dow >= 5
                weekend_factor = (
                    product["weekend_lift"] if is_weekend else (1.06 if dow == 4 else 0.97)
                )

                promo_p = 0.18 if product["Product_ID"] == "P04" else 0.10
                promotion = int(rng.random() < promo_p)
                promo_factor = product["promo_lift"] if promotion else 1.0
                season = _month_seasonality(product["Product_ID"], d.month)

                noise = rng.lognormal(mean=0.0, sigma=product["noise"])
                true_demand = int(
                    max(
                        0,
                        round(
                            product["base"]
                            * store["scale"]
                            * weekend_factor
                            * promo_factor
                            * season
                            * noise
                        ),
                    )
                )
                recent_demand.append(true_demand)
                if len(recent_demand) > 14:
                    recent_demand.pop(0)

                available = max(0, int(opening))
                observed = min(true_demand, available)
                closing = available - observed
                stockout = int(true_demand > available)

                # Overnight replenishment booked after close.
                avg_d = float(np.mean(recent_demand)) if recent_demand else product["base"]
                scheduled = (i % 7) == restock_offset
                emergency = closing <= max(1, int(0.15 * avg_d)) and i > 10
                restock = 0
                if scheduled or emergency:
                    cover_days = 8 if scheduled else 5
                    target_opening = int(round(avg_d * cover_days))
                    restock = max(0, target_opening - closing)
                    # Occasional under-ordering creates realistic stockouts.
                    if rng.random() < 0.22:
                        restock = int(restock * rng.uniform(0.30, 0.65))
                    if product["Product_ID"] == "P05" and rng.random() < 0.15:
                        restock = int(restock * 0.5)

                rows.append(
                    {
                        "Date": d.isoformat(),
                        "Store_ID": store["Store_ID"],
                        "Store_Name": store["Store_Name"],
                        "Product_ID": product["Product_ID"],
                        "Product_Name": product["Product_Name"],
                        "Day_of_Week": dow,
                        "Promotion": promotion,
                        "True_Demand": true_demand,
                        "Available_Stock": available,
                        "Observed_Sales": observed,
                        "Closing_Stock": closing,
                        "Restock_Units": int(restock),
                        "Stockout_Flag": stockout,
                        "Demo_Unit_Price": product["price"],
                    }
                )
                opening = closing + restock

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Store_ID", "Product_ID", "Date"]).reset_index(drop=True)


def save_dataset(path: Path | None = None, seed: int = SEED) -> Path:
    path = path or DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(seed=seed)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = save_dataset()
    df = pd.read_csv(out)
    sales_ok = (df["Observed_Sales"] == np.minimum(df["True_Demand"], df["Available_Stock"])).all()
    close_ok = (df["Closing_Stock"] == df["Available_Stock"] - df["Observed_Sales"]).all()
    print(f"Wrote {out} ({len(df)} rows)")
    print(f"Stockout rate: {df['Stockout_Flag'].mean():.1%}")
    print(f"Restock days: {(df['Restock_Units'] > 0).mean():.1%}")
    print(f"Sales identity OK: {bool(sales_ok)} | Closing identity OK: {bool(close_ok)}")
