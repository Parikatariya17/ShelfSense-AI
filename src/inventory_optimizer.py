"""Rule-based stockout risk and replenishment engine.

This is not a trained classifier. Current inventory is the latest Closing_Stock
(opening inventory of the next day before any new sales).
"""

from __future__ import annotations

import math

import pandas as pd

from src.data_generator import DEMO_PRICES, PRODUCTS

Z_SERVICE = 1.65  # ~95% service level in a normal-demand safety-stock formula
DEFAULT_LEAD_TIME = 3
DEFAULT_BUDGET = 2500.0


def classify_risk(days_remaining: float, lead_time: int) -> tuple[str, str]:
    """Transparent thresholds relative to supplier lead time."""
    if days_remaining <= 1:
        return (
            "Critical Risk",
            f"Only {days_remaining:.1f} day(s) of stock left — likely to stock out before replenishment.",
        )
    if days_remaining <= lead_time:
        return (
            "High Risk",
            f"{days_remaining:.1f} day(s) of stock is at or below the {lead_time}-day lead time.",
        )
    if days_remaining <= lead_time + 3:
        return (
            "Medium Risk",
            f"{days_remaining:.1f} day(s) of stock covers lead time but little buffer beyond it.",
        )
    return (
        "Low Risk",
        f"{days_remaining:.1f} day(s) of stock exceeds lead time plus a 3-day buffer.",
    )


def replenishment_math(
    avg_daily_demand: float,
    daily_std: float,
    lead_time: int,
    current_inventory: int,
) -> dict:
    """Safety stock, reorder point, target stock, and recommended order quantity."""
    avg_daily_demand = max(0.0, float(avg_daily_demand))
    daily_std = max(0.0, float(daily_std))
    lead_time = max(0, int(lead_time))
    current_inventory = max(0, int(round(current_inventory)))

    # Safety Stock = z * sigma_daily * sqrt(lead time)
    safety = Z_SERVICE * daily_std * math.sqrt(lead_time) if lead_time > 0 else 0.0
    safety_stock = int(round(safety))
    reorder_point = int(round(avg_daily_demand * lead_time + safety_stock))
    # Cover lead time plus a 7-day review/cycle horizon, plus safety stock.
    target_stock = int(round(avg_daily_demand * (lead_time + 7) + safety_stock))
    recommended_qty = max(0, target_stock - current_inventory)
    days_remaining = current_inventory / max(avg_daily_demand, 1e-6)
    risk, reason = classify_risk(days_remaining, lead_time)

    return {
        "avg_daily_demand": round(avg_daily_demand, 2),
        "daily_std": round(daily_std, 2),
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "target_stock": target_stock,
        "recommended_qty": int(recommended_qty),
        "days_remaining": round(days_remaining, 2),
        "risk": risk,
        "reason": reason,
        "current_inventory": current_inventory,
        "lead_time": lead_time,
    }


def sku_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.sort_values("Date").groupby(["Store_ID", "Product_ID"], as_index=False).tail(1)
    return latest[
        [
            "Date",
            "Store_ID",
            "Store_Name",
            "Product_ID",
            "Product_Name",
            "Closing_Stock",
            "Available_Stock",
            "Demo_Unit_Price",
        ]
    ].copy()


def build_recommendations(
    df: pd.DataFrame,
    forecasts: dict[tuple[str, str], pd.DataFrame],
    lead_time: int = DEFAULT_LEAD_TIME,
    inventory_overrides: dict[tuple[str, str], int] | None = None,
    budget: float | None = None,
) -> pd.DataFrame:
    """One recommendation row per store-product using latest Closing_Stock."""
    snap = sku_latest_snapshot(df)
    rows = []
    for _, sku in snap.iterrows():
        key = (sku["Store_ID"], sku["Product_ID"])
        fc = forecasts.get(key)
        hist = df[(df["Store_ID"] == sku["Store_ID"]) & (df["Product_ID"] == sku["Product_ID"])]
        hist_demand = hist["Estimated_Demand"].dropna()
        if fc is not None and not fc.empty:
            avg = float(fc["Forecast_Demand"].mean())
            std = float(fc["Forecast_Demand"].std(ddof=0)) if len(fc) > 1 else float(hist_demand.std(ddof=0) or 0)
        else:
            avg = float(hist_demand.tail(28).mean() or 0)
            std = float(hist_demand.tail(28).std(ddof=0) or 0)

        current = int(sku["Closing_Stock"])
        if inventory_overrides and key in inventory_overrides:
            current = int(inventory_overrides[key])

        calc = replenishment_math(avg, std, lead_time, current)
        price = float(sku.get("Demo_Unit_Price") or DEMO_PRICES.get(sku["Product_ID"], 0))
        cost = round(calc["recommended_qty"] * price, 2)
        rows.append(
            {
                **calc,
                "Store_ID": sku["Store_ID"],
                "Store_Name": sku["Store_Name"],
                "Product_ID": sku["Product_ID"],
                "Product_Name": sku["Product_Name"],
                "demo_unit_price": price,
                "estimated_purchase_cost": cost,
                "as_of_date": sku["Date"],
            }
        )

    recs = pd.DataFrame(rows)
    if recs.empty:
        return recs

    recs["budget_note"] = ""
    if budget is not None and budget >= 0:
        recs = recs.sort_values(["risk", "days_remaining"], key=_risk_sort)
        remaining = float(budget)
        adjusted_qty = []
        adjusted_cost = []
        notes = []
        for _, row in recs.iterrows():
            price = row["demo_unit_price"]
            qty = int(row["recommended_qty"])
            cost = qty * price
            if cost <= remaining:
                remaining -= cost
                adjusted_qty.append(qty)
                adjusted_cost.append(round(cost, 2))
                notes.append("Within demo purchasing budget.")
            else:
                affordable = int(remaining // price) if price > 0 else 0
                remaining -= affordable * price
                adjusted_qty.append(affordable)
                adjusted_cost.append(round(affordable * price, 2))
                if qty > 0:
                    notes.append(
                        f"Budget capped this line from {qty} to {affordable} units (demo prices)."
                    )
                else:
                    notes.append("No order required.")
        recs["recommended_qty"] = adjusted_qty
        recs["estimated_purchase_cost"] = adjusted_cost
        recs["budget_note"] = notes

    recs["explanation"] = recs.apply(_explain_row, axis=1)
    return recs.reset_index(drop=True)


def _risk_sort(series: pd.Series) -> pd.Series:
    if series.name != "risk":
        return series
    order = {"Critical Risk": 0, "High Risk": 1, "Medium Risk": 2, "Low Risk": 3}
    return series.map(order)


def _explain_row(row: pd.Series) -> str:
    return (
        f"Current inventory is closing stock ({int(row['current_inventory'])} units). "
        f"Average predicted daily demand is {row['avg_daily_demand']}. "
        f"Safety stock {int(row['safety_stock'])} uses z={Z_SERVICE} × demand std "
        f"× sqrt(lead time={int(row['lead_time'])}). "
        f"Reorder point is {int(row['reorder_point'])}; target cover is lead time plus 7 days. "
        f"Recommended order is {int(row['recommended_qty'])} units. {row['reason']}"
    )


def product_price_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Product_ID": p["Product_ID"],
                "Product_Name": p["Product_Name"],
                "Demo_Unit_Price": p["price"],
            }
            for p in PRODUCTS
        ]
    )
