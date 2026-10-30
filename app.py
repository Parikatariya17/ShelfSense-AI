"""SHELFSENSE AI — Streamlit dashboard.

Four sections: Executive Dashboard, Demand Forecasting, Inventory Intelligence,
and Business Simulator. Models are loaded once and reused across filters.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import DATA_PATH
from src.forecasting import MODEL_PATH, forecast_next_7, load_artifact, train_and_save
from src.inventory_optimizer import (
    DEFAULT_BUDGET,
    DEFAULT_LEAD_TIME,
    build_recommendations,
    replenishment_math,
)
from src.preprocessing import prepare_frame
from src.stockout_analysis import apply_estimated_demand, stockout_frequency

st.set_page_config(
    page_title="SHELFSENSE AI",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B1220"
CARD = "#121A2B"
TEAL = "#2EC4B6"
AMBER = "#F4A261"
RED = "#E63946"
TEXT = "#E8EEF7"
MUTED = "#93A0B8"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,26,43,0.6)",
    font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(1200px 600px at 10% -10%, #152238 0%, {NAVY} 45%);
            color: {TEXT};
        }}
        [data-testid="stSidebar"] {{
            background: #0A101C;
            border-right: 1px solid #1E2A44;
        }}
        .hero-title {{
            font-size: 2.1rem; font-weight: 700; letter-spacing: 0.04em; margin-bottom: 0.15rem;
        }}
        .hero-tag {{
            color: {TEAL}; font-size: 1.05rem; font-weight: 500; margin-bottom: 0.8rem;
        }}
        .demo-banner {{
            background: #1A2740;
            border: 1px solid {TEAL}55;
            color: {TEXT};
            padding: 0.7rem 1rem;
            border-radius: 10px;
            margin-bottom: 1.1rem;
            font-size: 0.92rem;
        }}
        div[data-testid="stMetric"] {{
            background: {CARD};
            border: 1px solid #1E2A44;
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Preparing data and demand model…")
def load_pipeline():
    if (not DATA_PATH.exists()) or (not MODEL_PATH.exists()):
        result = train_and_save()
        return result["frame"], result["artifact"]
    framed, _meta = prepare_frame()
    artifact = load_artifact()
    framed = apply_estimated_demand(framed, artifact["model"], artifact["feature_cols"])
    return framed, artifact


@st.cache_data(show_spinner=False)
def cached_forecasts(_train_end, frame_hash: int) -> dict:
    """Rebuild 7-day forecasts once per trained artifact / dataset."""
    frame, artifact = load_pipeline()
    out = {}
    for store_id, product_id in frame[["Store_ID", "Product_ID"]].drop_duplicates().itertuples(index=False):
        out[(store_id, product_id)] = forecast_next_7(
            frame, artifact["model"], artifact["feature_cols"], store_id, product_id
        )
    return out


def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#223049", zeroline=False)
    fig.update_yaxes(gridcolor="#223049", zeroline=False)
    return fig


def header():
    st.markdown('<div class="hero-title">SHELFSENSE AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-tag">Predict Demand. Prevent Stockouts. Optimize Inventory.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="demo-banner"><b>Synthetic demo data</b> — this prototype uses a generated '
        "180-day supermarket dataset. It is not real company data. True demand is known here "
        "only because the world is simulated; in live retail, demand during stockouts is unobserved."
        "</div>",
        unsafe_allow_html=True,
    )


def page_dashboard(frame: pd.DataFrame, recs: pd.DataFrame) -> None:
    header()
    total_sales = int(frame["Observed_Sales"].sum())
    hidden = float(frame["Hidden_Demand"].sum())
    stockouts = int(frame["Stockout_Flag"].sum())
    at_risk = int(recs["risk"].isin(["High Risk", "Critical Risk"]).sum()) if not recs.empty else 0
    units_on_hand = int(recs["current_inventory"].sum()) if not recs.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total recorded sales", f"{total_sales:,}")
    c2.metric("Estimated hidden demand", f"{hidden:,.0f}")
    c3.metric("Stockout observations", f"{stockouts:,}")
    c4.metric("SKUs at High/Critical risk", f"{at_risk}")
    c5.metric("Closing inventory (units)", f"{units_on_hand:,}")

    daily = (
        frame.groupby("Date", as_index=False)
        .agg(
            Observed_Sales=("Observed_Sales", "sum"),
            Estimated_Demand=("Estimated_Demand", "sum"),
            Hidden_Demand=("Hidden_Demand", "sum"),
        )
        .sort_values("Date")
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Observed_Sales"], name="Observed sales", line=dict(color=TEAL)))
    fig.add_trace(
        go.Scatter(x=daily["Date"], y=daily["Estimated_Demand"], name="Estimated demand", line=dict(color=AMBER, dash="dot"))
    )
    fig.update_layout(title="Daily demand trend (all stores)", hovermode="x unified")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    left, right = st.columns(2)
    with left:
        top = (
            frame.groupby("Product_Name", as_index=False)["Observed_Sales"]
            .sum()
            .sort_values("Observed_Sales", ascending=False)
        )
        fig_top = px.bar(top, x="Product_Name", y="Observed_Sales", title="Top-selling products (observed)")
        fig_top.update_traces(marker_color=TEAL)
        st.plotly_chart(style_fig(fig_top), use_container_width=True)
    with right:
        store = frame.groupby("Store_Name", as_index=False)["Observed_Sales"].sum()
        fig_store = px.bar(store, x="Store_Name", y="Observed_Sales", title="Store-wise observed sales")
        fig_store.update_traces(marker_color="#4C6EF5")
        st.plotly_chart(style_fig(fig_store), use_container_width=True)

    freq = stockout_frequency(frame)
    st.subheader("Stockout frequency by SKU")
    st.dataframe(
        freq.rename(
            columns={
                "stockout_days": "Stockout days",
                "stockout_rate": "Stockout rate",
                "observed_sales": "Observed sales",
                "hidden_demand": "Hidden demand",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Inventory status (latest closing stock)")
    if not recs.empty:
        status_cols = [
            "Store_ID",
            "Product_Name",
            "current_inventory",
            "days_remaining",
            "risk",
            "recommended_qty",
        ]
        st.dataframe(recs[status_cols], use_container_width=True, hide_index=True)


def page_forecasting(frame: pd.DataFrame, artifact: dict, forecasts: dict) -> None:
    header()
    stores = sorted(frame["Store_ID"].unique())
    products = sorted(frame["Product_ID"].unique())
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", stores, format_func=lambda s: f"{s} — {frame.loc[frame['Store_ID']==s, 'Store_Name'].iloc[0]}")
    product = c2.selectbox(
        "Product",
        products,
        format_func=lambda p: f"{p} — {frame.loc[frame['Product_ID']==p, 'Product_Name'].iloc[0]}",
    )

    sku = frame[(frame["Store_ID"] == store) & (frame["Product_ID"] == product)].sort_values("Date")
    fc = forecasts.get((store, product), pd.DataFrame())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sku["Date"], y=sku["Observed_Sales"], name="Historical observed sales", line=dict(color=TEAL)))
    fig.add_trace(
        go.Scatter(
            x=sku["Date"],
            y=sku["Estimated_Demand"],
            name="Estimated demand (stockout-aware)",
            line=dict(color=AMBER, dash="dot"),
        )
    )
    hidden_days = sku[sku["Stockout_Flag"] == 1]
    if not hidden_days.empty:
        fig.add_trace(
            go.Bar(
                x=hidden_days["Date"],
                y=hidden_days["Hidden_Demand"],
                name="Estimated hidden demand",
                marker_color=RED,
                opacity=0.45,
            )
        )
    if not fc.empty:
        fig.add_trace(
            go.Scatter(
                x=fc["Date"],
                y=fc["Forecast_Demand"],
                name="7-day forecast",
                line=dict(color="#A78BFA", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc["Date"],
                y=fc["Baseline_Naive"],
                name="Seasonal-naive baseline",
                line=dict(color=MUTED, dash="dash"),
            )
        )
    fig.update_layout(title="Historical versus predicted demand", hovermode="x unified", barmode="overlay")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Next seven days")
    if not fc.empty:
        st.dataframe(fc.rename(columns={"Forecast_Demand": "Predicted demand", "Baseline_Naive": "Seasonal naive"}), use_container_width=True, hide_index=True)

    metrics = artifact.get("metrics", {})
    experiment = artifact.get("experiment", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RF MAE (holdout vs observed)", f"{metrics.get('rf_mae_vs_observed', float('nan')):.2f}")
    m2.metric("RF RMSE (holdout vs observed)", f"{metrics.get('rf_rmse_vs_observed', float('nan')):.2f}")
    m3.metric("Naive MAE (holdout vs observed)", f"{metrics.get('naive_mae_vs_observed', float('nan')):.2f}")
    m4.metric("Naive RMSE (holdout vs observed)", f"{metrics.get('naive_rmse_vs_observed', float('nan')):.2f}")

    st.caption(metrics.get("note", ""))
    with st.expander("Stockout recovery and artificial-stockout experiment (synthetic evaluation)"):
        r1, r2, r3 = st.columns(3)
        r1.metric("Holdout stockout rows", metrics.get("stockout_holdout_n", 0))
        r2.metric(
            "Recovery MAE vs true demand",
            f"{metrics.get('recovery_mae_vs_true_demand', float('nan')):.2f}",
        )
        r3.metric(
            "Recovery RMSE vs true demand",
            f"{metrics.get('recovery_rmse_vs_true_demand', float('nan')):.2f}",
        )
        e1, e2, e3 = st.columns(3)
        e1.metric("Experiment n (capped in-stock days)", experiment.get("n", 0))
        e2.metric(
            "Experiment MAE vs known demand",
            f"{experiment.get('mae_predicted_vs_known_demand', float('nan')):.2f}",
        )
        e3.metric(
            "Mean estimated hidden demand",
            f"{experiment.get('mean_estimated_hidden_demand', float('nan')):.2f}",
        )
        st.write(experiment.get("note", ""))


def page_inventory(recs: pd.DataFrame) -> None:
    header()
    st.subheader("Inventory intelligence")
    st.caption("Risk levels are rule-based (not a trained ML classifier). Current inventory is latest closing stock.")
    if recs.empty:
        st.warning("No recommendations available.")
        return
    show = recs[
        [
            "Store_ID",
            "Product_Name",
            "current_inventory",
            "avg_daily_demand",
            "days_remaining",
            "risk",
            "reorder_point",
            "safety_stock",
            "recommended_qty",
            "demo_unit_price",
            "estimated_purchase_cost",
            "reason",
        ]
    ].rename(
        columns={
            "current_inventory": "Current inventory (closing)",
            "avg_daily_demand": "Avg daily predicted demand",
            "days_remaining": "Days remaining",
            "reorder_point": "Reorder point",
            "safety_stock": "Safety stock",
            "recommended_qty": "Recommended order qty",
            "demo_unit_price": "Demo unit price",
            "estimated_purchase_cost": "Estimated purchase cost (demo)",
            "reason": "Risk reason",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)

    fig = px.bar(
        recs.sort_values("days_remaining"),
        x="Product_Name",
        y="days_remaining",
        color="risk",
        facet_col="Store_ID",
        title="Estimated days of stock remaining",
        color_discrete_map={
            "Low Risk": TEAL,
            "Medium Risk": AMBER,
            "High Risk": "#F77F00",
            "Critical Risk": RED,
        },
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)


def page_simulator(frame: pd.DataFrame, forecasts: dict) -> None:
    header()
    st.subheader("Business simulator")
    stores = sorted(frame["Store_ID"].unique())
    products = sorted(frame["Product_ID"].unique())
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", stores, key="sim_store")
    product = c2.selectbox(
        "Product",
        products,
        key="sim_product",
        format_func=lambda p: f"{p} — {frame.loc[frame['Product_ID']==p, 'Product_Name'].iloc[0]}",
    )
    sku = frame[(frame["Store_ID"] == store) & (frame["Product_ID"] == product)].sort_values("Date")
    latest_close = int(sku.iloc[-1]["Closing_Stock"])
    price = float(sku.iloc[-1]["Demo_Unit_Price"])
    fc = forecasts.get((store, product), pd.DataFrame())
    avg = float(fc["Forecast_Demand"].mean()) if not fc.empty else float(sku["Estimated_Demand"].tail(28).mean())
    std = float(fc["Forecast_Demand"].std(ddof=0) or 0) if not fc.empty else float(sku["Estimated_Demand"].tail(28).std(ddof=0) or 0)

    inv = st.slider("Current inventory (closing stock override)", min_value=0, max_value=max(400, latest_close * 3), value=latest_close)
    lead = st.slider("Supplier lead time (days)", min_value=1, max_value=14, value=DEFAULT_LEAD_TIME)
    budget = st.number_input("Purchasing budget (demo currency)", min_value=0.0, value=float(DEFAULT_BUDGET), step=50.0)

    calc = replenishment_math(avg, std, lead, inv)
    qty = calc["recommended_qty"]
    cost = round(qty * price, 2)
    note = "Within demo purchasing budget."
    if cost > budget:
        affordable = int(budget // price) if price > 0 else 0
        note = f"Budget caps the order from {qty} to {affordable} units using labelled demo prices."
        qty = affordable
        cost = round(qty * price, 2)
        calc["recommended_qty"] = qty

    exposure = max(0.0, avg * lead - inv)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current stock", f"{inv}")
    k2.metric("Stockout exposure (units over lead time)", f"{exposure:.1f}")
    k3.metric("Recommended order qty", f"{qty}")
    k4.metric("Estimated purchase cost (demo)", f"{cost:,.2f}")

    s1, s2, s3 = st.columns(3)
    s1.metric("Reorder point", calc["reorder_point"])
    s2.metric("Safety stock", calc["safety_stock"])
    s3.metric("Risk", calc["risk"])

    st.info(
        f"{calc['reason']} Average daily predicted demand is {calc['avg_daily_demand']} units. "
        f"Latest recorded closing stock for this SKU was {latest_close}. {note} "
        f"Demo unit price for {sku.iloc[-1]['Product_Name']} is {price:.2f}."
    )
    st.caption("Demo product prices are synthetic labels for this prototype, not supplier quotes.")


def main() -> None:
    inject_css()
    frame, artifact = load_pipeline()
    frame_hash = int(frame["Observed_Sales"].sum())
    forecasts = cached_forecasts(str(artifact.get("train_end")), frame_hash)
    recs = build_recommendations(frame, forecasts, lead_time=DEFAULT_LEAD_TIME, budget=DEFAULT_BUDGET)

    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Executive Dashboard",
            "Demand Forecasting",
            "Inventory Intelligence",
            "Business Simulator",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("RandomForest demand model is cached. Changing store or product does not retrain.")
    st.sidebar.caption(f"Holdout start: {artifact.get('test_start')}")

    if page == "Executive Dashboard":
        page_dashboard(frame, recs)
    elif page == "Demand Forecasting":
        page_forecasting(frame, artifact, forecasts)
    elif page == "Inventory Intelligence":
        page_inventory(recs)
    else:
        page_simulator(frame, forecasts)


if __name__ == "__main__":
    main()
