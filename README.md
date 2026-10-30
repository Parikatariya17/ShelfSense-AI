# SHELFSENSE AI

**Predict Demand. Prevent Stockouts. Optimize Inventory.**

ShelfSense AI is a lightweight Streamlit prototype for stockout-aware demand estimation, seven-day forecasting, and rule-based inventory replenishment.

This project uses **synthetic demo data**. It is not real company data. True demand is known in the CSV only because the simulator generated it. In live retail, demand during a stockout is unobserved.

## What it does

1. Estimates hidden demand when recorded sales are censored by empty shelves.
2. Forecasts the next seven days of demand with a Random Forest, compared with a seasonal-naive baseline.
3. Flags SKUs that may stock out using a transparent rule-based risk engine (not a trained classifier).
4. Recommends replenishment using safety stock, reorder point, and a 7-day cover target.

## Stack

Python 3.11+ (this machine was tested with Python 3.13), Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Joblib.

## Mac terminal commands

```bash
cd ~/Desktop/ShelfSense-AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_generator
python -m src.forecasting
streamlit run app.py
```

The app will generate the CSV and train the model automatically if those files are missing. The commands above make that step explicit for testing.

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Project layout

```
ShelfSense-AI/
├── app.py
├── requirements.txt
├── README.md
├── data/demo_retail_data.csv
├── models/demand_model.joblib
└── src/
    ├── data_generator.py
    ├── preprocessing.py
    ├── forecasting.py
    ├── stockout_analysis.py
    └── inventory_optimizer.py
```

## Method notes

- `Available_Stock` is opening inventory. `Closing_Stock = Available_Stock - Observed_Sales`. Next-day opening is previous closing plus overnight `Restock_Units`.
- The demand model never uses `True_Demand`, same-day sales, `Available_Stock`, or `Stockout_Flag` as features. Lag and rolling features apply `shift(1)` first.
- Training uses only non-stockout rows before a chronological 28-day holdout.
- Inventory recommendations use latest **closing** stock, not opening stock.
- Demo unit prices are labelled synthetic values for cost illustrations.
