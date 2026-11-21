"""Streamlit front end: scenario planning for risk-adjusted manufacturing demand."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model

# Illustrative cumulative probabilities of reaching approval; replace with your own benchmarks.
DEFAULT_POS = {"Preclinical": 0.05, "Phase 1": 0.10, "Phase 2": 0.15, "Phase 3": 0.55,
               "Filed": 0.90, "Approved": 1.00}

st.set_page_config(page_title="Portfolio demand scenarios", layout="wide")
st.title("Risk-adjusted manufacturing demand")
st.caption("Synthetic portfolio for demonstration only. Edit the table, the success rates and the scenario, "
           "and see how demand compares with capacity.")

with st.sidebar:
    st.header("Scenario")
    pos_mult = st.slider("Success-rate multiplier", 0.5, 1.5, 1.0, 0.05,
                         help="Scales every phase's probability of reaching approval (capped at 100 %).")
    delay = st.slider("Launch delay (years)", 0, 3, 0)
    uplift = st.slider("Peak demand vs plan", 0.5, 1.5, 1.0, 0.05)
    capacity = st.number_input("Capacity (batches / year)", min_value=0, value=120, step=10)
    y0, y1 = st.slider("Years", 2024, 2040, (2025, 2036))
    years = np.arange(y0, y1 + 1)
    n_sims = st.select_slider("Simulations", [1000, 5000, 20000], value=5000)

left, right = st.columns([3, 2])
with right:
    st.subheader("Probability of reaching approval")
    pos_df = st.data_editor(pd.DataFrame({"phase": list(DEFAULT_POS), "probability": list(DEFAULT_POS.values())}),
                            hide_index=True, disabled=["phase"],
                            column_config={"probability": st.column_config.NumberColumn(min_value=0.0, max_value=1.0)})
    pos = dict(zip(pos_df["phase"], pos_df["probability"]))
    st.subheader("Portfolio")
    portfolio = st.data_editor(pd.read_csv(Path(__file__).parent / "data" / "portfolio.csv"),
                               num_rows="dynamic", hide_index=True,
                               column_config={"phase": st.column_config.SelectboxColumn(options=list(DEFAULT_POS))})

try:
    df = model.validate(portfolio, pos)
except ValueError as e:
    st.error(f"Cannot run the model: {e}")
    st.stop()

sims = model.simulate(df, pos, years, pos_mult, delay, uplift, n_sims=n_sims)
summary = model.summarise(sims, years, capacity)

with left:
    fig = go.Figure([
        go.Scatter(x=years, y=summary["P90"], line=dict(width=0), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=years, y=summary["P10"], fill="tonexty", line=dict(width=0), name="P10–P90",
                   fillcolor="rgba(31,119,180,0.2)"),
        go.Scatter(x=years, y=summary["P50"], name="P50", line=dict(color="rgb(31,119,180)")),
        go.Scatter(x=years, y=[capacity] * len(years), name="Capacity", line=dict(color="black", dash="dash")),
    ])
    fig.update_layout(yaxis_title="Batches per year", xaxis_title="Year", hovermode="x unified",
                      margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    peak = summary.loc[summary["P(demand > capacity)"].idxmax()]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"P50 demand {years[-1]}", f"{summary['P50'].iloc[-1]:.0f}")
    c2.metric("Highest risk of shortfall", f"{peak['P(demand > capacity)']:.0%}", f"in {int(peak['year'])}",
              delta_color="off")
    c3.metric("Molecules in model", len(df))
    st.dataframe(summary.style.format({"P10": "{:.0f}", "P50": "{:.0f}", "P90": "{:.0f}", "mean": "{:.1f}",
                                       "P(demand > capacity)": "{:.0%}"}), hide_index=True)
    st.download_button("Download table (CSV)", summary.to_csv(index=False), "demand_scenario.csv")
