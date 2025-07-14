import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained pipeline
pipeline = joblib.load("walmart_xgb_pipeline.pkl")

st.title("📦 Walmart Sales Prediction")

store = st.number_input("Store ID", min_value=1)
dept = st.number_input("Department ID", min_value=1)
year = st.selectbox("Year", [2010, 2011, 2012])
month = st.slider("Month", 1, 12)
week = st.slider("Week of Year", 1, 52)

# Create sin/cos
week_sin = np.sin(2 * np.pi * week / 52)
week_cos = np.cos(2 * np.pi * week / 52)

# Lags & rolling: for now, you can input manually or test with sample
lags = [st.number_input(f"Lag {i}", value=0.0) for i in range(1, 7)]
rolls = [st.number_input(f"Rolling Mean {w}", value=0.0) for w in [3, 4, 6]]

if st.button("Predict"):
    data = pd.DataFrame([[
        store, dept, year, month, week_sin, week_cos, *lags, *rolls
    ]], columns=[
        'Store', 'Dept', 'Year', 'Month', 'Week_Sin', 'Week_Cos',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Lag_6',
        'Rolling_Mean_3', 'Rolling_Mean_4', 'Rolling_Mean_6'
    ])
    pred = pipeline.predict(data)[0]
    st.success(f"📊 Predicted Weekly Sales: ${pred:,.2f}")
