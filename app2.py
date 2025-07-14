import streamlit as st
import pandas as pd
import numpy as np
import joblib

pipeline = joblib.load("walmart_xgb_pipeline.pkl")  

st.title("🛒 Walmart Sales Predictor")

# --- User Inputs ---
store = st.number_input("Store ID", min_value=1)
dept = st.number_input("Department ID", min_value=1)
date = st.date_input("Date")
is_holiday = st.checkbox("Is it a holiday?")
temperature = st.number_input("Temperature (°F)")
fuel_price = st.number_input("Fuel Price ($)")
cpi = st.number_input("CPI")
unemployment = st.number_input("Unemployment Rate (%)")
markdown1 = st.number_input("MarkDown1")
markdown2 = st.number_input("MarkDown2")
markdown3 = st.number_input("MarkDown3")
markdown4 = st.number_input("MarkDown4")
markdown5 = st.number_input("MarkDown5")
size = st.number_input("Store Size")
store_type = st.selectbox("Store Type", ["A", "B", "C"])

# --- Dummy Lag/Rolling Values (can be dynamic later) ---
lag_1 = 5000.0
lag_2 = 4800.0
lag_3 = 4600.0
lag_4 = 4400.0
lag_5 = 4200.0
lag_6 = 4000.0
rolling_3 = 4800.0
rolling_4 = 4700.0
rolling_6 = 4600.0

# --- Prediction Trigger ---
if st.button("Predict"):
    df = pd.DataFrame([{
        "Store": store,
        "Dept": dept,
        "Date": pd.to_datetime(date),
        "IsHoliday": int(is_holiday),
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
        "CPI": cpi,
        "Unemployment": unemployment,
        "MarkDown1": markdown1,
        "MarkDown2": markdown2,
        "MarkDown3": markdown3,
        "MarkDown4": markdown4,
        "MarkDown5": markdown5,
        "Size": size,
        "Type": store_type,
        "Lag_1": lag_1,
        "Lag_2": lag_2,
        "Lag_3": lag_3,
        "Lag_4": lag_4,
        "Lag_5": lag_5,
        "Lag_6": lag_6,
        "Rolling_Mean_3": rolling_3,
        "Rolling_Mean_4": rolling_4,
        "Rolling_Mean_6": rolling_6,
    }])

    # --- Feature Engineering ---
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Week_Sin"] = np.sin(2 * np.pi * df["Week"] / 52)
    df["Week_Cos"] = np.cos(2 * np.pi * df["Week"] / 52)
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["Quarter"] = df["Date"].dt.quarter
    df["IsYearEnd"] = df["Month"] >= 11
    df["IsBackToSchool"] = df["Month"].isin([7, 8]).astype(int)
    df["IsHolidaySeason"] = df["Month"].isin([11, 12]).astype(int)
    df["IsPreHoliday"] = 0
    df["IsPostHoliday"] = 0
    df["Sales_Ratio"] = 1.0
    df["IsSalesSpiking"] = 0

    # --- Drop problematic columns and fix dtypes ---
    df = df.drop(columns=["Date"])               # ❌ Drop datetime
    df["Type"] = df["Type"].astype(str)          # ✅ Make sure Type is string for one-hot encoding

    # --- Prediction ---
    pred = pipeline.predict(df)[0]
    st.success(f"📈 Predicted Weekly Sales: ${pred:.2f}")
