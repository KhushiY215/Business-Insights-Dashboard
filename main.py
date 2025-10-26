# file: main.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import timedelta

st.set_page_config(page_title="📊 Professional Business Insights Dashboard", layout="wide")

st.title("📊 Professional AI-Powered Business Insights Dashboard")

# --- 1. Upload CSV ---
uploaded_file = st.file_uploader("Upload your Superstore CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("First 5 rows of your dataset")
    st.write(df.head())

    # --- 2. Basic preprocessing ---
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df = df.sort_values('Order Date')
    st.subheader("Columns in dataset")
    st.write(df.columns.tolist())

    # --- 3. Check numeric columns ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("Numeric columns detected:", numeric_cols)

    # --- 4. Aggregate daily sales ---
    agg_dict = {'Sales': 'sum'}
    if 'Quantity' in numeric_cols:
        agg_dict['Quantity'] = 'sum'
    if 'Discount' in numeric_cols:
        agg_dict['Discount'] = 'mean'

    daily_sales = df.groupby('Order Date').agg(agg_dict).reset_index()

    # --- 5. Create lag and rolling features ---
    for lag in [1, 7, 14]:
        daily_sales[f'lag_{lag}'] = daily_sales['Sales'].shift(lag)
    daily_sales['rolling_7'] = daily_sales['Sales'].rolling(7).mean()
    daily_sales['rolling_30'] = daily_sales['Sales'].rolling(30).mean()
    daily_sales = daily_sales.dropna()

    # --- 6. Prepare features ---
    feature_cols = []
    if 'Quantity' in daily_sales.columns:
        feature_cols.append('Quantity')
    if 'Discount' in daily_sales.columns:
        feature_cols.append('Discount')
    feature_cols += [f'lag_{lag}' for lag in [1,7,14]] + ['rolling_7','rolling_30']

    X = daily_sales[feature_cols]
    y = daily_sales['Sales']

    # --- 7. Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- 8. Train XGBoost ---
    model = XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- 9. Model evaluation ---
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.subheader("Model Performance")
    st.write(f"R² Score: {r2:.2f}")
    st.write(f"RMSE: {rmse:.2f}")

    # --- 10. Forecast next 14 days ---
    last_row = daily_sales.iloc[-1].copy()
    future_preds = []
    future_dates = []

    for i in range(1,15):
        next_date = last_row['Order Date'] + timedelta(days=1)
        future_dates.append(next_date)

        # Create features for next day
        features = {}
        if 'Quantity' in last_row.index:
            features['Quantity'] = last_row['Quantity']  # Assume same
        if 'Discount' in last_row.index:
            features['Discount'] = last_row['Discount']  # Assume same

        # Lag features
        for lag in [1,7,14]:
            features[f'lag_{lag}'] = last_row['Sales'] if lag ==1 else daily_sales.iloc[-lag]['Sales']
        # Rolling features
        features['rolling_7'] = daily_sales['Sales'].iloc[-7:].mean()
        features['rolling_30'] = daily_sales['Sales'].iloc[-30:].mean()

        X_future = pd.DataFrame([features])
        pred = model.predict(X_future)[0]
        future_preds.append(pred)

        # Append prediction as last_row for next iteration
        last_row['Sales'] = pred

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Sales': future_preds
    })

    st.subheader("📅 Forecast for next 14 days")
    st.write(forecast_df)

    # --- 11. Visualizations ---
    st.subheader("Sales over Time")
    fig_sales = px.line(daily_sales, x='Order Date', y='Sales', title="Daily Sales Trend")
    st.plotly_chart(fig_sales)

    st.subheader("Predicted vs Actual (Test Set)")
    fig_pred = px.line(x=y_test.index, y=y_test.values, labels={'x':'Index','y':'Sales'}, title="Actual vs Predicted Sales")
    fig_pred.add_scatter(x=y_test.index, y=y_pred, mode='lines', name='Predicted')
    st.plotly_chart(fig_pred)

    # --- 12. Auto Insights ---
    top_region = df.groupby('Region')['Sales'].sum().idxmax()
    top_category = df.groupby('Category')['Sales'].sum().idxmax()
    avg_sales = daily_sales['Sales'].mean()

    st.subheader("🧠 Automated Insights")
    st.write(f"Average daily sales: {avg_sales:.2f}")
    st.write(f"Top Region: {top_region}")
    st.write(f"Top Product Category: {top_category}")

    # --- 13. Download forecast ---
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast CSV", data=csv, file_name='forecast_sales.csv', mime='text/csv')
