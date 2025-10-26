# file: app.py

import streamlit as st           # Streamlit for UI
import pandas as pd             # For data handling
import numpy as np              # For numeric operations
import plotly.express as px     # For interactive charts
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- 1. Title ---
st.title("📊 AI-Powered Business Insights Dashboard")

# --- 2. Specify CSV file path here ---
file_path = "train.csv"  # <-- Replace with your CSV path

# --- 3. Load data ---
try:
    df = pd.read_csv(file_path)
    st.success(f"Data loaded successfully from: {file_path}")
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

st.write("Here are the first rows of your data:", df.head())

# --- 4. Basic preprocessing ---
# --- 4. Basic preprocessing ---
# Convert Order Date to datetime (day first)
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
st.write("Data after adding Year & Month:", df[['Order Date', 'Year', 'Month']].head())


# --- 5. Exploratory Data Analysis (EDA) ---
monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

# Monthly Sales Trend
fig1 = px.line(
    monthly_sales,
    x=monthly_sales.apply(lambda row: f"{int(row['Year'])}-{int(row['Month'])}", axis=1),
    y='Sales',
    title="Monthly Total Sales Trend"
)
st.plotly_chart(fig1)

# Sales by Region
fig2 = px.bar(df, x='Region', y='Sales', color='Region', title="Sales by Region")
st.plotly_chart(fig2)

# --- 6. Feature preparation for ML model ---
X = monthly_sales[['Year', 'Month']]
y = monthly_sales['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 7. Build regression model ---
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
st.write(f"Model R² score: {r2:.2f}")

# --- 8. Forecast for next 3 months ---
last_year = monthly_sales['Year'].max()
last_month = monthly_sales[monthly_sales['Year'] == last_year]['Month'].max()

future_months = []
for i in range(1, 4):
    m = last_month + i
    y_ = last_year
    if m > 12:
        m -= 12
        y_ += 1
    future_months.append({'Year': y_, 'Month': m})

future_df = pd.DataFrame(future_months)
future_pred = model.predict(future_df)
future_df['Predicted_Sales'] = future_pred

st.write("📅 Forecast for next 3 months:")
st.write(future_df)

# --- 9. Auto-Generated Insight ---
avg_sales = monthly_sales['Sales'].mean()
top_region = df.groupby('Region')['Sales'].sum().idxmax()
st.write(f"🧠 Insight: Average monthly sales ≈ {avg_sales:.0f}. Top performing region: **{top_region}**.")

# --- 10. Download predictions ---
csv = future_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download predictions as CSV",
    data=csv,
    file_name='future_sales_predictions.csv',
    mime='text/csv'
)
