# streamlit_app.py (High Accuracy Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("housing.csv")

# Drop rows with missing values
df = df.dropna()

# Features & target
numerical = ["housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
categorical = ["ocean_proximity"]
target = "median_house_value"

X = df[numerical + categorical]
y = df[target]

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(drop="first"), categorical)
])

# Build pipeline
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="High Accuracy House Price Predictor", layout="centered")
st.title("🏡 House Price Predictor (Random Forest)")
st.write("This app uses Random Forest to predict *Median House Value* based on multiple housing features.")

# User input
st.sidebar.header("📥 Input Features")
user_input = {}
for feature in numerical:
    min_val = int(df[feature].min())
    max_val = int(df[feature].max())
    median_val = int(df[feature].median())
    user_input[feature] = st.sidebar.slider(f"{feature.replace('_', ' ').title()}", min_val, max_val, median_val)

user_input["ocean_proximity"] = st.sidebar.selectbox("Ocean Proximity", df["ocean_proximity"].unique())
input_df = pd.DataFrame([user_input])

# Predict for user
predicted_price = model.predict(input_df)[0]
st.success(f"🏠 Predicted Median House Value: ${predicted_price:,.2f}")

# Evaluation
with st.expander("📊 Model Evaluation"):
    st.write(f"*MAE:* ${mae:,.2f}")
    st.write(f"*RMSE:* ${rmse:,.2f}")
    st.write(f"*R² Score:* {r2:.4f}")

# Plot actual vs predicted
st.subheader("📉 Actual vs Predicted")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, y_pred, alpha=0.4, color="blue")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted House Value")
ax.grid(True)
st.pyplot(fig)