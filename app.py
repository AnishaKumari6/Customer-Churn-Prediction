import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and columns
model = joblib.load("churn_model.pkl")
columns = joblib.load("columns.pkl")

st.title("Customer Churn Prediction System")

# -------------------------
# SINGLE USER INPUT
# -------------------------
st.header("Single User Prediction")

age = st.slider("Age", 18, 70)
purchases = st.number_input("Total Purchases", 0, 100)
support_calls = st.number_input("Customer Service Calls", 0, 20)
discount = st.slider("Discount Usage Rate", 0.0, 1.0)
days_since = st.number_input("Days Since Last Purchase", 0, 365)

# Create input dictionary (basic features)
input_data = {
    "Age": age,
    "Total_Purchases": purchases,
    "Customer_Service_Calls": support_calls,
    "Discount_Usage_Rate": discount,
    "Days_Since_Last_Purchase": days_since,
}

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# Recreate engineered features (IMPORTANT)
input_df['Engagement_score'] = 0
input_df['Avg_Purchase_Per_Year'] = purchases / (1 + 1)
input_df['Value_Per_Purchase'] = 0
input_df['Recency_Score'] = 1 / (days_since + 1)
input_df['Abandonment_Impact'] = 0
input_df['Frustration_Score'] = support_calls
input_df['Loyalty_Score'] = purchases
input_df['Discount_Sensitivity'] = discount * purchases

# Handle missing columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Fix any inf/nan
input_df.replace([np.inf, -np.inf], 0, inplace=True)
input_df.fillna(0, inplace=True)

# Prediction
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]

    st.subheader(f"Churn Probability: {prob:.2f}")

    if prob > 0.7:
        st.error("High Risk → Give Discount")
    elif prob > 0.4:
        st.warning("Medium Risk → Send Email")
    else:
        st.success("Low Risk → No Action")

# -------------------------
# BATCH UPLOAD
# -------------------------
st.header("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    data = data.reindex(columns=columns, fill_value=0)

    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.fillna(0, inplace=True)

    probs = model.predict_proba(data)[:, 1]
    data['churn_probability'] = probs

    st.write(data.head())

    st.download_button(
        "Download Results",
        data.to_csv(index=False),
        file_name="predictions.csv"
    )

    st.markdown(
        "<h5 style='text-align: center;'>Built by Arpit Jain</h5>",
        unsafe_allow_html=True
    )