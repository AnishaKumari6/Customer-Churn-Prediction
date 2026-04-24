import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("churn_model.pkl")
columns = joblib.load("columns.pkl")

# -----------------------------
# HEADER
# -----------------------------
st.title("📊 Customer Churn Prediction System")

st.markdown("""
This system predicts whether a customer is likely to churn based on behavioral and transactional data.

**Use cases:**
- Identify high-risk users  
- Take proactive retention actions  
- Optimize marketing strategies  
""")

st.markdown("---")

# -----------------------------
# SINGLE USER PREDICTION
# -----------------------------
st.header("🔮 Single User Prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70)
    purchases = st.number_input("Total Purchases", 0, 100)
    days_since = st.number_input("Days Since Last Purchase", 0, 365)

with col2:
    support_calls = st.number_input("Customer Service Calls", 0, 20)
    discount = st.slider("Discount Usage Rate", 0.0, 1.0)

# -----------------------------
# CREATE INPUT DATA
# -----------------------------
input_data = {
    "Age": age,
    "Total_Purchases": purchases,
    "Customer_Service_Calls": support_calls,
    "Discount_Usage_Rate": discount,
    "Days_Since_Last_Purchase": days_since,
}

input_df = pd.DataFrame([input_data])

# -----------------------------
# FEATURE ENGINEERING (SAME AS TRAINING)
# -----------------------------
input_df['Engagement_score'] = 0
input_df['Avg_Purchase_Per_Year'] = purchases / (1 + 1)
input_df['Value_Per_Purchase'] = 0
input_df['Recency_Score'] = 1 / (days_since + 1)
input_df['Abandonment_Impact'] = 0
input_df['Frustration_Score'] = support_calls
input_df['Loyalty_Score'] = purchases
input_df['Discount_Sensitivity'] = discount * purchases

# match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# clean
input_df.replace([np.inf, -np.inf], 0, inplace=True)
input_df.fillna(0, inplace=True)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Churn", use_container_width=True):

    prob = model.predict_proba(input_df)[0][1]

    st.markdown("### 📊 Prediction Result")

    colA, colB, colC = st.columns(3)

    colA.metric("Churn Probability", f"{prob:.2f}")

    if prob > 0.7:
        colB.metric("Risk Level", "High 🔥")
        colC.metric("Recommended Action", "Give Discount")
        st.error("🚨 High Risk Customer → Immediate retention action needed")

    elif prob > 0.4:
        colB.metric("Risk Level", "Medium ⚠️")
        colC.metric("Recommended Action", "Send Email")
        st.warning("⚠️ Medium Risk → Engage user with offers")

    else:
        colB.metric("Risk Level", "Low ✅")
        colC.metric("Recommended Action", "No Action")
        st.success("✅ Low Risk → Customer is stable")

st.markdown("---")

# -----------------------------
# BATCH PREDICTION
# -----------------------------
st.header("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    data = data.reindex(columns=columns, fill_value=0)

    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.fillna(0, inplace=True)

    probs = model.predict_proba(data)[:, 1]
    data['churn_probability'] = probs

    st.subheader("Preview Results")
    st.dataframe(data.head())

    st.download_button(
        "⬇️ Download Predictions",
        data.to_csv(index=False),
        file_name="churn_predictions.csv",
        use_container_width=True
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "<h5 style='text-align: center;'>Built by Arpit Jain</h5>",
    unsafe_allow_html=True
)