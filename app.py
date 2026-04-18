import streamlit as st
import pandas as pd
import joblib
import os

# -------------------- Load Model and Metrics --------------------
MODEL_PATH = os.path.join("Models", "Best_Model.pkl")
RESULT_PATH = os.path.join("Models", "model_results.csv")

model = joblib.load(MODEL_PATH)
metrics = pd.read_csv(RESULT_PATH)
metrics.columns = metrics.columns.str.strip()  # remove extra spaces

# Best model by F1 Score
best_metrics = metrics.loc[metrics["F1_Score"].idxmax()]

# -------------------- Streamlit Page --------------------
st.set_page_config(page_title="Insurance Customer Response Prediction", layout="wide")

st.title(" Insurance Customer Response Prediction")
st.write("Predict whether a customer will respond to a vehicle insurance offer.")

# -------------------- Sidebar Inputs --------------------
st.sidebar.header("Customer Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
driving_license = st.sidebar.selectbox("Driving License", [0,1])
region_code = st.sidebar.number_input("Region Code", 1, 53, 1)
previously_insured = st.sidebar.selectbox("Previously Insured", [0,1])
vehicle_age = st.sidebar.selectbox("Vehicle Age", ["< 1 Year","1-2 Year","> 2 Years"])
vehicle_damage = st.sidebar.selectbox("Vehicle Damage", ["Yes","No"])
annual_premium = st.sidebar.number_input("Annual Premium", 1000, 100000, 30000, step=1000)
policy_sales_channel = st.sidebar.number_input("Policy Sales Channel", 1, 30, 26)
customer_vintage = st.sidebar.slider("Customer Vintage (days)", 0, 500, 100)

# -------------------- Encode Categorical Inputs --------------------
Gender_enc = 1 if gender == "Male" else 0
Vehicle_Damage_enc = 1 if vehicle_damage == "Yes" else 0
Vehicle_Age_map = {"< 1 Year":0, "1-2 Year":1, "> 2 Years":2}
Vehicle_Age_enc = Vehicle_Age_map[vehicle_age]

# -------------------- Prepare Input DataFrame --------------------
input_df = pd.DataFrame({
    "Age":[age],
    "Gender":[Gender_enc],
    "Driving_License":[driving_license],
    "Region_Code":[region_code],
    "Previously_Insured":[previously_insured],
    "Vehicle_Age":[Vehicle_Age_enc],
    "Vehicle_Damage":[Vehicle_Damage_enc],
    "Annual_Premium":[annual_premium],
    "Policy_Sales_Channel":[policy_sales_channel],
    "Vintage":[customer_vintage]
})

# -------------------- Prediction --------------------
if st.button("Predict Response"):
    prediction_class = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]

    if prediction_class == 1:
        st.success(f"✅ Customer is likely to respond. (Probability: {prediction_prob:.2f})")
    else:
        st.warning(f"❌ Customer is unlikely to respond. (Probability: {prediction_prob:.2f})")

# -------------------- Display Metrics --------------------
st.markdown("---")
st.subheader("Best Model Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", round(best_metrics["Accuracy"],2))
col2.metric("F1 Score", round(best_metrics["F1_Score"],2))
col3.metric("ROC_AUC", round(best_metrics["ROC_AUC"],2))

st.markdown("---")
st.write("Built with Streamlit")