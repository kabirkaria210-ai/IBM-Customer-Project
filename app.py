import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="centered"
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("✈️ Customer Churn Predictor")
st.markdown("### Travel Customer Churn Prediction using Random Forest")
st.markdown("Fill in the customer details below to predict whether they will churn.")
st.divider()

# ── Input Form ───────────────────────────────────────────────────────────────
st.subheader("📋 Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=70, value=30)
    frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes", "No Record"])
    annual_income = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])

with col2:
    services_opted = st.slider("Services Opted", min_value=1, max_value=6, value=3)
    account_synced = st.selectbox("Account Synced to Social Media", ["No", "Yes"])
    booked_hotel = st.selectbox("Booked Hotel or Not", ["No", "Yes"])

st.divider()

# ── Encode inputs (same as notebook) ─────────────────────────────────────────
def encode_inputs(age, frequent_flyer, annual_income, services_opted, account_synced, booked_hotel):
    ff_map = {"Yes": 1, "No": 0, "No Record": 2}
    income_map = {"Low Income": 0, "Middle Income": 1, "High Income": 2}
    binary_map = {"Yes": 1, "No": 0}

    return pd.DataFrame([{
        "Age": age,
        "FrequentFlyer": ff_map[frequent_flyer],
        "AnnualIncomeClass": income_map[annual_income],
        "ServicesOpted": services_opted,
        "AccountSyncedToSocialMedia": binary_map[account_synced],
        "BookedHotelOrNot": binary_map[booked_hotel]
    }])

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True):
    input_df = encode_inputs(age, frequent_flyer, annual_income,
                             services_opted, account_synced, booked_hotel)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    churn_prob   = round(probability[1] * 100, 2)
    no_churn_prob = round(probability[0] * 100, 2)

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is **LIKELY TO CHURN**")
        st.markdown(f"**Churn Probability: {churn_prob}%**")
        st.progress(int(churn_prob))
        st.warning("💡 Recommendation: Offer a discount or personalized deal to retain this customer.")
    else:
        st.success(f"✅ This customer is **NOT LIKELY TO CHURN**")
        st.markdown(f"**Retention Probability: {no_churn_prob}%**")
        st.progress(int(no_churn_prob))
        st.info("💡 Recommendation: Keep engaging this customer with loyalty rewards.")

    st.divider()
    st.subheader("📈 Probability Breakdown")
    prob_df = pd.DataFrame({
        "Outcome": ["Will NOT Churn", "Will Churn"],
        "Probability (%)": [no_churn_prob, churn_prob]
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("B.Tech – Gen AI | 2nd Semester | Customer Churn Prediction Project")
