import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor | Travel AI",
    page_icon="✈️",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #080C24 !important;
    color: #FFFFFF !important;
}

.stApp {
    background: linear-gradient(135deg, #080C24 0%, #0F1535 50%, #080C24 100%) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.hero-banner {
    background: linear-gradient(135deg, #0F1535, #111A3E);
    border: 1px solid #1A2456;
    border-top: 4px solid #00C2FF;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 194, 255, 0.15);
}
.hero-icon { font-size: 3.5rem; }
.hero-title { font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin: 0.3rem 0; }
.hero-title span { color: #00C2FF; }
.hero-subtitle { font-size: 1rem; color: #8899BB; margin-top: 0.3rem; }
.hero-badge {
    display: inline-block;
    background: rgba(0,194,255,0.12);
    border: 1px solid rgba(0,194,255,0.4);
    color: #00C2FF;
    padding: 0.3rem 1.2rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-top: 0.8rem;
    letter-spacing: 1.5px;
}

.section-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #00C2FF;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1A2456;
}

.stat-row {
    display: flex;
    gap: 0.8rem;
    margin: 1rem 0;
}
.stat-box {
    flex: 1;
    background: #0F1535;
    border: 1px solid #1A2456;
    border-top: 3px solid #00C2FF;
    border-radius: 10px;
    padding: 1rem 0.5rem;
    text-align: center;
}
.stat-number { font-size: 1.6rem; font-weight: 800; color: #00C2FF; }
.stat-label { font-size: 0.7rem; color: #8899BB; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }

.custom-divider { border: none; border-top: 1px solid #1A2456; margin: 1.5rem 0; }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #8899BB !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00C2FF, #0090CC) !important;
    color: #080C24 !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem !important;
    width: 100% !important;
    letter-spacing: 1.5px !important;
    box-shadow: 0 4px 20px rgba(0,194,255,0.35) !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #33CFFF, #00C2FF) !important;
    box-shadow: 0 6px 28px rgba(0,194,255,0.55) !important;
}

.result-card-churn {
    background: linear-gradient(135deg, #1A0808, #2D0F0F);
    border: 1px solid #FF4444;
    border-left: 6px solid #FF4444;
    border-radius: 16px;
    padding: 2.2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(255,68,68,0.2);
    margin: 1rem 0;
}
.result-card-safe {
    background: linear-gradient(135deg, #081A08, #0F2D15);
    border: 1px solid #00FF88;
    border-left: 6px solid #00FF88;
    border-radius: 16px;
    padding: 2.2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,255,136,0.2);
    margin: 1rem 0;
}
.result-icon { font-size: 2.8rem; }
.result-label-churn { font-size: 1.5rem; font-weight: 800; color: #FF4444; margin: 0.5rem 0; }
.result-label-safe  { font-size: 1.5rem; font-weight: 800; color: #00FF88; margin: 0.5rem 0; }
.result-prob { font-size: 3.5rem; font-weight: 800; color: #FFFFFF; }
.result-prob-label { font-size: 0.8rem; color: #8899BB; text-transform: uppercase; letter-spacing: 1.5px; }

.rec-box {
    background: rgba(0,194,255,0.08);
    border: 1px solid rgba(0,194,255,0.25);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    color: #CCECFF;
    font-size: 0.92rem;
}
.rec-box strong { color: #00C2FF; }

.prob-bar-bg {
    background: #1A2456;
    border-radius: 8px;
    height: 14px;
    overflow: hidden;
    margin: 0.4rem 0 1rem 0;
}
.prob-bar-green { height: 100%; background: linear-gradient(90deg, #00C2FF, #00FF88); border-radius: 8px; }
.prob-bar-red   { height: 100%; background: linear-gradient(90deg, #FF4444, #FF8888); border-radius: 8px; }

.footer {
    text-align: center;
    color: #556688;
    font-size: 0.78rem;
    padding: 1.5rem 0 0.5rem 0;
    border-top: 1px solid #1A2456;
    margin-top: 2rem;
}
.footer span { color: #00C2FF; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-icon">✈️</div>
    <div class="hero-title">Customer <span>Churn</span> Predictor</div>
    <div class="hero-subtitle">AI-powered prediction for travel customer retention</div>
    <div class="hero-badge">⚡ POWERED BY RANDOM FOREST</div>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
    <div class="stat-box"><div class="stat-number">954</div><div class="stat-label">Training Records</div></div>
    <div class="stat-box"><div class="stat-number">~90%</div><div class="stat-label">Model Accuracy</div></div>
    <div class="stat-box"><div class="stat-number">6</div><div class="stat-label">Input Features</div></div>
    <div class="stat-box"><div class="stat-number">100</div><div class="stat-label">Decision Trees</div></div>
</div>
<hr class="custom-divider">
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📋 Enter Customer Details</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    age             = st.slider("🎂  Age", min_value=18, max_value=70, value=30)
    frequent_flyer  = st.selectbox("✈️  Frequent Flyer", ["No", "Yes", "No Record"])
    annual_income   = st.selectbox("💰  Annual Income Class", ["Low Income", "Middle Income", "High Income"])

with col2:
    services_opted  = st.slider("🛎️  Services Opted", min_value=1, max_value=6, value=3)
    account_synced  = st.selectbox("📱  Account Synced to Social Media", ["No", "Yes"])
    booked_hotel    = st.selectbox("🏨  Booked Hotel or Not", ["No", "Yes"])

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ── Encode ────────────────────────────────────────────────────────────────────
def encode_inputs(age, frequent_flyer, annual_income, services_opted, account_synced, booked_hotel):
    return pd.DataFrame([{
        "Age": age,
        "FrequentFlyer": {"Yes": 1, "No": 0, "No Record": 2}[frequent_flyer],
        "AnnualIncomeClass": {"Low Income": 0, "Middle Income": 1, "High Income": 2}[annual_income],
        "ServicesOpted": services_opted,
        "AccountSyncedToSocialMedia": {"Yes": 1, "No": 0}[account_synced],
        "BookedHotelOrNot": {"Yes": 1, "No": 0}[booked_hotel]
    }])

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍  PREDICT CHURN", use_container_width=True):
    input_df    = encode_inputs(age, frequent_flyer, annual_income, services_opted, account_synced, booked_hotel)
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    churn_prob    = round(probability[1] * 100, 2)
    no_churn_prob = round(probability[0] * 100, 2)

    st.markdown('<p class="section-title">📊 Prediction Result</p>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-card-churn">
            <div class="result-icon">⚠️</div>
            <div class="result-label-churn">LIKELY TO CHURN</div>
            <div class="result-prob">{churn_prob}%</div>
            <div class="result-prob-label">Churn Probability</div>
        </div>
        <div class="rec-box">💡 <strong>Recommendation:</strong> This customer is at risk. Offer a personalized discount, loyalty reward, or exclusive travel deal to retain them.</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card-safe">
            <div class="result-icon">✅</div>
            <div class="result-label-safe">NOT LIKELY TO CHURN</div>
            <div class="result-prob">{no_churn_prob}%</div>
            <div class="result-prob-label">Retention Probability</div>
        </div>
        <div class="rec-box">💡 <strong>Recommendation:</strong> This customer is happy and loyal. Keep them engaged with regular updates and loyalty rewards.</div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider"><p class="section-title">📈 Probability Breakdown</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;">
            <span style="color:#00FF88;font-weight:600;font-size:0.9rem;">✅ Will NOT Churn</span>
            <span style="color:#FFFFFF;font-weight:700;">{no_churn_prob}%</span>
        </div>
        <div class="prob-bar-bg"><div class="prob-bar-green" style="width:{no_churn_prob}%;"></div></div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;margin-top:0.5rem;">
            <span style="color:#FF4444;font-weight:600;font-size:0.9rem;">⚠️ Will Churn</span>
            <span style="color:#FFFFFF;font-weight:700;">{churn_prob}%</span>
        </div>
        <div class="prob-bar-bg"><div class="prob-bar-red" style="width:{churn_prob}%;"></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider"><p class="section-title">🧾 Input Summary</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Feature": ["Age", "Frequent Flyer", "Annual Income", "Services Opted", "Social Media Sync", "Booked Hotel"],
        "Value":   [age, frequent_flyer, annual_income, services_opted, account_synced, booked_hotel]
    }), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>B.Tech – Gen AI</span> · 2nd Semester · Customer Churn Prediction · <span>Random Forest Model</span>
</div>
""", unsafe_allow_html=True)
