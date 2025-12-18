import streamlit as st
import numpy as np
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Gold Loan Price Prediction",
    page_icon="💰",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return joblib.load("gold_price_model.pkl")

model = load_model()

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background-color: #0f172a;
}
.card {
    background-color: #020617;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(255, 215, 0, 0.15);
    margin-top: 20px;
}
.title {
    text-align: center;
    color: #facc15;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
}
.result {
    font-size: 28px;
    font-weight: bold;
    color: #22c55e;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<h1 class='title'>💰 Gold Loan Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Machine Learning based Gold Value Estimator</p>", unsafe_allow_html=True)

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("🔧 Enter Details")

gold_weight = st.sidebar.number_input(
    "Gold Weight (grams)",
    min_value=1.0,
    max_value=1000.0,
    step=0.5
)

gold_purity = st.sidebar.selectbox(
    "Gold Purity (Carat)",
    [18, 20, 22, 24]
)

interest_rate = st.sidebar.slider(
    "Interest Rate (%)",
    min_value=5.0,
    max_value=25.0,
    step=0.5
)

loan_tenure = st.sidebar.slider(
    "Loan Tenure (Months)",
    min_value=3,
    max_value=36
)

# -------------------- PREDICTION --------------------
if st.sidebar.button("🔮 Predict Gold Loan Value"):
    try:
        input_data = np.array([[gold_weight, gold_purity, interest_rate, loan_tenure]])
        prediction = model.predict(input_data)[0]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subtitle'>Estimated Loan Amount</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='result'>₹ {prediction:,.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

# -------------------- FOOTER --------------------
st.markdown("""
<hr>
<p style="text-align:center; color:#94a3b8;">
Developed using Streamlit & Machine Learning 🚀
</p>
""", unsafe_allow_html=True)
