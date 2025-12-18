import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="💰",
    layout="centered"
)

# ---------------- CUSTOM CSS (DARK UI) ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.main {
    background-color: #0e1117;
}
.stButton>button {
    background-color: #f0b90b;
    color: black;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
.stNumberInput>div>div>input {
    background-color: #1e222d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("gold_price_model.pkl")

model = load_model()

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align: center;'>💰 Gold Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict gold prices using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- INPUTS ----------------
st.subheader("📊 Enter Market Details")

feature1 = st.number_input("Gold Open Price", min_value=0.0, format="%.2f")
feature2 = st.number_input("Gold High Price", min_value=0.0, format="%.2f")
feature3 = st.number_input("Gold Low Price", min_value=0.0, format="%.2f")
feature4 = st.number_input("Gold Volume", min_value=0.0, format="%.2f")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Gold Price"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)

    st.success(f"💵 Predicted Gold Price: ₹ {prediction[0]:,.2f}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:14px;'>Built with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
