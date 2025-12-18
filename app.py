import streamlit as st
import joblib
import numpy as np
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="💰",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "gold_price_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
}
h1, h2, h3, h4 {
    color: #FFD700;
}
.stButton>button {
    background-color: #FFD700;
    color: black;
    border-radius: 8px;
    height: 45px;
    font-size: 16px;
}
.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("💰 Gold Price Prediction App")
st.write("Predict gold prices using a Machine Learning model")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔢 Input Parameters")

open_price = st.sidebar.number_input("Open Price", min_value=0.0, step=1.0)
high_price = st.sidebar.number_input("High Price", min_value=0.0, step=1.0)
low_price = st.sidebar.number_input("Low Price", min_value=0.0, step=1.0)
volume = st.sidebar.number_input("Volume", min_value=0.0, step=1.0)

# ---------------- MAIN ----------------
st.subheader("📊 Enter Values and Predict")

input_data = np.array([[open_price, high_price, low_price, volume]])

if st.button("Predict Gold Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success("✅ Prediction Successful")
        st.metric(label="Predicted Gold Price", value=f"₹ {prediction:,.2f}")
    except Exception as e:
        st.error("❌ Error while predicting")
        st.write(e)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Sharim | ML & Streamlit Project")
