import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 🎯 Load the Trained Model
# -----------------------------
model = joblib.load("gradient_boosting_model.pkl")

# -----------------------------
# 🎨 Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🛍️ Online Shoppers Purchase Prediction",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# 💠 Custom Styling
# -----------------------------
st.markdown("""
    <style>
        header {visibility: hidden;}
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }

        body {
            background-color: #f4f6f9;
        }

        .main {
            background-color: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
        }

        h1, h2, h3 {
            color: #1f4172;
        }

        .stButton button {
            background-color: #1f4172;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            border: none;
            transition: 0.3s ease;
        }

        .stButton button:hover {
            background-color: #2c5aa0;
            transform: scale(1.02);
        }

        .prediction {
            font-size: 22px;
            font-weight: 600;
            text-align: center;
            color: #1f4172;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 🏷️ Title and Description
# -----------------------------
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🛒 Online Shoppers Purchase Prediction App")
st.markdown("""
This app predicts whether an **online shopping session** will result in a **purchase (Revenue = True)**  
based on the visitor’s browsing behavior, engagement, and session characteristics.
""")

# -----------------------------
# 📊 Input Section
# -----------------------------
st.subheader("Enter Session Details:")

col1, col2, col3 = st.columns(3)

with col1:
    Administrative = st.number_input("Administrative Pages", 0, 30, 2)
    Informational = st.number_input("Informational Pages", 0, 20, 0)
    ProductRelated = st.number_input("Product Related Pages", 0, 500, 10)
    BounceRates = st.number_input("Bounce Rate", 0.0, 1.0, 0.02)
    
with col2:
    Administrative_Duration = st.number_input("Administrative Duration (sec)", 0.0, 6000.0, 80.0)
    Informational_Duration = st.number_input("Informational Duration (sec)", 0.0, 2000.0, 40.0)
    ProductRelated_Duration = st.number_input("Product Related Duration (sec)", 0.0, 20000.0, 1200.0)
    ExitRates = st.number_input("Exit Rate", 0.0, 1.0, 0.05)

with col3:
    PageValues = st.number_input("Page Values", 0.0, 200.0, 10.0)
    SpecialDay = st.slider("Special Day (0 → not close, 1 → close)", 0.0, 1.0, 0.0)
    Month = st.selectbox("Month", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    VisitorType = st.selectbox("Visitor Type", ['Returning_Visitor', 'New_Visitor'])
    Weekend = st.selectbox("Weekend", [False, True])

# Encode categorical values
month_map = {'Feb': 2, 'Mar': 3, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
Month_encoded = month_map[Month]
Visitor_encoded = 1 if VisitorType == 'Returning_Visitor' else 0
Weekend_encoded = 1 if Weekend else 0

# -----------------------------
# 🧩 Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    'Administrative': [Administrative],
    'Administrative_Duration': [Administrative_Duration],
    'Informational': [Informational],
    'Informational_Duration': [Informational_Duration],
    'ProductRelated': [ProductRelated],
    'ProductRelated_Duration': [ProductRelated_Duration],
    'BounceRates': [BounceRates],
    'ExitRates': [ExitRates],
    'PageValues': [PageValues],
    'SpecialDay': [SpecialDay],
    'Month': [Month_encoded],
    'VisitorType': [Visitor_encoded],
    'Weekend': [Weekend_encoded]
})

# Scale numeric values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(input_data)

# -----------------------------
# 🤖 Make Prediction
# -----------------------------
if st.button("🔮 Predict Purchase Probability"):
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    if prediction[0] == 1:
        st.markdown(f"<div class='prediction'>✅ **Prediction:** Likely to make a purchase<br>💰 Probability: {probability:.2%}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction'>❌ **Prediction:** Unlikely to make a purchase<br>💡 Probability: {probability:.2%}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

