import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder  # For loading encoders if needed

# -----------------------------
# 1️⃣ Load trained assets
# -----------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_scaler(path):
    return joblib.load(path)

@st.cache_resource
def load_encoder(path):
    return joblib.load(path)

model = load_model('gradient_boosting_model.pkl')
scaler = load_scaler('scaler.pkl')
month_encoder = load_encoder('month_encoder.pkl')
visitor_type_encoder = load_encoder('visitor_type_encoder.pkl')

# -----------------------------
# 2️⃣ Define feature columns
# -----------------------------
feature_columns = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'
]

numerical_cols_for_scaler = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]

# -----------------------------
# 3️⃣ Streamlit Page Layout
# -----------------------------
st.set_page_config(page_title="Online Shopper Purchase Prediction 🛒", layout="wide")
st.title('🛒 Online Shopper Purchase Prediction')
st.markdown('Predict whether a user session will result in a purchase based on session features.')

# -----------------------------
# 4️⃣ Input Widgets in Page Body
# -----------------------------
st.subheader("Enter Session Details:")

col1, col2, col3 = st.columns(3)

with col1:
    administrative = st.number_input('Administrative Page Visits', min_value=0, max_value=50, value=2)
    administrative_duration = st.number_input('Administrative Duration (seconds)', min_value=0.0, max_value=5000.0, value=80.0)
    informational = st.number_input('Informational Page Visits', min_value=0, max_value=30, value=0)
    informational_duration = st.number_input('Informational Duration (seconds)', min_value=0.0, max_value=3000.0, value=40.0)

with col2:
    product_related = st.number_input('Product Related Page Visits', min_value=0, max_value=1000, value=10)
    product_related_duration = st.number_input('Product Related Duration (seconds)', min_value=0.0, max_value=70000.0, value=1200.0)
    bounce_rates = st.number_input('Bounce Rates', min_value=0.0, max_value=1.0, value=0.02, format="%.4f")
    exit_rates = st.number_input('Exit Rates', min_value=0.0, max_value=1.0, value=0.05, format="%.4f")

with col3:
    page_values = st.number_input('Page Values', min_value=0.0, max_value=400.0, value=10.0, format="%.2f")
    special_day = st.number_input('Special Day (Closeness 0-1)', min_value=0.0, max_value=1.0, value=0.0, format="%.1f")
    operating_systems = st.slider('Operating System', min_value=1, max_value=8, value=2)
    browser = st.slider('Browser', min_value=1, max_value=13, value=2)
    region = st.slider('Region', min_value=1, max_value=9, value=1)
    traffic_type = st.slider('Traffic Type', min_value=1, max_value=20, value=1)

month_options = month_encoder.classes_.tolist()
month_selected = st.selectbox('Month', options=month_options, index=month_options.index('Feb'))

visitor_type_options = visitor_type_encoder.classes_.tolist()
visitor_type_selected = st.selectbox('Visitor Type', options=visitor_type_options, index=visitor_type_options.index('Returning_Visitor'))

weekend = st.checkbox('Weekend Session')

# -----------------------------
# 5️⃣ Prediction Logic
# -----------------------------
if st.button('Predict Purchase Intention'):
    # Encode categorical features
    month_encoded = month_encoder.transform([month_selected])[0]
    visitor_type_encoded = visitor_type_encoder.transform([visitor_type_selected])[0]
    weekend_encoded = 1 if weekend else 0

    # Prepare DataFrame for prediction
    input_data = pd.DataFrame([{
        'Administrative': administrative,
        'Administrative_Duration': administrative_duration,
        'Informational': informational,
        'Informational_Duration': informational_duration,
        'ProductRelated': product_related,
        'ProductRelated_Duration': product_related_duration,
        'BounceRates': bounce_rates,
        'ExitRates': exit_rates,
        'PageValues': page_values,
        'SpecialDay': special_day,
        'Month': month_encoded,
        'OperatingSystems': operating_systems,
        'Browser': browser,
        'Region': region,
        'TrafficType': traffic_type,
        'VisitorType': visitor_type_encoded,
        'Weekend': weekend_encoded
    }])

    # Ensure correct column order
    input_data = input_data[feature_columns]

    # Scale numerical features only
    input_data[numerical_cols_for_scaler] = scaler.transform(input_data[numerical_cols_for_scaler])

    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.success(f'This session is predicted to result in a purchase! 🎉\n💰 Probability: {probability:.2%}')
    else:
        st.info(f'This session is predicted NOT to result in a purchase.\n💡 Probability: {probability:.2%}')
