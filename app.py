import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder # Although LabelEncoder itself is not used directly, it's good practice to import if its objects are loaded

# --- 1. Load trained assets ---
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

# --- 2. Define Feature Columns ---
feature_columns = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'
]

# Define numerical columns that need scaling
numerical_cols_for_scaler = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]

# --- 3. Streamlit App Layout ---
st.title('Online Shopper Intention Prediction 🛒')
st.markdown('Predict whether a user session will result in a purchase.')

# --- 4. Input Widgets ---
st.sidebar.header('User Session Features')

# Numerical Inputs
administrative = st.sidebar.number_input('Administrative Page Visits', min_value=0, max_value=50, value=2)
administrative_duration = st.sidebar.number_input('Administrative Duration (seconds)', min_value=0.0, max_value=5000.0, value=80.0)
informational = st.sidebar.number_input('Informational Page Visits', min_value=0, max_value=30, value=0)
informational_duration = st.sidebar.number_input('Informational Duration (seconds)', min_value=0.0, max_value=3000.0, value=40.0)
product_related = st.sidebar.number_input('Product Related Page Visits', min_value=0, max_value=1000, value=10)
product_related_duration = st.sidebar.number_input('Product Related Duration (seconds)', min_value=0.0, max_value=70000.0, value=1200.0)
bounce_rates = st.sidebar.number_input('Bounce Rates', min_value=0.0, max_value=1.0, value=0.02, format="%.4f")
exit_rates = st.sidebar.number_input('Exit Rates', min_value=0.0, max_value=1.0, value=0.05, format="%.4f")
page_values = st.sidebar.number_input('Page Values', min_value=0.0, max_value=400.0, value=10.0, format="%.2f")
special_day = st.sidebar.number_input('Special Day (Closeness 0-1)', min_value=0.0, max_value=1.0, value=0.0, format="%.1f")

# Other integer/categorical inputs
operating_systems = st.sidebar.slider('Operating System', min_value=1, max_value=8, value=2)
browser = st.sidebar.slider('Browser', min_value=1, max_value=13, value=2)
region = st.sidebar.slider('Region', min_value=1, max_value=9, value=1)
traffic_type = st.sidebar.slider('Traffic Type', min_value=1, max_value=20, value=1)

# Categorical Inputs for Month and VisitorType using their original categories
month_options = month_encoder.classes_.tolist()
month_selected = st.sidebar.selectbox('Month', options=month_options, index=month_options.index('Feb'))

visitor_type_options = visitor_type_encoder.classes_.tolist()
visitor_type_selected = st.sidebar.selectbox('Visitor Type', options=visitor_type_options, index=visitor_type_options.index('Returning_Visitor'))

weekend = st.sidebar.checkbox('Weekend Session')

# --- 5. Prediction Logic ---
if st.button('Predict Purchase Intention'):
    # Preprocess inputs
    month_encoded = month_encoder.transform([month_selected])[0]
    visitor_type_encoded = visitor_type_encoder.transform([visitor_type_selected])[0]
    weekend_encoded = 1 if weekend else 0

    # Create DataFrame for prediction
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
    
    # Ensure column order matches training data
    input_data = input_data[feature_columns]

    # Apply scaler to numerical columns only
    input_data[numerical_cols_for_scaler] = scaler.transform(input_data[numerical_cols_for_scaler])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.success('This session is predicted to result in a purchase! 🎉')
    else:
        st.info('This session is predicted NOT to result in a purchase.')
