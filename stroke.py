import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from warnings import filterwarnings

# Suppress the ConvergenceWarning from the Logistic Regression model
filterwarnings('ignore') 

# --- 1. Load Model and Encoders ---
try:
    # Load the trained model
    with open('logistic_model.pkl', 'rb') as file:
        Logr = pickle.load(file)

    # Load the Label Encoders dictionary
    with open('encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    
    st.sidebar.success("Model and Encoders loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or encoder files not found.")
    st.info("Please ensure 'logistic_model.pkl' and 'encoders.pkl' are in the same directory.")
    Logr, encoders = None, None

# --- 2. Streamlit App Layout ---

st.title("🧠 Stroke Prediction App")
st.markdown("Enter patient details below to predict the likelihood of a stroke.")
st.write("---")

if Logr and encoders:
    
    # --- 3. User Input Widgets ---

    col1, col2 = st.columns(2)

    # Input 1: Gender (Categorical - Encoded by LabelEncoder in notebook)
    gender_map = {0: 'Female', 1: 'Male', 2: 'Other'}
    gender_label = col1.selectbox("Gender", options=list(gender_map.values()))
    
    # Input 2: Age (Numerical)
    age = col2.slider("Age", min_value=0.08, max_value=82.0, value=40.0, step=0.1)

    # Input 3: Hypertension (Binary)
    hypertension = col1.selectbox("Hypertension", options=['No', 'Yes'])
    hyper_encoded = 1 if hypertension == 'Yes' else 0

    # Input 4: Heart Disease (Binary)
    heart_disease = col2.selectbox("Heart Disease", options=['No', 'Yes'])
    hd_encoded = 1 if heart_disease == 'Yes' else 0

    # Input 5: Ever Married (Categorical - Encoded by LabelEncoder in notebook)
    ever_married = col1.selectbox("Marital Status", options=['No', 'Yes'])
    
    # Input 6: Work Type (Categorical - Encoded by LabelEncoder in notebook)
    # Mapping based on your notebook's logic/dataset structure, likely:
    # 0: Children, 1: Gov_job, 2: Private, 3: Self-employed, 4: Never_worked
    work_options = {
        'Private': 2, 'Self-employed': 3, 'Govt_job': 1, 'children': 0, 'Never_worked': 4
    }
    work_type = col2.selectbox("Work Type", options=list(work_options.keys()))

    # Input 7: Residence Type (Categorical - Encoded by LabelEncoder in notebook)
    residence_map = {0: 'Rural', 1: 'Urban'}
    residence_type = col1.selectbox("Residence Type", options=list(residence_map.values()))

    # Input 8: Average Glucose Level (Numerical)
    avg_glucose_level = col2.number_input(
        "Average Glucose Level (mg/dL)", 
        min_value=50.0, max_value=300.0, value=90.0, step=1.0
    )

    # --- 4. Prediction Logic ---

    if st.button("Predict Stroke Risk"):
        
        # 4a. Encode Categorical Inputs using the loaded encoders

        # Encode 'gender'
        gender_encoder = encoders['gender']
        # The 'Other' option has code 2. Your notebook's label encoding mapped: Male=1, Female=0, Other=2.
        # We need to use the stored encoder to map the string back to the correct integer.
        try:
             gender_encoded = gender_encoder.transform([gender_label])[0]
        except ValueError:
            # Handle the case where the user input is 'Other', but it wasn't in the original fit list
            # Your notebook code handles 'Other', but for robustness, we'll ensure we map it correctly.
            # Assuming 'Other' is mapped to the highest integer (2) in the saved encoder.
            if gender_label == 'Other':
                gender_encoded = 2 
            else:
                 # This should ideally not happen if options are controlled by selectbox
                 st.error("Invalid gender selection.")
                 st.stop()


        # Encode 'ever_married' (Yes/No)
        ever_married_encoder = encoders['ever_married']
        ever_married_encoded = ever_married_encoder.transform([ever_married])[0]

        # Encode 'work_type'
        work_type_encoded = work_options[work_type] # Using the hardcoded map derived from exploration

        # Encode 'Residence_type'
        residence_encoder = encoders['Residence_type']
        residence_type_encoded = residence_encoder.transform([residence_type])[0]
        
        
        # 4b. Create a DataFrame for prediction
        input_data = pd.DataFrame([[
            gender_encoded,
            age,
            hyper_encoded,
            hd_encoded,
            ever_married_encoded,
            work_type_encoded,
            residence_type_encoded,
            avg_glucose_level
        ]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level'])

        # 4c. Make Prediction
        prediction = Logr.predict(input_data)[0]
        prediction_proba = Logr.predict_proba(input_data)[0][1]

        st.write("---")
        
        # 4d. Display Results
        if prediction == 1:
            st.error(f"**Prediction:** Patient is at risk of having a stroke (Risk Probability: **{prediction_proba:.2%}**)")
            st.markdown("⚠️ **Disclaimer:** This is a prediction based on a Logistic Regression model and should **not** replace professional medical advice.")
        else:
            st.success(f"**Prediction:** Patient is currently predicted not to have a stroke (Risk Probability: **{prediction_proba:.2%}**)")
            st.markdown("✅ **Disclaimer:** This is a prediction based on a Logistic Regression model and should **not** replace professional medical advice.")

else:
    st.warning("Prediction functionality is disabled due to missing model files.")
