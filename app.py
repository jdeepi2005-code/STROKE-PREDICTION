
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# --- 1. File Names and Feature Order ---
MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/panda/logistic_model.pkl'
ENCODER_PATH = '/content/drive/MyDrive/Colab Notebooks/panda/encoders.pkl'
FEATURE_COLUMNS = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level']
TARGET_COLUMN = 'stroke'

# --- 2. Model and Asset Loading Function ---
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and fitted encoders from PKL files."""
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        with open(ENCODER_PATH, 'rb') as file:
            encoders = pickle.load(file)
        return model, encoders
    except FileNotFoundError:
        st.error(f"Deployment Error: Model or encoder file not found. Ensure {MODEL_PATH} and {ENCODER_PATH} are in the repository.")
        return None, None

MODEL, ENCODERS = load_assets()

# --- 3. Streamlit App Interface ---
st.title("🧠 Stroke Prediction App (Logistic Regression)")
st.markdown("Enter the patient's data to predict the risk of stroke. This app uses a pre-trained model.")

if MODEL is not None:
    with st.form("prediction_form"):
        st.header("Patient Vitals")

        gender_options = ENCODERS['gender'].classes_
        work_type_options = ENCODERS['work_type'].classes_
        residence_type_options = ENCODERS['Residence_type'].classes_

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", gender_options)
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            age = st.slider("Age", 0, 100, 50)
            avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", value=100.0, step=0.1)

        with col2:
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
            work_type = st.selectbox("Work Type", work_type_options)
            residence_type = st.selectbox("Residence Type", residence_type_options)

        submitted = st.form_submit_button("Predict Stroke")

    if submitted:
        # --- 4. Preprocessing User Input ---
        input_data = {
            'gender': ENCODERS['gender'].transform([gender])[0],
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': 1 if ever_married == 'Yes' else 0,
            'work_type': ENCODERS['work_type'].transform([work_type])[0],
            'Residence_type': ENCODERS['Residence_type'].transform([residence_type])[0],
            'avg_glucose_level': avg_glucose_level,
        }

        features = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

        # --- 5. Prediction and Output ---
        prediction = MODEL.predict(features)[0]
        prediction_proba = MODEL.predict_proba(features)[0, 1]

        st.subheader("Prediction Result:")

        if prediction == 1:
            st.error(f"The model predicts a **HIGH** risk of stroke.")
            st.metric(label="Predicted Probability of Stroke", value=f"{prediction_proba*100:.2f}%")
        else:
            st.success(f"The model predicts a **LOW** risk of stroke.")
            st.metric(label="Predicted Probability of Stroke", value=f"{prediction_proba*100:.2f}%")

        st.warning("Disclaimer: This is a machine learning prediction for demonstration purposes only.")
else:
    st.error("Application Initialization Failed: Please check that `logistic_model.pkl` and `encoders.pkl` were uploaded to your GitHub repository.")
