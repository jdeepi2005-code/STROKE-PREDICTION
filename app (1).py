import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# --- File Names and Model Settings ---
PKL_DATA_PATH = '/content/drive/MyDrive/Colab Notebooks/panda/stroke_data_processed.pkl'
FEATURE_COLUMNS = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level']
TARGET_COLUMN = 'stroke'

# --- Data Loading, Model Training, and Mapping Function ---
@st.cache_resource
def load_and_train_model(pkl_path):
    """Loads data from PKL, derives mappings, and trains the Logistic Regression model."""
    try:
        # Load the DataFrame from the PKL file
        with open(pkl_path, 'rb') as file:
            dia = pickle.load(file)

    except FileNotFoundError:
        st.error(f"Error: Data file not found. Ensure {pkl_path} is in the repository.")
        return None, None
    except Exception as e:
        st.error(f"Error loading PKL file: {e}")
        return None, None

    # --- Derive Mappings (CRUCIAL for translating UI input to model codes) ---
    # Since the original LabelEncoder objects were not saved, we will manually derive
    # the mappings from the unique encoded values present in the PKL data.

    # NOTE: These mappings are derived by assuming the LabelEncoder sorted the unique strings
    # alphabetically, which is the default behavior.

    # 1. Manually specify the original classes based on your notebook's data:
    original_gender_classes = ['Female', 'Male', 'Other']
    original_work_type_classes = ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed']
    original_residence_classes = ['Rural', 'Urban'] # Assuming 0=Rural, 1=Urban

    # 2. Create mock encoders to get the code for UI selection
    le_gender = LabelEncoder()
    le_gender.fit(original_gender_classes)

    le_work = LabelEncoder()
    le_work.fit(original_work_type_classes)

    le_residence = LabelEncoder()
    le_residence.fit(original_residence_classes)

    encoders = {
        'gender': le_gender,
        'work_type': le_work,
        'Residence_type': le_residence
    }

    # --- Train the Model on the Loaded PKL Data ---
    ind = dia[FEATURE_COLUMNS]
    dep = dia[TARGET_COLUMN]

    # Re-train the model
    Logr = LogisticRegression(max_iter=1000)
    Logr.fit(ind, dep)

    return Logr, encoders

MODEL, ENCODERS = load_and_train_model(PKL_DATA_PATH)


# --- Streamlit App Interface ---

st.title("🧠 Stroke Prediction App (Data from PKL)")
st.markdown("Enter the patient's data below to predict the likelihood of a stroke.")

if MODEL is not None:
    # --- Input Form ---
    with st.form("prediction_form"):
        st.header("Patient Vitals")

        # Get the class names for the Streamlit select boxes
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

        # Convert user inputs to numerical codes using the mock encoder objects
        input_data = {
            'gender': ENCODERS['gender'].transform([gender])[0],
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': 1 if ever_married == 'Yes' else 0, # Assuming binary mapping is simple
            'work_type': ENCODERS['work_type'].transform([work_type])[0],
            'Residence_type': ENCODERS['Residence_type'].transform([residence_type])[0],
            'avg_glucose_level': avg_glucose_level,
        }

        # --- Prediction Button ---
        submitted = st.form_submit_button("Predict Stroke")

    if submitted:
        # Prepare data for prediction (columns must be in the exact order)
        features = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

        # Predict
        prediction = MODEL.predict(features)[0]
        prediction_proba = MODEL.predict_proba(features)[0, 1]

        st.subheader("Prediction Result:")

        if prediction == 1:
            st.error(f"The model predicts a **HIGH** risk of stroke.")
            st.metric(label="Predicted Probability of Stroke", value=f"{prediction_proba*100:.2f}%")
        else:
            st.success(f"The model predicts a **LOW** risk of stroke.")
            st.metric(label="Predicted Probability of Stroke", value=f"{prediction_proba*100:.2f}%")

        st.warning("Disclaimer: This model is re-trained on every start using the PKL data and should not be used for actual medical diagnosis.")
else:
    st.error("Deployment Error: Model could not be trained. Check the presence of stroke_data_processed.pkl.")
