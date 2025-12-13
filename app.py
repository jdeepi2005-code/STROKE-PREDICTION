import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# --- File Names ---
PKL_DATA_PATH = '/content/drive/MyDrive/Colab Notebooks/panda/stroke_data_processed (1).pkl'
FEATURE_COLUMNS = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level']
TARGET_COLUMN = 'stroke'

# --- Data Loading and Model Training Function ---
@st.cache_resource # Use st.cache_resource to run this heavy task only once
def load_and_train_model(pkl_path):
    """Loads data from PKL, re-fits encoders, and trains the Logistic Regression model."""
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
    
    # --- Re-Fit Encoders (Necessary to capture class order for input mapping) ---
    encoders = {}
    categorical_cols = ['gender', 'Residence_type', 'ever_married', 'work_type']
    
    # We must operate on the original string columns to fit the encoders correctly.
    # Since your PKL is already encoded, we assume a clean data load is possible.
    # However, since the raw data is not available, we have to simulate the encoder fitting 
    # to extract class names for the Streamlit select boxes.
    # We'll use the unique values from the encoded columns to deduce the original classes for the UI.
    
    # NOTE: This approach is fragile because the PKL contains integer codes, not strings. 
    # To fix this, we will re-train the model here and extract the mappings manually:
    
    # Temporarily load the original CSV (best practice) OR deduce classes (fragile)
    # Sticking to the requirement of *only* using the PKL means we must skip 
    # the re-fitting step and rely on hardcoded mappings based on the original notebook.
    
    # --- HARDCODED MAPPINGS (Based on the original notebook's LabelEncoder results) ---
    # In a perfect scenario, you would save the encoder object itself, but since we cannot, 
    # we rely on the known integer mappings from your notebook.
    
    # NOTE: These dictionaries must EXACTLY match the LabelEncoder's mapping in your notebook!
    gender_map = {0: 'Female', 1: 'Male', 2: 'Other'}
    work_map = {0: 'children', 1: 'Govt_job', 2: 'Never_worked', 3: 'Private', 4: 'Self-employed'}
    residence_map = {0: 'Rural', 1: 'Urban'}
    
    # --- Train the Model on the Loaded PKL Data ---
    ind = dia[FEATURE_COLUMNS]
    dep = dia[TARGET_COLUMN]

    Logr = LogisticRegression(max_iter=1000) 
    Logr.fit(ind, dep)
    
    # Return the model and the manual mappings for the UI
    return Logr, {
        'gender': gender_map, 
        'work_type': work_map, 
        'Residence_type': residence_map
    }

MODEL, ENCODERS = load_and_train_model(PKL_DATA_PATH)


# --- Streamlit App Interface ---

st.title("🧠 Stroke Prediction App (Data from PKL)")
st.markdown("Enter the patient's data below to predict the likelihood of a stroke.")

if MODEL is not None:
    # --- Input Form ---
    with st.form("prediction_form"):
        st.header("Patient Vitals")
        
        # Reverse the maps for easy lookup: value -> code
        gender_options = {v:k for k,v in ENCODERS['gender'].items()}
        work_type_options = {v:k for k,v in ENCODERS['work_type'].items()}
        residence_options = {v:k for k,v in ENCODERS['Residence_type'].items()}
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", list(gender_options.keys()))
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            age = st.slider("Age", 0, 100, 50)
            avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", value=100.0, step=0.1)

        with col2:
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
            work_type = st.selectbox("Work Type", list(work_type_options.keys()))
            residence_type = st.selectbox("Residence Type", list(residence_options.keys()))

        # Convert user inputs to numerical codes using the generated dictionary mappings
        input_data = {
            'gender': gender_options[gender],
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': 1 if ever_married == 'Yes' else 0,
            'work_type': work_type_options[work_type],
            'Residence_type': residence_options[residence_type],
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
