import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Chronic Kidney Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('ckd_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'ckd_model.pkl' not found. Please ensure the model file is in the same directory.")
        return None

# Load model
model = load_model()

# Title and description
st.title("Chronic Kidney Disease Prediction System")
st.markdown("---")
st.markdown("""
This application uses machine learning to predict the likelihood of chronic kidney disease based on various medical parameters. 
Please enter the patient's medical information below to get a prediction.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Prediction", "About the Model", "Dataset Info"])

if page == "Prediction":
    if model is not None:
        st.header("Patient Information Input")
        st.markdown("**Required fields are marked with * - Optional fields can be left as default values**")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 Essential Information")
            age = st.number_input("Age* (years)", min_value=1, max_value=120, value=50, help="Age is a critical factor in CKD prediction")
            blood_pressure = st.number_input("Blood Pressure* (mm Hg)", min_value=50, max_value=200, value=120, help="Systolic blood pressure is crucial for CKD prediction")
            specific_gravity = st.number_input("Specific Gravity* (urine)", min_value=1.005, max_value=1.025, value=1.020, step=0.005, help="Important kidney function indicator")
            
            st.subheader("🔴 Critical Blood Tests")
            serum_creatinine = st.number_input("Serum Creatinine* (mgs/dl)", min_value=0.5, max_value=15.0, value=1.2, step=0.1, help="Most important kidney function marker")
            haemoglobin = st.number_input("Haemoglobin* (gms)", min_value=5.0, max_value=20.0, value=12.5, step=0.1, help="Critical for CKD detection")
            blood_urea = st.number_input("Blood Urea* (mgs/dl)", min_value=10, max_value=200, value=40, help="Key kidney function indicator")
            
            st.subheader("🔴 Essential Urine Tests")
            albumin = st.selectbox("Albumin* (protein in urine)", [0, 1, 2, 3, 4, 5], help="Proteinuria is a key CKD indicator")
            sugar = st.selectbox("Sugar* (in urine)", [0, 1, 2, 3, 4, 5], help="Important for diabetes-related kidney disease")
            
            st.subheader("🔴 Medical History")
            diabetes_mellitus = st.selectbox("Diabetes Mellitus*", ["No", "Yes"], help="Major risk factor for CKD")
            hypertension = st.selectbox("Hypertension*", ["No", "Yes"], help="Critical risk factor for CKD progression")
        
        with col2:
            # Optional section with expander
            with st.expander("🟡 Optional Additional Tests", expanded=False):
                st.markdown("*These can help improve accuracy but are not essential*")
                
                st.subheader("Additional Blood Tests")
                blood_glucose_random = st.number_input("Blood Glucose Random (mgs/dl)", min_value=50, max_value=500, value=120)
                sodium = st.number_input("Sodium (mEq/L)", min_value=120, max_value=160, value=140)
                potassium = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=8.0, value=4.5, step=0.1)
                packed_cell_volume = st.number_input("Packed Cell Volume", min_value=20, max_value=60, value=40)
                white_blood_cell_count = st.number_input("White Blood Cell Count (cells/cumm)", min_value=3000, max_value=20000, value=7500)
                red_blood_cell_count = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=2.0, max_value=8.0, value=4.5, step=0.1)
                
                st.subheader("Additional Urine Tests")
                red_blood_cells = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
                pus_cell = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
                pus_cell_clumps = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
                bacteria = st.selectbox("Bacteria", ["Not Present", "Present"])
                
                st.subheader("Additional Medical History & Symptoms")
                coronary_artery_disease = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
                appetite = st.selectbox("Appetite", ["Good", "Poor"])
                peda_edema = st.selectbox("Pedal Edema", ["No", "Yes"])
                anemia = st.selectbox("Anemia", ["No", "Yes"])
            
            # Quick prediction option
            st.markdown("---")
            st.subheader("🚀 Quick Prediction")
            st.markdown("*Use only the essential parameters for a fast screening*")
            quick_predict = st.checkbox("Use Quick Prediction (Essential parameters only)", help="This will use default values for optional parameters")
        
        # Convert categorical inputs to numerical
        def convert_to_numeric(value_dict):
            conversions = {
                'red_blood_cells': {'Normal': 1, 'Abnormal': 0},
                'pus_cell': {'Normal': 1, 'Abnormal': 0},
                'pus_cell_clumps': {'Not Present': 0, 'Present': 1},
                'bacteria': {'Not Present': 0, 'Present': 1},
                'hypertension': {'No': 0, 'Yes': 1},
                'diabetes_mellitus': {'No': 0, 'Yes': 1},
                'coronary_artery_disease': {'No': 0, 'Yes': 1},
                'appetite': {'Good': 1, 'Poor': 0},
                'peda_edema': {'No': 0, 'Yes': 1},
                'anemia': {'No': 0, 'Yes': 1}
            }
            
            numeric_dict = {}
            for key, value in value_dict.items():
                if key in conversions:
                    numeric_dict[key] = conversions[key][value]
                else:
                    numeric_dict[key] = value
            return numeric_dict
        
        # Predict button
        predict_button_text = "Quick Predict (Essential Only)" if quick_predict else "Predict Kidney Disease (All Parameters)"
        if st.button(predict_button_text, type="primary"):
            # Prepare input data
            if quick_predict:
                # Use default values for optional parameters
                input_data = {
                    'age': age,
                    'blood_pressure': blood_pressure,
                    'specific_gravity': specific_gravity,
                    'albumin': albumin,
                    'sugar': sugar,
                    'red_blood_cells': "Normal",  # Default
                    'pus_cell': "Normal",  # Default
                    'pus_cell_clumps': "Not Present",  # Default
                    'bacteria': "Not Present",  # Default
                    'blood_glucose_random': 120,  # Default
                    'blood_urea': blood_urea,
                    'serum_creatinine': serum_creatinine,
                    'sodium': 140,  # Default
                    'potassium': 4.5,  # Default
                    'haemoglobin': haemoglobin,
                    'packed_cell_volume': 40,  # Default
                    'white_blood_cell_count': 7500,  # Default
                    'red_blood_cell_count': 4.5,  # Default
                    'hypertension': hypertension,
                    'diabetes_mellitus': diabetes_mellitus,
                    'coronary_artery_disease': "No",  # Default
                    'appetite': "Good",  # Default
                    'peda_edema': "No",  # Default
                    'anemia': "No"  # Default
                }
            else:
                input_data = {
                    'age': age,
                    'blood_pressure': blood_pressure,
                    'specific_gravity': specific_gravity,
                    'albumin': albumin,
                    'sugar': sugar,
                    'red_blood_cells': red_blood_cells,
                    'pus_cell': pus_cell,
                    'pus_cell_clumps': pus_cell_clumps,
                    'bacteria': bacteria,
                    'blood_glucose_random': blood_glucose_random,
                    'blood_urea': blood_urea,
                    'serum_creatinine': serum_creatinine,
                    'sodium': sodium,
                    'potassium': potassium,
                    'haemoglobin': haemoglobin,
                    'packed_cell_volume': packed_cell_volume,
                    'white_blood_cell_count': white_blood_cell_count,
                    'red_blood_cell_count': red_blood_cell_count,
                    'hypertension': hypertension,
                    'diabetes_mellitus': diabetes_mellitus,
                    'coronary_artery_disease': coronary_artery_disease,
                    'appetite': appetite,
                    'peda_edema': peda_edema,
                    'anemia': anemia
                }
            
            # Convert to numeric
            numeric_data = convert_to_numeric(input_data)
            
            # Create DataFrame with proper column order
            feature_names = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
                           'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
                           'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                           'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
                           'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
                           'coronary_artery_disease', 'appetite', 'peda_edema', 'anemia']
            
            input_df = pd.DataFrame([numeric_data], columns=feature_names)
            
            try:
                # Make prediction
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
                
                # Display results
                st.markdown("---")
                st.header("Prediction Results")
                
                if prediction == 1:
                    st.error("**Prediction: Chronic Kidney Disease Detected**")
                    st.markdown("The model indicates a high likelihood of chronic kidney disease. Please consult with a healthcare professional immediately for proper diagnosis and treatment.")
                else:
                    st.success("**Prediction: No Chronic Kidney Disease**")
                    st.markdown("The model indicates a low likelihood of chronic kidney disease. However, regular health check-ups are always recommended.")
                
                if prediction_proba is not None:
                    st.subheader("Prediction Confidence")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("No CKD Probability", f"{prediction_proba[0]:.2%}")
                    with col2:
                        st.metric("CKD Probability", f"{prediction_proba[1]:.2%}")
                
                # Display input summary
                st.subheader("Input Summary")
                if quick_predict:
                    st.info("🚀 Quick Prediction Mode: Only essential parameters were used. Default values applied to optional fields.")
                    # Show only essential parameters
                    essential_params = {
                        'Age': age,
                        'Blood Pressure': blood_pressure,
                        'Specific Gravity': specific_gravity,
                        'Albumin': albumin,
                        'Sugar': sugar,
                        'Serum Creatinine': serum_creatinine,
                        'Haemoglobin': haemoglobin,
                        'Blood Urea': blood_urea,
                        'Diabetes Mellitus': diabetes_mellitus,
                        'Hypertension': hypertension
                    }
                    summary_df = pd.DataFrame(list(essential_params.items()), columns=['Essential Parameter', 'Value'])
                else:
                    summary_df = pd.DataFrame(list(input_data.items()), columns=['Parameter', 'Value'])
                st.dataframe(summary_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.markdown("Please check that all input values are valid and try again.")

elif page == "About the Model":
    st.header("About the Prediction Model")
    
    st.markdown("""
    ### Model Overview
    This chronic kidney disease prediction system uses a machine learning model trained on medical data 
    to assess the likelihood of chronic kidney disease based on various clinical parameters.
    
    ### Features Used
    The model considers the following 24 medical parameters:
    
    **Basic Information:**
    - Age
    - Blood Pressure
    - Specific Gravity
    
    **Urine Tests:**
    - Albumin levels
    - Sugar levels
    - Red Blood Cells presence
    - Pus Cell presence
    - Pus Cell Clumps
    - Bacteria presence
    
    **Blood Tests:**
    - Blood Glucose Random
    - Blood Urea
    - Serum Creatinine
    - Sodium levels
    - Potassium levels
    - Haemoglobin levels
    - Packed Cell Volume
    - White Blood Cell Count
    - Red Blood Cell Count
    
    **Medical History:**
    - Hypertension
    - Diabetes Mellitus
    - Coronary Artery Disease
    
    **Symptoms:**
    - Appetite
    - Pedal Edema
    - Anemia
    
    ### Important Disclaimer
    This tool is for educational and screening purposes only. It should not be used as a substitute 
    for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
    professionals for proper medical evaluation and treatment decisions.
    """)

elif page == "Dataset Info":
    st.header("Dataset Information")
    
    st.markdown("""
    ### Dataset Overview
    The model was trained on a chronic kidney disease dataset containing medical records 
    with various clinical parameters and their corresponding diagnoses.
    
    ### Dataset Characteristics
    - **Total Features:** 24 clinical parameters
    - **Target Variable:** Binary classification (CKD vs No CKD)
    - **Data Types:** Mixed (numerical and categorical)
    - **Preprocessing:** Missing value imputation, outlier handling, feature scaling
    
    ### Data Preprocessing Steps
    1. **Missing Value Treatment:** Numerical features filled with mean, categorical with mode
    2. **Outlier Detection:** IQR method applied to remove extreme values
    3. **Feature Encoding:** Categorical variables converted to numerical format
    4. **Scaling:** StandardScaler applied for model optimization
    
    ### Model Performance Metrics
    The model's performance was evaluated using multiple metrics including:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix analysis
    
    ### Quality Assurance
    - Cross-validation performed to ensure model reliability
    - Multiple algorithms compared to select the best performing model
    - Feature importance analysis conducted
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
<small>Chronic Kidney Disease Prediction System | For Educational and Screening Purposes Only</small>
</div>
""", unsafe_allow_html=True)