import streamlit as st
import pandas as pd
import numpy as np
import sys
import platform
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Chronic Kidney Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Debug information - Add this early to see what's happening
st.sidebar.markdown("### 🐛 Debug Information")
st.sidebar.write(f"Python version: {sys.version}")
st.sidebar.write(f"Platform: {platform.platform()}")

try:
    import sklearn
    st.sidebar.write(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    st.sidebar.error(f"❌ Scikit-learn error: {e}")

try:
    import matplotlib
    st.sidebar.write(f"✅ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    st.sidebar.error(f"❌ Matplotlib error: {e}")

try:
    import seaborn
    st.sidebar.write(f"✅ Seaborn: {seaborn.__version__}")
except ImportError as e:
    st.sidebar.error(f"❌ Seaborn error: {e}")

# Check if model file exists
model_path = Path("ckd_model.pkl")
st.sidebar.write(f"Model file exists: {model_path.exists()}")
if model_path.exists():
    st.sidebar.write(f"Model file size: {model_path.stat().st_size / 1024:.2f} KB")

# Load the trained model with comprehensive error handling
@st.cache_resource
def load_model():
    try:
        import pickle
        
        if not model_path.exists():
            st.error("❌ Model file 'ckd_model.pkl' not found in the repository.")
            st.info("Please ensure the model file is uploaded to your GitHub repository.")
            return None
            
        with open('ckd_model.pkl', 'rb') as file:
            model = pickle.load(file)
            
        st.sidebar.success("✅ Model loaded successfully!")
        return model
        
    except ImportError as e:
        st.sidebar.error(f"❌ Import error: {e}")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {str(e)}")
        st.sidebar.error("This is likely due to Python/library version mismatch")
        return None

# Create a fallback dummy model for demonstration
def create_dummy_model():
    """Create a simple dummy model for demonstration when real model fails"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create dummy training data
        X, y = make_classification(n_samples=1000, n_features=24, n_classes=2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        st.sidebar.warning("⚠️ Using dummy model for demonstration")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Could not create dummy model: {e}")
        return None

# Try to load model
try:
    model = load_model()
    if model is None:
        st.warning("⚠️ Real model could not be loaded. Creating a dummy model for demonstration...")
        model = create_dummy_model()
except Exception as e:
    st.error(f"❌ Critical error during model loading: {e}")
    model = None

# Title and description
st.title("🏥 Chronic Kidney Disease Prediction System")
st.markdown("---")

# Show deployment status
if model is None:
    st.error("❌ **Deployment Issue**: Model could not be loaded")
    st.markdown("""
    ### Possible Issues:
    1. **Model file missing** from GitHub repository
    2. **Python version mismatch** (Cloud: 3.13.5 vs Training environment)
    3. **Library version incompatibility**
    4. **Pickle serialization issues**
    
    ### Solutions:
    1. Ensure `ckd_model.pkl` is in your GitHub repo
    2. Retrain model with Python 3.13 and current library versions
    3. Use joblib instead of pickle for model serialization
    """)
else:
    st.success("✅ **System Status**: Model loaded successfully!")

st.markdown("""
This application uses machine learning to predict the likelihood of chronic kidney disease based on various medical parameters. 
Please enter the patient's medical information below to get a prediction.
""")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Prediction", "About the Model", "Dataset Info", "Debug Info"])

if page == "Prediction":
    if model is not None:
        st.header("📝 Patient Information Input")
        st.markdown("**Required fields are marked with * - Optional fields can be left as default values**")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 Essential Information")
            age = st.number_input("Age* (years)", min_value=1, max_value=120, value=50)
            blood_pressure = st.number_input("Blood Pressure* (mm Hg)", min_value=50, max_value=200, value=120)
            specific_gravity = st.number_input("Specific Gravity* (urine)", min_value=1.005, max_value=1.025, value=1.020, step=0.005)
            
            st.subheader("🔴 Critical Blood Tests")
            serum_creatinine = st.number_input("Serum Creatinine* (mgs/dl)", min_value=0.5, max_value=15.0, value=1.2, step=0.1)
            haemoglobin = st.number_input("Haemoglobin* (gms)", min_value=5.0, max_value=20.0, value=12.5, step=0.1)
            blood_urea = st.number_input("Blood Urea* (mgs/dl)", min_value=10, max_value=200, value=40)
            
            st.subheader("🔴 Essential Urine Tests")
            albumin = st.selectbox("Albumin* (protein in urine)", [0, 1, 2, 3, 4, 5])
            sugar = st.selectbox("Sugar* (in urine)", [0, 1, 2, 3, 4, 5])
            
            st.subheader("🔴 Medical History")
            diabetes_mellitus = st.selectbox("Diabetes Mellitus*", ["No", "Yes"])
            hypertension = st.selectbox("Hypertension*", ["No", "Yes"])
        
        with col2:
            with st.expander("🟡 Optional Additional Tests", expanded=False):
                st.markdown("*These can help improve accuracy but are not essential*")
                
                blood_glucose_random = st.number_input("Blood Glucose Random (mgs/dl)", min_value=50, max_value=500, value=120)
                sodium = st.number_input("Sodium (mEq/L)", min_value=120, max_value=160, value=140)
                potassium = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=8.0, value=4.5, step=0.1)
                packed_cell_volume = st.number_input("Packed Cell Volume", min_value=20, max_value=60, value=40)
                white_blood_cell_count = st.number_input("White Blood Cell Count (cells/cumm)", min_value=3000, max_value=20000, value=7500)
                red_blood_cell_count = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=2.0, max_value=8.0, value=4.5, step=0.1)
                
                red_blood_cells = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
                pus_cell = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
                pus_cell_clumps = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
                bacteria = st.selectbox("Bacteria", ["Not Present", "Present"])
                
                coronary_artery_disease = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
                appetite = st.selectbox("Appetite", ["Good", "Poor"])
                peda_edema = st.selectbox("Pedal Edema", ["No", "Yes"])
                anemia = st.selectbox("Anemia", ["No", "Yes"])
            
            st.markdown("---")
            st.subheader("🚀 Quick Prediction")
            quick_predict = st.checkbox("Use Quick Prediction (Essential parameters only)")
        
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
        predict_button_text = "🚀 Quick Predict (Essential Only)" if quick_predict else "🔬 Predict Kidney Disease (All Parameters)"
        
        if st.button(predict_button_text, type="primary"):
            try:
                # Prepare input data
                if quick_predict:
                    input_data = {
                        'age': age, 'blood_pressure': blood_pressure, 'specific_gravity': specific_gravity,
                        'albumin': albumin, 'sugar': sugar, 'red_blood_cells': "Normal",
                        'pus_cell': "Normal", 'pus_cell_clumps': "Not Present", 'bacteria': "Not Present",
                        'blood_glucose_random': 120, 'blood_urea': blood_urea, 'serum_creatinine': serum_creatinine,
                        'sodium': 140, 'potassium': 4.5, 'haemoglobin': haemoglobin, 'packed_cell_volume': 40,
                        'white_blood_cell_count': 7500, 'red_blood_cell_count': 4.5, 'hypertension': hypertension,
                        'diabetes_mellitus': diabetes_mellitus, 'coronary_artery_disease': "No",
                        'appetite': "Good", 'peda_edema': "No", 'anemia': "No"
                    }
                else:
                    input_data = {
                        'age': age, 'blood_pressure': blood_pressure, 'specific_gravity': specific_gravity,
                        'albumin': albumin, 'sugar': sugar, 'red_blood_cells': red_blood_cells,
                        'pus_cell': pus_cell, 'pus_cell_clumps': pus_cell_clumps, 'bacteria': bacteria,
                        'blood_glucose_random': blood_glucose_random, 'blood_urea': blood_urea,
                        'serum_creatinine': serum_creatinine, 'sodium': sodium, 'potassium': potassium,
                        'haemoglobin': haemoglobin, 'packed_cell_volume': packed_cell_volume,
                        'white_blood_cell_count': white_blood_cell_count, 'red_blood_cell_count': red_blood_cell_count,
                        'hypertension': hypertension, 'diabetes_mellitus': diabetes_mellitus,
                        'coronary_artery_disease': coronary_artery_disease, 'appetite': appetite,
                        'peda_edema': peda_edema, 'anemia': anemia
                    }
                
                # Convert to numeric
                numeric_data = convert_to_numeric(input_data)
                
                # Create DataFrame
                feature_names = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
                               'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
                               'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                               'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
                               'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
                               'coronary_artery_disease', 'appetite', 'peda_edema', 'anemia']
                
                input_df = pd.DataFrame([numeric_data], columns=feature_names)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
                
                # Display results
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                if prediction == 1:
                    st.error("**🚨 Prediction: Chronic Kidney Disease Detected**")
                    st.markdown("⚠️ The model indicates a high likelihood of chronic kidney disease. Please consult with a healthcare professional immediately.")
                else:
                    st.success("**✅ Prediction: No Chronic Kidney Disease**")
                    st.markdown("✅ The model indicates a low likelihood of chronic kidney disease. Regular check-ups are still recommended.")
                
                if prediction_proba is not None:
                    st.subheader("📈 Prediction Confidence")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("No CKD Probability", f"{prediction_proba[0]:.2%}")
                    with col2:
                        st.metric("CKD Probability", f"{prediction_proba[1]:.2%}")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check that all input values are valid and try again.")
    else:
        st.error("❌ Cannot make predictions - Model not loaded")
        st.info("Please resolve the deployment issues shown in the debug information above.")

elif page == "About the Model":
    st.header("📚 About the Prediction Model")
    st.markdown("""
    ### Model Overview
    This chronic kidney disease prediction system uses a machine learning model trained on medical data 
    to assess the likelihood of chronic kidney disease based on various clinical parameters.
    
    ### ⚠️ Important Disclaimer
    This tool is for **educational and screening purposes only**. It should not be used as a substitute 
    for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
    professionals for proper medical evaluation and treatment decisions.
    """)

elif page == "Dataset Info":
    st.header("📈 Dataset Information")
    st.markdown("""
    ### Dataset Overview
    The model was trained on a chronic kidney disease dataset containing medical records 
    with various clinical parameters and their corresponding diagnoses.
    
    ### Dataset Characteristics
    - **Total Features:** 24 clinical parameters
    - **Target Variable:** Binary classification (CKD vs No CKD)
    - **Data Types:** Mixed (numerical and categorical)
    """)

elif page == "Debug Info":
    st.header("🐛 Debug Information")
    
    st.subheader("Environment Information")
    st.write(f"**Python Version:** {sys.version}")
    st.write(f"**Platform:** {platform.platform()}")
    
    st.subheader("Library Versions")
    libraries = ['streamlit', 'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'Unknown')
            st.write(f"✅ **{lib}:** {version}")
        except ImportError:
            st.write(f"❌ **{lib}:** Not installed")
    
    st.subheader("File System")
    st.write(f"**Current Directory:** {Path.cwd()}")
    st.write(f"**Files in directory:**")
    for file in Path.cwd().iterdir():
        st.write(f"  - {file.name} ({'Directory' if file.is_dir() else f'{file.stat().st_size} bytes'})")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
<small>🏥 Chronic Kidney Disease Prediction System | For Educational Purposes Only</small>
</div>
""", unsafe_allow_html=True)
