# Chronic Kidney Disease (CKD) Prediction System

A Streamlit-based machine learning application that predicts the likelihood of Chronic Kidney Disease using clinical and laboratory parameters. Provides interactive interface for patient data entry and generates predictions with confidence scores.

## Demo

[![Demo Video](https://img.youtube.com/vi/VpKk3xld2jM/0.jpg)](https://youtu.be/VpKk3xld2jM)

**[Watch Live Demo](https://youtu.be/VpKk3xld2jM)**

**[Live Application](https://ckd-prediction-system.streamlit.app/)**

## Features

- CKD prediction using essential clinical parameters
- Optional detailed medical test inputs for improved accuracy
- Interactive Streamlit interface with organized input sections
- Prediction results with confidence scores and input summary
- Educational and screening tool for healthcare awareness

## Parameters

**Basic Information:** Age, Blood Pressure, Specific Gravity

**Urine Tests:** Albumin, Sugar, Red Blood Cells, Pus Cell, Pus Cell Clumps, Bacteria

**Blood Tests:** Blood Glucose Random, Blood Urea, Serum Creatinine, Sodium, Potassium, Haemoglobin, Packed Cell Volume, White Blood Cell Count, Red Blood Cell Count

**Medical History:** Hypertension, Diabetes Mellitus, Coronary Artery Disease, Appetite, Pedal Edema, Anemia

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Model Details

- **Dataset:** CKD dataset with 24 clinical parameters
- **Preprocessing:** Missing value imputation, outlier handling, feature scaling
- **Model:** Machine learning classifier with probability scoring
- **File:** `ckd_model.pkl`

## Tech Stack

- **Python** - Core development
- **Streamlit** - Web interface
- **scikit-learn** - Machine learning
- **Pandas & NumPy** - Data processing
- **Matplotlib & Seaborn** - Visualization

## Docker Deployment

```bash
docker build -t ckd-prediction .
docker run -p 8501:8501 ckd-prediction
```

## Disclaimer

This application is for educational and screening purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for proper medical evaluation.

## License

MIT License
