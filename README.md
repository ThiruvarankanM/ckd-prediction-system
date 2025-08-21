# Chronic Kidney Disease (CKD) Prediction System 🏥

A **Streamlit-based machine learning application** that predicts the likelihood of **Chronic Kidney Disease (CKD)** using clinical and laboratory parameters. This tool provides an interactive interface for entering patient data and generates predictions along with confidence scores.

> **Note:** A `Dockerfile` is included for containerized deployment, but it is optional and not required to run the app locally.  
> **Live Demo:** The app is deployed on Streamlit Cloud — [Click here]([https://ckd-prediction-system.streamlit.app/])

---

## 🌟 Features

- Predict CKD using **essential parameters** for quick screening.
- Optional input of **detailed medical tests** for more accurate predictions.
- Interactive **Streamlit UI** with organized input sections.
- Shows **prediction results**, **confidence scores**, and **input summary**.
- Designed for **educational and screening purposes**.

---

## 🩺 Features & Parameters

**Basic Information:**
- Age, Blood Pressure, Specific Gravity  

**Urine Tests:**
- Albumin, Sugar, Red Blood Cells, Pus Cell, Pus Cell Clumps, Bacteria  

**Blood Tests:**
- Blood Glucose Random, Blood Urea, Serum Creatinine, Sodium, Potassium, Haemoglobin, Packed Cell Volume, White Blood Cell Count, Red Blood Cell Count  

**Medical History & Symptoms:**
- Hypertension, Diabetes Mellitus, Coronary Artery Disease, Appetite, Pedal Edema, Anemia  

---

## 🚀 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
````

---

## 📊 Dataset & Model

* Trained on a CKD dataset with 24 clinical parameters.
* Preprocessing: missing value imputation, outlier handling, feature scaling.
* Machine learning model: Supports probability scores for prediction confidence.
* Model file: `ckd_model.pkl`

---

## ⚠️ Disclaimer

This application is **for educational and screening purposes only**.
It is **not a substitute for professional medical advice, diagnosis, or treatment**.
Always consult with qualified healthcare professionals for proper evaluation.

---

## 📚 Tech Stack

* **Python**
* **Streamlit** (UI)
* **scikit-learn** (Machine Learning)
* **Pandas & NumPy** (Data handling)
* **Matplotlib & Seaborn** (Visualizations)

---
## 🎥 Working Demo Video

Watch the final demonstration of the **Chronic Kidney Disease (CKD) Prediction System** here:  
[![Demo Video](https://img.youtube.com/vi/VpKk3xld2jM/0.jpg)](https://youtu.be/VpKk3xld2jM)

Or click this link: [https://youtu.be/VpKk3xld2jM](https://youtu.be/VpKk3xld2jM)


## 💡 Author

**Thiruvarankan Mathurakaran**
[GitHub Profile](https://github.com/ThiruvarankanM)
