import streamlit as st
import pickle
import numpy as np

# Load trained models
with open("C:/Users/bhavyanshu/Desktop/New folder (2)/Model/diabetes.pkl", "rb") as f:
    diabetes_model = pickle.load(f)
with open("C:/Users/bhavyanshu/Desktop/New folder (2)/Model/liver.pkl", "rb") as f:
    liver_model = pickle.load(f)
with open("C:/Users/bhavyanshu/Desktop/New folder (2)/Model/kidney.pkl", "rb") as f:
    kidney_model = pickle.load(f)
with open("C:/Users/bhavyanshu/Desktop/New folder (2)/Model/cancer.pkl", "rb") as f:
    cancer_model = pickle.load(f)

st.title("Multiple Disease Prediction System")

# Disease selection
disease = st.selectbox("Select a Disease for Prediction", 
                       ["Diabetes", "Liver Disease", "Kidney Disease", "Cancer"])

def predict(model, inputs):
    data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(data)
    return "Positive" if prediction[0] == 1 else "Negative"

if disease == "Diabetes":
    st.header("Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 122)
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 846)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)
    if st.button("Predict"):
        result = predict(diabetes_model, [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age])
        st.success(f"Diabetes Prediction: {result}")

elif disease == "Liver Disease":
    st.header("Liver Disease Prediction")
    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin", 0.0, 10.0)
    direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 10.0)
    alk_phosphate = st.number_input("Alkaline Phosphotase", 0, 2000)
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase", 0, 2000)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", 0, 2000)
    total_proteins = st.number_input("Total Proteins", 0.0, 10.0)
    albumin = st.number_input("Albumin", 0.0, 6.0)
    ag_ratio = st.number_input("Albumin and Globulin Ratio", 0.0, 3.0)
    gender_binary = 1 if gender == "Male" else 0
    if st.button("Predict"):
        result = predict(liver_model, [
            age, gender_binary, total_bilirubin, direct_bilirubin,
            alk_phosphate, alamine_aminotransferase, aspartate_aminotransferase,
            total_proteins, albumin, ag_ratio
        ])
        st.success(f"Liver Disease Prediction: {result}")

elif disease == "Kidney Disease":
    st.header("Kidney Disease Prediction")

    # Kidney model expects 18 inputs
    age = st.number_input("Age", 1, 100)
    blood_pressure = st.number_input("Blood Pressure", 0, 200)
    specific_gravity = st.number_input("Specific Gravity", 1.000, 1.030)
    albumin = st.number_input("Albumin", 0, 5)
    sugar = st.number_input("Sugar", 0, 5)
    red_blood_cells = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
    pus_cell = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
    pus_cell_clumps = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
    bacteria = st.selectbox("Bacteria", ["Present", "Not Present"])
    blood_urea = st.number_input("Blood Urea", 0.0, 200.0)
    serum_creatinine = st.number_input("Serum Creatinine", 0.0, 20.0)
    sodium = st.number_input("Sodium", 100.0, 150.0)
    potassium = st.number_input("Potassium", 1.0, 10.0)
    hemoglobin = st.number_input("Hemoglobin", 3.0, 17.5)
    packed_cell_volume = st.number_input("Packed Cell Volume", 20, 55)
    white_blood_cell_count = st.number_input("White Blood Cell Count", 4000, 18000)
    red_blood_cell_count = st.number_input("Red Blood Cell Count", 2.0, 6.5)
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])

    # Convert categorical to numerical
    rbc = 1 if red_blood_cells == "Normal" else 0
    pc = 1 if pus_cell == "Normal" else 0
    pcc = 1 if pus_cell_clumps == "Present" else 0
    ba = 1 if bacteria == "Present" else 0
    htn = 1 if hypertension == "Yes" else 0

    if st.button("Predict"):
        result = predict(kidney_model, [
            age, blood_pressure, specific_gravity, albumin, sugar,
            rbc, pc, pcc, ba, blood_urea, serum_creatinine,
            sodium, potassium, hemoglobin, packed_cell_volume,
            white_blood_cell_count, red_blood_cell_count, htn
        ])
        st.success(f"Kidney Disease Prediction: {result}")

elif disease == "Cancer":
    st.header("Cancer Prediction")
    # Replace with real cancer model features
    radius_mean = st.number_input("Radius Mean", 0.0, 30.0)
    texture_mean = st.number_input("Texture Mean", 0.0, 40.0)
    perimeter_mean = st.number_input("Perimeter Mean", 0.0, 200.0)
    area_mean = st.number_input("Area Mean", 0.0, 2500.0)
    smoothness_mean = st.number_input("Smoothness Mean", 0.0, 1.0)
    if st.button("Predict"):
        result = predict(cancer_model, [
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
        ])
        st.success(f"Cancer Prediction: {result}")
