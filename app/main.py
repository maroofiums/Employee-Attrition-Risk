import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "employee_attrition_pipeline.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Employee Attrition Risk Predictor")
st.title("Employee Attrition Risk Assessment System")

st.markdown("Predict employee attrition risk using ML stacking model.")

with st.form("prediction_form"):

    Age = st.slider("Age", 18, 60, 30)
    BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    DailyRate = st.number_input("Daily Rate", 100, 2000, 800)
    Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    DistanceFromHome = st.slider("Distance From Home", 1, 30, 5)
    Education = st.slider("Education Level (1-5)", 1, 5, 3)
    EducationField = st.selectbox("Education Field", ["Life Sciences","Medical","Marketing","Technical Degree","Other","Human Resources"])
    EnvironmentSatisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    HourlyRate = st.slider("Hourly Rate", 30, 100, 60)
    JobRole = st.selectbox("Job Role", [
        "Sales Executive","Research Scientist","Laboratory Technician",
        "Manufacturing Director","Healthcare Representative",
        "Manager","Sales Representative","Research Director","Human Resources"
    ])
    MaritalStatus = st.selectbox("Marital Status", ["Single","Married","Divorced"])
    MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
    OverTime = st.selectbox("Over Time", ["Yes","No"])
    TotalWorkingYears = st.slider("Total Working Years", 0, 40, 5)
    YearsAtCompany = st.slider("Years At Company", 0, 40, 3)
    WorkLifeBalance = st.slider("Work Life Balance (1-4)", 1, 4, 3)

    submit = st.form_submit_button("Predict Risk")

if submit:

    input_data = pd.DataFrame([{
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "OverTime": OverTime,
        "TotalWorkingYears": TotalWorkingYears,
        "YearsAtCompany": YearsAtCompany,
        "WorkLifeBalance": WorkLifeBalance
    }])

    probability = model.predict_proba(input_data)[0][1]

    if probability > 0.7:
        risk = "High Risk"
        st.error(f"Attrition Probability: {probability:.2f}")
    elif probability > 0.4:
        risk = "Medium Risk"
        st.warning(f"Attrition Probability: {probability:.2f}")
    else:
        risk = "Low Risk"
        st.success(f"Attrition Probability: {probability:.2f}")

    st.subheader(f"Risk Level: {risk}")