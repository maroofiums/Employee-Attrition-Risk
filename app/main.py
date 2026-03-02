import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Employee Attrition Risk Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Risk Prediction System")
st.markdown("Predict employee attrition probability using a trained ML pipeline.")

# ---------------------------
# Load Model
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "employee_attrition_pipeline.pkl"
model = joblib.load(MODEL_PATH)
# ---------------------------
# Input Sections
# ---------------------------

st.header("👤 Personal Information")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", 18, 60, 30)

with col2:
    Gender = st.selectbox("Gender", ["Male", "Female"])

with col3:
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])


# ---------------------------
st.header("💼 Job Information")

col1, col2, col3 = st.columns(3)

with col1:
    Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])

with col2:
    JobRole = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative",
        "Manager", "Sales Representative", "Research Director", "Human Resources"
    ])

with col3:
    BusinessTravel = st.selectbox("Business Travel", [
        "Travel_Rarely", "Travel_Frequently", "Non-Travel"
    ])

OverTime = st.selectbox("OverTime", ["Yes", "No"])


# ---------------------------
st.header("💰 Compensation")

col1, col2 = st.columns(2)

with col1:
    MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)

with col2:
    PercentSalaryHike = st.slider("Percent Salary Hike", 10, 25, 15)

StockOptionLevel = st.slider("Stock Option Level", 0, 3, 1)


# ---------------------------
st.header("📈 Experience")

col1, col2, col3 = st.columns(3)

with col1:
    TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 5)

with col2:
    YearsAtCompany = st.number_input("Years At Company", 0, 40, 3)

with col3:
    YearsInCurrentRole = st.number_input("Years In Current Role", 0, 20, 2)

YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20, 2)
NumCompaniesWorked = st.number_input("Number of Companies Worked", 0, 10, 1)
TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 10, 2)


# ---------------------------
st.header("📊 Satisfaction & Performance")

col1, col2, col3 = st.columns(3)

with col1:
    EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 3)

with col2:
    JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)

with col3:
    RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)

JobInvolvement = st.slider("Job Involvement", 1, 4, 3)
WorkLifeBalance = st.slider("Work Life Balance", 1, 4, 3)
PerformanceRating = st.slider("Performance Rating", 1, 4, 3)


# ---------------------------
st.header("📍 Additional Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    DailyRate = st.number_input("Daily Rate", 100, 2000, 800)

with col2:
    HourlyRate = st.number_input("Hourly Rate", 10, 200, 60)

with col3:
    MonthlyRate = st.number_input("Monthly Rate", 1000, 30000, 15000)

DistanceFromHome = st.number_input("Distance From Home (km)", 0, 50, 5)
Education = st.slider("Education Level (1-5)", 1, 5, 3)
EducationField = st.selectbox("Education Field", [
    "Life Sciences", "Medical", "Marketing",
    "Technical Degree", "Human Resources", "Other"
])
JobLevel = st.slider("Job Level", 1, 5, 2)


# ---------------------------
# Prediction
# ---------------------------

st.divider()
submit = st.button("🔍 Predict Attrition Risk")

if submit:

    input_df = pd.DataFrame([{
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
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobSatisfaction": JobSatisfaction,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager
    }])

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if probability > 0.7:
        st.error(f"High Attrition Risk ⚠️\n\nProbability: {probability:.2f}")
    elif probability > 0.4:
        st.warning(f"Medium Attrition Risk ⚡\n\nProbability: {probability:.2f}")
    else:
        st.success(f"Low Attrition Risk ✅\n\nProbability: {probability:.2f}")