# Employee Attrition Risk Prediction

Live Demo:  
https://employee-attrition-risk-prediction.streamlit.app/

GitHub Repository:  
https://github.com/maroofiums/Employee-Attrition-Risk

Kaggle Notebook:  
https://www.kaggle.com/code/maroofiums/employee-attrition-risk-predictor

---

## Project Overview

This project is an end-to-end machine learning system designed to predict employee attrition risk using HR analytics data.

It includes:

- Data cleaning and preprocessing
- Feature engineering
- Ensemble model training using stacking
- Model evaluation and threshold tuning
- A deployed Streamlit web application for real-time predictions

The goal is to help HR teams identify employees at risk of leaving and support data-driven retention strategies.

---

## Repository Structure

```

Employee-Attrition-Risk/
│
├── Data
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── Notebook
│   └── employee-attrition-risk-predictor.ipynb
│
├── app
│   ├── models
│   │   └── employee_attrition_pipeline.pkl
│   └── main.py
│
├── README.md
└── requirements.txt

```

---

## Dataset

The project uses the IBM HR Analytics Attrition dataset.

- 1,470 employee records
- 35 attributes related to:
  - Demographics
  - Job role and department
  - Compensation
  - Satisfaction metrics
  - Experience and tenure

Target variable:  
Attrition (Yes / No)

---

## Machine Learning Pipeline

The final model is built using a stacking ensemble approach.

Base learners:
- RandomForestClassifier
- XGBoost Classifier
- LightGBM Classifier

Meta learner:
- Logistic Regression

The pipeline includes:
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- ColumnTransformer for preprocessing
- StackingClassifier for ensemble learning

The full pipeline is saved and loaded in the Streamlit application to ensure consistent preprocessing during inference.

---

## Model Performance

- ROC-AUC Score: ~0.81
- Threshold tuning performed to improve recall for attrition class
- Balanced accuracy optimized for minority class detection

Detailed evaluation metrics and confusion matrices are available in the notebook.

---

## Features Used

The model uses the following features:

Age  
BusinessTravel  
DailyRate  
Department  
DistanceFromHome  
Education  
EducationField  
EnvironmentSatisfaction  
Gender  
HourlyRate  
JobInvolvement  
JobLevel  
JobSatisfaction  
JobRole  
MaritalStatus  
MonthlyIncome  
MonthlyRate  
NumCompaniesWorked  
OverTime  
PercentSalaryHike  
PerformanceRating  
RelationshipSatisfaction  
StockOptionLevel  
TotalWorkingYears  
TrainingTimesLastYear  
WorkLifeBalance  
YearsAtCompany  
YearsInCurrentRole  
YearsSinceLastPromotion  
YearsWithCurrManager  

---

## Streamlit Application

The Streamlit application allows users to:

- Input employee information
- Generate attrition probability
- Classify risk level (Low, Medium, High)

The model is deployed on Streamlit Cloud for public access.

Live application:  
https://employee-attrition-risk-prediction.streamlit.app/

---

## How to Run Locally

1. Clone the repository:

```

git clone [https://github.com/maroofiums/Employee-Attrition-Risk.git](https://github.com/maroofiums/Employee-Attrition-Risk.git)
cd Employee-Attrition-Risk

```

2. Install dependencies:

```

pip install -r requirements.txt

```

3. Run the application:

```

streamlit run app/main.py

```

---

## Deployment Details

- Python 3.10
- Scikit-learn version pinned in requirements.txt
- Full preprocessing and model pipeline serialized using joblib

Using a consistent sklearn version is important to avoid pickle compatibility errors.

---

## Future Improvements

- Integrate SHAP for explainability
- Add batch prediction via CSV upload
- Implement automated retraining pipeline
- Containerize using Docker
- Deploy on cloud infrastructure (AWS / GCP / Azure)

---

## Author

Maroof  
AI / ML Developer  
GitHub: https://github.com/maroofiums

---
