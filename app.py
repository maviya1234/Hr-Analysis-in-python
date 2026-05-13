#%%
#%%

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

# ===============================
# TITLE
# ===============================
st.title("💼 HR Attrition Prediction System")

st.subheader("🔍 Key Drivers of Attrition")
c1, c2, c3 = st.columns(3)
c1.metric("1️⃣ Income", "Low Salary Impact")
c2.metric("2️⃣ OverTime", "High Workload")
c3.metric("3️⃣ Age", "Experience Factor")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("HR_Analytics.csv")
    df.drop("EmployeeNumber", axis=1, inplace=True)
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, drop_first=True)
    return df

df = load_data()

# ===============================
# PREPARE DATA
# ===============================
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# 🔥 SMOTE (fix imbalance)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# ===============================
# MODEL TRAINING
# ===============================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=3,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("📥 Employee Details")

age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
distance = st.sidebar.slider("Distance From Home", 1, 50, 10)
years = st.sidebar.slider("Years at Company", 0, 40, 5)
job_sat = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 2)
work_years = st.sidebar.slider("Total Working Years", 0, 40, 8)

overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
marital = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Manager"])

# ===============================
# INPUT DATAFRAME
# ===============================
input_dict = {
    "Age": age,
    "MonthlyIncome": income,
    "DistanceFromHome": distance,
    "YearsAtCompany": years,
    "JobSatisfaction": job_sat,
    "TotalWorkingYears": work_years,
    "OverTime_Yes": 1 if overtime == "Yes" else 0,
    "MaritalStatus_Single": 1 if marital == "Single" else 0,
    "MaritalStatus_Married": 1 if marital == "Married" else 0,
    "JobRole_Sales Executive": 1 if job_role == "Sales Executive" else 0,
    "JobRole_Research Scientist": 1 if job_role == "Research Scientist" else 0,
}

input_df = pd.DataFrame([input_dict])

# Match columns
input_df = input_df.reindex(columns=df.drop("Attrition", axis=1).columns, fill_value=0)

# Apply imputer
input_df = imputer.transform(input_df)

# ===============================
# PREDICTION
# ===============================
st.subheader("🎯 Prediction")

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]

    st.write(f"Attrition Probability: {prob:.2f}")
    st.progress(int(prob * 100))

    if prob > 0.4:
        st.error("⚠️ High Risk: Employee likely to leave")
    elif prob > 0.25:
        st.warning("⚠️ Medium Risk: Monitor employee")
    else:
        st.success("✅ Low Risk: Employee likely to stay")

# ===============================
# OPTIONAL DATA VIEW
# ===============================
if st.checkbox("Show Dataset"):
    st.write(df.head())
# %%