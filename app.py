import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
with open("logistic_model.pkl", "rb") as f:
    model = joblib.load("logistic_model.pkl")

st.title("🚢 Titanic Survival Predictor")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Preprocessing
sex_code = 1 if sex == "male" else 0
embarked_code = {"S": 2, "C": 0, "Q": 1}[embarked]

# Create input DataFrame
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex_code,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked_code
}])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 Survived with {probability:.2%} probability.")
    else:
        st.error(f"💀 Did not survive. Survival chance: {probability:.2%}.")
