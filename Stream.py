import streamlit as st
import pandas as pd
import numpy as np
from Insurance import InsuranceModel


st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")

@st.cache_resource(ttl=3600)
def load_model():
    model = InsuranceModel()
    model.load()
    return model




model = load_model()

def predict_insurance():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 18, 100, 25)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        
    with col2:
        children = st.number_input("Children", 0, 10, 0)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            'age': age, 'sex': sex, 'bmi': bmi,
            'children': children, 'smoker': smoker,
            'region': region
        }])
        
        prepared_data = model.prepare_data(input_data)
        prediction = model.predict(prepared_data)[0]
        st.success(f"Predicted Cost: ${prediction:,.2f}")

if __name__ == "__main__":
    predict_insurance()
