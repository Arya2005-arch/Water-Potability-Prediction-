import streamlit as st
import pandas as pd
from model import load_data, preprocess_data, train_and_evaluate_models, predict_potability

st.title("💧 Water Potability Prediction App")

# Load and show dataset
data = load_data()
st.subheader("Dataset Preview")
st.dataframe(data.head())

if st.checkbox("Show data summary"):
    st.write(data.describe())




# Live prediction
st.subheader("🔍 Predict Potability of Your Sample")

with st.form("prediction_form"):
    ph = st.number_input("pH value", min_value=0.0, max_value=14.0, value=7.0)
    hardness = st.number_input("Hardness", value=200.0)
    solids = st.number_input("Solids", value=15000.0)
    chloramines = st.number_input("Chloramines", value=7.0)
    sulfate = st.number_input("Sulfate", value=300.0)
    conductivity = st.number_input("Conductivity", value=400.0)
    organic_carbon = st.number_input("Organic Carbon", value=10.0)
    trihalomethanes = st.number_input("Trihalomethanes", value=60.0)
    turbidity = st.number_input("Turbidity", value=4.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        features = {
            "ph": ph,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity
        }
        result = predict_potability(model, features, imputer)
        st.success(f"Prediction: {'Potable' if result == 1 else 'Not Potable'}")
