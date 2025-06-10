import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from model import load_data, preprocess_data, train_and_save_model, predict_potability

# Load and preprocess data
data = load_data()
(X_train, X_test, y_train, y_test), imputer = preprocess_data(data)

# Train and save model (every time app runs)
model = train_and_save_model(X_train, X_test, y_train, y_test, imputer)

# Load from pkl
model = joblib.load("random_forest_model.pkl")
imputer = joblib.load("imputer.pkl")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

if page == "Home":
    st.title("💧 Water Potability Prediction App")
    st.markdown("Use the app to check if water is potable based on input parameters.")

elif page == "Dataset":
    st.title("📄 Dataset Preview")
    st.dataframe(data.head())

elif page == "Summary":
    st.title("📊 Data Summary")
    st.write(data.describe())

elif page == "Graphs":
    st.title("📈 Data Visualization")

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    st.subheader("Pairplot")
    sns.pairplot(data.dropna(), hue='Potability', palette='Set2')
    st.pyplot()

elif page == "Predict":
    st.title("🔍 Predict Water Potability")

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
            st.success(f"Prediction: {'✅ Potable Water' if result == 1 else '❌ Not Potable'}")
