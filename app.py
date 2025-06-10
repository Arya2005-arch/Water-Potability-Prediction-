import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from model import load_data, preprocess_data, train_and_evaluate_models, predict_potability

# Load data and preprocess
data = load_data()
(X_train, X_test, y_train, y_test), imputer = preprocess_data(data)
results = train_and_evaluate_models(X_train, X_test, y_train, y_test, imputer)

# Save success message
st.success("✅ Model and imputer have been saved as .pkl files.")

# Load model and imputer
model = joblib.load("random_forest_model.pkl")
imputer = joblib.load("imputer.pkl")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

# Page: Home
if page == "Home":
    st.title("💧 Water Potability Prediction App")
    st.markdown("""
        Welcome to the Water Potability Predictor!  
        Use the sidebar to explore the dataset, view statistics, graphs, and make predictions.
    """)

# Page: Dataset
elif page == "Dataset":
    st.title("📄 Dataset Preview")
    st.dataframe(data.head())

# Page: Summary
elif page == "Summary":
    st.title("📊 Data Summary")
    st.write(data.describe())

    st.subheader("📈 Accuracy of Different Models")
    st.table(pd.DataFrame(results.items(), columns=["Model", "Accuracy (%)"]))

# Page: Graphs
elif page == "Graphs":
    st.title("📊 Data Visualization")

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    st.subheader("Pairplot (this may take a moment)")
    sns.pairplot(data.dropna(), hue='Potability', palette='Set2')
    st.pyplot()

# Page: Predict
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
            st.success(f"Prediction: {'Potable 💧' if result == 1 else 'Not Potable ⚠️'}")
