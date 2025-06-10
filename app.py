import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# App config
st.set_page_config(page_title="Water Potability Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

# Home Page
if page == "Home":
    st.title("💧 Water Potability Prediction App")
    st.markdown("""
        Welcome to the **Water Potability Prediction App** built with **Streamlit** and **Machine Learning**.
        \nUse the sidebar to explore the dataset, view visualizations, or make predictions.
    """)

# Dataset Page
elif page == "Dataset":
    st.title("📊 Dataset Overview")
    df = pd.read_csv("water_potability.csv")  # Replace with your dataset path
    st.dataframe(df)

# Summary Page
elif page == "Summary":
    st.title("📌 Statistical Summary")
    df = pd.read_csv("water_potability.csv")
    st.write(df.describe())

# Graphs Page
elif page == "Graphs":
    st.title("📈 Data Visualization")
    df = pd.read_csv("water_potability.csv")

    st.subheader("Pairplot of Features")
    st.markdown("This shows relationships between variables based on potability.")
    fig = sns.pairplot(df.dropna(), hue='Potability')
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

# Prediction Page
elif page == "Predict":
    st.title("🔍 Water Potability Predictor")

    # Add sliders or input fields here
    ph = st.slider("pH Level", 0.0, 14.0, 7.0)
    hardness = st.slider("Hardness", 50, 500, 200)
    solids = st.slider("Solids", 500, 50000, 10000)
    conductivity = st.slider("Conductivity", 100, 1000, 400)
    # ... add more features as needed

    if st.button("Predict"):
        # Example prediction logic (replace with model)
        st.success("Prediction logic goes here")
