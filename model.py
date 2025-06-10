import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data():
    return pd.read_csv("water_potability.csv")[:800]  # Limit to 800 rows

def preprocess_data(df):
    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return train_test_split(X_imputed, y, test_size=0.3, random_state=42), imputer

def train_and_save_model(X_train, X_test, y_train, y_test, imputer):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "random_forest_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    print("✅ Model and imputer have been saved.")
    return model

def predict_potability(model, input_data, imputer):
    input_df = pd.DataFrame([input_data])
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    return model.predict(input_imputed)[0]
