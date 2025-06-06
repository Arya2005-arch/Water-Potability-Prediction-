import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data():
    return pd.read_csv("water_potability.csv")

def preprocess_data(df):
    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return train_test_split(X_imputed, y, test_size=0.3, random_state=42), imputer

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results[name] = round(accuracy * 100, 2)

    return results, models["Random Forest"]  # Return RF for live predictions

def predict_potability(model, input_data, imputer):
    input_df = pd.DataFrame([input_data])
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    prediction = model.predict(input_imputed)[0]
    return prediction
