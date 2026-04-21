import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ===================== LOAD DATA (AUTO-DETECT) ===================== #
def load_data():
    print("Loading data...")

    possible_paths = [
        "data/employee_data.csv",
        "employee_data.csv",
        "data/HR.csv",
        "HR.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found dataset: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError(
        "❌ No dataset found.\n"
        "Put your CSV file in:\n"
        " - project root OR\n"
        " - data/ folder\n"
    )


# ===================== PREPROCESS ===================== #
def preprocess_data(df):
    print("Preprocessing data...")

    # Target
    y = df["Attrition"].map({"Yes": 1, "No": 0})

    # Features
    X = df.drop("Attrition", axis=1)

    # Columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "string"]).columns

    # Encoder (FIXED)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    return X, y, preprocessor


# ===================== TRAIN MODEL ===================== #
def train_model(X, y, preprocessor):
    print("Splitting data...")

    if len(y) < 10:
        print("⚠️ Dataset too small → skipping split")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    print("Training model...")

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\nAccuracy: {acc:.4f}")

    return model


# ===================== SAVE ===================== #
def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("✅ Model saved successfully!")


# ===================== MAIN ===================== #
def main():
    df = load_data()

    print("\nClass distribution:")
    print(df["Attrition"].value_counts())

    X, y, preprocessor = preprocess_data(df)

    model = train_model(X, y, preprocessor)

    save_model(model)


if __name__ == "__main__":
    main()