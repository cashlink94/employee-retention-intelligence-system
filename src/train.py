import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from preprocess import load_data, split_data

DATA_PATH = "data/HR_data.csv"
MODEL_PATH = "models/model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
FEATURES_PATH = "models/features.pkl"


def encode_features(X_train, X_test):
    encoders = {}
    X_train = X_train.copy()
    X_test = X_test.copy()

    cat_cols = X_train.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        le = LabelEncoder()

        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

        le.fit(X_train[col])

        X_train[col] = le.transform(X_train[col])

        X_test[col] = X_test[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

        encoders[col] = le

    return X_train, X_test, encoders


def main():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test, encoders = encode_features(X_train, X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))

    os.makedirs("models", exist_ok=True)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(encoders, open(ENCODERS_PATH, "wb"))
    pickle.dump(X_train.columns.tolist(), open(FEATURES_PATH, "wb"))

    print("Model + encoders + features saved!")


if __name__ == "__main__":
    main()