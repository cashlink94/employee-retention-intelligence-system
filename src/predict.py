import pickle
import pandas as pd

MODEL_PATH = "models/model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
FEATURES_PATH = "models/features.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
encoders = pickle.load(open(ENCODERS_PATH, "rb"))
features = pickle.load(open(FEATURES_PATH, "rb"))


def predict_employee(input_dict):
    df = pd.DataFrame([input_dict])

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Ensure same feature order
    df = df.reindex(columns=features, fill_value=0)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability