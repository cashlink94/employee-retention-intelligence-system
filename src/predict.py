import pickle
import numpy as np
import pandas as pd
import shap

model = pickle.load(open("models/model.pkl", "rb"))
features = pickle.load(open("models/features.pkl", "rb"))

explainer = shap.TreeExplainer(model)


def predict_employee(data: dict):

    df = pd.DataFrame([data])

    df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})

    df = df[features].fillna(0)

    prob = model.predict_proba(df)[0][1]
    pred = int(prob > 0.5)

    risk_score = round(prob * 100, 2)

    # ================= SHAP FIX =================
    shap_values = explainer.shap_values(df)

    # binary classification handling
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # convert to numpy + flatten properly
    importances = np.array(shap_values[0]).flatten()

    return pred, risk_score, importances, features