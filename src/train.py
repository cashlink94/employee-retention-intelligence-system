import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "data/HR_data.csv"

MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/features.pkl"


FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "MonthlyIncome",
    "YearsAtCompany",
    "OverTime",
    "JobSatisfaction",
    "WorkLifeBalance"
]


def main():
    df = pd.read_csv(DATA_PATH)

    # target
    y = df["Attrition"].map({"No": 0, "Yes": 1})

    X = df[FEATURES].copy()

    # clean encoding
    X["OverTime"] = X["OverTime"].map({"Yes": 1, "No": 0})

    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))

    os.makedirs("models", exist_ok=True)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(FEATURES, open(FEATURES_PATH, "wb"))

    print("✅ Clean production model saved!")


if __name__ == "__main__":
    main()