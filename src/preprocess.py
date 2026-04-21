import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    df = df.copy()

    # Target encoding
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    return X, y


def split_data(df):
    X, y = preprocess(df)

    # Count class distribution
    class_counts = y.value_counts()

    # SAFE STRATIFY LOGIC (prevents sklearn crash)
    if class_counts.min() < 2 or len(df) < 20:
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    return X_train, X_test, y_train, y_test