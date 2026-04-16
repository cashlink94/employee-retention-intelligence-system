import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def split_data(df, target="Attrition"):
    # clean target
    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target].map({"Yes": 1, "No": 0})

    # REMOVE stratify safeguard (fix crash)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    return preprocessor