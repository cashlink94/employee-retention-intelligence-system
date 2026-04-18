import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from src.predict import predict_employee

# Load model for feature importance
model = pickle.load(open("models/model.pkl", "rb"))

st.set_page_config(page_title="Employee Retention AI", layout="wide")

st.title("Employee Retention Intelligence System")
st.markdown("### AI-powered employee attrition prediction")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Employee Details")

    age = st.slider("Age", 18, 60, 30)
    daily_rate = st.number_input("Daily Rate", 100, 1500, 500)
    distance = st.slider("Distance From Home", 1, 30, 5)

with col2:
    st.subheader("Work Information")

    income = st.number_input("Monthly Income", 1000, 20000, 5000)
    years = st.slider("Years at Company", 0, 40, 3)


if st.button("Predict Attrition Risk"):

    input_data = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance,
        "MonthlyIncome": income,
        "YearsAtCompany": years
    }

    prediction, prob = predict_employee(input_data)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠ High Risk of Leaving ({prob:.2%})")
    else:
        st.success(f"✅ Low Risk ({prob:.2%})")

    # Feature Importance
    st.subheader("Model Insights")

    importances = model.feature_importances_
    features = model.feature_names_in_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)