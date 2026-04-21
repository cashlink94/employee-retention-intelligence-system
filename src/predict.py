import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===================== LOAD MODEL ===================== #
model = joblib.load("models/model.pkl")

# ===================== PAGE CONFIG ===================== #
st.set_page_config(
    page_title="Employee Retention Intelligence System",
    layout="centered"
)

# ===================== HEADER ===================== #
st.title("🏢 Employee Retention Intelligence System")
st.caption("AI-powered employee attrition prediction & analytics dashboard")

st.markdown("---")

# ===================== INPUT SECTION ===================== #
st.subheader("👤 Employee Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    distance = st.slider("Distance From Home", 1, 30, 5)
    years = st.slider("Years at Company", 0, 40, 3)

with col2:
    daily_rate = st.number_input("Daily Rate", 100, 2000, 500)
    income = st.number_input("Monthly Income", 1000, 20000, 5000)

department = st.selectbox("Department", ["Sales", "HR", "R&D"])
overtime = st.selectbox("OverTime", ["Yes", "No"])

st.markdown("---")

# ===================== INPUT DATA ===================== #
input_data = {
    "Age": age,
    "DailyRate": daily_rate,
    "DistanceFromHome": distance,
    "MonthlyIncome": income,
    "YearsAtCompany": years,
    "Department": department,
    "OverTime": overtime
}

df = pd.DataFrame([input_data])

# ===================== PREDICTION ===================== #
if st.button("🚀 Predict Attrition Risk"):

    try:
        prob = model.predict_proba(df)[0][1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # ===================== RESULT ===================== #
    st.markdown("## 📊 Prediction Result")

    if prob < 0.4:
        st.success(f"🟢 Low Risk ({prob:.2%})")
        zone = "SAFE"
    elif prob < 0.7:
        st.warning(f"🟡 Medium Risk ({prob:.2%})")
        zone = "MODERATE"
    else:
        st.error(f"🔴 High Risk ({prob:.2%})")
        zone = "HIGH"

    # ===================== RISK VISUALISATION ===================== #
    st.markdown("### 📊 Risk Visualisation")

    st.progress(float(prob))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Low Risk", "0–40%")

    with col2:
        st.metric("Medium Risk", "40–70%")

    with col3:
        st.metric("High Risk", "70–100%")

    st.info(f"Current Risk Zone: **{zone}**")

    # ===================== MODEL INSIGHTS ===================== #
    st.markdown("### 🧠 Model Insights (Key Drivers)")

    try:
        clf = model.named_steps.get("classifier", None)

        if clf and hasattr(clf, "feature_importances_"):

            importances = clf.feature_importances_

            features = [
                "Age",
                "DailyRate",
                "DistanceFromHome",
                "MonthlyIncome",
                "YearsAtCompany"
            ]

            idx = np.argsort(importances)[::-1]

            for i in idx:
                feature = features[i]
                value = importances[i]

                bar = "█" * int(value * 40)

                direction = "🔴 increases risk" if feature == "DistanceFromHome" else "🟢 reduces risk"

                st.write(f"{feature} {bar} ({value:.3f}) {direction}")

        else:
            st.info("Model does not support feature importance")

    except:
        st.error("Could not generate model insights")

    # ===================== HR EXPLANATION ===================== #
    st.markdown("### 💡 HR Interpretation")

    if prob > 0.7:
        st.error("High risk employee — immediate retention action required")
    elif prob > 0.4:
        st.warning("Moderate risk — monitor employee closely")
    else:
        st.success("Low risk — employee is stable")

    st.markdown("""
### 📌 Key Business Drivers
- Low income + overtime → higher attrition risk  
- Short tenure → weaker job attachment  
- Long tenure → stronger retention stability  
""")