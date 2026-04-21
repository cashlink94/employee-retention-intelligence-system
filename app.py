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

# ===================== INPUT ===================== #
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

# ===================== INPUT FRAME ===================== #
input_df = pd.DataFrame([{
    "Age": age,
    "DailyRate": daily_rate,
    "DistanceFromHome": distance,
    "MonthlyIncome": income,
    "YearsAtCompany": years,
    "Department": department,
    "OverTime": overtime
}])

# ===================== PREDICTION ===================== #
if st.button("🚀 Predict Attrition Risk"):

    try:
        prob = model.predict_proba(input_df)[0][1]
    except Exception:
        st.error("Prediction failed — model mismatch")
        st.stop()

    # ===================== RESULT ===================== #
    st.markdown("## 📊 Prediction Result")

    if prob < 0.4:
        st.success(f"🟢 Low Risk ({prob:.2%})")
    elif prob < 0.7:
        st.warning(f"🟡 Medium Risk ({prob:.2%})")
    else:
        st.error(f"🔴 High Risk ({prob:.2%})")

    # ===================== RISK BAR ===================== #
    st.markdown("### 📊 Risk Visualisation")
    st.progress(float(prob))
    st.info("Risk Zones: Low (0–40%) | Medium (40–70%) | High (70–100%)")

    # ===================== MODEL INSIGHTS ===================== #
    st.markdown("### 🧠 Model Insights (Key Drivers)")

    features = ["Age", "DailyRate", "DistanceFromHome", "MonthlyIncome", "YearsAtCompany"]

    importances = None

    # SAFE IMPORTANCE EXTRACTION
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        clf = model.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_

    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # ===================== DISPLAY SAFE ===================== #
    if importances is not None:

        st.markdown("#### 🔍 Ranked Risk Drivers")

        safe_len = min(len(features), len(importances))
        sorted_idx = np.argsort(importances[:safe_len])[::-1]

        for rank, i in enumerate(sorted_idx, start=1):

            feature = features[i]
            importance = float(importances[i])
            value = input_df.iloc[0][feature]

            direction = (
                "🔴 increases risk"
                if feature in ["DistanceFromHome", "DailyRate"]
                else "🟢 reduces risk"
            )

            col1, col2, col3 = st.columns([1, 5, 6])

            with col1:
                st.write(f"**#{rank}**")

            with col2:
                st.write(feature)

            with col3:
                st.progress(importance)
                st.caption(f"{value} → {importance:.3f} ({direction})")

    else:
        st.info("Explainability not available for this model.")

    # ===================== HR DECISION ===================== #
    st.markdown("### 💡 HR Decision")

    if prob > 0.7:
        st.error("🔴 High Risk Employee — Immediate action required")
        summary = "High attrition risk detected."
    elif prob > 0.4:
        st.warning("🟡 Medium Risk Employee — Monitor closely")
        summary = "Early warning signs of attrition."
    else:
        st.success("🟢 Low Risk Employee — Stable employee")
        summary = "No significant retention risk."

    st.markdown(f"**Summary:** {summary}")