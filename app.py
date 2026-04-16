import streamlit as st
import pandas as pd
import numpy as np

from src.predict import predict_employee

st.set_page_config(
    page_title="HR Decision Intelligence System",
    layout="wide"
)

# ================= HEADER =================
st.markdown("""
# 🏢 HR Decision Intelligence System
### AI-powered Employee Attrition Prediction with Explainable AI
""")

st.markdown("---")

left, right = st.columns([1, 2], gap="large")

# ================= INPUT =================
with left:
    st.markdown("## 📥 Employee Profile")

    age = st.slider("Age", 18, 60, 30)
    daily_rate = st.number_input("Daily Rate", 100, 1500, 500)
    distance = st.slider("Distance From Home", 1, 30, 5)
    income = st.number_input("Monthly Income", 1000, 20000, 5000)
    years = st.slider("Years at Company", 0, 40, 3)

    overtime = st.selectbox("OverTime", ["No", "Yes"])
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    work_life = st.slider("Work Life Balance", 1, 4, 3)

    run = st.button("🚀 Run AI Risk Analysis")

# ================= OUTPUT =================
with right:
    st.markdown("## 📊 AI Decision Dashboard")

    if run:

        # -------- INPUT PACK --------
        input_data = {
            "Age": age,
            "DailyRate": daily_rate,
            "DistanceFromHome": distance,
            "MonthlyIncome": income,
            "YearsAtCompany": years,
            "OverTime": overtime,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life
        }

        # -------- PREDICTION --------
        pred, risk_score, importances, features = predict_employee(input_data)

        # ================= RISK SCORE =================
        st.markdown("### 🎯 Risk Score (0–100)")

        st.metric(
            label="Attrition Risk",
            value=f"{risk_score}"
        )

        if risk_score > 60:
            st.error("🚨 High Risk Employee")
        elif risk_score > 30:
            st.warning("⚠ Medium Risk Employee")
        else:
            st.success("✅ Low Risk Employee")

        st.markdown("---")

        # ================= SHAP IMPORTANCE FIX =================
        st.markdown("### 🧠 AI Explanation (SHAP-based Importance)")

        importances = np.array(importances).flatten()

        df_imp = pd.DataFrame({
            "Feature": features,
            "Importance": importances[:len(features)]
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(df_imp.set_index("Feature"))

        st.markdown("---")

        # ================= HR INSIGHTS =================
        st.markdown("### 💡 HR Insights")

        insights = []

        if income < 3000:
            insights.append("Low income is a strong attrition driver")

        if overtime == "Yes":
            insights.append("Overtime increases burnout risk significantly")

        if job_satisfaction <= 2:
            insights.append("Low job satisfaction is critical risk factor")

        if work_life <= 2:
            insights.append("Poor work-life balance increases attrition")

        if distance > 20:
            insights.append("Long commute increases turnover risk")

        for i in insights:
            st.warning(f"• {i}")

        st.markdown("---")

        # ================= HR RECOMMENDATION =================
        st.markdown("### 🧭 HR Recommendation")

        if risk_score > 60:
            st.error("Immediate intervention required")
        elif risk_score > 30:
            st.warning("Monitor employee closely")
        else:
            st.success("No action required")

        st.markdown("---")

        # ================= AI SUMMARY =================
        st.markdown("### 🧠 AI Summary")

        if risk_score > 60:
            st.info("High-risk employee based on multiple behavioral signals.")
        elif risk_score > 30:
            st.info("Moderate risk employee requiring attention.")
        else:
            st.info("Low-risk employee with stable profile.")

    else:
        st.info("👈 Enter employee details and run AI analysis")