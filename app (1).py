import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Student Burnout Risk Predictor",
    page_icon="🎓",
    layout="centered"
)

# -------------------- TITLE --------------------
st.title("🎓 Student Burnout Risk Predictor")
st.write(
    "ML-based burnout risk prediction using academic & behavioral indicators "
    "derived from real student performance data."
)

st.divider()

# -------------------- DEBUG CHECK --------------------
st.write("📁 Files in directory:")
st.write(os.listdir("."))

# -------------------- LOAD MODEL & SCALER --------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("burnout_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.success("✅ Model and scaler loaded successfully")
except Exception as e:
    st.error("❌ Failed to load model or scaler")
    st.exception(e)
    st.stop()

st.divider()

# -------------------- USER INPUTS --------------------
st.subheader("📥 Enter Student Details")

attendance = st.slider(
    "Attendance Percentage",
    min_value=40.0,
    max_value=100.0,
    value=75.0
)

studytime = st.slider(
    "Weekly Study Time (1 = very low, 4 = very high)",
    min_value=1,
    max_value=4,
    value=2
)

failures = st.slider(
    "Number of Past Course Failures",
    min_value=0,
    max_value=4,
    value=0
)

health = st.slider(
    "Health Status (1 = very poor, 5 = excellent)",
    min_value=1,
    max_value=5,
    value=3
)

goout = st.slider(
    "Social Activity Level (1 = low, 5 = high)",
    min_value=1,
    max_value=5,
    value=3
)

marks_trend = st.slider(
    "Marks Trend (Final Grade − Initial Grade)",
    min_value=-10.0,
    max_value=10.0,
    value=0.0
)

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Burnout Risk"):

    # Input order MUST match training
    input_data = np.array([
        [
            attendance,
            studytime,
            failures,
            health,
            goout,
            marks_trend
        ]
    ])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    risk_labels = {
        0: ("Low Risk 🟢", "Student shows healthy academic and behavioral patterns."),
        1: ("Medium Risk 🟡", "Some warning signs detected. Monitoring is advised."),
        2: ("High Risk 🔴", "Strong burnout indicators detected. Immediate support recommended.")
    }

    label, explanation = risk_labels[prediction]

    st.subheader(f"📊 Prediction: {label}")
    st.write(explanation)

    st.divider()

    # -------------------- EXPLAINABILITY --------------------
    st.subheader("📈 Why this prediction?")

    feature_names = [
        "Attendance %",
        "Study Time",
        "Failures",
        "Health",
        "Social Activity",
        "Marks Trend"
    ]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(feature_names, importances)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance (Model Explanation)")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")

# -------------------- FOOTER --------------------
st.divider()
st.caption(
    "⚠️ This tool predicts burnout risk for educational support purposes only. "
    "Predictions are based on inferred risk patterns from anonymized academic data."
)
