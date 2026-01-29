# 🎓 Student Burnout Risk Predictor

An end-to-end machine learning project that predicts student burnout risk using
academic and behavioral indicators derived from real-world student performance data.

🔗 **Live App:** https://<your-streamlit-link>.streamlit.app

---

## 📌 Problem Statement

Student burnout often goes undetected until academic performance declines
or dropout occurs. Educational institutions lack early warning systems
to identify at-risk students.

This project addresses that gap by predicting **Low / Medium / High burnout risk**
using interpretable machine learning models.

---

## 📊 Dataset

- Source: UCI Machine Learning Repository
- Files:
  - `student-mat.csv` (Mathematics course)
  - `student-por.csv` (Portuguese course)
- Records: ~1000 students
- Data is anonymized and publicly available

Since burnout labels do not exist in real datasets, burnout risk was **inferred**
using academic and behavioral indicators.

---

## 🧠 Feature Engineering

Key features used:

- Attendance percentage (derived from absences)
- Marks trend (G3 − G1)
- Study time
- Past failures
- Health status
- Social activity level

### Burnout Risk Definition

| Risk Level | Criteria |
|----------|---------|
| Low | Good attendance, stable/improving grades |
| Medium | Moderate absences or declining grades |
| High | Poor attendance, declining grades, low health |

This mirrors real-world risk modeling used in education analytics.

---

## 🤖 Models Used

- Logistic Regression (baseline)
- Random Forest (primary model)
- XGBoost (comparison)

Random Forest was selected for deployment due to its balance between
performance and interpretability.

---

## 📈 Model Explainability

Feature importance is used to explain predictions, highlighting
key contributors such as attendance and grade decline.

This makes the model suitable for decision support, not just prediction.

---

## 🖥️ Web Application

The trained model is deployed as an interactive Streamlit web application:

- User inputs student details
- Predicts burnout risk
- Displays explanation via feature importance

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## ⚠️ Disclaimer

This tool is intended for educational support and research purposes only.
Predictions represent inferred risk patterns and should not be used
as sole decision-making criteria.

---

## 👤 Author

**ZORO**  
Final-year Computer Science student  
