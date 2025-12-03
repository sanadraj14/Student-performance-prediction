import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests

# Load model
model = joblib.load("student_performance_model.pkl")

# Load Lottie animation from URL
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

animation_url = "https://assets2.lottiefiles.com/packages/lf20_tQ6nFS.json"
animation = load_lottie(animation_url)

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# Custom CSS styling
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 100px;
            font-weight: bold;
            color: #4CAF50;
        }
        .sub-title {
            text-align: center;
            color: #555;
            font-size: 20px;
        }
        .prediction-box {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #ddd;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<p class="main-title">🎓 Advanced Student Performance Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter details to predict the student\'s final score</p>', unsafe_allow_html=True)

# Display animation if loaded
if animation:
    st_lottie(animation, height=200)

# Input Section
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    hours = st.slider("📘 Hours Studied", 0, 12, 5)
    attendance = st.slider("📅 Attendance (%)", 0, 100, 75)
    prev_score = st.number_input("📝 Previous Score", 0, 100, 60)

with col2:
    parent_edu = st.selectbox("🎓 Parent Education Level", [0, 1, 2, 3,4])
    sleep = st.slider("😴 Sleep Hours", 0, 12, 7)
    extra = st.selectbox("📚 Extra Classes", [0, 1])

# Convert input to DataFrame
input_data = pd.DataFrame({
    "Hours_study": [hours],
    "Attendance": [attendance],
    "Previous_score": [prev_score],
    "Parent_edu": [parent_edu],
    "Sleep_hours": [sleep],
    "Extra_classes": [extra]
})

# Prediction Section
st.markdown("---")
if st.button("🔮 Predict Final Score", use_container_width=True):
    result = model.predict(input_data)[0]
    st.markdown("""
        <div class="prediction-box">
            <h3>📊 Prediction Result</h3>
            <p style="font-size: 30px; font-weight: bold; color: #2196F3;">{:.2f}</p>
        </div>
        """.format(result), unsafe_allow_html=True)
