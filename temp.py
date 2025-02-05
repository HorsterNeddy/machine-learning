import streamlit as st
#import pandas as pd
#import numpy as np
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import joblib


#@st.cache

model = joblib.load('CVD_prediction_model')
#joblib.dump(model,'CVD_prediction_model')


# Function to predict heart disease likelihood
def predict_cardiovascular_disease(model, user_data):
    user_data = np.array(user_data).reshape(1, -1)
    prediction_proba = model.predict_proba(user_data)
    probability = prediction_proba[0][1] * 100  # Probability of having heart disease
    return probability

# Streamlit interface
def main():
    # Custom styling
    st.markdown("""
        <style>
        body {
            background-color: #87CEEB;  /* Sky Blue background */
            font-family: 'Comic Sans MS', cursive, sans-serif;  /* Stylish font */
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            font-weight: bold;
        }
        .stTextInput>label {
            color: #555;
            font-family: 'Arial', sans-serif;
            font-size: 14px;
        }
        .stTextInput input {
            border: 2px solid #4CAF50;
            font-size: 14px;
        }
        .stSelectbox select {
            font-family: 'Arial', sans-serif;
            font-size: 14px;
        }
        h1 {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            font-size: 40px;
        }
        h2 {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            font-size: 30px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Cardiovascular Disease Prediction")
    
    st.subheader("Enter your details:")
    
    # User input fields
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Select Gender', ['Male', 'Female'])
    sex = 1 if sex == 'Male' else 0 # 0: female, 1: male
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])  # 0: typical angina, 1: atypical angina, etc.
    trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
    chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=700, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])  # 0: false, 1: true
    restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina', [0, 1])  # 0: no, 1: yes
    oldpeak = st.number_input('Depression Induced by Exercise', min_value=0.0, max_value=6.0, value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', [0, 1, 2, 3])  # 0: normal, 1: fixed defect, etc.
    
    # Prepare user input data
    user_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Train the model
    model = joblib.load('CVD_prediction_model')
    #train_model()

    # Prediction button
    if st.button("Predict"):
        # Get prediction
        probability = predict_cardiovascular_disease(model, user_data)

        # Categorize and color code the result
        if probability >= 75:
            result = "Critical"
            color = "red"
        elif 50 <= probability < 75:
            result = "Mild "
            color = "orange"
        elif 25 <= probability < 50:
            result = "Stable "
            color = "yellow"
        else:
            result = "No Threat "
            color = "green"

        # Display result with color coding
        st.markdown(f"""
            <div style="background-color:{color}; padding: 10px; font-size: 20px; text-align: center;">
                <strong>Prediction: {result}</strong><br>
                Likelihood: {probability:.2f}%
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
