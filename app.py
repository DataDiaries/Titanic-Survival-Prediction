import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score

# ... (Your existing code for data loading, preprocessing, and model building) ...

st.title("Titanic Survival Prediction")

# Sidebar for user input
st.sidebar.header("Passenger Information")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses", min_value=0, value=0)
parch = st.sidebar.number_input("Number of Parents/Children", min_value=0, value=0)
fare = st.sidebar.number_input("Fare", min_value=0.0, value=10.0)
embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])

# Create a dictionary with user input
user_input = {
    "Pclass": pclass,
    "Sex": 0 if sex == "male" else 1,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": 0 if embarked == "S" else (1 if embarked == "C" else 2)
}

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Make prediction
prediction = model.predict(input_df)[0]

# Display prediction
st.header("Prediction")
if prediction == 1:
    st.write("The passenger is predicted to have survived.")
else:
    st.write("The passenger is predicted to not have survived.")


# ... (Add more sections for visualizations and analysis) ...

# Example: Display the feature importance table
st.header("Feature Importance")
st.write(coefficients_df)
