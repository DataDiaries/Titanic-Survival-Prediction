import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression


# Load the Titanic dataset (replace with your actual dataset path)
train = pd.read_csv('Titanic_train.csv')
test = pd.read_csv('Titanic_test.csv')


# Preprocessing functions (similar to your existing code)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# Data Preprocessing
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis=1)
train.drop('Cabin', axis='columns', inplace=True)
test.drop('Cabin', axis='columns', inplace=True)
test.dropna(inplace=True)
train['Embarked'] = train['Embarked'].fillna('S')
train.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
X = train.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.25, random_state=42)


# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)


# Streamlit app
st.title('Titanic Survival Prediction')


# Create input fields for user
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('Number of Siblings/Spouses', min_value=0, value=0)
parch = st.number_input('Number of Parents/Children', min_value=0, value=0)
fare = st.number_input('Fare', min_value=0.0, value=10.0)
embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])


# Preprocess user input
sex_encoded = 1 if sex == 'female' else 0
embarked_encoded = 0 if embarked == 'S' else (1 if embarked == 'C' else 2)

user_input = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked_encoded]
})


# Make prediction
prediction = model.predict(user_input)


# Display prediction
if st.button('Predict'):
    if prediction[0] == 1:
        st.success('Passenger would likely survive.')
    else:
        st.error('Passenger would likely not survive.')

