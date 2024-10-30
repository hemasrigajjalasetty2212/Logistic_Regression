# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:23:17 2024

"""


# Import necessary libraries
import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


# 1. Data Exploration
train_data = pd.read_csv('Titanic_train.csv')  
test_data = pd.read_csv('Titanic_test.csv')

# 2. Data Preprocessing
imputer = SimpleImputer()
train_data[['Age']] = imputer.fit_transform(train_data[['Age']])
test_data[['Age']] = imputer.transform(test_data[['Age']])
encoder = LabelEncoder()
train_data['Sex'] = encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = encoder.transform(test_data['Sex'])

# 3. Model Building
X = train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1) # Dropping columns with string values 
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred)


# 6. Deployment with Streamlit
st.title('Logistic Regression Model')
st.write('Enter values for prediction:')
Pclass = st.selectbox('Pclass', [1, 2, 3])
Sex = st.selectbox('Sex', [0, 1])
Age = st.number_input('Age')
SibSp = st.number_input('SibSp')
Parch = st.number_input('Parch')
Fare = st.number_input('Fare')
if st.button('Predict'):
    prediction = model.predict([[Pclass, Sex, Age, SibSp, Parch, Fare]])
    st.write('Survival Probability:', prediction[0])

