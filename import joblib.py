import joblib
from fastapi import FastAPI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.metrics import classification_report
from sklearn.svm import SVR, SVC
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.preprocessing import StandardScaler

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regression": SVR(),
    "Support Vector Classifier": SVC(kernel='rbf', gamma='auto', C=2),
    "Logistic Regression": LogisticRegression()
}

# Load data
df = pd.read_excel('./PressureOutput.xlsx', sheet_name='Processed Data')

# Split data into features and labels
x = df.iloc[:, :-1]  # Features - All columns but last
y = df.iloc[:, -1]  # Labels

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=104, test_size=0.25, shuffle=True)

# Initialize and train the classifier
classifier = SVC(kernel='rbf', gamma='auto', C=2)
classifier.fit(x_train, y_train)

# Predict on the test set
y_test_predict = classifier.predict(x_test)
print(classification_report(y_test, y_test_predict))

# Predict on new set
new_set = np.array([0, 0, 0, 0, 0]).reshape(1, -1)

# Normalize/Standardize the new set using the same scaler as training data
scaler = StandardScaler().fit(x_train)
new_set_scaled = scaler.transform(new_set)

# Make prediction on the new set
new_set_prediction = classifier.predict(new_set_scaled)
print("Prediction for new set [0,0,0,0,0]:", new_set_prediction)
