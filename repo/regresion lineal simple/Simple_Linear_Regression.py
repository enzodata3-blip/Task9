# -*- coding: utf-8 -*-
"""
Multiple Linear Regression with Backward Elimination on 50 Startups dataset.

NOTE: This file is named "Simple_Linear_Regression.py" but actually implements
Multiple Linear Regression. The filename is misleading.
"""

# Library imports
import numpy as np        # For numerical operations (np alias)
import matplotlib.pyplot as plt  # For graphical representation (plt alias)
import pandas as pd       # For data manipulation (pd alias)

# Import the dataset
dataset = pd.read_csv("50_Startups.csv")

# Independent variables "X" and dependent variable "y"
X = dataset.iloc[:, :-1].values  # All rows, all columns except the last
y = dataset.iloc[:, 4].values    # Column to predict: Profit

# Encode categorical variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])  # Convert State text to numbers

# Dummy variables: convert categories into multiple binary columns (one for each category)
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')
X = np.array(onehotencoder.fit_transform(X), dtype=float)

# Remove one dummy variable to avoid multicollinearity
X = X[:, 1:]

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling (standardization and normalization)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  # Scale test data with training parameters

# Multiple Linear Regression model
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)  # Train the model

# Make predictions with the model
y_pred = regression.predict(X_test)

# Backward Elimination (to optimize the model)
import statsmodels.api as sm
X_be = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)  # Add intercept column
SL = 0.05  # Significance level

# Test different variable combinations, removing the least significant ones
X_opt = X_be[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print("Backward Elimination - Iteration 1 (all variables):")
print(regression_OLS.summary())

# Remove least significant variables iteratively
X_opt = X_be[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print("\nBackward Elimination - Iteration 2:")
print(regression_OLS.summary())

X_opt = X_be[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print("\nBackward Elimination - Iteration 3:")
print(regression_OLS.summary())

X_opt = X_be[:, [0, 3, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print("\nBackward Elimination - Iteration 4:")
print(regression_OLS.summary())

X_opt = X_be[:, [0, 3]]
regression_OLS = sm.OLS(y, X_opt).fit()
print("\nBackward Elimination - Iteration 5 (final):")
print(regression_OLS.summary())

# FIX: Visualization - scatter actual vs predicted (can't do 2D line plot with multi-dimensional X)
plt.scatter(range(len(y_test)), y_test, color="red", label="Actual")
plt.scatter(range(len(y_pred)), y_pred, color="blue", label="Predicted")
plt.title("Actual vs Predicted Profit (Test set)")
plt.xlabel("Observation Index")
plt.ylabel("Profit ($)")
plt.legend()
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.scatter(range(len(residuals)), residuals, color="purple")
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals (Test set)")
plt.xlabel("Observation Index")
plt.ylabel("Residual ($)")
plt.show()
