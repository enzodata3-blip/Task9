# -*- coding: utf-8 -*-
"""
Polynomial Regression on Position Salaries dataset.
We analyze employee positions to see the relationship between level and salary.
The idea is to predict salary based on position level using both linear and polynomial regression.

FIX: Removed incompatible 50_Startups encoding/backward elimination code that was
copy-pasted after the Position_Salaries section. Position_Salaries has only 3 columns
(Position, Level, Salary), so categorical encoding on column 3 would crash with IndexError.
FIX: Removed np.ones((50,1)) - Position_Salaries has 10 rows, not 50.
FIX: Fixed visualization to work with the actual data dimensions.
"""

# Library imports
import numpy as np        # For mathematical operations
import matplotlib.pyplot as plt  # For graphical representation
import pandas as pd       # For data manipulation

# Load the dataset
dataset = pd.read_csv("Position_Salaries.csv")

# Select the independent variable "X" and the dependent variable "y"
X = dataset.iloc[:, 1:2].values  # Level column (all rows, second column)
y = dataset.iloc[:, -1].values   # Salary column (what we want to predict)

# Linear Regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)  # Train the model with all data

# Polynomial Regression setup
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)  # Degree 2 polynomial
X_poly = poly_reg.fit_transform(X)       # Transform X to polynomial features

# Fit the polynomial regression
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualize Linear Regression results
plt.scatter(X, y, color="red")           # Original data points in red
plt.plot(X, lin_reg.predict(X), color="blue")  # Linear regression line in blue
plt.title("Linear Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary ($)")
plt.show()

# Visualize Polynomial Regression results
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")  # Polynomial regression curve in blue
plt.title("Polynomial Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary ($)")
plt.show()

# Predict a specific level
print(f"Linear prediction for level 6.5: ${lin_reg.predict([[6.5]])[0]:,.2f}")
print(f"Polynomial prediction for level 6.5: ${lin_reg2.predict(poly_reg.transform([[6.5]]))[0]:,.2f}")
