# -*- coding: utf-8 -*-
"""
MULTIPLE LINEAR REGRESSION
Constraints for using Multiple Regression:
    - Linearity: linear relationship between independent and dependent variables.
    - Homoscedasticity: constant variability of errors.
    - Normality: independent variables should be normally distributed.
    - Independence of errors: errors should not be related to each other.
    - No multicollinearity: avoid "competition" between dummy variables.

5 Methods to build multiple regression models:
    - All-in: include all variables at once.
    - Backward Elimination: remove the least significant variable one by one.
    - Forward Selection: add the most significant variable each time.
    - Bidirectional Elimination: combine forward and backward (more precise but tedious).
    - Score Comparison: test all combinations and keep the best score (Akaike).
"""

# Library imports
import numpy as np        # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd       # For data manipulation

# Load the dataset with startup data
dataset = pd.read_csv("50_Startups.csv")

# Extract independent variables (X) and dependent variable (y)
X = dataset.iloc[:, :-1].values  # All columns except the last (R&D, Admin, Marketing, State)
y = dataset.iloc[:, 4].values    # Last column: Profit (what we want to predict)

## Handle missing values (NaN)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])  # Apply mean imputation on columns with NaN

# Encode categorical variables (convert text to numbers)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])  # Encode State column

onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(onehotencoder.fit_transform(X), dtype=float)  # Binary dummy variables

# Remove first dummy variable to avoid multicollinearity (dummy variable trap)
X = X[:, 1:]

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling (standardization)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  # Test data scaled with training parameters

# MULTIPLE LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)  # Train the model

# Predict with test data
y_pred = regression.predict(X_test)

# Backward Elimination to simplify the model
import statsmodels.api as sm
X_be = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)  # Add intercept column

SL = 0.05  # Significance level

# Iteration 1: All variables
X_opt = X_be[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print("Backward Elimination - Iteration 1 (all variables):")
print(regression_OLS.summary())

# Iteration 2: Remove least significant variable
X_opt = X_be[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print("\nBackward Elimination - Iteration 2:")
print(regression_OLS.summary())

# FIX: Visualization uses first feature column for scatter plot (R&D Spend)
# This is a simplification since multiple regression operates in higher dimensions
plt.scatter(range(len(y_test)), y_test, color="red", label="Actual")
plt.scatter(range(len(y_pred)), y_pred, color="blue", label="Predicted")
plt.title("Actual vs Predicted Profit (Test set)")
plt.xlabel("Observation Index")
plt.ylabel("Profit ($)")
plt.legend()
plt.show()
