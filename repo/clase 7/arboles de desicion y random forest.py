# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:03:34 2024

@author: Carlos-Javier

Decision Tree and Random Forest REGRESSION on wine quality data.
Uses alcohol and fixed acidity to predict wine quality score.

FIX: Removed unnecessary StandardScaler - tree-based models are invariant
to feature scaling and do not require it.
FIX: Corrected meshgrid step=500 bug on scaled data (was producing empty plots).
"""

### Library imports
import numpy as np        # Library for number handling and math tools (np alias)
import matplotlib.pyplot as plt  # Library for graphical representation (pyplot sub-library) (plt alias)
import pandas as pd       # Library for data manipulation (pd alias)

## Import the Dataset
dataset = pd.read_csv("winequality-red.csv")

## Feature matrix
X = dataset.iloc[:, [0, 1]].values  # Independent variables: alcohol and fixed acidity
y = dataset.iloc[:, 11].values      # Dependent variable: wine quality

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# NOTE: StandardScaler removed - Decision Trees do not need feature scaling.
# Trees split on thresholds, so the scale of features does not affect their decisions.

# Decision Tree Regression model
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(criterion="squared_error", random_state=0)

# Train the model
dt.fit(X_train, y_train)

# Predict the results with the test set
y_pred = dt.predict(X_test)

# Evaluation: calculate Mean Squared Error
from sklearn.metrics import mean_squared_error
mse_dt = mean_squared_error(y_test, y_pred)
print(f"Decision Tree MSE: {mse_dt}")

# Visualize the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.1))
plt.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.title("Decision Tree Regressor (Training set)")
plt.xlabel("Alcohol")
plt.ylabel("Fixed Acidity")
plt.legend()
plt.show()

# Visualize the test set results
# FIX: Changed step from 500 to 0.1 - data is unscaled (range ~4-16), step=500 produced empty plot
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.1))
plt.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.title("Decision Tree Regressor (Test set)")
plt.xlabel("Alcohol")
plt.ylabel("Fixed Acidity")
plt.legend()
plt.show()


# ============================================================
# RANDOM FOREST REGRESSION
# ============================================================

## Import the Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = pd.read_csv("winequality-red.csv")

# Feature matrix (inputs) and dependent variable (output)
X = dataset.iloc[:, [0, 1]].values  # Independent variables (alcohol and fixed acidity)
y = dataset.iloc[:, 11].values      # Dependent variable (wine quality, column 11)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# NOTE: StandardScaler removed - Random Forest does not need feature scaling.

# Fit the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=0)  # 100 trees in the forest
rf.fit(X_train, y_train)

# Predict the results with the test set
y_pred = rf.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_rf = mean_squared_error(y_test, y_pred)
print(f"Random Forest MSE: {mse_rf}")

# Visualize the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.1))

plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title("Random Forest Regressor (Training set)")
plt.xlabel("Alcohol")
plt.ylabel("Fixed Acidity")
plt.legend()
plt.show()

# Visualize the test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.1))

plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title("Random Forest Regressor (Test set)")
plt.xlabel("Alcohol")
plt.ylabel("Fixed Acidity")
plt.legend()
plt.show()
