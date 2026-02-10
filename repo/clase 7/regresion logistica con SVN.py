# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:07:43 2024

@author: Carlos-Javier

Decision Tree Classification (CART)
Regression produces numerical variables.
Always use .values for tree-based models.

NOTE: This file is named "regresion logistica con SVN.py" but actually
implements a Decision Tree Classifier. The filename is misleading.
"""

### Library imports
import numpy as np        # Library for number handling and math tools (np alias)
import matplotlib.pyplot as plt  # Library for graphical representation (pyplot sub-library) (plt alias)
import pandas as pd       # Library for data manipulation (pd alias)

## Import the Dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

## Feature matrix
X = dataset.iloc[:, [2, 3]].values  # Independent variables: Age and EstimatedSalary
y = dataset.iloc[:, 4].values       # Dependent variable: Purchased

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Decision Tree Classifier (no scaling needed - trees don't require it)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
# Criterion='entropy'; Gini index is the default
# If entropy = 0, it means 100% correct classification
dt.fit(X_train, y_train)

## Predict the results with the test set
y_pred = dt.predict(X_test)

## Analyze results with the confusion matrix
from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)

## For verification, sum along the diagonal

### Visualize the training set results graphically
# Note: Using raw (unscaled) data, so step sizes reflect actual Age and Salary ranges
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=1),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=500))
plt.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("Decision Tree Classifier (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


### Visualize the test set results graphically
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=1),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=500))
plt.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("Decision Tree Classifier (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
