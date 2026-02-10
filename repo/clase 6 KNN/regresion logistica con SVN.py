# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:22:03 2024

@author: Carlos-Javier

SVM Classification with LINEAR Kernel
Uses a linear hyperplane to separate classes.
"""

### Library imports
import numpy as np        # Library for number handling and math tools (np alias)
import matplotlib.pyplot as plt  # Library for graphical representation (pyplot sub-library) (plt alias)
import pandas as pd       # Library for data manipulation (pd alias)

## Import the Dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

## Feature matrix
X = dataset.iloc[:, [2, 3]]  # Independent variables: Age and EstimatedSalary
y = dataset.iloc[:, -1]      # Dependent variable: Purchased

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## Fit the SVM model with linear kernel on the training set
from sklearn.svm import SVC
svma = SVC(kernel="linear", random_state=0)
svma.fit(X_train, y_train)

## Predict the results with the test set
y_pred = svma.predict(X_test)

## Analyze results with the confusion matrix
from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)

### Visualize the training set results graphically
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, svma.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("SVM Linear Kernel (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


### Visualize the test set results graphically
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, svma.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("SVM Linear Kernel (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
