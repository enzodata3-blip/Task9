# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:07:43 2024

@author: Carlos-Javier

KNN Classification Algorithm
Categorical data but the classification is no longer linear.
To begin, a new data point is placed to identify which side it will be closer to.
This is the nearest neighbor theory.

What this algorithm does:
1. Choose a number K of neighbors (recommended to be odd; Python default is 5)
2. Take the K nearest neighbors of the new data point using Euclidean distance
3. Among those neighbors, count the number of points belonging to each category
4. Assign the new data point to the category with the most neighbors

Only uses qualitative variables.

Probability comes into play here.
It's between 0 and 1 and responds to very simple actions,
which is why quantitative variables aren't used but rather qualitative ones.
A classification model called KNN.
When predicting categorical variables,
the word "classification" should appear.
Comparable variables should be scaled,
and scaling means putting them in the same range.
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

## Fit the KNN classifier on the training set
## Learn to predict classifications
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
# When p=1 it's Manhattan distance, when p=2 it's Euclidean distance
knn.fit(X_train, y_train)

## Predict the results with the test set
y_pred = knn.predict(X_test)

## Analyze results with the confusion matrix
## Check if the prediction is good
## Calculate the confusion matrix on the test model
## How many predictions are correct
from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)

## For verification, sum along the diagonal

### Visualize the training set results graphically
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("K-NN (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


### Visualize the test set results graphically
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("K-NN (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
