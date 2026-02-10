# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:03:34 2024

@author: Carlos-Javier
"""

"Arbol "
### Importación de librerías
import numpy as np  ## Librería para tratamiento de números y herramientas matemáticas (np Alias)
import matplotlib.pyplot as plt  ## Librería para representación gráfica (pyplot es una sublibrearía - gráficos) (plt Alias)
import pandas as pd  ## Librería para manipular datos (pd Alias)

## Importar el Dataset
dataset = pd.read_csv("winequality-red.csv")

## Matriz de características 
X = dataset.iloc[:, [0, 1]].values  # Variable independientes, en este caso 'Age' y 'Estimated Salary'
y = dataset.iloc[:, 11].values  # Variable dependiente

# Dividir el dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Escalado de variables (opcional, puede no ser necesario para árboles de decisión)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Modelo de Regresión con Árbol de Decisión
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(criterion="squared_error", random_state=0)  # Cambiado a 'squared_error'

# Entrenamos el modelo
dt.fit(X_train, y_train)

# Predicción de los resultados con el conjunto de test
y_pred = dt.predict(X_test)

# Evaluación de los resultados (opcional, sin la métrica de error cuadrático medio como en el original)
# Se realiza la matriz de confusión solo si se tratara de clasificación, pero no se incluye aquí porque es un modelo de regresión.

# Visualización del algoritmo de entrenamiento con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.1))
plt.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.title("Árbol de Decisión (Conjunto de Entrenamiento)")
plt.xlabel("alcohol")
plt.ylabel("fixed acidity")
plt.legend()
plt.show()

# Visualización del algoritmo de prueba con los resultados
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.title("Árbol de Decisión (Conjunto de Prueba)")
plt.xlabel("Alcohol")
plt.ylabel("Fixed acidity")
plt.legend()
plt.show()



"Random forest"
## Importar el Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  ## Usamos RandomForestRegressor para regresión
from sklearn.metrics import mean_squared_error  ## Para evaluar la predicción con MSE

# Cargar el dataset
dataset  = pd.read_csv("winequality-red.csv")

# Matriz de características (entradas) y variable dependiente (salida)
X = dataset.iloc[:,[0,1]].values  # Variables independientes (por ejemplo, alcohol y acidez fija)
y = dataset.iloc[:,11].values     # Variable dependiente (calidad del vino, columna 11)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Escalado de las características
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajuste del modelo Random Forest
rf = RandomForestRegressor(n_estimators = 100, random_state = 0)  # Usamos 100 árboles en el bosque
rf.fit(X_train, y_train)

# Predicción de los resultados con el conjunto de prueba
y_pred = rf.predict(X_test)

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio (MSE): {mse}")

# Visualizar los resultados para el conjunto de entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title("Random Forest (Training set)")
plt.xlabel("Alcohol")
plt.ylabel("Fixed Acidity")
plt.legend()
plt.show()

# Visualizar los resultados para el conjunto de prueba
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title("Random Forest (Test set)")
plt.xlabel("Alcohol")
plt.ylabel("Fixed Acidity")
plt.legend()
plt.show()
