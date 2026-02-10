## Analizamos empresas startups para decidir en cuál invertir según varios criterios.
## La idea es ver si hay alguna relación entre ganancias y ubicación y predecir cuánto gana la empresa.

# Primero, importamos las librerías que necesitamos
import numpy as np  # para operaciones matemáticas
import matplotlib.pyplot as plt  # para hacer gráficos bonitos
import pandas as pd  # para trabajar con datos en forma de tablas

# Cargamos el dataset
dataset = pd.read_csv("Position_Salaries.csv")

# Seleccionamos la columna independiente "X" y la variable dependiente "y"
X = dataset.iloc[:, 1:2].values  # Aquí agarramos todas las filas y la segunda columna
y = dataset.iloc[:, -1].values  # Y en 'y' ponemos la última columna, que es lo que queremos predecir

# Ahora importamos el modelo de regresión lineal
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)  # Entrenamos el modelo con los datos que tenemos en X e y

# Configuración de regresión polinómica
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)  # Elegimos que el modelo sea de grado 2
X_poly = poly_reg.fit_transform(X)  # Transformamos X a su versión polinómica

# Ajustamos los datos a la regresión polinómica
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Graficamos los resultados de la regresión lineal
plt.scatter(X, y, color="red")  # Puntos originales en rojo
plt.plot(X, lin_reg.predict(X), color="blue")  # Línea de regresión lineal en azul
plt.title("Modelo de regresión lineal")
plt.xlabel("Posición del empleador")
plt.ylabel("Sueldo en $")
plt.show()

# Graficamos los resultados de la regresión polinómica
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")  # Línea de regresión polinómica en azul
plt.title("Modelo de regresión polinómica")
plt.xlabel("Posición del empleador")
plt.ylabel("Sueldo en $")
plt.show()

# NOTE: Position_Salaries.csv has a single numeric predictor (Level).
# Categorical encoding is not needed here — that section belongs to 50_Startups scripts.
# The linear vs polynomial comparison above is the correct use of this dataset.
